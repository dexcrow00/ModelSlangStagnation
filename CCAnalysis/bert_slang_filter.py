#!/usr/bin/env python3
"""
bert_slang_filter.py — Fine-tuned transformer slang sense classifier.

Supports BERT, RoBERTa, or both models simultaneously. When both are loaded
an ensemble score (average P(slang)) is also produced.

  score = 2 * P(slang) - 1  (range [-1, 1]; positive = slang usage)

Threshold filtering uses: ensemble_score (both models) > bert_score > roberta_score.

Requires: torch, transformers, scikit-learn

Usage:
    # Fine-tune BERT only, then score
    python bert_slang_filter.py contexts/ --output-dir scored/ \\
        --target-words target_words.txt --score-all \\
        --ft-annotations annotations/ --ft-model-dir ./ft_model/

    # Fine-tune RoBERTa only, then score
    python bert_slang_filter.py contexts/ --output-dir scored/ \\
        --target-words target_words.txt --score-all \\
        --roberta-annotations annotations/ --roberta-model-dir ./roberta_model/

    # Fine-tune both models and produce ensemble scores
    python bert_slang_filter.py contexts/ --output-dir scored/ \\
        --target-words target_words.txt --score-all \\
        --ft-annotations annotations/ --ft-model-dir ./ft_model/ \\
        --roberta-annotations annotations/ --roberta-model-dir ./roberta_model/

    # Load saved models (skip training) and filter above threshold
    python bert_slang_filter.py contexts/*.csv -o filtered.csv \\
        --target-words target_words.txt \\
        --ft-model-dir ./ft_model/ --roberta-model-dir ./roberta_model/ \\
        --ft-threshold 0.1
"""

from __future__ import annotations

import argparse
import csv
import glob
import logging
import sys
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Tuple

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(line_buffering=True)  # type: ignore[attr-defined]

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stderr)],
)
log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _batched(items: list, size: int) -> Iterator[list]:
    for i in range(0, len(items), size):
        yield items[i : i + size]


def _read_rows(path: Path) -> List[Dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as fh:
        rows = list(csv.DictReader(fh))
    # Normalise any quoted column names produced by Excel/PowerShell BOM exports
    normalised = []
    for row in rows:
        normalised.append({k.strip('"').strip(): v for k, v in row.items()})
    return normalised


def load_target_words(path: Path) -> List[str]:
    with path.open(encoding="utf-8") as fh:
        words = [line.strip() for line in fh if line.strip() and not line.startswith("#")]
    if not words:
        log.error("No words found in %s", path)
        sys.exit(1)
    log.info("Loaded %d target words from %s", len(words), path)
    return words


# ---------------------------------------------------------------------------
# Shared dataset + annotation loader
# ---------------------------------------------------------------------------

class _SlangDataset(Dataset):
    """Tokenised binary classification dataset stored in memory."""

    def __init__(self, texts: List[str], labels: List[int], tokenizer, max_len: int = 256) -> None:
        enc = tokenizer(texts, padding=True, truncation=True, max_length=max_len, return_tensors="pt")
        self.input_ids      = enc["input_ids"]
        self.attention_mask = enc["attention_mask"]
        self.labels         = torch.tensor(labels, dtype=torch.long)

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, i: int) -> Dict:
        return {
            "input_ids":      self.input_ids[i],
            "attention_mask": self.attention_mask[i],
            "labels":         self.labels[i],
        }


def load_annotation_examples(annotation_dir: Path) -> List[Tuple[str, int]]:
    """Recursively load (text, label) pairs from CSVs with an 'is_slang' column.

    Expects rows where is_slang == '1' (slang) or '0' (not slang).
    Rows with blank or missing values are silently skipped.
    """
    examples: List[Tuple[str, int]] = []
    for csv_path in sorted(annotation_dir.rglob("*.csv")):
        with csv_path.open(newline="", encoding="utf-8-sig") as fh:
            for row in csv.DictReader(fh):
                a = (row.get("is_slang") or "").strip()
                if a == "1":
                    examples.append((row["target_context"], 1))
                elif a == "0":
                    examples.append((row["target_context"], 0))
    n_pos = sum(e[1] for e in examples)
    log.info(
        "Loaded %d annotated examples from %s (%d slang, %d not-slang)",
        len(examples), annotation_dir, n_pos, len(examples) - n_pos,
    )
    return examples


# ---------------------------------------------------------------------------
# Generic transformer classifier (works for BERT, RoBERTa, etc.)
# ---------------------------------------------------------------------------

class TransformerSlangClassifier:
    """Binary transformer classifier fine-tuned on human-annotated slang examples.

    Works with any HuggingFace AutoModel (BERT, RoBERTa, DistilBERT, …).
    Scores are mapped P(slang) ∈ [0, 1] → [-1, 1] via ``2p - 1``.

    Parameters
    ----------
    base_model : HuggingFace model ID or local path (e.g. 'bert-base-uncased',
                 'roberta-base').
    label      : Short name used in log messages, e.g. 'BERT' or 'RoBERTa'.
    words      : Target words used to select relevant rows during scoring.
    device     : PyTorch device string, or None for auto-detect.
    """

    _DEFAULT_BERT_BASE    = "bert-base-uncased"
    _DEFAULT_ROBERTA_BASE = "roberta-base"

    def __init__(
        self,
        base_model: str,
        label: str = "model",
        words: Optional[List[str]] = None,
        device: Optional[str] = None,
    ) -> None:
        self.base_model = base_model
        self.label      = label
        self.words      = [w.lower() for w in (words or [])]

        if device:
            self.device = device
        elif torch.cuda.is_available():
            self.device = "cuda"
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            self.device = "mps"
        else:
            self.device = "cpu"

        self.tokenizer = None
        self.model     = None

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def train(
        self,
        examples: List[Tuple[str, int]],
        model_save_dir: Path,
        epochs: int = 10,
        lr: float = 2e-5,
        batch_size: int = 16,
    ) -> None:
        """Fine-tune a sequence classifier and save the best val-acc checkpoint."""
        try:
            from transformers import (  # type: ignore
                AutoModelForSequenceClassification,
                AutoTokenizer,
                get_linear_schedule_with_warmup,
            )
            from sklearn.model_selection import train_test_split  # type: ignore
        except ImportError as exc:
            log.error("Fine-tuning requires transformers and scikit-learn: %s", exc)
            sys.exit(1)

        if len(examples) < 10:
            log.error("[%s] Too few annotated examples (%d) for reliable fine-tuning.",
                      self.label, len(examples))
            sys.exit(1)

        texts  = [e[0] for e in examples]
        labels = [e[1] for e in examples]

        train_texts, val_texts, train_labels, val_labels = train_test_split(
            texts, labels, test_size=0.2, random_state=42, stratify=labels
        )
        log.info(
            "[%s] Fine-tune split — Train: %d (%d pos / %d neg) | Val: %d (%d pos / %d neg)",
            self.label,
            len(train_texts), sum(train_labels), len(train_labels) - sum(train_labels),
            len(val_texts),   sum(val_labels),   len(val_labels)   - sum(val_labels),
        )

        log.info("[%s] Loading base model: %s", self.label, self.base_model)
        tokenizer = AutoTokenizer.from_pretrained(self.base_model)
        model     = AutoModelForSequenceClassification.from_pretrained(
            self.base_model, num_labels=2
        ).to(self.device)

        n_pos = sum(labels)
        n_neg = len(labels) - n_pos
        class_weights = torch.tensor(
            [1.0, n_neg / max(n_pos, 1)], dtype=torch.float, device=self.device
        )
        log.info("[%s] Class weights: not-slang=%.2f, slang=%.2f",
                 self.label, class_weights[0].item(), class_weights[1].item())

        train_ds     = _SlangDataset(train_texts, train_labels, tokenizer)
        val_ds       = _SlangDataset(val_texts,   val_labels,   tokenizer)
        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
        val_loader   = DataLoader(val_ds,   batch_size=batch_size)

        optimizer   = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
        total_steps = len(train_loader) * epochs
        scheduler   = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=max(1, total_steps // 10),
            num_training_steps=total_steps,
        )

        model_save_dir.mkdir(parents=True, exist_ok=True)
        best_val_acc = -1.0

        for epoch in range(1, epochs + 1):
            model.train()
            total_loss = 0.0
            for batch in train_loader:
                input_ids      = batch["input_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)
                batch_labels   = batch["labels"].to(self.device)
                logits = model(input_ids=input_ids, attention_mask=attention_mask).logits
                loss   = F.cross_entropy(logits, batch_labels, weight=class_weights)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                total_loss += loss.item()

            model.eval()
            correct = total = 0
            with torch.no_grad():
                for batch in val_loader:
                    input_ids      = batch["input_ids"].to(self.device)
                    attention_mask = batch["attention_mask"].to(self.device)
                    preds = (
                        model(input_ids=input_ids, attention_mask=attention_mask)
                        .logits.argmax(dim=1).cpu()
                    )
                    correct += (preds == batch["labels"]).sum().item()
                    total   += len(batch["labels"])

            val_acc  = correct / total if total else 0.0
            avg_loss = total_loss / len(train_loader)
            log.info("[%s] Epoch %d/%d — loss: %.4f, val_acc: %.3f",
                     self.label, epoch, epochs, avg_loss, val_acc)

            if val_acc >= best_val_acc:
                best_val_acc = val_acc
                model.save_pretrained(str(model_save_dir))
                tokenizer.save_pretrained(str(model_save_dir))
                log.info("[%s]   Saved best model (val_acc=%.3f) -> %s",
                         self.label, val_acc, model_save_dir)

        log.info("[%s] Fine-tuning complete. Best val_acc: %.3f", self.label, best_val_acc)
        self.tokenizer = tokenizer
        self.model     = model
        self.model.eval()

    # ------------------------------------------------------------------
    # Loading
    # ------------------------------------------------------------------

    def load(self, model_dir: Path) -> None:
        """Load a previously saved fine-tuned model."""
        try:
            from transformers import AutoTokenizer, AutoModelForSequenceClassification  # type: ignore
        except ImportError as exc:
            log.error("transformers is required to load a fine-tuned model: %s", exc)
            sys.exit(1)
        log.info("[%s] Loading model from %s", self.label, model_dir)
        self.tokenizer = AutoTokenizer.from_pretrained(str(model_dir))
        self.model     = (
            AutoModelForSequenceClassification.from_pretrained(str(model_dir)).to(self.device)
        )
        self.model.eval()

    # ------------------------------------------------------------------
    # Scoring
    # ------------------------------------------------------------------

    def score_rows(
        self,
        rows: List[Dict[str, str]],
        batch_size: int,
    ) -> List[Optional[float]]:
        """Return a score ∈ [-1, 1] for each row (None if no target word matched)."""
        assert self.model is not None and self.tokenizer is not None, \
            f"[{self.label}] Call train() or load() before score_rows()."

        texts  = [r["target_context"] for r in rows]
        scores: List[Optional[float]] = [None] * len(rows)

        for word in self.words:
            relevant = [i for i, t in enumerate(texts) if word in t.lower()]
            if not relevant:
                continue
            log.info("  [%s] scoring '%s': %d rows ...", self.label, word, len(relevant))
            rel_texts = [texts[i] for i in relevant]

            all_probs: List[float] = []
            for batch_texts in _batched(rel_texts, batch_size):
                enc = self.tokenizer(
                    batch_texts, padding=True, truncation=True,
                    max_length=256, return_tensors="pt",
                ).to(self.device)
                with torch.no_grad():
                    probs = (
                        torch.softmax(self.model(**enc).logits, dim=1)[:, 1].cpu().tolist()
                    )
                all_probs.extend(probs)

            for idx, p in zip(relevant, all_probs):
                s = float(2 * p - 1)          # [0,1] -> [-1,1]
                if scores[idx] is None or s > scores[idx]:
                    scores[idx] = s

        return scores


# Backward-compatibility alias
FineTunedSlangClassifier = TransformerSlangClassifier


# ---------------------------------------------------------------------------
# Scoring helpers
# ---------------------------------------------------------------------------

def _primary_score(row: Dict, bert_col: bool, roberta_col: bool, ensemble_col: bool) -> Optional[float]:
    """Return the best available score for threshold/sort decisions."""
    def _f(key: str) -> Optional[float]:
        v = row.get(key, "")
        return float(v) if v else None

    if ensemble_col:
        return _f("ensemble_score")
    if bert_col:
        return _f("bert_score")
    return _f("roberta_score")


def _sort_rows(rows: List[Dict], bert_col: bool, roberta_col: bool, ensemble_col: bool) -> None:
    _NEG_INF = float("-inf")
    rows.sort(
        key=lambda r: -(
            _primary_score(r, bert_col, roberta_col, ensemble_col) or _NEG_INF
        )
    )


def _score_file(
    path: Path,
    bert_clf: Optional[TransformerSlangClassifier],
    roberta_clf: Optional[TransformerSlangClassifier],
    args: argparse.Namespace,
    fieldnames: List[str],
    bert_col: bool,
    roberta_col: bool,
    ensemble_col: bool,
) -> Tuple[List[Dict], int]:
    """Score a single CSV file with whichever models are loaded."""
    rows = _read_rows(path)
    if not rows:
        log.info("Skipping empty file: %s", path.name)
        return [], 0

    log.info("Processing %s (%d rows) ...", path.name, len(rows))
    n = len(rows)

    bert_scores    = bert_clf.score_rows(rows, args.batch_size)    if bert_clf    else [None] * n
    roberta_scores = roberta_clf.score_rows(rows, args.batch_size) if roberta_clf else [None] * n

    out_rows: List[Dict] = []
    for row, bs, rs in zip(rows, bert_scores, roberta_scores):
        # Ensemble: average of available scores
        available = [s for s in (bs, rs) if s is not None]
        es: Optional[float] = sum(available) / len(available) if available else None

        # Build the output row
        out_row: Dict = {
            "uri":            row.get("uri", ""),
            "target_context": row.get("target_context", ""),
        }
        if bert_col:
            out_row["bert_score"]     = f"{bs:.4f}" if bs is not None else ""
        if roberta_col:
            out_row["roberta_score"]  = f"{rs:.4f}" if rs is not None else ""
        if ensemble_col:
            out_row["ensemble_score"] = f"{es:.4f}" if es is not None else ""

        # Threshold filtering (skip when not score_all)
        if not args.score_all:
            primary = _primary_score(out_row, bert_col, roberta_col, ensemble_col)
            if primary is not None and primary < args.ft_threshold:
                continue

        out_rows.append(out_row)

    log.info("  %s: %d/%d rows kept (%.1f%%)",
             path.name, len(out_rows), n,
             100.0 * len(out_rows) / n if n else 0.0)
    return out_rows, n


def _write_csv(path: Path, rows: List[Dict], fieldnames: List[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=(
            "Score slang sense using fine-tuned BERT and/or RoBERTa classifiers. "
            "When both models are loaded, an ensemble score is also produced."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # BERT only — fine-tune then score
  python bert_slang_filter.py contexts/ --output-dir scored/ \\
      --target-words target_words.txt --score-all \\
      --ft-annotations annotations/ --ft-model-dir ./ft_model/

  # RoBERTa only — fine-tune then score
  python bert_slang_filter.py contexts/ --output-dir scored/ \\
      --target-words target_words.txt --score-all \\
      --roberta-annotations annotations/ --roberta-model-dir ./roberta_model/

  # Both models — ensemble scoring
  python bert_slang_filter.py contexts/ --output-dir scored/ \\
      --target-words target_words.txt --score-all \\
      --ft-annotations annotations/ --ft-model-dir ./ft_model/ \\
      --roberta-annotations annotations/ --roberta-model-dir ./roberta_model/

  # Load saved models (no training) and filter above threshold
  python bert_slang_filter.py contexts/*.csv -o filtered.csv \\
      --target-words target_words.txt \\
      --ft-model-dir ./ft_model/ --roberta-model-dir ./roberta_model/ \\
      --ft-threshold 0.1
        """,
    )

    # Inputs / outputs
    p.add_argument("inputs", nargs="+", metavar="CSV",
                   help="Input CSV file(s), glob patterns, or a directory of CSVs.")
    out_group = p.add_mutually_exclusive_group(required=True)
    out_group.add_argument("-o", "--output", metavar="FILE",
                           help="Output CSV path (all inputs merged into one file).")
    out_group.add_argument("--output-dir", metavar="DIR", dest="output_dir",
                           help="Output directory (one output file per input).")

    # Shared options
    p.add_argument("--target-words", default="target_words.txt", metavar="FILE",
                   dest="target_words",
                   help="Text file of target words, one per line (default: target_words.txt).")
    p.add_argument("--score-all", action="store_true", dest="score_all",
                   help="Write every row with scores attached (no filtering).")
    p.add_argument("--ft-threshold", type=float, default=0.0, metavar="F",
                   dest="ft_threshold",
                   help="Min score to keep a row when filtering (default: 0.0). "
                        "Applied to: ensemble_score > bert_score > roberta_score.")
    p.add_argument("--batch-size", type=int, default=32, metavar="N", dest="batch_size",
                   help="Inference batch size for both models (default: 32).")
    p.add_argument("--device", default=None, metavar="DEV",
                   help="PyTorch device, e.g. cpu, cuda, mps (default: auto-detect).")

    # BERT group
    bert = p.add_argument_group(
        "BERT model",
        "Omit both --ft-annotations and --ft-model-dir to disable BERT entirely.",
    )
    bert.add_argument("--ft-annotations", metavar="DIR", dest="ft_annotations",
                      help="Annotated CSVs to fine-tune BERT (recursive *.csv scan, "
                           "requires 'is_slang' column).")
    bert.add_argument("--ft-model-dir", metavar="DIR", dest="ft_model_dir",
                      help="Directory to save (training) or load (inference) the BERT model.")
    bert.add_argument("--ft-base-model", default=TransformerSlangClassifier._DEFAULT_BERT_BASE,
                      metavar="ID", dest="ft_base_model",
                      help=f"Base HuggingFace model ID for BERT fine-tuning "
                           f"(default: {TransformerSlangClassifier._DEFAULT_BERT_BASE}).")
    bert.add_argument("--ft-epochs", type=int, default=10, metavar="N", dest="ft_epochs",
                      help="BERT fine-tuning epochs (default: 10).")
    bert.add_argument("--ft-lr", type=float, default=2e-5, metavar="F", dest="ft_lr",
                      help="BERT learning rate (default: 2e-5).")

    # RoBERTa group
    roberta = p.add_argument_group(
        "RoBERTa model",
        "Omit both --roberta-annotations and --roberta-model-dir to disable RoBERTa entirely.",
    )
    roberta.add_argument("--roberta-annotations", metavar="DIR", dest="roberta_annotations",
                         help="Annotated CSVs to fine-tune RoBERTa (recursive *.csv scan, "
                              "requires 'is_slang' column). Defaults to --ft-annotations "
                              "when --roberta-model-dir is given but this flag is omitted.")
    roberta.add_argument("--roberta-model-dir", metavar="DIR", dest="roberta_model_dir",
                         help="Directory to save (training) or load (inference) the RoBERTa model.")
    roberta.add_argument("--roberta-base-model",
                         default=TransformerSlangClassifier._DEFAULT_ROBERTA_BASE,
                         metavar="ID", dest="roberta_base_model",
                         help=f"Base HuggingFace model ID for RoBERTa fine-tuning "
                              f"(default: {TransformerSlangClassifier._DEFAULT_ROBERTA_BASE}).")
    roberta.add_argument("--roberta-epochs", type=int, default=10, metavar="N",
                         dest="roberta_epochs",
                         help="RoBERTa fine-tuning epochs (default: 10).")
    roberta.add_argument("--roberta-lr", type=float, default=2e-5, metavar="F",
                         dest="roberta_lr",
                         help="RoBERTa learning rate (default: 2e-5).")

    return p


def _resolve_classifier(
    model_dir_str: Optional[str],
    annotation_dir_str: Optional[str],
    fallback_annotation_dir_str: Optional[str],
    base_model: str,
    label: str,
    words: List[str],
    device: Optional[str],
    epochs: int,
    lr: float,
    batch_size: int,
    parser: argparse.ArgumentParser,
) -> Optional[TransformerSlangClassifier]:
    """
    Build, train/load, and return a TransformerSlangClassifier, or None if
    neither model_dir nor annotation_dir was supplied (model disabled).
    """
    if not model_dir_str and not annotation_dir_str:
        return None   # model disabled

    clf = TransformerSlangClassifier(base_model=base_model, label=label,
                                     words=words, device=device)

    ann_dir_str = annotation_dir_str or fallback_annotation_dir_str

    if ann_dir_str:
        ann_dir = Path(ann_dir_str)
        if not ann_dir.exists():
            parser.error(f"[{label}] Annotation directory not found: {ann_dir}")
        examples = load_annotation_examples(ann_dir)
        if not examples:
            parser.error(f"[{label}] No annotated examples found in {ann_dir}.")
        model_dir = Path(model_dir_str) if model_dir_str else None
        if model_dir is None:
            parser.error(f"[{label}] --{'ft' if label=='BERT' else 'roberta'}-model-dir "
                         f"is required when supplying annotations.")
        clf.train(examples, model_dir, epochs=epochs, lr=lr, batch_size=batch_size)
    else:
        # No annotations — load a pre-trained model
        model_dir = Path(model_dir_str)  # type: ignore[arg-type]
        if not model_dir.exists():
            parser.error(
                f"[{label}] Model directory '{model_dir}' not found. "
                f"Provide annotations to train a model first."
            )
        clf.load(model_dir)

    return clf


def main() -> None:
    parser = build_parser()
    args   = parser.parse_args()

    # Require at least one model
    if not args.ft_model_dir and not args.roberta_model_dir:
        parser.error(
            "At least one of --ft-model-dir (BERT) or --roberta-model-dir (RoBERTa) "
            "must be supplied."
        )

    # Expand inputs
    input_paths: List[Path] = []
    for pattern in args.inputs:
        p = Path(pattern)
        if p.is_dir():
            input_paths.extend(sorted(p.glob("*.csv")))
        else:
            matched = sorted(glob.glob(pattern, recursive=True))
            input_paths.extend(Path(m) for m in matched) if matched else input_paths.append(p)

    missing = [p for p in input_paths if not p.exists()]
    if missing:
        parser.error(f"File(s) not found: {', '.join(str(p) for p in missing)}")
    if not input_paths:
        parser.error("No input files found.")
    if len(input_paths) > 1 and args.output_dir is None:
        log.warning(
            "%d input files with -o set — all rows merged into one file. "
            "Use --output-dir for per-file output.",
            len(input_paths),
        )

    words  = load_target_words(Path(args.target_words))
    device = args.device
    if device:
        log.info("Using device: %s", device)

    # Build BERT classifier (optional)
    bert_clf = _resolve_classifier(
        model_dir_str            = args.ft_model_dir,
        annotation_dir_str       = args.ft_annotations,
        fallback_annotation_dir_str = None,
        base_model               = args.ft_base_model,
        label                    = "BERT",
        words                    = words,
        device                   = device,
        epochs                   = args.ft_epochs,
        lr                       = args.ft_lr,
        batch_size               = args.batch_size,
        parser                   = parser,
    )

    # Build RoBERTa classifier (optional)
    # Falls back to --ft-annotations when --roberta-annotations is not given
    roberta_clf = _resolve_classifier(
        model_dir_str            = args.roberta_model_dir,
        annotation_dir_str       = args.roberta_annotations,
        fallback_annotation_dir_str = args.ft_annotations,
        base_model               = args.roberta_base_model,
        label                    = "RoBERTa",
        words                    = words,
        device                   = device,
        epochs                   = args.roberta_epochs,
        lr                       = args.roberta_lr,
        batch_size               = args.batch_size,
        parser                   = parser,
    )

    # Determine which score columns to emit
    bert_col     = bert_clf    is not None
    roberta_col  = roberta_clf is not None
    ensemble_col = bert_col and roberta_col

    fieldnames = ["uri", "target_context"]
    if bert_col:
        fieldnames.append("bert_score")
    if roberta_col:
        fieldnames.append("roberta_score")
    if ensemble_col:
        fieldnames.append("ensemble_score")

    score_kwargs = dict(
        bert_clf     = bert_clf,
        roberta_clf  = roberta_clf,
        args         = args,
        fieldnames   = fieldnames,
        bert_col     = bert_col,
        roberta_col  = roberta_col,
        ensemble_col = ensemble_col,
    )

    # ── Per-file mode ──────────────────────────────────────────────────────────
    if args.output_dir is not None:
        out_dir = Path(args.output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        grand_in = grand_out = grand_skipped = 0
        for path in input_paths:
            out_path = out_dir / path.name
            if out_path.exists():
                log.info("Skipping %s — already exists in %s", path.name, out_dir)
                grand_skipped += 1
                continue
            file_rows, n_in = _score_file(path, **score_kwargs)
            _sort_rows(file_rows, bert_col, roberta_col, ensemble_col)
            _write_csv(out_path, file_rows, fieldnames)
            grand_in  += n_in
            grand_out += len(file_rows)
        log.info(
            "Done. %d/%d rows kept (%.1f%%) across %d file(s); %d skipped -> %s",
            grand_out, grand_in,
            100.0 * grand_out / grand_in if grand_in else 0.0,
            len(input_paths) - grand_skipped, grand_skipped, out_dir,
        )
        return

    # ── Single-file mode ───────────────────────────────────────────────────────
    output_path = Path(args.output)
    grand_in    = 0
    all_rows: List[Dict] = []
    for path in input_paths:
        file_rows, n_in = _score_file(path, **score_kwargs)
        all_rows.extend(file_rows)
        grand_in += n_in

    _sort_rows(all_rows, bert_col, roberta_col, ensemble_col)
    _write_csv(output_path, all_rows, fieldnames)
    log.info(
        "Done. %d/%d rows kept (%.1f%%), sorted by score -> %s",
        len(all_rows), grand_in,
        100.0 * len(all_rows) / grand_in if grand_in else 0.0,
        output_path,
    )


if __name__ == "__main__":
    main()
