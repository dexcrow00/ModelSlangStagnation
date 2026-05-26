#!/usr/bin/env python3
"""
bert_slang_filter.py — Slang sense scorer with three independent methods.

Scores each context row by how likely the target word is used in its slang
sense, running up to three methods for comparison:

  Method 1 — SBERT prototype matching (--sbert-model):
    Encodes the full context sentence with a sentence-transformer model trained
    for semantic similarity. Compares to per-word sense centroids built from
    slang.yaml examples.
    score = cos_sim(context, slang_proto) - cos_sim(context, standard_proto)
    For slang-only words (no standard examples): score = cos_sim(context, slang_proto)

  Method 2 — Zero-shot NLI (--nli-model):
    Frames sense classification as natural language inference.
    score = P(entailment | "X used as slang") - P(entailment | "X used literally")
    Needs no prototype examples — sense disambiguation is handled by the NLI model.

  Method 3 — Fine-tuned BERT classifier (--ft-model-dir / --ft-annotations):
    Binary BERT sequence classifier fine-tuned on human-annotated CSV examples.
    Learns directly from labeled (context, slang/not-slang) pairs.
    score = 2 * P(slang) - 1  (mapped to [-1, 1] to match NLI score range)
    Train once with --ft-annotations, then reuse with --ft-model-dir.

All active scores are written as separate columns for side-by-side comparison.
Rows containing no defined slang word get empty scores and are passed through.

Requires: sentence-transformers, pyyaml  (see requirements_bert_slang_filter.txt)
Fine-tuning also requires: scikit-learn (pip install scikit-learn)

Usage:
    # Compare both zero-shot methods — write all rows with scores
    python3 bert_slang_filter.py input.csv -o scored.csv \\
        --slang-defs slang.yaml --score-all

    # Fine-tune on annotated CSVs, then score (saves model to ./ft_model/)
    python3 bert_slang_filter.py input.csv -o scored.csv \\
        --slang-defs slang.yaml --score-all \\
        --ft-annotations lang_quality_filtered_contexts/ \\
        --ft-model-dir ./ft_model/

    # Reuse saved fine-tuned model (skip training)
    python3 bert_slang_filter.py input.csv -o scored.csv \\
        --slang-defs slang.yaml --score-all --ft-model-dir ./ft_model/

    # Process a whole directory, one output file per input
    python3 bert_slang_filter.py contexts/ --output-dir scored/ \\
        --slang-defs slang.yaml --score-all

    # Filter using all three scores
    python3 bert_slang_filter.py contexts/*.csv -o filtered.csv \\
        --slang-defs slang.yaml --sbert-threshold 0.1 --nli-threshold 0.1 --ft-threshold 0.1

    # SBERT only, lighter model
    python3 bert_slang_filter.py input.csv -o out.csv \\
        --slang-defs slang.yaml --no-nli --sbert-model all-MiniLM-L6-v2
"""

from __future__ import annotations

import argparse
import csv
import glob
import logging
import sys
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from sentence_transformers import CrossEncoder, SentenceTransformer

try:
    import yaml  # type: ignore
    HAS_YAML = True
except ImportError:
    HAS_YAML = False

try:
    from tqdm import tqdm  # type: ignore
    HAS_TQDM = True
except ImportError:
    HAS_TQDM = False

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

_SBERT_MODEL = "all-mpnet-base-v2"
_NLI_MODEL   = "cross-encoder/nli-deberta-v3-large"

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _batched(items: list, size: int) -> Iterator[list]:
    for i in range(0, len(items), size):
        yield items[i : i + size]


def _read_rows(path: Path) -> List[Dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def load_slang_definitions(path: Path) -> Dict:
    if not HAS_YAML:
        log.error("PyYAML is required. pip install pyyaml")
        sys.exit(1)
    with path.open(encoding="utf-8") as fh:
        data = yaml.safe_load(fh)
    if not isinstance(data, dict):
        log.error("Slang definitions file must be a YAML mapping of word -> senses.")
        sys.exit(1)
    return data

# ---------------------------------------------------------------------------
# Method 1 — SBERT prototype matching
# ---------------------------------------------------------------------------

class SBERTSlangClassifier:
    """Sentence-BERT prototype-based slang sense scorer.

    Encodes full context sentences and compares to per-word sense centroids
    built from slang.yaml example sentences.
    """

    def __init__(self, model: SentenceTransformer, definitions: Dict) -> None:
        self.model = model
        self.prototypes: Dict[str, Tuple[torch.Tensor, Optional[torch.Tensor]]] = {}
        self._build_prototypes(definitions)

    def _encode(self, sentences: List[str], batch_size: int = 32) -> torch.Tensor:
        embs = self.model.encode(
            sentences, batch_size=batch_size,
            show_progress_bar=HAS_TQDM, convert_to_numpy=True,
        )
        return F.normalize(torch.tensor(embs), dim=1)

    def _build_prototypes(self, definitions: Dict) -> None:
        for word, senses in definitions.items():
            word_lc     = word.lower()
            slang_sents = senses.get("slang", [])
            std_sents   = senses.get("standard", [])

            if not slang_sents:
                log.warning("SBERT: skipping '%s' — no slang examples.", word)
                continue

            slang_proto = F.normalize(self._encode(slang_sents).mean(dim=0), dim=0)

            if std_sents:
                std_proto = F.normalize(self._encode(std_sents).mean(dim=0), dim=0)
                self.prototypes[word_lc] = (slang_proto, std_proto)
                log.info("  SBERT '%s' — %d slang / %d standard examples",
                         word_lc, len(slang_sents), len(std_sents))
            else:
                self.prototypes[word_lc] = (slang_proto, None)
                log.info("  SBERT '%s' — %d slang examples, slang-only mode",
                         word_lc, len(slang_sents))

    def score_rows(self, rows: List[Dict[str, str]], batch_size: int) -> List[Optional[float]]:
        texts  = [r["target_context"] for r in rows]
        scores: List[Optional[float]] = [None] * len(rows)

        for word, (slang_proto, std_proto) in self.prototypes.items():
            relevant = [i for i, t in enumerate(texts) if word in t.lower()]
            if not relevant:
                continue

            log.info("  SBERT scoring '%s': %d rows ...", word, len(relevant))
            embs = self._encode([texts[i] for i in relevant], batch_size)  # (N, D), L2-normed
            slang_sims = (embs @ slang_proto).tolist()

            if std_proto is not None:
                std_sims    = (embs @ std_proto).tolist()
                word_scores = [float(s - t) for s, t in zip(slang_sims, std_sims)]
            else:
                word_scores = [float(s) for s in slang_sims]

            for idx, s in zip(relevant, word_scores):
                if scores[idx] is None or s > scores[idx]:
                    scores[idx] = s

        return scores

# ---------------------------------------------------------------------------
# Method 2 — Zero-shot NLI
# ---------------------------------------------------------------------------

class NLISlangClassifier:
    """Zero-shot NLI slang sense scorer.

    Uses a cross-encoder NLI model to score entailment for slang vs. standard
    sense hypotheses. No prototype examples needed.
    """

    _SLANG_TMPL = "In this text, the word '{word}' is used as internet slang or informal language."
    _STD_TMPL   = "In this text, the word '{word}' is used with its literal, standard dictionary meaning."

    def __init__(self, model: CrossEncoder, words: List[str]) -> None:
        self.model = model
        self.words = [w.lower() for w in words]
        label2id   = getattr(model.model.config, "label2id", {})
        label2id_lc = {k.lower(): v for k, v in label2id.items()}
        self.entailment_idx = label2id_lc.get("entailment", 1)
        log.info("  NLI entailment class index: %d  (label2id: %s)",
                 self.entailment_idx, label2id)

    def _entailment_probs(self, pairs: List[Tuple[str, str]], batch_size: int) -> np.ndarray:
        logits = self.model.predict(pairs, batch_size=batch_size,
                                    show_progress_bar=HAS_TQDM)  # (N, 3)
        if logits.ndim == 1:
            logits = logits.reshape(1, -1)
        exp_x = np.exp(logits - logits.max(axis=1, keepdims=True))
        probs  = exp_x / exp_x.sum(axis=1, keepdims=True)
        return probs[:, self.entailment_idx]

    def score_rows(self, rows: List[Dict[str, str]], batch_size: int) -> List[Optional[float]]:
        texts  = [r["target_context"] for r in rows]
        scores: List[Optional[float]] = [None] * len(rows)

        for word in self.words:
            relevant = [i for i, t in enumerate(texts) if word in t.lower()]
            if not relevant:
                continue

            rel_texts   = [texts[i] for i in relevant]
            slang_pairs = [(t, self._SLANG_TMPL.format(word=word)) for t in rel_texts]
            std_pairs   = [(t, self._STD_TMPL.format(word=word))   for t in rel_texts]

            # Single predict call: slang pairs first, then standard
            log.info("  NLI scoring '%s': %d rows ...", word, len(rel_texts))
            all_probs   = self._entailment_probs(slang_pairs + std_pairs, batch_size)
            n           = len(rel_texts)
            slang_probs = all_probs[:n]
            std_probs   = all_probs[n:]

            for idx, sp, tp in zip(relevant, slang_probs, std_probs):
                s = float(sp - tp)
                if scores[idx] is None or s > scores[idx]:
                    scores[idx] = s

        return scores

# ---------------------------------------------------------------------------
# Method 3 — Fine-tuned BERT classifier
# ---------------------------------------------------------------------------

class _SlangDataset(Dataset):
    """Tokenised binary classification dataset stored in memory."""

    def __init__(self, texts: List[str], labels: List[int], tokenizer, max_len: int = 256) -> None:
        enc = tokenizer(texts, padding=True, truncation=True, max_length=max_len, return_tensors="pt")
        self.input_ids = enc["input_ids"]
        self.attention_mask = enc["attention_mask"]
        self.labels = torch.tensor(labels, dtype=torch.long)

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, i: int) -> Dict:
        return {
            "input_ids": self.input_ids[i],
            "attention_mask": self.attention_mask[i],
            "labels": self.labels[i],
        }


def load_annotation_examples(annotation_dir: Path) -> List[Tuple[str, int]]:
    """Recursively load (text, label) pairs from CSVs with an 'annotation' column.

    Expects rows where annotation == 'True' (slang) or 'False' (not slang).
    Rows with blank or missing annotations are silently skipped.
    """
    examples: List[Tuple[str, int]] = []
    for csv_path in sorted(annotation_dir.rglob("*.csv")):
        with csv_path.open(newline="", encoding="utf-8") as fh:
            for row in csv.DictReader(fh):
                a = (row.get("annotation") or "").strip()
                if a == "True":
                    examples.append((row["target_context"], 1))
                elif a == "False":
                    examples.append((row["target_context"], 0))
    n_pos = sum(e[1] for e in examples)
    log.info(
        "Loaded %d annotated examples from %s (%d slang, %d not-slang)",
        len(examples), annotation_dir, n_pos, len(examples) - n_pos,
    )
    return examples


class FineTunedSlangClassifier:
    """Binary BERT classifier fine-tuned on human-annotated slang context examples.

    Scores are mapped from P(slang) ∈ [0, 1] → [-1, 1] via ``2p - 1`` so they
    are on the same scale as the NLI scores and thresholds are interchangeable.
    """

    _DEFAULT_BASE = "bert-base-uncased"

    def __init__(
        self,
        base_model: str = _DEFAULT_BASE,
        words: Optional[List[str]] = None,
        device: Optional[str] = None,
    ) -> None:
        self.base_model = base_model
        self.words = [w.lower() for w in (words or [])]
        if device:
            self.device = device
        elif torch.cuda.is_available():
            self.device = "cuda"
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            self.device = "mps"
        else:
            self.device = "cpu"
        self.tokenizer = None
        self.model = None

    def train(
        self,
        examples: List[Tuple[str, int]],
        model_save_dir: Path,
        epochs: int = 10,
        lr: float = 2e-5,
        batch_size: int = 16,
    ) -> None:
        """Fine-tune a BERT sequence classifier and save the best checkpoint."""
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
            log.error("Too few annotated examples (%d) for reliable fine-tuning.", len(examples))
            sys.exit(1)

        texts  = [e[0] for e in examples]
        labels = [e[1] for e in examples]

        train_texts, val_texts, train_labels, val_labels = train_test_split(
            texts, labels, test_size=0.2, random_state=42, stratify=labels
        )
        log.info(
            "Fine-tune split — Train: %d (%d pos / %d neg) | Val: %d (%d pos / %d neg)",
            len(train_texts), sum(train_labels), len(train_labels) - sum(train_labels),
            len(val_texts),  sum(val_labels),  len(val_labels)  - sum(val_labels),
        )

        log.info("Loading base model for fine-tuning: %s", self.base_model)
        tokenizer = AutoTokenizer.from_pretrained(self.base_model)
        model = AutoModelForSequenceClassification.from_pretrained(
            self.base_model, num_labels=2
        ).to(self.device)

        n_pos = sum(labels)
        n_neg = len(labels) - n_pos
        class_weights = torch.tensor(
            [1.0, n_neg / max(n_pos, 1)], dtype=torch.float, device=self.device
        )
        log.info("Class weights: not-slang=%.2f, slang=%.2f",
                 class_weights[0].item(), class_weights[1].item())

        train_ds  = _SlangDataset(train_texts, train_labels, tokenizer)
        val_ds    = _SlangDataset(val_texts,   val_labels,   tokenizer)
        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
        val_loader   = DataLoader(val_ds,   batch_size=batch_size)

        optimizer    = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
        total_steps  = len(train_loader) * epochs
        scheduler    = get_linear_schedule_with_warmup(
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
                    preds = model(input_ids=input_ids, attention_mask=attention_mask).logits.argmax(dim=1).cpu()
                    correct += (preds == batch["labels"]).sum().item()
                    total   += len(batch["labels"])

            val_acc = correct / total if total else 0.0
            avg_loss = total_loss / len(train_loader)
            log.info("Epoch %d/%d — loss: %.4f, val_acc: %.3f", epoch, epochs, avg_loss, val_acc)

            if val_acc >= best_val_acc:
                best_val_acc = val_acc
                model.save_pretrained(str(model_save_dir))
                tokenizer.save_pretrained(str(model_save_dir))
                log.info("  Saved best model (val_acc=%.3f) -> %s", val_acc, model_save_dir)

        log.info("Fine-tuning complete. Best val_acc: %.3f", best_val_acc)
        self.tokenizer = tokenizer
        self.model     = model
        self.model.eval()

    def load(self, model_dir: Path) -> None:
        """Load a previously saved fine-tuned model."""
        try:
            from transformers import AutoTokenizer, AutoModelForSequenceClassification  # type: ignore
        except ImportError as exc:
            log.error("transformers is required to load a fine-tuned model: %s", exc)
            sys.exit(1)
        log.info("Loading fine-tuned model from %s", model_dir)
        self.tokenizer = AutoTokenizer.from_pretrained(str(model_dir))
        self.model     = AutoModelForSequenceClassification.from_pretrained(str(model_dir)).to(self.device)
        self.model.eval()

    def score_rows(self, rows: List[Dict[str, str]], batch_size: int) -> List[Optional[float]]:
        assert self.model is not None and self.tokenizer is not None, \
            "Call train() or load() before score_rows()."
        texts  = [r["target_context"] for r in rows]
        scores: List[Optional[float]] = [None] * len(rows)

        for word in self.words:
            relevant = [i for i, t in enumerate(texts) if word in t.lower()]
            if not relevant:
                continue
            log.info("  FT scoring '%s': %d rows ...", word, len(relevant))
            rel_texts = [texts[i] for i in relevant]

            all_probs: List[float] = []
            for batch_texts in _batched(rel_texts, batch_size):
                enc = self.tokenizer(
                    batch_texts, padding=True, truncation=True,
                    max_length=256, return_tensors="pt",
                ).to(self.device)
                with torch.no_grad():
                    probs = torch.softmax(
                        self.model(**enc).logits, dim=1
                    )[:, 1].cpu().tolist()  # P(slang)
                all_probs.extend(probs)

            for idx, p in zip(relevant, all_probs):
                s = float(2 * p - 1)  # [0,1] → [-1,1] to match NLI score range
                if scores[idx] is None or s > scores[idx]:
                    scores[idx] = s

        return scores


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Score slang sense using SBERT prototype matching and zero-shot NLI.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Compare both methods on all rows (recommended starting point)
  python bert_slang_filter.py input.csv -o scored.csv \\
      --slang-defs slang.yaml --score-all

  # Process a whole directory, one output file per input
  python bert_slang_filter.py contexts/ --output-dir scored/ \\
      --slang-defs slang.yaml --score-all

  # Filter using both scores
  python bert_slang_filter.py contexts/*.csv -o filtered.csv \\
      --slang-defs slang.yaml --sbert-threshold 0.1 --nli-threshold 0.1

  # SBERT only, lighter model
  python bert_slang_filter.py input.csv -o out.csv \\
      --slang-defs slang.yaml --no-nli --sbert-model all-MiniLM-L6-v2

  # NLI only
  python bert_slang_filter.py input.csv -o out.csv \\
      --slang-defs slang.yaml --no-sbert
        """,
    )

    p.add_argument("inputs", nargs="+", metavar="CSV",
                   help="Input CSV file(s), glob patterns, or a directory of CSVs. "
                        "When a directory is given, all *.csv files within it are processed.")

    out_group = p.add_mutually_exclusive_group(required=True)
    out_group.add_argument("-o", "--output", metavar="FILE",
                           help="Output CSV path (single-file mode — all inputs merged).")
    out_group.add_argument("--output-dir", metavar="DIR", dest="output_dir",
                           help="Output directory (per-file mode — one output file per input, "
                                "same filename). Created if it does not exist.")
    p.add_argument("--slang-defs", required=True, metavar="YAML", dest="slang_defs",
                   help="YAML file of slang word definitions (slang.yaml).")
    p.add_argument("--score-all", action="store_true", dest="score_all",
                   help="Write every row with scores attached, without filtering. "
                        "Recommended for comparing the two methods.")

    # Thresholds
    p.add_argument("--sbert-threshold", type=float, default=0.0, metavar="F",
                   dest="sbert_threshold",
                   help="Min SBERT score to keep a row (default: 0.0).")
    p.add_argument("--nli-threshold", type=float, default=0.0, metavar="F",
                   dest="nli_threshold",
                   help="Min NLI score to keep a row (default: 0.0).")

    # Method toggles
    p.add_argument("--no-sbert", action="store_true", dest="no_sbert",
                   help="Skip SBERT scoring.")
    p.add_argument("--no-nli", action="store_true", dest="no_nli",
                   help="Skip NLI scoring.")

    # Models
    p.add_argument("--sbert-model", default=_SBERT_MODEL, metavar="ID", dest="sbert_model",
                   help=f"Sentence-transformer model for SBERT scoring "
                        f"(default: {_SBERT_MODEL}). "
                        f"Lighter alternative: all-MiniLM-L6-v2 (~80 MB).")
    p.add_argument("--nli-model", default=_NLI_MODEL, metavar="ID", dest="nli_model",
                   help=f"Cross-encoder NLI model "
                        f"(default: {_NLI_MODEL}). "
                        f"Lighter alternative: cross-encoder/nli-MiniLM2-L6-H768 (~100 MB).")
    p.add_argument("--batch-size", type=int, default=32, metavar="N", dest="batch_size",
                   help="Inference batch size for both methods (default: 32). "
                        "Reduce if hitting GPU out-of-memory errors.")
    p.add_argument("--device", default=None, metavar="DEV", dest="device",
                   help="PyTorch device for inference, e.g. cpu, cuda, mps. "
                        "Defaults to auto-detect. Use --device cpu to avoid MPS OOM errors.")

    # Fine-tuned classifier
    ft = p.add_argument_group(
        "Fine-tuned classifier",
        "Trains a BERT sequence classifier on human-annotated CSVs and scores with it. "
        "Supply --ft-annotations to (re-)train, or just --ft-model-dir to load a saved model.",
    )
    ft.add_argument("--ft-annotations", metavar="DIR", dest="ft_annotations",
                    help="Directory of annotated CSVs (scanned recursively for *.csv files "
                         "with an 'annotation' column containing 'True'/'False' labels).")
    ft.add_argument("--ft-model-dir", metavar="DIR", dest="ft_model_dir",
                    help="Where to save (training) or load (inference) the fine-tuned model.")
    ft.add_argument("--ft-base-model", default=FineTunedSlangClassifier._DEFAULT_BASE,
                    metavar="ID", dest="ft_base_model",
                    help=f"Base HuggingFace model to fine-tune "
                         f"(default: {FineTunedSlangClassifier._DEFAULT_BASE}).")
    ft.add_argument("--ft-epochs", type=int, default=10, metavar="N", dest="ft_epochs",
                    help="Number of fine-tuning epochs (default: 10).")
    ft.add_argument("--ft-lr", type=float, default=2e-5, metavar="F", dest="ft_lr",
                    help="Learning rate for fine-tuning (default: 2e-5).")
    ft.add_argument("--ft-threshold", type=float, default=0.0, metavar="F", dest="ft_threshold",
                    help="Min ft_score to keep a row when filtering (default: 0.0).")

    return p


def _score_file(
    path: Path,
    sbert_clf: Optional[SBERTSlangClassifier],
    nli_clf: Optional[NLISlangClassifier],
    ft_clf: Optional[FineTunedSlangClassifier],
    args: argparse.Namespace,
    fieldnames: List[str],
) -> Tuple[List[Dict], int]:
    """Score a single CSV file. Returns (out_rows, n_input_rows)."""
    rows = _read_rows(path)
    if not rows:
        log.info("Skipping empty file: %s", path.name)
        return [], 0

    log.info("Processing %s (%d rows) ...", path.name, len(rows))
    n = len(rows)
    sbert_scores = sbert_clf.score_rows(rows, args.batch_size) if sbert_clf else [None] * n
    nli_scores   = nli_clf.score_rows(rows, args.batch_size)   if nli_clf   else [None] * n
    ft_scores    = ft_clf.score_rows(rows, args.batch_size)    if ft_clf    else [None] * n

    out_rows: List[Dict] = []
    for row, ss, ns, fs in zip(rows, sbert_scores, nli_scores, ft_scores):
        if not args.score_all:
            if sbert_clf and ss is not None and ss < args.sbert_threshold:
                continue
            if nli_clf   and ns is not None and ns < args.nli_threshold:
                continue
            if ft_clf    and fs is not None and fs < args.ft_threshold:
                continue

        out_row: Dict = {"uri": row["uri"], "target_context": row["target_context"]}
        if not args.no_sbert:
            out_row["sbert_score"] = f"{ss:.4f}" if ss is not None else ""
        if not args.no_nli:
            out_row["nli_score"] = f"{ns:.4f}" if ns is not None else ""
        if ft_clf is not None:
            out_row["ft_score"] = f"{fs:.4f}" if fs is not None else ""
        out_rows.append(out_row)

    log.info("  %s: %d/%d rows kept (%.1f%%)",
             path.name, len(out_rows), n,
             100.0 * len(out_rows) / n if n else 0.0)
    return out_rows, n


def _sort_rows(rows: List[Dict]) -> None:
    """Sort rows in-place: descending sbert_score, nli_score, ft_score; empty sorts last."""
    _NEG_INF = float("-inf")

    def _key(r: Dict) -> Tuple[float, float, float]:
        ss = float(r["sbert_score"]) if r.get("sbert_score") else _NEG_INF
        ns = float(r["nli_score"])   if r.get("nli_score")   else _NEG_INF
        fs = float(r["ft_score"])    if r.get("ft_score")    else _NEG_INF
        return (-ss, -ns, -fs)

    rows.sort(key=_key)


def _write_csv(path: Path, rows: List[Dict], fieldnames: List[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = build_parser()
    args   = parser.parse_args()

    using_ft = bool(args.ft_model_dir or args.ft_annotations)
    if args.no_sbert and args.no_nli and not using_ft:
        parser.error("All methods are disabled — nothing to do.")

    # Expand inputs: directories → *.csv within; glob patterns; plain paths
    input_paths: List[Path] = []
    for pattern in args.inputs:
        p = Path(pattern)
        if p.is_dir():
            input_paths.extend(sorted(p.glob("*.csv")))
        else:
            matched = sorted(glob.glob(pattern, recursive=True))
            if matched:
                input_paths.extend(Path(m) for m in matched)
            else:
                input_paths.append(p)

    missing = [p for p in input_paths if not p.exists()]
    if missing:
        parser.error(f"File(s) not found: {', '.join(str(p) for p in missing)}")

    if not input_paths:
        parser.error("No input files found.")

    if len(input_paths) > 1 and args.output_dir is None:
        log.warning(
            "%d input files detected but -o is set — all rows will be merged into one file "
            "and nothing is written until the entire run finishes. "
            "Use --output-dir to write one output file per input file.",
            len(input_paths),
        )

    definitions = load_slang_definitions(Path(args.slang_defs))

    # ── Load models ───────────────────────────────────────────────────────────
    device = args.device  # None → auto-detect
    if device:
        log.info("Using device: %s", device)

    sbert_clf: Optional[SBERTSlangClassifier] = None
    if not args.no_sbert:
        log.info("Loading SBERT model: %s", args.sbert_model)
        sbert_clf = SBERTSlangClassifier(
            SentenceTransformer(args.sbert_model, device=device), definitions
        )

    nli_clf: Optional[NLISlangClassifier] = None
    if not args.no_nli:
        log.info("Loading NLI model: %s", args.nli_model)
        nli_clf = NLISlangClassifier(
            CrossEncoder(args.nli_model, num_labels=3, device=device),
            list(definitions.keys()),
        )

    ft_clf: Optional[FineTunedSlangClassifier] = None
    if using_ft:
        if not args.ft_model_dir:
            parser.error("--ft-annotations requires --ft-model-dir to specify where to save the model.")
        ft_model_dir = Path(args.ft_model_dir)
        ft_clf = FineTunedSlangClassifier(
            base_model=args.ft_base_model,
            words=list(definitions.keys()),
            device=device,
        )
        if args.ft_annotations:
            annotation_dir = Path(args.ft_annotations)
            if not annotation_dir.exists():
                parser.error(f"Annotation directory not found: {annotation_dir}")
            examples = load_annotation_examples(annotation_dir)
            if not examples:
                parser.error("No annotated examples found — check that CSVs have an 'annotation' column.")
            ft_clf.train(
                examples, ft_model_dir,
                epochs=args.ft_epochs, lr=args.ft_lr, batch_size=args.batch_size,
            )
        elif ft_model_dir.exists():
            ft_clf.load(ft_model_dir)
        else:
            parser.error(
                f"--ft-model-dir '{ft_model_dir}' does not exist. "
                "Provide --ft-annotations to train a model first."
            )

    fieldnames = ["uri", "target_context"]
    if not args.no_sbert:
        fieldnames.append("sbert_score")
    if not args.no_nli:
        fieldnames.append("nli_score")
    if ft_clf is not None:
        fieldnames.append("ft_score")

    # ── Per-file mode (--output-dir) ──────────────────────────────────────────
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
            file_rows, n_in = _score_file(path, sbert_clf, nli_clf, ft_clf, args, fieldnames)
            _sort_rows(file_rows)
            _write_csv(out_path, file_rows, fieldnames)
            grand_in  += n_in
            grand_out += len(file_rows)
        log.info("Done. %d/%d rows kept (%.1f%%) across %d file(s); %d skipped (already complete) -> %s",
                 grand_out, grand_in,
                 100.0 * grand_out / grand_in if grand_in else 0.0,
                 len(input_paths) - grand_skipped, grand_skipped, out_dir)
        return

    # ── Single-file mode (-o) ─────────────────────────────────────────────────
    output_path = Path(args.output)
    grand_in  = 0
    all_rows: List[Dict] = []
    for path in input_paths:
        file_rows, n_in = _score_file(path, sbert_clf, nli_clf, ft_clf, args, fieldnames)
        all_rows.extend(file_rows)
        grand_in += n_in

    _sort_rows(all_rows)
    _write_csv(output_path, all_rows, fieldnames)

    log.info("Done. %d/%d rows kept (%.1f%%), sorted by score -> %s",
             len(all_rows), grand_in,
             100.0 * len(all_rows) / grand_in if grand_in else 0.0,
             output_path)


if __name__ == "__main__":
    main()
