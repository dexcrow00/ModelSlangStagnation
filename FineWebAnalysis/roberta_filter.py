#!/usr/bin/env python3
"""
roberta_filter.py — Score/filter context rows with a fine-tuned RoBERTa slang
classifier.

  score = 2 * P(slang) - 1  (range [-1, 1]; positive = slang usage)

This is the inference/filtering half of the pipeline; fine-tune the model first
with finetune_roberta.py, which saves the model directory loaded here.

Requires: torch, transformers

Usage:

    # Score every row in a directory of Parquet contexts (from fineweb_context.py)
    python roberta_filter.py contexts/ --output-dir scored/ --score-all \\
        --roberta-model-dir ./roberta_model/

    # Filter above a score threshold into a single CSV
    python roberta_filter.py contexts/*.parquet -o filtered.csv \\
        --roberta-model-dir ./roberta_model/ --threshold 0.1
"""

from __future__ import annotations

import argparse
import csv
import glob
import logging
import sys
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Tuple

import pyarrow.parquet as pq

import torch

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

def _pick_device(device: Optional[str]) -> str:
    """Resolve an explicit device string, else auto-detect cuda/mps/cpu."""
    
    if device:
        return device
    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def _batched(items: list, size: int) -> Iterator[list]:
    for i in range(0, len(items), size):
        yield items[i : i + size]


def _read_rows(path: Path) -> List[Dict[str, str]]:
    """Read context rows from a Parquet (fineweb_context.py) or CSV file.

    Both yield dicts keyed by ``target`` / ``uri`` / ``target_context`` (the
    ``target`` column is present only for the Parquet output).
    """
    if path.suffix.lower() == ".parquet":
        table = pq.read_table(path)
        return [{k: ("" if v is None else str(v)) for k, v in row.items()}
                for row in table.to_pylist()]

    with path.open(newline="", encoding="utf-8-sig") as fh:
        rows = list(csv.DictReader(fh))
    # Normalise any quoted column names produced by Excel/PowerShell BOM exports
    normalised = []
    for row in rows:
        normalised.append({k.strip('"').strip(): v for k, v in row.items()})
    return normalised


# ---------------------------------------------------------------------------
# Transformer classifier (inference only)
# ---------------------------------------------------------------------------

class TransformerSlangClassifier:
    """Binary transformer classifier for slang-sense scoring (inference only).

    Loads a model fine-tuned by finetune_roberta.py and maps
    P(slang) ∈ [0, 1] → [-1, 1] via ``2p - 1``. Works with any HuggingFace
    AutoModelForSequenceClassification checkpoint (RoBERTa, BERT, …).

    Parameters
    ----------
    label  : Short name used in log messages, e.g. 'RoBERTa'.
    device : PyTorch device string, or None for auto-detect.
    """

    def __init__(self, label: str = "RoBERTa", device: Optional[str] = None) -> None:
        self.label     = label
        self.device    = _pick_device(device)
        if self.device:
            log.info("Using device: %s", self.device)
        self.tokenizer = None
        self.model     = None

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

    def score_rows(
        self,
        rows: List[Dict[str, str]],
        batch_size: int,
    ) -> List[Optional[float]]:
        """Return a score ∈ [-1, 1] for each row.

        Every row is scored directly on its ``target_context``: the context was
        already extracted around a specific target (carried in the ``target``
        column), so there is no need to re-select rows by word.
        """
        assert self.model is not None and self.tokenizer is not None, \
            f"[{self.label}] Call load() before score_rows()."

        texts = [r["target_context"] for r in rows]
        if not texts:
            return []
        log.info("  [%s] scoring %d rows ...", self.label, len(texts))

        scores: List[Optional[float]] = []
        for batch_texts in _batched(texts, batch_size):
            enc = self.tokenizer(
                batch_texts, padding=True, truncation=True,
                max_length=256, return_tensors="pt",
            ).to(self.device)
            with torch.no_grad():
                probs = (
                    torch.softmax(self.model(**enc).logits, dim=1)[:, 1].cpu().tolist()
                )
            scores.extend(float(2 * p - 1) for p in probs)  # [0,1] -> [-1,1]

        return scores


# ---------------------------------------------------------------------------
# Scoring helpers
# ---------------------------------------------------------------------------

_FIELDNAMES = ["target", "uri", "target_context", "roberta_score"]


def _sort_rows(rows: List[Dict]) -> None:
    _NEG_INF = float("-inf")
    rows.sort(key=lambda r: -(float(r["roberta_score"]) if r.get("roberta_score") else _NEG_INF))


def _score_file(
    path: Path,
    roberta_clf: TransformerSlangClassifier,
    args: argparse.Namespace,
) -> Tuple[List[Dict], int]:
    """Score a single context file with the RoBERTa classifier."""
    rows = _read_rows(path)
    if not rows:
        log.info("Skipping empty file: %s", path.name)
        return [], 0

    log.info("Processing %s (%d rows) ...", path.name, len(rows))
    n = len(rows)
    roberta_scores = roberta_clf.score_rows(rows, args.batch_size)

    out_rows: List[Dict] = []
    for row, rs in zip(rows, roberta_scores):
        out_row = {
            "target":         row.get("target", ""),
            "uri":            row.get("uri", ""),
            "target_context": row.get("target_context", ""),
            "roberta_score":  f"{rs:.4f}" if rs is not None else "",
        }

        if not args.score_all:
            if rs is None or rs < args.threshold:
                continue

        out_rows.append(out_row)

    log.info("  %s: %d/%d rows kept (%.1f%%)",
             path.name, len(out_rows), n,
             100.0 * len(out_rows) / n if n else 0.0)
    return out_rows, n


def _write_csv(path: Path, rows: List[Dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=_FIELDNAMES, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Score/filter slang sense using a fine-tuned RoBERTa classifier.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Score every row in a directory of Parquet contexts
  python roberta_filter.py contexts/ --output-dir scored/ --score-all \\
      --roberta-model-dir ./roberta_model/

  # Filter above a score threshold into a single CSV
  python roberta_filter.py contexts/*.parquet -o filtered.csv \\
      --roberta-model-dir ./roberta_model/ --threshold 0.1

  (Fine-tune the model first with finetune_roberta.py.)
        """,
    )

    # Inputs / outputs
    p.add_argument("inputs", nargs="+", metavar="FILE",
                   help="Input context file(s) — Parquet (from fineweb_context.py) "
                        "or CSV — glob patterns, or a directory of such files.")
    out_group = p.add_mutually_exclusive_group(required=True)
    out_group.add_argument("-o", "--output", metavar="FILE",
                           help="Output CSV path (all inputs merged into one file).")
    out_group.add_argument("--output-dir", metavar="DIR", dest="output_dir",
                           help="Output directory (one output file per input).")

    # Shared options
    p.add_argument("--score-all", action="store_true", dest="score_all",
                   help="Write every row with scores attached (no filtering).")
    p.add_argument("--threshold", type=float, default=0.0, metavar="F",
                   help="Min roberta_score to keep a row when filtering (default: 0.0).")
    p.add_argument("--batch-size", type=int, default=32, metavar="N", dest="batch_size",
                   help="Inference batch size (default: 32).")
    p.add_argument("--device", default=None, metavar="DEV",
                   help="PyTorch device, e.g. cpu, cuda, mps (default: auto-detect).")

    # RoBERTa model
    p.add_argument("--roberta-model-dir", default="FineWebAnalysis/ft_model_roberta", metavar="DIR",
                   dest="roberta_model_dir",
                   help="Directory of the fine-tuned RoBERTa model to load "
                        "(produced by finetune_roberta.py).")

    return p


def _load_classifier(
    model_dir_str: str,
    device: Optional[str],
    parser: argparse.ArgumentParser,
    label: str = "RoBERTa",
) -> TransformerSlangClassifier:
    """Load a fine-tuned TransformerSlangClassifier for inference."""
    model_dir = Path(model_dir_str)
    if not model_dir.exists():
        parser.error(
            f"[{label}] Model directory '{model_dir}' not found. "
            f"Fine-tune one first with finetune_roberta.py."
        )
    clf = TransformerSlangClassifier(label=label, device=device)
    clf.load(model_dir)
    return clf


def main() -> None:
    parser = build_parser()
    args   = parser.parse_args()

    # Expand inputs
    input_paths: List[Path] = []
    for pattern in args.inputs:
        p = Path(pattern)
        if p.is_dir():
            input_paths.extend(sorted(list(p.glob("*.parquet")) + list(p.glob("*.csv"))))
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

    device = args.device

    roberta_clf = _load_classifier(args.roberta_model_dir, device, parser)

    score_kwargs = dict(roberta_clf=roberta_clf, args=args)

    # ── Per-file mode ──────────────────────────────────────────────────────────
    if args.output_dir is not None:
        out_dir = Path(args.output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        grand_in = grand_out = grand_skipped = 0
        for path in input_paths:
            out_path = out_dir / (path.stem + ".csv")
            if out_path.exists():
                log.info("Skipping %s — already exists in %s", path.name, out_dir)
                grand_skipped += 1
                continue
            file_rows, n_in = _score_file(path, **score_kwargs)
            _sort_rows(file_rows)
            _write_csv(out_path, file_rows)
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

    _sort_rows(all_rows)
    _write_csv(output_path, all_rows)
    log.info(
        "Done. %d/%d rows kept (%.1f%%), sorted by score -> %s",
        len(all_rows), grand_in,
        100.0 * len(all_rows) / grand_in if grand_in else 0.0,
        output_path,
    )


if __name__ == "__main__":
    main()
