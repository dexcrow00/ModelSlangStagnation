#!/usr/bin/env python3
"""
bert_filter.py — Filter word_context.py CSV output using BERT-based classifiers.

Two classifiers run on each target_context chunk:

  1. Language — papluca/xlm-roberta-base-language-detection
     Drops rows where English-class confidence < --lang-threshold (default 0.85).

  2. Writing quality — textattack/bert-base-uncased-CoLA
     BERT base fine-tuned on the Corpus of Linguistic Acceptability.
     Drops rows where the "acceptable" confidence < --quality-threshold (default 0.70).
     Low-quality text (keyword strings, SEO spam, nav menus, etc.) scores poorly
     on linguistic acceptability and is removed.

Both filters can be disabled independently. Use --score-all to write every row
with its scores attached (useful for calibrating thresholds before filtering).

Usage:
    python bert_filter.py contexts/word_context_CC-MAIN-2024-10.csv -o filtered.csv
    python bert_filter.py contexts/*.csv -o filtered.csv --quality-threshold 0.8
    python bert_filter.py input.csv -o scored.csv --score-all --keep-scores
    python bert_filter.py input.csv -o out.csv --no-lang-filter --device cuda
"""

from __future__ import annotations

import argparse
import csv
import glob
import logging
import os
import sys
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Tuple

# Suppress tokenizer fork warnings before any transformers import.
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

try:
    import torch
    from transformers import pipeline as hf_pipeline  # type: ignore
except ImportError:
    print(
        "ERROR: transformers and torch are required.\n"
        "  pip install transformers torch",
        file=sys.stderr,
    )
    sys.exit(1)

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

# ---------------------------------------------------------------------------
# Default models
# ---------------------------------------------------------------------------

_LANG_MODEL    = "papluca/xlm-roberta-base-language-detection"
_QUALITY_MODEL = "textattack/bert-base-uncased-CoLA"

# ---------------------------------------------------------------------------
# Device detection
# ---------------------------------------------------------------------------

def _detect_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"

# ---------------------------------------------------------------------------
# Score extraction
# ---------------------------------------------------------------------------

def _english_score(label_scores: List[Dict]) -> float:
    """Probability assigned to the 'en' class by the language detection model."""
    for r in label_scores:
        if r["label"] == "en":
            return float(r["score"])
    return 0.0


def _acceptable_score(label_scores: List[Dict]) -> float:
    """Probability assigned to the 'acceptable' class by the CoLA model.

    textattack/bert-base-uncased-CoLA maps:
      LABEL_0 → unacceptable
      LABEL_1 → acceptable
    """
    for r in label_scores:
        if r["label"] in ("LABEL_1", "acceptable"):
            return float(r["score"])
    return 0.0

# ---------------------------------------------------------------------------
# CSV helpers
# ---------------------------------------------------------------------------

def _read_rows(path: Path) -> List[Dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def _batched(items: list, size: int) -> Iterator[list]:
    for i in range(0, len(items), size):
        yield items[i : i + size]

# ---------------------------------------------------------------------------
# Core filtering
# ---------------------------------------------------------------------------

def score_rows(
    rows: List[Dict[str, str]],
    lang_pipe,
    quality_pipe,
    batch_size: int,
    lang_threshold: float,
) -> List[Tuple[Dict[str, str], float, float]]:
    """
    Run both classifiers and return (row, lang_score, quality_score) for every row.

    Quality scoring is skipped for rows that already fail the language threshold,
    saving inference time.
    """
    n = len(rows)
    texts = [r["target_context"] for r in rows]
    lang_scores  = [1.0] * n  # default: pass (used when lang filter disabled)
    quality_scores = [1.0] * n

    # ── Language pass ─────────────────────────────────────────────────────────
    if lang_pipe is not None:
        batches = list(_batched(texts, batch_size))
        it = tqdm(batches, desc="Language", unit="batch", leave=False) if HAS_TQDM else batches
        offset = 0
        for batch in it:
            results = lang_pipe(batch)
            for j, label_list in enumerate(results):
                lang_scores[offset + j] = _english_score(label_list)
            offset += len(batch)

    # ── Quality pass (only on rows that pass the language threshold) ──────────
    if quality_pipe is not None:
        passing_indices = [i for i, ls in enumerate(lang_scores) if ls >= lang_threshold]
        quality_texts = [texts[i] for i in passing_indices]

        q_scores_flat: List[float] = []
        batches = list(_batched(quality_texts, batch_size))
        it = tqdm(batches, desc="Quality", unit="batch", leave=False) if HAS_TQDM else batches
        for batch in it:
            results = quality_pipe(batch)
            for label_list in results:
                q_scores_flat.append(_acceptable_score(label_list))

        for i, qs in zip(passing_indices, q_scores_flat):
            quality_scores[i] = qs

    return [(row, ls, qs) for row, ls, qs in zip(rows, lang_scores, quality_scores)]

# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Filter word_context.py CSV output using BERT-based classifiers.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Filter a single dump
  python bert_filter.py contexts/word_context_CC-MAIN-2024-10.csv -o filtered.csv

  # Filter all dumps, stricter quality gate, GPU
  python bert_filter.py contexts/*.csv -o filtered.csv --quality-threshold 0.8 --device cuda

  # Write all rows with scores (no filtering) to calibrate thresholds
  python bert_filter.py input.csv -o scored.csv --score-all --keep-scores

  # Language filter only (skip quality classifier)
  python bert_filter.py input.csv -o out.csv --no-quality-filter
        """,
    )

    parser.add_argument("inputs", nargs="+", metavar="CSV",
                        help="Input CSV file(s) from word_context.py. Glob patterns accepted.")
    parser.add_argument("-o", "--output", required=True, metavar="FILE",
                        help="Output CSV path.")

    # Thresholds
    parser.add_argument("--lang-threshold", type=float, default=0.85, metavar="F",
                        dest="lang_threshold",
                        help="Minimum English confidence to keep a row (default: 0.85).")
    parser.add_argument("--quality-threshold", type=float, default=0.70, metavar="F",
                        dest="quality_threshold",
                        help="Minimum linguistic-acceptability confidence to keep a row "
                             "(default: 0.70).")

    # Filter toggles
    parser.add_argument("--no-lang-filter", action="store_true", dest="no_lang",
                        help="Skip language detection.")
    parser.add_argument("--no-quality-filter", action="store_true", dest="no_quality",
                        help="Skip writing-quality classification.")

    # Output options
    parser.add_argument("--keep-scores", action="store_true", dest="keep_scores",
                        help="Append lang_score and quality_score columns to the output.")
    parser.add_argument("--score-all", action="store_true", dest="score_all",
                        help="Write every row (no filtering), useful for threshold calibration. "
                             "Implies --keep-scores.")

    # Inference options
    parser.add_argument("--batch-size", type=int, default=32, metavar="N", dest="batch_size",
                        help="Inference batch size (default: 32; increase for GPU).")
    parser.add_argument("--device", default=None,
                        help="Inference device: cpu | cuda | cuda:N | mps. "
                             "Auto-detected if omitted.")

    # Model overrides
    parser.add_argument("--lang-model", default=_LANG_MODEL, metavar="ID", dest="lang_model",
                        help=f"HuggingFace language-detection model (default: {_LANG_MODEL}).")
    parser.add_argument("--quality-model", default=_QUALITY_MODEL, metavar="ID",
                        dest="quality_model",
                        help=f"HuggingFace quality-classification model "
                             f"(default: {_QUALITY_MODEL}).")

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    # --score-all implies --keep-scores
    if args.score_all:
        args.keep_scores = True

    # Expand glob patterns
    input_paths: List[Path] = []
    for pattern in args.inputs:
        matched = sorted(glob.glob(pattern, recursive=True))
        input_paths.extend(Path(p) for p in matched) if matched else input_paths.append(Path(pattern))

    missing = [p for p in input_paths if not p.exists()]
    if missing:
        parser.error(f"File(s) not found: {', '.join(str(p) for p in missing)}")

    device = args.device or _detect_device()
    log.info("Device: %s", device)

    # ── Load models ───────────────────────────────────────────────────────────
    lang_pipe = None
    quality_pipe = None

    if not args.no_lang:
        log.info("Loading language model: %s", args.lang_model)
        lang_pipe = hf_pipeline(
            "text-classification",
            model=args.lang_model,
            device=device,
            top_k=None,
        )

    if not args.no_quality:
        log.info("Loading quality model: %s", args.quality_model)
        quality_pipe = hf_pipeline(
            "text-classification",
            model=args.quality_model,
            device=device,
            top_k=None,
        )

    if lang_pipe is None and quality_pipe is None:
        parser.error("Both filters are disabled — nothing to do.")

    # ── Set up output ─────────────────────────────────────────────────────────
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = ["uri", "target_context"]
    if args.keep_scores:
        fieldnames += ["lang_score", "quality_score"]

    grand_in = grand_out = 0

    with output_path.open("w", newline="", encoding="utf-8") as out_fh:
        writer = csv.DictWriter(out_fh, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()

        for path in input_paths:
            rows = _read_rows(path)
            if not rows:
                log.info("Skipping empty file: %s", path.name)
                continue

            log.info("Processing %s (%d rows) …", path.name, len(rows))

            scored = score_rows(
                rows=rows,
                lang_pipe=lang_pipe,
                quality_pipe=quality_pipe,
                batch_size=args.batch_size,
                lang_threshold=args.lang_threshold,
            )

            kept = 0
            for row, ls, qs in scored:
                passes = args.score_all or (
                    ls >= args.lang_threshold and qs >= args.quality_threshold
                )
                if passes:
                    out_row: Dict = {"uri": row["uri"], "target_context": row["target_context"]}
                    if args.keep_scores:
                        out_row["lang_score"]    = f"{ls:.4f}"
                        out_row["quality_score"] = f"{qs:.4f}"
                    writer.writerow(out_row)
                    kept += 1

            grand_in  += len(rows)
            grand_out += kept
            log.info("  %s: %d/%d rows kept (%.1f%%)",
                     path.name, kept, len(rows),
                     100.0 * kept / len(rows) if rows else 0.0)

    log.info(
        "Done. %d/%d rows kept (%.1f%%) → %s",
        grand_out, grand_in,
        100.0 * grand_out / grand_in if grand_in else 0.0,
        output_path,
    )


if __name__ == "__main__":
    main()
