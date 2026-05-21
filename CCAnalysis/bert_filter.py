#!/usr/bin/env python3
"""
bert_filter.py — Filter word_context.py CSV output using BERT-based classifiers.

Three classifiers run on each target_context chunk:

  1. Language — langdetect (lightweight, no neural model)
     Drops rows where English-language probability < --lang-threshold (default 0.85).

  2. Writing quality — heuristic scorer (no ML, no torch required)
     Scores based on function-word ratio, comma density, and separator density.
     Catches keyword strings, SEO spam, and nav menus without loading any model.
     Drops rows below --quality-threshold (default 0.70).

This script runs on langdetect + pure-Python heuristics only.
Slang detection will now be handled offline by /bert_slang_filter.py

Usage:
    # Language + quality only (no torch needed)
    python bert_filter.py contexts/*.csv -o filtered.csv
"""

from __future__ import annotations

import argparse
import csv
import glob
import logging
import sys
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Tuple

try:
    from langdetect import DetectorFactory, LangDetectException, detect_langs  # type: ignore
    DetectorFactory.seed = 0
    HAS_LANGDETECT = True
except ImportError:
    HAS_LANGDETECT = False

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
# Filter 1 — Language detection (langdetect, no neural model)
# ---------------------------------------------------------------------------

def _english_score(text: str) -> float:
    """Return estimated probability that text is English via langdetect."""
    try:
        for lang in detect_langs(text):
            if lang.lang == "en":
                return float(lang.prob)
        return 0.0
    except LangDetectException:
        return 0.0

# ---------------------------------------------------------------------------
# Filter 2 — Writing quality (heuristic, no ML)
# ---------------------------------------------------------------------------

# Common English function words. Their presence signals grammatical prose;
# their absence signals keyword lists or nav menus.
_FUNCTION_WORDS = frozenset({
    "a", "an", "the", "is", "are", "was", "were", "be", "been", "being",
    "i", "you", "he", "she", "it", "we", "they", "me", "him", "her", "us",
    "in", "on", "at", "to", "of", "and", "or", "but", "if", "not", "so",
    "for", "with", "as", "by", "from", "do", "does", "did", "will",
    "would", "could", "should", "have", "has", "had", "that", "this",
    "than", "then", "when", "what", "which", "who", "how", "there",
})


def _quality_score(text: str) -> float:
    """Heuristic writing-quality score in [0, 1].

    Penalises keyword strings (low function-word ratio), comma-separated
    lists (high comma density), and nav menus (high pipe/slash density).
    Returns 0.0 for texts shorter than 8 tokens.
    """
    tokens = text.split()
    n = len(tokens)
    if n < 8:
        return 0.0

    lower = [t.strip(".,!?;:\"'()[]{}-").lower() for t in tokens]

    func_ratio    = sum(1 for t in lower if t in _FUNCTION_WORDS) / n
    comma_density = text.count(",") / n
    sep_density   = (text.count("|") + text.count(" / ")) / n

    score = 1.0
    if func_ratio < 0.04:
        score -= 0.50
    elif func_ratio < 0.08:
        score -= 0.20

    if comma_density > 0.12:
        score -= 0.40
    elif comma_density > 0.06:
        score -= 0.15

    if sep_density > 0.04:
        score -= 0.40

    return max(0.0, min(1.0, score))

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
# Core scoring
# ---------------------------------------------------------------------------

def score_rows(
    rows: List[Dict[str, str]],
    lang_enabled: bool,
    quality_enabled: bool,
    batch_size: int,
    lang_threshold: float,
    quality_threshold: float,
) -> List[Tuple[Dict[str, str], float, float]]:
    """Run all enabled classifiers. Returns (row, lang, quality) per row."""
    n = len(rows)
    texts = [r["target_context"] for r in rows]
    lang_scores    = [1.0] * n
    quality_scores = [1.0] * n

    # ── Language ──────────────────────────────────────────────────────────────
    if lang_enabled:
        if not HAS_LANGDETECT:
            log.warning("langdetect not installed — skipping language filter. pip install langdetect")
        else:
            it = (
                tqdm(enumerate(texts), desc="Language", total=n, unit="row", leave=False)
                if HAS_TQDM else enumerate(texts)
            )
            for i, text in it:
                lang_scores[i] = _english_score(text)

    # ── Quality heuristic (only on language-passing rows) ─────────────────────
    if quality_enabled:
        for i, text in enumerate(texts):
            if lang_scores[i] >= lang_threshold:
                quality_scores[i] = _quality_score(text)

    return list(zip(rows, lang_scores, quality_scores))

# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Filter word_context.py CSV output using language and quality filters.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Language + quality only (no torch/transformers needed)
  python bert_filter.py contexts/*.csv -o filtered.csv

  # All three filters
  python bert_filter.py contexts/*.csv -o filtered.csv

  # Score everything for threshold calibration
  python bert_filter.py input.csv -o scored.csv --score-all --keep-scores 
      """,
    )

    parser.add_argument("inputs", nargs="+", metavar="CSV",
                        help="Input CSV file(s) from word_context.py. Glob patterns accepted.")
    parser.add_argument("-o", "--output", required=True, metavar="FILE",
                        help="Output CSV path.")

    # Thresholds
    parser.add_argument("--lang-threshold", type=float, default=0.85, metavar="F",
                        dest="lang_threshold",
                        help="Min English confidence to keep a row (default: 0.85).")
    parser.add_argument("--quality-threshold", type=float, default=0.70, metavar="F",
                        dest="quality_threshold",
                        help="Min heuristic quality score to keep a row (default: 0.70). "
                             "Penalises keyword strings, comma lists, and nav menus.")

    # Filter toggles
    parser.add_argument("--no-lang-filter", action="store_true", dest="no_lang",
                        help="Skip language detection.")
    parser.add_argument("--no-quality-filter", action="store_true", dest="no_quality",
                        help="Skip writing-quality scoring.")

    # Output options
    parser.add_argument("--keep-scores", action="store_true", dest="keep_scores",
                        help="Append lang_score, quality_score columns.")
    parser.add_argument("--score-all", action="store_true", dest="score_all",
                        help="Write every row without filtering. Implies --keep-scores.")

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    if args.score_all:
        args.keep_scores = True

    # Expand glob patterns
    input_paths: List[Path] = []
    for pattern in args.inputs:
        matched = sorted(glob.glob(pattern, recursive=True))
        if matched:
            input_paths.extend(Path(p) for p in matched)
        else:
            input_paths.append(Path(pattern))

    missing = [p for p in input_paths if not p.exists()]
    if missing:
        parser.error(f"File(s) not found: {', '.join(str(p) for p in missing)}")

    if args.no_lang and args.no_quality is None:
        parser.error("All filters are disabled — nothing to do.")

    # ── Output setup ──────────────────────────────────────────────────────────
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

            log.info("Processing %s (%d rows) ...", path.name, len(rows))

            scored = score_rows(
                rows=rows,
                lang_enabled=not args.no_lang,
                quality_enabled=not args.no_quality,
                batch_size=args.batch_size,
                lang_threshold=args.lang_threshold,
                quality_threshold=args.quality_threshold,
            )

            kept = 0
            for row, ls, qs in scored:
                if not args.score_all:
                    if ls < args.lang_threshold:
                        continue
                    if qs < args.quality_threshold:
                        continue

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

    log.info("Done. %d/%d rows kept (%.1f%%) -> %s",
             grand_out, grand_in,
             100.0 * grand_out / grand_in if grand_in else 0.0,
             output_path)


if __name__ == "__main__":
    main()
