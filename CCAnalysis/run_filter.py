#!/usr/bin/env python3
"""
run_filter.py — Re-filter scored CSV output by score thresholds.

Reads CSVs produced by bert_filter.py (lang_score, quality_score) and/or
bert_slang_filter.py (sbert_score, nli_score) and writes only rows that meet
all specified thresholds. Score columns absent from a file are ignored.

The crawl-date range flags (--from-crawl, --to-crawl) select which
input files to process by matching the CC-MAIN-YYYY-WW identifier in each
filename. Files with no recognisable crawl ID in their name are always included.

Rows with an empty score (word not present in that context) are never filtered
by the corresponding threshold.

Usage:
    python run_filter.py scored/*.csv -o filtered.csv
    python run_filter.py scored/*.csv -o filtered.csv --lang-threshold 0.9 --quality-threshold 0.8
    python run_filter.py slang_scored/*.csv -o filtered.csv --sbert-threshold 0.1 --nli-threshold 0.05
    python run_filter.py scored/ -o filtered.csv --keep-scores \\
        --from-crawl CC-MAIN-2020-05 --to-crawl CC-MAIN-2022-49
"""

from __future__ import annotations

import argparse
import csv
import glob
import re
import sys
from datetime import date
from pathlib import Path
from typing import Dict, List, Optional, Tuple

_CRAWL_RE = re.compile(r"CC-MAIN-(\d{4})-(\d{2,4})")

# ---------------------------------------------------------------------------
# Date helpers (shared pattern with word_context.py / word_variation.py)
# ---------------------------------------------------------------------------

def _crawl_date(crawl_id: str) -> date:
    m = _CRAWL_RE.search(crawl_id)
    if not m:
        raise ValueError(f"Cannot parse date from: {crawl_id!r}")
    year = int(m.group(1))
    week = max(1, min(int(m.group(2)[:2]), 53))
    return date.fromisocalendar(year, week, 1)


def _crawl_id_from_path(path: Path) -> Optional[str]:
    m = _CRAWL_RE.search(path.name)
    return m.group(0) if m else None


def _in_range(
    crawl_id: str,
    from_crawl: Optional[str],
    to_crawl: Optional[str],
) -> bool:
    try:
        d = _crawl_date(crawl_id)
    except ValueError:
        return True
    if from_crawl is not None:
        try:
            if d < _crawl_date(from_crawl):
                return False
        except ValueError:
            pass
    if to_crawl is not None:
        try:
            if d > _crawl_date(to_crawl):
                return False
        except ValueError:
            pass
    return True

# ---------------------------------------------------------------------------
# Row filtering
# ---------------------------------------------------------------------------

def _threshold_check(row: Dict[str, str], col: str, threshold: float, default: float = 1.0) -> bool:
    """Return False if the column is present, non-empty, and below threshold."""
    raw = (row.get(col) or "").strip()
    if not raw:
        return True  # absent or empty — don't filter
    try:
        return float(raw) >= threshold
    except ValueError:
        return True


def _passes(
    row: Dict[str, str],
    lang_t: float,
    quality_t: float,
    sbert_t: float,
    nli_t: float,
) -> bool:
    return (
        _threshold_check(row, "lang_score",    lang_t)
        and _threshold_check(row, "quality_score", quality_t)
        and _threshold_check(row, "sbert_score",   sbert_t)
        and _threshold_check(row, "nli_score",     nli_t)
    )

# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Re-filter scored CSV output by score thresholds.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run_filter.py scored/*.csv -o filtered.csv
  python run_filter.py scored/*.csv -o filtered.csv --lang-threshold 0.9
  python run_filter.py slang_scored/*.csv --output-dir filtered/ --sbert-threshold 0.1 --nli-threshold 0.05
  python run_filter.py scored/ --output-dir filtered/ --keep-scores \\
      --from-crawl CC-MAIN-2020-05 --to-crawl CC-MAIN-2022-49
        """,
    )

    parser.add_argument("inputs", nargs="+", metavar="CSV",
                        help="Scored CSV file(s) or directory. Glob patterns accepted.")

    out_group = parser.add_mutually_exclusive_group(required=True)
    out_group.add_argument("-o", "--output", metavar="FILE",
                           help="Output CSV path (single-file mode — all inputs merged).")
    out_group.add_argument("--output-dir", metavar="DIR", dest="output_dir",
                           help="Output directory (per-file mode — one output file per input, "
                                "same filename, preserving crawl date in filename). "
                                "Created if it does not exist.")

    # Thresholds
    parser.add_argument("--lang-threshold", type=float, default=0.85, metavar="F",
                        dest="lang_threshold",
                        help="Min lang_score to keep (default: 0.85).")
    parser.add_argument("--quality-threshold", type=float, default=0.70, metavar="F",
                        dest="quality_threshold",
                        help="Min quality_score to keep (default: 0.70).")
    parser.add_argument("--sbert-threshold", type=float, default=0.0, metavar="F",
                        dest="sbert_threshold",
                        help="Min sbert_score to keep when the column is present (default: 0.0). "
                             "Rows with an empty sbert_score are unaffected.")
    parser.add_argument("--nli-threshold", type=float, default=0.0, metavar="F",
                        dest="nli_threshold",
                        help="Min nli_score to keep when the column is present (default: 0.0). "
                             "Rows with an empty nli_score are unaffected.")

    # Date range so we can easily reupload dirs back to AWS for more samples
    parser.add_argument("--from-crawl", default=None, metavar="ID", dest="from_crawl",
                        help="Only process files at or after this crawl ID "
                             "(e.g. CC-MAIN-2020-05).")
    parser.add_argument("--to-crawl", default=None, metavar="ID", dest="to_crawl",
                        help="Only process files up to and including this crawl ID.")

    # Output
    parser.add_argument("--keep-scores", action="store_true", dest="keep_scores",
                        help="Include score columns in output (lang_score, quality_score, "
                             "sbert_score, nli_score — whichever are present in the input). "
                             "Default is to strip them and write only uri and target_context.")

    return parser


def _filter_file(
    path: Path,
    crawl_id: Optional[str],
    out_fh,
    out_fields: List[str],
    args: argparse.Namespace,
) -> Tuple[int, int]:
    """Filter one file, writing kept rows to out_fh. Returns (n_in, n_kept)."""
    with path.open(newline="", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))

    if not rows:
        print(f"Skipping empty file: {path.name}", file=sys.stderr)
        return 0, 0

    writer = csv.DictWriter(out_fh, fieldnames=out_fields, extrasaction="ignore")
    kept = 0
    for row in rows:
        if _passes(row, args.lang_threshold, args.quality_threshold,
                   args.sbert_threshold, args.nli_threshold):
            writer.writerow(row)
            kept += 1

    tag = f" [{crawl_id}]" if crawl_id else ""
    pct = 100.0 * kept / len(rows) if rows else 0.0
    print(f"{path.name}{tag}: {kept}/{len(rows)} kept ({pct:.1f}%)", file=sys.stderr)
    return len(rows), kept


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    # Expand globs and directory inputs
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

    if len(input_paths) > 1 and args.output_dir is None:
        print(
            f"Warning: {len(input_paths)} input files will be merged into one output file. "
            "Use --output-dir to write a separate file per input.",
            file=sys.stderr,
        )

    # Apply date range filter to file list
    in_range: List[Tuple[Path, Optional[str]]] = []
    for path in input_paths:
        crawl_id = _crawl_id_from_path(path)
        if crawl_id and not _in_range(crawl_id, args.from_crawl, args.to_crawl):
            print(f"Skipping {path.name} (outside date range)", file=sys.stderr)
            continue
        in_range.append((path, crawl_id))

    if not in_range:
        print("No input files matched the date range.", file=sys.stderr)
        sys.exit(1)

    base_fields = ["uri", "target_context"]
    score_fields = ["lang_score", "quality_score", "sbert_score", "nli_score"]
    out_fields = base_fields + (score_fields if args.keep_scores else [])

    grand_in = grand_out = 0

    # ── Per-file mode (--output-dir) ─────────────────────────────────────────
    if args.output_dir is not None:
        out_dir = Path(args.output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        for path, crawl_id in in_range:
            out_path = out_dir / path.name
            with out_path.open("w", newline="", encoding="utf-8") as out_fh:
                writer = csv.DictWriter(out_fh, fieldnames=out_fields, extrasaction="ignore")
                writer.writeheader()
                n_in, n_kept = _filter_file(path, crawl_id, out_fh, out_fields, args)
            grand_in += n_in
            grand_out += n_kept
        pct = 100.0 * grand_out / grand_in if grand_in else 0.0
        print(f"\nDone. {grand_out:,}/{grand_in:,} rows kept ({pct:.1f}%) -> {out_dir}",
              file=sys.stderr)
        return

    # ── Single-file mode (-o) ─────────────────────────────────────────────────
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="", encoding="utf-8") as out_fh:
        writer = csv.DictWriter(out_fh, fieldnames=out_fields, extrasaction="ignore")
        writer.writeheader()
        for path, crawl_id in in_range:
            n_in, n_kept = _filter_file(path, crawl_id, out_fh, out_fields, args)
            grand_in += n_in
            grand_out += n_kept

    pct = 100.0 * grand_out / grand_in if grand_in else 0.0
    print(f"\nDone. {grand_out:,}/{grand_in:,} rows kept ({pct:.1f}%) -> {out_path}",
          file=sys.stderr)


if __name__ == "__main__":
    main()
