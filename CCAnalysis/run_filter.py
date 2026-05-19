#!/usr/bin/env python3
"""
run_filter.py — Re-filter bert_filter.py --score-all output by score thresholds.

Reads CSVs with uri, target_context, lang_score, quality_score, slang_score
columns and writes only rows that meet all specified thresholds. Use this to
experiment with different cutoffs without re-running the BERT models.

The crawl-date range flags (--from-crawl, --to-crawl) select which
input files to process by matching the CC-MAIN-YYYY-WW identifier in each
filename. Files with no recognisable crawl ID in their name are always included.

Rows with an empty slang_score (word not present in that context) are never
filtered by --slang-threshold, matching bert_filter.py's behaviour.

Usage:
    python run_filter.py scored/*.csv -o filtered.csv
    python run_filter.py scored/*.csv -o filtered.csv --lang-threshold 0.9 --quality-threshold 0.8
    python run_filter.py scored/*.csv -o filtered.csv \\
        --from-crawl CC-MAIN-2020-05 --to-crawl CC-MAIN-2022-49
    python run_filter.py scored/ -o filtered.csv --keep-scores
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

def _passes(
    row: Dict[str, str],
    lang_t: float,
    quality_t: float,
    slang_t: float,
) -> bool:
    try:
        if float(row.get("lang_score") or "1") < lang_t:
            return False
    except ValueError:
        pass
    try:
        if float(row.get("quality_score") or "1") < quality_t:
            return False
    except ValueError:
        pass
    slang_raw = (row.get("slang_score") or "").strip()
    if slang_raw:
        try:
            if float(slang_raw) < slang_t:
                return False
        except ValueError:
            pass
    return True

# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Re-filter bert_filter.py --score-all output by score thresholds.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run_filter.py scored/*.csv -o filtered.csv
  python run_filter.py scored/*.csv -o filtered.csv --lang-threshold 0.9
  python run_filter.py scored/*.csv -o filtered.csv \\
      --from-crawl CC-MAIN-2020-05 --to-crawl CC-MAIN-2022-49
        """,
    )

    parser.add_argument("inputs", nargs="+", metavar="CSV",
                        help="Scored CSV file(s) or directory. Glob patterns accepted.")
    parser.add_argument("-o", "--output", required=True, metavar="FILE",
                        help="Output CSV path.")

    # Thresholds — same defaults as bert_filter.py
    parser.add_argument("--lang-threshold", type=float, default=0.85, metavar="F",
                        dest="lang_threshold",
                        help="Min lang_score to keep (default: 0.85).")
    parser.add_argument("--quality-threshold", type=float, default=0.70, metavar="F",
                        dest="quality_threshold",
                        help="Min quality_score to keep (default: 0.70).")
    parser.add_argument("--slang-threshold", type=float, default=0.0, metavar="F",
                        dest="slang_threshold",
                        help="Min slang_score to keep when the score is present (default: 0.0). "
                             "Rows with an empty slang_score are unaffected.")

    # Date range so we can easily reupload dirs back to AWS for more samples
    parser.add_argument("--from-crawl", default=None, metavar="ID", dest="from_crawl",
                        help="Only process files at or after this crawl ID "
                             "(e.g. CC-MAIN-2020-05).")
    parser.add_argument("--to-crawl", default=None, metavar="ID", dest="to_crawl",
                        help="Only process files up to and including this crawl ID.")

    # Output
    parser.add_argument("--keep-scores", action="store_true", dest="keep_scores",
                        help="Include lang_score, quality_score, slang_score columns in output. "
                             "Default is to strip them and write only uri and target_context.")

    return parser


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

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    base_fields = ["uri", "target_context"]
    score_fields = ["lang_score", "quality_score", "slang_score"]
    out_fields = base_fields + (score_fields if args.keep_scores else [])

    grand_in = grand_out = 0

    with out_path.open("w", newline="", encoding="utf-8") as out_fh:
        writer = csv.DictWriter(out_fh, fieldnames=out_fields, extrasaction="ignore")
        writer.writeheader()

        for path, crawl_id in in_range:
            with path.open(newline="", encoding="utf-8") as fh:
                rows = list(csv.DictReader(fh))

            if not rows:
                print(f"Skipping empty file: {path.name}", file=sys.stderr)
                continue

            kept = 0
            for row in rows:
                if _passes(row, args.lang_threshold, args.quality_threshold, args.slang_threshold):
                    writer.writerow(row)
                    kept += 1

            grand_in += len(rows)
            grand_out += kept
            tag = f" [{crawl_id}]" if crawl_id else ""
            pct = 100.0 * kept / len(rows) if rows else 0.0
            print(f"{path.name}{tag}: {kept}/{len(rows)} kept ({pct:.1f}%)", file=sys.stderr)

    pct = 100.0 * grand_out / grand_in if grand_in else 0.0
    print(
        f"\nDone. {grand_out:,}/{grand_in:,} rows kept ({pct:.1f}%) -> {out_path}",
        file=sys.stderr,
    )


if __name__ == "__main__":
    main()
