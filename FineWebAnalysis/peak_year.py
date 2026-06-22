#!/usr/bin/env python3
"""
peak_year.py — Find each target word's peak-popularity year from the scored
FineWeb context data.

Reads the same per-crawl scored CSVs as ``word_rate_plotter.py`` (columns:
target, uri, target_context, roberta_score; filenames carry the crawl id, e.g.
``word_context_CC-MAIN-2019-35.csv``), keeps rows scoring at/above the slang-sense
threshold, then aggregates above-threshold hits and FineWeb sample token totals
by *calendar year*. A word's peak year is the year of its highest token-normalised
rate (occurrences per million sample tokens) --- aggregating by year rather than by
individual dump avoids letting a single small/large dump dominate the estimate.

Token totals come from the ``fineweb_10BT_dump_sizes.json`` cache that
``word_rate_plotter.py`` builds (run that once if the cache is missing, or pass
``--raw-counts`` to rank peaks by raw count instead). This script itself is
stdlib-only.

Usage:
    python peak_year.py
    python peak_year.py --threshold 0.99 --words lol sick swag vibe vibes slay aura
    python peak_year.py --threshold 0.5 -o peak_years.json
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
DEFAULT_SCORED_DIR = HERE / "prompt_scored"
DEFAULT_SIZES_CACHE = HERE / "fineweb_10BT_dump_sizes.json"
CRAWL_ID_RE = re.compile(r"CC-MAIN-(\d{4})-(\d{2})")


def _crawl_year(name: str) -> int | None:
    """Calendar year from a ``CC-MAIN-<year>-<week>`` crawl id, else None."""
    m = CRAWL_ID_RE.search(name)
    return int(m.group(1)) if m else None


def load_year_counts(scored_dir: Path, threshold: float) -> dict[int, dict[str, int]]:
    """{year: {target: above-threshold hits}} pooled over that year's dumps."""
    counts: dict[int, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    n_files = 0
    for path in sorted(scored_dir.glob("*.csv")):
        year = _crawl_year(path.stem)
        if year is None:
            print(f"  skipping {path.name} — no CC-MAIN-<year>-<week> in name", file=sys.stderr)
            continue
        n_files += 1
        with path.open(newline="", encoding="utf-8-sig") as fh:
            for row in csv.DictReader(fh):
                if float(row["roberta_score"]) >= threshold:
                    counts[year][row["target"]] += 1
    print(f"  loaded {n_files} dump CSV(s) across {len(counts)} year(s) "
          f"(threshold {threshold:g})", file=sys.stderr)
    return counts


def load_year_tokens(cache_path: Path) -> dict[int, int]:
    """{year: FineWeb sample token total}, summed from the dump-size cache."""
    if not cache_path.is_file():
        sys.exit(f"Token-size cache not found: {cache_path}\n"
                 f"Run word_rate_plotter.py once to build it, or pass --raw-counts.")
    raw = json.loads(cache_path.read_text(encoding="utf-8"))
    tokens: dict[int, int] = defaultdict(int)
    for dump_id, tok in raw.items():
        year = _crawl_year(dump_id)
        if year is not None:
            tokens[year] += int(tok)
    return tokens


def peak_years(
    year_counts: dict[int, dict[str, int]],
    year_tokens: dict[int, int] | None,
    words: list[str] | None,
    min_count: int,
) -> list[dict]:
    """One record per word: peak year (argmax yearly rate), that rate, and totals."""
    totals: dict[str, int] = defaultdict(int)
    for per_word in year_counts.values():
        for w, c in per_word.items():
            totals[w] += c

    if words:
        missing = [w for w in words if w not in totals]
        if missing:
            print(f"  no above-threshold hits for: {', '.join(missing)}", file=sys.stderr)
        targets = [w for w in words if w in totals]
    else:
        targets = [w for w, c in totals.items() if c >= min_count]
    if not targets:
        sys.exit("No target words to report (raise data, lower --threshold/--min-count).")

    years = sorted(year_counts)
    records: list[dict] = []
    for w in targets:
        # rate per year: normalised to occurrences/M tokens unless --raw-counts.
        rates: dict[int, float] = {}
        for y in years:
            k = year_counts[y].get(w, 0)
            if year_tokens is not None:
                tok = year_tokens.get(y, 0)
                rates[y] = (k / tok * 1_000_000) if tok else 0.0
            else:
                rates[y] = float(k)
        # Peak = highest-rate year; ties broken toward the earlier year.
        peak = max(years, key=lambda y: (rates[y], -y))
        records.append({
            "word": w,
            "peak_year": peak,
            "peak_rate_per_million": round(rates[peak], 3),
            "total_hits": totals[w],
            "rate_by_year": {str(y): round(rates[y], 3) for y in years},
        })
    records.sort(key=lambda r: (r["peak_year"], r["word"]))
    return records


def main() -> None:
    p = argparse.ArgumentParser(
        description="Find each target word's peak-popularity year from scored FineWeb data.")
    p.add_argument("--scored-dir", type=Path, default=DEFAULT_SCORED_DIR, metavar="DIR",
                   dest="scored_dir",
                   help=f"Directory of scored crawl CSVs (default: {DEFAULT_SCORED_DIR.name}/).")
    p.add_argument("--threshold", type=float, default=0.5, metavar="F",
                   help="Min roberta_score for a row to count as the slang sense "
                        "(default: 0.5; the paper highlight figures use 0.99).")
    p.add_argument("--words", nargs="+", metavar="WORD",
                   help="Only report these target words (default: all with >= --min-count hits).")
    p.add_argument("--min-count", type=int, default=30, metavar="N", dest="min_count",
                   help="When --words is omitted, skip words with fewer than N total "
                        "above-threshold hits as noise (default: 30).")
    p.add_argument("--raw-counts", action="store_true", dest="raw_counts",
                   help="Rank peaks by raw yearly count instead of occurrences per "
                        "million sample tokens (skips the token-size cache).")
    p.add_argument("--sizes-cache", type=Path, default=DEFAULT_SIZES_CACHE, metavar="FILE",
                   dest="sizes_cache",
                   help=f"JSON cache of per-dump sample token totals "
                        f"(default: {DEFAULT_SIZES_CACHE.name}).")
    p.add_argument("-o", "--output", type=Path, metavar="FILE",
                   help="Write the full results (incl. per-year rates) to this JSON file.")
    args = p.parse_args()

    if not args.scored_dir.is_dir():
        p.error(f"Scored directory not found: {args.scored_dir}")

    year_counts = load_year_counts(args.scored_dir, args.threshold)
    if not year_counts:
        p.error(f"No crawl CSVs found in {args.scored_dir}.")
    year_tokens = None if args.raw_counts else load_year_tokens(args.sizes_cache)

    records = peak_years(year_counts, year_tokens, args.words, args.min_count)

    unit = "raw count" if args.raw_counts else "per-M-token rate"
    width = max(len(r["word"]) for r in records)
    print(f"\n{'word':<{width}}  peak  {unit:>14}  total_hits")
    print(f"{'-' * width}  ----  {'-' * 14}  ----------")
    for r in records:
        print(f"{r['word']:<{width}}  {r['peak_year']}  "
              f"{r['peak_rate_per_million']:>14.3f}  {r['total_hits']:>10}")

    if args.output:
        args.output.write_text(json.dumps(records, indent=2), encoding="utf-8")
        print(f"\nWrote {len(records)} records to {args.output}", file=sys.stderr)


if __name__ == "__main__":
    main()
