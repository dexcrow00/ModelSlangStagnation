#!/usr/bin/env python3
"""
sense_filter_rates.py — What fraction of each target word's extracted occurrences
the RoBERTa sense classifier discards at a given threshold.

``peak_year.py`` and the paper's figures keep only rows scoring
``roberta_score >= threshold``; everything below is dropped as a non-slang use of
the word. This reports, per word, how much of the raw extraction that removes --
a direct read on polysemy, since a word with a common standard-English sense
(\\textit{sick}, \\textit{troll}) loses far more of its occurrences than a
slang-only one.

Scanning the scored CSVs costs a couple of minutes, so the first run caches a
per-word histogram of scores (``score_hist.json``); later runs answer any
threshold from the cache instantly. Pass ``--rebuild`` after rescoring.

Usage (run from FineWebAnalysis/):
    python sense_filter_rates.py                      # threshold 0.99
    python sense_filter_rates.py -t 0.5
    python sense_filter_rates.py -t 0 0.5 0.9 0.99    # one column per threshold
    python sense_filter_rates.py -t 0.99 --csv rates.csv
    python sense_filter_rates.py --rebuild
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
DEFAULT_DIRS = [HERE / "prompt_scored", HERE / "scenario_prompt_scored"]
DEFAULT_CACHE = HERE / "score_hist.json"
SCALE = 10_000          # scores are stored to 4 decimal places; bin exactly


def scan(dirs: list[Path]) -> dict[str, dict[str, int]]:
    """{word: {scaled score: count}} over every scored row in every CSV."""
    csv.field_size_limit(10_000_000)
    hist: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    files = sorted(p for d in dirs for p in d.glob("*.csv"))
    if not files:
        sys.exit(f"No CSVs found in {', '.join(str(d) for d in dirs)}.")
    for n, path in enumerate(files, 1):
        print(f"  [{n}/{len(files)}] {path.parent.name}/{path.name}", end="\r",
              file=sys.stderr, flush=True)
        with path.open(newline="", encoding="utf-8-sig") as fh:
            reader = csv.reader(fh)
            header = next(reader, None)
            if not header:
                continue
            ti, si = header.index("target"), header.index("roberta_score")
            for row in reader:
                try:
                    key = str(int(round(float(row[si]) * SCALE)))
                except (ValueError, IndexError):
                    continue
                hist[row[ti]][key] += 1
    print(f"  scanned {len(files)} file(s), {len(hist)} target word(s)" + " " * 30,
          file=sys.stderr)
    return {w: dict(c) for w, c in hist.items()}


def load_hist(cache: Path, dirs: list[Path], rebuild: bool) -> dict[str, dict[str, int]]:
    if cache.is_file() and not rebuild:
        return json.loads(cache.read_text(encoding="utf-8"))
    hist = scan(dirs)
    cache.write_text(json.dumps(hist), encoding="utf-8")
    print(f"  cached to {cache.name}", file=sys.stderr)
    return hist


def rates(hist: dict[str, dict[str, int]], thresholds: list[float]) -> list[dict]:
    """Per word: total occurrences and the share dropped at each threshold."""
    out = []
    for word, counts in hist.items():
        items = [(int(k), v) for k, v in counts.items()]
        total = sum(v for _, v in items)
        row = {"word": word, "total": total, "pct": [], "kept": []}
        for t in thresholds:
            cut = int(round(t * SCALE))
            # peak_year.py keeps score >= threshold, so "filtered" is score < threshold.
            kept = sum(v for s, v in items if s >= cut)
            row["kept"].append(kept)
            row["pct"].append(100.0 * (total - kept) / total if total else 0.0)
        out.append(row)
    out.sort(key=lambda r: (-r["pct"][0], -r["total"]))
    return out


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[1],
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("-t", "--threshold", type=float, nargs="+", default=[0.99],
                   metavar="F", help="Sense-score threshold(s); rows below are "
                                     "filtered out (default: 0.99).")
    p.add_argument("--scored-dir", type=Path, nargs="+", default=DEFAULT_DIRS,
                   metavar="DIR", dest="scored_dir",
                   help="Directories of scored crawl CSVs (default: prompt_scored/ "
                        "and scenario_prompt_scored/).")
    p.add_argument("--cache", type=Path, default=DEFAULT_CACHE, metavar="FILE",
                   help=f"Score-histogram cache (default: {DEFAULT_CACHE.name}).")
    p.add_argument("--rebuild", action="store_true",
                   help="Rescan the CSVs even if the cache exists.")
    p.add_argument("--min-total", type=int, default=0, metavar="N", dest="min_total",
                   help="Skip words with fewer than N extracted occurrences.")
    p.add_argument("--words", nargs="+", metavar="WORD", help="Only these target words.")
    p.add_argument("--csv", type=Path, metavar="FILE", help="Also write the table as CSV.")
    args = p.parse_args()

    hist = load_hist(args.cache, args.scored_dir, args.rebuild)
    if args.words:
        missing = [w for w in args.words if w not in hist]
        if missing:
            print(f"  no scored rows for: {', '.join(missing)}", file=sys.stderr)
        hist = {w: hist[w] for w in args.words if w in hist}
    rows = [r for r in rates(hist, args.threshold) if r["total"] >= args.min_total]
    if not rows:
        sys.exit("No words to report.")

    width = max(len(r["word"]) for r in rows)
    head = f"{'word':<{width}}  {'occurrences':>11}"
    for t in args.threshold:
        head += f"  {'kept@' + format(t, 'g'):>10}  {'%filtered':>9}"
    print(head)
    print("-" * len(head))
    for r in rows:
        line = f"{r['word']:<{width}}  {r['total']:>11,}"
        for kept, pct in zip(r["kept"], r["pct"]):
            line += f"  {kept:>10,}  {pct:>8.1f}%"
        print(line)

    grand = sum(r["total"] for r in rows)
    line = f"{'ALL':<{width}}  {grand:>11,}"
    for i in range(len(args.threshold)):
        kept = sum(r["kept"][i] for r in rows)
        line += f"  {kept:>10,}  {100.0 * (grand - kept) / grand:>8.1f}%"
    print("-" * len(head))
    print(line)

    if args.csv:
        with args.csv.open("w", newline="", encoding="utf-8") as fh:
            w = csv.writer(fh)
            cols = ["word", "occurrences"]
            for t in args.threshold:
                cols += [f"kept_at_{t:g}", f"pct_filtered_at_{t:g}"]
            w.writerow(cols)
            for r in rows:
                out = [r["word"], r["total"]]
                for kept, pct in zip(r["kept"], r["pct"]):
                    out += [kept, round(pct, 2)]
                w.writerow(out)
        print(f"\nWrote {args.csv}", file=sys.stderr)


if __name__ == "__main__":
    main()
