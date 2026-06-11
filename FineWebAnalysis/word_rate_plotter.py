#!/usr/bin/env python3
"""
word_rate_plotter.py — Plot target-word slang occurrences per crawl dump from
scored context CSVs (script version of DataProcessingTools/word_rate_plotter.ipynb).

Reads the per-crawl CSVs produced by roberta_filter.py --score-all (columns:
target, uri, target_context, roberta_score; filenames carry the crawl id,
e.g. word_context_CC-MAIN-2019-35.csv), keeps rows with roberta_score >= the
threshold, and plots one line per target word with one data point per crawl
dump (the CC-MAIN year-week mapped to a calendar date).

Usage:
    python word_rate_plotter.py --threshold 0.5
    python word_rate_plotter.py --threshold 0.8 --top 20 --log -o rates.png
    python word_rate_plotter.py --threshold 0.5 --words epic fire sus
"""

from __future__ import annotations

import argparse
import csv
import logging
import platform
import re
import sys
from collections import Counter, defaultdict
from datetime import date
from pathlib import Path
from typing import Dict, List, Optional

import matplotlib.dates as mdates
import matplotlib.pyplot as plt

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stderr)],
)
log = logging.getLogger(__name__)

DEFAULT_SCORED_DIR = Path(__file__).resolve().parent / "prompt_scored"
CRAWL_ID_RE = re.compile(r"CC-MAIN-(\d{4})-(\d{2})")


def _use_emoji_font() -> None:
    """Pick a system font with emoji glyphs so emoji targets render."""
    emoji_font = {
        "Darwin": "Apple Color Emoji",
        "Windows": "Segoe UI Emoji",
    }.get(platform.system(), "Noto Color Emoji")
    plt.rcParams["font.family"] = "sans-serif"
    plt.rcParams["font.sans-serif"] = [emoji_font, "DejaVu Sans"]


def load_dump_counts(scored_dir: Path, threshold: float) -> Dict[date, Counter]:
    """Count above-threshold rows per (dump, target) across scored crawl CSVs.

    Returns {dump date: Counter of target occurrences}; the CC-MAIN-<year>-<week>
    crawl id is mapped to the Monday of that ISO week.
    """
    counts: Dict[date, Counter] = defaultdict(Counter)
    for path in sorted(scored_dir.glob("*.csv")):
        m = CRAWL_ID_RE.search(path.stem)
        if not m:
            log.warning("Skipping %s — no CC-MAIN-<year>-<week> in filename.", path.name)
            continue
        dump_date = date.fromisocalendar(int(m.group(1)), int(m.group(2)), 1)
        with path.open(newline="", encoding="utf-8-sig") as fh:
            for row in csv.DictReader(fh):
                if float(row["roberta_score"]) >= threshold:
                    counts[dump_date][row["target"]] += 1

    n_rows = sum(sum(c.values()) for c in counts.values())
    log.info("Loaded %d crawl dump(s); %d rows >= threshold %.2f",
             len(counts), n_rows, threshold)
    return counts


def plot_rates(
    counts: Dict[date, Counter],
    threshold: float,
    words: Optional[List[str]],
    top: Optional[int],
    log_scale: bool,
    output: Optional[Path],
) -> None:
    dumps = sorted(counts)
    totals: Counter = Counter()
    for dump_counts in counts.values():
        totals.update(dump_counts)

    if words:
        missing = [w for w in words if w not in totals]
        if missing:
            log.warning("No above-threshold occurrences for: %s", ", ".join(missing))
        targets = [w for w in words if w in totals]
    else:
        targets = [w for w, _ in totals.most_common(top)]
    if not targets:
        sys.exit("Nothing to plot — no target words above the threshold.")

    fig, ax = plt.subplots(figsize=(12, 6))
    for target in targets:
        ax.plot(dumps, [counts[d][target] for d in dumps],
                marker="o", markersize=3, linewidth=1, label=target)

    ax.set_title(f"Slang usage per crawl dump ({len(targets)} words, "
                 f"roberta_score >= {threshold:g})")
    ax.set_xlabel("Crawl dump date")
    ax.set_ylabel(f"Occurrences in dump{' (log scale)' if log_scale else ''}")
    if log_scale:
        ax.set_yscale("log")
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    fig.autofmt_xdate()
    ax.legend(bbox_to_anchor=(1.01, 1), loc="upper left", fontsize="small",
              ncols=1 + len(targets) // 30)
    ax.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()

    if output:
        plt.savefig(output, dpi=150)
        log.info("Plot saved to %s", output)
    else:
        plt.show()


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Plot target-word slang occurrences per crawl dump from scored CSVs.")
    p.add_argument("--threshold", type=float, default=0.0, metavar="F",
                   help="Min roberta_score for a row to count (default: 0.0).")
    p.add_argument("--scored-dir", type=Path, default=DEFAULT_SCORED_DIR,
                   metavar="DIR", dest="scored_dir",
                   help=f"Directory of scored crawl CSVs (default: {DEFAULT_SCORED_DIR}).")
    p.add_argument("--words", nargs="+", metavar="WORD",
                   help="Only plot these target words (default: all).")
    p.add_argument("--top", type=int, metavar="N",
                   help="Only plot the N most frequent target words.")
    p.add_argument("--log", action="store_true", dest="log_scale",
                   help="Log-scale the y axis.")
    p.add_argument("-o", "--output", type=Path, metavar="FILE",
                   help="Save the plot here instead of displaying it.")
    return p


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    if not args.scored_dir.is_dir():
        parser.error(f"Scored directory not found: {args.scored_dir}")

    counts = load_dump_counts(args.scored_dir, args.threshold)
    if not counts:
        parser.error(f"No crawl CSVs found in {args.scored_dir}.")

    _use_emoji_font()
    plot_rates(counts, args.threshold, args.words, args.top,
               args.log_scale, args.output)


if __name__ == "__main__":
    main()
