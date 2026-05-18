#!/usr/bin/env python3
"""
word_variation.py — Find and plot words with the largest frequency variation
across Common Crawl crawls, using word-sample CSVs from word_sample.py.

Each CSV has a single "word" column.  The script:
  1. Counts each word's frequency per crawl and normalises to a sampling rate.
  2. Scores each word by log2(max_rate / min_rate) across crawls where it appears.
  3. Plots the top-N most variable words as a time series, plus a variation
     bar chart sorted by score.

Usage:
    python word_variation.py /path/to/Run0/
    python word_variation.py /path/to/Run0/ --top 25 --min-crawls 10 -o variation.png
    python word_variation.py /path/to/Run0/ --min-avg-rate 5e-5 --top 30
"""

from __future__ import annotations

import argparse
import csv
import re
import sys
from collections import Counter
from datetime import date
from pathlib import Path

import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np


# ---------------------------------------------------------------------------
# Date parsing
# ---------------------------------------------------------------------------

_CRAWL_RE = re.compile(r"CC-MAIN-(\d{4})-(\d{2,4})")


def _crawl_date(crawl_id: str) -> date:
    """Convert CC-MAIN-YYYY-WW to an approximate calendar date (Monday of that ISO week)."""
    m = _CRAWL_RE.search(crawl_id)
    if not m:
        raise ValueError(f"Cannot parse date from: {crawl_id!r}")
    year = int(m.group(1))
    week = max(1, min(int(m.group(2)[:2]), 53))
    return date.fromisocalendar(year, week, 1)


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_directory(
    directory: Path,
) -> tuple[list[date], dict[str, list[int]], list[int]]:
    """
    Load all word-sample CSVs from directory.

    Returns:
        dates   — sorted list of crawl dates (one per CSV)
        counts  — {word: [raw_count_crawl_0, raw_count_crawl_1, ...]} aligned to dates
        totals  — total words sampled per crawl (for computing rates)
    """
    crawl_entries: list[tuple[date, Counter]] = []

    for csv_path in sorted(directory.glob("word_sample_CC-MAIN-*.csv")):
        m = _CRAWL_RE.search(csv_path.name)
        if not m:
            continue
        try:
            d = _crawl_date(m.group(0))
        except ValueError:
            continue

        counter: Counter = Counter()
        with csv_path.open(encoding="utf-8") as fh:
            reader = csv.reader(fh)
            next(reader, None)  # skip header
            for row in reader:
                if row and row[0].strip():
                    counter[row[0].strip().lower()] += 1
        crawl_entries.append((d, counter))

    if not crawl_entries:
        raise SystemExit(f"No word_sample_CC-MAIN-*.csv files found in {directory}")

    crawl_entries.sort(key=lambda x: x[0])
    dates = [d for d, _ in crawl_entries]
    counters = [c for _, c in crawl_entries]
    totals = [max(sum(c.values()), 1) for c in counters]

    all_words: set[str] = set()
    for c in counters:
        all_words.update(c.keys())

    counts: dict[str, list[int]] = {w: [c[w] for c in counters] for w in all_words}

    return dates, counts, totals


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------

def score_words(
    rates: dict[str, list[float]],
    min_crawls: int,
    min_avg_rate: float,
) -> dict[str, float]:
    """
    Return {word: log2(max_rate / min_nonzero_rate)} for words that pass filters.

    Filters:
      - Word must appear (rate > 0) in at least min_crawls crawls.
      - Word's average rate across all crawls must be >= min_avg_rate.
    """
    scores: dict[str, float] = {}
    for word, rs in rates.items():
        nonzero = [r for r in rs if r > 0]
        if len(nonzero) < min_crawls:
            continue
        avg = sum(rs) / len(rs)
        if avg < min_avg_rate:
            continue
        lo, hi = min(nonzero), max(nonzero)
        if lo <= 0:
            continue
        scores[word] = np.log2(hi / lo)
    return scores


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def _bucket_by_year(
    dates: list[date], values: dict[str, list[float]]
) -> tuple[list[date], dict[str, list[float | None]]]:
    """Return (year_midpoints, {word: [mean_value_or_None per year]}).

    Works for both rates and raw counts; zero entries are treated as absent.
    """
    years = sorted({d.year for d in dates})
    year_dates = [date(y, 7, 1) for y in years]
    bucketed: dict[str, list[float | None]] = {}
    for word, vs in values.items():
        by_year: dict[int, list[float]] = {y: [] for y in years}
        for d, v in zip(dates, vs):
            if v:
                by_year[d.year].append(v)
        bucketed[word] = [
            sum(ys) / len(ys) if ys else None
            for ys in (by_year[y] for y in years)
        ]
    return year_dates, bucketed


def plot(
    dates: list[date],
    counts: dict[str, list[int]],
    totals: list[int],
    scores: dict[str, float],
    top: int,
    output: Path | None,
    bucket_by_year: bool = False,
    log_scale: bool = False,
) -> None:
    top_words = sorted(scores, key=lambda w: -scores[w])[:top]
    n = len(top_words)
    colors = cm.tab20(np.linspace(0, 1, n)) if n <= 20 else cm.turbo(np.linspace(0.05, 0.95, n))

    fig, (ax_ts, ax_bar) = plt.subplots(
        1, 2,
        figsize=(max(14, n * 0.4 + 8), 7),
        gridspec_kw={"width_ratios": [2.5, 1]},
    )

    # ── Choose values: rates (log) or raw counts (linear) ────────────────────
    if log_scale:
        values: dict[str, list[float]] = {
            w: [c / t for c, t in zip(cs, totals)]
            for w, cs in counts.items()
        }
        ylabel = "Sampling rate (log scale)"
    else:
        values = {w: [float(c) for c in cs] for w, cs in counts.items()}
        ylabel = "Count per sample"

    # ── Bucket by year or keep per-crawl ─────────────────────────────────────
    if bucket_by_year:
        plot_dates, plot_values = _bucket_by_year(dates, values)
        xlabel = "Year"
        title_suffix = "yearly avg"
        tick_labels: list[str] | None = [str(d.year) for d in plot_dates]
    else:
        plot_dates = dates
        plot_values = {w: [v if v else None for v in vs] for w, vs in values.items()}
        xlabel = "Crawl date"
        title_suffix = "per crawl"
        tick_labels = None

    # ── Time series ───────────────────────────────────────────────────────────
    for word, color in zip(top_words, colors):
        vs = plot_values[word]
        xs = [d for d, v in zip(plot_dates, vs) if v is not None]
        ys = [v for v in vs if v is not None]
        if len(xs) < 2:
            continue
        ax_ts.plot(
            xs, ys,
            label=word,
            color=color,
            linewidth=1.4,
            marker="o",
            markersize=4 if bucket_by_year else 2.5,
            alpha=0.85,
        )

    if log_scale:
        ax_ts.set_yscale("log")
    ax_ts.set_xlabel(xlabel, fontsize=10)
    ax_ts.set_ylabel(ylabel, fontsize=10)
    ax_ts.set_title(f"Top {n} most variable words over time ({title_suffix})", fontsize=11)
    if tick_labels is not None:
        ax_ts.set_xticks(plot_dates)
        ax_ts.set_xticklabels(tick_labels, fontsize=8)
    ax_ts.legend(fontsize=7.5, ncol=max(1, n // 15), loc="upper left", framealpha=0.6)
    ax_ts.spines[["top", "right"]].set_visible(False)
    ax_ts.tick_params(axis="x", rotation=30)

    # ── Total count bar chart ─────────────────────────────────────────────────
    bar_words = top_words[::-1]  # highest variation at top
    bar_totals = [sum(counts[w]) for w in bar_words]
    ax_bar.barh(
        range(n), bar_totals,
        color=colors[::-1], edgecolor="white", linewidth=0.4,
    )
    ax_bar.set_yticks(range(n))
    ax_bar.set_yticklabels(bar_words, fontsize=8.5)
    ax_bar.set_xlabel("Total count across all crawls", fontsize=9)
    ax_bar.set_title("Total occurrences", fontsize=11)
    ax_bar.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{int(x):,}"))
    ax_bar.spines[["top", "right"]].set_visible(False)

    fig.suptitle(
        "Word frequency variation across Common Crawl snapshots\n"
        f"({len(dates)} crawls  ·  {len(scores):,} words)",
        fontsize=12, y=1.01,
    )
    fig.tight_layout()

    if output:
        output.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output, dpi=160, bbox_inches="tight")
        print(f"Saved: {output}")
    else:
        plt.show()
    plt.close(fig)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Plot words with the largest frequency variation across CC crawls.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python word_variation.py Run0/
  python word_variation.py Run0/ --top 30 --min-crawls 10 -o variation.png
  python word_variation.py Run0/ --min-avg-rate 1e-4 --top 40
        """,
    )
    parser.add_argument(
        "directory",
        help="Directory containing word_sample_CC-MAIN-*.csv files.",
    )
    parser.add_argument(
        "--top", type=int, default=20, metavar="N",
        help="Number of top-variation words to plot (default: 20).",
    )
    parser.add_argument(
        "--min-crawls", type=int, default=5, metavar="K", dest="min_crawls",
        help="Minimum crawls a word must appear in to be scored (default: 5).",
    )
    parser.add_argument(
        "--min-avg-rate", type=float, default=2e-5, metavar="F", dest="min_avg_rate",
        help="Minimum average sampling rate across all crawls (default: 2e-5). "
             "Filters out words too rare to distinguish from noise.",
    )
    parser.add_argument(
        "-o", "--output", default=None,
        help="Output PNG path (displays interactively if omitted).",
    )
    parser.add_argument(
        "--words", metavar="FILE", default=None,
        help="Plain-text file of words to plot (one per line, # for comments). "
             "When provided, only these words are shown and --top / --min-* filters "
             "are ignored for selection (variation scores are still computed).",
    )
    parser.add_argument(
        "--bucket-by-year", action="store_true", dest="bucket_by_year",
        help="Average values across all crawls within each year before plotting. "
             "Default is one point per crawl sample file.",
    )
    parser.add_argument(
        "--log", action="store_true",
        help="Show log-scale sampling rate on the y-axis. "
             "Default is linear scale showing raw word counts.",
    )
    parser.add_argument(
        "--print-top", type=int, default=50, metavar="N", dest="print_top",
        help="Number of top words to print to stdout (default: 50). "
             "Ignored when --words is used.",
    )
    return parser


def _load_word_list(path: Path) -> list[str]:
    """Load words from a plain-text file, one per line. Case-normalised to lowercase."""
    words = []
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            word = line.strip().lower()
            if word and not word.startswith("#"):
                words.append(word)
    return words


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    directory = Path(args.directory)
    if not directory.is_dir():
        print(f"Not a directory: {directory}", file=sys.stderr)
        sys.exit(1)

    print(f"Loading CSVs from {directory} …")
    dates, counts, totals = load_directory(directory)
    print(f"  {len(dates)} crawls  ·  {len(counts):,} unique words")

    rates = {w: [c / t for c, t in zip(cs, totals)] for w, cs in counts.items()}

    # ── Word list mode ────────────────────────────────────────────────────────
    if args.words:
        # _load_word_list already lowercases; counts keys are also lowercase from
        # load_directory, so lookup is case-insensitive end-to-end.
        word_list = _load_word_list(Path(args.words))
        missing = [w for w in word_list if w.lower() not in counts]
        if missing:
            print(f"  Warning: {len(missing)} word(s) not found in any crawl: {missing}")
        present = [w.lower() for w in word_list if w.lower() in counts]
        if not present:
            print("None of the specified words appear in the data.", file=sys.stderr)
            sys.exit(1)
        scores = {w: score_words({w: rates[w]}, 1, 0.0).get(w, 0.0) for w in present}
        print(f"  Plotting {len(present)} word(s) from {args.words}\n")
        print(f"  {'word':<25}  {'total count':>11}  {'crawls':>6}")
        print(f"  {'-'*25}  {'-'*11}  {'-'*6}")
        for w in present:
            total = sum(counts[w])
            n_appear = sum(1 for c in counts[w] if c > 0)
            print(f"  {w:<25}  {total:>11,}  {n_appear:>6}")
        plot(dates, counts, totals, scores, len(present),
             Path(args.output) if args.output else None,
             bucket_by_year=args.bucket_by_year, log_scale=args.log)
        return

    # ── Auto mode: rank by variation score ───────────────────────────────────
    scores = score_words(rates, args.min_crawls, args.min_avg_rate)
    print(f"  {len(scores):,} words passed filters "
          f"(min_crawls={args.min_crawls}, min_avg_rate={args.min_avg_rate:.1e})\n")

    if not scores:
        print("No words passed the filters. Try lowering --min-crawls or --min-avg-rate.")
        sys.exit(1)

    top_all = sorted(scores, key=lambda w: -scores[w])
    print(f"Top {min(args.print_top, len(top_all))} words by variation (most changed first):")
    print(f"  {'word':<25}  {'total count':>11}  {'crawls':>6}")
    print(f"  {'-'*25}  {'-'*11}  {'-'*6}")
    for w in top_all[:args.print_top]:
        total = sum(counts[w])
        n_appear = sum(1 for c in counts[w] if c > 0)
        print(f"  {w:<25}  {total:>11,}  {n_appear:>6}")

    plot(
        dates, counts, totals, scores, args.top,
        Path(args.output) if args.output else None,
        bucket_by_year=args.bucket_by_year, log_scale=args.log,
    )


if __name__ == "__main__":
    main()
