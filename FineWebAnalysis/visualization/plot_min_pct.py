#!/usr/bin/env python3
"""
plot_min_pct.py — Plot each word's *minimum* year as a percentage of its own peak
year, from the ``pct_of_peak_by_year`` map written by peak_year.py.

A word whose minimum sits near 100% held a flat rate across 2013--2024; one whose
minimum approaches 0% all but vanished in at least one year, i.e. it has a real
lifecycle rather than a steady presence. Bars are sorted ascending, so the most
dramatic risers/decliners are at the top.

Percent-of-peak amplifies sampling noise for rare words (a word with nine total
hits can read 0% in several years), so ``MIN_HITS`` screens those out.

The word filters (MIN_HITS, MAX_MIN_PCT, MAX_FILTERED_PCT) are constants below
rather than flags, so any change to them is tracked in git.

Usage (run from FineWebAnalysis/visualization/):
    python plot_min_pct.py
    python plot_min_pct.py --top 40 -o figures/min_pct.png
    python plot_min_pct.py peak_years_pct.json -            # display interactively
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]  # FineWebAnalysis/
DEFAULT_INPUT = ROOT / "peak_years_pct.json"
DEFAULT_OUTPUT = ROOT / "figures" / "min_pct_of_peak.png"
DEFAULT_FILTER_RATES = ROOT / "word_filter_rates_99.csv"
BAR, INK, GRID = "#2a78d6", "#52514e", "#e1e0d9"

# Skip words with fewer than this many total slang-sense hits, whose percentages
# are mostly sampling noise (0 keeps all).
MIN_HITS = 80
# Skip words whose lowest year still sits above this % of their own peak, i.e.
# keep only words that fall by at least (100 - MAX_MIN_PCT)% at some point
# (None keeps all).
MAX_MIN_PCT: float | None = 50
# Skip words whose share of hits rejected by the sense filter exceeds this %,
# read from --filter-rates (None keeps all).
MAX_FILTERED_PCT: float | None = 80


def load_filter_rates(path: Path) -> dict[str, float]:
    """``{word: pct filtered}`` from word_filter_rates_*.csv."""
    with path.open(encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        col = next((c for c in reader.fieldnames or [] if c.startswith("pct_filtered")), None)
        if col is None:
            sys.exit(f"{path} has no 'pct_filtered_*' column.")
        return {r["word"]: float(r[col]) for r in reader}


def load(path: Path, min_hits: int, max_min_pct: float | None,
         filter_rates: dict[str, float] | None = None,
         max_filtered_pct: float | None = None) -> list[tuple[str, float, int, str]]:
    """``[(word, min pct, hits, year of that minimum)]``, ascending by percentage."""
    records = json.loads(path.read_text(encoding="utf-8"))
    rows = []
    for r in records:
        pct = r.get("pct_of_peak_by_year")
        if not pct:
            sys.exit(f"{path} has no 'pct_of_peak_by_year' — regenerate it with peak_year.py.")
        if r["total_hits"] < min_hits:
            continue
        if max_filtered_pct is not None:
            rate = filter_rates.get(r["word"])
            if rate is None:
                print(f"warning: {r['word']!r} not in filter-rates CSV; keeping it.",
                      file=sys.stderr)
            elif rate > max_filtered_pct:
                continue
        year, value = min(pct.items(), key=lambda kv: kv[1])
        if max_min_pct is not None and value > max_min_pct:
            continue
        rows.append((r["word"], value, r["total_hits"], year))
    if not rows:
        sys.exit(f"No words with >= {min_hits} hits"
                 f"{f' whose minimum is <= {max_min_pct:g}% of peak' if max_min_pct is not None else ''}"
                 f"{f' and <= {max_filtered_pct:g}% filtered' if max_filtered_pct is not None else ''}.")
    return sorted(rows, key=lambda t: t[1])


def render(rows, min_hits: int, max_min_pct: float | None, output: Path | None,
           max_filtered_pct: float | None = None) -> None:
    words = [w for w, _, _, _ in rows]
    values = [v for _, v, _, _ in rows]

    fig, ax = plt.subplots(figsize=(9, max(3.5, len(rows) * 0.19)))
    ax.barh(range(len(rows)), values, color=BAR, height=0.72)
    # Name the year each minimum falls in, just past the bar.
    for i, (_, v, hits, year) in enumerate(rows):
        ax.text(v + 1.2, i, f"{year} · {hits:,} hits", va="center",
                fontsize=6, color="#898781")

    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels(words, fontsize=7)
    ax.invert_yaxis()                      # lowest minimum at the top
    ax.set_xlim(0, 100)
    ax.set_xlabel("lowest year, as % of the word's own peak year", fontsize=9)
    ax.set_title("How far each word falls below its own peak\n"
                 f"({len(rows)} words, minimum over 2013--2024"
                 f"{f', >= {min_hits:,} slang-sense hits' if min_hits else ''}"
                 f"{f', minimum <= {max_min_pct:g}% of peak' if max_min_pct is not None else ''}"
                 f"{f', <= {max_filtered_pct:g}% filtered' if max_filtered_pct is not None else ''})",
                 fontsize=11, loc="left")
    ax.tick_params(axis="y", length=0)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.grid(True, axis="x", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    fig.tight_layout()

    if output:
        output.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output, dpi=150, bbox_inches="tight")
        print(f"Saved: {output}")
    else:
        plt.show()
    plt.close(fig)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[1],
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("input", nargs="?", type=Path, default=DEFAULT_INPUT,
                   help=f"peak_years JSON with percentages (default: {DEFAULT_INPUT.name}).")
    p.add_argument("--filter-rates", type=Path, default=DEFAULT_FILTER_RATES, metavar="CSV",
                   dest="filter_rates",
                   help=f"Per-word filter rates CSV, used when MAX_FILTERED_PCT is set "
                        f"(default: {DEFAULT_FILTER_RATES.name}).")
    p.add_argument("--top", type=int, metavar="N",
                   help="Only plot the N words with the lowest minimum.")
    p.add_argument("-o", "--output", type=Path, default=DEFAULT_OUTPUT,
                   help=f"Output path (default: {DEFAULT_OUTPUT}). Pass '-' to display.")
    args = p.parse_args()

    if not args.input.is_file():
        p.error(f"Input not found: {args.input}")
    filter_rates = None
    if MAX_FILTERED_PCT is not None:
        if not args.filter_rates.is_file():
            p.error(f"Filter rates not found: {args.filter_rates}")
        filter_rates = load_filter_rates(args.filter_rates)
    rows = load(args.input, MIN_HITS, MAX_MIN_PCT, filter_rates, MAX_FILTERED_PCT)
    if args.top:
        rows = rows[:args.top]
    render(rows, MIN_HITS, MAX_MIN_PCT,
           None if str(args.output) == "-" else args.output, MAX_FILTERED_PCT)


if __name__ == "__main__":
    main()
