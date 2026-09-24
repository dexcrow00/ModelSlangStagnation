#!/usr/bin/env python3
"""
plot_min_pct.py — Plot each word's *minimum* year as a percentage of its own peak
year, from the ``pct_of_peak_by_year`` map written by peak_year.py.

A word whose minimum sits near 100% held a flat rate across 2013--2024; one whose
minimum approaches 0% all but vanished in at least one year, i.e. it has a real
lifecycle rather than a steady presence. Bars are sorted ascending, so the most
dramatic risers/decliners are at the top.

Percent-of-peak amplifies sampling noise for rare words (a word with nine total
hits can read 0% in several years), so ``--min-hits`` screens those out.

Usage (run from FineWebAnalysis/):
    python plot_min_pct.py
    python plot_min_pct.py --min-hits 100 --top 40 -o figures/min_pct.png
    python plot_min_pct.py --max-min-pct 50      # only words that halve from peak
    python plot_min_pct.py peak_years_pct.json -            # display interactively
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
DEFAULT_INPUT = HERE / "peak_years_pct.json"
DEFAULT_OUTPUT = HERE / "figures" / "min_pct_of_peak.png"
BAR, INK, GRID = "#2a78d6", "#52514e", "#e1e0d9"


def load(path: Path, min_hits: int,
         max_min_pct: float | None) -> list[tuple[str, float, int, str]]:
    """``[(word, min pct, hits, year of that minimum)]``, ascending by percentage."""
    records = json.loads(path.read_text(encoding="utf-8"))
    rows = []
    for r in records:
        pct = r.get("pct_of_peak_by_year")
        if not pct:
            sys.exit(f"{path} has no 'pct_of_peak_by_year' — regenerate it with peak_year.py.")
        if r["total_hits"] < min_hits:
            continue
        year, value = min(pct.items(), key=lambda kv: kv[1])
        if max_min_pct is not None and value > max_min_pct:
            continue
        rows.append((r["word"], value, r["total_hits"], year))
    if not rows:
        sys.exit(f"No words with >= {min_hits} hits"
                 f"{f' whose minimum is <= {max_min_pct:g}% of peak' if max_min_pct is not None else ''}.")
    return sorted(rows, key=lambda t: t[1])


def render(rows, min_hits: int, max_min_pct: float | None, output: Path | None) -> None:
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
                 f"{f', minimum <= {max_min_pct:g}% of peak' if max_min_pct is not None else ''})",
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
    p.add_argument("--min-hits", type=int, default=100, metavar="N", dest="min_hits",
                   help="Skip words with fewer than N total slang-sense hits, whose "
                        "percentages are mostly sampling noise (default: 100; 0 keeps all).")
    p.add_argument("--max-min-pct", type=float, metavar="PCT", dest="max_min_pct",
                   help="Skip words whose lowest year still sits above PCT%% of their "
                        "own peak, i.e. keep only words that fall by at least "
                        "(100 - PCT)%% at some point (default: keep all).")
    p.add_argument("--top", type=int, metavar="N",
                   help="Only plot the N words with the lowest minimum.")
    p.add_argument("-o", "--output", type=Path, default=DEFAULT_OUTPUT,
                   help=f"Output path (default: {DEFAULT_OUTPUT}). Pass '-' to display.")
    args = p.parse_args()

    if not args.input.is_file():
        p.error(f"Input not found: {args.input}")
    rows = load(args.input, args.min_hits, args.max_min_pct)
    if args.top:
        rows = rows[:args.top]
    render(rows, args.min_hits, args.max_min_pct,
           None if str(args.output) == "-" else args.output)


if __name__ == "__main__":
    main()
