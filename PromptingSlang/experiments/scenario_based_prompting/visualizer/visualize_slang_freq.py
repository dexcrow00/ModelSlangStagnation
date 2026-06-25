#!/usr/bin/env python3
"""Scenario-based prompting: slang-frequency chart.

Counts every word from the curated ``target_words.txt`` list in the free-text
scenario responses and draws two stacked bar charts of response frequency, each
with its own x-axis. The **top panel colours each bar by the word's corpus peak
year** (recency, ordered oldest -> newest) and the **bottom panel by its total
corpus occurrence** (overall frequency, ordered most -> least), both from the
FineWeb ``peak_years.json``. Each panel's title reports the Pearson correlation
between its corpus statistic and response frequency, quantifying the experiment's
central question: does a model's naturalistic slang usage track recency, or
simply overall corpus frequency?

Words whose corpus statistics rest on fewer than ``--min-peak-hits`` (default
100) sense-filtered occurrences are unreliable and excluded from the
correlations and drawn grey in both panels. By default the chart aggregates over
all models; ``--model`` restricts to one.

Usage (run from the PromptingSlang root):
    python experiments/scenario_based_prompting/visualizer/visualize_slang_freq.py
    python experiments/scenario_based_prompting/visualizer/visualize_slang_freq.py --model llama
    python experiments/scenario_based_prompting/visualizer/visualize_slang_freq.py --rate -o figures/slang_freq.png
"""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.colors import LogNorm, Normalize

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT))
from src.response_utils import model_short, read_responses  # noqa: E402

DEFAULT_RESPONSES = Path(__file__).resolve().parents[1] / "responses"
DEFAULT_OUTPUT = Path(__file__).resolve().parents[1] / "figures" / "slang_freq.png"
# target_words.txt and peak_years.json live in the sibling FineWebAnalysis project.
FINEWEB = REPO_ROOT.parent / "FineWebAnalysis"
DEFAULT_WORDS = FINEWEB / "target_words.txt"
DEFAULT_PEAKS = FINEWEB / "peak_years.json"


def load_target_words(path: Path) -> list[str]:
    return [w.strip().lower() for w in path.read_text(encoding="utf-8").splitlines()
            if w.strip() and not w.strip().startswith("#")]


def load_peak_years(path: Path) -> dict[str, tuple[int, int]]:
    """{word: (peak_year, total_hits)}; total_hits gauges how trustworthy the peak is."""
    if not path.is_file():
        return {}
    return {r["word"].lower(): (int(r["peak_year"]), int(r.get("total_hits", 0)))
            for r in json.loads(path.read_text(encoding="utf-8"))
            if "word" in r and "peak_year" in r}


def _pattern(word: str) -> re.Pattern:
    # Word-boundary-ish match tolerating the space/hyphen in multi-word targets
    # ("red pill", "glow-up", "he ate"); case-insensitive.
    return re.compile(rf"(?<![a-z]){re.escape(word)}(?![a-z])", re.IGNORECASE)


def pick_model(records: list[dict], requested: str | None) -> tuple[list[dict], str | None]:
    models = sorted({r.get("model", "unknown") for r in records})
    if requested is None:
        return records, None
    matches = [m for m in models if requested.lower() in m.lower()]
    if not matches:
        sys.exit(f"No model matching '{requested}'. Available: {', '.join(models)}")
    if len(matches) > 1:
        sys.exit(f"'{requested}' matches several models: {', '.join(matches)}. Be more specific.")
    return [r for r in records if r.get("model") == matches[0]], matches[0]


def collect(records: list[dict], words: list[str]) -> tuple[Counter, int]:
    """Total occurrences per target word across *records*, and the response count."""
    pats = {w: _pattern(w) for w in words}
    totals: Counter = Counter()
    n = 0
    for rec in records:
        text = rec.get("response") or ""
        if not text:
            continue
        n += 1
        for w, pat in pats.items():
            hits = len(pat.findall(text))
            if hits:
                totals[w] += hits
    return totals, n


def _pearson(xs: list[float], ys: list[float]) -> float:
    """Pearson correlation coefficient of two equal-length sequences.

    r = sum((x-mx)(y-my)) / sqrt(sum((x-mx)^2) * sum((y-my)^2)).
    """
    n = len(xs)
    if n < 2:
        return float("nan")
    mx, my = sum(xs) / n, sum(ys) / n
    sxy = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    sxx = sum((x - mx) ** 2 for x in xs)
    syy = sum((y - my) ** 2 for y in ys)
    if sxx == 0 or syy == 0:
        return float("nan")
    return sxy / math.sqrt(sxx * syy)


def _draw_panel(fig, ax, words, values, colors, cmap, norm, cbar_label,
                ylabel, xlabel, int_ticks: bool = False) -> None:
    """Response-frequency bars (one per word, in the given order) with own x-axis."""
    ax.bar(range(len(words)), values, color=colors, edgecolor="white", linewidth=0.4)
    ax.set_xticks(range(len(words)))
    ax.set_xticklabels(words, rotation=45, ha="right", fontsize=8)
    ax.set_xlabel(xlabel, fontsize=9)
    ax.set_ylabel(ylabel, fontsize=9)
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    sm = cm.ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, pad=0.01)
    cbar.set_label(cbar_label, fontsize=9)
    if int_ticks:
        cbar.ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{int(v)}"))


def render(totals: Counter, n_responses: int, peaks: dict[str, tuple[int, int]],
           model_label: str | None, as_rate: bool, min_hits: int, output: Path | None) -> None:
    appeared = [w for w in totals if totals[w] > 0]
    if not appeared:
        sys.exit("No target words found in any response.")

    # A corpus statistic is trusted only if backed by at least `min_hits`
    # sense-filtered occurrences; rarer words are greyed out in both panels.
    def reliable_of(w: str) -> bool:
        pk = peaks.get(w)
        return bool(pk and pk[1] >= min_hits)

    reliable_words = [w for w in appeared if reliable_of(w)]
    grey_words = [w for w in appeared if not reliable_of(w)]
    n_greyed = len(grey_words)

    scale = (1.0 / max(n_responses, 1)) if as_rate else 1.0
    val = {w: totals[w] * scale for w in appeared}

    # --- Pearson correlations over the reliable (coloured) words --------------
    # Order-independent: the same data points, only plotted in different orders.
    rel_year = [peaks[w][0] for w in reliable_words]
    rel_occ = [peaks[w][1] for w in reliable_words]
    rel_freq = [totals[w] for w in reliable_words]  # response counts (rate is a constant scale)
    r_year = _pearson(rel_year, rel_freq)
    r_occ = _pearson(rel_occ, rel_freq)
    n_corr = len(reliable_words)

    print(f"\nPearson correlations over {n_corr} reliable words (>= {min_hits} corpus hits), "
          "response frequency vs:")
    print(f"  corpus peak year        r = {r_year:+.3f}")
    print(f"  corpus total occurrence r = {r_occ:+.3f}")
    print(f"\n  {'word':<12}{'peak_yr':>8}{'corpus_occ':>12}{'resp_freq':>11}")
    for w in sorted(reliable_words, key=lambda w: -totals[w]):
        print(f"  {w:<12}{peaks[w][0]:>8}{peaks[w][1]:>12}{totals[w]:>11}")

    # Colour scales from the reliable words.
    if reliable_words:
        peak_norm = Normalize(vmin=min(rel_year),
                              vmax=max(rel_year) if max(rel_year) > min(rel_year) else min(rel_year) + 1)
        occ_norm = LogNorm(vmin=max(min(rel_occ), 1), vmax=max(max(rel_occ), min(rel_occ) + 1))
    else:
        peak_norm, occ_norm = Normalize(2013, 2024), LogNorm(1, 10)
    plasma, viridis = plt.get_cmap("plasma"), plt.get_cmap("viridis")

    # Each panel has its own word order: panel 1 by corpus peak year (oldest ->
    # newest), panel 2 by corpus total occurrence (most -> least). Greyed words last.
    order1 = (sorted(reliable_words, key=lambda w: (peaks[w][0], -totals[w], w))
              + sorted(grey_words, key=lambda w: (-totals[w], w)))
    order2 = (sorted(reliable_words, key=lambda w: (-peaks[w][1], w))
              + sorted(grey_words, key=lambda w: (-totals[w], w)))
    colors1 = [plasma(peak_norm(peaks[w][0])) if reliable_of(w) else "#bbbbbb" for w in order1]
    colors2 = [viridis(occ_norm(peaks[w][1])) if reliable_of(w) else "#bbbbbb" for w in order2]

    width = max(10, max(len(order1), len(order2)) * 0.42)
    fig, (ax_top, ax_bot) = plt.subplots(2, 1, figsize=(width, 10))
    ylabel = "occurrences per response" if as_rate else "occurrences in responses"

    _draw_panel(fig, ax_top, order1, [val[w] for w in order1], colors1,
                plasma, peak_norm, "corpus peak year", ylabel,
                f"slang word (ordered by corpus peak year; grey = <{min_hits} corpus hits)",
                int_ticks=True)
    ax_top.set_title(f"coloured by corpus peak year (recency)  ---  "
                     f"Pearson $r$(peak year, response freq) $= {r_year:+.2f}$  ($n={n_corr}$)",
                     fontsize=10)
    _draw_panel(fig, ax_bot, order2, [val[w] for w in order2], colors2,
                viridis, occ_norm, "corpus total occurrences (log)", ylabel,
                f"slang word (ordered by corpus total occurrence, most $\\to$ least; "
                f"grey = <{min_hits} corpus hits)")
    ax_bot.set_title(f"coloured by total corpus occurrence (overall frequency)  ---  "
                     f"Pearson $r$(corpus count, response freq) $= {r_occ:+.2f}$  ($n={n_corr}$)",
                     fontsize=10)

    total_hits = sum(totals.values())
    who = model_short(model_label) if model_label else "all models"
    greyed_note = f"; {n_greyed} grey (<{min_hits} corpus hits)" if n_greyed else ""
    fig.suptitle(f"Scenario-based prompting --- slang frequency in responses ({who})\n"
                 f"{n_responses} responses, {total_hits} slang hits "
                 f"({total_hits / max(n_responses, 1):.2f} per response){greyed_note}", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    if output:
        output.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output, dpi=150, bbox_inches="tight")
        print(f"Saved: {output}")
    else:
        plt.show()
    plt.close(fig)


def main() -> None:
    p = argparse.ArgumentParser(
        description="Bar chart of target-word slang frequency in scenario responses, coloured by corpus peak year.")
    p.add_argument("responses", nargs="?", default=str(DEFAULT_RESPONSES),
                   help=f"Response JSONL file or directory (default: {DEFAULT_RESPONSES}).")
    p.add_argument("-m", "--model", default=None,
                   help="Restrict to one model (substring match); default aggregates all.")
    p.add_argument("--words", type=Path, default=DEFAULT_WORDS,
                   help=f"Target words file (default: {DEFAULT_WORDS}).")
    p.add_argument("--peaks", type=Path, default=DEFAULT_PEAKS,
                   help=f"peak_years.json with corpus peaks (default: {DEFAULT_PEAKS}).")
    p.add_argument("--rate", action="store_true",
                   help="Plot occurrences per response instead of raw counts.")
    p.add_argument("--min-peak-hits", type=int, default=100, dest="min_peak_hits", metavar="N",
                   help="Grey out (treat peak year as unreliable) words backed by fewer than N "
                        "sense-filtered corpus hits (default: 100).")
    p.add_argument("-o", "--output", type=Path, default=DEFAULT_OUTPUT,
                   help=f"Output path (default: {DEFAULT_OUTPUT}). Pass '-' to display interactively.")
    args = p.parse_args()

    records = read_responses(args.responses)
    if not records:
        sys.exit(f"No response records found in {args.responses}.")
    words = load_target_words(args.words)
    peaks = load_peak_years(args.peaks)
    if not peaks:
        print(f"Warning: no peak years from {args.peaks}; bars will be uncoloured grey.", file=sys.stderr)
    records, model_label = pick_model(records, args.model)
    totals, n = collect(records, words)
    render(totals, n, peaks, model_label, args.rate, args.min_peak_hits,
           None if str(args.output) == "-" else args.output)


if __name__ == "__main__":
    main()
