#!/usr/bin/env python3
"""Scenario-based prompting: slang-frequency chart.

Counts every word from the curated ``target_words.txt`` list in the free-text
scenario responses and draws two stacked bar charts of response frequency, each
with its own x-axis. The **top panel colours each bar by the word's corpus peak
year** (recency, ordered oldest -> newest) and the **bottom panel by its total
corpus occurrence** (overall frequency, ordered most -> least), both from the
FineWeb ``peak_years.json``. The top panel (year) reports Spearman correlation and the bottom panel (total
corpus frequency) reports Pearson correlation, each with a two-tailed p-value,
quantifying the experiment's central question: does a model's naturalistic slang
usage track recency, or simply overall corpus frequency?

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

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.colors import Normalize
from scipy.stats import t as t_dist

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT))
from src.response_utils import model_short, read_responses  # noqa: E402

DEFAULT_RESPONSES = Path(__file__).resolve().parents[1] / "responses"
DEFAULT_COUNTS_JSON = Path(__file__).resolve().parents[1] / "slang_counts_by_model.json"
DEFAULT_OUTPUT = Path(__file__).resolve().parents[1] / "figures" / "slang_freq.png"
# Word lists and peak_years.json live in the sibling FineWebAnalysis project.
FINEWEB = REPO_ROOT.parent / "FineWebAnalysis"
DEFAULT_WORDS = [FINEWEB / "target_words.txt", FINEWEB / "scenario_words.txt"]
DEFAULT_PEAKS = FINEWEB / "peak_years.json"


def load_target_words(paths: list[Path]) -> list[str]:
    """Union of words/phrases across the given files, lowercased, order-stable."""
    seen: set[str] = set()
    words: list[str] = []
    for p in paths:
        for line in p.read_text(encoding="utf-8").splitlines():
            w = line.strip().lower()
            if w and not w.startswith("#") and w not in seen:
                seen.add(w)
                words.append(w)
    return words


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


def load_from_counts_json(path: Path, model: str | None) -> tuple[Counter, int]:
    """Build (totals Counter, n_responses) from a pre-aggregated slang_counts_by_model.json.

    When ``model`` is given (substring match), sums only that model's counts;
    otherwise sums across all models.
    """
    data = json.loads(path.read_text(encoding="utf-8"))
    models_data = data.get("models", {})
    if model:
        matches = [m for m in models_data if model.lower() in m.lower()]
        if not matches:
            sys.exit(f"No model matching '{model}'. Available: {', '.join(models_data)}")
        if len(matches) > 1:
            sys.exit(f"'{model}' matches several models: {', '.join(matches)}. Be more specific.")
        models_data = {matches[0]: models_data[matches[0]]}
    totals: Counter = Counter()
    n_responses = 0
    for info in models_data.values():
        n_responses += info.get("responses", 0)
        for w, c in info.get("words", {}).items():
            totals[w] += c
    return totals, n_responses


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


def _rank(xs: list[float]) -> list[float]:
    """Fractional ranks of *xs* (ties get the average of their positions)."""
    order = sorted(range(len(xs)), key=lambda i: xs[i])
    ranks = [0.0] * len(xs)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and xs[order[j + 1]] == xs[order[i]]:
            j += 1
        avg = (i + j) / 2.0 + 1.0  # 1-based average rank for the tie block
        for k in range(i, j + 1):
            ranks[order[k]] = avg
        i = j + 1
    return ranks


def _spearman(xs: list[float], ys: list[float]) -> float:
    """Spearman rank correlation = Pearson correlation of the ranks (tie-corrected)."""
    return _pearson(_rank(xs), _rank(ys))


def _pvalue(r: float, n: int) -> float:
    """Two-tailed p-value for a correlation coefficient r with sample size n."""
    if n < 3 or math.isnan(r) or abs(r) >= 1.0:
        return float("nan")
    t_stat = r * math.sqrt(n - 2) / math.sqrt(1.0 - r * r)
    return float(2 * t_dist.sf(abs(t_stat), df=n - 2))


def _fmt_p(p: float) -> str:
    if math.isnan(p):
        return "n/a"
    if p < 0.001:
        return "p < 0.001"
    return f"p = {p:.3f}"


def _draw_panel(fig, ax, words, values, colors, cmap, norm, cbar_label,
                ylabel, xlabel, int_ticks: bool = False, log_y: bool = False) -> None:
    """Scatter plot of response frequency (one point per word) with own x-axis."""
    xs = np.arange(len(words), dtype=float)
    ax.scatter(xs, values, c=colors, s=60, zorder=3)
    # Trend line: fit in log space when log_y so it appears straight on the log axis;
    # fit in linear space otherwise.
    ys = np.array(values, dtype=float)
    valid = ys > 0
    if valid.sum() > 1:
        if log_y:
            coeffs = np.polyfit(xs[valid], np.log(ys[valid]), 1)
            trend = np.exp(np.polyval(coeffs, xs))
        else:
            coeffs = np.polyfit(xs[valid], ys[valid], 1)
            trend = np.polyval(coeffs, xs)
        ax.plot(xs, trend, color="black", linewidth=1.2, linestyle="--", alpha=0.55, zorder=2)
    ax.set_xticks(xs)
    ax.set_xticklabels(words, rotation=45, ha="right", fontsize=8)
    ax.set_xlabel(xlabel, fontsize=9)
    ax.set_ylabel(ylabel, fontsize=9)
    if log_y:
        ax.set_yscale("log")
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    if cmap is not None:
        sm = cm.ScalarMappable(norm=norm, cmap=cmap)
        sm.set_array([])
        cbar = fig.colorbar(sm, ax=ax, pad=0.01)
        cbar.set_label(cbar_label, fontsize=9)
        if int_ticks:
            cbar.ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{int(v)}"))


def render(totals: Counter, n_responses: int, peaks: dict[str, tuple[int, int]],
           model_label: str | None, as_rate: bool, min_hits: int, output: Path | None,
           drop_unreliable: bool = False, log_y: bool = False,
           caption: str = "") -> None:
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

    # --- Spearman rank correlations over the reliable (coloured) words --------
    # Order-independent: the same data points, only plotted in different orders.
    # Rank-based, so robust to the heavy outliers (bro/omg/u) in response usage.
    rel_year = [peaks[w][0] for w in reliable_words]
    rel_occ = [peaks[w][1] for w in reliable_words]
    rel_freq = [totals[w] for w in reliable_words]  # response counts (rate is a constant scale)
    r_year = _spearman(rel_year, rel_freq)
    r_occ = _pearson(rel_occ, rel_freq)
    n_corr = len(reliable_words)
    p_year = _pvalue(r_year, n_corr)
    p_occ = _pvalue(r_occ, n_corr)

    print(f"\nCorrelations over {n_corr} reliable words (>= {min_hits} corpus hits), "
          "response frequency vs:")
    print(f"  corpus peak year        Spearman rho = {r_year:+.3f}  {_fmt_p(p_year)}")
    print(f"  corpus total occurrence Pearson  r   = {r_occ:+.3f}  {_fmt_p(p_occ)}")
    print(f"\n  {'word':<12}{'peak_yr':>8}{'corpus_occ':>12}{'resp_freq':>11}")
    for w in sorted(reliable_words, key=lambda w: -totals[w]):
        print(f"  {w:<12}{peaks[w][0]:>8}{peaks[w][1]:>12}{totals[w]:>11}")

    # Colour scale for top panel (peak year → plasma).
    if reliable_words:
        peak_norm = Normalize(vmin=min(rel_year),
                              vmax=max(rel_year) if max(rel_year) > min(rel_year) else min(rel_year) + 1)
    else:
        peak_norm = Normalize(2013, 2024)
    plasma = plt.get_cmap("plasma")

    # Each panel has its own word order: panel 1 by corpus peak year (oldest ->
    # newest), panel 2 by corpus total occurrence (most -> least).
    # Unreliable words are appended greyed-out unless --drop-unreliable is set.
    tail = [] if drop_unreliable else sorted(grey_words, key=lambda w: (-totals[w], w))
    order1 = sorted(reliable_words, key=lambda w: (peaks[w][0], -totals[w], w)) + tail
    order2 = sorted(reliable_words, key=lambda w: (-peaks[w][1], w)) + tail
    colors1 = [plasma(peak_norm(peaks[w][0])) if reliable_of(w) else "#bbbbbb" for w in order1]
    # Bottom panel: x-position already encodes corpus frequency order, no colour needed.
    colors2 = ["#4878cf" if reliable_of(w) else "#bbbbbb" for w in order2]

    width = max(10, max(len(order1), len(order2)) * 0.42)
    fig, (ax_top, ax_bot) = plt.subplots(2, 1, figsize=(width, 10))
    scale_note = " (log scale)" if log_y else ""
    ylabel = f"occurrences per response{scale_note}" if as_rate else f"occurrences in responses{scale_note}"

    xlabel1 = "slang word (ordered by corpus peak year, oldest → newest)"
    if not drop_unreliable:
        xlabel1 += f"; grey = <{min_hits} corpus hits"
    _draw_panel(fig, ax_top, order1, [val[w] for w in order1], colors1,
                plasma, peak_norm, "corpus peak year", ylabel, xlabel1,
                int_ticks=True, log_y=log_y)
    ax_top.set_title(f"coloured by corpus peak year (recency)  ---  "
                     f"Spearman $\\rho$(peak year, response freq) $= {r_year:+.2f}$  "
                     f"($n={n_corr}$, {_fmt_p(p_year)})",
                     fontsize=10)
    xlabel2 = "slang word (descending corpus frequency)"
    if not drop_unreliable:
        xlabel2 += f"; grey = <{min_hits} corpus hits"
    _draw_panel(fig, ax_bot, order2, [val[w] for w in order2], colors2,
                None, None, None, ylabel, xlabel2, log_y=log_y)
    ax_bot.set_title(f"ordered by total corpus occurrence (most → least)  ---  "
                     f"Pearson $r$(corpus count, response freq) $= {r_occ:+.2f}$  "
                     f"($n={n_corr}$, {_fmt_p(p_occ)})",
                     fontsize=10)

    total_hits = sum(totals.values())
    who = model_short(model_label) if model_label else "all models"
    if drop_unreliable:
        greyed_note = f"; {n_greyed} word(s) with <{min_hits} corpus hits excluded" if n_greyed else ""
    else:
        greyed_note = f"; {n_greyed} grey (<{min_hits} corpus hits)" if n_greyed else ""
    fig.suptitle(f"Scenario-based prompting --- slang frequency in responses ({who})\n"
                 f"{n_responses} responses, {total_hits} slang hits "
                 f"({total_hits / max(n_responses, 1):.2f} per response){greyed_note}", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    if caption:
        fig.text(0.5, 0.01, caption, ha="center", va="bottom", fontsize=8,
                 style="italic", wrap=True)
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
    p.add_argument("--words", type=Path, nargs="+", default=DEFAULT_WORDS,
                   help="One or more word-list files; their union is the slang vocabulary "
                        f"(default: {', '.join(p.name for p in DEFAULT_WORDS)}).")
    p.add_argument("--peaks", type=Path, default=DEFAULT_PEAKS,
                   help=f"peak_years.json with corpus peaks (default: {DEFAULT_PEAKS}).")
    p.add_argument("--rate", action="store_true",
                   help="Plot occurrences per response instead of raw counts.")
    p.add_argument("--min-peak-hits", type=int, default=100, dest="min_peak_hits", metavar="N",
                   help="Grey out (treat peak year as unreliable) words backed by fewer than N "
                        "sense-filtered corpus hits (default: 100).")
    p.add_argument("--drop-unreliable", action="store_true", dest="drop_unreliable",
                   help="Exclude words below --min-peak-hits from the chart entirely "
                        "instead of greying them out.")
    p.add_argument("--log-y", action="store_true", dest="log_y",
                   help="Use a log scale on the y-axis (trend line fitted in log space).")
    p.add_argument("--exclude", nargs="*", default=[], metavar="WORD",
                   help="Words to remove from the analysis entirely (case-insensitive). "
                        "E.g. --exclude 💀 lol")
    p.add_argument("--counts-json", type=Path, default=None, dest="counts_json",
                   metavar="FILE",
                   help="Use a pre-aggregated slang_counts_by_model.json instead of raw "
                        "response files. Used automatically when the responses path does not "
                        f"exist (default: {DEFAULT_COUNTS_JSON}).")
    p.add_argument("-o", "--output", type=Path, default=DEFAULT_OUTPUT,
                   help=f"Output path (default: {DEFAULT_OUTPUT}). Pass '-' to display interactively.")
    args = p.parse_args()

    peaks = load_peak_years(args.peaks)
    if not peaks:
        print(f"Warning: no peak years from {args.peaks}; bars will be uncoloured grey.", file=sys.stderr)
    words = load_target_words(args.words)

    use_json = args.counts_json or (not Path(args.responses).exists())
    if use_json:
        counts_path = args.counts_json or DEFAULT_COUNTS_JSON
        if not counts_path.is_file():
            sys.exit(f"Counts JSON not found: {counts_path}")
        totals, n = load_from_counts_json(counts_path, args.model)
        model_label = args.model
        print(f"Loaded from {counts_path.name}: {n} responses, {sum(totals.values())} hits")
    else:
        records = read_responses(args.responses)
        if not records:
            sys.exit(f"No response records found in {args.responses}.")
        records, model_label = pick_model(records, args.model)
        totals, n = collect(records, words)

    for w in {x.lower() for x in args.exclude}:
        totals.pop(w, None)

    # ── Figure caption ────────────────────────────────────────────────────────
    # Edit this string, then regenerate to embed the caption in the image.
    caption = (
        "Experiment 4: We generated 100 responses per model for a user simulation prompt (see appendix C or something)" \
        "in order to collect natural slang usage metrics. We then calculated peak usage year and total count for each word" \
        "in the FineWeb corpus. The results show that, expectedly, models tend to more frequently use slang words that appear in the corpus" \
        "more often. Our finding is that corpus slang volume also correlates with peak slang usage age (older slang is more " \
        "represented in the corpus), leading to models overusing out-of-date slang."
    )

    render(totals, n, peaks, model_label, args.rate, args.min_peak_hits,
           None if str(args.output) == "-" else args.output,
           drop_unreliable=args.drop_unreliable, log_y=args.log_y,
           caption=caption)


if __name__ == "__main__":
    main()
