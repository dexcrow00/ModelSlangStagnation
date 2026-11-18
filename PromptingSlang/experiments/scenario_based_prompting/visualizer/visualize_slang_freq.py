#!/usr/bin/env python3
"""Scenario-based prompting: slang-frequency chart.

Counts every word from the curated ``target_words.txt`` list in the free-text
scenario responses and plots each word's response frequency, ordered by its
total corpus occurrence (most -> least, from the FineWeb ``peak_years_pct.json``),
titled with the Pearson correlation between corpus count and response frequency
and its two-tailed p-value. The Spearman correlation with corpus peak year is
still computed and printed, but not drawn.

Words whose corpus statistics rest on fewer than ``--min-peak-hits`` (default
100) sense-filtered occurrences are considered unreliable and excluded from the
correlations and dropped from both panels. By default the chart aggregates over
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
import sys
import textwrap
from collections import Counter
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from scipy.stats import t as t_dist

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT))
from src.analysis_utils import load_peak_records, load_vocab, pick_model, word_pattern  # noqa: E402
from src.response_utils import model_short, read_responses  # noqa: E402

DEFAULT_RESPONSES = Path(__file__).resolve().parents[1] / "responses"
DEFAULT_COUNTS_JSON = Path(__file__).resolve().parents[1] / "slang_counts_by_model.json"
DEFAULT_OUTPUT = Path(__file__).resolve().parents[1] / "figures" / "slang_freq.png"
# Word lists and peak-year files live in the sibling FineWebAnalysis project.
FINEWEB = REPO_ROOT.parent / "FineWebAnalysis"
DEFAULT_WORDS = [FINEWEB / "target_words.txt", FINEWEB / "scenario_words.txt"]
# target_words and scenario_words peaks come from separate peak_year.py runs.
DEFAULT_PEAKS = [FINEWEB / "peak_years_pct.json", FINEWEB / "scenario_peak_years_pct.json"]


def filter_to_model(records: list[dict], requested: str | None) -> tuple[list[dict], str | None]:
    """Records for the one model matching *requested*; all records when it is None."""
    if requested is None:
        return records, None
    chosen = pick_model({r.get("model", "unknown") for r in records}, requested,
                        "No records found.")
    return [r for r in records if r.get("model") == chosen], chosen


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
    pats = {w: word_pattern(w) for w in words}
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
    ax.set_xticks([])  # one point per word; only the ordering matters, not the labels
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


def render(totals: Counter, n_responses: int, peaks: dict[str, dict],
           model_label: str | None, as_rate: bool, min_hits: int, output: Path | None,
           show_unreliable: bool = False, log_y: bool = False,
           caption: str = "") -> None:
    appeared = [w for w in totals if totals[w] > 0]
    if not appeared:
        sys.exit("No target words found in any response.")

    # A corpus statistic is trusted only if backed by at least `min_hits`
    # sense-filtered occurrences; rarer words are greyed out in both panels.
    def reliable_of(w: str) -> bool:
        pk = peaks.get(w)
        return bool(pk and pk["total_hits"] >= min_hits)

    reliable_words = [w for w in appeared if reliable_of(w)]
    grey_words = [w for w in appeared if not reliable_of(w)]

    scale = (1.0 / max(n_responses, 1)) if as_rate else 1.0
    val = {w: totals[w] * scale for w in appeared}

    # --- Spearman rank correlations over the reliable (coloured) words --------
    # Order-independent: the same data points, only plotted in different orders.
    # Rank-based, so robust to the heavy outliers (bro/omg/u) in response usage.
    rel_year = [peaks[w]["peak_year"] for w in reliable_words]
    rel_occ = [peaks[w]["total_hits"] for w in reliable_words]
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
        print(f"  {w:<12}{peaks[w]['peak_year']:>8}{peaks[w]['total_hits']:>12}{totals[w]:>11}")

    # Words ordered by corpus total occurrence (most -> least). Unreliable words are
    # dropped unless --show-unreliable is set, in which case they are shown in grey.
    tail = [] if not show_unreliable else sorted(grey_words, key=lambda w: (-totals[w], w))
    order = sorted(reliable_words, key=lambda w: (-peaks[w]["total_hits"], w)) + tail
    # The x-position already encodes corpus frequency order, so no colour scale is needed.
    colors = ["#4878cf" if reliable_of(w) else "#bbbbbb" for w in order]

    fig, ax = plt.subplots(figsize=(10, 5))
    scale_note = " (log scale)" if log_y else ""
    ylabel = f"occurrences per response{scale_note}" if as_rate else f"occurrences in responses{scale_note}"

    xlabel = "slang words, ordered by descending corpus frequency"
    if show_unreliable:
        xlabel += f"; grey = <{min_hits} corpus hits"
    _draw_panel(fig, ax, order, [val[w] for w in order], colors,
                None, None, None, ylabel, xlabel, log_y=log_y)
    who = model_short(model_label) if model_label else "all models"
    ax.set_title(f"Slang frequency in responses ({who})  ---  "
                 f"Pearson $r$(corpus count, response freq) $= {r_occ:+.2f}$  "
                 f"($n={n_corr}$, {_fmt_p(p_occ)})",
                 fontsize=10)

    if caption:
        # Wrapped by hand: matplotlib's wrap=True ignores the reserved strip below the axes.
        fig.tight_layout(rect=(0, 0.16, 1, 1))
        fig.text(0.01, 0.01, textwrap.fill(caption, 150), ha="left", va="bottom",
                 fontsize=8, style="italic")
    else:
        fig.tight_layout()
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
    p.add_argument("--peaks", type=Path, nargs="+", default=DEFAULT_PEAKS,
                   help="One or more peak_year.py JSON files; their union gives the corpus peaks "
                        f"(default: {', '.join(p.name for p in DEFAULT_PEAKS)}).")
    p.add_argument("--rate", action="store_true",
                   help="Plot occurrences per response instead of raw counts.")
    p.add_argument("--min-peak-hits", type=int, default=100, dest="min_peak_hits", metavar="N",
                   help="Grey out (treat peak year as unreliable) words backed by fewer than N "
                        "sense-filtered corpus hits (default: 100).")
    p.add_argument("--show-unreliable", action="store_true", dest="show_unreliable",
                   help="Show words below --min-peak-hits as greyed out on the chart, instead of droppin them.")
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

    peaks: dict[str, dict] = {}
    for path in args.peaks:
        recs = load_peak_records(path)
        if not recs:
            print(f"Warning: no peak years from {path}.", file=sys.stderr)
        peaks.update(recs)
    if not peaks:
        print("Warning: no peak years loaded; bars will be uncoloured grey.", file=sys.stderr)
    words = load_vocab(args.words)

    use_json = args.counts_json or (not Path(args.responses).exists())
    if use_json:
        counts_path = args.counts_json or DEFAULT_COUNTS_JSON
        if not counts_path.is_file():
            sys.exit(f"Counts JSON not found: {counts_path}")
        totals, n = load_from_counts_json(counts_path, args.model)
        model_label = args.model
        n_models = 1 if args.model else len(json.loads(counts_path.read_text(encoding="utf-8"))["models"])
        print(f"Loaded from {counts_path.name}: {n} responses, {sum(totals.values())} hits")
    else:
        records = read_responses(args.responses)
        if not records:
            sys.exit(f"No response records found in {args.responses}.")
        records, model_label = filter_to_model(records, args.model)
        totals, n = collect(records, words)
        n_models = len({r.get("model") for r in records})

    for w in {x.lower() for x in args.exclude}:
        totals.pop(w, None)

    # ── Figure caption ────────────────────────────────────────────────────────
    # Edit this string, then regenerate to embed the caption in the image.
    # The response counts are filled in from the data.
    caption = (
        f"Experiment 4: We generated about {round(n / max(n_models, 1))} responses per model "
        f"({n} in total) for a user-simulation prompt (see the Experiment 4 appendix) in order "
        "to collect natural slang usage. We then calculated the peak usage year and total "
        "count of each word in the FineWeb corpus. As expected, models more frequently use "
        "slang words that appear in the corpus more often."
    )

    render(totals, n, peaks, model_label, args.rate, args.min_peak_hits,
           None if str(args.output) == "-" else args.output,
           show_unreliable=args.show_unreliable, log_y=args.log_y,
           caption=caption)


if __name__ == "__main__":
    main()
