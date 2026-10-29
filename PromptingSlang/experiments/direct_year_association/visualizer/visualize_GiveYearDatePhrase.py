#!/usr/bin/env python3
"""GiveYear / DatePhrase visualizer.

Charts the two "what year is this slang from?" probes — ``GiveYear``
("The slang word {word} is most associated with what year?") and ``DatePhrase``
("You hear someone saying {word}..., which year are they most likely saying this
in?"). Both take a *word* and the model answers with a *year*.

For a **single** model it draws one timeline per word: words sit on the y-axis
**ordered by their true corpus peak year** (read from the FineWeb
``peak_years_pct.json``), and the x-axis is calendar year (2010--2024). On each
word's horizontal lane we mark the **true peak year** (corpus ground truth)
against the **model's responded years**, aggregating the GiveYear and DatePhrase
samples together, with a connector showing the gap between the two. Responses
(or peaks) outside the year window are clamped to the axis edge and drawn with an
arrow marker.

Usage (run from the PromptingSlang root):
    python experiments/direct_year_association/visualizer/visualize_GiveYearDatePhrase.py
    python experiments/direct_year_association/visualizer/visualize_GiveYearDatePhrase.py --model claude
    python experiments/direct_year_association/visualizer/visualize_GiveYearDatePhrase.py -o figures/giveyear.png
"""

from __future__ import annotations

import argparse
import re
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT))
from src.analysis_utils import load_peak_years, pick_model  # noqa: E402
from src.response_utils import model_short, read_responses  # noqa: E402

DEFAULT_RESPONSES = Path(__file__).resolve().parents[1] / "responses"
DEFAULT_OUTPUT = Path(__file__).resolve().parents[1] / "figures" / "give_year_date_phrase.png"
# True corpus peak years. Edit this path (not a CLI flag) to change it, so the
# choice is tracked in git. peak_years_pct.json lives in the sibling
# FineWebAnalysis project (repo root = parents[4]) and covers all 144 candidates,
# including every current target word.
PEAK_YEARS = Path(__file__).resolve().parents[4] / "FineWebAnalysis" / "peak_years_pct.json"
PROMPT_IDS = ("GiveYear", "DatePhrase")
XMIN, XMAX = 2010, 2024  # x-axis (calendar year) window
# Words the plot font can't draw, shown under a text label instead.
LABELS = {"💀": "skull emoji"}

_YEAR_RE = re.compile(r"(?<!\d)(1\d{3}|20\d{2})(?!\d)")


def responded_year(rec: dict) -> int | None:
    """First 4-digit year (1000-2099) in the response, e.g. '2000s' -> 2000, '1600' -> 1600."""
    m = _YEAR_RE.search(str(rec.get("response") or ""))
    return int(m.group(1)) if m else None


def collect(records: list[dict]) -> dict[str, dict[str, list[int]]]:
    """{model: {word: [responded year, ...]}} pooling GiveYear + DatePhrase together."""
    data: dict[str, dict[str, list[int]]] = defaultdict(lambda: defaultdict(list))
    for rec in records:
        if rec.get("prompt_id") not in PROMPT_IDS:
            continue
        word = (rec.get("variables") or {}).get("word")
        year = responded_year(rec)
        if word and year is not None:
            data[rec.get("model", "unknown")][str(word)].append(year)
    return data


def _clamp(year: float) -> float:
    return min(max(year, XMIN), XMAX)


def _marker(year: float, in_range: str) -> str:
    """Arrow marker when the value falls outside the axis window, else *in_range*."""
    if year > XMAX:
        return ">"
    if year < XMIN:
        return "<"
    return in_range


def render(data: dict, model: str, peaks: dict[str, int], output: Path | None) -> None:
    per_word = data[model]
    # Words ordered by true peak year (earliest at the bottom); words with no
    # known peak are sorted last, alphabetically.
    words = sorted(per_word, key=lambda w: (peaks.get(w, 9999), w))
    if not words:
        sys.exit(f"No responses to chart for model '{model}'.")

    fig, ax = plt.subplots(figsize=(11, max(4.5, len(words) * 0.5)))
    rng = np.random.default_rng(0)
    abs_errors: list[float] = []
    median_abs_errors: list[float] = []
    for i, word in enumerate(words):
        ys = per_word[word]
        ax.plot([XMIN, XMAX], [i, i], color="#dddddd", lw=0.8, zorder=0)

        # Model responded years: faint jittered cloud + a solid mean marker.
        jitter = rng.uniform(-0.16, 0.16, size=len(ys))
        ax.scatter([_clamp(y) for y in ys], i + jitter, s=16, color="#3b6ea5",
                   alpha=0.30, edgecolors="none", zorder=2)
        mean = sum(ys) / len(ys)
        median = float(np.median(ys))

        pk = peaks.get(word)
        if pk is not None:
            abs_errors.append(abs(mean - pk))
            median_abs_errors.append(abs(median - pk))
            # Connector showing the gap between true peak and model mean.
            ax.plot([_clamp(pk), _clamp(mean)], [i, i], color="#999999", lw=1.3,
                    alpha=0.7, zorder=1)
        ax.scatter(_clamp(mean), i, marker=_marker(mean, "D"), s=60, color="#1f3b57",
                   edgecolors="white", linewidths=0.6, zorder=4)
        # Median as a tall orange tick: robust to the occasional origin-year answer
        # (e.g. 1600 for 'methinks') that drags the mean far off.
        in_window = XMIN <= median <= XMAX
        # Out-of-window arrows sit just above the mean's arrow at the same edge.
        ax.scatter(_clamp(median), i if in_window else i + 0.25, marker=_marker(median, "|"),
                   s=180 if in_window else 45,
                   color="#e08a1e", linewidths=2.2 if in_window else 0.6,
                   edgecolors=None if in_window else "white", zorder=4)
        if pk is not None:
            ax.scatter(_clamp(pk), i, marker=_marker(pk, "*"), s=170, color="#d1495b",
                       edgecolors="black", linewidths=0.4, zorder=5)

    ax.set_yticks(range(len(words)))
    labels = [LABELS.get(w, w) for w in words]
    ax.set_yticklabels([f"{l} ({peaks[w]})" if w in peaks else l for w, l in zip(words, labels)],
                       fontsize=8)
    ax.set_ylim(-0.6, len(words) - 0.4)
    ax.set_xticks(range(XMIN, XMAX + 1))
    ax.set_xticklabels([str(y) for y in range(XMIN, XMAX + 1)], rotation=45, ha="right", fontsize=8)
    ax.set_xlim(XMIN - 0.6, XMAX + 0.6)
    ax.set_xlabel("year", fontsize=9)
    ax.set_ylabel("word (true corpus peak year)", fontsize=9)
    ax.grid(axis="x", linestyle=":", alpha=0.4, zorder=0)

    handles = [
        plt.Line2D([], [], marker="*", linestyle="", color="#d1495b", markersize=13,
                   markeredgecolor="black", markeredgewidth=0.4, label="true corpus peak"),
        plt.Line2D([], [], marker="D", linestyle="", color="#1f3b57", markersize=7,
                   markeredgecolor="white", label="model mean response"),
        plt.Line2D([], [], marker="|", linestyle="", color="#e08a1e", markersize=12,
                   markeredgewidth=2.2, label="model median response"),
        plt.Line2D([], [], marker="o", linestyle="", color="#3b6ea5", alpha=0.5,
                   markersize=6, label="individual response"),
        plt.Line2D([], [], marker=">", linestyle="", color="#555555", markersize=7,
                   label=f"beyond {XMIN}–{XMAX}"),
    ]
    ax.legend(handles=handles, loc="lower right", fontsize=8, framealpha=0.9)

    mae = f"{sum(abs_errors) / len(abs_errors):.1f}" if abs_errors else "n/a"
    mae_median = (f"{sum(median_abs_errors) / len(median_abs_errors):.1f}"
                  if median_abs_errors else "n/a")
    ax.set_title(f"Slang → year association — {model_short(model)}\n"
                 f"GiveYear + DatePhrase aggregated; model response vs true corpus peak "
                 f"(mean abs. error: {mae} yr from means, {mae_median} yr from medians)",
                 fontsize=11)
    fig.tight_layout()
    if output:
        output.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output, dpi=110, bbox_inches="tight")
        print(f"Saved: {output}  (mean abs. error vs corpus peak: {mae} yr from means, "
              f"{mae_median} yr from medians)")
    else:
        plt.show()
    plt.close(fig)


def main() -> None:
    p = argparse.ArgumentParser(
        description="Per-word timeline of GiveYear/DatePhrase responses vs true corpus peak, single model.")
    p.add_argument("responses", nargs="?", default=str(DEFAULT_RESPONSES),
                   help=f"Response JSONL file or directory (default: {DEFAULT_RESPONSES}).")
    p.add_argument("-m", "--model", default=None,
                   help="Substring of the model id to chart (default: first available).")
    p.add_argument("-o", "--output", type=Path, default=None,
                   help=f"Output path (default: {DEFAULT_OUTPUT}, with '_<model>' appended to "
                        "the name when --model is given). Pass '-' to display interactively.")
    args = p.parse_args()
    if args.output is None:
        args.output = DEFAULT_OUTPUT
        if args.model:
            # e.g. --model claude -> give_year_date_phrase_claude.png
            suffix = re.sub(r"[^\w.-]+", "_", args.model)
            args.output = DEFAULT_OUTPUT.with_name(f"{DEFAULT_OUTPUT.stem}_{suffix}{DEFAULT_OUTPUT.suffix}")

    records = read_responses(args.responses)
    if not records:
        sys.exit(f"No response records found in {args.responses}.")
    peaks = load_peak_years(PEAK_YEARS)
    if not peaks:
        print(f"Warning: no peak years loaded from {PEAK_YEARS}; "
              "words will be ordered alphabetically and true-peak markers omitted.", file=sys.stderr)
    data = collect(records)
    model = pick_model(
        data, args.model,
        f"No {' / '.join(PROMPT_IDS)} records with usable years found.")
    render(data, model, peaks, None if str(args.output) == "-" else args.output)


if __name__ == "__main__":
    main()
