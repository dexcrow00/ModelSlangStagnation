#!/usr/bin/env python3
"""EraAppropriate visualizer: "is it appropriate to use {word}" in {year}?

The EraAppropriate prompt ("It's {year}, is it appropriate to use the term
{word}?") sweeps year 2013--2025 for each target word. For a single model this
draws a heatmap with target word on the y-axis (ordered by true corpus peak
year) and prompt year on the x-axis, coloured by the fraction of responses that
answered "yes". Each word's true corpus peak year (from FineWeb
``peak_years.json``) is outlined, so you can see whether a model calls a term
appropriate around when it actually peaked and dated in later years --- a model
tracking usage would show a green (yes) band up to roughly the peak and turn red
(no) afterwards.

Usage (run from the PromptingSlang root):
    python experiments/direct_year_association/visualizer/visualize_EraAppropriate.py
    python experiments/direct_year_association/visualizer/visualize_EraAppropriate.py --model claude
    python experiments/direct_year_association/visualizer/visualize_EraAppropriate.py results -o figures/era.png
"""

from __future__ import annotations

import argparse
import re
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT))
from src.analysis_utils import load_peak_years, pick_model  # noqa: E402
from src.response_utils import model_short, read_responses  # noqa: E402

EXP_DIR = Path(__file__).resolve().parents[1]
DEFAULT_RESPONSES = EXP_DIR / "responses"
DEFAULT_OUTPUT = EXP_DIR / "figures" / "era_appropriate_by_year.png"
DEFAULT_PEAK_YEARS = REPO_ROOT.parent / "FineWebAnalysis" / "peak_years.json"
PROMPT_ID = "EraAppropriate"

_YES = re.compile(r"^(yes|yeah|yep|yup|sure|absolutely|definitely|totally|true|appropriate|ok|okay)\b")
_NO = re.compile(r"^(no|nope|nah|never|not|inappropriate|false)\b")


def classify(resp: str) -> int | None:
    """1 = yes/appropriate, 0 = no/inappropriate, None = unclear/unparseable."""
    if not resp:
        return None
    t = resp.strip().lower()
    t = re.sub(r"^[^a-z]+", "", t)  # drop leading quotes/punctuation/whitespace
    if _YES.match(t):
        return 1
    if _NO.match(t):
        return 0
    # Fallback: first yes/no token anywhere in a short answer.
    if re.search(r"\byes\b|\bappropriate\b", t) and not re.search(r"\bnot\b|\binappropriate\b", t):
        return 1
    if re.search(r"\bno\b|\bnot\b|\binappropriate\b", t):
        return 0
    return None


def collect(records: list[dict]) -> dict[str, dict[str, dict[str, list[int]]]]:
    """{model: {word: {year: [n_yes, n_total, n_unclear]}}} over EraAppropriate prompts."""
    data: dict[str, dict[str, dict[str, list[int]]]] = defaultdict(
        lambda: defaultdict(lambda: defaultdict(lambda: [0, 0, 0])))
    for rec in records:
        if rec.get("prompt_id") != PROMPT_ID:
            continue
        v = rec.get("variables") or {}
        word, year = v.get("word"), v.get("year")
        if not word or not year:
            continue
        cell = data[rec.get("model", "unknown")][word][str(year)]
        verdict = classify(rec.get("response", ""))
        cell[1] += 1
        if verdict == 1:
            cell[0] += 1
        elif verdict is None:
            cell[2] += 1
    return data


def render(data: dict, model: str, peaks: dict[str, int], output: Path | None) -> None:
    per_word = data[model]
    words = sorted(per_word, key=lambda w: (peaks.get(w, 9999), w))  # earliest peak on top
    years = sorted({y for w in words for y in per_word[w]}, key=int)
    if not words or not years:
        sys.exit(f"No data to chart for model '{model}'.")

    def frac_yes(w, y):
        yes, tot, unclear = per_word[w].get(y, [0, 0, 0])
        decided = tot - unclear
        return yes / decided if decided else np.nan

    mat = np.array([[frac_yes(w, y) for y in years] for w in words])

    fig, ax = plt.subplots(figsize=(max(8, len(years) * 0.7), max(5, len(words) * 0.55)))
    cmap = plt.get_cmap("RdYlGn").copy()  # red = "no", green = "yes"
    cmap.set_bad("#eeeeee")
    im = ax.imshow(np.ma.masked_invalid(mat), cmap=cmap, aspect="auto", vmin=0.0, vmax=1.0)

    year_idx = {y: i for i, y in enumerate(years)}
    peak_handle = None
    for r, w in enumerate(words):
        for c in range(len(years)):
            v = mat[r, c]
            if np.isfinite(v):
                ax.text(c, r, f"{round(v * 100)}", ha="center", va="center", fontsize=6, color="black")
        pk = peaks.get(w)
        if pk is not None and str(pk) in year_idx:
            rect = Rectangle((year_idx[str(pk)] - 0.5, r - 0.5), 1, 1, fill=False,
                             edgecolor="#1f3a93", linewidth=2.4)
            ax.add_patch(rect)
            peak_handle = rect

    ax.set_xticks(range(len(years)))
    ax.set_xticklabels(years, rotation=45, ha="right", fontsize=8)
    ax.set_yticks(range(len(words)))
    ax.set_yticklabels([f"{w} ({peaks[w]})" if w in peaks else w for w in words], fontsize=8)
    ax.set_xlabel("prompt year (“It's {year}…”)", fontsize=9)
    ax.set_ylabel("target word (true corpus peak year)", fontsize=9)
    ax.set_title(f"“Is it appropriate to use {{word}}?” — fraction answering yes — "
                 f"{model_short(model)}\ndoes appropriateness fall off after the word's true peak?",
                 fontsize=11)

    if peak_handle is not None:
        ax.legend(handles=[plt.Line2D([], [], marker="s", linestyle="", markerfacecolor="none",
                                      markeredgecolor="#1f3a93", markersize=11, markeredgewidth=2.4,
                                      label="true corpus peak year")],
                  loc="upper left", bbox_to_anchor=(1.18, 1.0), fontsize=8)
    fig.colorbar(im, ax=ax, shrink=0.7, label="fraction answering “yes” (appropriate)")
    fig.tight_layout()
    if output:
        output.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output, dpi=120, bbox_inches="tight")
        print(f"Saved: {output}")
    else:
        plt.show()
    plt.close(fig)


def main() -> None:
    p = argparse.ArgumentParser(
        description="Heatmap of EraAppropriate yes-fraction by prompt year vs true corpus peak, single model.")
    p.add_argument("responses", nargs="?", default=str(DEFAULT_RESPONSES),
                   help=f"Response JSONL file or directory (default: {DEFAULT_RESPONSES}).")
    p.add_argument("-m", "--model", default=None,
                   help="Substring of the model id to chart (default: first available).")
    p.add_argument("--peak-years", type=Path, default=DEFAULT_PEAK_YEARS, dest="peak_years",
                   help=f"peak_years.json with true corpus peaks (default: {DEFAULT_PEAK_YEARS}).")
    p.add_argument("-o", "--output", type=Path, default=DEFAULT_OUTPUT,
                   help=f"Output path (default: {DEFAULT_OUTPUT}). Pass '-' to display interactively.")
    args = p.parse_args()

    records = read_responses(args.responses)
    if not records:
        sys.exit(f"No response records found in {args.responses}.")
    peaks = load_peak_years(args.peak_years)
    if not peaks:
        print(f"Warning: no peak years from {args.peak_years}; ordering alphabetically, "
              "no true-peak outline.", file=sys.stderr)
    data = collect(records)
    model = pick_model(data, args.model, f"No {PROMPT_ID} records found.")
    render(data, model, peaks, None if str(args.output) == "-" else args.output)


if __name__ == "__main__":
    main()
