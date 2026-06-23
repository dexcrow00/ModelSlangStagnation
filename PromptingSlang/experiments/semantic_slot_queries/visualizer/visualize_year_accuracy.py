#!/usr/bin/env python3
"""Semantic-slot target-word recovery accuracy, resolved by prompt year.

The target-recovery prompts are now prefixed with "The year is {year}." for each
year 2013--2025. This visualizer asks: does telling the model a year change which
slang it produces, and if so, is a word recovered best in the year it was
actually most popular?

For a **single** model it draws a heatmap with **target word on the y-axis
(ordered by true corpus peak year) and prompt year on the x-axis**, coloured by
the fraction of responses (pooled over the three phrasings, 30 per cell) that
recovered the intended word. Each word's **true corpus peak year** (from the
FineWeb ``peak_years.json``) is outlined in red, and the model's own
best-accuracy year is marked with a circle --- a model aware of its data
distribution would have the two coincide.

Usage (run from the PromptingSlang root):
    python experiments/semantic_slot_queries/visualizer/visualize_year_accuracy.py
    python experiments/semantic_slot_queries/visualizer/visualize_year_accuracy.py --model claude
    python experiments/semantic_slot_queries/visualizer/visualize_year_accuracy.py -o figures/byyear.png
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT))
from src.response_utils import model_short, read_responses, responded_word  # noqa: E402

# Reuse the matching logic (is_correct / target_prompt_id / WORDS_CHRONO_ORDER)
# from the sibling visualizer so the two stay in lockstep.
_VIZ = Path(__file__).resolve().parent / "visualize.py"
_spec = importlib.util.spec_from_file_location("_slot_viz", _VIZ)
_slot = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_slot)
is_correct, target_prompt_id = _slot.is_correct, _slot.target_prompt_id
WORDS_CHRONO_ORDER = _slot.WORDS_CHRONO_ORDER

EXP_DIR = Path(__file__).resolve().parents[1]
DEFAULT_RESPONSES = EXP_DIR / "responses"
DEFAULT_OUTPUT = EXP_DIR / "figures" / "target_word_accuracy_by_year.png"
DEFAULT_PEAK_YEARS = REPO_ROOT.parent / "FineWebAnalysis" / "peak_years.json"


def load_peak_years(path: Path) -> dict[str, int]:
    """{word: true peak year} from FineWebAnalysis/peak_year.py output."""
    if not path.is_file():
        return {}
    records = json.loads(path.read_text(encoding="utf-8"))
    return {r["word"]: int(r["peak_year"]) for r in records if "word" in r and "peak_year" in r}


def collect(records: list[dict]) -> dict[str, dict[str, dict[str, list[int]]]]:
    """{model: {word: {year: [n_correct, n_total]}}} over the target-word prompts."""
    id_to_word = {target_prompt_id(w): w for w in WORDS_CHRONO_ORDER}
    data: dict[str, dict[str, dict[str, list[int]]]] = defaultdict(
        lambda: defaultdict(lambda: defaultdict(lambda: [0, 0])))
    for rec in records:
        word = id_to_word.get(rec.get("prompt_id"))
        if word is None:
            continue
        year = (rec.get("variables") or {}).get("year")
        if not year:
            continue  # pre-year-dimension data; skip
        cell = data[rec.get("model", "unknown")][word][str(year)]
        cell[1] += 1
        if is_correct(responded_word(rec), word):
            cell[0] += 1
    return data


def pick_model(data: dict, requested: str | None) -> str:
    models = sorted(data)
    if not models:
        sys.exit("No year-resolved target-word records found.")
    if requested is None:
        chosen = models[0]
        if len(models) > 1:
            print(f"Charting model '{chosen}'. Other models present "
                  f"(use --model to pick): {', '.join(m for m in models if m != chosen)}")
        return chosen
    matches = [m for m in models if requested.lower() in m.lower()]
    if not matches:
        sys.exit(f"No model matching '{requested}'. Available: {', '.join(models)}")
    if len(matches) > 1:
        sys.exit(f"'{requested}' matches several models: {', '.join(matches)}. Be more specific.")
    return matches[0]


def render(data: dict, model: str, peaks: dict[str, int], output: Path | None) -> None:
    per_word = data[model]
    words = sorted(per_word, key=lambda w: (peaks.get(w, 9999), w))  # earliest peak first (top)
    years = sorted({y for w in words for y in per_word[w]}, key=int)
    if not words or not years:
        sys.exit(f"No data to chart for model '{model}'.")

    def acc(w, y):
        c, n = per_word[w].get(y, [0, 0])
        return c / n if n else np.nan

    mat = np.array([[acc(w, y) for y in years] for w in words])

    fig, ax = plt.subplots(figsize=(max(8, len(years) * 0.7), max(5, len(words) * 0.55)))
    cmap = plt.get_cmap("viridis").copy()
    cmap.set_bad("#eeeeee")
    im = ax.imshow(np.ma.masked_invalid(mat), cmap=cmap, aspect="auto", vmin=0.0, vmax=1.0)

    year_idx = {y: i for i, y in enumerate(years)}
    peak_handle = None
    for r, w in enumerate(words):
        # Annotate each cell with its accuracy percentage.
        for c in range(len(years)):
            v = mat[r, c]
            if np.isfinite(v):
                ax.text(c, r, f"{round(v * 100)}", ha="center", va="center", fontsize=6,
                        color="white" if v < 0.55 else "black")
        # Outline the true corpus peak year for this word.
        pk = peaks.get(w)
        if pk is not None and str(pk) in year_idx:
            rect = Rectangle((year_idx[str(pk)] - 0.5, r - 0.5), 1, 1, fill=False,
                             edgecolor="#d1495b", linewidth=2.2)
            ax.add_patch(rect)
            peak_handle = rect

    ax.set_xticks(range(len(years)))
    ax.set_xticklabels(years, rotation=45, ha="right", fontsize=8)
    ax.set_yticks(range(len(words)))
    ax.set_yticklabels([f"{w} ({peaks[w]})" if w in peaks else w for w in words], fontsize=8)
    ax.set_xlabel("prompt year (“The year is {year}.”)", fontsize=9)
    ax.set_ylabel("target word (true corpus peak year)", fontsize=9)
    ax.set_title(f"Semantic slot recovery accuracy by prompt year — {model_short(model)}\n"
                 "does accuracy peak at the word's true peak year?", fontsize=11)

    if peak_handle is not None:
        ax.legend(handles=[plt.Line2D([], [], marker="s", linestyle="", markerfacecolor="none",
                                      markeredgecolor="#d1495b", markersize=11, markeredgewidth=2.2,
                                      label="true corpus peak year")],
                  loc="upper left", bbox_to_anchor=(1.18, 1.0), fontsize=8)
    fig.colorbar(im, ax=ax, shrink=0.7, label="recovery accuracy")
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
        description="Heatmap of semantic-slot recovery accuracy by prompt year vs true peak, single model.")
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
    model = pick_model(data, args.model)
    render(data, model, peaks, None if str(args.output) == "-" else args.output)


if __name__ == "__main__":
    main()
