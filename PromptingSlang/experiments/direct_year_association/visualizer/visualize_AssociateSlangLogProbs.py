#!/usr/bin/env python3
"""Direct-year-association visualizer.

Charts only the ``AssociateSlangLogprobs`` prompt — the year×word association
probe ("A standout word in term of popularity increase in {year} was {word}").
For each model it draws a heatmap with **year on the y-axis and word on the
x-axis**, coloured by the mean completion logprob the model assigned to that
(year, word) pairing (higher / closer to 0 = stronger association).

Usage (run from the PromptingSlang root):
    python experiments/direct_year_association/visualizer/visualize.py
    python experiments/direct_year_association/visualizer/visualize.py -o figures/assoc.png
"""

from __future__ import annotations

import argparse
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT))
from src.response_utils import joint_logprob, model_short, read_responses  # noqa: E402

DEFAULT_RESPONSES = Path(__file__).resolve().parents[1] / "responses"
DEFAULT_OUTPUT = Path(__file__).resolve().parents[1] / "figures" / "associate_slang_logprobs.png"
PROMPT_ID = "AssociateSlangLogprobs"
WORDS_CHRONO_ORDER = ["lol","sick","troll","bro","swag","vibe","vibes","alpha","red pill","slay","gaslight","glow-up","aura"] 


def collect(records: list[dict]) -> dict[str, dict[tuple[str, str], list[float]]]:
    """{model: {(year, word): [completion logprob, ...]}} for the assoc prompt."""
    data: dict[str, dict[tuple[str, str], list[float]]] = defaultdict(lambda: defaultdict(list))
    for rec in records:
        if rec.get("prompt_id") != PROMPT_ID:
            continue
        v = rec.get("variables") or {}
        year, word = v.get("year"), v.get("word")
        jl = joint_logprob(rec)
        if year and word and jl is not None:
            data[rec.get("model", "unknown")][(str(year), str(word))].append(jl)
    return data


def _model_output(output: Path, model: str) -> Path:
    """Insert the model short name before the suffix: assoc.png -> assoc_<model>.png."""
    safe = model_short(model).replace("/", "_")
    return output.with_name(f"{output.stem}_{safe}{output.suffix}")


def render(data: dict, output: Path | None) -> None:
    models = sorted(data)
    if not models:
        sys.exit(f"No {PROMPT_ID} records with usable logprobs found.")

    # Years ascending; drawn with origin="lower" below so they increment upward.
    years = sorted({y for m in models for (y, _w) in data[m]}, key=lambda s: (len(s), s))
    # X-axis follows WORDS_CHRONO_ORDER, then any stragglers not in that list.
    present = {w for m in models for (_y, w) in data[m]}
    words = [w for w in WORDS_CHRONO_ORDER if w in present] + sorted(present - set(WORDS_CHRONO_ORDER))

    def cell(m, y, w):
        vals = data[m].get((y, w))
        return sum(vals) / len(vals) if vals else np.nan

    mats = {m: np.array([[cell(m, y, w) for w in words] for y in years]) for m in models}
    finite = [v for mat in mats.values() for v in mat.ravel() if np.isfinite(v)]
    vmin, vmax = (min(finite), max(finite)) if finite else (-1.0, 0.0)

    cmap = plt.get_cmap("viridis").copy()
    cmap.set_bad("#eeeeee")
    panel_w = max(5.5, len(words) * 0.5)
    panel_h = max(3.5, len(years) * 0.4)

    # One figure per model so each model gets its own chart, with a shared
    # colour scale (vmin/vmax) so panels stay directly comparable.
    for m in models:
        fig, ax = plt.subplots(figsize=(panel_w, panel_h))
        im = ax.imshow(np.ma.masked_invalid(mats[m]), cmap=cmap, aspect="auto",
                       vmin=vmin, vmax=vmax, origin="lower")
        ax.set_xticks(range(len(words)))
        ax.set_xticklabels(words, rotation=45, ha="right", fontsize=7)
        ax.set_yticks(range(len(years)))
        ax.set_yticklabels(years, fontsize=7)
        ax.set_xlabel("word", fontsize=8)
        ax.set_ylabel("year", fontsize=8)
        for r in range(len(years)):
            for c in range(len(words)):
                val = mats[m][r, c]
                if np.isfinite(val):
                    ax.text(c, r, f"{val:.1f}", ha="center", va="center", fontsize=5.5,
                            color="white" if (val - vmin) < (vmax - vmin) * 0.5 else "black")

        fig.suptitle(f"Direct year association — AssociateSlangLogprobs — {model_short(m)}\n"
                     "year × word (mean completion logprob)", fontsize=11)
        fig.colorbar(im, ax=ax, shrink=0.7, label="mean completion logprob")
        if output:
            out = _model_output(output, m)
            out.parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(out, dpi=110, bbox_inches="tight")
            print(f"Saved: {out}")
        else:
            plt.show()
        plt.close(fig)


def main() -> None:
    p = argparse.ArgumentParser(description="Heatmap of AssociateSlangLogprobs year x word associations per model.")
    p.add_argument("responses", nargs="?", default=str(DEFAULT_RESPONSES),
                   help=f"Response JSONL file or directory (default: {DEFAULT_RESPONSES}).")
    p.add_argument("-o", "--output", type=Path, default=DEFAULT_OUTPUT,
                   help=f"Output path (default: {DEFAULT_OUTPUT}). Pass '-' to display interactively.")
    args = p.parse_args()

    records = read_responses(args.responses)
    if not records:
        sys.exit(f"No response records found in {args.responses}.")
    render(collect(records), None if str(args.output) == "-" else args.output)


if __name__ == "__main__":
    main()
