#!/usr/bin/env python3
"""GiveYear / DatePhrase visualizer.

Charts the two "what year is this slang from?" probes — ``GiveYear``
("The slang word {word} is most associated with what year?") and
``DatePhrase`` ("You hear someone saying {word}..., which year are they most
likely saying this in?"). Both take a *word* and the model answers with a
*year*, so unlike AssociateSlangLogprobs the year is the response, not a
variable.

For a **single** model it draws one strip-plot panel per prompt with the
**responded year on the y-axis and the word on the x-axis**: every sample is a
jittered point and the per-word mean is marked, so you can read both the
central guess and its spread.

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
from src.response_utils import model_short, read_responses  # noqa: E402

DEFAULT_RESPONSES = Path(__file__).resolve().parents[1] / "responses"
DEFAULT_OUTPUT = Path(__file__).resolve().parents[1] / "figures" / "give_year_date_phrase.png"
PROMPT_IDS = ("GiveYear", "DatePhrase")
WORDS_CHRONO_ORDER = ["lol","sick","troll","bro","swag","vibe","vibes","alpha","red pill","slay","gaslight","glow-up","aura"] 

_YEAR_RE = re.compile(r"(1[89]\d{2}|20\d{2})")


def responded_year(rec: dict) -> int | None:
    """First plausible 4-digit year (18xx-20xx) in the response, e.g. '2000s' -> 2000."""
    m = _YEAR_RE.search(str(rec.get("response") or ""))
    return int(m.group(1)) if m else None


def collect(records: list[dict]) -> dict[str, dict[str, dict[str, list[int]]]]:
    """{model: {prompt_id: {word: [responded year, ...]}}}."""
    data: dict[str, dict[str, dict[str, list[int]]]] = defaultdict(
        lambda: defaultdict(lambda: defaultdict(list)))
    for rec in records:
        pid = rec.get("prompt_id")
        if pid not in PROMPT_IDS:
            continue
        word = (rec.get("variables") or {}).get("word")
        year = responded_year(rec)
        if word and year is not None:
            data[rec.get("model", "unknown")][pid][str(word)].append(year)
    return data


def pick_model(data: dict, requested: str | None) -> str:
    models = sorted(data)
    if not models:
        sys.exit(f"No {' / '.join(PROMPT_IDS)} records with usable years found.")
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


def render(data: dict, model: str, output: Path | None) -> None:
    per_prompt = data[model]
    prompts = [p for p in PROMPT_IDS if per_prompt.get(p)]
    # X-axis follows WORDS_CHRONO_ORDER, then any stragglers not in that list.
    present = {w for p in prompts for w in per_prompt[p]}
    words = [w for w in WORDS_CHRONO_ORDER if w in present] + sorted(present - set(WORDS_CHRONO_ORDER))

    all_years = [y for p in prompts for ys in per_prompt[p].values() for y in ys]
    ymin, ymax = (min(all_years) - 1, max(all_years) + 1) if all_years else (2000, 2025)

    fig, axes = plt.subplots(1, len(prompts), figsize=(max(6.0, len(words) * 0.55) * len(prompts),
                                                       5.0), squeeze=False, sharey=True)
    rng = np.random.default_rng(0)
    for col, pid in enumerate(prompts):
        ax = axes[0][col]
        for xi, word in enumerate(words):
            ys = per_prompt[pid].get(word, [])
            if not ys:
                continue
            jitter = rng.uniform(-0.18, 0.18, size=len(ys))
            ax.scatter(xi + jitter, ys, s=28, alpha=0.55, color="#3b6ea5",
                       edgecolors="none", zorder=2)
            mean = sum(ys) / len(ys)
            ax.scatter(xi, mean, marker="_", s=520, color="#d1495b", linewidths=2.2, zorder=3)
        ax.set_xticks(range(len(words)))
        ax.set_xticklabels(words, rotation=45, ha="right", fontsize=8)
        ax.set_xlim(-0.6, len(words) - 0.4)
        ax.set_ylim(ymin, ymax)
        ax.set_xlabel("word", fontsize=9)
        if col == 0:
            ax.set_ylabel("responded year", fontsize=9)
        ax.set_title(pid, fontsize=10)
        ax.grid(axis="y", linestyle=":", alpha=0.4, zorder=0)

    handles = [
        plt.Line2D([], [], marker="o", linestyle="", color="#3b6ea5", alpha=0.6, label="sample"),
        plt.Line2D([], [], marker="_", linestyle="", color="#d1495b", markersize=12,
                   markeredgewidth=2.2, label="per-word mean"),
    ]
    fig.legend(handles=handles, loc="upper right", fontsize=8, frameon=False)
    fig.suptitle(f"Slang → year association — {model_short(model)}\n"
                 "responded year × word", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    if output:
        output.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output, dpi=110, bbox_inches="tight")
        print(f"Saved: {output}")
    else:
        plt.show()
    plt.close(fig)


def main() -> None:
    p = argparse.ArgumentParser(
        description="Strip plot of GiveYear/DatePhrase responded-year x word for a single model.")
    p.add_argument("responses", nargs="?", default=str(DEFAULT_RESPONSES),
                   help=f"Response JSONL file or directory (default: {DEFAULT_RESPONSES}).")
    p.add_argument("-m", "--model", default=None,
                   help="Substring of the model id to chart (default: first available).")
    p.add_argument("-o", "--output", type=Path, default=DEFAULT_OUTPUT,
                   help=f"Output path (default: {DEFAULT_OUTPUT}). Pass '-' to display interactively.")
    args = p.parse_args()

    records = read_responses(args.responses)
    if not records:
        sys.exit(f"No response records found in {args.responses}.")
    data = collect(records)
    model = pick_model(data, args.model)
    render(data, model, None if str(args.output) == "-" else args.output)


if __name__ == "__main__":
    main()
