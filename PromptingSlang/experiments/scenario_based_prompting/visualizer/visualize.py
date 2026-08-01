#!/usr/bin/env python3
"""Scenario-based prompting visualizer.

Counts how often each word from the target_words list appears in the models'
free-text responses, and plots the usage count per model as a grouped bar chart
(one group of bars per target word, one bar per model).

Usage (run from the PromptingSlang root):
    python experiments/scenario_based_prompting/visualizer/visualize.py
    python experiments/scenario_based_prompting/visualizer/visualize.py \
        RESPONSES.jsonl -o figures/scenario.png
"""

from __future__ import annotations

import argparse
import sys
from collections import Counter, defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT))
from src.analysis_utils import load_vocab, word_pattern  # noqa: E402
from src.response_utils import model_short, read_responses  # noqa: E402

DEFAULT_RESPONSES = Path(__file__).resolve().parents[1] / "responses"
DEFAULT_WORDS = REPO_ROOT / "data" / "prompts" / "target_words.txt"


def count_usage(records: list[dict], words: list[str]) -> dict[str, Counter]:
    """Return {model: Counter(word -> total occurrences in its responses)}."""
    pats = {w: word_pattern(w) for w in words}
    per_model: dict[str, Counter] = defaultdict(Counter)
    for rec in records:
        text = rec.get("response") or ""
        if not text:
            continue
        c = per_model[rec.get("model", "unknown")]
        for w, pat in pats.items():
            hits = len(pat.findall(text))
            if hits:
                c[w] += hits
    return per_model


def render(per_model: dict[str, Counter], words: list[str], output: Path | None) -> None:
    models = sorted(per_model)
    # Keep only words that appeared at least once (across any model), preserving order.
    used = [w for w in words if any(per_model[m][w] for m in models)]
    if not used:
        sys.exit("No target words found in any response.")

    x = np.arange(len(used))
    n = len(models)
    width = 0.8 / max(n, 1)
    fig, ax = plt.subplots(figsize=(max(10, len(used) * 0.9), 6))
    for i, m in enumerate(models):
        counts = [per_model[m][w] for w in used]
        ax.bar(x + (i - (n - 1) / 2) * width, counts, width, label=model_short(m))

    ax.set_xticks(x)
    ax.set_xticklabels(used, rotation=30, ha="right")
    ax.set_xlabel("Target word")
    ax.set_ylabel("Occurrences in responses")
    ax.set_title("Scenario prompting — target-word usage per model")
    ax.legend(fontsize="small")
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    fig.tight_layout()
    if output:
        output.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output, dpi=150)
        print(f"Saved: {output}")
    else:
        plt.show()
    plt.close(fig)


def main() -> None:
    p = argparse.ArgumentParser(description="Visualize target-word usage per model in scenario responses.")
    p.add_argument("responses", nargs="?", default=str(DEFAULT_RESPONSES),
                   help=f"Response JSONL file or directory (default: {DEFAULT_RESPONSES}).")
    p.add_argument("--words", type=Path, default=DEFAULT_WORDS,
                   help=f"Target words file (default: {DEFAULT_WORDS}).")
    p.add_argument("-o", "--output", type=Path, default=None,
                   help="Save path; displays interactively if omitted.")
    args = p.parse_args()

    records = read_responses(args.responses)
    if not records:
        sys.exit(f"No response records found in {args.responses}.")
    words = load_vocab(args.words)
    per_model = count_usage(records, words)
    render(per_model, words, args.output)


if __name__ == "__main__":
    main()
