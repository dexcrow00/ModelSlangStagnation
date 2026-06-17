#!/usr/bin/env python3
"""Direct-year-association visualizer.

One bar chart per prompt type (prompt_id). Each chart groups bars by model;
within a model's group there is one bar per distinct year/word the model
responded with, and the bar height is how often it gave that response.

Usage (run from the PromptingSlang root):
    python experiments/direct_year_association/visualizer/visualize.py
    python experiments/direct_year_association/visualizer/visualize.py -o figures/dya.png
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
from src.response_utils import model_short, read_responses, responded_word  # noqa: E402

DEFAULT_RESPONSES = Path(__file__).resolve().parents[1] / "responses"


def by_prompt(records: list[dict]) -> dict[str, dict[str, Counter]]:
    """{prompt_id: {model: Counter(response_value -> count)}}."""
    out: dict[str, dict[str, Counter]] = defaultdict(lambda: defaultdict(Counter))
    for rec in records:
        val = responded_word(rec)
        if val:
            out[rec.get("prompt_id", "unknown")][rec.get("model", "unknown")][val] += 1
    return out


def render(prompt_id: str, per_model: dict[str, Counter], top: int, output: Path | None) -> None:
    models = sorted(per_model)
    totals: Counter = Counter()
    for c in per_model.values():
        totals.update(c)
    values = [v for v, _ in totals.most_common(top)]  # year/word responses to show
    if not values:
        print(f"  (no responses for {prompt_id}) — skipping")
        return

    x = np.arange(len(models))          # one group per model
    nv = len(values)
    width = 0.8 / max(nv, 1)
    fig, ax = plt.subplots(figsize=(max(9, len(models) * 2.2), 6))
    for j, val in enumerate(values):    # one bar series per response value
        heights = [per_model[m][val] for m in models]
        ax.bar(x + (j - (nv - 1) / 2) * width, heights, width, label=val)

    ax.set_xticks(x)
    ax.set_xticklabels([model_short(m) for m in models])
    ax.set_xlabel("Model")
    ax.set_ylabel("Times responded")
    ax.set_title(f"Direct year association — {prompt_id}\nresponses per model")
    ax.legend(title="response", fontsize="small", ncols=max(1, nv // 8))
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
    p = argparse.ArgumentParser(description="Visualize direct-year-association responses per model, one chart per prompt type.")
    p.add_argument("responses", nargs="?", default=str(DEFAULT_RESPONSES),
                   help=f"Response JSONL file or directory (default: {DEFAULT_RESPONSES}).")
    p.add_argument("--top", type=int, default=10,
                   help="Distinct response values (years/words) to show per chart (default: 10).")
    p.add_argument("-o", "--output", type=Path, default=None,
                   help="Base output path; the prompt_id is appended (<stem>_<prompt_id>.png). "
                        "Displays interactively if omitted.")
    args = p.parse_args()

    records = read_responses(args.responses)
    if not records:
        sys.exit(f"No response records found in {args.responses}.")

    grouped = by_prompt(records)
    for prompt_id in sorted(grouped):
        out = None
        if args.output:
            out = args.output.with_name(f"{args.output.stem}_{prompt_id}{args.output.suffix}")
        render(prompt_id, grouped[prompt_id], args.top, out)


if __name__ == "__main__":
    main()
