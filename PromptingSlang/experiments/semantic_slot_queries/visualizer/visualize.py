#!/usr/bin/env python3
"""Semantic-slot-queries visualizer.

One chart per input prompt JSONL file. Each chart shows the incidence of every
word the models produced (the slang term each prompt evoked), as a grouped bar
chart — one group of bars per response word, one bar per model.

Records are matched to a prompt file by their prompt_id (each file owns a
distinct set of ids), so the two slot files (semantic_neighbor and
target_words) get separate charts even though their responses share a directory.

Usage (run from the PromptingSlang root):
    python experiments/semantic_slot_queries/visualizer/visualize.py
    python experiments/semantic_slot_queries/visualizer/visualize.py \
        --prompts experiments/semantic_slot_queries/prompts/exp3_semantic_neighbor.jsonl \
        -o figures/semantic.png
"""

from __future__ import annotations

import argparse
import glob
import sys
from collections import Counter, defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT))
from src.prompts import load_prompts  # noqa: E402
from src.response_utils import model_short, read_responses, responded_word  # noqa: E402

EXP_DIR = Path(__file__).resolve().parents[1]
DEFAULT_RESPONSES = EXP_DIR / "responses"
DEFAULT_PROMPTS = sorted(glob.glob(str(EXP_DIR / "prompts" / "*.jsonl")))


def incidence_by_model(records: list[dict], ids: set[str]) -> dict[str, Counter]:
    """{model: Counter(response_word -> count)} over records whose prompt_id is in ids."""
    per_model: dict[str, Counter] = defaultdict(Counter)
    for rec in records:
        if rec.get("prompt_id") not in ids:
            continue
        word = responded_word(rec)
        if word:
            per_model[rec.get("model", "unknown")][word] += 1
    return per_model


def render(per_model: dict[str, Counter], title: str, top: int, output: Path | None) -> None:
    models = sorted(per_model)
    totals: Counter = Counter()
    for c in per_model.values():
        totals.update(c)
    words = [w for w, _ in totals.most_common(top)]
    if not words:
        print(f"  (no responses for {title}) — skipping")
        return

    x = np.arange(len(words))
    n = len(models)
    width = 0.8 / max(n, 1)
    fig, ax = plt.subplots(figsize=(max(10, len(words) * 0.7), 6))
    for i, m in enumerate(models):
        ax.bar(x + (i - (n - 1) / 2) * width, [per_model[m][w] for w in words],
               width, label=model_short(m))
    ax.set_xticks(x)
    ax.set_xticklabels(words, rotation=40, ha="right", fontsize=8)
    ax.set_xlabel("Response word")
    ax.set_ylabel("Incidence (count of responses)")
    ax.set_title(f"Semantic slot queries — {title}\nresponse-word incidence per model")
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
    p = argparse.ArgumentParser(description="Visualize response-word incidence per model, one chart per prompt file.")
    p.add_argument("responses", nargs="?", default=str(DEFAULT_RESPONSES),
                   help=f"Response JSONL file or directory (default: {DEFAULT_RESPONSES}).")
    p.add_argument("--prompts", nargs="+", default=DEFAULT_PROMPTS,
                   help="Prompt JSONL file(s); one chart is produced per file.")
    p.add_argument("--top", type=int, default=15, help="Show the N most common response words (default: 15).")
    p.add_argument("-o", "--output", type=Path, default=None,
                   help="Output path. With multiple prompt files, the file stem is "
                        "appended (<stem>_<promptfile>.png). Displays if omitted.")
    args = p.parse_args()

    records = read_responses(args.responses)
    if not records:
        sys.exit(f"No response records found in {args.responses}.")

    for pf in args.prompts:
        ids = {t.id for t, _lp, _echo in load_prompts(pf)}
        per_model = incidence_by_model(records, ids)
        stem = Path(pf).stem
        out = None
        if args.output:
            out = (args.output if len(args.prompts) == 1
                   else args.output.with_name(f"{args.output.stem}_{stem}{args.output.suffix}"))
        render(per_model, stem, args.top, out)


if __name__ == "__main__":
    main()
