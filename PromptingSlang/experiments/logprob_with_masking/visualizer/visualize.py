#!/usr/bin/env python3
"""Logprob-with-masking visualizer.

Produces two charts:

  * <out>_logprob.png  — for models that returned logprobs (open prompts): the
    top first-token candidates ranked by mean probability, grouped per model.
  * <out>_sampled.png  — for non-logprob models (closed prompts, sampled
    repeatedly at temperature): the percentage of samples that produced each
    response word — a sampling-based ("simulated logprob") distribution.

Usage (run from the PromptingSlang root):
    python experiments/logprob_with_masking/visualizer/visualize.py
    python experiments/logprob_with_masking/visualizer/visualize.py -o figures/exp2.png
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
from src.response_utils import (  # noqa: E402
    first_token_distribution, model_short, read_responses, responded_word,
)

DEFAULT_RESPONSES = Path(__file__).resolve().parents[1] / "responses"


def logprob_means(records: list[dict]) -> dict[str, dict[str, float]]:
    """{model: {token: mean first-token probability}} over records with logprobs."""
    acc: dict[str, dict[str, list]] = defaultdict(lambda: defaultdict(list))
    n: dict[str, int] = defaultdict(int)
    for rec in records:
        dist = first_token_distribution(rec)
        if not dist:
            continue
        m = rec.get("model", "unknown")
        n[m] += 1
        for tok, prob in dist.items():
            acc[m][tok].append(prob)
    # mean over records where the token appeared in the top-k
    return {m: {tok: sum(ps) / n[m] for tok, ps in toks.items()} for m, toks in acc.items()}


def sampled_pct(records: list[dict]) -> dict[str, dict[str, float]]:
    """{model: {word: pct of that model's sampled responses}} for text responses."""
    counts: dict[str, Counter] = defaultdict(Counter)
    for rec in records:
        if not rec.get("response"):
            continue
        counts[rec.get("model", "unknown")][responded_word(rec)] += 1
    out = {}
    for m, c in counts.items():
        tot = sum(c.values())
        out[m] = {w: 100.0 * v / tot for w, v in c.items()} if tot else {}
    return out


def _grouped(per_model: dict[str, dict[str, float]], top: int, ylabel: str,
             title: str, output: Path | None) -> None:
    models = sorted(per_model)
    if not models:
        print(f"  (no data for: {title}) — skipping")
        return
    agg: Counter = Counter()
    for d in per_model.values():
        for k, v in d.items():
            agg[k] += v
    keys = [k for k, _ in agg.most_common(top)]
    x = np.arange(len(keys))
    nm = len(models)
    width = 0.8 / max(nm, 1)
    fig, ax = plt.subplots(figsize=(max(10, len(keys) * 0.8), 6))
    for i, m in enumerate(models):
        ax.bar(x + (i - (nm - 1) / 2) * width, [per_model[m].get(k, 0) for k in keys],
               width, label=model_short(m))
    ax.set_xticks(x)
    ax.set_xticklabels(keys, rotation=40, ha="right", fontsize=8)
    ax.set_xlabel("Token / word responded with")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
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
    p = argparse.ArgumentParser(description="Visualize masked-slot logprob and sampled-response distributions.")
    p.add_argument("responses", nargs="?", default=str(DEFAULT_RESPONSES),
                   help=f"Response JSONL file or directory (default: {DEFAULT_RESPONSES}).")
    p.add_argument("--top", type=int, default=15, help="Top N tokens/words to show (default: 15).")
    p.add_argument("-o", "--output", type=Path, default=None,
                   help="Base output path; '_logprob'/'_sampled' suffixes are added. Displays if omitted.")
    args = p.parse_args()

    records = read_responses(args.responses)
    if not records:
        sys.exit(f"No response records found in {args.responses}.")

    lp_out = sm_out = None
    if args.output:
        lp_out = args.output.with_name(f"{args.output.stem}_logprob{args.output.suffix}")
        sm_out = args.output.with_name(f"{args.output.stem}_sampled{args.output.suffix}")

    _grouped(logprob_means(records), args.top,
             "Mean first-token probability",
             "Masked slot — top logprob candidates (logprob models)", lp_out)
    _grouped(sampled_pct(records), args.top,
             "% of sampled responses",
             "Masked slot — sampled response distribution (non-logprob models)", sm_out)


if __name__ == "__main__":
    main()
