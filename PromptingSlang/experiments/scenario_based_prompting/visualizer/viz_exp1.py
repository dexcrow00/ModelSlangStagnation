#!/usr/bin/env python3
"""Visualize Experiment 1: slang word frequencies in free-generated responses.

Figure layout (single PNG):
  Left panel  — Horizontal bar chart of detected slang words grouped by era.
                Words that appeared are fully coloured; words chosen as
                era markers that did NOT appear are shown as faint outlines,
                making the era gap visible.
  Right panel — Prompt-type breakdown: how many slang hits each scenario
                type produced, with per-word stacked colour.

Usage (run from the PromptingSlang root):
    python experiments/1_scenario_based_prompting/visualizer/viz_exp1.py data/responses/exp1_scenario_generation_<id>.jsonl
    python experiments/1_scenario_based_prompting/visualizer/viz_exp1.py data/responses/exp1_*.jsonl -o figures/exp1.png
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

# ── Era palette ───────────────────────────────────────────────────────────────
ERA_COLORS: dict[str, str] = {
    "pre-2012":  "#5B8DB8",
    "2012–2015": "#67AE6E",
    "2016–2019": "#E8A838",
    "2020–2022": "#D65F4A",
    "2023–2024": "#9B6BB5",
}

# Words to show in the chart, grouped by era.  Non-appearing words make the
# gap visible; we show them as outline-only bars.
CHART_WORDS: dict[str, list[str]] = {
    "pre-2012":  ["sick", "dope", "epic", "legit", "rad"],
    "2012–2015": ["bae", "fleek", "savage", "squad", "yolo"],
    "2016–2019": ["lit", "fire", "vibe", "bruh", "cancelled"],
    "2020–2022": ["bussin", "based", "mid", "sus", "hits different"],
    "2023–2024": ["rizz", "delulu", "ate", "snatched", "brat"],
}

PROMPT_LABELS: dict[str, str] = {
    "exp1_text_message":   "Text message",
    "exp1_reddit_comment": "Reddit comment",
    "exp1_group_chat":     "Group chat",
    "exp1_reaction_post":  "Reaction post",
}


def _make_pattern(word: str) -> re.Pattern:
    return re.compile(r"(?<![a-z])" + re.escape(word) + r"(?![a-z])", re.IGNORECASE)

_PATTERNS: dict[str, re.Pattern] = {
    w: _make_pattern(w)
    for words in CHART_WORDS.values()
    for w in words
}


def load_records(paths: list[Path]) -> list[dict]:
    decoder = json.JSONDecoder()
    records = []
    for path in paths:
        text = path.read_text(encoding="utf-8")
        pos = 0
        while pos < len(text):
            while pos < len(text) and text[pos] in " \t\n\r":
                pos += 1
            if pos >= len(text):
                break
            rec, end = decoder.raw_decode(text, pos)
            pos = end
            if rec.get("response"):
                records.append(rec)
    return records


def count_words(records: list[dict]) -> tuple[Counter, dict[str, Counter]]:
    """Return (global_counts, {prompt_id: counts})."""
    total: Counter = Counter()
    by_prompt: dict[str, Counter] = defaultdict(Counter)
    for rec in records:
        text = rec["response"]
        pid = rec.get("prompt_id", "unknown")
        for word, pat in _PATTERNS.items():
            hits = len(pat.findall(text))
            if hits:
                total[word] += hits
                by_prompt[pid][word] += hits
    return total, by_prompt


def render(records: list[dict], output: Path | None) -> None:
    n = len(records)
    total, by_prompt = count_words(records)

    all_words = [w for words in CHART_WORDS.values() for w in words]
    all_eras   = [era for era, words in CHART_WORDS.items() for _ in words]
    counts     = [total.get(w, 0) for w in all_words]

    # ── Figure & layout ───────────────────────────────────────────────────────
    fig = plt.figure(figsize=(16, 8))
    gs  = fig.add_gridspec(1, 2, width_ratios=[3, 1.4], wspace=0.35)
    ax_bar  = fig.add_subplot(gs[0])
    ax_prmp = fig.add_subplot(gs[1])

    # ── Left: horizontal bar chart ────────────────────────────────────────────
    y_pos   = np.arange(len(all_words))
    max_cnt = max(counts) if max(counts) > 0 else 1

    for i, (word, era, cnt) in enumerate(zip(all_words, all_eras, counts)):
        color = ERA_COLORS[era]
        if cnt > 0:
            ax_bar.barh(i, cnt, color=color, edgecolor="white", linewidth=0.5, height=0.7)
            ax_bar.text(cnt + 0.15, i, f"{cnt}", va="center", fontsize=8.5, color="#333")
        else:
            # outline-only bar for absent words
            ax_bar.barh(i, max_cnt * 0.04, color="none",
                        edgecolor=color, linewidth=1.2, linestyle="--",
                        height=0.7, alpha=0.55)

    # Era dividers and labels
    era_list = list(CHART_WORDS.keys())
    boundaries = []
    idx = 0
    for era in era_list:
        n_words = len(CHART_WORDS[era])
        mid = idx + n_words / 2 - 0.5
        ax_bar.text(-0.6, mid, era, ha="right", va="center", fontsize=8,
                    color=ERA_COLORS[era], fontweight="bold")
        if idx > 0:
            ax_bar.axhline(idx - 0.5, color="#dddddd", linewidth=0.8)
        idx += n_words

    ax_bar.set_yticks(y_pos)
    ax_bar.set_yticklabels(all_words, fontsize=9)
    ax_bar.invert_yaxis()
    ax_bar.set_xlabel("Occurrences across all responses", fontsize=9)
    ax_bar.set_title(
        f"Slang word frequency — Experiment 1\n"
        f"({n} responses, temperature 0.8, 2 models)",
        fontsize=11, pad=10
    )
    ax_bar.spines[["top", "right"]].set_visible(False)
    ax_bar.set_xlim(left=-3.5)

    # Faint dashed-bar legend note
    absent_patch = mpatches.Patch(
        facecolor="none", edgecolor="#888", linestyle="--",
        label="Not observed (outline = era marker)"
    )
    ax_bar.legend(handles=[absent_patch], fontsize=7.5, loc="lower right",
                  framealpha=0.7)

    # ── Right: prompt-type breakdown ──────────────────────────────────────────
    prompt_ids  = list(PROMPT_LABELS.keys())
    prompt_lbls = [PROMPT_LABELS[p] for p in prompt_ids]
    x = np.arange(len(prompt_ids))
    width = 0.65

    # Stacked by era
    bottoms = np.zeros(len(prompt_ids))
    era_totals: dict[str, np.ndarray] = {}
    for era, words in CHART_WORDS.items():
        era_cnts = np.array([
            sum(by_prompt.get(pid, {}).get(w, 0) for w in words)
            for pid in prompt_ids
        ], dtype=float)
        era_totals[era] = era_cnts
        ax_prmp.bar(x, era_cnts, width, bottom=bottoms,
                    color=ERA_COLORS[era], label=era, edgecolor="white", linewidth=0.5)
        bottoms += era_cnts

    ax_prmp.set_xticks(x)
    ax_prmp.set_xticklabels(prompt_lbls, rotation=20, ha="right", fontsize=8.5)
    ax_prmp.set_ylabel("Total slang hits", fontsize=9)
    ax_prmp.set_title("Hits by scenario type", fontsize=10, pad=8)
    ax_prmp.spines[["top", "right"]].set_visible(False)
    ax_prmp.legend(fontsize=7.5, loc="upper right", framealpha=0.7)

    # Total label above each bar
    for xi, total_cnt in enumerate(bottoms):
        if total_cnt > 0:
            ax_prmp.text(xi, total_cnt + 0.1, str(int(total_cnt)),
                         ha="center", va="bottom", fontsize=8.5)

    fig.tight_layout()
    _save_or_show(fig, output)


def _save_or_show(fig: plt.Figure, output: Path | None) -> None:
    if output:
        output.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output, dpi=180, bbox_inches="tight")
        print(f"Saved: {output}")
    else:
        plt.show()
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Visualize Experiment 1 slang frequency results.")
    parser.add_argument("inputs", nargs="+", help="One or more exp1 response JSONL files.")
    parser.add_argument("-o", "--output", default=None,
                        help="Save path (PNG/PDF). Displays interactively if omitted.")
    args = parser.parse_args()

    paths = [Path(p) for p in args.inputs]
    missing = [p for p in paths if not p.exists()]
    if missing:
        print(f"File(s) not found: {missing}", file=sys.stderr); sys.exit(1)

    records = load_records(paths)
    if not records:
        print("No text response records found.", file=sys.stderr); sys.exit(1)

    render(records, Path(args.output) if args.output else None)


if __name__ == "__main__":
    main()
