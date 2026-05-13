#!/usr/bin/env python3
"""Visualize Experiment 2: fill-in-blank logprob probing results.

Figure layout (single PNG):
  Two columns — one per model (Qwen left, Llama right).
  Each column is a horizontal bar chart:
    Y-axis : templates, sorted from most casual (top) to most formal (bottom).
    X-axis : joint probability of the generated word (0–1).
    Bar colour : era of the generated word (grey = standard/neutral English).
    Bar label  : the actual word the model produced.

  Below each column a second strip shows the top-5 first-token alternatives
  as a small heatmap (template × alternative token), giving a sense of how
  much probability mass the model spread across candidates.

Usage:
    python visualizer/viz_exp2.py data/responses/exp2_logprob_probing_<id>.jsonl
    python visualizer/viz_exp2.py data/responses/exp2_logprob_probing_<id>.jsonl -o figures/exp2.png
"""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

ERA_COLORS: dict[str, str] = {
    "pre-2012":  "#5B8DB8",
    "2012–2015": "#67AE6E",
    "2016–2019": "#E8A838",
    "2020–2022": "#D65F4A",
    "2023–2024": "#9B6BB5",
    "":          "#BBBBBB",   # standard / neutral English
}

SLANG_ERAS: dict[str, str] = {
    "sick": "pre-2012", "dope": "pre-2012", "rad": "pre-2012",
    "wicked": "pre-2012", "fresh": "pre-2012", "epic": "pre-2012",
    "legit": "pre-2012", "stoked": "pre-2012",
    "fleek": "2012–2015", "ratchet": "2012–2015", "savage": "2012–2015",
    "extra": "2012–2015", "salty": "2012–2015",
    "lit": "2016–2019", "fire": "2016–2019", "slay": "2016–2019",
    "iconic": "2016–2019", "lowkey": "2016–2019", "highkey": "2016–2019",
    "slaps": "2016–2019", "vibes": "2016–2019", "vibe": "2016–2019",
    "bussin": "2020–2022", "based": "2020–2022", "goated": "2020–2022",
    "mid": "2020–2022", "valid": "2020–2022",
    "rizz": "2023–2024", "snatched": "2023–2024", "ate": "2023–2024",
    "delulu": "2023–2024", "brat": "2023–2024",
}

# Template order: casual / colloquial → formal
TEMPLATE_ORDER = [
    "Bro that movie was",
    "The party last night was",
    "This new show is lowkey",
    "That song",
    "This new track",
    "Their latest single",
    "The beat on that record",
    "Their chemistry is",
    "The new album is",
    "Her comeback was",
    "That was absolutely",
    "That outfit is honestly",
    "That performance was",
    "That speech was genuinely",
]

_STOP = re.compile(r"[.\s!?,\n]|^<")


def _word_and_prob(logprobs_obj: dict) -> tuple[str, float]:
    tokens: list[str] = logprobs_obj.get("tokens") or []
    lps: list[float | None] = logprobs_obj.get("token_logprobs") or []
    parts, total_lp = [], 0.0
    for tok, lp in zip(tokens, lps):
        if _STOP.search(tok):
            break
        parts.append(tok)
        if lp is not None:
            total_lp += float(lp)
    word = "".join(parts).strip().strip("\"'").lower()
    return word, math.exp(total_lp) if parts else 0.0


def _first_token_alts(logprobs_obj: dict) -> list[tuple[str, float]]:
    top_lps = logprobs_obj.get("top_logprobs") or []
    if not top_lps:
        return []
    first = top_lps[0]
    alts = [(tok.strip().lower(), math.exp(float(lp))) for tok, lp in first.items() if tok.strip()]
    alts.sort(key=lambda x: -x[1])
    total = sum(p for _, p in alts)
    return [(tok, p / total) for tok, p in alts] if total else alts


def _era(word: str) -> str:
    return SLANG_ERAS.get(word.lower(), "")


def load_records(path: Path) -> list[dict]:
    decoder = json.JSONDecoder()
    records = []
    text = path.read_text(encoding="utf-8")
    pos = 0
    while pos < len(text):
        while pos < len(text) and text[pos] in " \t\n\r":
            pos += 1
        if pos >= len(text):
            break
        rec, end = decoder.raw_decode(text, pos)
        pos = end
        if rec.get("logprobs"):
            records.append(rec)
    return records


def _model_data(records: list[dict]) -> dict[str, dict[str, tuple[str, float, list]]]:
    """Return {model: {prefix: (word, prob, [(alt_tok, alt_prob), ...])}}."""
    data: dict[str, dict[str, tuple]] = defaultdict(dict)
    for rec in records:
        model = rec["model"].split("/")[-1]
        prefix = (rec.get("variables") or {}).get("prefix", rec.get("prompt_text", ""))
        lp = rec["logprobs"]
        word, prob = _word_and_prob(lp)
        alts = _first_token_alts(lp)
        data[model][prefix] = (word, prob, alts)
    return dict(data)


def render(records: list[dict], output: Path | None) -> None:
    model_data = _model_data(records)
    models = sorted(model_data.keys())

    n_templates = len(TEMPLATE_ORDER)
    n_models    = len(models)

    # Figure: n_models columns, 2 rows (main bars + alternatives strip)
    fig = plt.figure(figsize=(7 * n_models, 11))
    gs  = fig.add_gridspec(
        2, n_models,
        height_ratios=[3, 1],
        hspace=0.08,
        wspace=0.35,
    )

    era_legend_added = False

    for col, model in enumerate(models):
        ax_main = fig.add_subplot(gs[0, col])
        ax_alt  = fig.add_subplot(gs[1, col])
        pdata   = model_data[model]

        # ── Build ordered template list (use TEMPLATE_ORDER, skip any missing) ──
        ordered = [t for t in TEMPLATE_ORDER if t in pdata] + \
                  [t for t in pdata if t not in TEMPLATE_ORDER]

        y_pos = np.arange(len(ordered))
        probs = [pdata[t][1] if t in pdata else 0.0 for t in ordered]
        words = [pdata[t][0] if t in pdata else "?" for t in ordered]
        colors = [ERA_COLORS.get(_era(w), ERA_COLORS[""]) for w in words]

        # ── Main bar chart ────────────────────────────────────────────────────
        bars = ax_main.barh(
            y_pos, probs, color=colors,
            edgecolor="white", linewidth=0.6, height=0.72,
        )

        for i, (word, prob) in enumerate(zip(words, probs)):
            x_lbl = max(prob + 0.02, 0.04)
            ax_main.text(
                x_lbl, i, word,
                va="center", ha="left", fontsize=9,
                color="#222",
                fontweight="bold" if _era(word) else "normal",
            )

        ax_main.set_yticks(y_pos)
        ax_main.set_yticklabels(
            [f'"{t} ___"' for t in ordered],
            fontsize=8.5,
        )
        ax_main.invert_yaxis()
        ax_main.set_xlim(0, 1.35)
        ax_main.set_xlabel("P(generated word)", fontsize=9)
        ax_main.set_title(model, fontsize=11, pad=8)
        ax_main.spines[["top", "right"]].set_visible(False)
        ax_main.axvline(1.0, color="#ccc", linewidth=0.8, linestyle=":")

        # ── Alternatives heatmap strip ────────────────────────────────────────
        # Collect unique top-alt tokens across all templates
        all_alts_list = [pdata[t][2][:5] if t in pdata else [] for t in ordered]
        alt_tokens = list(dict.fromkeys(
            tok for alts in all_alts_list for tok, _ in alts
        ))[:8]

        matrix = np.zeros((len(ordered), len(alt_tokens)))
        for row, alts in enumerate(all_alts_list):
            for tok, prob in alts:
                if tok in alt_tokens:
                    matrix[row, alt_tokens.index(tok)] = prob

        im = ax_alt.imshow(
            matrix, aspect="auto", cmap="YlOrRd",
            vmin=0, vmax=1, interpolation="nearest",
        )
        ax_alt.set_xticks(range(len(alt_tokens)))
        ax_alt.set_xticklabels(alt_tokens, fontsize=7.5, rotation=35, ha="right")
        ax_alt.set_yticks([])
        ax_alt.set_ylabel("Alt. tok.\ndist.", fontsize=7.5)
        ax_alt.set_title("First-token alternatives (P)", fontsize=8, pad=4)

    # ── Era colour legend ─────────────────────────────────────────────────────
    handles = [
        mpatches.Patch(color=c, label=era if era else "Standard English")
        for era, c in ERA_COLORS.items()
    ]
    fig.legend(
        handles=handles,
        title="Era of generated word",
        loc="lower center",
        ncol=len(ERA_COLORS),
        fontsize=8,
        title_fontsize=8.5,
        framealpha=0.8,
        bbox_to_anchor=(0.5, -0.02),
    )

    fig.suptitle(
        "Experiment 2 — Fill-in-Blank Logprob Probing\n"
        "Colour = era of generated word · Bar length = joint probability",
        fontsize=12, y=1.01,
    )

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
    parser = argparse.ArgumentParser(description="Visualize Experiment 2 logprob probing results.")
    parser.add_argument("input", help="exp2 response JSONL file.")
    parser.add_argument("-o", "--output", default=None,
                        help="Save path (PNG/PDF). Displays interactively if omitted.")
    args = parser.parse_args()

    path = Path(args.input)
    if not path.exists():
        print(f"File not found: {path}", file=sys.stderr); sys.exit(1)

    records = load_records(path)
    if not records:
        print("No logprob records found.", file=sys.stderr); sys.exit(1)

    render(records, Path(args.output) if args.output else None)


if __name__ == "__main__":
    main()
