#!/usr/bin/env python3
"""Visualize Experiment 3: semantic neighbour probing results.

Figure layout (single PNG):
  A 3×2 grid — one panel per semantic slot.
  Each panel contains a "word tile" grid: rows = phrasings (1–4),
  columns = models (Qwen | Llama). Each tile is coloured by the era of the
  generated word and annotated with the word itself; tile brightness encodes
  confidence (joint probability).

  Below the main grid a summary strip shows, for each slot, the average
  first-token probability mass landing on each known era — a compact way to
  compare where each model's probability is concentrated.

Usage (run from the PromptingSlang root):
    python experiments/3_semantic_slot_queries/visualizer/viz_exp3.py data/responses/exp3_semantic_neighbor_<id>.jsonl
    python experiments/3_semantic_slot_queries/visualizer/viz_exp3.py data/responses/exp3_semantic_neighbor_<id>.jsonl -o figures/exp3.png
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
import matplotlib.colors as mcolors
import numpy as np

ERA_COLORS: dict[str, str] = {
    "pre-2012":  "#5B8DB8",
    "2012–2015": "#67AE6E",
    "2016–2019": "#E8A838",
    "2020–2022": "#D65F4A",
    "2023–2024": "#9B6BB5",
    "":          "#CCCCCC",
}

SLANG_ERAS: dict[str, str] = {
    "sick": "pre-2012", "dope": "pre-2012", "rad": "pre-2012",
    "wicked": "pre-2012", "fresh": "pre-2012", "epic": "pre-2012",
    "legit": "pre-2012", "stoked": "pre-2012", "gnarly": "pre-2012",
    "fleek": "2012–2015", "savage": "2012–2015", "swag": "2012–2015",
    "bae": "2012–2015", "extra": "2012–2015", "salty": "2012–2015",
    "lit": "2016–2019", "fire": "2016–2019", "slay": "2016–2019",
    "iconic": "2016–2019", "lowkey": "2016–2019", "highkey": "2016–2019",
    "slaps": "2016–2019", "vibe": "2016–2019", "vibes": "2016–2019",
    "bussin": "2020–2022", "based": "2020–2022", "goated": "2020–2022",
    "mid": "2020–2022", "sus": "2020–2022", "valid": "2020–2022",
    "rizz": "2023–2024", "snatched": "2023–2024", "ate": "2023–2024",
    "delulu": "2023–2024", "brat": "2023–2024",
}

SLOT_LABELS: dict[str, str] = {
    "exp3_cool_impressive":   "Cool / impressive",
    "exp3_mediocre":          "Mediocre / disappointing",
    "exp3_charisma":          "Charisma / magnetism",
    "exp3_agreement":         "Enthusiastic agreement",
    "exp3_suspicious":        "Suspicious / untrustworthy",
    "exp3_excellent_sensory": "Excellent (music / food)",
}

SLOT_ORDER = list(SLOT_LABELS.keys())

_STOP = re.compile(r"[.\s!?,\n]|^<")


def _normalize_logprobs(lp: dict) -> tuple[list[str], list[float], list[dict]]:
    """Handle Together-native and OpenAI-compatible logprob formats.

    Strips leading routing-token sequences (<|special|> [routing_arg] ...) produced
    by models like gpt-oss-120b before returning token data.
    """
    content = lp.get("content")
    if content and isinstance(content, list):
        tokens = [item["token"] for item in content]
        lps = [float(item["logprob"]) for item in content]
        top_lps = [
            {a["token"]: a["logprob"] for a in (item.get("top_logprobs") or [])}
            for item in content
        ]
    else:
        tokens = lp.get("tokens") or []
        lps_raw = lp.get("token_logprobs") or []
        lps = [float(x) if x is not None else 0.0 for x in lps_raw]
        top_lps = lp.get("top_logprobs") or []

    # Strip leading routing-token sequences: <|special|> [routing_arg] <|special|> ...
    i = 0
    while i < len(tokens):
        if tokens[i].startswith("<|"):
            i += 1
            if i < len(tokens) and not tokens[i].startswith("<|"):
                i += 1  # skip routing argument (e.g. "analysis")
        else:
            break
    return tokens[i:], lps[i:], (top_lps[i:] if top_lps else [])


def _word_and_prob(lp: dict) -> tuple[str, float]:
    tokens, lps, _ = _normalize_logprobs(lp)
    parts, total = [], 0.0
    for tok, lp_val in zip(tokens, lps):
        if _STOP.search(tok):
            break
        parts.append(tok)
        total += lp_val
    word = "".join(parts).strip().strip("\"'").lower()
    return word, math.exp(total) if parts else 0.0


def _era_prob_mass(lp: dict) -> dict[str, float]:
    """Aggregate first-token alternative probability into era buckets."""
    _, _, top_lps = _normalize_logprobs(lp)
    if not top_lps:
        return {}
    era_mass: dict[str, float] = defaultdict(float)
    first = top_lps[0]
    total = sum(math.exp(float(v)) for v in first.values())
    for tok, lp_val in first.items():
        if tok.startswith("<|"):
            continue
        clean = tok.strip().lower()
        era = SLANG_ERAS.get(clean, "")
        era_mass[era] += math.exp(float(lp_val)) / max(total, 1e-9)
    return dict(era_mass)


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
        if rec.get("logprobs") and rec.get("prompt_id", "").startswith("exp3_"):
            records.append(rec)
    return records


def _organise(records: list[dict]) -> dict[str, dict[str, list[tuple[str, float, str]]]]:
    """Return {slot: {model_short: [(word, prob, phrasing_snippet), ...]}}."""
    data: dict[str, dict[str, list]] = defaultdict(lambda: defaultdict(list))
    for rec in records:
        slot  = rec["prompt_id"]
        model = rec["model"].split("/")[-1]
        word, prob = _word_and_prob(rec["logprobs"])
        snippet    = rec.get("prompt_text", "")[:45]
        data[slot][model].append((word, prob, snippet))
    return {s: dict(m) for s, m in data.items()}


def render(records: list[dict], output: Path | None) -> None:
    data = _organise(records)
    models = sorted({m for slot_data in data.values() for m in slot_data})
    n_models  = len(models)
    n_slots   = len(SLOT_ORDER)
    n_rows_grid = (n_slots + 1) // 2   # 3 rows of 2 slots each

    fig = plt.figure(figsize=(6.5 * 2, 4.5 * n_rows_grid + 2.5))
    outer_gs = fig.add_gridspec(n_rows_grid + 1, 2,
                                 height_ratios=[4.5] * n_rows_grid + [2.5],
                                 hspace=0.55, wspace=0.32)

    for slot_idx, slot in enumerate(SLOT_ORDER):
        row = slot_idx // 2
        col = slot_idx % 2
        ax  = fig.add_subplot(outer_gs[row, col])
        slot_data = data.get(slot, {})
        label     = SLOT_LABELS.get(slot, slot)

        # How many phrasings per model?
        n_phrasings = max((len(v) for v in slot_data.values()), default=0)
        if n_phrasings == 0:
            ax.set_visible(False)
            continue

        # Build a grid: rows = phrasings, cols = models
        # Each cell = coloured tile with word text
        tile_h, tile_w = 1.0, 1.0
        ax.set_xlim(-0.1, n_models * tile_w + 0.1)
        ax.set_ylim(-0.1, n_phrasings * tile_h + 0.5)
        ax.set_aspect("equal")
        ax.axis("off")
        ax.set_title(label, fontsize=11, fontweight="bold", pad=6)

        # Column headers (model names)
        for col_i, model in enumerate(models):
            ax.text(
                col_i * tile_w + tile_w / 2, n_phrasings * tile_h + 0.2,
                model.replace("-Instruct-Turbo", ""),
                ha="center", va="bottom", fontsize=8, color="#444",
            )

        for col_i, model in enumerate(models):
            entries = slot_data.get(model, [])
            for row_i in range(n_phrasings):
                x0 = col_i * tile_w
                y0 = (n_phrasings - 1 - row_i) * tile_h
                if row_i < len(entries):
                    word, prob, snippet = entries[row_i]
                    era = _era(word)
                    base_hex = ERA_COLORS.get(era, ERA_COLORS[""])
                    # Darken/lighten by confidence: prob in [0,1]
                    base_rgb = mcolors.to_rgb(base_hex)
                    # Blend toward white for low confidence
                    alpha_eff = 0.35 + 0.65 * prob
                    tile_rgb = tuple(1 - alpha_eff * (1 - c) for c in base_rgb)
                    text_color = "#111" if sum(tile_rgb) / 3 > 0.55 else "#fff"
                else:
                    word, era = "–", ""
                    tile_rgb  = (0.93, 0.93, 0.93)
                    text_color = "#999"
                    prob = 0.0

                rect = plt.Rectangle(
                    (x0 + 0.04, y0 + 0.04),
                    tile_w - 0.08, tile_h - 0.08,
                    facecolor=tile_rgb, edgecolor="white", linewidth=1.5,
                )
                ax.add_patch(rect)
                ax.text(
                    x0 + tile_w / 2, y0 + tile_h / 2,
                    word, ha="center", va="center",
                    fontsize=9.5, color=text_color, fontweight="bold",
                )
                # Confidence annotation (small, bottom-right of tile)
                ax.text(
                    x0 + tile_w - 0.07, y0 + 0.10,
                    f"{prob:.2f}", ha="right", va="bottom",
                    fontsize=6.5, color=text_color, alpha=0.75,
                )

    # ── Bottom strip: era probability mass per slot × model ──────────────────
    ax_strip = fig.add_subplot(outer_gs[n_rows_grid, :])
    eras_ordered = list(ERA_COLORS.keys())[:-1]   # skip "" (neutral)
    x = np.arange(n_slots)
    bar_width = 0.38
    offsets = np.linspace(-bar_width * (n_models - 1) / 2,
                           bar_width * (n_models - 1) / 2, n_models)

    for m_i, model in enumerate(models):
        for era_i, era in enumerate(eras_ordered):
            era_masses = []
            for slot in SLOT_ORDER:
                slot_recs = [r for r in records
                             if r["prompt_id"] == slot and r["model"].split("/")[-1] == model]
                masses = [_era_prob_mass(r["logprobs"]).get(era, 0.0) for r in slot_recs]
                era_masses.append(sum(masses) / max(len(masses), 1))

            bottom = np.zeros(n_slots)
            for ei in range(era_i):
                prev_era = eras_ordered[ei]
                for si, slot in enumerate(SLOT_ORDER):
                    slot_recs = [r for r in records
                                 if r["prompt_id"] == slot and r["model"].split("/")[-1] == model]
                    masses = [_era_prob_mass(r["logprobs"]).get(prev_era, 0.0) for r in slot_recs]
                    bottom[si] += sum(masses) / max(len(masses), 1)

            ax_strip.bar(
                x + offsets[m_i], era_masses, bar_width,
                bottom=bottom,
                color=ERA_COLORS[era],
                edgecolor="white", linewidth=0.5,
                label=era if m_i == 0 else "_",
            )

    ax_strip.set_xticks(x)
    ax_strip.set_xticklabels(
        [SLOT_LABELS.get(s, s).replace(" / ", "\n") for s in SLOT_ORDER],
        fontsize=8.5,
    )
    ax_strip.set_ylabel("Avg first-token P\nmass on slang era", fontsize=8.5)
    model_labels = " · ".join(m.replace("-Instruct-Turbo", "") for m in models)
    ax_strip.set_title(
        f"Era probability mass in first-token alternatives, by slot\n({model_labels})",
        fontsize=9, pad=6,
    )
    ax_strip.spines[["top", "right"]].set_visible(False)
    ax_strip.set_ylim(0, 0.65)

    # ── Era colour legend ─────────────────────────────────────────────────────
    handles = [
        mpatches.Patch(color=ERA_COLORS[era], label=era)
        for era in eras_ordered
    ]
    handles.append(mpatches.Patch(color=ERA_COLORS[""], label="Standard / neutral"))
    fig.legend(
        handles=handles,
        title="Era of word",
        loc="lower center",
        ncol=len(handles),
        fontsize=8,
        title_fontsize=8.5,
        framealpha=0.8,
        bbox_to_anchor=(0.5, -0.04),
    )

    fig.suptitle(
        "Experiment 3 — Semantic Neighbour Probing\n"
        "Tile colour = era of generated word · Tile brightness = confidence (joint P)",
        fontsize=13, y=1.01,
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
    parser = argparse.ArgumentParser(description="Visualize Experiment 3 semantic neighbour results.")
    parser.add_argument("input", help="exp3 response JSONL file.")
    parser.add_argument("-o", "--output", default=None,
                        help="Save path (PNG/PDF). Displays interactively if omitted.")
    args = parser.parse_args()

    path = Path(args.input)
    if not path.exists():
        print(f"File not found: {path}", file=sys.stderr); sys.exit(1)

    records = load_records(path)
    if not records:
        print("No exp3 logprob records found.", file=sys.stderr); sys.exit(1)

    render(records, Path(args.output) if args.output else None)


if __name__ == "__main__":
    main()
