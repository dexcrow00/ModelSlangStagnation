#!/usr/bin/env python3
"""Scenario-based prompting: slang-frequency chart.

Counts every word from the curated ``target_words.txt`` list in the free-text
scenario responses and draws a bar chart of each word that appeared, ordered by
and **coloured by its true corpus peak year** (from the FineWeb
``peak_years.json``). The colour gradient makes the experiment's central
question legible at a glance: when a model writes naturally, does it reach for
old or recent slang?

Words with no corpus peak (no sense-filtered FineWeb occurrences) are drawn in
grey at the right. By default the chart aggregates over all models; ``--model``
restricts to one.

Usage (run from the PromptingSlang root):
    python experiments/scenario_based_prompting/visualizer/visualize_slang_freq.py
    python experiments/scenario_based_prompting/visualizer/visualize_slang_freq.py --model llama
    python experiments/scenario_based_prompting/visualizer/visualize_slang_freq.py --rate -o figures/slang_freq.png
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.colors import Normalize

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT))
from src.response_utils import model_short, read_responses  # noqa: E402

DEFAULT_RESPONSES = Path(__file__).resolve().parents[1] / "responses"
DEFAULT_OUTPUT = Path(__file__).resolve().parents[1] / "figures" / "slang_freq.png"
# target_words.txt and peak_years.json live in the sibling FineWebAnalysis project.
FINEWEB = REPO_ROOT.parent / "FineWebAnalysis"
DEFAULT_WORDS = FINEWEB / "target_words.txt"
DEFAULT_PEAKS = FINEWEB / "peak_years.json"


def load_target_words(path: Path) -> list[str]:
    return [w.strip().lower() for w in path.read_text(encoding="utf-8").splitlines()
            if w.strip() and not w.strip().startswith("#")]


def load_peak_years(path: Path) -> dict[str, int]:
    if not path.is_file():
        return {}
    return {r["word"].lower(): int(r["peak_year"])
            for r in json.loads(path.read_text(encoding="utf-8"))
            if "word" in r and "peak_year" in r}


def _pattern(word: str) -> re.Pattern:
    # Word-boundary-ish match tolerating the space/hyphen in multi-word targets
    # ("red pill", "glow-up", "he ate"); case-insensitive.
    return re.compile(rf"(?<![a-z]){re.escape(word)}(?![a-z])", re.IGNORECASE)


def pick_model(records: list[dict], requested: str | None) -> tuple[list[dict], str | None]:
    models = sorted({r.get("model", "unknown") for r in records})
    if requested is None:
        return records, None
    matches = [m for m in models if requested.lower() in m.lower()]
    if not matches:
        sys.exit(f"No model matching '{requested}'. Available: {', '.join(models)}")
    if len(matches) > 1:
        sys.exit(f"'{requested}' matches several models: {', '.join(matches)}. Be more specific.")
    return [r for r in records if r.get("model") == matches[0]], matches[0]


def collect(records: list[dict], words: list[str]) -> tuple[Counter, int]:
    """Total occurrences per target word across *records*, and the response count."""
    pats = {w: _pattern(w) for w in words}
    totals: Counter = Counter()
    n = 0
    for rec in records:
        text = rec.get("response") or ""
        if not text:
            continue
        n += 1
        for w, pat in pats.items():
            hits = len(pat.findall(text))
            if hits:
                totals[w] += hits
    return totals, n


def render(totals: Counter, n_responses: int, peaks: dict[str, int],
           model_label: str | None, as_rate: bool, output: Path | None) -> None:
    appeared = [w for w in totals if totals[w] > 0]
    if not appeared:
        sys.exit("No target words found in any response.")
    # Words with a known corpus peak, oldest -> newest; undated words last.
    dated = sorted((w for w in appeared if w in peaks), key=lambda w: (peaks[w], -totals[w], w))
    undated = sorted((w for w in appeared if w not in peaks), key=lambda w: (-totals[w], w))
    words = dated + undated

    scale = (1.0 / max(n_responses, 1)) if as_rate else 1.0
    values = [totals[w] * scale for w in words]

    cmap = plt.get_cmap("plasma")
    if dated:
        lo, hi = min(peaks[w] for w in dated), max(peaks[w] for w in dated)
        norm = Normalize(vmin=lo, vmax=hi if hi > lo else lo + 1)
    else:
        norm = Normalize(vmin=2013, vmax=2024)
    colors = [cmap(norm(peaks[w])) if w in peaks else "#bbbbbb" for w in words]

    fig, ax = plt.subplots(figsize=(max(10, len(words) * 0.42), 6))
    ax.bar(range(len(words)), values, color=colors, edgecolor="white", linewidth=0.4)
    ax.set_xticks(range(len(words)))
    ax.set_xticklabels(words, rotation=45, ha="right", fontsize=8)
    ax.set_xlabel("slang word (ordered by corpus peak year; grey = no corpus peak)", fontsize=9)
    ax.set_ylabel("occurrences per response" if as_rate else "occurrences in responses", fontsize=9)
    ax.grid(axis="y", linestyle="--", alpha=0.4)

    sm = cm.ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, pad=0.01)
    cbar.set_label("corpus peak year", fontsize=9)
    cbar.ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{int(v)}"))

    total_hits = sum(totals.values())
    who = model_short(model_label) if model_label else "all models"
    ax.set_title(f"Scenario-based prompting --- slang frequency by corpus peak year ({who})\n"
                 f"{n_responses} responses, {total_hits} slang hits "
                 f"({total_hits / max(n_responses, 1):.2f} per response)", fontsize=11)
    fig.tight_layout()
    if output:
        output.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output, dpi=150, bbox_inches="tight")
        print(f"Saved: {output}")
    else:
        plt.show()
    plt.close(fig)


def main() -> None:
    p = argparse.ArgumentParser(
        description="Bar chart of target-word slang frequency in scenario responses, coloured by corpus peak year.")
    p.add_argument("responses", nargs="?", default=str(DEFAULT_RESPONSES),
                   help=f"Response JSONL file or directory (default: {DEFAULT_RESPONSES}).")
    p.add_argument("-m", "--model", default=None,
                   help="Restrict to one model (substring match); default aggregates all.")
    p.add_argument("--words", type=Path, default=DEFAULT_WORDS,
                   help=f"Target words file (default: {DEFAULT_WORDS}).")
    p.add_argument("--peaks", type=Path, default=DEFAULT_PEAKS,
                   help=f"peak_years.json with corpus peaks (default: {DEFAULT_PEAKS}).")
    p.add_argument("--rate", action="store_true",
                   help="Plot occurrences per response instead of raw counts.")
    p.add_argument("-o", "--output", type=Path, default=DEFAULT_OUTPUT,
                   help=f"Output path (default: {DEFAULT_OUTPUT}). Pass '-' to display interactively.")
    args = p.parse_args()

    records = read_responses(args.responses)
    if not records:
        sys.exit(f"No response records found in {args.responses}.")
    words = load_target_words(args.words)
    peaks = load_peak_years(args.peaks)
    if not peaks:
        print(f"Warning: no peak years from {args.peaks}; bars will be uncoloured grey.", file=sys.stderr)
    records, model_label = pick_model(records, args.model)
    totals, n = collect(records, words)
    render(totals, n, peaks, model_label, args.rate,
           None if str(args.output) == "-" else args.output)


if __name__ == "__main__":
    main()
