#!/usr/bin/env python3
"""Semantic-slot visualizer for the forced-choice (multiple-choice) variant.

The free-response prompts ask the model to name a slang word outright; this
variant hands it four candidates ("A) lmao B) lol C) haha D) rofl") and asks for
a letter. Scoring is therefore a lookup rather than a text match, so the sibling
visualizers --- which compare the response text against the target word --- read
these records as an unbroken run of "b" and score them near zero. This script
scores the letter-resolved rows instead and draws them as a ridgeline: one
ridge per target word (earliest true peak at the top), tracing the target
selection rate across prompt years averaged over all models (each model
weighted equally). Each ridge is also coloured by that rate on a diverging
scale centred on chance: blue where models pick the intended word more often
than guessing would, red where they fall below chance (i.e. systematically
prefer a distractor). A dark vertical line through each ridge marks that word's
true corpus peak year, so a model population tracking the data distribution
would put the crest of each ridge on its line.

Cells are scored over *valid* responses only --- a record whose letter could not
be parsed is excluded rather than counted as a miss --- so a partial run shows up
as a thin cell, not a wrong one. ``--min-n`` drops cells too thin to read from
the model average.

Usage (run from the PromptingSlang root):
    python experiments/semantic_slot_queries/visualizer/exp3_multi_choice_visualizer.py
    python experiments/semantic_slot_queries/visualizer/exp3_multi_choice_visualizer.py --min-n 10
"""

from __future__ import annotations

import argparse
import importlib.util
import sys
from collections import Counter, defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import patheffects
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm
from matplotlib.path import Path as MplPath

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT))
from src.analysis_utils import load_peak_years  # noqa: E402
from src.response_utils import model_short, read_responses  # noqa: E402

EXP_DIR = Path(__file__).resolve().parents[1]


def _load(path: Path, name: str):
    """Import a sibling script by path (they are scripts, not an importable package)."""
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# Word order and slot-id mapping come from the free-response visualizer, and the
# A/B/C/D parsing from the forced-choice analysis script, so all three stay in
# lockstep with this one.
_slot = _load(Path(__file__).resolve().parent / "visualize.py", "_slot_viz")
_cf = _load(EXP_DIR / "analysis" / "exp3_choice_freq.py", "_exp3_choice_freq")
WORDS_CHRONO_ORDER = _slot.WORDS_CHRONO_ORDER
target_prompt_id = _slot.target_prompt_id

DEFAULT_RESPONSES = EXP_DIR / "results_from_list"
DEFAULT_PEAK_YEARS = REPO_ROOT.parent / "FineWebAnalysis" / "peak_years.json"
DEFAULT_OUTPUT = EXP_DIR / "figures" / "multi_choice_ridge_by_year.png"


def collect(records: list[dict]) -> tuple[dict, Counter]:
    """``{model: {word: {year: [n_target, n_valid]}}}`` and parse health."""
    rows, health = _cf.score(records)
    id_to_word = {target_prompt_id(w): w for w in WORDS_CHRONO_ORDER}
    data: dict[str, dict[str, dict[str, list[int]]]] = defaultdict(
        lambda: defaultdict(lambda: defaultdict(lambda: [0, 0])))
    for r in rows:
        word = id_to_word.get(r["slot"])
        if word is None:
            continue
        year = r["year"]
        if not year or year == "?":
            continue  # pre-year-dimension data; skip
        cell = data[r["model"]][word][str(year)]
        cell[1] += 1                 # valid (letter parsed) responses
        cell[0] += r["is_target"]    # of which chose the intended word
    return data, health


def chance_rate(records: list[dict]) -> float:
    """Mean 1/(number of options), the rate a model guessing at random would score."""
    sizes = [len(opts) for rec in records
             if (opts := _cf.parse_options(rec.get("prompt_text", "")))]
    return float(np.mean([1.0 / n for n in sizes])) if sizes else 0.25


def report_coverage(data: dict, min_n: int) -> None:
    """Warn about words a model never answered, or answered too thinly to read."""
    for model in sorted(data):
        missing = [w for w in WORDS_CHRONO_ORDER if w not in data[model]]
        thin = sorted({w for w, years in data[model].items()
                       for n in (c[1] for c in years.values()) if n < min_n})
        if missing or thin:
            print(f"Coverage note — {model_short(model)}:", file=sys.stderr)
            if missing:
                print(f"    no responses for: {', '.join(missing)}", file=sys.stderr)
            if thin:
                print(f"    cells below --min-n={min_n} (left out of the mean): {', '.join(thin)}",
                      file=sys.stderr)


def aggregate_rates(data: dict, min_n: int) -> dict[str, dict[str, float]]:
    """``{word: {year: rate}}`` averaged over models, each model weighted equally.

    A mean of per-model rates rather than pooled counts, so a model with more
    valid responses in a cell does not outvote the others. Models with no (or
    too few) responses in a cell are left out of that cell's mean.
    """
    per_cell: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    for model_data in data.values():
        for w, years in model_data.items():
            for y, (target, valid) in years.items():
                if valid >= min_n:
                    per_cell[w][y].append(target / valid)
    return {w: {y: float(np.mean(v)) for y, v in years.items()} for w, years in per_cell.items()}


# Diverging red <-> blue around a neutral gray: below chance reads warm, above cool.
RATE_CMAP = LinearSegmentedColormap.from_list(
    "below_above_chance",
    ["#a82a2b", "#e34948", "#f5c4c2", "#f0efec", "#b7d3f6", "#5598e7", "#1c5cab"])


def render_ridge(data: dict, peaks: dict[str, int], chance: float, min_n: int,
                 output: Path | None) -> None:
    """Ridgeline: one ridge per word of the model-averaged selection rate by prompt year."""
    rates = aggregate_rates(data, min_n)
    words = sorted(rates, key=lambda w: (peaks.get(w, 9999), w))  # earliest peak first (top)
    years = sorted({int(y) for w in words for y in rates[w]})
    if not words or not years:
        sys.exit("No year-resolved forced-choice records found.")

    # Ridge height at 100%, in row spacings. Most rates sit near the ceiling, so
    # ridges taller than a row would hide the one above; keep each in its own band.
    overlap = 0.9
    edge, peak_color = "#52514e", "#0b0b0b"
    norm = TwoSlopeNorm(vmin=0.0, vcenter=chance, vmax=1.0)
    x = np.array(years)
    xs = np.linspace(x[0], x[-1], 600)
    n = len(words)
    fig, ax = plt.subplots(figsize=(max(9, len(years) * 0.75) + 1.5, 1.5 + n * 0.5))

    im = None
    for i, w in enumerate(words):
        base = n - 1 - i
        y = np.array([rates[w].get(str(yr), np.nan) for yr in years])
        ok = np.isfinite(y)
        if not ok.any():
            continue
        top = base + overlap * y
        # Draw top to bottom so each ridge sits in front of the one above it.
        z = 3 * i
        # Colour the area under the ridge by the rate at each x: a one-row gradient
        # image, clipped to the ridge's outline (gaps for missing years stay empty).
        grad = np.interp(xs, x[ok], y[ok])[np.newaxis, :]
        im = ax.imshow(grad, cmap=RATE_CMAP, norm=norm, aspect="auto", origin="lower",
                       interpolation="bilinear", extent=(x[0], x[-1], base, base + overlap),
                       zorder=z)
        outline = ax.fill_between(x, base, top, where=ok, facecolor="none", linewidth=0)
        im.set_clip_path(MplPath.make_compound_path(*outline.get_paths()), ax.transData)
        outline.remove()
        ax.plot(x, top, color=edge, linewidth=1.5, zorder=z + 2)
        ax.plot([x[0], x[-1]], [base, base], color="#c3c2b7", linewidth=0.8, zorder=z + 2)
        pk = peaks.get(w)
        if pk is not None and years[0] <= pk <= years[-1]:
            ax.vlines(pk, base, base + overlap, color=peak_color, linewidth=2, zorder=z + 1,
                      path_effects=[patheffects.withStroke(linewidth=4, foreground="white")])

    # Scale key for ridge height, beside the bottom ridge.
    kx = x[-1] + 0.6
    ax.plot([kx, kx], [0, overlap], color="#52514e", linewidth=1, clip_on=False)
    for frac, label in ((0, "0%"), (chance, f"chance {100 * chance:.0f}%"), (1, "100%")):
        ax.plot([kx - 0.08, kx], [overlap * frac] * 2, color="#52514e", linewidth=1, clip_on=False)
        ax.text(kx + 0.12, overlap * frac, label, va="center", fontsize=7, color="#52514e")

    ax.set_xticks(years)
    ax.set_xticklabels([str(yr) for yr in years], rotation=45, ha="right", fontsize=8)
    ax.set_xlim(x[0] - 0.3, x[-1] + 0.3)
    ax.set_yticks(np.arange(n) + overlap / 2)   # label the middle of each ridge's band
    ax.set_yticklabels([f"{w} ({peaks[w]})" if w in peaks else w for w in reversed(words)],
                       fontsize=8)
    ax.set_ylim(-0.3, n - 1 + overlap + 0.2)
    ax.tick_params(axis="y", length=0)
    for side in ("left", "right", "top"):
        ax.spines[side].set_visible(False)
    ax.grid(axis="x", color="#e1e0d9", linewidth=0.6, zorder=-1)
    ax.set_axisbelow(True)

    ax.set_xlabel("prompt year (“The year is {year}.”)", fontsize=9)
    ax.set_ylabel("target word (true corpus peak year)", fontsize=9)
    ax.set_title("Forced-choice target selection by prompt year — mean of "
                 f"{len(data)} models\ndoes selection peak at the word's true peak year?",
                 fontsize=11, loc="left")
    ax.legend(handles=[plt.Line2D([], [], color=peak_color, linewidth=2,
                                  label="true corpus peak year")],
              loc="lower right", bbox_to_anchor=(1.0, 1.0), fontsize=8, frameon=False)
    if im is not None:
        cbar = fig.colorbar(im, ax=ax, shrink=0.35, anchor=(0.0, 1.0), pad=0.1)
        ticks = [0.0, chance, 0.5, 0.75, 1.0]
        cbar.set_ticks(ticks, labels=[f"{100 * t:.0f}%" + (" (chance)" if t == chance else "")
                                      for t in ticks])
        cbar.ax.tick_params(labelsize=7)
        cbar.set_label("target selection rate", fontsize=8)
        cbar.outline.set_visible(False)
    fig.tight_layout()
    _save(fig, output)


def _save(fig, output: Path | None) -> None:
    if output:
        output.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output, dpi=120, bbox_inches="tight")
        print(f"Saved: {output}")
    else:
        plt.show()
    plt.close(fig)


def main() -> None:
    p = argparse.ArgumentParser(
        description="Charts for the forced-choice (A/B/C/D) semantic-slot queries.")
    p.add_argument("responses", nargs="?", default=str(DEFAULT_RESPONSES),
                   help=f"Response JSONL file or directory (default: {DEFAULT_RESPONSES}).")
    p.add_argument("--peak-years", type=Path, default=DEFAULT_PEAK_YEARS, dest="peak_years",
                   help=f"peak_years.json with true corpus peaks (default: {DEFAULT_PEAK_YEARS}).")
    p.add_argument("--min-n", type=int, default=1, dest="min_n",
                   help="Minimum valid responses for a model's cell to count toward the mean "
                        "(default: 1).")
    p.add_argument("-o", "--output", type=Path, default=None,
                   help=f"Output path (default: {DEFAULT_OUTPUT}). "
                        "Pass '-' to display interactively.")
    args = p.parse_args()

    records = read_responses(args.responses)
    if not records:
        sys.exit(f"No response records found in {args.responses}.")
    data, health = collect(records)
    if not data:
        sys.exit("No forced-choice records found — is this a free-response results directory?")
    parsed = health["clean"] + health["lenient"]
    total = sum(health.values())
    if total:
        print(f"{total} records · {parsed} scored ({100 * parsed / total:.1f}%)")
    report_coverage(data, args.min_n)

    output = args.output if args.output is not None else DEFAULT_OUTPUT
    output = None if str(output) == "-" else Path(output)

    peaks = load_peak_years(args.peak_years)
    if not peaks:
        print(f"Warning: no peak years from {args.peak_years}; ordering alphabetically, "
              "no true-peak lines.", file=sys.stderr)
    render_ridge(data, peaks, chance_rate(records), args.min_n, output)

if __name__ == "__main__":
    main()
