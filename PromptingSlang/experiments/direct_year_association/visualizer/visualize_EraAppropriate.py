#!/usr/bin/env python3
"""EraAppropriate visualizer: "is it appropriate to use {word}" in {year}?

The EraAppropriate prompt ("It's {year}, is it appropriate to use the term
{word}?") sweeps year 2013--2025 for each target word. The score for a cell is
the fraction of decided (yes/no) responses that answered "yes". Two views:

``--mode ridge`` (default) --- a ridgeline aggregated over all models (each
model weighted equally): one ridge per target word (earliest true corpus peak
at the top) tracing the yes-rate across prompt years, coloured blue where most
answers say "appropriate" and red where most say "no". A dark vertical line
through each ridge marks the word's true corpus peak year (from FineWeb
``peak_years_pct.json``) --- a model population tracking usage would keep each
ridge high up to its line and let it drop off afterwards.

CAPTION under the chart says how answers are scored.

Usage (run from the PromptingSlang root):
    python experiments/direct_year_association/visualizer/visualize_EraAppropriate.py
    python experiments/direct_year_association/visualizer/visualize_EraAppropriate.py results -o figures/era.png
"""

from __future__ import annotations

import argparse
import re
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT))
from src.analysis_utils import load_peak_years, pick_model  # noqa: E402
from src.response_utils import model_short, read_responses  # noqa: E402
from src.ridgeline import mean_rates, render_ridgeline  # noqa: E402

EXP_DIR = Path(__file__).resolve().parents[1]
DEFAULT_RESPONSES = EXP_DIR / "responses"
DEFAULT_OUTPUT = {
    "ridge": EXP_DIR / "figures" / "era_appropriate_ridge_by_year.png",
}
DEFAULT_PEAK_YEARS = REPO_ROOT.parent / "FineWebAnalysis" / "peak_years_pct.json"
PROMPT_ID = "EraAppropriate"

# Printed under the chart. Keep it in step with _YES / _NO below.
CAPTION = (
    "Answers of \u201cdated\u201d, \u201canachronistic\u201d, \u201carchaic\u201d or "
    "\u201cearly\u201d count as no, and \u201ccurrent\u201d or \u201ccontemporary\u201d as "
    "yes; other answers (e.g. \u201ccontext-dependent\u201d) are left out. For the "
    "offensive terms cuck, cucked, thot and stfu, a \u201cno\u201d may reflect the "
    "word\u2019s meaning rather than its era.")

# The prompt offers "appropriate" vs "anachronistic or dated", so the models often
# answer with an era word rather than yes/no: current/contemporary read as yes,
# dated/anachronistic/archaic/early (too soon for the term) as no.
_YES = re.compile(r"^(yes|yeah|yep|yup|sure|absolutely|definitely|totally|true|appropriate|ok|okay"
                  r"|current|contemporary)\b")
_NO = re.compile(r"^(no|nope|nah|never|not|inappropriate|false"
                 r"|dated|anachronistic|archaic|early)\b")


def classify(resp: str) -> int | None:
    """1 = yes/appropriate, 0 = no/inappropriate, None = unclear/unparseable."""
    if not resp:
        return None
    t = resp.strip().lower()
    t = re.sub(r"^[^a-z]+", "", t)  # drop leading quotes/punctuation/whitespace
    if _YES.match(t):
        return 1
    if _NO.match(t):
        return 0
    # Fallback: first yes/no token anywhere in a short answer.
    if re.search(r"\byes\b|\bappropriate\b", t) and not re.search(r"\bnot\b|\binappropriate\b", t):
        return 1
    if re.search(r"\bno\b|\bnot\b|\binappropriate\b", t):
        return 0
    return None


def collect(records: list[dict]) -> dict[str, dict[str, dict[str, list[int]]]]:
    """{model: {word: {year: [n_yes, n_total, n_unclear]}}} over EraAppropriate prompts."""
    data: dict[str, dict[str, dict[str, list[int]]]] = defaultdict(
        lambda: defaultdict(lambda: defaultdict(lambda: [0, 0, 0])))
    for rec in records:
        if rec.get("prompt_id") != PROMPT_ID:
            continue
        v = rec.get("variables") or {}
        word, year = v.get("word"), v.get("year")
        if not word or not year:
            continue
        cell = data[rec.get("model", "unknown")][word][str(year)]
        verdict = classify(rec.get("response", ""))
        cell[1] += 1
        if verdict == 1:
            cell[0] += 1
        elif verdict is None:
            cell[2] += 1
    return data


def yes_rate(cell: list[int]) -> float:
    """Fraction of decided (non-unclear) responses answering yes; NaN if none were decided."""
    yes, tot, unclear = cell
    decided = tot - unclear
    return yes / decided if decided else np.nan


def render_ridge(data: dict, peaks: dict[str, int], output: Path | None,
                 caption: str | None = None) -> None:
    """Ridgeline of the yes-rate by prompt year, averaged over all models."""
    per_model = {m: {w: {y: yes_rate(c) for y, c in years.items()} for w, years in words.items()}
                 for m, words in data.items()}
    render_ridgeline(
        mean_rates(per_model), peaks, center=0.5, center_label="even split",
        title=f"“Is it appropriate to use {{word}}?” — fraction answering yes — mean of "
              f"{len(data)} models\ndoes appropriateness fall off after the word's true peak?",
        xlabel="prompt year (“It's {year}…”)",
        cbar_label="fraction answering “yes” (appropriate)", output=output,
        caption=caption)

def main() -> None:
    p = argparse.ArgumentParser(
        description="EraAppropriate yes-fraction by prompt year vs true corpus peak.")
    p.add_argument("responses", nargs="?", default=str(DEFAULT_RESPONSES),
                   help=f"Response JSONL file or directory (default: {DEFAULT_RESPONSES}).")
    p.add_argument("-m", "--model", default=None,
                   help="Substring of the model id to chart (heatmap mode; default: first available).")
    p.add_argument("--peak-years", type=Path, default=DEFAULT_PEAK_YEARS, dest="peak_years",
                   help=f"peak_years_pct.json with true corpus peaks (default: {DEFAULT_PEAK_YEARS}).")
    p.add_argument("-o", "--output", type=Path, default=None,
                   help="Output path (default: figures/era_appropriate_{ridge_,}by_year.png). "
                        "Pass '-' to display interactively.")
    args = p.parse_args()

    records = read_responses(args.responses)
    if not records:
        sys.exit(f"No response records found in {args.responses}.")
    peaks = load_peak_years(args.peak_years)
    if not peaks:
        print(f"Warning: no peak years from {args.peak_years}; ordering alphabetically, "
              "no true-peak outline.", file=sys.stderr)
    words = {str(r["variables"]["word"]).lower() for r in records
             if r.get("prompt_id") == PROMPT_ID and (r.get("variables") or {}).get("word")}
    data = collect(records)
    output = args.output if args.output is not None else DEFAULT_OUTPUT["ridge"]
    output = None if str(output) == "-" else Path(output)
    if not data:
        sys.exit(f"No {PROMPT_ID} records found.")
    render_ridge(data, peaks, output, CAPTION)


if __name__ == "__main__":
    main()
