#!/usr/bin/env python3
"""Render logprob response files as a year × word probability heatmap.

For each model found in the file, one figure is produced.
- X-axis (columns): the value of the first variable in each record (e.g. year).
- Y-axis (rows):    unique response words, ordered by first appearance.
- Cell colour:      combined probability of the response word for that column.

Only the cell where a word was actually selected for a given year is coloured;
all other cells in that column are left blank.

A matching text-response record (same prompt_id / model / variables) is used
to identify which tokens form the response word. Without a match, all
non-stop tokens are combined.

Usage
-----
    python visualizer/visualize.py data/responses/single_word_logprobs_<id>.jsonl
    python visualizer/visualize.py data/responses/single_word_logprobs_<id>.jsonl -o out.png
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

def _load_records(path: Path) -> list[dict]:
    """Parse a pretty-printed JSONL file into a list of dicts."""
    decoder = json.JSONDecoder()
    text = path.read_text(encoding="utf-8")
    records: list[dict] = []
    pos = 0
    while pos < len(text):
        while pos < len(text) and text[pos] in " \t\n\r":
            pos += 1
        if pos >= len(text):
            break
        obj, end = decoder.raw_decode(text, pos)
        records.append(obj)
        pos = end
    return records


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _is_stop_token(token: str) -> bool:
    """Return True for special control/stop tokens (e.g. <|eot_id|>)."""
    return token.startswith("<|") and token.endswith("|>")


def _col_label(record: dict) -> str:
    """X-axis label for a record: value of the first variable, or prompt_id."""
    variables = record.get("variables") or {}
    if variables:
        return str(next(iter(variables.values())))
    return record.get("prompt_id", "")


def _find_text_response(lp_record: dict, all_records: list[dict]) -> str | None:
    """Return the text response matching *lp_record* by prompt_id/model/variables."""
    for r in all_records:
        if (r.get("response") is not None
                and r.get("prompt_id") == lp_record.get("prompt_id")
                and r.get("model") == lp_record.get("model")
                and r.get("variables") == lp_record.get("variables")):
            return r["response"].strip()
    return None


def _combined_prob(
    tokens: list[str],
    token_logprobs: list[float],
    response_text: str,
) -> tuple[str, float]:
    """Find the tokens that form *response_text* and return (word, combined_prob).

    Searches for a contiguous token span whose stripped concatenation matches
    the response text (case-insensitive). Falls back to all tokens if no span
    is found. ``combined_prob`` is the product of per-token probabilities,
    i.e. exp(sum of log-probabilities).
    """
    target_nospace = response_text.strip().replace(" ", "")
    n = len(tokens)

    best_lps: list[float] = []
    for start in range(n):
        for end in range(start + 1, n + 1):
            candidate = "".join(t.strip() for t in tokens[start:end])
            if candidate.lower() == target_nospace.lower():
                best_lps = token_logprobs[start:end]
                break
        if best_lps:
            break

    if not best_lps:
        # Fallback: use all non-stop tokens
        best_lps = token_logprobs

    return response_text.strip(), math.exp(sum(best_lps))


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------

def render(records: list[dict], output: Path | None = None) -> None:
    """Build and display (or save) a year × word heatmap for *records*.

    One figure is produced per model. Records without logprobs data are ignored.

    Layout
    ------
    - Columns (x-axis): one per record, labelled by the first variable value
      (e.g. the year).
    - Rows (y-axis): unique response words ordered by first appearance.
    - Cell colour: combined probability where that word was the model's
      response; blank elsewhere.
    """
    lp_records = [r for r in records if (r.get("logprobs") or {}).get("top_logprobs")]
    if not lp_records:
        n_text = sum(1 for r in records if r.get("response"))
        print(
            f"No records with logprobs data found "
            f"({len(records)} total record(s), {n_text} text response(s) ignored).",
            file=sys.stderr,
        )
        sys.exit(1)

    # Group records by model so each model gets its own figure.
    models = list(dict.fromkeys(r["model"] for r in lp_records))

    for model_idx, model in enumerate(models):
        model_records = [r for r in lp_records if r["model"] == model]

        # Build one (col_label, word, prob) entry per record.
        entries: list[tuple[str, str, float]] = []
        for record in model_records:
            lp = record["logprobs"]

            pairs = [
                (tok, lp_val)
                for tok, lp_val in zip(lp["tokens"], lp["token_logprobs"])
                if not _is_stop_token(tok)
            ]
            if not pairs:
                continue

            filtered_tokens, filtered_lps = zip(*pairs)

            response_text = _find_text_response(record, records)
            if response_text:
                word, prob = _combined_prob(list(filtered_tokens), list(filtered_lps), response_text)
            else:
                word = "".join(t.strip() for t in filtered_tokens)
                prob = math.exp(sum(filtered_lps))

            entries.append((_col_label(record), word, prob))

        if not entries:
            continue

        # Build ordered unique axes, preserving first-appearance order.
        unique_years = list(dict.fromkeys(lbl for lbl, _, _ in entries))
        unique_words = list(dict.fromkeys(w   for _, w, _ in entries))

        year_idx = {y: i for i, y in enumerate(unique_years)}
        word_idx = {w: i for i, w in enumerate(unique_words)}

        n_rows = len(unique_years)
        n_cols = len(unique_words)

        # Fill matrix; cells with no matching entry stay NaN.
        prob_matrix = np.full((n_rows, n_cols), np.nan)
        for label, word, prob in entries:
            prob_matrix[year_idx[label], word_idx[word]] = prob

        # --- figure layout ------------------------------------------------
        CELL_W, CELL_H = 2.2, 0.55
        fig_w = n_cols * CELL_W + 2.0
        fig_h = n_rows * CELL_H + 1.0

        fig, ax = plt.subplots(figsize=(fig_w, fig_h))

        norm = plt.Normalize(vmin=0, vmax=1)
        im = ax.imshow(
            prob_matrix,
            aspect="auto",
            cmap="RdYlBu_r",
            norm=norm,
            interpolation="nearest",
        )

        # --- cell annotations ---------------------------------------------
        cmap_obj = plt.cm.RdYlBu_r
        for label, word, prob in entries:
            row = year_idx[label]
            col = word_idx[word]
            r, g, b, _ = cmap_obj(norm(prob))
            text_color = "black" if (0.299 * r + 0.587 * g + 0.114 * b) > 0.45 else "white"
            ax.text(
                col, row,
                f"{prob:.3f}",
                ha="center", va="center",
                fontsize=8,
                color=text_color,
            )

        # --- axes ---------------------------------------------------------
        ax.set_xticks(range(n_cols))
        ax.set_xticklabels(unique_words, fontsize=8)
        ax.set_xlabel("word", fontsize=8)

        ax.set_yticks(range(n_rows))
        ax.set_yticklabels(unique_years, fontsize=8)
        ax.set_ylabel(
            next(iter((model_records[0].get("variables") or {}).keys()), ""),
            fontsize=8,
        )

        ax.set_title(model, fontsize=8, pad=6)

        # --- colorbar -----------------------------------------------------
        cbar = fig.colorbar(im, ax=ax, shrink=0.5, pad=0.02)
        cbar.set_label("probability", fontsize=7)
        cbar.ax.tick_params(labelsize=7)

        fig.tight_layout()

        if output:
            out_path = (
                output.with_name(f"{output.stem}_{model_idx}{output.suffix or '.png'}")
                if len(models) > 1
                else output
            )
            fig.savefig(out_path, dpi=200, bbox_inches="tight")
            print(f"Saved: {out_path}")
        else:
            plt.show()

        plt.close(fig)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Render a logprob JSONL response file as a word-probability chart."
    )
    parser.add_argument(
        "input",
        help="Path to a logprobs response JSONL file (from data/responses/).",
    )
    parser.add_argument(
        "--output", "-o",
        default=None,
        help="Save the figure to this path (PNG/PDF/SVG) instead of displaying it.",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    records = _load_records(Path(args.input))
    render(records, Path(args.output) if args.output else None)


if __name__ == "__main__":
    main()
