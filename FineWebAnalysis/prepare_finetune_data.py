#!/usr/bin/env python3
"""
prepare_finetune_data.py — Build an annotation-ready fine-tuning set for the
RoBERTa slang-sense classifier from extracted word-context CSVs.

Streams the per-crawl context CSVs in a contexts directory (default
``10BT_scenario_merged``; columns ``target, uri, target_context``), reservoir-
samples up to ``--per-word`` cleaned, de-duplicated contexts for each target
word, and writes one CSV per word with the columns ``finetune_roberta.py``
expects --- plus a blank ``is_slang`` column for a human (or LLM) to fill:

    target, is_slang, target_context, uri

TODO(dexcrow): Is this true? I guess we'll find out
Every word's occurrences are a natural mix of the slang sense and the standard
sense (e.g. ``extra`` = dramatic vs. additional), so labelling the ``is_slang``
column (1 = slang sense, 0 = not) yields a balanced binary training set. Once
labelled, point ``finetune_roberta.py --annotations <out-dir>`` at the result.

Words in ``slang_constants.ALWAYS_SLANG`` have no standard sense and are always
passed by ``roberta_filter.py``, so they need no classifier training and are
excluded by default (pass them explicitly via ``--words`` to override).

Usage:
    python prepare_finetune_data.py
    python prepare_finetune_data.py --contexts 10BT_scenario_merged \\
        --out scenario_finetune_annotations --per-word 150
    python prepare_finetune_data.py --words extra fresh goals woke
"""

from __future__ import annotations

import argparse
import csv
import logging
import random
import re
import sys
from collections import defaultdict
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s",
                    handlers=[logging.StreamHandler(sys.stderr)])
log = logging.getLogger(__name__)

HERE = Path(__file__).resolve().parent
DEFAULT_CONTEXTS = HERE / "10BT_scenario_merged"
DEFAULT_OUT = HERE / "scenario_finetune_annotations"
MIN_CHARS, MAX_CHARS = 40, 600  # drop fragments and over-long boilerplate

# Shared with roberta_filter.py: words with no standard sense are pre-labelled.
sys.path.insert(0, str(HERE))
from slang_constants import ALWAYS_SLANG  # noqa: E402


def _slug(word: str) -> str:
    """Filesystem-safe stem for a target word (handles spaces, emoji, etc.)."""
    s = re.sub(r"\s+", "_", word.strip())
    s = re.sub(r"[^A-Za-z0-9_]+", "", s)
    return s or f"u{hex(ord(word[0]))[2:]}" if word else "word"


def _clean(context: str) -> str:
    """Collapse whitespace/newlines to a single readable line."""
    return " ".join(context.split())


def load_excluded_contexts(paths: list[Path]) -> set[str]:
    """Cleaned, lower-cased ``target_context`` values from existing CSVs/dirs.

    Used to keep a held-out set (e.g. validation) disjoint from an already-built
    set (e.g. the training annotations) so metrics computed on it are unbiased.
    """
    csvs: list[Path] = []
    for p in paths:
        csvs.extend(sorted(p.glob("*.csv")) if p.is_dir() else [p])
    excluded: set[str] = set()
    for c in csvs:
        with c.open(newline="", encoding="utf-8-sig") as fh:
            for row in csv.DictReader(fh):
                ctx = _clean(row.get("target_context") or "")
                if ctx:
                    excluded.add(ctx.lower())
    log.info("Loaded %d context(s) to exclude from %d file(s).", len(excluded), len(csvs))
    return excluded


def sample_contexts(contexts_dir: Path, words: set[str] | None, per_word: int,
                    seed: int, exclude: set[str] | None = None) -> dict[str, list[tuple[str, str]]]:
    """Reservoir-sample up to ~4x per_word cleaned contexts per target word.

    Oversampling leaves room to de-duplicate down to ``per_word`` afterwards.
    Contexts whose cleaned/lower-cased text is in ``exclude`` are skipped, so a
    validation set can be drawn disjoint from the training set.
    Returns {word: [(uri, context), ...]}.
    """
    rng = random.Random(seed)
    over = max(per_word * 4, 400)
    reservoir: dict[str, list[tuple[str, str]]] = defaultdict(list)
    seen: dict[str, int] = defaultdict(int)

    files = sorted(contexts_dir.glob("*.csv"))
    if not files:
        sys.exit(f"No context CSVs found in {contexts_dir}.")
    for path in files:
        with path.open(newline="", encoding="utf-8-sig") as fh:
            for row in csv.DictReader(fh):
                w = (row.get("target") or "").strip()
                if not w or (words is not None and w not in words):
                    continue
                # Always-slang words are handled by the filter allowlist and need
                # no training data; skip them unless explicitly requested.
                if words is None and w.casefold() in ALWAYS_SLANG:
                    continue
                ctx = _clean(row.get("target_context") or "")
                if not (MIN_CHARS <= len(ctx) <= MAX_CHARS):
                    continue
                if exclude and ctx.lower() in exclude:
                    continue
                seen[w] += 1
                res = reservoir[w]
                rec = (row.get("uri") or "", ctx)
                if len(res) < over:
                    res.append(rec)
                else:
                    j = rng.randint(0, seen[w] - 1)
                    if j < over:
                        res[j] = rec
    log.info("Streamed %d context file(s); %d target word(s) sampled.",
             len(files), len(reservoir))
    return reservoir


def write_annotations(reservoir: dict[str, list[tuple[str, str]]], out_dir: Path,
                      per_word: int, seed: int) -> list[tuple[str, int]]:
    """De-duplicate, cap at per_word, and write one annotation CSV per word."""
    rng = random.Random(seed)
    out_dir.mkdir(parents=True, exist_ok=True)
    summary: list[tuple[str, int]] = []
    for word, recs in sorted(reservoir.items(), key=lambda kv: -len(kv[1])):
        rng.shuffle(recs)
        seen_ctx: set[str] = set()
        rows: list[tuple[str, str]] = []
        for uri, ctx in recs:
            key = ctx.lower()
            if key in seen_ctx:
                continue
            seen_ctx.add(key)
            rows.append((uri, ctx))
            if len(rows) >= per_word:
                break
        path = out_dir / f"{_slug(word)}_annotations.csv"
        with path.open("w", newline="", encoding="utf-8") as fh:
            writer = csv.writer(fh)
            writer.writerow(["target", "is_slang", "target_context", "uri"])
            for uri, ctx in rows:
                writer.writerow([word, "", ctx, uri])
        summary.append((word, len(rows)))
    return summary


def write_readme(out_dir: Path, summary: list[tuple[str, int]], per_word: int,
                 excluded_from: list[str] | None = None) -> None:
    total = sum(n for _, n in summary)
    if excluded_from:
        lines = [
            "# Scenario-word held-out evaluation set (UNLABELED)",
            "",
            "Held-out contexts for evaluating the RoBERTa slang-sense classifier on the",
            "new scenario words, sampled from `10BT_scenario_merged` by",
            "`prepare_finetune_data.py` and kept **disjoint** from the training set",
            f"({', '.join(excluded_from)}). One CSV per target word; columns:",
            "`target, is_slang, target_context, uri`.",
            "",
            "## To use",
            "1. Fill the `is_slang` column: `1` if the target word is used in its slang",
            "   sense in that context, `0` if it is the ordinary sense. Leave blank to skip.",
            "2. Score with the fine-tuned model and compute metrics:",
            "   `python roberta_filter.py <this-dir> --score-all \\",
            "       --output-dir scored_eval/ --roberta-model-dir ./ft_model_roberta_scenario/`",
            "   then treat `roberta_score >= 0` as predicted-slang and compare against",
            "   `is_slang` for precision / recall / F1 (per word and overall).",
        ]
    else:
        lines = [
            "# Scenario-word fine-tuning annotations (UNLABELED)",
            "",
            "Annotation-ready contexts for the RoBERTa slang-sense classifier, sampled",
            "from `10BT_scenario_merged` by `prepare_finetune_data.py`. One CSV per",
            "target word; columns: `target, is_slang, target_context, uri`.",
            "",
            "## To use",
            "1. Fill the `is_slang` column in each CSV: `1` if the target word is used",
            "   in its slang sense in that context, `0` if it is the ordinary sense",
            "   (e.g. `extra` = dramatic -> 1, `extra` = additional -> 0). Leave a row",
            "   blank to skip it.",
            "2. Fine-tune:",
            "   `python finetune_roberta.py --annotations scenario_finetune_annotations \\",
            "       --model-dir ./ft_model_roberta_scenario/`",
            "",
            "Words with no standard sense (`slang_constants.ALWAYS_SLANG`: "
            f"{', '.join(sorted(ALWAYS_SLANG))}) are **excluded** --- `roberta_filter.py` "
            "always passes them, so they need no classifier training.",
        ]
    lines += [
        "",
        f"## Contents ({len(summary)} words, {total} contexts, up to {per_word} each)",
        "",
        "| word | contexts |",
        "| --- | --- |",
    ]
    lines += [f"| {w} | {n} |" for w, n in summary]
    (out_dir / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    p = argparse.ArgumentParser(description="Prepare annotation-ready RoBERTa fine-tuning data.")
    p.add_argument("--contexts", type=Path, default=DEFAULT_CONTEXTS,
                   help=f"Directory of word-context CSVs (default: {DEFAULT_CONTEXTS.name}).")
    p.add_argument("--out", type=Path, default=DEFAULT_OUT,
                   help=f"Output directory for annotation CSVs (default: {DEFAULT_OUT.name}).")
    p.add_argument("--per-word", type=int, default=120, dest="per_word",
                   help="Max contexts to keep per word (default: 120).")
    p.add_argument("--words", nargs="+", default=None,
                   help="Restrict to these target words (default: all in the contexts dir).")
    p.add_argument("--exclude", nargs="+", type=Path, default=None, metavar="PATH",
                   help="CSV file(s) or dir(s) whose contexts to exclude, e.g. the training "
                        "set when building a disjoint validation set.")
    p.add_argument("--seed", type=int, default=0, help="Sampling seed (default: 0).")
    args = p.parse_args()

    words = set(args.words) if args.words else None
    exclude = load_excluded_contexts(args.exclude) if args.exclude else None
    reservoir = sample_contexts(args.contexts, words, args.per_word, args.seed, exclude)
    summary = write_annotations(reservoir, args.out, args.per_word, args.seed)
    write_readme(args.out, summary, args.per_word,
                 excluded_from=[str(p) for p in args.exclude] if args.exclude else None)

    total = sum(n for _, n in summary)
    log.info("Wrote %d annotation CSV(s), %d contexts total, to %s",
             len(summary), total, args.out)
    for w, n in summary:
        print(f"  {w:18} {n:>4}")


if __name__ == "__main__":
    main()
