#!/usr/bin/env python3
"""Experiment 2 analysis: extract and compare logprobs for target words across fill-in-blank templates.

Run Experiment 2 with: --max-tokens 1 --temperature 0.0
The echo+logprobs combination returns the model's logprob for every prompt token.
For each (template, word) pair we extract the logprob of the target word token(s)
at the end of the echoed sentence, giving a direct measure of P(word | context).

Usage:
    python analysis/exp2_logprob_compare.py data/responses/<exp2_logprobs_run>.jsonl
    python analysis/exp2_logprob_compare.py data/responses/<exp2_logprobs_run>.jsonl --family adj
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections import defaultdict
from pathlib import Path


def load_records(path: Path, family: str | None) -> list[dict]:
    records = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            rec = json.loads(line)
        except json.JSONDecodeError:
            continue
        if rec.get("logprobs") is None:
            continue
        pid = rec.get("prompt_id", "")
        if family and not pid.endswith(family):
            continue
        records.append(rec)
    return records


def _logprob_of_last_tokens(logprobs_obj: dict, target_word: str) -> float | None:
    """Return the summed logprob of target_word at the END of the echoed token sequence.

    Together's logprobs object has parallel arrays: tokens[] and token_logprobs[].
    With echo=true, these span the full prompt + generated tokens.
    We find the last occurrence of the target word (allowing a leading space token
    like ' lit') and return its logprob. For multi-token words, we sum logprobs.
    """
    if not isinstance(logprobs_obj, dict):
        return None

    tokens: list[str] = logprobs_obj.get("tokens") or []
    token_lps: list[float | None] = logprobs_obj.get("token_logprobs") or []

    if not tokens or not token_lps or len(tokens) != len(token_lps):
        return None

    # Normalise: strip leading space from each token for matching
    normalised = [t.lstrip() for t in tokens]
    target_lower = target_word.lower()

    # Walk backwards to find the last position where the word appears
    # as a single token (most slang words are one token)
    for i in range(len(normalised) - 1, -1, -1):
        if normalised[i].lower() == target_lower:
            lp = token_lps[i]
            return float(lp) if lp is not None else None

    # Multi-token fallback: try to match the word reconstructed from consecutive tokens
    target_tokens = target_word.lower().split()
    n = len(target_tokens)
    for i in range(len(normalised) - n, -1, -1):
        chunk = [normalised[j].lower() for j in range(i, i + n)]
        if chunk == target_tokens:
            lps = [token_lps[j] for j in range(i, i + n) if token_lps[j] is not None]
            return float(sum(lps)) if len(lps) == n else None

    return None


def build_table(records: list[dict]) -> dict[str, dict[str, float]]:
    """Return {prefix: {word: logprob}} for all records."""
    table: dict[str, dict[str, float]] = defaultdict(dict)
    for rec in records:
        variables = rec.get("variables", {})
        word = variables.get("word", "")
        prefix = variables.get("prefix", "")
        if not word or not prefix:
            continue
        lp = _logprob_of_last_tokens(rec["logprobs"], word)
        if lp is not None:
            table[prefix][word] = lp
    return dict(table)


def _prob_str(lp: float) -> str:
    p = math.exp(lp)
    return f"{p:.4f}"


def report(table: dict[str, dict[str, float]]) -> None:
    if not table:
        print("No usable records found — check that logprobs and echo are present.", file=sys.stderr)
        return

    # Collect all words across all prefixes
    all_words: list[str] = sorted({w for words in table.values() for w in words})

    print(f"\n{'─' * 60}")
    print("  Experiment 2 — Logprob comparison (P(word | template context))")
    print(f"  {len(table)} template(s)   {len(all_words)} target word(s)")
    print(f"{'─' * 60}\n")

    for prefix, word_lps in sorted(table.items()):
        print(f"\n  Template: \"{prefix.strip()}___\"")
        print(f"  {'Word':<22} {'LogProb':>9}  {'Prob':>8}  Rank")
        print(f"  {'─' * 22} {'─' * 9}  {'─' * 8}  {'─' * 4}")
        ranked = sorted(word_lps.items(), key=lambda kv: -kv[1])
        for rank, (word, lp) in enumerate(ranked, 1):
            print(f"  {word:<22} {lp:>9.3f}  {_prob_str(lp):>8}  #{rank}")

    # Cross-template: for each word, what is its average logprob?
    print(f"\n\n  {'─' * 60}")
    print("  Average logprob per word (across all templates)")
    print(f"  {'─' * 60}")
    print(f"  {'Word':<22} {'Avg LogProb':>11}  {'Avg Prob':>10}  {'Templates':>10}")
    print(f"  {'─' * 22} {'─' * 11}  {'─' * 10}  {'─' * 10}")

    word_lps_all: dict[str, list[float]] = defaultdict(list)
    for word_lps in table.values():
        for w, lp in word_lps.items():
            word_lps_all[w].append(lp)

    avg_lps = {w: sum(lps) / len(lps) for w, lps in word_lps_all.items()}
    for word, avg_lp in sorted(avg_lps.items(), key=lambda kv: -kv[1]):
        n = len(word_lps_all[word])
        avg_p = math.exp(avg_lp)
        print(f"  {word:<22} {avg_lp:>11.3f}  {avg_p:>10.4f}  {n:>10}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare logprobs across fill-in-blank templates (Experiment 2).")
    parser.add_argument("responses", help="Path to response JSONL file from an exp2 run.")
    parser.add_argument(
        "--family",
        choices=["adj", "verb"],
        help="Filter to a specific template family (adj=exp2_adj_probe, verb=exp2_verb_probe).",
    )
    args = parser.parse_args()

    path = Path(args.responses)
    if not path.exists():
        print(f"File not found: {path}", file=sys.stderr)
        sys.exit(1)

    records = load_records(path, args.family)
    if not records:
        print("No records with logprobs found.", file=sys.stderr)
        sys.exit(1)

    table = build_table(records)
    report(table)


if __name__ == "__main__":
    main()
