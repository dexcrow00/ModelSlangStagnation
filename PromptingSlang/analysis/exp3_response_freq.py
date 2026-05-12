#!/usr/bin/env python3
"""Experiment 3 analysis: distribution of words the model produces for each semantic slot.

Two complementary views are reported:

  1. Top-logprob distribution (from logprobs.top_logprobs[0]):
     The model's probability mass over first-token candidates at temperature=0.
     This is the cleaner, statistically sound measure — no sampling needed.

  2. Response frequency (from the 'response' field):
     If you ran at temperature > 0 with multiple runs, this tallies the actual
     sampled words. Useful for checking whether logprob rank predicts sampling
     frequency.

Usage:
    python analysis/exp3_response_freq.py data/responses/<exp3_run>.jsonl
    python analysis/exp3_response_freq.py data/responses/*.jsonl --slot exp3_charisma
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path


def load_records(paths: list[Path], slot_filter: str | None) -> list[dict]:
    records = []
    for path in paths:
        for line in path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            pid = rec.get("prompt_id", "")
            if not pid.startswith("exp3_"):
                continue
            if slot_filter and pid != slot_filter:
                continue
            records.append(rec)
    return records


def _first_token_dist(logprobs_obj: dict) -> dict[str, float] | None:
    """Extract probability distribution over first generated token.

    Together returns top_logprobs as a list (one entry per position).
    Each entry is a dict mapping token string → logprob.
    We take position 0 — the first generated token.
    """
    if not isinstance(logprobs_obj, dict):
        return None
    top_lps = logprobs_obj.get("top_logprobs")
    if not top_lps or not isinstance(top_lps, list) or len(top_lps) == 0:
        return None
    first = top_lps[0]
    if not isinstance(first, dict):
        return None
    # Convert logprobs to probabilities; strip leading whitespace from tokens
    dist = {tok.strip(): math.exp(lp) for tok, lp in first.items() if tok.strip()}
    total = sum(dist.values())
    # Normalise (top-K renormalisation)
    return {tok: p / total for tok, p in sorted(dist.items(), key=lambda kv: -kv[1])}


def _clean_response(text: str | None) -> str:
    if not text:
        return ""
    # Take only the first word, lowercase
    return text.strip().split()[0].lower().strip(".,!?\"'") if text.strip() else ""


def report_slot(slot: str, records: list[dict]) -> None:
    print(f"\n{'═' * 60}")
    print(f"  Slot: {slot}  ({len(records)} record(s))")
    print(f"{'═' * 60}")

    # --- View 1: logprob-based distribution ---
    aggregated: dict[str, list[float]] = defaultdict(list)
    for rec in records:
        lp_obj = rec.get("logprobs")
        if lp_obj is None:
            continue
        dist = _first_token_dist(lp_obj)
        if dist is None:
            continue
        for tok, prob in dist.items():
            aggregated[tok].append(prob)

    if aggregated:
        # Average probability across all phrasings (robustness check)
        n_phrasings = len({rec.get("prompt_text") for rec in records})
        avg_dist = {tok: sum(probs) / len(probs) for tok, probs in aggregated.items()}
        print(f"\n  [Logprob distribution — avg over {n_phrasings} phrasing(s)]")
        print(f"  {'Token':<22}  {'Avg P':>8}  {'# phrasings':>12}  Bar")
        print(f"  {'─' * 22}  {'─' * 8}  {'─' * 12}  {'─' * 20}")
        top = sorted(avg_dist.items(), key=lambda kv: -kv[1])[:25]
        max_p = top[0][1] if top else 1
        for tok, p in top:
            bar = "█" * round(20 * p / max_p)
            n_present = len(aggregated[tok])
            print(f"  {tok:<22}  {p:>8.4f}  {n_present:>12}  {bar}")

    # --- View 2: sampled response frequency ---
    response_counts: Counter = Counter()
    for rec in records:
        word = _clean_response(rec.get("response"))
        if word:
            response_counts[word] += 1

    if response_counts:
        total = sum(response_counts.values())
        print(f"\n  [Response frequency — {total} sampled response(s)]")
        print(f"  {'Word':<22}  {'Count':>6}  {'Freq':>8}")
        print(f"  {'─' * 22}  {'─' * 6}  {'─' * 8}")
        for word, count in response_counts.most_common(20):
            print(f"  {word:<22}  {count:>6}  {count / total:>8.3f}")

    if not aggregated and not response_counts:
        print("  (no usable data)")


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyse Experiment 3 semantic neighbour responses.")
    parser.add_argument("responses", nargs="+", help="Path(s) to response JSONL file(s).")
    parser.add_argument("--slot", help="Filter to a specific semantic slot (e.g. exp3_charisma).")
    args = parser.parse_args()

    paths = [Path(p) for p in args.responses]
    missing = [p for p in paths if not p.exists()]
    if missing:
        print(f"File not found: {missing}", file=sys.stderr)
        sys.exit(1)

    records = load_records(paths, args.slot)
    if not records:
        print("No exp3 records found — check prompt_ids start with 'exp3_'.", file=sys.stderr)
        sys.exit(1)

    # Group by slot
    by_slot: dict[str, list[dict]] = defaultdict(list)
    for rec in records:
        by_slot[rec["prompt_id"]].append(rec)

    print(f"\nExperiment 3 — Semantic Neighbour Probing")
    print(f"  {len(records)} records across {len(by_slot)} slot(s)\n")

    for slot in sorted(by_slot.keys()):
        report_slot(slot, by_slot[slot])

    print()


if __name__ == "__main__":
    main()
