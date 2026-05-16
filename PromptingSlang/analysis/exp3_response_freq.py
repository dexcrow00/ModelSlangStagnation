#!/usr/bin/env python3
"""Experiment 3 analysis: distribution of words the model produces for each semantic slot.

Two complementary views per slot:

  1. Generated word (from logprobs.tokens, since response is None when logprobs
     are requested): the model's top-1 word at temperature=0, plus its joint
     log-probability.

  2. First-token alternatives (top_logprobs[0]): the model's probability mass
     over the top-5 candidates at the first position, giving a richer view of
     the distribution even without sampling at higher temperature.

Usage:
    python analysis/exp3_response_freq.py data/responses/<exp3_run>.jsonl
    python analysis/exp3_response_freq.py data/responses/*.jsonl --slot exp3_charisma
"""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

SLOT_LABELS = {
    "exp3_cool_impressive":   "Cool / impressive",
    "exp3_mediocre":          "Mediocre / disappointing",
    "exp3_charisma":          "Charisma / social magnetism",
    "exp3_agreement":         "Enthusiastic agreement",
    "exp3_suspicious":        "Suspicious / untrustworthy",
    "exp3_excellent_sensory": "Excellent in music or food",
}

_STOP_TOK = re.compile(r"[.\s!?,\n]|^<")


def _normalize_logprobs(lp: dict) -> tuple[list[str], list[float], list[dict]]:
    """Handle Together-native and OpenAI-compatible logprob formats.

    Returns (tokens, token_logprobs, top_logprobs_list) where each element of
    top_logprobs_list is a dict mapping token_str → logprob_float.

    Leading routing tokens of the form <|special|> [channel_name] <|special|> ...
    (as produced by gpt-oss-120b) are stripped before returning.
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


def _word_from_tokens(logprobs_obj: dict) -> tuple[str, float | None]:
    """Reconstruct the generated word and its joint logprob from logprobs tokens."""
    if not isinstance(logprobs_obj, dict):
        return "", None
    tokens, lps, _ = _normalize_logprobs(logprobs_obj)
    parts, total = [], 0.0
    for tok, lp in zip(tokens, lps):
        if _STOP_TOK.search(tok):
            break
        parts.append(tok)
        total += lp
    word = "".join(parts).strip().strip("\"'").lower()
    return word, (total if parts else None)


def _first_token_alts(logprobs_obj: dict) -> list[tuple[str, float]]:
    """Return (token, probability) pairs from top_logprobs[0], normalised."""
    if not isinstance(logprobs_obj, dict):
        return []
    _, _, top_lps = _normalize_logprobs(logprobs_obj)
    if not top_lps:
        return []
    first = top_lps[0]
    if not isinstance(first, dict):
        return []
    alts = [(tok.strip().lower(), math.exp(float(lp))) for tok, lp in first.items() if tok.strip()]
    alts.sort(key=lambda x: -x[1])
    total = sum(p for _, p in alts)
    return [(tok, p / total) for tok, p in alts] if total else alts


def load_records(paths: list[Path], slot_filter: str | None) -> list[dict]:
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
            try:
                rec, end = decoder.raw_decode(text, pos)
            except json.JSONDecodeError:
                break
            pos = end
            pid = rec.get("prompt_id", "")
            if not pid.startswith("exp3_"):
                continue
            if slot_filter and pid != slot_filter:
                continue
            records.append(rec)
    return records


def _bar(p: float, max_p: float, width: int = 22) -> str:
    filled = round(width * p / max_p) if max_p > 0 else 0
    return "█" * filled + "░" * (width - filled)


def report_slot(slot: str, records: list[dict]) -> None:
    label = SLOT_LABELS.get(slot, slot)
    print(f"\n{'═' * 68}")
    print(f"  Slot : {label}")
    print(f"  ID   : {slot}  ({len(records)} record(s))")
    print(f"{'═' * 68}")

    by_model: dict[str, list[dict]] = defaultdict(list)
    for rec in records:
        by_model[rec.get("model", "unknown")].append(rec)

    for model, recs in sorted(by_model.items()):
        short = model.split("/")[-1]
        print(f"\n  ── {short} ──")

        # Collect generated words and first-token alternatives across phrasings
        generated: list[tuple[str, float | None, str]] = []  # (word, logprob, phrasing)
        alt_agg: dict[str, list[float]] = defaultdict(list)

        for rec in recs:
            lp_obj = rec.get("logprobs") or {}
            word, jlp = _word_from_tokens(lp_obj)
            phrasing = rec.get("prompt_text", "")[:60]
            generated.append((word, jlp, phrasing))
            for tok, p in _first_token_alts(lp_obj):
                alt_agg[tok].append(p)

        # Generated words table
        print(f"\n  Generated words (one per phrasing variant):")
        print(f"  {'Word':<20}  {'LogP':>8}  Phrasing (truncated)")
        print(f"  {'─' * 20}  {'─' * 8}  {'─' * 40}")
        for word, jlp, phrasing in generated:
            lp_str = f"{jlp:.3f}" if jlp is not None else "   ?"
            print(f"  {word:<20}  {lp_str:>8}  {phrasing}")

        # Aggregated first-token alternatives across all phrasings
        if alt_agg:
            avg_alts = sorted(
                ((tok, sum(ps) / len(ps)) for tok, ps in alt_agg.items()),
                key=lambda x: -x[1],
            )
            print(f"\n  Avg first-token probability across {len(recs)} phrasing(s):")
            print(f"  {'Token':<20}  {'Avg P':>7}  Bar")
            print(f"  {'─' * 20}  {'─' * 7}  {'─' * 22}")
            max_p = avg_alts[0][1] if avg_alts else 1
            for tok, avg_p in avg_alts[:10]:
                print(f"  {tok:<20}  {avg_p:>7.4f}  {_bar(avg_p, max_p)}")


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
        print("No exp3 records found.", file=sys.stderr)
        sys.exit(1)

    by_slot: dict[str, list[dict]] = defaultdict(list)
    for rec in records:
        by_slot[rec["prompt_id"]].append(rec)

    print(f"\nExperiment 3 — Semantic Neighbour Probing")
    print(f"  {len(records)} records  ·  {len(by_slot)} slot(s)\n")

    for slot in sorted(by_slot.keys()):
        report_slot(slot, by_slot[slot])

    print()


if __name__ == "__main__":
    main()
