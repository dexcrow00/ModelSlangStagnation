#!/usr/bin/env python3
"""Experiment 2 analysis: compare the model's predicted next-word distribution
across fill-in-blank syntactic frames.

Each template is an incomplete sentence (e.g. "That was absolutely").
With max-tokens > 1, the model generates a full word or short phrase. We report:
  - The actual generated word (from response text) and its joint log-probability
    (product of per-token logprobs up to the first stop character).
  - Top-5 first-token alternatives (top_logprobs[0]), resolved to likely full
    words where the generated sequence provides enough context.

Usage (run from the PromptingSlang root):
    python experiments/2_logprob_with_masking/analysis/exp2_logprob_compare.py data/responses/<exp2_run>.jsonl
    python experiments/2_logprob_with_masking/analysis/exp2_logprob_compare.py data/responses/<exp2_run>.jsonl --family adj
    python experiments/2_logprob_with_masking/analysis/exp2_logprob_compare.py data/responses/<exp2_run>.jsonl --slang-only
"""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
from collections import defaultdict
from pathlib import Path

SLANG_ERAS: dict[str, str] = {
    "sick": "pre-2012", "dope": "pre-2012", "rad": "pre-2012",
    "wicked": "pre-2012", "fresh": "pre-2012", "gnarly": "pre-2012",
    "epic": "pre-2012", "legit": "pre-2012", "stoked": "pre-2012",
    "fleek": "2012–2015", "ratchet": "2012–2015", "basic": "2012–2015",
    "savage": "2012–2015", "extra": "2012–2015", "salty": "2012–2015",
    "lit": "2016–2019", "fire": "2016–2019", "slay": "2016–2019",
    "iconic": "2016–2019", "lowkey": "2016–2019", "highkey": "2016–2019",
    "slaps": "2016–2019", "vibes": "2016–2019", "vibe": "2016–2019",
    "bussin": "2020–2022", "based": "2020–2022", "goated": "2020–2022",
    "mid": "2020–2022", "valid": "2020–2022",
    "hits": "2020–2022", "cooks": "2020–2022", "cooked": "2020–2022",
    "rizz": "2023–2024", "snatched": "2023–2024", "ate": "2023–2024",
    "delulu": "2023–2024", "brat": "2023–2024",
}

_STOP_TOK = re.compile(r"[.\s!?,\n]|^<")  # whitespace/punct or EOS token like <|eot_id|>


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


def _word_and_logprob(logprobs_obj: dict) -> tuple[str, float | None]:
    """Reconstruct the generated word and its joint logprob from logprobs tokens."""
    if not isinstance(logprobs_obj, dict):
        return "", None
    tokens, lps, _ = _normalize_logprobs(logprobs_obj)
    if not tokens:
        return "", None

    word_parts, total_lp = [], 0.0
    for tok, lp in zip(tokens, lps):
        if _STOP_TOK.search(tok):
            break
        word_parts.append(tok)
        total_lp += lp

    word = "".join(word_parts).strip().strip("\"'").lower()
    return word, total_lp if word_parts else None


def _first_token_alts(logprobs_obj: dict) -> list[tuple[str, float]]:
    """Return top alternatives at position 0 as (token, prob) pairs."""
    if not isinstance(logprobs_obj, dict):
        return []
    _, _, top_lps = _normalize_logprobs(logprobs_obj)
    if not top_lps:
        return []
    first = top_lps[0]
    if not isinstance(first, dict):
        return []
    alts = [(tok.strip(), math.exp(float(lp))) for tok, lp in first.items() if tok.strip()]
    alts.sort(key=lambda x: -x[1])
    total = sum(p for _, p in alts)
    return [(tok, p / total) for tok, p in alts]


def load_records(path: Path, family: str | None) -> list[dict]:
    decoder = json.JSONDecoder()
    records = []
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
        if rec.get("logprobs") is None:
            continue
        pid = rec.get("prompt_id", "")
        if family and not pid.endswith(family):
            continue
        records.append(rec)
    return records


def _era(word: str) -> str:
    return SLANG_ERAS.get(word.lower().rstrip("s"), SLANG_ERAS.get(word.lower(), ""))


def _bar(p: float, max_p: float, width: int = 20) -> str:
    filled = round(width * p / max_p) if max_p > 0 else 0
    return "█" * filled + "░" * (width - filled)


def report(records: list[dict], slang_only: bool) -> None:
    if not records:
        print("No usable records.", file=sys.stderr)
        return

    # Group by model → prefix
    by_model: dict[str, dict[str, list[dict]]] = defaultdict(lambda: defaultdict(list))
    for rec in records:
        prefix = rec.get("variables", {}).get("prefix", rec.get("prompt_text", ""))
        by_model[rec.get("model", "unknown")][prefix].append(rec)

    # Track generated words for cross-template summary
    word_probs: dict[str, list[float]] = defaultdict(list)

    for model, prefixes in sorted(by_model.items()):
        short = model.split("/")[-1]
        print(f"\n{'═' * 68}")
        print(f"  Model: {short}")
        print(f"{'═' * 68}")

        for prefix, recs in sorted(prefixes.items()):
            print(f"\n  Template: \"{prefix} ___\"")

            for rec in recs:
                lp_obj = rec.get("logprobs", {})
                word, jlp = _word_and_logprob(lp_obj)
                alts = _first_token_alts(lp_obj)

                if not word and not alts:
                    continue

                era_tag = f"[{_era(word)}]" if word and _era(word) else ""
                prob_str = f"P={math.exp(jlp):.4f}" if jlp is not None else "P=?"

                print(f"    Generated: \"{word}\"  {prob_str}  {era_tag}")

                if word:
                    if jlp is not None:
                        word_probs[word].append(math.exp(jlp))

                if not slang_only and alts:
                    max_p = alts[0][1] if alts else 1
                    print(f"    {'Alt (tok)':<18}  {'P':>7}  {'Era':<12}  Bar")
                    print(f"    {'─' * 18}  {'─' * 7}  {'─' * 12}  {'─' * 20}")
                    for tok, p in alts:
                        era_label = f"[{_era(tok)}]" if _era(tok) else ""
                        print(f"    {tok:<18}  {p:>7.4f}  {era_label:<12}  {_bar(p, max_p)}")

    # Cross-template summary
    print(f"\n\n{'═' * 68}")
    print("  Summary: generated words ranked by avg probability across templates")
    print(f"{'═' * 68}")
    print(f"  {'Word':<22}  {'Avg P':>8}  {'Era':<12}  {'N frames':>8}")
    print(f"  {'─' * 22}  {'─' * 8}  {'─' * 12}  {'─' * 8}")

    rows = sorted(word_probs.items(), key=lambda kv: -(sum(kv[1]) / len(kv[1])))
    if slang_only:
        rows = [(w, ps) for w, ps in rows if _era(w)]

    for word, probs in rows:
        avg = sum(probs) / len(probs)
        print(f"  {word:<22}  {avg:>8.4f}  [{_era(word)}]  {len(probs):>8}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyse Experiment 2 full-word completions.")
    parser.add_argument("responses", help="Path to response JSONL file from an exp2 run.")
    parser.add_argument("--family", choices=["adj", "verb"],
                        help="Filter to adj or verb template family.")
    parser.add_argument("--slang-only", action="store_true",
                        help="Summary shows only known slang terms; hides first-token alt table.")
    args = parser.parse_args()

    path = Path(args.responses)
    if not path.exists():
        print(f"File not found: {path}", file=sys.stderr)
        sys.exit(1)

    records = load_records(path, args.family)
    if not records:
        print("No records with logprobs found.", file=sys.stderr)
        sys.exit(1)

    report(records, slang_only=args.slang_only)
    print()


if __name__ == "__main__":
    main()
