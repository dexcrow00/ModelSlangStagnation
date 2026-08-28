#!/usr/bin/env python3
"""Experiment 1 analysis: count target slang word frequencies in free-generated responses.

Usage (run from the PromptingSlang root):
    python experiments/1_scenario_based_prompting/analysis/exp1_word_freq.py data/responses/<exp1_run>.jsonl
    python experiments/1_scenario_based_prompting/analysis/exp1_word_freq.py data/responses/*.jsonl --model meta-llama/...
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

# Target words grouped by approximate era of peak usage.
# Single-word and hyphenated/compound forms are included; phrase detection
# uses a simple regex boundary check.
# TODO(dexcrow): Update this with actual experimental buckets.
ERA_WORDS: dict[str, list[str]] = {
    "pre-2012 (older)": [
        "sick", "dope", "rad", "gnarly", "fresh", "wicked", "banging", "epic",
        "legit", "stoked", "pumped", "tight", "fly",
    ],
    "2012–2015": [
        "yolo", "swag", "bae", "basic", "goals", "squad", "fleek", "on fleek",
        "turnt", "ratchet", "thirsty", "salty", "savage", "extra", "omg",
    ],
    "2016–2019": [
        "lit", "fire", "slay", "vibe", "lowkey", "highkey", "yeet", "fam",
        "bruh", "sis", "woke", "cancelled", "shade", "throwing shade",
        "no cap", "cap", "clout", "ghosting",
    ],
    "2020–2022": [
        "bussin", "based", "mid", "sus", "hits different", "understood the assignment",
        "main character", "rent free", "gatekeep", "gaslight", "girlboss",
        "situationship", "cheugy", "vibing", "lmao", "lmaooo"
    ],
    "2023–2024": [
        "rizz", "delulu", "it's giving", "snatched", "ate", "left no crumbs",
        "brat", "demure", "very demure", "NPC", "roman empire", "era",
        "in my era", "💀",
    ],
}

# Flat list preserving order for reporting
ALL_WORDS: list[str] = [w for words in ERA_WORDS.values() for w in words]


def _pattern(word: str) -> re.Pattern:
    escaped = re.escape(word)
    return re.compile(rf"(?<![a-z]){escaped}(?![a-z])", re.IGNORECASE)


_PATTERNS: dict[str, re.Pattern] = {w: _pattern(w) for w in ALL_WORDS}


def count_in_text(text: str) -> Counter:
    counts: Counter = Counter()
    for word, pat in _PATTERNS.items():
        hits = len(pat.findall(text))
        if hits:
            counts[word] += hits
    return counts


def load_records(paths: list[Path], model_filter: str | None) -> list[dict]:
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
            if model_filter and rec.get("model", "") != model_filter:
                continue
            if rec.get("response"):
                records.append(rec)
    return records


def _bar(count: int, total: int, width: int = 30) -> str:
    filled = round(width * count / max(total, 1))
    return "█" * filled + "░" * (width - filled)


def report(records: list[dict], top_n: int = 30) -> None:
    n = len(records)
    print(f"\n{'─' * 60}")
    print(f"  Responses analysed : {n}")
    print(f"{'─' * 60}\n")

    total_counts: Counter = Counter()
    prompt_counts: dict[str, Counter] = defaultdict(Counter)

    for rec in records:
        text = rec["response"]
        pid = rec.get("prompt_id", "unknown")
        c = count_in_text(text)
        total_counts.update(c)
        prompt_counts[pid].update(c)

    total_tokens = sum(total_counts.values())
    print(f"  Total slang hits   : {total_tokens}  ({total_tokens / max(n, 1):.2f} per response)\n")

    print(f"  {'Word':<28} {'Count':>6}  {'Per resp':>8}  Distribution")
    print(f"  {'─' * 28} {'─' * 6}  {'─' * 8}  {'─' * 30}")

    for era, words in ERA_WORDS.items():
        era_total = sum(total_counts[w] for w in words)
        if era_total == 0:
            continue
        print(f"\n  ── {era} ──")
        for word in sorted(words, key=lambda w: -total_counts[w]):
            c = total_counts[word]
            if c == 0:
                continue
            rate = c / n
            print(f"  {word:<28} {c:>6}  {rate:>8.3f}  {_bar(c, max(total_counts.values()))}")

    if len(prompt_counts) > 1:
        print(f"\n\n  {'─' * 60}")
        print("  Breakdown by prompt_id (top 5 words each)\n")
        for pid, counts in sorted(prompt_counts.items()):
            top = counts.most_common(5)
            top_str = ", ".join(f"{w}={c}" for w, c in top)
            print(f"  {pid:<30} {top_str}")

    # Log-odds gap: which words appear at a rate meaningfully above chance?
    print(f"\n\n  Top {top_n} words by raw count")
    print(f"  {'─' * 50}")
    for word, c in total_counts.most_common(top_n):
        era = next((e for e, ws in ERA_WORDS.items() if word in ws), "?")
        print(f"  {word:<28} {c:>5}  [{era}]")


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyse slang word frequency in Experiment 1 responses.")
    parser.add_argument("responses", nargs="+", help="Path(s) to response JSONL file(s).")
    parser.add_argument("--model", help="Filter to a specific model ID.")
    parser.add_argument("--top", type=int, default=30, help="Number of top words to list (default 30).")
    args = parser.parse_args()

    paths = [Path(p) for p in args.responses]
    missing = [p for p in paths if not p.exists()]
    if missing:
        print(f"File not found: {missing}", file=sys.stderr)
        sys.exit(1)

    records = load_records(paths, args.model)
    if not records:
        print("No usable response records found.", file=sys.stderr)
        sys.exit(1)

    report(records, top_n=args.top)


if __name__ == "__main__":
    main()
