#!/usr/bin/env python3
"""Count slang-word usage in the scenario-based-prompting responses, per model.

The slang vocabulary is the union of ``FineWebAnalysis/target_words.txt`` and
``FineWebAnalysis/scenario_words.txt``. Every word/phrase from that vocabulary is
matched (case-insensitive, word-boundary, multi-word/emoji tolerant) in each
model's free-text responses, and the counts are written as JSON.

Usage (from the PromptingSlang root):
    python experiments/scenario_based_prompting/analysis/count_slang_by_model.py
    python experiments/scenario_based_prompting/analysis/count_slang_by_model.py -o counts.json
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]           # PromptingSlang
PROJECT_ROOT = REPO_ROOT.parent                            # CommonCrawlAnalysis
sys.path.insert(0, str(REPO_ROOT))
from src.analysis_utils import load_vocab, word_pattern  # noqa: E402
from src.response_utils import read_responses  # noqa: E402

DEFAULT_RESPONSES = Path(__file__).resolve().parents[1] / "responses"
WORD_FILES = [PROJECT_ROOT / "FineWebAnalysis" / "target_words.txt",
              PROJECT_ROOT / "FineWebAnalysis" / "scenario_words.txt"]
DEFAULT_OUTPUT = Path(__file__).resolve().parents[1] / "slang_counts_by_model.json"


def main() -> None:
    p = argparse.ArgumentParser(description="Count slang-word usage per model in scenario responses.")
    p.add_argument("responses", nargs="?", default=str(DEFAULT_RESPONSES),
                   help=f"Response JSONL file or directory (default: {DEFAULT_RESPONSES}).")
    p.add_argument("-o", "--output", type=Path, default=DEFAULT_OUTPUT,
                   help=f"Output JSON path (default: {DEFAULT_OUTPUT}).")
    args = p.parse_args()

    vocab = load_vocab(WORD_FILES)
    pats = {w: word_pattern(w) for w in vocab}

    records = read_responses(args.responses)
    counts: dict[str, Counter] = defaultdict(Counter)
    n_resp: Counter = Counter()
    for rec in records:
        text = rec.get("response") or ""
        if not text:
            continue
        model = rec.get("model", "unknown")
        n_resp[model] += 1
        for w, pat in pats.items():
            hits = len(pat.findall(text))
            if hits:
                counts[model][w] += hits

    models = {}
    for model in sorted(counts | Counter({m: 0 for m in n_resp})):
        c = counts.get(model, Counter())
        models[model] = {
            "responses": n_resp[model],
            "total_slang_hits": sum(c.values()),
            "words": dict(c.most_common()),  # word -> count, descending
        }

    out = {
        "sources": [str(f.relative_to(PROJECT_ROOT)) for f in WORD_FILES],
        "vocabulary_size": len(vocab),
        "models": models,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Wrote {args.output}")
    for m, d in models.items():
        print(f"  {m.split('/')[-1]:32} {d['total_slang_hits']:>4} hits / {d['responses']} responses")


if __name__ == "__main__":
    main()
