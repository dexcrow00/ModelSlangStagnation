#!/usr/bin/env python3
"""Semantic-slot-queries visualizer.

Charts how reliably each model recovers the *expected* slang term from a
definition-only prompt. The target-word prompts (``exp3_<word>``) describe a
slang term by its meaning and ask for the word; the target for each is the word
itself (e.g. ``exp3_red_pill`` -> "red pill").

The chart buckets the x-axis by target word in ``WORDS_CHRONO_ORDER`` and draws
one bar per model in each bucket, giving the **percentage of that word's
responses where the model answered with the expected target word** on the
y-axis. Matching is stem-based: responses are case-folded, stripped of
spaces/hyphens/punctuation, and reduced by a light suffix stemmer, so
inflections count (``gaslighting``->gaslight, ``slayed``->slay,
``redpilled``->red pill, and ``vibe``/``vibes`` match each other). A response
that contains the target as a run of whole words also counts (``take the red
pill``, ``alpha male``).

Usage (run from the PromptingSlang root):
    python experiments/semantic_slot_queries/visualizer/visualize.py
    python experiments/semantic_slot_queries/visualizer/visualize.py -o figures/accuracy.png
"""

from __future__ import annotations

import argparse
import re
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT))
from src.response_utils import model_short, read_responses, responded_word  # noqa: E402

EXP_DIR = Path(__file__).resolve().parents[1]
DEFAULT_RESPONSES = EXP_DIR / "responses"
DEFAULT_OUTPUT = EXP_DIR / "figures" / "target_word_accuracy.png"
WORDS_CHRONO_ORDER = ["lol","sick","troll","bro","swag","omg","meh","lmao","vibe","vibes","alpha","red pill","legit","slay","gaslight","glow-up","aura","lowkey","situationship"]


def _norm(s: str) -> str:
    """Lowercase and strip everything but letters/digits, so 'glow-up' == 'glowup'."""
    return re.sub(r"[^a-z0-9]", "", s.lower())


def _stem(s: str) -> str:
    """Light suffix stemmer on a normalized string: drop -ing/-ed/-s (keep >=3 char root).

    Enough to fold common inflections (gaslighting->gaslight, slayed->slay,
    alphas->alpha) without a full stemming dependency. '-er' is deliberately
    *not* stripped so 'swagger' stays distinct from 'swag'.
    """
    s = _norm(s)
    for suf in ("ing", "ed", "s"):
        if s.endswith(suf) and len(s) - len(suf) >= 3:
            return s[: -len(suf)]
    return s


def _stem_tokens(s: str) -> list[str]:
    """Stem each whitespace/punctuation-separated token of *s*."""
    return [_stem(t) for t in re.split(r"[^a-z0-9]+", s.lower()) if t]


def is_correct(response: str, word: str) -> bool:
    """True if *response* names *word*.

    Matches the joined inflected form (``redpilled``->red pill) and also credits
    a response that contains the target as a run of whole words (``take the red
    pill``, ``alpha male``), while keeping unrelated words distinct (``brother``
    is not ``bro``).
    """
    if _stem(response) == _stem(word):
        return True
    target, resp = _stem_tokens(word), _stem_tokens(response)
    n = len(target)
    return bool(target) and any(resp[i:i + n] == target for i in range(len(resp) - n + 1))


def target_prompt_id(word: str) -> str:
    """'red pill' -> 'exp3_red_pill', 'glow-up' -> 'exp3_glow_up'."""
    return "exp3_" + re.sub(r"[ \-]", "_", word)


def accuracy_by_model(records: list[dict]) -> dict[str, dict[str, tuple[int, int]]]:
    """{model: {word: (correct, total)}} over the target-word prompts."""
    id_to_word = {target_prompt_id(w): w for w in WORDS_CHRONO_ORDER}
    tallies: dict[str, dict[str, list[int]]] = defaultdict(
        lambda: defaultdict(lambda: [0, 0]))
    for rec in records:
        word = id_to_word.get(rec.get("prompt_id"))
        if word is None:
            continue
        cell = tallies[rec.get("model", "unknown")][word]
        cell[1] += 1  # total
        if is_correct(responded_word(rec), word):
            cell[0] += 1  # correct
    return {m: {w: tuple(c) for w, c in wc.items()} for m, wc in tallies.items()}


def render(per_model: dict[str, dict[str, tuple[int, int]]], output: Path | None) -> None:
    models = sorted(per_model)
    words = [w for w in WORDS_CHRONO_ORDER if any(per_model[m].get(w) for m in models)]
    if not models or not words:
        sys.exit("No target-word responses found.")

    x = np.arange(len(words))
    n = len(models)
    width = 0.8 / max(n, 1)
    fig, ax = plt.subplots(figsize=(max(11, len(words) * 0.9), 6))
    for i, m in enumerate(models):
        pct = []
        for w in words:
            correct, total = per_model[m].get(w, (0, 0))
            pct.append(100.0 * correct / total if total else 0.0)
        ax.bar(x + (i - (n - 1) / 2) * width, pct, width, label=model_short(m))

    ax.set_xticks(x)
    ax.set_xticklabels(words, rotation=40, ha="right", fontsize=9)
    ax.set_xlabel("Target word")
    ax.set_ylabel("% responses matching the expected target word")
    ax.set_ylim(0, 100)
    ax.set_title("Semantic slot queries — target-word recovery accuracy per model")
    ax.legend(fontsize="small", loc="upper left", bbox_to_anchor=(1.01, 1.0))
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    fig.tight_layout()
    if output:
        output.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output, dpi=150)
        print(f"Saved: {output}")
    else:
        plt.show()
    plt.close(fig)


def main() -> None:
    p = argparse.ArgumentParser(
        description="Per-model % of target-word prompts answered with the expected word.")
    p.add_argument("responses", nargs="?", default=str(DEFAULT_RESPONSES),
                   help=f"Response JSONL file or directory (default: {DEFAULT_RESPONSES}).")
    p.add_argument("-o", "--output", type=Path, default=DEFAULT_OUTPUT,
                   help=f"Output path (default: {DEFAULT_OUTPUT}). Pass '-' to display interactively.")
    args = p.parse_args()

    records = read_responses(args.responses)
    if not records:
        sys.exit(f"No response records found in {args.responses}.")
    render(accuracy_by_model(records), None if str(args.output) == "-" else args.output)


if __name__ == "__main__":
    main()
