#!/usr/bin/env python3
"""
generate_synthetic_scenario.py — Synthetic slang/non-slang examples for the
scenario words, split disjointly between the training (``scenario_finetune_
annotations``) and held-out (``scenario_finetune_validation``) sets.

Mirrors DataProcessingTools/synthetic_data_generation/generate_synthetic_
annotations.py: for each word it asks Claude Opus 4.7 for a balanced set of
labelled example sentences (is_slang 1/0). The pool is then split 75/25 into a
train portion (-> annotations dir) and a val portion (-> validation dir) so the
two sets never share a synthetic sentence; words present only in the training
set get all their synthetic rows there.

Output CSVs match the scenario format (``target, is_slang, target_context,
uri``) with ``uri = synthetic://generated/<word>/<i>``, written as
``<stem>_synthetic.csv`` alongside the real-context ``<stem>_annotations.csv``.

Resumable: a word whose train synthetic CSV already exists is skipped.

Usage:
    python generate_synthetic_scenario.py --limit 1   # test one word first
    python generate_synthetic_scenario.py             # all remaining words
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import random
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ANNOT_DIR = HERE / "scenario_finetune_annotations"
VAL_DIR = HERE / "scenario_finetune_validation"
DROP = {"squad"}              # words to skip entirely
VAL_FRAC = 0.25               # fraction of each class routed to validation

sys.path.insert(0, str(HERE))
from Keys import ANTHROPIC_API_KEY  # FineWebAnalysis/Keys.py  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s",
                    handlers=[logging.StreamHandler(sys.stderr)])
log = logging.getLogger("synth")

MODEL = "claude-opus-4-7"

# Reused (verbatim) from the original generate_synthetic_annotations.py so the
# synthetic style matches the existing dataset; static for prompt-caching.
SYSTEM_PROMPT = """You generate labelled example sentences for training a slang-detection classifier.

You will be given a single TARGET word or phrase. Produce a set of natural example sentences, each of which uses the target, and label every sentence:
  is_slang = 1  -> the target is used as internet / informal SLANG in this sentence
  is_slang = 0  -> the target is used in an ordinary, literal, non-slang sense

DECIDE the word's nature first:
  - If the target ALSO has a common everyday / literal / dictionary meaning
    (e.g. "extra", "fresh", "goals", "tight", "woke", "cancelled", "clout",
    "pumped", "shaking", "era", "dude"), set has_non_slang_meaning = true and
    produce a roughly 50/50 mix of is_slang=1 and is_slang=0 sentences.
  - If the target is essentially slang-only with no standard non-slang meaning
    (e.g. "hits different", "left no crumbs", "rent free"), set
    has_non_slang_meaning = false and make EVERY sentence is_slang=1.

QUANTITY: produce between 60 and 90 sentences total. When the word has a
non-slang meaning, keep the two classes within ~10% of each other.

STYLE (match the existing dataset exactly):
  - one sentence per example, written all in lowercase
  - no terminal punctuation (no period/question mark/exclamation at the end)
  - avoid commas inside the sentence
  - apostrophes inside contractions are fine (i've, it's, that's)
  - roughly 8 to 28 words, sounding like real web text, comments, reviews,
    forum posts, captions, articles — vary the register and domain widely
  - the target word/phrase MUST appear verbatim in every sentence
  - for is_slang=0 sentences, use the literal meaning in a believable context
    (news, recipes, sports, science, how-to, history, etc.)
  - do not number the sentences and do not add quotation marks around them
  - make every sentence distinct; do not reuse the same sentence frame repeatedly

OUTPUT: respond with a SINGLE JSON object and nothing else, of the form
{"has_non_slang_meaning": <true|false>, "examples": [{"text": "<sentence>", "is_slang": <0|1>}, ...]}
"""


def words_from_dir(d: Path) -> dict[str, str]:
    """{stem: target_word} for every ``<stem>_annotations.csv`` in *d*."""
    out: dict[str, str] = {}
    for f in sorted(d.glob("*_annotations.csv")):
        stem = f.name[: -len("_annotations.csv")]
        with f.open(newline="", encoding="utf-8-sig") as fh:
            for row in csv.DictReader(fh):
                t = (row.get("target") or "").strip()
                if t:
                    out[stem] = t
                    break
    return out


def generate_for_word(client, word: str) -> list[tuple[str, int]]:
    """Call Claude once for *word*; return [(sentence, is_slang), ...]."""
    user_msg = (f'TARGET: "{word}"\n\nGenerate the labelled example sentences for this '
                "target now, following all the rules. Respond with only the JSON object.")
    with client.messages.stream(
        model=MODEL,
        max_tokens=16000,
        thinking={"type": "adaptive"},
        system=[{"type": "text", "text": SYSTEM_PROMPT, "cache_control": {"type": "ephemeral"}}],
        messages=[{"role": "user", "content": user_msg}],
    ) as stream:
        final = stream.get_final_message()
    text = "".join(b.text for b in final.content if getattr(b, "type", None) == "text").strip()
    start, end = text.find("{"), text.rfind("}")
    if start == -1 or end == -1:
        raise ValueError(f"No JSON object in response for {word!r}: {text[:200]!r}")
    data = json.loads(text[start:end + 1])
    rows: list[tuple[str, int]] = []
    seen: set[str] = set()
    for ex in data.get("examples", []):
        t = (ex.get("text") or "").strip()
        try:
            lbl = int(ex.get("is_slang"))
        except (TypeError, ValueError):
            continue
        if t and lbl in (0, 1) and t.lower() not in seen:
            seen.add(t.lower())
            rows.append((t, lbl))
    return rows


def _write(path: Path, word: str, rows: list[tuple[str, int]], start_idx: int) -> None:
    with path.open("w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["target", "is_slang", "target_context", "uri"])
        for i, (text, lbl) in enumerate(rows, start_idx):
            w.writerow([word, lbl, text, f"synthetic://generated/{word}/{i}"])


def split_pool(rows: list[tuple[str, int]], to_val: bool, rng: random.Random
               ) -> tuple[list, list]:
    """Split a pool into (train, val), holding out VAL_FRAC of each class for val."""
    if not to_val:
        return rows, []
    train, val = [], []
    for label in (0, 1):
        cls = [r for r in rows if r[1] == label]
        rng.shuffle(cls)
        k = round(len(cls) * VAL_FRAC)
        val.extend(cls[:k])
        train.extend(cls[k:])
    rng.shuffle(train)
    rng.shuffle(val)
    return train, val


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--limit", type=int, default=None, help="Only process the first N words.")
    ap.add_argument("--dry-run", action="store_true", help="List words and exit.")
    ap.add_argument("--overwrite", action="store_true", help="Regenerate existing synthetic CSVs.")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    annot_words = words_from_dir(ANNOT_DIR)
    val_stems = set(words_from_dir(VAL_DIR))
    todo = [(stem, word) for stem, word in annot_words.items()
            if word.casefold() not in DROP and stem not in DROP]
    if not args.overwrite:
        todo = [(s, w) for s, w in todo if not (ANNOT_DIR / f"{s}_synthetic.csv").exists()]
    if args.limit is not None:
        todo = todo[: args.limit]

    log.info("%d word(s) to generate (val split for those in %s).", len(todo), VAL_DIR.name)
    if args.dry_run:
        for stem, word in todo:
            print(f"  {word:18} -> train{'+val' if stem in val_stems else ''}")
        return

    import anthropic
    client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)
    rng = random.Random(args.seed)

    n_ok = n_fail = 0
    for i, (stem, word) in enumerate(todo, 1):
        try:
            pool = generate_for_word(client, word)
            if len(pool) < 30:
                log.warning("[%d/%d] %r returned only %d rows.", i, len(todo), word, len(pool))
            train, val = split_pool(pool, stem in val_stems, rng)
            _write(ANNOT_DIR / f"{stem}_synthetic.csv", word, train, 0)
            if val:
                _write(VAL_DIR / f"{stem}_synthetic.csv", word, val, len(train))
            pos = sum(l for _, l in pool)
            log.info("[%d/%d] %-16s pool=%d (%d slang/%d not) -> train=%d val=%d",
                     i, len(todo), word, len(pool), pos, len(pool) - pos, len(train), len(val))
            n_ok += 1
        except Exception as exc:  # noqa: BLE001
            log.error("[%d/%d] FAILED for %r: %s", i, len(todo), word, exc)
            n_fail += 1

    log.info("Done. %d ok, %d failed.", n_ok, n_fail)


if __name__ == "__main__":
    main()
