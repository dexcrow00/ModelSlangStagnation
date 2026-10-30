#!/usr/bin/env python3
"""
generate_synthetic_scenario.py — Top up the minority class of imbalanced
scenario words with synthetic examples, written straight into the per-word
CSVs of the training (``scenario_finetune_annotations``) and held-out
(``scenario_finetune_validation``) sets.

For every ``<stem>_annotations.csv`` in each directory, the labelled rows are
counted. If the gap between ``is_slang == 1`` and ``is_slang == 0`` is larger
than ``IMBALANCE_THRESHOLD``, Claude is asked for ``N_PER_TOP_UP`` new
contexts of the lacking class, which are appended to that same file with
``uri = "synthetic"``. Unlabelled (blank) rows are not counted, so a file that
has not been annotated yet is left alone.

A few of the file's real labelled rows are included in the prompt so the model
sees which sense of the word counts as slang and what the contexts look like.
Generated contexts that already appear in either directory's file for the word
are dropped, so train and validation never share a synthetic row.

Each run tops up a word once; rerun to add another batch if it is still
imbalanced.

Usage:
    python generate_synthetic_scenario.py --dry-run   # show counts and what would be generated
    python generate_synthetic_scenario.py --limit 1   # test one file first
    python generate_synthetic_scenario.py             # all imbalanced files
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
IMBALANCE_THRESHOLD = 30      # top up only when |n_slang - n_not| exceeds this
N_PER_TOP_UP = 20             # synthetic rows added to the lacking class per run
N_REFERENCE = 5               # real labelled rows per class shown to the model
SYNTHETIC_URI = "synthetic"

sys.path.insert(0, str(HERE))
from Keys import ANTHROPIC_API_KEY  # FineWebAnalysis/Keys.py  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s",
                    handlers=[logging.StreamHandler(sys.stderr)])
log = logging.getLogger("synth")

MODEL = "claude-sonnet-5-5"

# Static so it can be prompt-cached across words.
SYSTEM_PROMPT = """You generate labelled training contexts for a slang-sense classifier.

You will be given a TARGET word or phrase, the LABEL to produce, how many contexts to write, and some real labelled contexts for that word taken from web text.

  is_slang = 1  -> the target is used in the specific internet / informal SLANG sense shown by the real is_slang=1 examples (or, if none are given, its most common current internet-slang sense)
  is_slang = 0  -> the target is used in an ordinary, literal or non-slang sense, or in a different sense than the slang one above

STYLE (match the real contexts):
  - each context is a window of roughly 25 to 60 words cut out of a web page: forum posts, comments, reviews, blogs, news, product pages, captions
  - normal web casing and punctuation; it may start or end mid-sentence
  - the target MUST appear verbatim exactly as given, once, somewhere in the context
  - vary topic, register and sentence structure widely; do not reuse a frame
  - do not copy or lightly paraphrase the real examples

OUTPUT: respond with a SINGLE JSON object and nothing else, of the form
{"examples": ["<context>", "<context>", ...]}
"""


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as fh:
        return list(csv.DictReader(fh))


def label_counts(rows: list[dict[str, str]]) -> tuple[int, int]:
    """(n_slang, n_not) over the labelled rows."""
    labels = [(r.get("is_slang") or "").strip() for r in rows]
    return labels.count("1"), labels.count("0")


def _reference_block(rows: list[dict[str, str]], rng: random.Random) -> str:
    lines = []
    for label in ("1", "0"):
        cls = [r["target_context"] for r in rows
               if (r.get("is_slang") or "").strip() == label and r.get("uri") != SYNTHETIC_URI]
        for ctx in rng.sample(cls, min(N_REFERENCE, len(cls))):
            lines.append(f"is_slang={label}: {' '.join(ctx.split())[:400]}")
    return "\n".join(lines) or "(none)"


def generate_for_word(client, word: str, label: int, n: int, reference: str) -> list[str]:
    """Call Claude once; return up to *n* new contexts of class *label* for *word*."""
    user_msg = (f'TARGET: "{word}"\nLABEL: is_slang = {label}\nCOUNT: {n}\n\n'
                f"Real labelled contexts for this word:\n{reference}\n\n"
                f"Write {n} new is_slang={label} contexts now. Respond with only the JSON object.")
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
    out: list[str] = []
    for ex in data.get("examples", []):
        t = " ".join(str(ex).split())
        if t and word.casefold() in t.casefold():
            out.append(t)
    return out


def append_rows(path: Path, word: str, label: int, contexts: list[str]) -> None:
    with path.open("a", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        for ctx in contexts:
            w.writerow([word, label, ctx, SYNTHETIC_URI])


def find_todo() -> list[tuple[Path, str, int, int, int]]:
    """[(csv_path, word, lacking_label, n_slang, n_not)] for every file over the threshold."""
    todo = []
    for d in (ANNOT_DIR, VAL_DIR):
        for f in sorted(d.glob("*_annotations.csv")):
            stem = f.name[: -len("_annotations.csv")]
            rows = read_rows(f)
            word = next(((r.get("target") or "").strip() for r in rows if (r.get("target") or "").strip()), "")
            if not word or stem in DROP or word.casefold() in DROP:
                continue
            n1, n0 = label_counts(rows)
            if abs(n1 - n0) > IMBALANCE_THRESHOLD:
                todo.append((f, word, 1 if n1 < n0 else 0, n1, n0))
    return todo


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--limit", type=int, default=None, help="Only process the first N files.")
    ap.add_argument("--dry-run", action="store_true", help="List files to top up and exit.")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    todo = find_todo()
    if args.limit is not None:
        todo = todo[: args.limit]

    log.info("%d file(s) with |slang - not| > %d.", len(todo), IMBALANCE_THRESHOLD)
    if args.dry_run:
        for f, word, label, n1, n0 in todo:
            print(f"  {f.parent.name:30} {word:18} slang={n1:3} not={n0:3} -> +{N_PER_TOP_UP} is_slang={label}")
        return

    import anthropic
    client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)
    rng = random.Random(args.seed)

    n_ok = n_fail = 0
    for i, (f, word, label, n1, n0) in enumerate(todo, 1):
        try:
            rows = read_rows(f)
            # Existing contexts for this word in both sets, to keep them disjoint.
            seen = {" ".join(r["target_context"].split()).casefold() for r in rows}
            for d in (ANNOT_DIR, VAL_DIR):
                other = d / f.name
                if other != f and other.exists():
                    seen |= {" ".join(r["target_context"].split()).casefold() for r in read_rows(other)}

            new: list[str] = []
            for ctx in generate_for_word(client, word, label, N_PER_TOP_UP, _reference_block(rows, rng)):
                if ctx.casefold() not in seen:
                    seen.add(ctx.casefold())
                    new.append(ctx)
            new = new[:N_PER_TOP_UP]
            if len(new) < N_PER_TOP_UP:
                log.warning("[%d/%d] %r returned only %d usable rows.", i, len(todo), word, len(new))
            append_rows(f, word, label, new)
            log.info("[%d/%d] %-30s %-16s slang=%d not=%d -> +%d is_slang=%d",
                     i, len(todo), f.parent.name, word, n1, n0, len(new), label)
            n_ok += 1
        except Exception as exc:  # noqa: BLE001
            log.error("[%d/%d] FAILED for %r (%s): %s", i, len(todo), word, f.parent.name, exc)
            n_fail += 1

    log.info("Done. %d ok, %d failed.", n_ok, n_fail)


if __name__ == "__main__":
    main()
