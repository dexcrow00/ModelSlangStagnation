#!/usr/bin/env python3
"""
generate_synthetic_annotations.py — Build synthetic slang-annotation CSVs.

For every unique target word in FineWebAnalysis/target_words.txt, ask Claude to
produce 50-100 natural example sentences that use the word, each labelled as
slang (is_slang=1) or ordinary/non-slang usage (is_slang=0).  The output mirrors
the style of the existing fine_web_10BT annotation files:

    columns : uri,target_context,is_slang,synthetic
    every row here is fully synthetic, so  uri="synthetic"  and  synthetic=1.

The target word itself is encoded in the FILENAME (`<word>_annotations.csv`),
exactly like the fine_web_10BT files (auc_eval.py recovers it from the stem).

Balance rule:
  - Words that also have an everyday / non-slang meaning  -> ~50/50 slang vs not.
  - Words with no standard non-slang meaning (skibidi, rizz, ...) -> all slang.
The model decides which case applies per word.

The run is resumable: any word whose CSV already exists (non-empty) is skipped.

Usage:
    C:\\Projects\\CommonCrawlAnalysis\\.venv\\Scripts\\python.exe \\
        DataProcessingTools/generate_synthetic_annotations.py [--limit N] [--dry-run]
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import sys
from pathlib import Path
from typing import List, Tuple

REPO_ROOT = Path(__file__).resolve().parents[2]
WORDS_FILE = REPO_ROOT / "FineWebAnalysis" / "target_words.txt"
OUT_DIR = REPO_ROOT / "DataProcessingTools" / "completed_annotations" / "synthetic_annotations"

sys.path.insert(0, str(REPO_ROOT / "FineWebAnalysis"))
from Keys import ANTHROPIC_API_KEY # pyright: ignore[reportMissingImports]

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stderr)],
)
log = logging.getLogger("synth")

MODEL = "claude-opus-4-7"

# A large, *static* system prompt so it can be prompt-cached across every
# per-word request (cache_control on the last system block).
SYSTEM_PROMPT = """You generate labelled example sentences for training a slang-detection classifier.

You will be given a single TARGET word or phrase. Produce a set of natural example sentences, each of which uses the target, and label every sentence:
  is_slang = 1  -> the target is used as internet / informal SLANG in this sentence
  is_slang = 0  -> the target is used in an ordinary, literal, non-slang sense

DECIDE the word's nature first:
  - If the target ALSO has a common everyday / literal / dictionary meaning
    (e.g. "fire", "sick", "salty", "troll", "karen", "tea", "savage", "lit",
    "mid", "bop", "snatched", "cooked", "basic", "stan"), set
    has_non_slang_meaning = true and produce a roughly 50/50 mix of is_slang=1
    and is_slang=0 sentences.
  - If the target is essentially slang-only with no standard non-slang meaning
    (e.g. "skibidi", "rizz", "gyatt", "delulu", "sybau", "sussy baka",
    "no cap", "yeet", "rizz", "huzz"), set has_non_slang_meaning = false and make
    EVERY sentence is_slang=1.

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


def load_words() -> List[str]:
    """Read target_words.txt, strip whitespace, drop blanks, dedupe (order-stable)."""
    seen = set()
    words: List[str] = []
    for raw in WORDS_FILE.read_text(encoding="utf-8").splitlines():
        w = raw.strip()
        if not w:
            continue
        key = w.lower()
        if key in seen:
            continue
        seen.add(key)
        words.append(w)
    return words


def safe_filename(word: str) -> str:
    """`<word>_annotations.csv`, with characters illegal on Windows replaced."""
    cleaned = word
    for bad in '<>:"/\\|?*':
        cleaned = cleaned.replace(bad, "")
    return f"{cleaned}_annotations.csv"


def generate_for_word(client, word: str) -> Tuple[bool, List[Tuple[str, int]]]:
    """Call Claude once for `word`; return (has_non_slang_meaning, [(text, is_slang), ...])."""
    user_msg = (
        f'TARGET: "{word}"\n\n'
        "Generate the labelled example sentences for this target now, "
        "following all the rules. Respond with only the JSON object."
    )

    with client.messages.stream(
        model=MODEL,
        max_tokens=16000,
        thinking={"type": "adaptive"},
        system=[
            {
                "type": "text",
                "text": SYSTEM_PROMPT,
                "cache_control": {"type": "ephemeral"},
            }
        ],
        messages=[{"role": "user", "content": user_msg}],
    ) as stream:
        final = stream.get_final_message()

    text = "".join(
        block.text for block in final.content if getattr(block, "type", None) == "text"
    ).strip()

    # Be forgiving: pull out the outermost JSON object if the model added stray text.
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1:
        raise ValueError(f"No JSON object in response for {word!r}: {text[:200]!r}")
    data = json.loads(text[start : end + 1])

    has_non_slang = bool(data.get("has_non_slang_meaning", False))
    rows: List[Tuple[str, int]] = []
    for ex in data.get("examples", []):
        t = (ex.get("text") or "").strip()
        try:
            lbl = int(ex.get("is_slang"))
        except (TypeError, ValueError):
            continue
        if t and lbl in (0, 1):
            rows.append((t, lbl))
    return has_non_slang, rows


def write_csv(path: Path, rows: List[Tuple[str, int]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["uri", "target_context", "is_slang", "synthetic"])
        for text, label in rows:
            w.writerow(["synthetic", text, label, 1])


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--limit", type=int, default=None,
                    help="Only process the first N (not-yet-done) words. For a test run.")
    ap.add_argument("--dry-run", action="store_true",
                    help="List the words that would be generated and exit.")
    ap.add_argument("--overwrite", action="store_true",
                    help="Regenerate even if an output CSV already exists.")
    args = ap.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    words = load_words()
    log.info("Loaded %d unique target words.", len(words))

    todo = []
    for w in words:
        out_path = OUT_DIR / safe_filename(w)
        if out_path.exists() and out_path.stat().st_size > 0 and not args.overwrite:
            continue
        todo.append(w)

    if args.limit is not None:
        todo = todo[: args.limit]

    log.info("%d words to generate (%d already done).", len(todo), len(words) - len(todo))

    if args.dry_run:
        for w in todo:
            print(f"  {w}  ->  {safe_filename(w)}")
        return

    import anthropic  # late import so --dry-run works without the package

    client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)

    n_ok = n_fail = 0
    for i, word in enumerate(todo, 1):
        out_path = OUT_DIR / safe_filename(word)
        try:
            has_non_slang, rows = generate_for_word(client, word)
            if len(rows) < 40:
                log.warning("[%d/%d] %r returned only %d rows.", i, len(todo), word, len(rows))
            write_csv(out_path, rows)
            n_pos = sum(lbl for _, lbl in rows)
            log.info("[%d/%d] %-16s -> %3d rows (%d slang / %d not, non_slang_meaning=%s)",
                     i, len(todo), word, len(rows), n_pos, len(rows) - n_pos, has_non_slang)
            n_ok += 1
        except Exception as exc:  # noqa: BLE001
            log.error("[%d/%d] FAILED for %r: %s", i, len(todo), word, exc)
            n_fail += 1

    log.info("Done. %d succeeded, %d failed. Output in %s", n_ok, n_fail, OUT_DIR)


if __name__ == "__main__":
    main()
