#!/usr/bin/env python3
"""
fineweb_context.py — Extract target-word context windows from a pre-sampled
FineWeb HuggingFace dataset config.

Streams through the chosen pre-sampled config (sample-10BT by default) in a
single pass, accumulates context rows per CC dump, and writes one CSV per dump
to the output directory. Dumps that already have enough rows are skipped.

Output CSV (one per dump):
  uri            — source URL of the document
  target_context — ±K-token window of text around the match (includes the match)

Usage:
    # All dumps in the default sample
    python fineweb_context.py --output-dir contexts/ --words target_words.txt

    # Larger sample, from 2020 onwards, 500 context rows per dump
    python fineweb_context.py --output-dir contexts/ --words target_words.txt \\
        --sample sample-100BT --since 2020 --n-rows 500

    # Explicit dump range
    python fineweb_context.py --output-dir contexts/ --words target_words.txt \\
        --from-dump CC-MAIN-2022-05 --to-dump CC-MAIN-2024-10

Requires: datasets
"""

from __future__ import annotations

import argparse
import csv
import logging
import re
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, NamedTuple, Optional, Set, Tuple

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(line_buffering=True)  # type: ignore[attr-defined]

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stderr)],
)
log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Tokenizer  (identical to word_context.py)
# ---------------------------------------------------------------------------

_WORD_RE = re.compile(r"[a-zA-Z'''\-]+")


def tokenize(text: str) -> List[str]:
    return [w.lower() for w in _WORD_RE.findall(text)]


# ---------------------------------------------------------------------------
# Target words / phrases  (identical to word_context.py)
# ---------------------------------------------------------------------------

class Targets(NamedTuple):
    single_words: Set[str]
    phrase_index: Dict[str, List[Tuple[str, ...]]]


def load_target_words(path: str) -> Targets:
    single_words: Set[str] = set()
    phrases: List[Tuple[str, ...]] = []
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            raw = line.strip()
            if not raw or raw.startswith("#"):
                continue
            parts = raw.lower().split()
            if len(parts) == 1:
                single_words.add(parts[0])
            else:
                phrases.append(tuple(parts))
    phrase_index: Dict[str, List[Tuple[str, ...]]] = {}
    for phrase in phrases:
        phrase_index.setdefault(phrase[0], []).append(phrase)
    log.info("Loaded %d single words and %d phrases from %s",
             len(single_words), len(phrases), path)
    return Targets(single_words=single_words, phrase_index=phrase_index)


# ---------------------------------------------------------------------------
# Context extraction  (identical to word_context.py)
# ---------------------------------------------------------------------------

def extract_contexts(tokens: List[str], targets: Targets, k: int) -> List[Tuple[str, str]]:
    """Find all target matches and return (matched_word, context_string) pairs.

    Phrases take priority over single-word matches at the same position.
    Advances past matched tokens to avoid double-counting overlapping matches.
    """
    results: List[Tuple[str, str]] = []
    i = 0
    while i < len(tokens):
        token = tokens[i]
        matched_key: Optional[str] = None
        matched_len = 0

        if token in targets.phrase_index:
            for phrase in targets.phrase_index[token]:
                n = len(phrase)
                if tuple(tokens[i:i + n]) == phrase:
                    matched_key = " ".join(phrase)
                    matched_len = n
                    break

        if matched_key is None and token in targets.single_words:
            matched_key = token
            matched_len = 1

        if matched_key:
            start = max(0, i - k)
            end   = min(len(tokens), i + matched_len + k)
            context = " ".join(tokens[start:end])
            results.append((matched_key, context))
            i += matched_len
        else:
            i += 1
    return results


# ---------------------------------------------------------------------------
# CSV helpers
# ---------------------------------------------------------------------------

_FIELDNAMES = ["uri", "target_context"]


def _count_rows(path: Path) -> int:
    if not path.exists():
        return 0
    with path.open(encoding="utf-8") as f:
        return max(0, sum(1 for _ in f) - 1)  # subtract header


def _write_rows(path: Path, rows: List[Dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=_FIELDNAMES)
        writer.writeheader()
        writer.writerows(rows)


# ---------------------------------------------------------------------------
# Dump filter
# ---------------------------------------------------------------------------

_CC_MAIN_RE = re.compile(r"^CC-MAIN-\d{4}-\d{2,4}$")


def _dump_matches(
    dump_id: str,
    since: Optional[int],
    from_dump: Optional[str],
    to_dump: Optional[str],
) -> bool:
    if not _CC_MAIN_RE.match(dump_id):
        return False
    if since and int(dump_id[8:12]) < since:
        return False
    if from_dump and dump_id < from_dump:
        return False
    if to_dump and dump_id > to_dump:
        return False
    return True


# ---------------------------------------------------------------------------
# Timing helper
# ---------------------------------------------------------------------------

def _fmt_duration(secs: float) -> str:
    s = int(secs)
    h, rem = divmod(s, 3600)
    m, s   = divmod(rem, 60)
    if h:
        return f"{h}h {m:02d}m"
    return f"{m}m {s:02d}s"


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

_SAMPLE_CONFIGS = ["sample-10BT", "sample-100BT", "sample-350BT"]


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Extract target-word context windows from a pre-sampled FineWeb config.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Default sample (sample-10BT), all dumps
  python fineweb_context.py --output-dir contexts/ --words target_words.txt

  # Larger sample, from 2020 onwards, 500 context rows per dump
  python fineweb_context.py --output-dir contexts/ --words target_words.txt \\
      --sample sample-100BT --since 2020 --n-rows 500

  # Explicit dump range
  python fineweb_context.py --output-dir contexts/ --words target_words.txt \\
      --from-dump CC-MAIN-2022-05 --to-dump CC-MAIN-2024-10
        """,
    )

    # Output
    p.add_argument("--output-dir", required=True, metavar="DIR", dest="output_dir",
                   type=Path,
                   help="Directory for output CSV files (one per dump).")

    # Target words
    p.add_argument("--words", default="target_words.txt", metavar="FILE",
                   help="Plain-text target words/phrases file, one per line "
                        "(default: target_words.txt).")

    # FineWeb config
    p.add_argument("--sample", default="sample-10BT", metavar="CONFIG",
                   choices=_SAMPLE_CONFIGS,
                   help="Pre-sampled FineWeb config to stream "
                        f"({', '.join(_SAMPLE_CONFIGS)}; default: sample-10BT).")

    # Context
    p.add_argument("--context-window", type=int, default=10, metavar="K",
                   dest="context_window",
                   help="Tokens on each side of a match to include (default: 10).")

    # Row target
    p.add_argument("--n-rows", type=int, default=500, metavar="N", dest="n_rows",
                   help="Target number of context rows per dump (default: 500). "
                        "A dump's output file is written and collection stops as "
                        "soon as this many rows are accumulated.")

    # Dump range filters
    p.add_argument("--since", type=int, default=None, metavar="YEAR",
                   help="Only collect rows for dumps from this year onwards (e.g. 2020).")
    p.add_argument("--from-dump", default=None, metavar="ID", dest="from_dump",
                   help="Only collect rows for dumps at or after this ID "
                        "(e.g. CC-MAIN-2020-05).")
    p.add_argument("--to-dump", default=None, metavar="ID", dest="to_dump",
                   help="Only collect rows for dumps up to and including this ID.")

    # Re-run
    p.add_argument("--force", action="store_true",
                   help="Re-process dumps whose output files already have enough rows.")

    return p


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    targets = load_target_words(args.words)

    try:
        from datasets import load_dataset  # type: ignore
    except ImportError:
        log.error("datasets package is required. pip install datasets")
        sys.exit(1)

    log.info("Streaming FineWeb config '%s' ...", args.sample)
    log.info("  output-dir     : %s", args.output_dir)
    log.info("  n-rows         : %d per dump", args.n_rows)
    log.info("  context-window : %d", args.context_window)

    ds = load_dataset("HuggingFaceFW/fineweb", name=args.sample, streaming=True)["train"]

    # Per-dump accumulators
    dump_rows: Dict[str, List[Dict[str, str]]] = defaultdict(list)
    dump_done: Set[str] = set()   # dumps written out or skipped

    docs_seen   = 0
    t0          = time.monotonic()
    log_every   = 100_000         # log a progress line every N documents

    for record in ds:
        dump_id = record.get("dump") or ""

        # Ignore records whose dump doesn't match the filter
        if not _dump_matches(dump_id, args.since, args.from_dump, args.to_dump):
            continue

        # First time we encounter this dump: check for an existing output file
        if dump_id not in dump_done and dump_id not in dump_rows:
            out = args.output_dir / f"word_context_{dump_id}.csv"
            if not args.force and _count_rows(out) >= args.n_rows:
                log.info("Skipping %s — already complete (%d rows)", dump_id, _count_rows(out))
                dump_done.add(dump_id)

        if dump_id in dump_done:
            continue

        # Extract context rows from this document
        text = record.get("text") or ""
        url  = record.get("url")  or ""
        if text:
            tokens = tokenize(text)
            for _, context in extract_contexts(tokens, targets, args.context_window):
                dump_rows[dump_id].append({"uri": url, "target_context": context})

        docs_seen += 1

        # If this dump has reached n_rows, write it out immediately and free memory
        if len(dump_rows[dump_id]) >= args.n_rows:
            out = args.output_dir / f"word_context_{dump_id}.csv"
            _write_rows(out, dump_rows[dump_id])
            elapsed = _fmt_duration(time.monotonic() - t0)
            log.info("✓ %s — %d rows → %s  [%s, %d docs seen]",
                     dump_id, len(dump_rows[dump_id]), out, elapsed, docs_seen)
            dump_done.add(dump_id)
            del dump_rows[dump_id]

        if docs_seen % log_every == 0:
            elapsed = _fmt_duration(time.monotonic() - t0)
            log.info("  ... %d docs scanned | %d dumps complete | %d dumps in progress  [%s]",
                     docs_seen, len(dump_done), len(dump_rows), elapsed)

    # Write any dumps that never reached n_rows
    for dump_id, rows in dump_rows.items():
        if rows:
            out = args.output_dir / f"word_context_{dump_id}.csv"
            _write_rows(out, rows)
            log.info("✓ %s — %d rows (below target) → %s", dump_id, len(rows), out)

    log.info("All done. %d docs scanned | %d dump file(s) written  [%s]",
             docs_seen, len(dump_done) + len(dump_rows),
             _fmt_duration(time.monotonic() - t0))


if __name__ == "__main__":
    main()
