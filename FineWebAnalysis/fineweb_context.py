#!/usr/bin/env python3
"""
fineweb_context.py — Extract target-word context windows from the FineWeb HuggingFace dataset.

For each FineWeb dump (CC-MAIN-* config), streams records using shard-based random
sampling, extracts ±K-token context windows around target words/phrases, and writes
one CSV per dump to the output directory.

Output CSV (one per dump):
  uri            — source URL of the document
  target_context — ±K-token window of text around the match (includes the match)

Dumps are discovered automatically from the dataset's available configs and can be
filtered by year or ID range. Already-complete output files are skipped unless
--force is given.

Usage:
    # All dumps
    python fineweb_context.py --output-dir contexts/ --words target_words.txt

    # From 2020 onwards, 500 context rows per dump
    python fineweb_context.py --output-dir contexts/ --words target_words.txt \\
        --since 2020 --n-rows 500

    # Explicit range
    python fineweb_context.py --output-dir contexts/ --words target_words.txt \\
        --from-dump CC-MAIN-2022-05 --to-dump CC-MAIN-2024-10

Requires: datasets, torch (for HuggingFace), pandas
"""

from __future__ import annotations

import argparse
import csv
import logging
import random
import re
import sys
import time
from pathlib import Path
from typing import Dict, Iterator, List, NamedTuple, Optional, Set, Tuple

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
# Dump discovery
# ---------------------------------------------------------------------------

_CC_MAIN_RE = re.compile(r"^CC-MAIN-\d{4}-\d{2,4}$")


def _get_dump_configs() -> List[str]:
    """Return all CC-MAIN-* config names available in HuggingFaceFW/fineweb."""
    try:
        from datasets import get_dataset_config_names  # type: ignore
    except ImportError:
        log.error("datasets package is required. pip install datasets")
        sys.exit(1)

    log.info("Fetching available FineWeb configs from HuggingFace ...")
    configs = get_dataset_config_names("HuggingFaceFW/fineweb")
    dumps = sorted(c for c in configs if _CC_MAIN_RE.match(c))
    log.info("Found %d CC-MAIN dump configs.", len(dumps))
    return dumps


# ---------------------------------------------------------------------------
# Per-dump processing
# ---------------------------------------------------------------------------

def _crawl_seed(base_seed: int, crawl_index: int) -> int:
    return (base_seed * 1_000_003 + crawl_index) & 0x7FFF_FFFF


def process_dump(
    dump_id: str,
    targets: Targets,
    context_window: int,
    n_rows: int,
    seed: int,
    shards_per_dump: int,
    records_per_shard: int,
    total_shards: int,
) -> List[Dict[str, str]]:
    """Stream a random selection of shards from one FineWeb dump config and
    extract context rows until n_rows is reached or all selected shards are
    exhausted.

    Returns a list of {"uri": ..., "target_context": ...} dicts.
    """
    try:
        from datasets import load_dataset  # type: ignore
    except ImportError:
        log.error("datasets package is required. pip install datasets")
        sys.exit(1)

    rng = random.Random(seed)
    n_shards = min(shards_per_dump, total_shards)
    shard_indices = sorted(rng.sample(range(total_shards), n_shards))

    log.info("  Loading dump config '%s' (sampling %d/%d shards, up to %d records each) ...",
             dump_id, n_shards, total_shards, records_per_shard)

    ds = load_dataset("HuggingFaceFW/fineweb", name=dump_id, streaming=True)["train"]

    rows: List[Dict[str, str]] = []
    for shard_idx in shard_indices:
        if len(rows) >= n_rows:
            break
        shard_ds = ds.shard(num_shards=total_shards, index=shard_idx)
        docs_seen = 0
        for record in shard_ds:
            if docs_seen >= records_per_shard or len(rows) >= n_rows:
                break
            text = record.get("text") or ""
            url  = record.get("url")  or ""
            if not text:
                continue
            tokens = tokenize(text)
            for _, context in extract_contexts(tokens, targets, context_window):
                rows.append({"uri": url, "target_context": context})
            docs_seen += 1

        log.info("    Shard %4d: %d docs scanned  |  %d context rows so far",
                 shard_idx, docs_seen, len(rows))

    return rows


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

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Extract target-word context windows from the FineWeb HuggingFace dataset.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # All available dumps, default settings
  python fineweb_context.py --output-dir contexts/ --words target_words.txt

  # Dumps from 2020 onwards, 500 context rows each
  python fineweb_context.py --output-dir contexts/ --words target_words.txt \\
      --since 2020 --n-rows 500

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

    # Context
    p.add_argument("--context-window", type=int, default=10, metavar="K",
                   dest="context_window",
                   help="Tokens on each side of a match to include (default: 10).")

    # Row target
    p.add_argument("--n-rows", type=int, default=500, metavar="N", dest="n_rows",
                   help="Target number of context rows per dump (default: 500). "
                        "Processing stops early for a dump once this is reached.")

    # Sampling parameters
    p.add_argument("--total-shards", type=int, default=200, metavar="N",
                   dest="total_shards",
                   help="Assumed total shard count per dump config (default: 200). "
                        "Used for shard() calls — safe to set higher than actual count.")
    p.add_argument("--shards-per-dump", type=int, default=50, metavar="N",
                   dest="shards_per_dump",
                   help="Number of randomly selected shards to visit per dump (default: 50). "
                        "Increase for better coverage at the cost of speed.")
    p.add_argument("--records-per-shard", type=int, default=200, metavar="N",
                   dest="records_per_shard",
                   help="Max documents to read from each shard (default: 200).")

    # Reproducibility
    p.add_argument("--seed", type=int, default=42,
                   help="Base random seed (default: 42). Each dump gets a derived seed.")

    # Dump range filters
    p.add_argument("--since", type=int, default=None, metavar="YEAR",
                   help="Only process dumps from this year onwards (e.g. 2020).")
    p.add_argument("--from-dump", default=None, metavar="ID", dest="from_dump",
                   help="Start from this dump ID inclusive (e.g. CC-MAIN-2020-05).")
    p.add_argument("--to-dump", default=None, metavar="ID", dest="to_dump",
                   help="Stop after this dump ID inclusive (e.g. CC-MAIN-2024-10).")

    # Re-run
    p.add_argument("--force", action="store_true",
                   help="Re-process dumps even if the output file already exists "
                        "with enough rows.")

    return p


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    targets = load_target_words(args.words)

    # Discover and filter dumps
    dump_ids = _get_dump_configs()

    if args.since:
        dump_ids = [d for d in dump_ids if int(d[8:12]) >= args.since]
    if args.from_dump:
        dump_ids = [d for d in dump_ids if d >= args.from_dump]
    if args.to_dump:
        dump_ids = [d for d in dump_ids if d <= args.to_dump]

    if not dump_ids:
        log.error("No dumps matched the specified filters.")
        sys.exit(1)

    log.info("Processing %d dump(s): %s → %s", len(dump_ids), dump_ids[0], dump_ids[-1])
    log.info("  n-rows          : %d", args.n_rows)
    log.info("  context-window  : %d", args.context_window)
    log.info("  shards-per-dump : %d / %d", args.shards_per_dump, args.total_shards)
    log.info("  records-per-shard: %d", args.records_per_shard)

    n_total = len(dump_ids)
    elapsed_times: List[float] = []

    for i, dump_id in enumerate(dump_ids):
        out = args.output_dir / f"word_context_{dump_id}.csv"
        n_remaining = n_total - i - 1

        # Skip if already complete
        if not args.force and out.exists():
            n_existing = _count_rows(out)
            if n_existing >= args.n_rows:
                log.info("[%d/%d] %s — skipped (%d rows already collected)",
                         i + 1, n_total, dump_id, n_existing)
                continue
            log.info("[%d/%d] %s — output exists but only %d/%d rows; reprocessing ...",
                     i + 1, n_total, dump_id, n_existing, args.n_rows)

        seed = _crawl_seed(args.seed, i)
        t0   = time.monotonic()

        try:
            rows = process_dump(
                dump_id=dump_id,
                targets=targets,
                context_window=args.context_window,
                n_rows=args.n_rows,
                seed=seed,
                shards_per_dump=args.shards_per_dump,
                records_per_shard=args.records_per_shard,
                total_shards=args.total_shards,
            )
            _write_rows(out, rows)
            log.info("%s — %d context rows → %s", dump_id, len(rows), out)
        except Exception as exc:
            log.error("[%d/%d] ✗  %s — %s", i + 1, n_total, dump_id, exc)
            continue

        elapsed = time.monotonic() - t0
        elapsed_times.append(elapsed)

        if elapsed_times and n_remaining > 0:
            avg = sum(elapsed_times) / len(elapsed_times)
            eta = f"ETA {_fmt_duration(avg * n_remaining)}"
        elif n_remaining == 0:
            eta = "done"
        else:
            eta = "ETA unknown"

        log.info("[%d/%d] ✓  %s — %s  |  %s",
                 i + 1, n_total, dump_id, _fmt_duration(elapsed), eta)

    log.info("All done.")


if __name__ == "__main__":
    main()
