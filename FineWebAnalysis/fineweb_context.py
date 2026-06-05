#!/usr/bin/env python3
"""
fineweb_context.py — Extract target-word context windows from a pre-sampled
FineWeb HuggingFace dataset config.

Processes each CC dump independently: only that dump's parquet files are
streamed, and the stream is terminated as soon as n_rows context rows have
been collected.  Dumps that already have a complete output file are skipped.

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

Requires: datasets, huggingface_hub
"""

from __future__ import annotations

import argparse
import csv
from datasets import load_dataset
from huggingface_hub import HfFileSystem
import logging
import re
import string
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from Keys import HF_TOKEN

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
logging.getLogger("httpx").setLevel(logging.WARNING)

# ---------------------------------------------------------------------------
# Target words / phrases
# ---------------------------------------------------------------------------

def load_target_words(path: str) -> List[str]:
    """Load target words/phrases as lowercase strings, one per line."""
    targets: List[str] = []
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            raw = line.strip()
            if not raw or raw.startswith("#"):
                continue
            targets.append(raw.lower())
    log.info("Loaded %d target words/phrases from %s", len(targets), path)
    return targets


# ---------------------------------------------------------------------------
# Context extraction  (identical to word_context.py)
# ---------------------------------------------------------------------------

def extract_contexts(text: str, targets: List[str], k: int) -> List[Tuple[str, str]]:
    """Find all target matches and return (matched_target, context_string) pairs.

    Splits the text on whitespace and matches each target by checking the first
    character as a fast pre-filter, then comparing the joined word slice against
    the full target string. Advances past matched words to avoid double-counting.
    """
    words = [w.strip(string.punctuation) for w in text.lower().split()]
    results: List[Tuple[str, str]] = []
    i = 0
    while i < len(words):
        for target in targets:
            n = len(target.split())
            if words[i][0:1] == target[0] and " ".join(words[i:i + n]) == target:
                start = max(0, i - k)
                end   = min(len(words), i + n + k)
                results.append((target, " ".join(words[start:end])))
                i += n
                break
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



# ---------------------------------------------------------------------------
# Dump filter
# ---------------------------------------------------------------------------

_CC_MAIN_RE = re.compile(r"CC-MAIN-\d{4}-\d{2,4}")
_CC_MAIN_FULL_RE = re.compile(r"^CC-MAIN-\d{4}-\d{2,4}$")


def _dump_matches(
    dump_id: str,
    since: Optional[int],
    from_dump: Optional[str],
    to_dump: Optional[str],
) -> bool:
    if not _CC_MAIN_FULL_RE.match(dump_id):
        return False
    if since and int(dump_id[8:12]) < since:
        return False
    if from_dump and dump_id < from_dump:
        return False
    if to_dump and dump_id > to_dump:
        return False
    return True


# ---------------------------------------------------------------------------
# HuggingFace file enumeration
# ---------------------------------------------------------------------------

def _list_dump_files(sample: str) -> Dict[str, List[str]]:
    """Return {dump_id: [repo-relative parquet path, ...]} for every CC dump
    present in the given FineWeb sample config.

    Uses HfFileSystem to enumerate the parquet files without downloading them.
    Tries several common path layouts to be robust against dataset restructuring.
    """

    fs = HfFileSystem()
    repo_prefix = "datasets/HuggingFaceFW/fineweb/"

    # Overzealous Claude, but this should just run once.
    # Patterns to try in order (FineWeb layout has changed over time)
    patterns = [
        f"{repo_prefix}{sample}/data/CC-MAIN-*/*.parquet",   # nested per-dump dirs
        f"{repo_prefix}{sample}/data/CC-MAIN-*.parquet",     # flat per-dump files
        f"{repo_prefix}data/CC-MAIN-*/*.parquet",            # root-level (full dataset)
    ]

    for pattern in patterns:
        try:
            files = fs.glob(pattern)
        except Exception as exc:
            log.debug("Pattern %s: %s", pattern, exc)
            continue

        if not files:
            continue

        dump_files: Dict[str, List[str]] = {}
        for f in files:
            m = _CC_MAIN_RE.search(f)
            if not m:
                continue
            dump_id = m.group(0)
            # Strip HfFileSystem prefix to get a path relative to the HF repo root
            rel = f.removeprefix(repo_prefix)
            dump_files.setdefault(dump_id, []).append(rel)

        if dump_files:
            log.info("Found %d dumps (%s parquet files) via pattern: .../%s",
                     len(dump_files),
                     sum(len(v) for v in dump_files.values()),
                     pattern.removeprefix(repo_prefix))
            return dump_files

    log.error(
        "Could not find any parquet files for sample '%s'. "
        "Verify the dataset name and your network connection.",
        sample,
    )
    sys.exit(1)


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
        description=(
            "Extract target-word context windows from a pre-sampled FineWeb config. "
            "Each CC dump is streamed independently and the stream is terminated as "
            "soon as --n-rows context rows have been collected."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    # Output
    p.add_argument("--output-dir", required=True, metavar="DIR", dest="output_dir",
                   type=Path,
                   help="Directory for output CSV files (one per dump).")

    # Target words
    p.add_argument("--words", default="FineWebAnalysis/target_words.txt", metavar="FILE",
                   help="Plain-text target words/phrases file, one per line "
                        "(default: target_words.txt).")

    # FineWeb config
    p.add_argument("--sample", default="sample-10BT", metavar="CONFIG",
                   choices=_SAMPLE_CONFIGS,
                   help="Pre-sampled FineWeb config to use "
                        f"({', '.join(_SAMPLE_CONFIGS)}; default: sample-10BT).")

    # Context
    p.add_argument("--context-window", type=int, default=20, metavar="K",
                   dest="context_window",
                   help="Tokens on each side of a match to include (default: 20).")

    # Row target
    p.add_argument("--n-rows", type=int, default=None, metavar="N", dest="n_rows",
                   help="Target number of context rows per dump (default: sys.maxsize). "
                        "Streaming for each dump stops as soon as this many rows "
                        "are accumulated.")

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


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    targets = load_target_words(args.words)

    # ── Step 1: enumerate dumps ───────────────────────────────────────────────
    log.info("Enumerating dumps in FineWeb '%s' ...", args.sample)
    all_dump_files = _list_dump_files(args.sample)

    target_dumps = sorted(
        dump_id for dump_id in all_dump_files
        if _dump_matches(dump_id, args.since, args.from_dump, args.to_dump)
    )

    log.info("  output-dir     : %s", args.output_dir)
    if args.n_rows:
        log.info("  n-rows         : %d per dump", args.n_rows)
    log.info("  context-window : %d", args.context_window)
    log.info("  dumps to process: %d (of %d total in sample)",
             len(target_dumps), len(all_dump_files))

    if not target_dumps:
        log.error("No dumps matched the requested filters.")
        sys.exit(1)

    t0 = time.monotonic()
    dumps_written, dumps_skipped = 0, 0

    # ── Step 2: process each dump independently ───────────────────────────────
    for dump_idx, dump_id in enumerate(target_dumps, 1):
        out = args.output_dir / f"word_context_{dump_id}.csv"

        if not args.force and args.n_rows and _count_rows(out) >= args.n_rows:
            log.info("[%d/%d] Skipping %s — already complete",
                     dump_idx, len(target_dumps), dump_id)
            dumps_skipped += 1
            continue

        log.info("[%d/%d] Streaming %s ...", dump_idx, len(target_dumps), dump_id)
        t_dump = time.monotonic()

        # Load only this dump's parquet files — stream terminates as soon as
        # we break, so no unnecessary records are consumed.
        ds_dump = load_dataset(
            "HuggingFaceFW/fineweb",
            data_files={"train": all_dump_files[dump_id]},
            streaming=True,
            split="train",
            token=HF_TOKEN,
        )

        rows_written = 0
        rows_seen = 0
        docs_seen = 0
        out.parent.mkdir(parents=True, exist_ok=True)

        with out.open("w", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=_FIELDNAMES)
            writer.writeheader()

            for record in ds_dump:
                rows_seen += 1
                if rows_seen % 5000 == 0:
                    log.info("Rows seen: %i", rows_seen)
                text = record.get("text") or ""
                url  = record.get("url")  or ""
                if text:
                    for _, context in extract_contexts(text, targets, args.context_window):
                        writer.writerow({"uri": url, "target_context": context})
                        rows_written += 1
                        if args.n_rows and rows_written >= args.n_rows:
                            break           # stop accumulating mid-document
                        if rows_written % 5000 == 0:
                            log.info("Written %i rows", rows_written)
                docs_seen += 1
                if args.n_rows and rows_written >= args.n_rows:
                    break                   # early-terminate the dump stream

        elapsed_dump  = _fmt_duration(time.monotonic() - t_dump)
        elapsed_total = _fmt_duration(time.monotonic() - t0)

        if args.n_rows and rows_written < args.n_rows:
            log.info("  ✓ %s — %d/%d rows (dump exhausted) → %s  "
                     "[dump %s | total %s | %d docs scanned]",
                     dump_id, rows_written, args.n_rows, out.name,
                     elapsed_dump, elapsed_total, docs_seen)
        else:
            log.info("  ✓ %s — %d rows → %s  "
                     "[dump %s | total %s | %d docs scanned]",
                     dump_id, rows_written, out.name,
                     elapsed_dump, elapsed_total, docs_seen)

        dumps_written += 1

    log.info(
        "All done. %d dump(s) written, %d skipped  [%s]",
        dumps_written, dumps_skipped,
        _fmt_duration(time.monotonic() - t0),
    )


if __name__ == "__main__":
    main()
