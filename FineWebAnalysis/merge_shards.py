#!/usr/bin/env python3
"""
merge_shards.py — Merge the sharded word-context CSVs produced by
fineweb_context.py when run with multiple workers.

fineweb_context.py --num-workers N writes one shard per worker:

    word_context_{dump}__{target}.part000.csv
    word_context_{dump}__{target}.part001.csv
    ...

This tool concatenates every shard that shares the same `{dump}__{target}` base
into a single file, keeping exactly one header row:

    word_context_{dump}__{target}.csv

Each output is written to a temp file and atomically renamed, so merging in
place (output dir == input dir) can never truncate a shard before it's read.

Usage:
    # Merge all shards in ./contexts in place
    python merge_shards.py --input-dir contexts/

    # Merge into a separate directory
    python merge_shards.py --input-dir contexts/ --output-dir merged/

    # Merge in place and delete the .partNNN shards afterwards
    python merge_shards.py --input-dir contexts/ --delete-shards
"""

from __future__ import annotations

import argparse
import csv
import logging
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List

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

# Matches "<base>.partNNN" (the shard suffix added by fineweb_context.py).
_SHARD_RE = re.compile(r"^(?P<base>.+)\.part\d+$")

# CSV field size can be large (long context windows); lift the default limit.
csv.field_size_limit(min(sys.maxsize, 2**31 - 1))


# ---------------------------------------------------------------------------
# Grouping
# ---------------------------------------------------------------------------

def group_shards(input_dir: Path, pattern: str) -> Dict[str, List[Path]]:
    """Group CSV files by their {dump}__{target} base name.

    A file named ``<base>.partNNN.csv`` is grouped under ``<base>``; a file
    named ``<base>.csv`` (already merged / single-worker) is grouped under
    ``<base>`` too, so re-running the merge is idempotent.
    """
    groups: Dict[str, List[Path]] = defaultdict(list)
    for p in sorted(input_dir.glob(pattern)):
        if not p.is_file():
            continue
        stem = p.name[:-4]  # strip ".csv" (glob guarantees the suffix)
        m = _SHARD_RE.match(stem)
        base = m.group("base") if m else stem
        groups[base].append(p)
    return groups


def _is_shard(path: Path) -> bool:
    return bool(_SHARD_RE.match(path.name[:-4]))


# ---------------------------------------------------------------------------
# Merge
# ---------------------------------------------------------------------------

def merge_group(base: str, files: List[Path], output_dir: Path) -> int:
    """Concatenate `files` into `<output_dir>/<base>.csv`. Returns data-row count.

    Writes to a temp file and atomically renames, so an in-place merge never
    truncates a source before it has been fully read.
    """
    out = output_dir / f"{base}.csv"
    tmp = out.with_name(out.name + ".tmp")

    rows = 0
    with tmp.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        header_written = False
        for src in sorted(files):
            with src.open("r", newline="", encoding="utf-8") as rf:
                reader = csv.reader(rf)
                try:
                    header = next(reader)
                except StopIteration:
                    continue  # empty shard (not even a header)
                if not header_written:
                    writer.writerow(header)
                    header_written = True
                for row in reader:
                    writer.writerow(row)
                    rows += 1

    if not header_written:
        # Every source was empty; produce nothing rather than an empty file.
        tmp.unlink(missing_ok=True)
        return 0

    tmp.replace(out)
    return rows


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=("Merge sharded word-context CSVs (word_context_*.partNNN.csv) "
                     "into one file per {dump}__{target}, keeping a single header."),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--input-dir", required=True, type=Path, dest="input_dir",
                   metavar="DIR",
                   help="Directory containing the shard CSV files.")
    p.add_argument("--output-dir", type=Path, default=None, dest="output_dir",
                   metavar="DIR",
                   help="Directory for merged files (default: same as --input-dir).")
    p.add_argument("--pattern", default="word_context_*.csv", metavar="GLOB",
                   help="Glob for shard files (default: word_context_*.csv).")
    p.add_argument("--delete-shards", action="store_true", dest="delete_shards",
                   help="Delete the .partNNN shard files after a successful merge.")
    return p


def main() -> None:
    args = build_parser().parse_args()

    input_dir: Path = args.input_dir
    output_dir: Path = args.output_dir or input_dir

    if not input_dir.is_dir():
        log.error("Input dir does not exist: %s", input_dir)
        sys.exit(1)
    output_dir.mkdir(parents=True, exist_ok=True)

    groups = group_shards(input_dir, args.pattern)
    if not groups:
        log.error("No files matched %s in %s", args.pattern, input_dir)
        sys.exit(1)

    n_inputs = sum(len(v) for v in groups.values())
    log.info("Merging %d shard file(s) into %d output file(s) → %s",
             n_inputs, len(groups), output_dir)

    files_written = 0
    rows_total = 0
    for i, (base, files) in enumerate(sorted(groups.items()), 1):
        # Skip a no-op: a lone, already-merged file written back to the same path.
        if (len(files) == 1 and output_dir == input_dir
                and files[0].name == f"{base}.csv"):
            continue

        rows = merge_group(base, files, output_dir)
        files_written += 1
        rows_total += rows
        if i % 200 == 0:
            log.info("  merged %d/%d groups ...", i, len(groups))

        if args.delete_shards:
            for src in files:
                if _is_shard(src):
                    src.unlink(missing_ok=True)

    log.info("Done. %d merged file(s), %d data row(s)%s.",
             files_written, rows_total,
             " (shards deleted)" if args.delete_shards else "")


if __name__ == "__main__":
    main()
