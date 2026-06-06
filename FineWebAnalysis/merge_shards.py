#!/usr/bin/env python3
"""
merge_shards.py — Merge the sharded word-context Parquet files produced by
fineweb_context.py when run with multiple workers.

fineweb_context.py --num-workers N writes one shard per worker per dump:

    word_context_{dump}.part000.parquet
    word_context_{dump}.part001.parquet
    ...

This tool concatenates every shard that shares the same `{dump}` base into a
single file, preserving the schema (target, uri, target_context):

    word_context_{dump}.parquet

Each output is written to a temp file and atomically renamed, so merging in
place (output dir == input dir) can never truncate a shard before it's read.
Shards are streamed row-group by row-group, so memory stays bounded.

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
import logging
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List

import pyarrow.parquet as pq

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


# ---------------------------------------------------------------------------
# Grouping
# ---------------------------------------------------------------------------

def group_shards(input_dir: Path, pattern: str) -> Dict[str, List[Path]]:
    """Group Parquet files by their {dump} base name.

    A file named ``<base>.partNNN.parquet`` is grouped under ``<base>``; a file
    named ``<base>.parquet`` (already merged / single-worker) is grouped under
    ``<base>`` too, so re-running the merge is idempotent.
    """
    groups: Dict[str, List[Path]] = defaultdict(list)
    for p in sorted(input_dir.glob(pattern)):
        if not p.is_file():
            continue
        stem = p.name[:-8]  # strip ".parquet" (glob guarantees the suffix)
        m = _SHARD_RE.match(stem)
        base = m.group("base") if m else stem
        groups[base].append(p)
    return groups


def _is_shard(path: Path) -> bool:
    return bool(_SHARD_RE.match(path.name[:-8]))


# ---------------------------------------------------------------------------
# Merge
# ---------------------------------------------------------------------------

def merge_group(base: str, files: List[Path], output_dir: Path) -> int:
    """Concatenate `files` into `<output_dir>/<base>.parquet`. Returns row count.

    Writes to a temp file and atomically renames, so an in-place merge never
    truncates a source before it has been fully read. Sources are streamed
    row-group by row-group, so peak memory is one row group.
    """
    out = output_dir / f"{base}.parquet"
    tmp = out.with_name(out.name + ".tmp")

    rows = 0
    writer: pq.ParquetWriter | None = None
    try:
        for src in sorted(files):
            pf = pq.ParquetFile(src)
            if writer is None:
                writer = pq.ParquetWriter(str(tmp), pf.schema_arrow,
                                          compression="zstd")
            for batch in pf.iter_batches():
                writer.write_batch(batch)
                rows += batch.num_rows
    finally:
        if writer is not None:
            writer.close()

    if writer is None:
        # No source files at all; produce nothing rather than an empty file.
        tmp.unlink(missing_ok=True)
        return 0

    tmp.replace(out)
    return rows


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=("Merge sharded word-context Parquet files "
                     "(word_context_*.partNNN.parquet) into one file per {dump}."),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--input-dir", required=True, type=Path, dest="input_dir",
                   metavar="DIR",
                   help="Directory containing the shard Parquet files.")
    p.add_argument("--output-dir", type=Path, default=None, dest="output_dir",
                   metavar="DIR",
                   help="Directory for merged files (default: same as --input-dir).")
    p.add_argument("--pattern", default="word_context_*.parquet", metavar="GLOB",
                   help="Glob for shard files (default: word_context_*.parquet).")
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
                and files[0].name == f"{base}.parquet"):
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
