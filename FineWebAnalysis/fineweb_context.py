#!/usr/bin/env python3
"""
fineweb_context.py — Extract target-word context windows from a pre-sampled
FineWeb HuggingFace dataset config.

Reads a FineWeb sample config (sample/10BT, sample/100BT, sample/350BT) in a
single streaming pass. The sample parquet files are flat — rows from many CC
dumps are interleaved — so each row is routed to a per-dump output CSV based on
its `dump` column. Files are read in row-group batches (text/url/dump columns
only). Every matching row is written.

Output CSV (one per dump per target word, e.g. word_context_CC-MAIN-2024-10__lol.csv):
  uri            — source URL of the document
  target_context — ±K-token window of text around the match (includes the match)

Usage:
    # All dumps in the default sample
    python fineweb_context.py --output-dir contexts/ --words target_words.txt

    # Larger sample, from 2020 onwards
    python fineweb_context.py --output-dir contexts/ --words target_words.txt \\
        --sample sample-100BT --since 2020

    # Explicit dump range
    python fineweb_context.py --output-dir contexts/ --words target_words.txt \\
        --from-dump CC-MAIN-2022-05 --to-dump CC-MAIN-2024-10

    # Parallel workers (e.g. 4-way): run one process per worker-id, 0..3.
    # Parquet files are partitioned round-robin so each is processed once;
    # each worker writes its own .partNNN.csv shard per dump.
    python fineweb_context.py --output-dir contexts/ --num-workers 4 --worker-id 0
    python fineweb_context.py --output-dir contexts/ --num-workers 4 --worker-id 1
    ...

Requires: datasets, huggingface_hub
"""

from __future__ import annotations

import argparse
import csv
import pyarrow.parquet as pq
from huggingface_hub import HfFileSystem
import logging
import re
import string
import sys
import time
from collections import OrderedDict
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

def compile_targets(targets: List[str]) -> Dict[str, List[Tuple[str, int]]]:
    """Bucket targets by first character and precompute word counts once.

    Returns {first_char: [(target, word_count), ...]}.  At match time each word
    only needs to compare against the handful of targets sharing its first
    character, instead of scanning the entire target list.
    """
    compiled: Dict[str, List[Tuple[str, int]]] = {}
    for t in targets:
        if not t:
            continue
        compiled.setdefault(t[0], []).append((t, len(t.split())))
    return compiled


def extract_contexts(
    text: str,
    compiled: Dict[str, List[Tuple[str, int]]],
    k: int,
) -> List[Tuple[str, str]]:
    """Find all target matches and return (matched_target, context_string) pairs.

    Splits the text on whitespace and, for each word, only compares against the
    targets that share its first character (see compile_targets). Single-word
    targets are matched directly; multiword targets compare the joined slice.
    Advances past matched words to avoid double-counting.
    """
    words = [w.strip(string.punctuation) for w in text.lower().split()]
    results: List[Tuple[str, str]] = []
    n_words = len(words)
    i = 0
    while i < n_words:
        w = words[i]
        candidates = compiled.get(w[:1])
        if candidates:
            for target, n in candidates:
                if (w == target) if n == 1 else (" ".join(words[i:i + n]) == target):
                    start = max(0, i - k)
                    end   = min(n_words, i + n + k)
                    results.append((target, " ".join(words[start:end])))
                    i += n
                    break
            else:
                i += 1
        else:
            i += 1
    return results


# ---------------------------------------------------------------------------
# CSV helpers
# ---------------------------------------------------------------------------

_FIELDNAMES = ["uri", "target_context"]

# Cap on simultaneously-open CSV writers. With one file per (dump, target) the
# total number of files is large, so we keep at most this many handles open at
# once (LRU eviction) and reopen evicted files in append mode as needed. Stays
# well under the typical 1024 open-file ulimit.
_MAX_OPEN_WRITERS = 512


def _safe_target(target: str) -> str:
    """Make a filesystem-safe filename component from a target word/phrase.

    Targets are already lowercased; spaces/punctuation collapse to underscores
    (e.g. "set fire" -> "set_fire").
    """
    safe = re.sub(r"[^a-z0-9]+", "_", target).strip("_")
    return safe or "x"


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

_REPO_PREFIX = "datasets/HuggingFaceFW/fineweb/"


def _sample_repo_dir(sample: str) -> str:
    """Map a --sample value to its repo-relative directory.

    The FineWeb sample configs live under ``sample/<size>`` (e.g. ``sample/10BT``),
    so ``sample-10BT`` → ``sample/10BT``.
    """
    return "sample/" + sample.removeprefix("sample-")


def _list_sample_files(sample: str) -> List[str]:
    """Return the repo-relative parquet paths for the given FineWeb sample config.

    Sample files are flat (e.g. ``sample/10BT/000_00000.parquet``) and are NOT
    organised by CC dump — every file mixes rows from many dumps, so dump
    filtering happens per row via the ``dump`` column at read time.
    """
    fs = HfFileSystem(token=HF_TOKEN)
    sample_dir = _sample_repo_dir(sample)
    pattern = f"{_REPO_PREFIX}{sample_dir}/*.parquet"

    try:
        files = fs.glob(pattern)
    except Exception as exc:
        log.error("Failed to list sample files (%s): %s", pattern, exc)
        sys.exit(1)

    rel_files = sorted(f.removeprefix(_REPO_PREFIX) for f in files
                       if f.endswith(".parquet"))
    if not rel_files:
        log.error(
            "Could not find any parquet files for sample '%s' at .../%s. "
            "Verify the dataset name and your network connection.",
            sample, sample_dir,
        )
        sys.exit(1)

    log.info("Found %d parquet files in sample '%s' (.../%s)",
             len(rel_files), sample, sample_dir)
    return rel_files


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
            "The sample is read in a single streaming pass and each matching row is "
            "routed to a per-dump output CSV by its `dump` column."
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

    # Parquet batch size
    p.add_argument("--batch-size", type=int, default=1000, metavar="N", dest="batch_size",
                   help="Rows decoded per parquet row-group batch (default: 1000).")

    # Dump range filters
    p.add_argument("--since", type=int, default=None, metavar="YEAR",
                   help="Only collect rows for dumps from this year onwards (e.g. 2020).")
    p.add_argument("--from-dump", default=None, metavar="ID", dest="from_dump",
                   help="Only collect rows for dumps at or after this ID "
                        "(e.g. CC-MAIN-2020-05).")
    p.add_argument("--to-dump", default=None, metavar="ID", dest="to_dump",
                   help="Only collect rows for dumps up to and including this ID.")

    # Parallel workers (parquet-file sharding)
    p.add_argument("--num-workers", type=int, default=1, metavar="N", dest="num_workers",
                   help="Total number of parallel workers sharing this dataset "
                        "(default: 1). The sample's parquet files are partitioned "
                        "round-robin across workers so each file is processed once.")
    p.add_argument("--worker-id", type=int, default=0, metavar="I", dest="worker_id",
                   help="This worker's index in [0, num-workers). Processes files "
                        "files[worker_id::num_workers] and writes its own shard of "
                        "output files (default: 0).")

    return p


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    if args.num_workers < 1:
        parser.error("--num-workers must be >= 1")
    if not (0 <= args.worker_id < args.num_workers):
        parser.error("--worker-id must be in [0, num-workers)")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    targets = load_target_words(args.words)
    compiled = compile_targets(targets)

    # Precompute a filesystem-safe filename component per target (output is one
    # CSV per dump per target), warning on any name collisions.
    target_files: Dict[str, str] = {}
    safe_seen: Dict[str, str] = {}
    for t in targets:
        safe = _safe_target(t)
        prev = safe_seen.get(safe)
        if prev is not None and prev != t:
            log.warning("Target filename collision: %r and %r both map to '%s' — "
                        "their contexts will share one file.", prev, t, safe)
        safe_seen.setdefault(safe, t)
        target_files[t] = safe

    # ── Step 1: enumerate the sample's (flat) parquet files ───────────────────
    log.info("Enumerating parquet files in FineWeb '%s' ...", args.sample)
    all_files = _list_sample_files(args.sample)

    # Partition files round-robin across workers so each file is processed by
    # exactly one worker — no coordination needed and no duplicated work.
    sample_files = all_files[args.worker_id::args.num_workers]

    # Worker-unique output filenames so workers never write the same path.
    # (Each dump's rows are spread across files/workers, so per-worker outputs
    # are partial shards that you concatenate afterwards.)
    shard_suffix = f".part{args.worker_id:03d}" if args.num_workers > 1 else ""

    log.info("  output-dir     : %s", args.output_dir)
    log.info("  context-window : %d", args.context_window)
    if args.num_workers > 1:
        log.info("  worker         : %d of %d → %d of %d files",
                 args.worker_id, args.num_workers, len(sample_files), len(all_files))
    if args.since or args.from_dump or args.to_dump:
        log.info("  dump filter    : since=%s from=%s to=%s",
                 args.since, args.from_dump, args.to_dump)

    if not sample_files:
        log.warning("No files assigned to worker %d/%d — nothing to do.",
                    args.worker_id, args.num_workers)
        return

    t0 = time.monotonic()
    started_at = time.strftime("%Y-%m-%d %H:%M:%S")
    fs = HfFileSystem(token=HF_TOKEN)

    # One CSV per (dump, target). The sample is flat (rows from many dumps
    # interleaved) so writers are created lazily and keyed by (dump, target).
    # Open handles are bounded by an LRU: when the cap is hit the least-recently
    # used writer is closed, and a reopened file is appended to (header already
    # written), so we never exceed the OS open-file limit.
    writers: "OrderedDict[Tuple[str, str], dict]" = OrderedDict()
    created: set = set()               # output paths opened this run (header written)
    match_cache: Dict[str, bool] = {}  # dump_id -> passes filters (memoised)

    def _dump_ok(dump_id: str) -> bool:
        ok = match_cache.get(dump_id)
        if ok is None:
            ok = _dump_matches(dump_id, args.since, args.from_dump, args.to_dump)
            match_cache[dump_id] = ok
        return ok

    def _writer_for(dump_id: str, target: str):
        """Return (creating/reopening if needed) the writer for one (dump, target)."""
        key = (dump_id, target)
        st = writers.get(key)
        if st is not None:
            writers.move_to_end(key)
            return st
        if len(writers) >= _MAX_OPEN_WRITERS:
            _, old = writers.popitem(last=False)   # evict least-recently-used
            old["fh"].close()
        out = args.output_dir / f"word_context_{dump_id}__{target_files[target]}{shard_suffix}.csv"
        if out in created:
            fh = out.open("a", newline="", encoding="utf-8")
            w = csv.DictWriter(fh, fieldnames=_FIELDNAMES)
        else:
            fh = out.open("w", newline="", encoding="utf-8")
            w = csv.DictWriter(fh, fieldnames=_FIELDNAMES)
            w.writeheader()
            created.add(out)
        st = {"fh": fh, "writer": w}
        writers[key] = st
        return st

    docs_seen = 0
    rows_total = 0
    read_secs = 0.0   # time producing/decoding parquet batches (I/O + decompress)
    match_secs = 0.0  # time routing + extract_contexts + writerow (CPU)

    # ── Step 2: single streaming pass over the sample ─────────────────────────
    try:
        for file_idx, rel_path in enumerate(sample_files, 1):
            log.info("[%d/%d] Reading %s ...", file_idx, len(sample_files),
                     rel_path.rsplit("/", 1)[-1])
            with fs.open(f"{_REPO_PREFIX}{rel_path}") as raw:
                pf = pq.ParquetFile(raw)
                t_read = time.monotonic()
                for batch in pf.iter_batches(batch_size=args.batch_size,
                                             columns=["text", "url", "dump"]):
                    cols = batch.to_pydict()
                    read_secs += time.monotonic() - t_read

                    t_match = time.monotonic()
                    for text, url, dump in zip(cols["text"], cols["url"], cols["dump"]):
                        docs_seen += 1
                        if docs_seen % 10000 == 0:
                            log.info("  docs scanned: %d | rows written: %d",
                                     docs_seen, rows_total)
                        if not text or not _dump_ok(dump):
                            continue
                        for target, context in extract_contexts(
                                text, compiled, args.context_window):
                            st = _writer_for(dump, target)
                            st["writer"].writerow({"uri": url or "",
                                                   "target_context": context})
                            rows_total += 1
                            if rows_total % 10000 == 0:
                                st["fh"].flush()  # push buffered rows to disk
                    match_secs += time.monotonic() - t_match
                    t_read = time.monotonic()
    finally:
        for st in writers.values():
            st["fh"].close()

    elapsed = _fmt_duration(time.monotonic() - t0)
    log.info("  timing: read/decompress %.1fs | matching %.1fs",
             read_secs, match_secs)
    log.info(
        "All done. %d output file(s), %d rows, %d docs scanned  [%s]",
        len(created), rows_total, docs_seen, elapsed,
    )

    # Persist the final run summary to a (per-worker) plain-text file.
    summary_lines = [
        "fineweb_context.py run summary",
        f"started         : {started_at}",
        f"finished        : {time.strftime('%Y-%m-%d %H:%M:%S')}",
        f"elapsed         : {elapsed}",
        f"sample          : {args.sample}",
        f"words file      : {args.words}",
        f"target words    : {len(targets)}",
        f"context-window  : {args.context_window}",
    ]
    if args.num_workers > 1:
        summary_lines.append(f"worker          : {args.worker_id} of {args.num_workers}")
        summary_lines.append(f"files processed : {len(sample_files)} of {len(all_files)}")
    else:
        summary_lines.append(f"files processed : {len(sample_files)}")
    if args.since or args.from_dump or args.to_dump:
        summary_lines.append(
            f"dump filter     : since={args.since} from={args.from_dump} to={args.to_dump}")
    summary_lines += [
        f"output files    : {len(created)}",
        f"rows written    : {rows_total}",
        f"docs scanned    : {docs_seen}",
        f"read/decompress : {read_secs:.1f}s",
        f"matching        : {match_secs:.1f}s",
    ]
    summary_path = args.output_dir / f"run_summary{shard_suffix}.txt"
    summary_path.write_text("\n".join(summary_lines) + "\n", encoding="utf-8")
    log.info("Wrote run summary → %s", summary_path)


if __name__ == "__main__":
    main()
