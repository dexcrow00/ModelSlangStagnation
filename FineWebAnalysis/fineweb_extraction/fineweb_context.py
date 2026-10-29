#!/usr/bin/env python3
"""
fineweb_context.py — Extract target-word context windows from a pre-sampled
FineWeb HuggingFace dataset config.

Reads a FineWeb sample config (sample/10BT, sample/100BT, sample/350BT) in a
single streaming pass. The sample parquet files are flat — rows from many CC
dumps are interleaved — so each match is routed to a per-dump output Parquet
file based on its `dump` column. Files are read in row-group batches (text/url/
dump columns only). A vectorised regex pre-filter skips documents containing no
target before any Python tokenisation; every match in a surviving document is
written.

Output Parquet (one per dump, e.g. word_context_CC-MAIN-2024-10.parquet):
  target         — the matched target word/phrase (filter with WHERE target=...)
  uri            — source URL of the document
  target_context — ±K-token window of the ORIGINAL text around the match
                   (casing/punctuation preserved for the downstream classifier)

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
    # each worker writes its own .partNNN.parquet shard per dump.
    python fineweb_context.py --output-dir contexts/ --num-workers 4 --worker-id 0
    python fineweb_context.py --output-dir contexts/ --num-workers 4 --worker-id 1
    ...

Requires: pyarrow, huggingface_hub
"""

from __future__ import annotations

import argparse
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq
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

    Matching is done on normalised tokens (whitespace-split, lowercased, with
    leading/trailing punctuation stripped); each token is only compared against
    the targets sharing its first character (see compile_targets). Single-word
    targets match directly; multiword targets compare the joined slice. Advances
    past matched words to avoid double-counting.

    The emitted context is sliced from the ORIGINAL ``text`` (not the normalised
    tokens), so casing, punctuation and internal spacing are preserved for the
    downstream classifier. Token character spans are tracked to map the matched
    ±k-token window back onto the raw string.
    """
    # (char_start, char_end) of each whitespace-delimited token in the original.
    spans = [(m.start(), m.end()) for m in re.finditer(r"\S+", text)]
    words = [text[s:e].strip(string.punctuation).lower() for s, e in spans]
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
                    results.append((target, text[spans[start][0]:spans[end - 1][1]]))
                    i += n
                    break
            else:
                i += 1
        else:
            i += 1
    return results


# ---------------------------------------------------------------------------
# Parquet output
# ---------------------------------------------------------------------------

# Output schema: one row per match, files partitioned by dump. Downstream
# filtering for a single word is just `WHERE target = '...'`.
_SCHEMA = pa.schema([
    ("target", pa.string()),
    ("uri", pa.string()),
    ("target_context", pa.string()),
])

# Rows buffered per dump before a row group is flushed to its ParquetWriter.
_FLUSH_ROWS = 2000


# ---------------------------------------------------------------------------
# Vectorised document pre-filter
# ---------------------------------------------------------------------------

def build_prefilter(targets: List[str]) -> Optional[str]:
    """Build one RE2 regex that matches any document containing a target.

    Used with ``pyarrow.compute.match_substring_regex`` to compute a boolean
    mask over a whole text column at once, so documents with zero targets are
    discarded before any per-row Python tokenisation.

    The pattern mirrors the tokeniser's semantics exactly, so it is a true
    superset of ``extract_contexts`` (no false negatives):
      * ``(?:^|\\s)`` + ``[[:punct:]]*`` allows the leading punctuation the
        tokeniser strips (incl. ``_``, which ``\\b`` would treat as a word char);
      * internal whitespace in multiword targets becomes ``\\s+``;
      * ``[[:punct:]]*`` + ``(?:\\s|$)`` allows trailing punctuation/whitespace.
    Case-insensitive via the inline ``(?i)`` flag.
    """
    alts = [r"\s+".join(re.escape(w) for w in t.split()) for t in targets if t]
    if not alts:
        return None
    return r"(?i)(?:^|\s)[[:punct:]]*(?:" + "|".join(alts) + r")[[:punct:]]*(?:\s|$)"


# ---------------------------------------------------------------------------
# Dump filter
# ---------------------------------------------------------------------------

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
            "The sample is read in a single streaming pass and each match is routed "
            "to a per-dump output Parquet file by its `dump` column."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    # Output
    p.add_argument("--output-dir", required=True, metavar="DIR", dest="output_dir",
                   type=Path,
                   help="Directory for output Parquet files (one per dump).")

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
    prefilter = build_prefilter(targets)
    if prefilter is None:
        log.error("No usable target words in %s — nothing to do.", args.words)
        return

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

    # One Parquet file per dump (target is a column). The sample is flat (rows
    # from many dumps interleaved) so writers are created lazily, keyed by dump.
    # The number of distinct dumps is small (~100), so all handles stay open for
    # the run. Rows are buffered per dump and flushed as parquet row groups.
    writers: Dict[str, pq.ParquetWriter] = {}      # dump_id -> open ParquetWriter
    buffers: Dict[str, Tuple[List[str], List[str], List[str]]] = {}  # dump -> cols
    created: set = set()               # output paths created this run
    match_cache: Dict[str, bool] = {}  # dump_id -> passes filters (memoised)

    def _dump_ok(dump_id: str) -> bool:
        ok = match_cache.get(dump_id)
        if ok is None:
            ok = _dump_matches(dump_id, args.since, args.from_dump, args.to_dump)
            match_cache[dump_id] = ok
        return ok

    def _flush_dump(dump_id: str) -> None:
        """Write the buffered rows for one dump as a parquet row group."""
        buf = buffers.get(dump_id)
        if not buf or not buf[0]:
            return
        table = pa.table({"target": buf[0], "uri": buf[1], "target_context": buf[2]},
                         schema=_SCHEMA)
        w = writers.get(dump_id)
        if w is None:
            out = args.output_dir / f"word_context_{dump_id}{shard_suffix}.parquet"
            w = pq.ParquetWriter(str(out), _SCHEMA, compression="zstd")
            writers[dump_id] = w
            created.add(out)
        w.write_table(table)
        buf[0].clear(); buf[1].clear(); buf[2].clear()

    docs_seen = 0     # documents scanned (before pre-filter)
    cand_docs = 0     # documents surviving the regex pre-filter
    rows_total = 0
    last_log = 0
    read_secs = 0.0   # time producing/decoding parquet batches (I/O + decompress)
    match_secs = 0.0  # time pre-filtering + extract_contexts + buffering (CPU)

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
                    read_secs += time.monotonic() - t_read

                    t_match = time.monotonic()
                    docs_seen += batch.num_rows
                    # Vectorised pre-filter: keep only docs that contain a target
                    # (nulls in the text column → null mask → dropped by filter).
                    mask = pc.match_substring_regex(batch.column("text"), prefilter)
                    sub = batch.filter(mask)
                    cand_docs += sub.num_rows
                    cols = sub.to_pydict()
                    for text, url, dump in zip(cols["text"], cols["url"], cols["dump"]):
                        if not _dump_ok(dump):
                            continue
                        for target, context in extract_contexts(
                                text, compiled, args.context_window):
                            buf = buffers.get(dump)
                            if buf is None:
                                buf = ([], [], [])
                                buffers[dump] = buf
                            buf[0].append(target)
                            buf[1].append(url or "")
                            buf[2].append(context)
                            rows_total += 1
                            if len(buf[0]) >= _FLUSH_ROWS:
                                _flush_dump(dump)
                    match_secs += time.monotonic() - t_match

                    if docs_seen - last_log >= 100000:
                        log.info("  docs scanned: %d | candidates: %d | rows written: %d",
                                 docs_seen, cand_docs, rows_total)
                        last_log = docs_seen
                    t_read = time.monotonic()
    finally:
        for dump_id in list(buffers):
            _flush_dump(dump_id)
        for w in writers.values():
            w.close()

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
        f"candidate docs  : {cand_docs}",
        f"read/decompress : {read_secs:.1f}s",
        f"matching        : {match_secs:.1f}s",
    ]
    summary_path = args.output_dir / f"run_summary{shard_suffix}.txt"
    summary_path.write_text("\n".join(summary_lines) + "\n", encoding="utf-8")
    log.info("Wrote run summary → %s", summary_path)


if __name__ == "__main__":
    main()
