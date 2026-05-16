#!/usr/bin/env python3
"""
word_sample.py — Reservoir-sampled word list from Common Crawl WET/WARC files.

For each of N samples the algorithm:
  1. Picks a random WET file from the file list.
  2. Streams that file, selecting one word uniformly at random via
     reservoir-1 sampling.  Early stopping (--max-records) bounds how
     many WARC records are read per sample so runtime stays proportional
     to N rather than to the total corpus size.
  3. Records the word and the source filename.

Files chosen more than once (when N > number of available files) are read
a single time with a proportionally larger reservoir, so each unique file
is opened at most once regardless of N.

Output is a CSV with one row per sampled word.  The filename embeds the CC
crawl identifier (extracted automatically from file paths or overridden with
--cc-date) and the script run datetime, e.g.:

    word_sample_CC-MAIN-2024-10_20260516_143201.csv

Supports local files, glob patterns, gzip, S3 paths (s3://...), and Common
Crawl .paths.gz index files — the same inputs as word_count.py.

Usage:
    python word_sample.py /data/*.wet.gz --n 10000 --seed 42
    python word_sample.py /data/*.wet.gz --n 10000 --seed 42 --max-records 50
    python word_sample.py s3://commoncrawl/crawl-data/CC-MAIN-2024-10/.../file.wet.gz \\
        --n 50000 --seed 0
    python word_sample.py --file-list wet.paths.gz --n 200000 --seed 42
"""

import argparse
import csv
import glob
import gzip
import io
import logging
import os
import random
import re
import sys
import time
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Tuple

try:
    from warcio.archiveiterator import ArchiveIterator  # pyright: ignore[reportMissingImports]
except ImportError:
    print("ERROR: warcio is required. Install with: pip install warcio", file=sys.stderr)
    sys.exit(1)

try:
    import boto3  # pyright: ignore[reportMissingImports]
    from botocore.config import Config as BotocoreConfig  # pyright: ignore[reportMissingImports]
    from botocore.exceptions import BotoCoreError, ClientError  # pyright: ignore[reportMissingImports]
    HAS_BOTO3 = True
except ImportError:
    HAS_BOTO3 = False

try:
    from bs4 import BeautifulSoup  # pyright: ignore[reportMissingImports]
    HAS_BS4 = True
except ImportError:
    HAS_BS4 = False

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(line_buffering=True)  # pyright: ignore[reportAttributeAccessIssue]

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stderr)],
)
log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Tokenizer
# ---------------------------------------------------------------------------

_WORD_RE = re.compile(r"[a-zA-Z'''\-]+")


def tokenize(text: str, min_length: int = 1) -> List[str]:
    words = _WORD_RE.findall(text)
    if min_length > 1:
        return [w.lower() for w in words if len(w) >= min_length]
    return [w.lower() for w in words]


# ---------------------------------------------------------------------------
# CC crawl identifier extraction
# ---------------------------------------------------------------------------

_CC_CRAWL_RE = re.compile(r"CC-MAIN-\d{4}-\d{2}")


def extract_cc_crawl_id(paths: List[str]) -> str:
    """Extract the CC-MAIN-YYYY-WW identifier from the first matching path."""
    for p in paths:
        m = _CC_CRAWL_RE.search(p)
        if m:
            return m.group()
    return "unknown"


# ---------------------------------------------------------------------------
# Text extraction
# ---------------------------------------------------------------------------

def _extract_text(record, format_hint: str) -> Optional[str]:
    if format_hint == "wet" or record.rec_type == "conversion":
        if record.rec_type != "conversion":
            return None
        try:
            return record.content_stream().read().decode("utf-8", errors="replace")
        except Exception:
            return None

    if format_hint == "warc" or record.rec_type == "response":
        if record.rec_type != "response":
            return None
        content_type = record.http_headers.get_header("Content-Type", "") if record.http_headers else ""
        if "text/html" not in content_type and "text/plain" not in content_type:
            return None
        try:
            raw = record.content_stream().read().decode("utf-8", errors="replace")
        except Exception:
            return None
        if "text/html" in content_type:
            if HAS_BS4:
                soup = BeautifulSoup(raw, "lxml")
                for tag in soup(["script", "style", "noscript"]):
                    tag.decompose()
                return soup.get_text(separator=" ")
            return re.sub(r"<[^>]+>", " ", raw)
        return raw

    # Auto: try both record types
    if record.rec_type == "conversion":
        try:
            return record.content_stream().read().decode("utf-8", errors="replace")
        except Exception:
            return None
    if record.rec_type == "response":
        return _extract_text(record, "warc")
    return None


# ---------------------------------------------------------------------------
# File / S3 helpers
# ---------------------------------------------------------------------------

_S3_CLIENT_CONFIG = BotocoreConfig(
    retries={"mode": "adaptive", "max_attempts": 11}
) if HAS_BOTO3 else None

_S3_RETRY_BASE_SECS = 10
_S3_RETRY_MAX_SECS = 300
_S3_RETRY_MAX_ATTEMPTS = 10
_s3_client = None


def _get_s3_client():
    global _s3_client
    if _s3_client is None:
        _s3_client = boto3.client("s3", config=_S3_CLIENT_CONFIG)
    return _s3_client


def open_s3(s3_uri: str):
    if not HAS_BOTO3:
        log.error("boto3 is required for S3 support.")
        sys.exit(1)
    bucket, _, key = s3_uri[5:].partition("/")
    s3 = _get_s3_client()
    for attempt in range(1, _S3_RETRY_MAX_ATTEMPTS + 2):
        try:
            return s3.get_object(Bucket=bucket, Key=key)["Body"]
        except (BotoCoreError, ClientError) as exc:
            if attempt > _S3_RETRY_MAX_ATTEMPTS:
                log.error("Exhausted retries for %s: %s", s3_uri, exc)
                sys.exit(1)
            cap = min(_S3_RETRY_BASE_SECS * (2 ** (attempt - 1)), _S3_RETRY_MAX_SECS)
            delay = random.uniform(0, cap)
            log.warning("S3 retry %d for %s in %.1fs: %s", attempt, s3_uri, delay, exc)
            time.sleep(delay)


def is_s3_uri(path: str) -> bool:
    return path.startswith("s3://")


def expand_paths(raw_paths: List[str]) -> List[str]:
    expanded = []
    for p in raw_paths:
        if is_s3_uri(p):
            expanded.append(p)
        else:
            matched = sorted(glob.glob(p, recursive=True))
            expanded.extend(matched if matched else [p])
    return expanded


_CC_S3_BUCKET = "s3://commoncrawl"


def parse_paths_file(paths_file: str) -> Iterator[str]:
    log.info("Reading paths file: %s", paths_file)
    if is_s3_uri(paths_file):
        raw = open_s3(paths_file)
        if raw is None:
            raise ValueError(f"Could not open paths file: {paths_file}")
        byte_stream = io.BytesIO(raw.read())
    else:
        byte_stream = open(paths_file, "rb")
    with gzip.open(byte_stream, "rt", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            yield line if is_s3_uri(line) else f"{_CC_S3_BUCKET}/{line}"


# ---------------------------------------------------------------------------
# Core sampling: one file → k words
# ---------------------------------------------------------------------------

def _sample_words_from_file(
    path: str,
    k: int,
    seed: int,
    format_hint: str,
    min_length: int,
    max_records: Optional[int],
) -> List[str]:
    """Return k words sampled uniformly at random from a WET/WARC file.

    Uses reservoir sampling so a single streaming pass suffices.  Stops
    after ``max_records`` content records when set (early stopping for speed).
    Returns fewer than k words only if the file contains fewer than k tokens.
    """
    rng = random.Random(seed)
    reservoir: List[str] = []
    total_words = 0
    records_seen = 0

    stream = open_s3(path) if is_s3_uri(path) else open(path, "rb")
    try:
        for record in ArchiveIterator(stream):
            text = _extract_text(record, format_hint)
            if not text:
                continue
            records_seen += 1
            for word in tokenize(text, min_length=min_length):
                total_words += 1
                if len(reservoir) < k:
                    reservoir.append(word)
                else:
                    j = rng.randint(0, total_words - 1)
                    if j < k:
                        reservoir[j] = word
            if max_records is not None and records_seen >= max_records:
                break
    except Exception as exc:
        log.error("Error processing %s: %s", path, exc)
    finally:
        if stream and hasattr(stream, "close"):
            stream.close()

    return reservoir


def _worker(args: tuple) -> Tuple[str, List[str]]:
    """Top-level worker function (must be module-level for pickling)."""
    path, k, seed, format_hint, min_length, max_records = args
    words = _sample_words_from_file(path, k, seed, format_hint, min_length, max_records)
    return path, words


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------

def write_sample_csv(rows: List[Tuple[str, str]], output_path: str) -> None:
    with open(output_path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(["word", "source"])
        writer.writerows(rows)
    log.info("Wrote %d sampled words to %s", len(rows), output_path)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Randomly sample N words from Common Crawl WET/WARC files.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Runtime scales with N, not with corpus size: for each sample the script
picks one random WET file and reads at most --max-records records from it.

Examples:
  # 10 000 words from local files, stop after 50 records per file
  python word_sample.py /data/*.wet.gz --n 10000 --seed 42 --max-records 50

  # 50 000 words from S3, full-file reservoir sampling (slower, unbiased)
  python word_sample.py s3://commoncrawl/crawl-data/CC-MAIN-2024-10/.../file.wet.gz \\
      --n 50000 --seed 0

  # Using a CC paths index file
  python word_sample.py --file-list wet.paths.gz --n 200000 --seed 42 --max-records 100
        """,
    )
    parser.add_argument(
        "files", nargs="*",
        help="WET/WARC file paths (local or s3://), supports glob patterns.",
    )
    parser.add_argument(
        "--file-list", metavar="PATHS_FILE", default=None,
        help="Gzip-compressed Common Crawl paths file (.paths.gz), local or s3://.",
    )
    parser.add_argument(
        "--n", type=int, required=True, metavar="N",
        help="Number of words to sample.",
    )
    parser.add_argument(
        "--seed", type=int, default=None, metavar="SEED",
        help="Random seed for reproducibility (default: random).",
    )
    parser.add_argument(
        "--max-records", type=int, default=None, metavar="K",
        dest="max_records",
        help="Stop after reading K content records per selected file.  "
             "Lower values are faster but only sample from each file's early records.  "
             "Omit to read the full file (unbiased, slower).",
    )
    parser.add_argument(
        "--output", "-o", default=None,
        help="Output CSV path. Defaults to word_sample_<cc-id>_<datetime>.csv.",
    )
    parser.add_argument(
        "--cc-date", default=None, metavar="ID",
        help="Override the CC crawl identifier in the output filename "
             "(e.g. CC-MAIN-2024-10). Auto-detected from file paths if omitted.",
    )
    parser.add_argument(
        "--min-length", type=int, default=1, metavar="N",
        help="Minimum word length to include (default: 1).",
    )
    parser.add_argument(
        "--type", dest="format_hint", choices=["wet", "warc", "auto"], default="auto",
        help="File type hint: wet | warc | auto (default: auto).",
    )
    parser.add_argument(
        "--workers", "-w", type=int, default=None, metavar="N",
        help="Parallel worker processes (default: os.cpu_count()).",
    )
    parser.add_argument(
        "--verbose", "-v", action="store_true",
        help="Enable debug logging.",
    )
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    if args.verbose:
        log.setLevel(logging.DEBUG)

    # Build file list
    file_list: List[str] = []
    if args.file_list:
        file_list.extend(parse_paths_file(args.file_list))
    if args.files:
        file_list.extend(expand_paths(args.files))
    if not file_list:
        parser.error("Provide at least one WET/WARC file or use --file-list.")

    log.info("Files available : %d", len(file_list))

    # Determine output path
    run_ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    cc_id = args.cc_date or extract_cc_crawl_id(file_list)
    output_path = args.output or f"word_sample_{cc_id}_{run_ts}.csv"

    log.info("Sample size     : %d", args.n)
    log.info("Seed            : %s", args.seed if args.seed is not None else "random")
    log.info("Max records/file: %s", args.max_records if args.max_records is not None else "unlimited")
    log.info("CC crawl        : %s", cc_id)
    log.info("Output          : %s", output_path)

    # ── Step 1: Select N file indices with seeded RNG ─────────────────────────
    master_rng = random.Random(args.seed)
    selections = [master_rng.randrange(len(file_list)) for _ in range(args.n)]

    # ── Step 2: Group selections by file (read each unique file once) ─────────
    # file_path → number of words needed from it
    counts: Dict[str, int] = defaultdict(int)
    for idx in selections:
        counts[file_list[idx]] += 1

    unique_files = len(counts)
    log.info(
        "Unique files to read: %d  (%.1f%% of available, avg %.1f words/file)",
        unique_files,
        100.0 * unique_files / len(file_list),
        args.n / unique_files,
    )

    # Per-file seeds derived deterministically from master seed + file path
    def _file_seed(path: str) -> int:
        return master_rng.randint(0, 2**31) ^ hash(path) & 0x7FFFFFFF

    tasks = [
        (path, k, _file_seed(path), args.format_hint, args.min_length, args.max_records)
        for path, k in counts.items()
    ]

    # ── Step 3: Process files in parallel ─────────────────────────────────────
    # words_by_file[path] = [word, word, ...]
    words_by_file: Dict[str, List[str]] = {}
    n_workers = args.workers or os.cpu_count()
    log.info("Using %d worker process(es).", n_workers)

    with ProcessPoolExecutor(max_workers=n_workers) as executor:
        future_to_path = {executor.submit(_worker, t): t[0] for t in tasks}
        completed = 0
        for future in as_completed(future_to_path):
            path = future_to_path[future]
            completed += 1
            try:
                _, words = future.result()
            except Exception as exc:
                log.error("Worker failed for %s: %s", path, exc)
                words = []
            words_by_file[path] = words
            log.info(
                "[%d/%d] %s — got %d/%d words",
                completed, unique_files, Path(path).name,
                len(words), counts[path],
            )

    # ── Step 4: Reconstruct sample in original selection order ────────────────
    # Track how many words we've consumed from each file so far
    consumed: Dict[str, int] = defaultdict(int)
    rows: List[Tuple[str, str]] = []
    missing = 0

    for idx in selections:
        path = file_list[idx]
        source = Path(path).name
        file_words = words_by_file.get(path, [])
        pos = consumed[path]
        if pos < len(file_words):
            rows.append((file_words[pos], source))
            consumed[path] += 1
        else:
            missing += 1

    if missing:
        log.warning(
            "%d sample(s) could not be filled (file had too few tokens). "
            "Try reducing --max-records or using larger files.",
            missing,
        )

    # Shuffle the output so selection order (file grouping) isn't visible
    master_rng.shuffle(rows)

    log.info(
        "Sampling complete. Requested: %d | Collected: %d | Missing: %d",
        args.n, len(rows), missing,
    )

    write_sample_csv(rows, output_path)
    log.info("Done.")


if __name__ == "__main__":
    main()
