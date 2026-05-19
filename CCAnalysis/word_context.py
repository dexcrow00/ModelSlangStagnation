#!/usr/bin/env python3
"""
word_context.py — Sample URIs from Common Crawl WET files and extract
context windows around target words/phrases.

For each sampled URI the script finds every occurrence of a target word or
phrase in the page text and saves the surrounding ±K tokens as a row in
the output CSV.

Output CSV (one per dump):
  uri            — WARC-Target-URI of the source page
  target_context — ±K-token window of text around the match (includes the match)

Single-dump mode (explicit file list):
    python word_context.py \\
        --file-list s3://commoncrawl/crawl-data/CC-MAIN-2024-10/wet.paths.gz \\
        --cc-date CC-MAIN-2024-10 --n-uris 500 --words target_words.txt \\
        --output-dir contexts/

Multi-crawl mode (fetches CC index automatically):
    python word_context.py --since 2020 --n-uris 500 \\
        --words target_words.txt --output-dir contexts/
    python word_context.py --from-crawl CC-MAIN-2019-04 --to-crawl CC-MAIN-2022-49 \\
        --n-uris 1000 --context-window 15 --words target_words.txt --output-dir contexts/
"""

from __future__ import annotations

import argparse
import csv
import glob
import gzip
import io
import json
import logging
import math
import os
import random
import re
import sys
import time
import urllib.request
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterator, List, NamedTuple, Optional, Set, Tuple

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

_WORD_RE = re.compile(r"[a-zA-Z''’\-]+")


def tokenize(text: str) -> List[str]:
    return [w.lower() for w in _WORD_RE.findall(text)]


# ---------------------------------------------------------------------------
# Target words / phrases
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
# Context extraction
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
            end = min(len(tokens), i + matched_len + k)
            context = " ".join(tokens[start:end])
            results.append((matched_key, context))
            i += matched_len
        else:
            i += 1
    return results


# ---------------------------------------------------------------------------
# S3 / file helpers
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


def is_s3_uri(path: str) -> bool:
    return path.startswith("s3://")


def _parse_s3_uri(uri: str) -> Tuple[str, str]:
    without_scheme = uri[5:]
    bucket, _, key = without_scheme.partition("/")
    return bucket, key


def open_s3(s3_uri: str):
    if not HAS_BOTO3:
        raise RuntimeError("boto3 is required for S3 support. pip install boto3")
    bucket, key = _parse_s3_uri(s3_uri)
    s3 = _get_s3_client()
    for attempt in range(1, _S3_RETRY_MAX_ATTEMPTS + 2):
        try:
            return s3.get_object(Bucket=bucket, Key=key)["Body"]
        except (BotoCoreError, ClientError) as exc:
            if attempt > _S3_RETRY_MAX_ATTEMPTS:
                raise RuntimeError(f"Exhausted S3 retries for {s3_uri}") from exc
            cap = min(_S3_RETRY_BASE_SECS * (2 ** (attempt - 1)), _S3_RETRY_MAX_SECS)
            delay = random.uniform(0, cap)
            log.warning("S3 error attempt %d/%d for %s: %s — retrying in %.1fs",
                        attempt, _S3_RETRY_MAX_ATTEMPTS, s3_uri, exc, delay)
            time.sleep(delay)


def expand_paths(raw_paths: List[str]) -> List[str]:
    expanded: List[str] = []
    for p in raw_paths:
        if is_s3_uri(p):
            expanded.append(p)
        else:
            matched = sorted(glob.glob(p, recursive=True))
            expanded.extend(matched or [p])
    return expanded


_CC_S3_BUCKET = "s3://commoncrawl"


def parse_paths_file(paths_file: str) -> Iterator[str]:
    log.debug("Reading paths file: %s", paths_file)
    if is_s3_uri(paths_file):
        raw_stream = open_s3(paths_file)
        byte_stream = io.BytesIO(raw_stream.read())
    else:
        byte_stream = open(paths_file, "rb")  # type: ignore[assignment]
    with gzip.open(byte_stream, "rt", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            yield line if is_s3_uri(line) else f"{_CC_S3_BUCKET}/{line}"


# ---------------------------------------------------------------------------
# Per-file worker (module-level so ProcessPoolExecutor can pickle it)
# ---------------------------------------------------------------------------

def _sample_file_contexts(
    path: str,
    n_records: int,
    targets: Targets,
    context_window: int,
    seed: int,
    max_records: Optional[int],
) -> List[Dict[str, str]]:
    """Reservoir-sample n_records URIs from one WET file, extract context rows."""
    rng = random.Random(seed)
    reservoir: List[Tuple[str, str]] = []  # (uri, text)
    seen = 0

    stream = open_s3(path) if is_s3_uri(path) else open(path, "rb")
    try:
        for record in ArchiveIterator(stream):
            if record.rec_type != "conversion":
                continue
            lang = record.rec_headers.get_header("WARC-Identified-Content-Language")
            if lang and "eng" not in lang.lower().split(","):
                continue
            try:
                text = record.content_stream().read().decode("utf-8", errors="replace")
            except Exception:
                continue
            if not text.strip():
                continue

            uri = record.rec_headers.get_header("WARC-Target-URI") or ""
            seen += 1

            if len(reservoir) < n_records:
                reservoir.append((uri, text))
            else:
                j = rng.randint(0, seen - 1)
                if j < n_records:
                    reservoir[j] = (uri, text)

            if max_records is not None and seen >= max_records:
                break
    except Exception as exc:
        log.error("Error reading %s: %s", path, exc)
    finally:
        if hasattr(stream, "close"):
            stream.close()

    rows: List[Dict[str, str]] = []
    for uri, text in reservoir:
        tokens = tokenize(text)
        for _word, context in extract_contexts(tokens, targets, context_window):
            rows.append({"uri": uri, "target_context": context})
    return rows


# ---------------------------------------------------------------------------
# Dump-level orchestration
# ---------------------------------------------------------------------------

_CC_CRAWL_RE = re.compile(r"CC-MAIN-\d{4}-\d{2,4}")


def _infer_cc_id(file_list: List[str]) -> Optional[str]:
    for path in file_list:
        m = _CC_CRAWL_RE.search(path)
        if m:
            return m.group(0)
    return None


_FIELDNAMES = ["uri", "target_context"]


def _read_existing_rows(path: Path) -> Tuple[List[Dict[str, str]], Set[str]]:
    """Read an existing output CSV. Returns (rows, set_of_unique_uris)."""
    rows: List[Dict[str, str]] = []
    uris: Set[str] = set()
    with path.open(newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            rows.append(row)
            uris.add(row["uri"])
    return rows, uris


def _write_rows(path: Path, rows: List[Dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=_FIELDNAMES)
        writer.writeheader()
        writer.writerows(rows)


def process_dump(
    file_list: List[str],
    n_uris: int,
    n_files: Optional[int],
    targets: Targets,
    context_window: int,
    seed: int,
    workers: Optional[int],
    max_records: Optional[int],
    cc_id: str,
    exclude_uris: Optional[Set[str]] = None,
) -> List[Dict[str, str]]:
    """Sample URIs from a dump and return context rows.

    exclude_uris: rows whose URI is in this set are dropped from the result,
    useful when topping up an existing file.
    """
    rng = random.Random(seed)
    n_workers = workers or os.cpu_count() or 1

    default_files = max(n_workers, math.ceil(n_uris / 100))
    n_files_actual = min(
        n_files if n_files is not None else default_files,
        len(file_list),
    )
    selected = rng.sample(file_list, n_files_actual)
    n_per_file = math.ceil(n_uris / n_files_actual)
    file_seeds = [rng.randint(0, 2**31 - 1) for _ in selected]

    log.debug("%s — sampling %d URIs from %d/%d files (%d per file)",
              cc_id, n_uris, n_files_actual, len(file_list), n_per_file)

    all_rows: List[Dict[str, str]] = []

    with ProcessPoolExecutor(max_workers=n_workers) as executor:
        futures = {
            executor.submit(
                _sample_file_contexts, path, n_per_file, targets,
                context_window, fseed, max_records,
            ): path
            for path, fseed in zip(selected, file_seeds)
        }
        for future in as_completed(futures):
            path = futures[future]
            try:
                all_rows.extend(future.result())
            except Exception as exc:
                log.error("Worker failed for %s: %s", path, exc)

    if exclude_uris:
        all_rows = [r for r in all_rows if r["uri"] not in exclude_uris]

    return all_rows


# ---------------------------------------------------------------------------
# Multi-crawl discovery
# ---------------------------------------------------------------------------

COLLINFO_URL = "https://index.commoncrawl.org/collinfo.json"
_CC_BUCKET_NAME = "commoncrawl"
_WET_MIN_CRAWL = "CC-MAIN-2013"
_CC_MAIN_RE = re.compile(r"^CC-MAIN-\d{4}-\d{2,4}$")


def _fetch_crawl_ids() -> List[str]:
    log.info("Fetching crawl list from %s", COLLINFO_URL)
    with urllib.request.urlopen(COLLINFO_URL, timeout=30) as resp:
        data = json.loads(resp.read())
    ids = [e["id"] for e in data if _CC_MAIN_RE.match(e.get("id", ""))]
    ids.sort()
    log.info("Found %d CC-MAIN crawl(s).", len(ids))
    return ids


def _wet_paths_uri(crawl_id: str) -> str:
    return f"s3://{_CC_BUCKET_NAME}/crawl-data/{crawl_id}/wet.paths.gz"


def _output_path_for(crawl_id: str, output_dir: Path) -> Path:
    return output_dir / f"word_context_{crawl_id}.csv"


def _fmt_duration(secs: float) -> str:
    s = int(secs)
    h, rem = divmod(s, 3600)
    m, s = divmod(rem, 60)
    if h:
        return f"{h}h {m:02d}m"
    return f"{m}m {s:02d}s"


def _crawl_seed(base_seed: int, crawl_index: int) -> int:
    return (base_seed * 1_000_003 + crawl_index) & 0x7FFF_FFFF


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Extract target-word context windows from Common Crawl WET files.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Single dump via S3 paths file
  python word_context.py \\
      --file-list s3://commoncrawl/crawl-data/CC-MAIN-2024-10/wet.paths.gz \\
      --cc-date CC-MAIN-2024-10 --n-uris 500 --output-dir contexts/

  # Multi-crawl: all crawls from 2020 onwards
  python word_context.py --since 2020 --n-uris 500 --output-dir contexts/

  # Multi-crawl: explicit date range
  python word_context.py \\
      --from-crawl CC-MAIN-2019-04 --to-crawl CC-MAIN-2022-49 \\
      --n-uris 1000 --context-window 15 --output-dir contexts/
        """,
    )

    # Input
    parser.add_argument("files", nargs="*",
                        help="WET file paths (local or s3://). "
                             "Use --file-list for a .paths.gz index.")
    parser.add_argument("--file-list", metavar="PATHS_FILE", default=None,
                        help="Gzip-compressed CC paths file (.paths.gz), local or s3://.")

    # Sampling
    parser.add_argument("--n-uris", type=int, default=500, metavar="N", dest="n_uris",
                        help="Total URIs (records) to sample per dump (default: 500).")
    parser.add_argument("--sample-files", type=int, default=None, metavar="M",
                        dest="sample_files",
                        help="WET files to sample from per dump. "
                             "Default: max(workers, ceil(n_uris/100)).")
    parser.add_argument("--max-records", type=int, default=None, metavar="K",
                        dest="max_records",
                        help="Stop reading after K content records per WET file. "
                             "Trades sampling uniformity for speed.")

    # Target words
    parser.add_argument("--words", metavar="WORDS_FILE", default="target_words.txt",
                        help="Plain-text target words/phrases file, one per line "
                             "(default: target_words.txt).")

    # Context
    parser.add_argument("--context-window", type=int, default=10, metavar="K",
                        dest="context_window",
                        help="Tokens on each side of a match to include (default: 10).")

    # Output
    parser.add_argument("--output-dir", required=True, metavar="DIR", dest="output_dir",
                        help="Directory for output CSV files.")
    parser.add_argument("--cc-date", default=None, metavar="ID", dest="cc_date",
                        help="Crawl ID for the output filename (e.g. CC-MAIN-2024-10). "
                             "Inferred from file paths if omitted.")

    # Parallelism / reproducibility
    parser.add_argument("--workers", type=int, default=None, metavar="N",
                        help="Parallel worker processes (default: os.cpu_count()).")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed (default: 42).")

    # Multi-crawl range filters
    parser.add_argument("--since", type=int, default=None, metavar="YEAR",
                        help="Multi-crawl: only process crawls from this year onwards.")
    parser.add_argument("--from-crawl", default=None, metavar="ID", dest="from_crawl",
                        help="Multi-crawl: start from this crawl ID (inclusive).")
    parser.add_argument("--to-crawl", default=None, metavar="ID", dest="to_crawl",
                        help="Multi-crawl: stop after this crawl ID (inclusive).")
    parser.add_argument("--force", action="store_true",
                        help="Multi-crawl: re-run crawls even if output already exists.")

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    targets = load_target_words(args.words)

    # ── Multi-crawl mode ──────────────────────────────────────────────────────
    if not args.files and not args.file_list:
        if not (args.since or args.from_crawl or args.to_crawl):
            parser.error(
                "Provide files, --file-list, or a crawl range "
                "(--since / --from-crawl / --to-crawl)."
            )

        crawl_ids = _fetch_crawl_ids()
        crawl_ids = [c for c in crawl_ids if c >= _WET_MIN_CRAWL]

        if args.since:
            crawl_ids = [c for c in crawl_ids if int(c[8:12]) >= args.since]
        if args.from_crawl:
            crawl_ids = [c for c in crawl_ids if c >= args.from_crawl]
        if args.to_crawl:
            crawl_ids = [c for c in crawl_ids if c <= args.to_crawl]

        if not crawl_ids:
            log.error("No crawls matched the specified filters.")
            sys.exit(1)

        output_dir.mkdir(parents=True, exist_ok=True)
        log.info("Processing %d crawl(s): %s → %s",
                 len(crawl_ids), crawl_ids[0], crawl_ids[-1])

        n_total = len(crawl_ids)
        elapsed_times: List[float] = []

        for i, crawl_id in enumerate(crawl_ids):
            out = _output_path_for(crawl_id, output_dir)
            n_remaining = n_total - i - 1

            existing_rows: List[Dict[str, str]] = []
            existing_uris: Set[str] = set()
            if not args.force and out.exists():
                existing_rows, existing_uris = _read_existing_rows(out)
                n_existing = len(existing_uris)
                if n_existing >= args.n_uris:
                    log.info("[%d/%d] %s — skipped (%d URIs already collected)",
                             i + 1, n_total, crawl_id, n_existing)
                    continue
                log.info("[%d/%d] %s — topping up (%d/%d URIs, need %d more)",
                         i + 1, n_total, crawl_id, n_existing, args.n_uris,
                         args.n_uris - n_existing)

            seed = _crawl_seed(args.seed, i)
            # Use a distinct seed for top-up passes so file/record selection differs.
            effective_seed = seed if not existing_uris else seed ^ 0x5A5A5A5A
            n_needed = args.n_uris - len(existing_uris)
            t0 = time.monotonic()

            try:
                file_list = list(parse_paths_file(_wet_paths_uri(crawl_id)))
                new_rows = process_dump(
                    file_list=file_list,
                    n_uris=n_needed,
                    n_files=args.sample_files,
                    targets=targets,
                    context_window=args.context_window,
                    seed=effective_seed,
                    workers=args.workers,
                    max_records=args.max_records,
                    cc_id=crawl_id,
                    exclude_uris=existing_uris if existing_uris else None,
                )
                all_rows = existing_rows + new_rows
                _write_rows(out, all_rows)
                log.info("%s — %d context rows (%d new) → %s",
                         crawl_id, len(all_rows), len(new_rows), out)
            except Exception as exc:
                log.error("[%d/%d] ✗  %s — %s", i + 1, n_total, crawl_id, exc)
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
                     i + 1, n_total, crawl_id, _fmt_duration(elapsed), eta)
        return

    # ── Single-dump mode ──────────────────────────────────────────────────────
    file_list: List[str] = []
    if args.file_list:
        file_list.extend(parse_paths_file(args.file_list))
    if args.files:
        file_list.extend(expand_paths(args.files))

    if not file_list:
        parser.error("No WET files found.")

    cc_id = args.cc_date or _infer_cc_id(file_list) or "unknown"
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = output_dir / f"word_context_{cc_id}_{timestamp}.csv"
    output_dir.mkdir(parents=True, exist_ok=True)

    rows = process_dump(
        file_list=file_list,
        n_uris=args.n_uris,
        n_files=args.sample_files,
        targets=targets,
        context_window=args.context_window,
        seed=args.seed,
        workers=args.workers,
        max_records=args.max_records,
        cc_id=cc_id,
    )
    _write_rows(output_path, rows)
    log.info("%s — %d context rows → %s", cc_id, len(rows), output_path)


if __name__ == "__main__":
    main()
