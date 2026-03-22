#!/usr/bin/env python3
"""
Word count script for WET/WARC format files from Common Crawl dumps.
Designed to run on AWS EC2.

Supports:
  - Local WET/WARC files (plain or gzip-compressed)
  - S3 paths (s3://bucket/key) via boto3 (Common Crawl is already hosted on s3 with free access)
  - Glob patterns for multiple local files

Usage:
  python word_count.py file1.warc.gz file2.wet.gz --output counts.json
  python word_count.py s3://bucket/path/to/file.wet.gz --output counts.csv --format csv
  python word_count.py /data/*.wet.gz --top 50000 --min-length 3

Dependencies:
  pip install warcio boto3 beautifulsoup4 lxml
"""

import argparse
import csv
import glob
import gzip
import io
import json
import logging
import random
import re
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Dict, Iterator, List, NamedTuple, Optional, Set, Tuple

try:
    from warcio.archiveiterator import ArchiveIterator
except ImportError:
    print("ERROR: warcio is required. Install with: pip install warcio", file=sys.stderr)
    sys.exit(1)

# Optional: S3 support
try:
    import boto3
    from botocore.config import Config as BotocoreConfig
    from botocore.exceptions import BotoCoreError, ClientError
    HAS_BOTO3 = True
except ImportError:
    HAS_BOTO3 = False

# Optional: HTML parsing for WARC files
try:
    from bs4 import BeautifulSoup
    HAS_BS4 = True
except ImportError:
    HAS_BS4 = False

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

# Force line-buffering on stderr so log messages appear in real-time even
# when output is piped or redirected (e.g. python ... 2>&1 | tee run.log).
sys.stderr.reconfigure(line_buffering=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stderr)],
)
log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Word tokenizer
# ---------------------------------------------------------------------------

_WORD_RE = re.compile(r"[a-zA-Z''\u2019\-]+")


def tokenize(text: str, min_length: int = 1, lowercase: bool = True) -> List[str]:
    """Extract words from a block of plain text."""
    words = _WORD_RE.findall(text)
    if lowercase:
        words = [w.lower() for w in words]
    if min_length > 1:
        words = [w for w in words if len(w) >= min_length]
    return words


# ---------------------------------------------------------------------------
# Target words / phrases
# ---------------------------------------------------------------------------

class Targets(NamedTuple):
    """Compiled target list ready for matching against a token stream."""
    # Single-token targets for O(1) membership checks.
    single_words: Set[str]
    # Maps the first token of each phrase to all phrases that start with it.
    # Phrases are stored as tuples so they can be compared against token slices.
    phrase_index: Dict[str, List[Tuple[str, ...]]]


def load_target_words(path: str, lowercase: bool = True) -> "Targets":
    """Load a plain-text file of target words and phrases (one per line).

    Lines with a single token are treated as individual words.
    Lines with multiple whitespace-separated tokens are treated as phrases.
    Blank lines and lines starting with '#' are ignored.
    All entries are normalised to match the tokeniser's case behaviour.
    """
    single_words: Set[str] = set()
    phrases: List[Tuple[str, ...]] = []

    with open(path, encoding="utf-8") as fh:
        for line in fh:
            raw = line.strip()
            if not raw or raw.startswith("#"):
                continue
            parts = raw.lower().split() if lowercase else raw.split()
            if len(parts) == 1:
                single_words.add(parts[0])
            else:
                phrases.append(tuple(parts))

    # Index phrases by their first token for efficient lookup.
    phrase_index: Dict[str, List[Tuple[str, ...]]] = {}
    for phrase in phrases:
        phrase_index.setdefault(phrase[0], []).append(phrase)

    log.info(
        "Loaded %d single words and %d phrases from %s",
        len(single_words), len(phrases), path,
    )
    return Targets(single_words=single_words, phrase_index=phrase_index)


def count_matches(tokens: List[str], targets: Targets) -> Counter:
    """Return a Counter of target hits found in a token list.

    Each position is checked against single-word targets first, then against
    any phrases whose first token matches. Phrase keys in the counter are the
    original space-joined strings (e.g. 'oh my god').
    """
    matches: Counter = Counter()
    for i, token in enumerate(tokens):
        if token in targets.single_words:
            matches[token] += 1
        if token in targets.phrase_index:
            for phrase in targets.phrase_index[token]:
                n = len(phrase)
                if tuple(tokens[i:i + n]) == phrase:
                    matches[" ".join(phrase)] += 1
    return matches


# ---------------------------------------------------------------------------
# Text extraction from WARC records
# ---------------------------------------------------------------------------

def extract_text_from_wet_record(record) -> str:
    """Extract plain text from a WET 'conversion' record."""
    if record.rec_type != "conversion":
        return None
    try:
        return record.content_stream().read().decode("utf-8", errors="replace")
    except Exception as exc:
        log.debug("Failed to read WET record: %s", exc)
        return None


def extract_text_from_warc_record(record) -> str:
    """
    Extract plain text from a WARC 'response' record.
    Uses BeautifulSoup if available, otherwise falls back to a simple regex strip.
    """
    if record.rec_type != "response":
        return

    content_type = record.http_headers.get_header("Content-Type", "") if record.http_headers else ""
    if "text/html" not in content_type and "text/plain" not in content_type:
        return

    try:
        raw = record.content_stream().read().decode("utf-8", errors="replace")
    except Exception as exc:
        log.debug("Failed to read WARC record: %s", exc)
        return

    if "text/html" in content_type:
        if HAS_BS4:
            soup = BeautifulSoup(raw, "lxml")
            for tag in soup(["script", "style", "noscript"]):
                tag.decompose()
            return soup.get_text(separator=" ")
        else:
            # Crude HTML tag stripper if bs4 is unavailable
            return re.sub(r"<[^>]+>", " ", raw)

    return raw


# ---------------------------------------------------------------------------
# File / S3 stream helpers
# ---------------------------------------------------------------------------

def open_local(path: str):
    """Open a local file (handles .gz transparently via warcio)."""
    return open(path, "rb")


_S3_CLIENT_CONFIG = BotocoreConfig(
    retries={
        "mode": "adaptive",
        "max_attempts": 11,  # 1 initial attempt + 10 retries
    }
) if HAS_BOTO3 else None

# Application-level retry settings (on top of botocore's per-request retries).
# Base and max are in seconds; full-jitter exponential backoff is used.
_S3_RETRY_BASE_SECS = 10
_S3_RETRY_MAX_SECS = 300   # cap at 5 minutes per attempt
_S3_RETRY_MAX_ATTEMPTS = 10


def open_s3(s3_uri: str):
    """Stream an S3 object as a file-like binary object.

    Retries up to _S3_RETRY_MAX_ATTEMPTS times with exponential backoff and
    full jitter, in addition to botocore's own per-request adaptive retries.
    """
    if not HAS_BOTO3:
        log.error("boto3 is required for S3 support. Install with: pip install boto3")
        sys.exit(1)

    parsed = _parse_s3_uri(s3_uri)
    bucket, key = parsed["bucket"], parsed["key"]
    log.info("Streaming from S3: s3://%s/%s", bucket, key)

    s3 = boto3.client("s3", config=_S3_CLIENT_CONFIG)
    for attempt in range(1, _S3_RETRY_MAX_ATTEMPTS + 2):
        try:
            return s3.get_object(Bucket=bucket, Key=key)["Body"]
        except (BotoCoreError, ClientError) as exc:
            if attempt > _S3_RETRY_MAX_ATTEMPTS:
                log.error("Exhausted %d retries for %s: %s", _S3_RETRY_MAX_ATTEMPTS, s3_uri, exc)
                sys.exit(1)
            cap = min(_S3_RETRY_BASE_SECS * (2 ** (attempt - 1)), _S3_RETRY_MAX_SECS)
            delay = random.uniform(0, cap)  # full jitter
            log.warning("S3 error on attempt %d/%d for %s: %s — retrying in %.1fs",
                        attempt, _S3_RETRY_MAX_ATTEMPTS, s3_uri, exc, delay)
            time.sleep(delay)


def _parse_s3_uri(uri: str) -> dict:
    """Parse s3://bucket/key into components."""
    if not uri.startswith("s3://"):
        raise ValueError(f"Not an S3 URI: {uri}")
    without_scheme = uri[5:]
    bucket, _, key = without_scheme.partition("/")
    return {"bucket": bucket, "key": key}


def is_s3_uri(path: str) -> bool:
    return path.startswith("s3://")


def expand_paths(raw_paths: List[str]) -> List[str]:
    """Expand glob patterns and return a flat list of paths/URIs."""
    expanded = []
    for p in raw_paths:
        if is_s3_uri(p):
            expanded.append(p)
        else:
            matched = sorted(glob.glob(p, recursive=True))
            if matched:
                expanded.extend(matched)
            else:
                # Treat as literal path even if it doesn't exist yet (will fail later)
                expanded.append(p)
    return expanded


# Common Crawl S3 bucket used when resolving bare relative paths from a paths file
_CC_S3_BUCKET = "s3://commoncrawl"


def parse_paths_file(paths_file: str) -> Iterator[str]:
    """
    Read a Common Crawl .paths.gz file and yield one S3 URI per line.

    The file may be local or an S3 URI. Each line is expected to be either:
      - A relative path  (e.g. crawl-data/CC-MAIN-2024-10/.../file.wet.gz)
      - A full S3 URI    (e.g. s3://commoncrawl/crawl-data/...)

    Relative paths are automatically prefixed with s3://commoncrawl/.
    Blank lines and lines starting with '#' are skipped.
    """
    log.info("Reading paths file: %s", paths_file)

    if is_s3_uri(paths_file):
        raw_stream = open_s3(paths_file)
        # boto3 body is not directly seekable; wrap in BytesIO for gzip
        raw_bytes = raw_stream.read()
        byte_stream = io.BytesIO(raw_bytes)
    else:
        byte_stream = open(paths_file, "rb")

    with gzip.open(byte_stream, "rt", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if is_s3_uri(line):
                yield line
            else:
                yield f"{_CC_S3_BUCKET}/{line}"


# ---------------------------------------------------------------------------
# Core processing
# ---------------------------------------------------------------------------

def process_file(
    path: str,
    counter: Counter,
    args: argparse.Namespace,
    targets: Optional[Targets] = None,
    file_index: int = 0,
    total_files: int = 0,
) -> int:
    """
    Process one WET or WARC file and update `counter` in-place.
    Returns the number of records processed.

    When `targets` is provided, only matching words and phrases are counted
    and --min-length is ignored (the targets file defines exactly what to count).
    """
    record_count = 0
    lowercase = not args.keep_case
    pct = 100.0 * file_index / total_files if total_files else 0.0

    stream = open_s3(path) if is_s3_uri(path) else open_local(path)

    try:
        for record in ArchiveIterator(stream):
            text = None

            if args.format_hint == "wet" or record.rec_type == "conversion":
                text = extract_text_from_wet_record(record)
            elif args.format_hint == "warc" or record.rec_type == "response":
                text = extract_text_from_warc_record(record)
            else:
                # Auto-detect: try both
                text = extract_text_from_wet_record(record)
                if text is None:
                    text = extract_text_from_warc_record(record)

            if text:
                if targets is not None:
                    # Tokenize without min_length — phrase matching needs every token.
                    tokens = tokenize(text, lowercase=lowercase)
                    counter.update(count_matches(tokens, targets))
                else:
                    counter.update(tokenize(text, min_length=args.min_length, lowercase=lowercase))
                record_count += 1

                if record_count % 1000 == 0:
                    log.info("  [%d/%d] (%.1f%%) %s — processed %d records, %d unique words so far",
                             file_index, total_files, pct, path, record_count, len(counter))
    except Exception as exc:
        log.error("Error processing %s: %s", path, exc)
    finally:
        if hasattr(stream, "close"):
            stream.close()

    return record_count


# ---------------------------------------------------------------------------
# Checkpoint helpers
# ---------------------------------------------------------------------------

def load_checkpoint(checkpoint_path: str) -> Tuple[Counter, Set[str]]:
    """Load a checkpoint file and return (counter, set of completed file paths).

    Returns empty defaults if the file does not exist or is unreadable.
    """
    try:
        with open(checkpoint_path, encoding="utf-8") as fh:
            data = json.load(fh)
        counter = Counter(data.get("counter", {}))
        completed = set(data.get("completed_files", []))
        log.info(
            "Checkpoint loaded from %s — %d files already done, %d unique words",
            checkpoint_path, len(completed), len(counter),
        )
        return counter, completed
    except FileNotFoundError:
        return Counter(), set()
    except Exception as exc:
        log.warning("Could not read checkpoint %s: %s — starting fresh", checkpoint_path, exc)
        return Counter(), set()


def save_checkpoint(checkpoint_path: str, counter: Counter, completed: Set[str]) -> None:
    """Atomically write the current counter and completed-file list to disk."""
    tmp_path = checkpoint_path + ".tmp"
    try:
        with open(tmp_path, "w", encoding="utf-8") as fh:
            json.dump({"completed_files": list(completed), "counter": dict(counter)}, fh)
        Path(tmp_path).replace(checkpoint_path)
    except Exception as exc:
        log.warning("Failed to save checkpoint to %s: %s", checkpoint_path, exc)


# ---------------------------------------------------------------------------
# Output writers
# ---------------------------------------------------------------------------

def write_json(counter: Counter, path: str, top: Optional[int]) -> None:
    data = counter.most_common(top) if top else counter.most_common()
    payload = {
        "total_unique_words": len(counter),
        "total_word_occurrences": sum(counter.values()),
        "words": [{"word": w, "count": c} for w, c in data],
    }
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, ensure_ascii=False, indent=2)
    log.info("JSON output written to %s", path)


def write_csv(counter: Counter, path: str, top: Optional[int]) -> None:
    data = counter.most_common(top) if top else counter.most_common()
    with open(path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(["word", "count"])
        writer.writerows(data)
    log.info("CSV output written to %s", path)


def write_tsv(counter: Counter, path: str, top: Optional[int]) -> None:
    data = counter.most_common(top) if top else counter.most_common()
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("word\tcount\n")
        for word, count in data:
            fh.write(f"{word}\t{count}\n")
    log.info("TSV output written to %s", path)


OUTPUT_WRITERS = {
    "json": write_json,
    "csv": write_csv,
    "tsv": write_tsv,
}


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Count word instances in WET/WARC files. Outputs a file suitable for local analysis.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Count words in local WET files and write JSON
  python word_count.py /data/*.wet.gz --output counts.json

  # Count only specific target words
  python word_count.py /data/*.wet.gz --words targets.txt --output counts.json

  # Use a Common Crawl .paths.gz file list (local)
  python word_count.py --file-list wet.paths.gz --output counts.json

  # Use a Common Crawl .paths.gz file list directly from S3
  python word_count.py --file-list s3://commoncrawl/crawl-data/CC-MAIN-2024-10/wet.paths.gz \\
      --output counts.csv --output-format csv

  # Stream individual files from S3, top 100k words only
  python word_count.py s3://commoncrawl/crawl-data/.../file.wet.gz \\
      --output counts.csv --output-format csv --top 100000

  # WARC files with HTML extraction, minimum word length 4
  python word_count.py /data/*.warc.gz --type warc --min-length 4 --output counts.tsv --output-format tsv
        """,
    )

    parser.add_argument(
        "files",
        nargs="*",
        help="WET/WARC file paths (local or s3://), supports glob patterns. "
             "Use --file-list instead to provide a Common Crawl .paths.gz index.",
    )
    parser.add_argument(
        "--file-list",
        metavar="PATHS_FILE",
        default=None,
        help="Path to a gzip-compressed Common Crawl paths file (.paths.gz), "
             "local or s3://. Each line is a relative CC path or full S3 URI. "
             "Can be combined with positional file arguments.",
    )
    parser.add_argument(
        "--output", "-o",
        default="word_counts.json",
        help="Output file path (default: word_counts.json)",
    )
    parser.add_argument(
        "--output-format", "-f",
        choices=["json", "csv", "tsv"],
        default=None,
        help="Output format. Inferred from --output extension if not set.",
    )
    parser.add_argument(
        "--type",
        dest="format_hint",
        choices=["wet", "warc", "auto"],
        default="auto",
        help="File type hint: wet | warc | auto (default: auto)",
    )
    parser.add_argument(
        "--top",
        type=int,
        default=None,
        metavar="N",
        help="Only write the top N most frequent words (default: all)",
    )
    parser.add_argument(
        "--min-length",
        type=int,
        default=1,
        metavar="N",
        help="Minimum word length to count (default: 1)",
    )
    parser.add_argument(
        "--keep-case",
        action="store_true",
        help="Preserve original casing instead of lowercasing all words",
    )
    parser.add_argument(
        "--words",
        metavar="WORDS_FILE",
        default=None,
        help="Path to a plain-text file of target words to count, one per line. "
             "When provided, only these words are tallied and all others are ignored. "
             "Blank lines and lines starting with '#' are skipped.",
    )
    parser.add_argument(
        "--checkpoint",
        metavar="FILE",
        default=None,
        help="Path to checkpoint file for resuming interrupted runs "
             "(default: <output>.checkpoint.json). Pass 'none' to disable checkpointing.",
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable verbose/debug logging",
    )

    return parser


def infer_output_format(output_path: str) -> str:
    ext = Path(output_path).suffix.lstrip(".")
    return ext if ext in OUTPUT_WRITERS else "json"


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    if args.verbose:
        log.setLevel(logging.DEBUG)

    # Resolve output format
    fmt = args.output_format or infer_output_format(args.output)
    if fmt not in OUTPUT_WRITERS:
        log.error("Unsupported output format: %s. Choose from: %s", fmt, ", ".join(OUTPUT_WRITERS))
        sys.exit(1)

    # Build the file list from both sources
    file_list: List[str] = []

    if args.file_list:
        file_list.extend(parse_paths_file(args.file_list))

    if args.files:
        file_list.extend(expand_paths(args.files))

    if not file_list:
        parser.error("Provide at least one WET/WARC file or use --file-list with a .paths.gz index.")

    log.info("Files to process: %d", len(file_list))

    targets: Optional[Targets] = None
    if args.words:
        targets = load_target_words(args.words, lowercase=not args.keep_case)

    # Resolve checkpoint path (None means disabled)
    checkpoint_path: Optional[str] = None
    if args.checkpoint and args.checkpoint.lower() != "none":
        checkpoint_path = args.checkpoint
    elif args.checkpoint is None:
        checkpoint_path = args.output + ".checkpoint.json"

    write = OUTPUT_WRITERS[fmt]
    counter: Counter
    completed_files: Set[str]

    if checkpoint_path:
        counter, completed_files = load_checkpoint(checkpoint_path)
        skipped = [p for p in file_list if p in completed_files]
        if skipped:
            log.info("Skipping %d already-completed file(s).", len(skipped))
    else:
        counter, completed_files = Counter(), set()

    remaining = [p for p in file_list if p not in completed_files]
    total_files = len(file_list)
    total_records = 0

    for path in remaining:
        file_index = len(completed_files) + 1
        pct = 100.0 * (file_index - 1) / total_files if total_files else 0.0
        log.info("[%d/%d] (%.1f%%) Processing: %s", file_index, total_files, pct, path)
        n = process_file(path, counter, args, targets=targets,
                         file_index=file_index, total_files=total_files)
        total_records += n
        completed_files.add(path)
        pct_done = 100.0 * len(completed_files) / total_files if total_files else 0.0
        log.info("[%d/%d] (%.1f%%) Done — %d records, running unique word count: %d",
                 len(completed_files), total_files, pct_done, n, len(counter))

        # Persist progress after every file
        write(counter, args.output, args.top)
        if checkpoint_path:
            save_checkpoint(checkpoint_path, counter, completed_files)

    if checkpoint_path and Path(checkpoint_path).exists():
        Path(checkpoint_path).unlink()
        log.info("Checkpoint removed — all files processed.")

    log.info("Processing complete. Total records: %d | Unique words: %d | Total occurrences: %d",
             total_records, len(counter), sum(counter.values()))
    log.info("Done. Download %s for local analysis.", args.output)


if __name__ == "__main__":
    main()
