#!/usr/bin/env python3
"""
run_all_crawls.py — Run word_sample.py against every Common Crawl WET dump on S3.

Fetches the full crawl list from Common Crawl's public index API, verifies
that a wet.paths.gz exists on S3 for each crawl, then invokes word_sample.py
once per crawl.  Outputs one CSV per crawl in --output-dir.

Already-completed crawls (output file present) are skipped automatically,
making the script safe to interrupt and resume.

Designed to run on AWS EC2 with an IAM role that has s3:GetObject access to
s3://commoncrawl (the bucket is public so no credentials are required).

Usage:
    python run_all_crawls.py --n 100000 --seed 42 --output-dir /data/samples/
    python run_all_crawls.py --n 50000  --seed 42 --max-records 50 --output-dir samples/
    python run_all_crawls.py --n 100000 --seed 42 --since 2018 --output-dir samples/
    python run_all_crawls.py --n 100000 --seed 42 --crawls CC-MAIN-2024-10 CC-MAIN-2023-40
    python run_all_crawls.py --n 100000 --seed 42 --dry-run
"""

import argparse
import json
import logging
import os
import re
import subprocess
import sys
import time
import urllib.request
from datetime import datetime
from pathlib import Path
from typing import List, Optional

try:
    import boto3  # pyright: ignore[reportMissingImports]
    from botocore import UNSIGNED  # pyright: ignore[reportMissingImports]
    from botocore.config import Config as BotocoreConfig  # pyright: ignore[reportMissingImports]
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
# Constants
# ---------------------------------------------------------------------------

COLLINFO_URL = "https://index.commoncrawl.org/collinfo.json"
CC_BUCKET    = "commoncrawl"
# WET files are only available from CC-MAIN-2013-20 onwards
_WET_MIN_CRAWL = "CC-MAIN-2013"
_CC_CRAWL_RE   = re.compile(r"^CC-MAIN-\d{4}-\d{2,4}$")

# ---------------------------------------------------------------------------
# Crawl discovery
# ---------------------------------------------------------------------------

def fetch_crawl_ids() -> List[str]:
    """Return all CC-MAIN-* crawl IDs from the public collinfo API, oldest first."""
    log.info("Fetching crawl list from %s", COLLINFO_URL)
    with urllib.request.urlopen(COLLINFO_URL, timeout=30) as resp:
        data = json.loads(resp.read())
    ids = [entry["id"] for entry in data if _CC_CRAWL_RE.match(entry.get("id", ""))]
    ids.sort()  # lexicographic = chronological for YYYY-WW format
    log.info("Found %d CC-MAIN crawl(s) in index.", len(ids))
    return ids


def has_wet_paths(crawl_id: str, s3_client) -> bool:
    """Return True if wet.paths.gz exists for this crawl on S3."""
    key = f"crawl-data/{crawl_id}/wet.paths.gz"
    try:
        s3_client.head_object(Bucket=CC_BUCKET, Key=key)
        return True
    except Exception:
        return False


def wet_paths_uri(crawl_id: str) -> str:
    return f"s3://{CC_BUCKET}/crawl-data/{crawl_id}/wet.paths.gz"


# ---------------------------------------------------------------------------
# Per-crawl seed derivation
# ---------------------------------------------------------------------------

def crawl_seed(base_seed: int, crawl_index: int) -> int:
    """Derive a stable, unique seed for a crawl from the base seed and its index."""
    return (base_seed * 1_000_003 + crawl_index) & 0x7FFF_FFFF


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def _fmt_duration(secs: float) -> str:
    s = int(secs)
    h, rem = divmod(s, 3600)
    m, s = divmod(rem, 60)
    if h:
        return f"{h}h {m:02d}m"
    return f"{m}m {s:02d}s"


# ---------------------------------------------------------------------------
# Running word_sample
# ---------------------------------------------------------------------------

WORD_SAMPLE = Path(__file__).parent / "word_sample.py"


def output_path_for(crawl_id: str, output_dir: Path) -> Path:
    return output_dir / f"word_sample_{crawl_id}.csv"


def run_crawl(
    crawl_id: str,
    seed: int,
    n: int,
    output_dir: Path,
    max_records: Optional[int],
    sample_files: Optional[int],
    min_length: int,
    workers: Optional[int],
    dry_run: bool,
) -> tuple:
    """Invoke word_sample.py for one crawl. Returns (success, elapsed_seconds)."""
    out = output_path_for(crawl_id, output_dir)

    cmd: List[str] = [
        sys.executable, str(WORD_SAMPLE),
        "--file-list", wet_paths_uri(crawl_id),
        "--n",         str(n),
        "--seed",      str(seed),
        "--output",    str(out),
        "--cc-date",   crawl_id,
    ]
    if max_records is not None:
        cmd += ["--max-records", str(max_records)]
    if sample_files is not None:
        cmd += ["--sample-files", str(sample_files)]
    if min_length > 1:
        cmd += ["--min-length", str(min_length)]
    if workers is not None:
        cmd += ["--workers", str(workers)]

    log.debug("CMD: %s", " ".join(cmd))

    if dry_run:
        return True, 0.0

    t0 = time.monotonic()
    result = subprocess.run(cmd)
    elapsed = time.monotonic() - t0

    ok = result.returncode == 0
    if not ok:
        log.error("word_sample exited %d for %s", result.returncode, crawl_id)
    return ok, elapsed


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run word_sample.py against every Common Crawl WET dump on S3.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Sample 100k words from every crawl since 2018, stop after 50 records/file
  python run_all_crawls.py --n 100000 --seed 42 --since 2018 \\
      --max-records 50 --output-dir /data/samples/

  # Specific crawls only
  python run_all_crawls.py --n 50000 --seed 0 \\
      --crawls CC-MAIN-2024-10 CC-MAIN-2023-40 CC-MAIN-2022-49 \\
      --output-dir samples/

  # Preview without running anything
  python run_all_crawls.py --n 100000 --seed 42 --dry-run
        """,
    )
    # TODO(): Add an argument that just passes total samples we want to collect and derives n from that.
    parser.add_argument(
        "--n", type=int, required=True, metavar="N",
        help="Words to sample per crawl (passed to word_sample --n).",
    )
    parser.add_argument(
        "--seed", type=int, required=True,
        help="Base random seed.  Each crawl gets a unique derived seed so runs "
             "are reproducible across the whole collection.",
    )
    parser.add_argument(
        "--output-dir", required=True, metavar="DIR",
        help="Directory for output CSVs.  Created if it does not exist.",
    )
    parser.add_argument(
        "--max-records", type=int, default=None, metavar="K",
        dest="max_records",
        help="Stop after K content records per WET file (passed to word_sample). "
             "Greatly reduces runtime. Omit for full-file unbiased sampling.",
    )
    parser.add_argument(
        "--sample-files", type=int, default=None, metavar="M",
        dest="sample_files",
        help="WET files to randomly select per crawl (passed to word_sample --sample-files). "
             "words-per-file = ceil(N / M). Defaults to min(N, total_files).",
    )
    parser.add_argument(
        "--min-length", type=int, default=1, metavar="N",
        dest="min_length",
        help="Minimum word length (default: 1, passed to word_sample).",
    )
    parser.add_argument(
        "--workers", type=int, default=None, metavar="N",
        help="Parallel worker processes per crawl (passed to word_sample --workers). "
             "Defaults to os.cpu_count() inside word_sample.",
    )
    parser.add_argument(
        "--crawls", nargs="+", metavar="ID", default=None,
        help="Run only these specific crawl IDs instead of all available ones.",
    )
    parser.add_argument(
        "--since", type=int, default=None, metavar="YEAR",
        help="Only process crawls from this year onwards (e.g. --since 2018).",
    )
    parser.add_argument(
        "--from-crawl", default=None, metavar="ID",
        dest="from_crawl",
        help="Start from this crawl ID (inclusive, chronological order).  "
             "Useful for resuming a partial run without --since.",
    )
    parser.add_argument(
        "--to-crawl", default=None, metavar="ID",
        dest="to_crawl",
        help="Stop after this crawl ID (inclusive).",
    )
    parser.add_argument(
        "--force", action="store_true",
        help="Re-run crawls even if the output file already exists.",
    )
    parser.add_argument(
        "--dry-run", action="store_true", dest="dry_run",
        help="Print the commands that would be run without executing them.",
    )
    parser.add_argument(
        "--skip-wet-check", action="store_true", dest="skip_wet_check",
        help="Skip the S3 head-object check for wet.paths.gz existence "
             "(faster startup, but may fail for crawls without WET files).",
    )
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # ── Enumerate crawls ──────────────────────────────────────────────────────
    if args.crawls:
        crawl_ids = sorted(args.crawls)
        log.info("Using %d explicitly specified crawl(s).", len(crawl_ids))
    else:
        crawl_ids = fetch_crawl_ids()

        # Filter out pre-WET crawls (before CC-MAIN-2013)
        crawl_ids = [c for c in crawl_ids if c >= _WET_MIN_CRAWL]

        if args.since:
            crawl_ids = [c for c in crawl_ids if int(c[8:12]) >= args.since]

        if args.from_crawl:
            crawl_ids = [c for c in crawl_ids if c >= args.from_crawl]

        if args.to_crawl:
            crawl_ids = [c for c in crawl_ids if c <= args.to_crawl]

        log.info("Crawls to process after filtering: %d", len(crawl_ids))

    if not crawl_ids:
        log.error("No crawls matched the specified filters.")
        sys.exit(1)

    # ── Optionally verify wet.paths.gz exists on S3 ───────────────────────────
    if not args.skip_wet_check and not args.dry_run:
        if not HAS_BOTO3:
            log.warning(
                "boto3 not available — skipping wet.paths.gz existence check. "
                "Install with: pip install boto3"
            )
        else:
            log.info("Checking S3 for wet.paths.gz availability...")
            # Public bucket: use unsigned requests
            s3 = boto3.client(
                "s3",
                config=BotocoreConfig(signature_version=UNSIGNED),
            )
            verified, skipped = [], []
            for cid in crawl_ids:
                if has_wet_paths(cid, s3):
                    verified.append(cid)
                else:
                    log.warning("No wet.paths.gz for %s — skipping.", cid)
                    skipped.append(cid)
            log.info(
                "S3 check: %d crawl(s) have WET files, %d skipped.",
                len(verified), len(skipped),
            )
            crawl_ids = verified

    if not crawl_ids:
        log.error("No crawls have WET files. Exiting.")
        sys.exit(1)

    # ── Summary ───────────────────────────────────────────────────────────────
    run_ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    log.info("=" * 60)
    log.info("Run started     : %s", run_ts)
    log.info("Crawls          : %d  (%s → %s)", len(crawl_ids), crawl_ids[0], crawl_ids[-1])
    log.info("Words per crawl : %d", args.n)
    log.info("Base seed       : %d", args.seed)
    log.info("Max records     : %s", args.max_records or "unlimited")
    log.info("Output dir      : %s", output_dir.resolve())
    log.info("Dry run         : %s", args.dry_run)
    log.info("=" * 60)

    # ── Process each crawl ───────────────────────────────────────────────────
    succeeded, skipped_existing, failed = [], [], []
    elapsed_times: List[float] = []
    n_total = len(crawl_ids)

    for i, crawl_id in enumerate(crawl_ids):
        out = output_path_for(crawl_id, output_dir)
        seed = crawl_seed(args.seed, i)
        n_remaining = n_total - i - 1

        # Skip if output already exists (unless --force)
        if not args.force and out.exists() and not args.dry_run:
            log.info("[%d/%d] %s — skipped", i + 1, n_total, crawl_id)
            skipped_existing.append(crawl_id)
            continue

        ok, elapsed = run_crawl(
            crawl_id=crawl_id,
            seed=seed,
            n=args.n,
            output_dir=output_dir,
            max_records=args.max_records,
            sample_files=args.sample_files,
            min_length=args.min_length,
            workers=args.workers,
            dry_run=args.dry_run,
        )

        (succeeded if ok else failed).append(crawl_id)
        if ok and not args.dry_run:
            elapsed_times.append(elapsed)

        if elapsed_times and n_remaining > 0:
            avg = sum(elapsed_times) / len(elapsed_times)
            eta = f"ETA {_fmt_duration(avg * n_remaining)}"
        elif n_remaining == 0:
            eta = "done"
        else:
            eta = "ETA unknown"

        status = "✓" if ok else "✗"
        log.info(
            "[%d/%d] %s  %s — %s  |  %s",
            i + 1, n_total, status, crawl_id, _fmt_duration(elapsed), eta,
        )

    # ── Final summary ─────────────────────────────────────────────────────────
    log.info("=" * 60)
    log.info("Run complete.")
    log.info("  Succeeded     : %d", len(succeeded))
    log.info("  Skipped       : %d  (output already existed)", len(skipped_existing))
    log.info("  Failed        : %d", len(failed))
    if failed:
        log.error("Failed crawls: %s", ", ".join(failed))
        sys.exit(1)


if __name__ == "__main__":
    main()
