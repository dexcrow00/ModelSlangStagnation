#!/usr/bin/env python3
"""
run_pipeline.py — End-to-end slang context collection and filtering pipeline.

Orchestrates word_context.py, bert_filter.py, and run_filter.py in a loop until
every crawl file in --output-dir has at least --n-uris rows that pass the score
filters.

Loop per iteration:
  1. word_context.py  — collect or top-up context samples per crawl
  2. bert_filter.py   — score any new rows (model loaded ONCE per iteration,
                        not once per file); scored files saved to --scored-dir
  3. run_filter.py    — apply thresholds; replace each output file with its
                        filtered version so the next top-up only adds new rows
  Repeat until all crawl files reach --n-uris passing rows, or --max-iters hit.

The scored files in --scored-dir are a persistent audit trail. On subsequent
runs only rows not already present in the scored file are sent to bert_filter,
so re-runs and interrupted jobs continue cheaply from where they left off.

Usage:
    python3 run_pipeline.py \\
        --output-dir contexts/ --n-uris 200 \\
        --words target_words.txt --slang-defs slang.yaml --since 2018

    python3 run_pipeline.py \\
        --output-dir contexts/ --n-uris 200 --words target_words.txt \\
        --from-crawl CC-MAIN-2020-05 --to-crawl CC-MAIN-2022-49
"""

from __future__ import annotations

import argparse
import csv
import logging
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Dict, List

_HERE = Path(__file__).parent

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
# CSV helpers
# ---------------------------------------------------------------------------

_CONTEXT_FIELDS = ["uri", "target_context"]
_SCORED_FIELDS  = ["uri", "target_context", "lang_score", "quality_score", "slang_score"]


def _count_rows(path: Path) -> int:
    if not path.exists():
        return 0
    with path.open(encoding="utf-8") as f:
        return max(0, sum(1 for _ in f) - 1)  # subtract header


def _read_rows(path: Path) -> List[Dict[str, str]]:
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _write_rows(path: Path, rows: List[Dict], fieldnames: List[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)


def _append_rows(path: Path, rows: List[Dict], fieldnames: List[str]) -> None:
    if not rows:
        return
    mode = "a" if path.exists() else "w"
    with path.open(mode, newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        if mode == "w":
            w.writeheader()
        w.writerows(rows)


def _unscored_rows(output_file: Path, scored_file: Path) -> List[Dict[str, str]]:
    """Return rows from output_file whose (uri, target_context) key is absent from scored_file."""
    output_rows = _read_rows(output_file)
    if not output_rows:
        return []
    scored_keys = {
        (r["uri"], r["target_context"])
        for r in _read_rows(scored_file)
    }
    return [r for r in output_rows
            if (r["uri"], r["target_context"]) not in scored_keys]

# ---------------------------------------------------------------------------
# Subprocess runner
# ---------------------------------------------------------------------------

def _run(cmd: List, label: str) -> None:
    log.info("[%s] %s", label, " ".join(str(c) for c in cmd))
    result = subprocess.run(cmd)
    if result.returncode != 0:
        log.error("[%s] exited with code %d", label, result.returncode)
        sys.exit(result.returncode)

# ---------------------------------------------------------------------------
# Step 1 — collect
# ---------------------------------------------------------------------------

def step_collect(args: argparse.Namespace) -> None:
    """Run word_context.py to collect / top-up all crawl files to --n-uris rows."""
    cmd = [
        sys.executable, str(_HERE / "word_context.py"),
        "--words",          args.words,
        "--output-dir",     str(args.output_dir),
        "--n-uris",         str(args.n_uris),
        "--seed",           str(args.seed),
        "--context-window", str(args.context_window),
    ]
    cmd += ["--workers", str(args.workers)]
    if args.sample_files is not None:
        cmd += ["--sample-files", str(args.sample_files)]
    if args.since is not None:
        cmd += ["--since", str(args.since)]
    if args.from_crawl:
        cmd += ["--from-crawl", args.from_crawl]
    if args.to_crawl:
        cmd += ["--to-crawl", args.to_crawl]
    _run(cmd, "collect")

# ---------------------------------------------------------------------------
# Step 2 — score
# ---------------------------------------------------------------------------

def step_score(args: argparse.Namespace) -> int:
    """Score all unscored rows with bert_filter.py.

    Gathers new rows from every crawl file into one batch so the BERT model
    is loaded only once per iteration. Distributes scored rows back to their
    per-crawl scored files in --scored-dir.

    Returns the number of rows scored.
    """
    output_files = sorted(args.output_dir.glob("*CC-MAIN-*.csv"))
    if not output_files:
        log.warning("[score] No CC-MAIN-*.csv files in %s", args.output_dir)
        return 0

    # Gather new (unscored) rows per file, preserving file order
    new_rows_by_file: Dict[str, List[Dict]] = {}
    all_new_rows: List[Dict] = []
    for f in output_files:
        new = _unscored_rows(f, args.scored_dir / f.name)
        new_rows_by_file[f.name] = new
        all_new_rows.extend(new)

    if not all_new_rows:
        log.info("[score] No new rows to score.")
        return 0

    n_with_new = sum(1 for v in new_rows_by_file.values() if v)
    log.info("[score] %d new rows across %d file(s) — running bert_filter.py ...",
             len(all_new_rows), n_with_new)

    with tempfile.TemporaryDirectory() as tmp:
        tmp_in  = Path(tmp) / "batch_input.csv"
        tmp_out = Path(tmp) / "batch_scored.csv"

        _write_rows(tmp_in, all_new_rows, _CONTEXT_FIELDS)

        cmd = [
            sys.executable, str(_HERE / "bert_filter.py"),
            str(tmp_in), "-o", str(tmp_out),
            "--score-all", "--keep-scores",
            "--batch-size", str(args.batch_size),
        ]
        if args.slang_defs:
            cmd += ["--slang-defs", args.slang_defs]
        if args.bert_model:
            cmd += ["--bert-model", args.bert_model]
        _run(cmd, "score")

        # Distribute scored rows back to per-crawl files in the same order they
        # were submitted (bert_filter preserves row order with --score-all).
        scored_rows = _read_rows(tmp_out)
        offset = 0
        for fname, rows in new_rows_by_file.items():
            n = len(rows)
            chunk = scored_rows[offset : offset + n]
            if chunk:
                _append_rows(args.scored_dir / fname, chunk, _SCORED_FIELDS)
                log.info("[score]   %s: appended %d scored rows", fname, len(chunk))
            offset += n

    return len(all_new_rows)

# ---------------------------------------------------------------------------
# Step 3 — filter
# ---------------------------------------------------------------------------

def step_filter(args: argparse.Namespace) -> Dict[str, int]:
    """Filter every scored file via run_filter.py and replace the output files.

    Returns {filename: n_kept_rows}.
    """
    scored_files = sorted(args.scored_dir.glob("*CC-MAIN-*.csv"))
    if not scored_files:
        log.warning("[filter] No scored files found in %s", args.scored_dir)
        return {}

    counts: Dict[str, int] = {}
    for scored_file in scored_files:
        output_file = args.output_dir / scored_file.name

        with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as tmp:
            tmp_filtered = Path(tmp.name)

        try:
            cmd = [
                sys.executable, str(_HERE / "run_filter.py"),
                str(scored_file), "-o", str(tmp_filtered),
                "--lang-threshold",    str(args.lang_threshold),
                "--quality-threshold", str(args.quality_threshold),
                "--slang-threshold",   str(args.slang_threshold),
            ]
            _run(cmd, f"filter({scored_file.name})")

            shutil.copy(tmp_filtered, output_file)
            n = _count_rows(output_file)
            counts[scored_file.name] = n
            log.info("[filter]   %s: %d rows kept", scored_file.name, n)
        finally:
            tmp_filtered.unlink(missing_ok=True)

    return counts

# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------

def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    if args.scored_dir is None:
        args.scored_dir = args.output_dir.parent / (args.output_dir.name + "_scored")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.scored_dir.mkdir(parents=True, exist_ok=True)

    # Validate that required scripts exist
    for script in ["word_context.py", "bert_filter.py", "run_filter.py"]:
        if not (_HERE / script).exists():
            parser.error(f"Required script not found: {_HERE / script}")

    log.info("Pipeline starting.")
    log.info("  output-dir  : %s", args.output_dir)
    log.info("  scored-dir  : %s", args.scored_dir)
    log.info("  n-uris      : %d", args.n_uris)
    log.info("  max-iters   : %d", args.max_iters)

    for iteration in range(1, args.max_iters + 1):
        log.info("")
        log.info("--- Iteration %d / %d ---", iteration, args.max_iters)

        log.info("[1/3] Collecting context samples ...")
        step_collect(args)

        log.info("[2/3] Scoring new rows ...")
        step_score(args)

        log.info("[3/3] Filtering and replacing output files ...")
        counts = step_filter(args)

        # Progress check
        all_files = sorted(args.output_dir.glob("*CC-MAIN-*.csv"))
        n_done  = sum(1 for f in all_files if _count_rows(f) >= args.n_uris)
        n_total = len(all_files)

        log.info("")
        log.info("Progress: %d / %d crawl files have >= %d passing rows",
                 n_done, n_total, args.n_uris)

        if n_total > 0 and n_done >= 1:
            done = [f for f in all_files if _count_rows(f) >= args.n_uris]
            log.info("First crawl file complete: %s", done[0].name)
            break

        if iteration == args.max_iters:
            best = max(all_files, key=_count_rows, default=None)
            log.warning(
                "Reached max iterations (%d). No file reached %d rows. Best: %s (%d rows)",
                args.max_iters, args.n_uris,
                best.name if best else "none", _count_rows(best) if best else 0,
            )

    log.info("Pipeline done.")

# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Full slang context collection and filtering pipeline.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run_pipeline.py \\
      --output-dir contexts/ --n-uris 200 \\
      --words target_words.txt --slang-defs slang.yaml --since 2018

  python run_pipeline.py \\
      --output-dir contexts/ --n-uris 200 --words target_words.txt \\
      --from-crawl CC-MAIN-2020-05 --to-crawl CC-MAIN-2022-49
        """,
    )

    # Directories
    parser.add_argument(
        "--output-dir", required=True, metavar="DIR", type=Path, dest="output_dir",
        help="Directory for context CSVs. Files here are replaced with filtered "
             "versions after each pass, so word_context.py can top them up.",
    )
    parser.add_argument(
        "--scored-dir", default=None, metavar="DIR", type=Path, dest="scored_dir",
        help="Directory for scored CSVs (all rows + score columns). "
             "Defaults to <output-dir>_scored. Kept as an audit trail across runs.",
    )

    # Pipeline target
    parser.add_argument(
        "--n-uris", type=int, required=True, metavar="N", dest="n_uris",
        help="Target number of passing rows per crawl file.",
    )
    parser.add_argument(
        "--max-iters", type=int, default=10, metavar="N", dest="max_iters",
        help="Maximum loop iterations before stopping (default: 10).",
    )

    # word_context.py passthrough
    parser.add_argument(
        "--words", default="target_words.txt", metavar="FILE",
        help="Plain-text file of target slang words (default: target_words.txt).",
    )
    parser.add_argument(
        "--since", type=int, default=None, metavar="YEAR",
        help="Only process crawls from this year onward.",
    )
    parser.add_argument(
        "--from-crawl", default=None, metavar="ID", dest="from_crawl",
        help="Only process crawls at or after this ID (e.g. CC-MAIN-2020-05).",
    )
    parser.add_argument(
        "--to-crawl", default=None, metavar="ID", dest="to_crawl",
        help="Only process crawls up to and including this ID.",
    )
    parser.add_argument(
        "--seed", type=int, default=42, metavar="N",
        help="Random seed passed to word_context.py (default: 42).",
    )
    parser.add_argument(
        "--workers", type=int, default=16, metavar="N",
        help="Parallel workers for word_context.py (default: 16).",
    )
    parser.add_argument(
        "--sample-files", type=int, default=None, metavar="M", dest="sample_files",
        help="WET files to sample per crawl dump (default: word_context.py default).",
    )
    parser.add_argument(
        "--context-window", type=int, default=10, metavar="K", dest="context_window",
        help="Token context window on each side of a match (default: 10).",
    )

    # bert_filter.py passthrough
    parser.add_argument(
        "--slang-defs", default=None, metavar="YAML", dest="slang_defs",
        help="Path to slang.yaml for slang-sense filtering. Slang filter is "
             "skipped if omitted.",
    )
    parser.add_argument(
        "--bert-model", default=None, metavar="ID", dest="bert_model",
        help="Override the BERT checkpoint used by bert_filter.py.",
    )
    parser.add_argument(
        "--batch-size", type=int, default=32, metavar="N", dest="batch_size",
        help="Inference batch size for bert_filter.py (default: 32).",
    )

    # run_filter.py thresholds
    parser.add_argument(
        "--lang-threshold", type=float, default=0.85, metavar="F",
        dest="lang_threshold",
        help="Min lang_score to keep (default: 0.85).",
    )
    parser.add_argument(
        "--quality-threshold", type=float, default=0.70, metavar="F",
        dest="quality_threshold",
        help="Min quality_score to keep (default: 0.70).",
    )
    parser.add_argument(
        "--slang-threshold", type=float, default=0.5, metavar="F",
        dest="slang_threshold",
        help="Min slang_score to keep (default: 0.5).",
    )

    return parser


if __name__ == "__main__":
    main()
