#!/usr/bin/env python3
"""
word_rate_plotter.py — Plot target-word slang occurrences per crawl dump from
scored context CSVs (script version of DataProcessingTools/word_rate_plotter.ipynb).

Reads the per-crawl CSVs produced by roberta_filter.py --score-all (columns:
target, uri, target_context, roberta_score; filenames carry the crawl id,
e.g. word_context_CC-MAIN-2019-35.csv), keeps rows with roberta_score >= the
threshold, and plots one line per target word with one data point per crawl
dump (the CC-MAIN year-week mapped to a calendar date).

Values are normalised to occurrences per million tokens of the dump within the
original FineWeb sample (the same sample fineweb_context.py extracted the
contexts from), since the scored context files are already target-filtered and
their sizes don't reflect dump sizes. The per-dump token totals are fetched
once from HuggingFace (dump + token_count columns of the sample parquets) and
cached next to this script; --raw-counts skips normalisation entirely.

Usage:
    python word_rate_plotter.py --threshold 0.5
    python word_rate_plotter.py --threshold 0.8 --top 20 --log -o rates.png
    python word_rate_plotter.py --threshold 0.5 --words epic fire sus
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import platform
import re
import sys
from collections import Counter, defaultdict
from datetime import date
from pathlib import Path
from typing import Dict, List, Optional

import matplotlib.dates as mdates
import matplotlib.pyplot as plt

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stderr)],
)
log = logging.getLogger(__name__)
logging.getLogger("httpx").setLevel(logging.WARNING)

DEFAULT_SCORED_DIR = Path(__file__).resolve().parent / "prompt_scored"
DEFAULT_SIZES_CACHE = Path(__file__).resolve().parent / "fineweb_10BT_dump_sizes.json"
HF_SAMPLE_DIR = "datasets/HuggingFaceFW/fineweb/sample/10BT"
CRAWL_ID_RE = re.compile(r"CC-MAIN-(\d{4})-(\d{2})")


def _crawl_date(stem: str) -> Optional[date]:
    """Map a CC-MAIN-<year>-<week> crawl id to the Monday of that ISO week."""
    m = CRAWL_ID_RE.search(stem)
    return date.fromisocalendar(int(m.group(1)), int(m.group(2)), 1) if m else None


def _use_emoji_font() -> None:
    """Pick a system font with emoji glyphs so emoji targets render."""
    emoji_font = {
        "Darwin": "Apple Color Emoji",
        "Windows": "Segoe UI Emoji",
    }.get(platform.system(), "Noto Color Emoji")
    plt.rcParams["font.family"] = "sans-serif"
    plt.rcParams["font.sans-serif"] = [emoji_font, "DejaVu Sans"]


def load_dump_counts(scored_dir: Path, threshold: float) -> Dict[date, Counter]:
    """Count above-threshold rows per (dump, target) across scored crawl CSVs."""
    counts: Dict[date, Counter] = defaultdict(Counter)
    for path in sorted(scored_dir.glob("*.csv")):
        dump_date = _crawl_date(path.stem)
        if dump_date is None:
            log.warning("Skipping %s — no CC-MAIN-<year>-<week> in filename.", path.name)
            continue
        counts.setdefault(dump_date, Counter())
        with path.open(newline="", encoding="utf-8-sig") as fh:
            for row in csv.DictReader(fh):
                if float(row["roberta_score"]) >= threshold:
                    counts[dump_date][row["target"]] += 1

    n_rows = sum(sum(c.values()) for c in counts.values())
    log.info("Loaded %d crawl dump(s); %d rows >= threshold %.2f",
             len(counts), n_rows, threshold)
    return counts


def fetch_dump_token_counts(cache_path: Path) -> Dict[date, int]:
    """Per-dump token totals of the FineWeb 10BT sample, as {dump date: tokens}.

    Computed once by streaming just the dump + token_count columns of the
    sample's parquet files from HuggingFace (the same sample fineweb_context.py
    extracted the contexts from) and cached as JSON next to this script.

    The fetch is resumable: each file's per-dump totals are written to a
    ``<cache>.partial.json`` sidecar as it finishes, so a sleep/network drop
    only costs the in-flight file. Once every file is present the sidecar is
    collapsed into ``cache_path`` and removed.
    """
    if not cache_path.is_file():
        _fetch_to_cache(cache_path)

    raw = json.loads(cache_path.read_text(encoding="utf-8"))
    sizes: Dict[date, int] = {}
    for dump_id, tok in raw.items():
        d = _crawl_date(dump_id)
        if d is not None:
            sizes[d] = sizes.get(d, 0) + tok
    return sizes


def _fetch_to_cache(cache_path: Path) -> None:
    """Aggregate per-dump token totals from HuggingFace into ``cache_path``."""
    from concurrent.futures import ThreadPoolExecutor, as_completed

    import pyarrow.parquet as pq
    from huggingface_hub import HfFileSystem
    from Keys import HF_TOKEN

    files = sorted(f for f in HfFileSystem(token=HF_TOKEN)
                   .glob(f"{HF_SAMPLE_DIR}/*.parquet"))
    if not files:
        sys.exit(f"No parquet files found at {HF_SAMPLE_DIR} on HuggingFace.")

    # Resume: {filename: {dump_id: tokens}} for files already aggregated.
    partial_path = cache_path.with_suffix(".partial.json")
    done: Dict[str, Dict[str, int]] = (
        json.loads(partial_path.read_text(encoding="utf-8"))
        if partial_path.is_file() else {})
    todo = [f for f in files if f.rsplit("/", 1)[-1] not in done]
    log.info("Aggregating dump token counts from %d sample parquet files on "
             "HuggingFace (one-time; cached to %s); %d done, %d to go ...",
             len(files), cache_path.name, len(done), len(todo))

    def _file_tokens(rel_path: str) -> Dict[str, int]:
        # cache_type="none" + pre_buffer: exact coalesced range reads of just
        # the two needed columns instead of buffered whole-file reads.
        fs = HfFileSystem(token=HF_TOKEN)
        with fs.open(rel_path, cache_type="none") as raw:
            table = pq.read_table(raw, columns=["dump", "token_count"],
                                  pre_buffer=True)
        agg = table.group_by("dump").aggregate([("token_count", "sum")])
        return dict(zip(agg["dump"].to_pylist(), agg["token_count_sum"].to_pylist()))

    with ThreadPoolExecutor(max_workers=8) as pool:
        futures = {pool.submit(_file_tokens, f): f for f in todo}
        for i, fut in enumerate(as_completed(futures), 1):
            name = futures[fut].rsplit("/", 1)[-1]
            done[name] = fut.result()
            partial_path.write_text(json.dumps(done), encoding="utf-8")  # checkpoint
            log.info("  [%d/%d] %s", len(done), len(files), name)

    tokens: Counter = Counter()
    for per_dump in done.values():
        tokens.update(per_dump)
    cache_path.write_text(json.dumps(dict(tokens), indent=1), encoding="utf-8")
    partial_path.unlink()


def _word_series(
    counts: Dict[date, Counter],
    dump_tokens: Optional[Dict[date, int]],
    dumps: List[date],
    target: str,
    smooth: int,
) -> List[float]:
    """Per-dump value series for one word (token-normalised unless raw), smoothed."""
    ys = [counts[d][target] if dump_tokens is None
          else counts[d][target] / dump_tokens[d] * 1_000_000
          for d in dumps]
    if smooth > 1:  # centered moving average over neighbouring dumps
        ys = [sum(w := ys[max(0, i - smooth // 2):i + smooth // 2 + 1]) / len(w)
              for i in range(len(ys))]
    return ys


def _render_chart(
    dumps: List[date],
    series: Dict[str, List[float]],
    dump_tokens: Optional[Dict[date, int]],
    log_scale: bool,
    title: str,
    output: Optional[Path],
) -> None:
    """Draw one chart from precomputed per-word series."""
    fig, ax = plt.subplots(figsize=(12, 6))
    for target, ys in series.items():
        ax.plot(dumps, ys, marker="o", markersize=3, linewidth=1, label=target)

    unit = ("Occurrences in dump" if dump_tokens is None
            else "Occurrences per million sample tokens")
    ax.set_title(title)
    ax.set_xlabel("Crawl dump date")
    ax.set_ylabel(f"{unit}{' (log scale)' if log_scale else ''}")
    if log_scale:
        ax.set_yscale("log")
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    fig.autofmt_xdate()
    ax.legend(bbox_to_anchor=(1.01, 1), loc="upper left", fontsize="small",
              ncols=1 + len(series) // 30)
    ax.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()

    if output:
        plt.savefig(output, dpi=150)
        log.info("Plot saved to %s", output)
    else:
        plt.show()
    plt.close(fig)


def _select_targets(totals: Counter, words: Optional[List[str]], top: Optional[int]) -> List[str]:
    if words:
        missing = [w for w in words if w not in totals]
        if missing:
            log.warning("No above-threshold occurrences for: %s", ", ".join(missing))
        targets = [w for w in words if w in totals]
    else:
        targets = [w for w, _ in totals.most_common(top)]
    if not targets:
        sys.exit("Nothing to plot — no target words above the threshold.")
    return targets


def plot_rates(
    counts: Dict[date, Counter],
    dump_tokens: Optional[Dict[date, int]],
    threshold: float,
    words: Optional[List[str]],
    top: Optional[int],
    smooth: int,
    log_scale: bool,
    output: Optional[Path],
) -> None:
    dumps = sorted(counts)
    totals: Counter = Counter()
    for dump_counts in counts.values():
        totals.update(dump_counts)

    targets = _select_targets(totals, words, top)
    series = {t: _word_series(counts, dump_tokens, dumps, t, smooth) for t in targets}
    title = (f"Slang usage per crawl dump ({len(targets)} words, "
             f"roberta_score >= {threshold:g})")
    _render_chart(dumps, series, dump_tokens, log_scale, title, output)


def plot_segmented_by_peak(
    counts: Dict[date, Counter],
    dump_tokens: Optional[Dict[date, int]],
    threshold: float,
    smooth: int,
    log_scale: bool,
    per_chart: int,
    min_count: int,
    output: Path,
) -> None:
    """One chart per cohort of words sharing a peak-popularity era (<= per_chart each).

    Each word's peak dump is the argmax of its smoothed, token-normalised series;
    words are sorted by peak date and chunked into consecutive groups, so each
    chart shows a contiguous cohort of words that crested around the same time.
    Words with fewer than ``min_count`` total above-threshold hits are dropped as
    noise. Output files are named ``<stem>_peakNN_<lo>_<hi><ext>``.
    """
    dumps = sorted(counts)
    totals: Counter = Counter()
    for dump_counts in counts.values():
        totals.update(dump_counts)

    kept = [w for w, c in totals.items() if c >= min_count]
    dropped = len(totals) - len(kept)
    if not kept:
        sys.exit(f"No words with >= {min_count} above-threshold occurrences.")

    # Peak dump = argmax of each word's smoothed normalised series.
    series = {w: _word_series(counts, dump_tokens, dumps, w, smooth) for w in kept}
    peak = {w: dumps[max(range(len(dumps)), key=lambda i: ys[i])]
            for w, ys in series.items()}
    ordered = sorted(kept, key=lambda w: (peak[w], -totals[w]))
    log.info("Segmenting %d words by peak era (%d dropped: < %d hits); "
             "<= %d per chart -> %d chart(s).",
             len(kept), dropped, min_count, per_chart,
             -(-len(ordered) // per_chart))

    for idx in range(0, len(ordered), per_chart):
        group = ordered[idx:idx + per_chart]
        lo, hi = peak[group[0]], peak[group[-1]]
        n = idx // per_chart + 1
        out = output.with_name(
            f"{output.stem}_peak{n:02d}_{lo:%Y-%m}_{hi:%Y-%m}{output.suffix}")
        title = (f"Slang peaking {lo:%Y-%m} to {hi:%Y-%m} "
                 f"({len(group)} words, roberta_score >= {threshold:g})")
        _render_chart(dumps, {w: series[w] for w in group},
                      dump_tokens, log_scale, title, out)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Plot target-word slang occurrences per crawl dump from scored CSVs.")
    p.add_argument("--threshold", type=float, default=0.0, metavar="F",
                   help="Min roberta_score for a row to count (default: 0.0).")
    p.add_argument("--scored-dir", type=Path, default=DEFAULT_SCORED_DIR,
                   metavar="DIR", dest="scored_dir",
                   help=f"Directory of scored crawl CSVs (default: {DEFAULT_SCORED_DIR}).")
    p.add_argument("--words", nargs="+", metavar="WORD",
                   help="Only plot these target words (default: all).")
    p.add_argument("--top", type=int, metavar="N",
                   help="Only plot the N most frequent target words.")
    p.add_argument("--raw-counts", action="store_true", dest="raw_counts",
                   help="Plot raw per-dump counts instead of occurrences per "
                        "million tokens of the FineWeb sample dump.")
    p.add_argument("--sizes-cache", type=Path, default=DEFAULT_SIZES_CACHE,
                   metavar="FILE", dest="sizes_cache",
                   help="JSON cache of per-dump sample token totals; fetched "
                        f"from HuggingFace if missing (default: {DEFAULT_SIZES_CACHE.name}).")
    p.add_argument("--smooth", type=int, default=1, metavar="N",
                   help="Centered moving average over N dumps to damp per-dump "
                        "noise (default: 1 = off).")
    p.add_argument("--log", action="store_true", dest="log_scale",
                   help="Log-scale the y axis.")
    p.add_argument("--segment-by-peak", action="store_true", dest="segment_by_peak",
                   help="Emit one chart per cohort of words sharing a peak era "
                        "(sorted by peak dump, <= --per-chart words each). "
                        "Requires -o; writes <stem>_peakNN_<lo>_<hi><ext> files.")
    p.add_argument("--per-chart", type=int, default=10, metavar="N", dest="per_chart",
                   help="Max words per chart in --segment-by-peak mode (default: 10).")
    p.add_argument("--min-count", type=int, default=50, metavar="N", dest="min_count",
                   help="In --segment-by-peak mode, drop words with fewer than N "
                        "total above-threshold hits as noise (default: 50).")
    p.add_argument("-o", "--output", type=Path, metavar="FILE",
                   help="Save the plot here instead of displaying it.")
    return p


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    if not args.scored_dir.is_dir():
        parser.error(f"Scored directory not found: {args.scored_dir}")

    counts = load_dump_counts(args.scored_dir, args.threshold)
    if not counts:
        parser.error(f"No crawl CSVs found in {args.scored_dir}.")

    dump_tokens = None
    if not args.raw_counts:
        dump_tokens = fetch_dump_token_counts(args.sizes_cache)
        missing = [d for d in counts if d not in dump_tokens]
        if missing:
            sys.exit(f"No sample token totals for dump(s): "
                     f"{', '.join(d.isoformat() for d in sorted(missing))} — "
                     f"delete {args.sizes_cache} to re-fetch, or use --raw-counts.")

    _use_emoji_font()
    if args.segment_by_peak:
        if args.output is None:
            parser.error("--segment-by-peak writes multiple files; -o is required.")
        plot_segmented_by_peak(counts, dump_tokens, args.threshold, args.smooth,
                               args.log_scale, args.per_chart, args.min_count,
                               args.output)
    else:
        plot_rates(counts, dump_tokens, args.threshold, args.words, args.top,
                   args.smooth, args.log_scale, args.output)


if __name__ == "__main__":
    main()
