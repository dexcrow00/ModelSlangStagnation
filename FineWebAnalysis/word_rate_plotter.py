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

--percent-of-peak rescales every word to its own maximum instead: each series is
plotted as a percentage of that word's highest (smoothed) dump value, so words
of very different absolute frequency can be compared by the shape of their rise
and fall. Confidence bands are scaled by the same per-word divisor.

With --confidence P, each line gets a shaded Poisson sampling-uncertainty band:
the 10BT sample is a random draw from FineWeb, so a per-dump occurrence count k
is k ~ Poisson(rate * tokens); the exact Poisson interval on the (window-pooled)
count bounds the true rate. Bands are wide for rare words, tight for common ones.

Usage:
    python word_rate_plotter.py --threshold 0.5
    python word_rate_plotter.py --threshold 0.8 --top 20 --log -o rates.png
    python word_rate_plotter.py --threshold 0.5 --words epic fire sus --confidence 0.95
    python word_rate_plotter.py --threshold 0.99 --highlight-bands   # -> ../writing/highlight_*_count.png
    python word_rate_plotter.py --threshold 0.5 --words lol aura --percent-of-peak
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
from scipy.stats import chi2

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stderr)],
)
log = logging.getLogger(__name__)
logging.getLogger("httpx").setLevel(logging.WARNING)

DEFAULT_SCORED_DIR = Path(__file__).resolve().parent / "prompt_scored"
SCENARIO_SCORED_DIR = Path(__file__).resolve().parent / "scenario_prompt_scored"
DEFAULT_SIZES_CACHE = Path(__file__).resolve().parent / "fineweb_10BT_dump_sizes.json"
WRITING_DIR = (Path(__file__).resolve().parent / ".." / "writing").resolve()
HF_SAMPLE_DIR = "datasets/HuggingFaceFW/fineweb/sample/10BT"
CRAWL_ID_RE = re.compile(r"CC-MAIN-(\d{4})-(\d{2})")

# --highlight-bands mode: fixed cohorts of ephemeral words (pooled from both
# prompt_scored and scenario_prompt_scored), each saved as
# highlight_<band>_count.png. The pre-2018 band gets a broken y-axis so lol
# (peak ~16/M) does not flatten the rest (peak <4/M).
HIGHLIGHT_BANDS = {
    "pre2018":    ["lol", "sick", "troll", "bro", "swag", "omg", "meh", "lmao"],
    "around2020": ["alpha", "red pill"],
    "2022_24":    ["slay", "gaslight", "glow-up", "aura", "lowkey", "situationship",
                   "vibe", "vibes", "legit"],
}
HIGHLIGHT_TITLES = {
    "pre2018":    "Pre-2018 band (declining)",
    "around2020": "Around-2020 band",
    "2022_24":    "2022--2024 band (recent risers)",
}
# broken-axis split per band: (top_lo, top_hi, bot_lo, bot_hi); absent => single axis.
HIGHLIGHT_BROKEN = {"pre2018": (4.5, 17.0, 0.0, 4.4)}


# Fixed categorical hues for --clean, assigned in this order (never cycled): a
# chart with more series than this keeps matplotlib's default cycle instead.
CLEAN_COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100",
                "#e87ba4", "#008300", "#4a3aa7", "#e34948"]


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


def _poisson_ci(k: int, confidence: float) -> tuple[float, float]:
    """Exact (Garwood) Poisson confidence interval for an observed count ``k``.

    The 10BT sample is a random draw from FineWeb, so an observed count k in T
    tokens is k ~ Poisson(rate * T); this interval on k therefore bounds the
    true rate given the sample.
    """
    alpha = 1.0 - confidence
    lower = chi2.ppf(alpha / 2, 2 * k) / 2 if k > 0 else 0.0
    upper = chi2.ppf(1.0 - alpha / 2, 2 * k + 2) / 2
    return lower, upper


def _word_series_ci(
    counts: Dict[date, Counter],
    dump_tokens: Optional[Dict[date, int]],
    dumps: List[date],
    target: str,
    smooth: int,
    confidence: Optional[float],
) -> tuple[List[float], List[float], List[float]]:
    """Central line and, if ``confidence`` is set, Poisson CI band for one word.

    Counts are pooled over the centered smoothing window, then scaled to the
    plotted unit (per the pooled token total when normalised, else averaged over
    the window's dumps for raw counts). The CI is the exact Poisson interval on
    the pooled count scaled the same way, so it is centered on the line and
    tightens as a wider window aggregates more of the sample. For ``smooth=1``
    the central line is identical to the unsmoothed per-dump value.
    """
    half = smooth // 2
    central: List[float] = []
    lower: List[float] = []
    upper: List[float] = []
    for i in range(len(dumps)):
        window = dumps[max(0, i - half):i + half + 1]
        k = sum(counts[d][target] for d in window)
        scale = (1_000_000 / sum(dump_tokens[d] for d in window)
                 if dump_tokens is not None else 1.0 / len(window))
        central.append(k * scale)
        if confidence is not None:
            lo, hi = _poisson_ci(k, confidence)
            lower.append(lo * scale)
            upper.append(hi * scale)
    return central, lower, upper


def _build_series(
    counts: Dict[date, Counter],
    dump_tokens: Optional[Dict[date, int]],
    dumps: List[date],
    targets: List[str],
    smooth: int,
    confidence: Optional[float],
    percent_of_peak: bool = False,
    min_change: Optional[float] = None,
    max_ci: Optional[float] = None,
) -> tuple[Dict[str, List[float]], Optional[Dict[str, tuple]]]:
    """Central series per word, plus the matching CI bands when ``confidence`` is set.

    Returns ``(series, bands)`` ready to hand to the renderers; ``bands`` is None
    when no confidence level was requested, which is how the renderers decide
    whether to shade at all. With ``percent_of_peak`` each word is rescaled to a
    percentage of its own maximum (see --percent-of-peak), and with
    ``min_change`` only words that vary by at least that percentage of their own
    peak are kept (see --min-change), and with ``max_ci`` only words whose
    confidence band stays narrower than that width (see --max-ci).
    """
    series: Dict[str, List[float]] = {}
    bands: Optional[Dict[str, tuple]] = {} if confidence else None
    for t in targets:
        central, lo, hi = _word_series_ci(counts, dump_tokens, dumps, t, smooth, confidence)
        if percent_of_peak:
            # Rescale to a percentage of this word's own peak dump, so words of
            # very different absolute frequency are comparable by shape. The CI
            # band is divided by the same peak, keeping it centred on the line.
            peak = max(central, default=0.0)
            if peak > 0:
                central = [100.0 * y / peak for y in central]
                lo = [100.0 * y / peak for y in lo]
                hi = [100.0 * y / peak for y in hi]
        series[t] = central
        if bands is not None:
            bands[t] = (lo, hi)
    if min_change is not None:
        series, bands = _filter_by_change(series, bands, min_change)
    if max_ci is not None:
        series, bands = _filter_by_ci(series, bands, max_ci)
    return series, bands


def _filter_by_ci(
    series: Dict[str, List[float]],
    bands: Optional[Dict[str, tuple]],
    max_ci: float,
) -> tuple[Dict[str, List[float]], Optional[Dict[str, tuple]]]:
    """Keep only words whose CI band never spans more than ``max_ci`` at a drawn dump.

    Width is measured in the plotted y unit --- percentage points of the word's
    own peak under --percent-of-peak, occurrences per million otherwise --- and
    only at dumps that are actually drawn (a zero dump is dropped by the
    renderers, so its band is not on the chart either). A wide band means the
    shape is mostly sampling noise, which is what this screens out.
    """
    if bands is None:
        sys.exit("--max-ci needs confidence bands — pass --confidence P (e.g. 0.95).")
    kept = {}
    for w, ys in series.items():
        lo, hi = bands[w]
        widths = [h - l for y, l, h in zip(ys, lo, hi) if y > 0]
        if widths and max(widths) <= max_ci:
            kept[w] = ys
    dropped = len(series) - len(kept)
    if dropped:
        log.info("--max-ci %g: kept %d word(s), dropped %d whose confidence band "
                 "exceeds that width at some dump.", max_ci, len(kept), dropped)
    if not kept:
        sys.exit(f"No words keep a confidence band within {max_ci:g} — raise --max-ci, "
                 f"widen --smooth, or lower --confidence.")
    return kept, {w: bands[w] for w in kept}


def _filter_by_change(
    series: Dict[str, List[float]],
    bands: Optional[Dict[str, tuple]],
    min_change: float,
) -> tuple[Dict[str, List[float]], Optional[Dict[str, tuple]]]:
    """Keep only words whose series falls at least ``min_change``% below its own peak.

    The test is on the smoothed series and ignores dumps with no occurrences (a
    zero is "not seen in this sample", not a real trough --- the renderers drop
    those points too). It is scale-free, so it gives the same answer before or
    after --percent-of-peak rescaling: a word qualifies when its smallest non-zero
    dump sits below (100 - min_change)% of its largest.
    """
    floor = 1.0 - min_change / 100.0
    kept = {}
    for w, ys in series.items():
        nz = [y for y in ys if y > 0]
        if len(nz) >= 2 and min(nz) < floor * max(nz):
            kept[w] = ys
    dropped = len(series) - len(kept)
    if dropped:
        log.info("--min-change %g%%: kept %d word(s), dropped %d that never fall "
                 "below %g%% of their peak.", min_change, len(kept), dropped, 100 * floor)
    if not kept:
        sys.exit(f"No words vary by >= {min_change:g}% of their peak — lower --min-change.")
    return kept, ({w: bands[w] for w in kept} if bands else None)


def _apply_clean_style(ax, n_series: int) -> None:
    """Recessive chrome for --clean: fixed hues, hairline y-grid, no boxed-in axes."""
    if n_series <= len(CLEAN_COLORS):
        ax.set_prop_cycle(color=CLEAN_COLORS[:n_series])
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.grid(True, axis="y", color="#e1e0d9", linewidth=0.8)
    ax.set_axisbelow(True)


def _draw_series(ax, dumps, series, bands=None, clean=False):
    """Draw one markered line (+ optional CI band) per word on ``ax``.

    Dumps where a word has zero occurrences are dropped rather than plotted at 0:
    a 0 is invalid on a log axis (and a misleading dive to the floor on a linear
    one), so we skip those points and let the line connect the neighboring dumps.
    """
    for target, ys in series.items():
        kept = [(d, y) for d, y in zip(dumps, ys) if y > 0]
        if not kept:
            continue  # word has no non-zero dumps to draw
        xs_t, ys_t = zip(*kept)
        # --clean drops the per-dump markers: at 95 dumps they read as noise, and
        # the line already shows where the data is.
        line, = ax.plot(xs_t, ys_t, marker="" if clean else "o", markersize=3,
                        linewidth=1.6 if clean else 1, label=target)
        if bands and target in bands:
            lo, hi = bands[target]
            # Mask the band to the same non-zero dumps so it tracks the line.
            band = [(d, l, h) for d, y, l, h in zip(dumps, ys, lo, hi) if y > 0]
            if band:
                bx, bl, bh = zip(*band)
                ax.fill_between(bx, bl, bh, color=line.get_color(),
                                alpha=0.18, linewidth=0)


def _settings_note(threshold: float, n_dirs: int, smooth: int,
                   dump_tokens: Optional[Dict[date, int]], percent_of_peak: bool,
                   confidence: Optional[float], min_change: Optional[float],
                   max_ci: Optional[float]) -> str:
    """One-line summary of how the figure was produced, drawn in its bottom margin."""
    parts = [f"word_rate_plotter.py \u00b7 roberta_score \u2265 {threshold:g}",
             "target + scenario contexts pooled" if n_dirs > 1 else "target contexts",
             "occurrences per million sample tokens" if dump_tokens is not None
             else "raw per-dump counts"]
    if smooth > 1:
        parts.append(f"{smooth}-dump centred moving average")
    if percent_of_peak:
        parts.append("rescaled to % of each word's peak dump")
    if confidence:
        parts.append(f"{confidence:.0%} Poisson CI")
    if min_change is not None:
        parts.append(f"words dropping \u2265 {min_change:g}% below peak")
    if max_ci is not None:
        parts.append(f"CI width \u2264 {max_ci:g}")
    return "  \u00b7  ".join(parts)


def _unit_label(dump_tokens: Optional[Dict[date, int]],
                percent_of_peak: bool = False) -> str:
    """Y-axis unit: % of each word's peak, raw per-dump counts, or the normalised rate."""
    if percent_of_peak:
        return ("% of the word's peak dump"
                f" {'count' if dump_tokens is None else 'rate'}")
    return ("Occurrences in dump" if dump_tokens is None
            else "Occurrences per million sample tokens")


def _render_chart(
    dumps: List[date],
    series: Dict[str, List[float]],
    dump_tokens: Optional[Dict[date, int]],
    log_scale: bool,
    title: str,
    output: Optional[Path],
    bands: Optional[Dict[str, tuple]] = None,
    percent_of_peak: bool = False,
    clean: bool = False,
    note: Optional[str] = None,
) -> None:
    """Draw one chart from precomputed per-word series (with optional CI bands)."""
    fig, ax = plt.subplots(figsize=(12, 6))
    if clean:
        _apply_clean_style(ax, len(series))
    _draw_series(ax, dumps, series, bands, clean)

    ax.set_title(title)
    ax.set_xlabel("Crawl dump date")
    ax.set_ylabel(f"{_unit_label(dump_tokens, percent_of_peak)}"
                  f"{' (log2 scale)' if log_scale else ''}")
    if log_scale:
        ax.set_yscale("log", base=2)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    fig.autofmt_xdate()
    ax.legend(bbox_to_anchor=(1.01, 1), loc="upper left", fontsize="small",
              ncols=1 + len(series) // 30, frameon=not clean)
    if not clean:
        ax.grid(True, linestyle="--", alpha=0.5)
    if note:
        # Reserve a strip under the x-label so the note never lands on the axes.
        fig.tight_layout(rect=(0, 0.055, 1, 1))
        fig.text(0.008, 0.015, note, fontsize=7, color="#898781", ha="left", va="bottom")
    else:
        fig.tight_layout()

    if output:
        output.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output, dpi=150)
        log.info("Plot saved to %s", output)
    else:
        plt.show()
    plt.close(fig)


def _render_broken(
    dumps: List[date],
    series: Dict[str, List[float]],
    dump_tokens: Optional[Dict[date, int]],
    title: str,
    output: Path,
    bands: Optional[Dict[str, tuple]],
    split: tuple,
    percent_of_peak: bool = False,
    clean: bool = False,
    note: Optional[str] = None,
) -> None:
    """Like _render_chart, but split over two y-scales with an axis break so one
    dominant word (e.g. lol) does not flatten the rest. Every word is drawn on
    both panels; the y-limits ``split`` = (top_lo, top_hi, bot_lo, bot_hi) decide
    which panel each is visible in."""
    top_lo, top_hi, bot_lo, bot_hi = split
    fig, (top, bot) = plt.subplots(
        2, 1, sharex=True, figsize=(12, 6),
        gridspec_kw={"height_ratios": [1, 2.3], "hspace": 0.07})
    if clean:
        for ax in (top, bot):
            _apply_clean_style(ax, len(series))
    _draw_series(top, dumps, series, bands, clean)
    _draw_series(bot, dumps, series, bands, clean)
    top.set_ylim(top_lo, top_hi)
    bot.set_ylim(bot_lo, bot_hi)

    top.spines["bottom"].set_visible(False)
    bot.spines["top"].set_visible(False)
    top.tick_params(bottom=False)
    d = 0.008  # diagonal break marks straddling the cut
    kw = dict(transform=top.transAxes, color="k", clip_on=False, lw=0.8)
    top.plot((-d, +d), (-d * 2.3, +d * 2.3), **kw)
    top.plot((1 - d, 1 + d), (-d * 2.3, +d * 2.3), **kw)
    kw.update(transform=bot.transAxes)
    bot.plot((-d, +d), (1 - d, 1 + d), **kw)
    bot.plot((1 - d, 1 + d), (1 - d, 1 + d), **kw)

    top.set_title(title)
    bot.set_xlabel("Crawl dump date")
    fig.supylabel(_unit_label(dump_tokens, percent_of_peak))
    bot.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    fig.autofmt_xdate()
    if not clean:
        for ax in (top, bot):
            ax.grid(True, linestyle="--", alpha=0.5)
    else:
        top.spines["bottom"].set_visible(False)   # re-hide: clean style restores it
        bot.spines["top"].set_visible(False)
    top.legend(bbox_to_anchor=(1.01, 1), loc="upper left", fontsize="small",
               frameon=not clean)

    if note:
        fig.text(0.008, 0.005, note, fontsize=7, color="#898781", ha="left", va="bottom")
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=150, bbox_inches="tight")
    log.info("Plot saved to %s", output)
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
    confidence: Optional[float] = None,
    percent_of_peak: bool = False,
    min_change: Optional[float] = None,
    max_ci: Optional[float] = None,
    clean: bool = False,
    note: Optional[str] = None,
) -> None:
    dumps = sorted(counts)
    totals: Counter = Counter()
    for dump_counts in counts.values():
        totals.update(dump_counts)

    targets = _select_targets(totals, words, top)
    series, bands = _build_series(counts, dump_tokens, dumps, targets, smooth,
                                  confidence, percent_of_peak, min_change, max_ci)
    targets = [t for t in targets if t in series]
    pct_note = ", % of each word's peak" if percent_of_peak else ""
    title = (f"Slang usage per crawl dump ({len(targets)} words, "
             f"roberta_score >= {threshold:g}"
             f"{f', {confidence:.0%} CI' if confidence else ''}"
             f"{pct_note})")
    _render_chart(dumps, series, dump_tokens, log_scale, title, output, bands,
                  percent_of_peak, clean, note)


def plot_segmented_by_peak(
    counts: Dict[date, Counter],
    dump_tokens: Optional[Dict[date, int]],
    threshold: float,
    smooth: int,
    log_scale: bool,
    per_chart: int,
    min_count: int,
    output: Path,
    confidence: Optional[float] = None,
    percent_of_peak: bool = False,
    min_change: Optional[float] = None,
    max_ci: Optional[float] = None,
    clean: bool = False,
    note: Optional[str] = None,
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
    series, bands = _build_series(counts, dump_tokens, dumps, kept, smooth,
                                  confidence, percent_of_peak, min_change, max_ci)
    kept = [w for w in kept if w in series]
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
                 f"({len(group)} words, roberta_score >= {threshold:g}"
                 f"{f', {confidence:.0%} CI' if confidence else ''})")
        _render_chart(dumps, {w: series[w] for w in group},
                      dump_tokens, log_scale, title, out,
                      {w: bands[w] for w in group} if bands else None,
                      percent_of_peak, clean, note)


def plot_highlight_bands(
    counts: Dict[date, Counter],
    dump_tokens: Optional[Dict[date, int]],
    threshold: float,
    smooth: int,
    out_dir: Path,
    confidence: Optional[float] = None,
    percent_of_peak: bool = False,
    min_change: Optional[float] = None,
    max_ci: Optional[float] = None,
    clean: bool = False,
    note: Optional[str] = None,
) -> None:
    """Emit the fixed ephemeral-word band figures, one PNG per band.

    Each band (HIGHLIGHT_BANDS) is a cohort of words on one linear-axis chart;
    bands listed in HIGHLIGHT_BROKEN get a split y-axis so a dominant word does
    not flatten the rest. Files are written as ``highlight_<band>_count.png``
    (``highlight_<band>_pct.png`` under --percent-of-peak, whose rescaling makes
    the fixed broken-axis limits meaningless, so the split is skipped).
    """
    dumps = sorted(counts)
    for band, words in HIGHLIGHT_BANDS.items():
        present = [w for w in words if any(counts[d].get(w) for d in dumps)]
        missing = [w for w in words if w not in present]
        if missing:
            log.warning("Band %s: no above-threshold occurrences for %s",
                        band, ", ".join(missing))
        series, bands = _build_series(counts, dump_tokens, dumps, present, smooth,
                                      confidence, percent_of_peak, min_change, max_ci)
        out = out_dir / f"highlight_{band}_{'pct' if percent_of_peak else 'count'}.png"
        title = HIGHLIGHT_TITLES[band]
        if band in HIGHLIGHT_BROKEN and not percent_of_peak:
            _render_broken(dumps, series, dump_tokens, title, out, bands,
                           HIGHLIGHT_BROKEN[band], percent_of_peak, clean, note)
        else:
            _render_chart(dumps, series, dump_tokens, False, title, out, bands,
                          percent_of_peak, clean, note)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Plot target-word slang occurrences per crawl dump from scored CSVs.")
    p.add_argument("--threshold", type=float, default=0.0, metavar="F",
                   help="Min roberta_score for a row to count (default: 0.0).")
    p.add_argument("--scored-dir", type=Path, nargs="+", default=None,
                   metavar="DIR", dest="scored_dir",
                   help="One or more directories of scored crawl CSVs (pooled). "
                        f"Default: {DEFAULT_SCORED_DIR.name} (or both "
                        f"{DEFAULT_SCORED_DIR.name} and {SCENARIO_SCORED_DIR.name} "
                        "with --highlight-bands).")
    p.add_argument("--words", nargs="+", metavar="WORD",
                   help="Only plot these target words (default: all).")
    p.add_argument("--top", type=int, metavar="N",
                   help="Only plot the N most frequent target words.")
    p.add_argument("--raw-counts", action="store_true", dest="raw_counts",
                   help="Plot raw per-dump counts instead of occurrences per "
                        "million tokens of the FineWeb sample dump.")
    p.add_argument("--percent-of-peak", action="store_true", dest="percent_of_peak",
                   help="Plot each word as a percentage of its own peak dump value "
                        "instead of an absolute rate, so words of very different "
                        "frequency can be compared by shape. Confidence bands are "
                        "scaled by the same per-word peak.")
    p.add_argument("--min-change", type=float, default=None, metavar="PCT",
                   help="Only plot words whose (smoothed) rate drops at least PCT%% "
                        "below its own peak at some dump — i.e. words that actually "
                        "rise or fall over time. Zero dumps are ignored.")
    p.add_argument("--max-ci", type=float, default=None, metavar="WIDTH",
                   help="Only plot words whose confidence band never spans more than "
                        "WIDTH at any drawn dump, in the plotted y unit (percentage "
                        "points of the word's peak under --percent-of-peak). Screens "
                        "out words whose shape is mostly sampling noise. Requires "
                        "--confidence.")
    p.add_argument("--no-ci", action="store_true", dest="no_ci",
                   help="Never shade confidence bands, overriding the 95%% default "
                        "that --highlight-bands would otherwise apply.")
    p.add_argument("--clean", action="store_true",
                   help="Recessive chart styling: no per-dump markers, thicker lines, "
                        "fixed categorical hues, hairline y-grid only, and no boxed-in "
                        "axes or legend frame.")
    p.add_argument("--settings-note", action="store_true", dest="settings_note",
                   help="Print a one-line summary of the settings used (threshold, "
                        "pooling, normalisation, smoothing, filters) in the figure's "
                        "bottom margin, so the chart is self-documenting.")
    p.add_argument("--sizes-cache", type=Path, default=DEFAULT_SIZES_CACHE,
                   metavar="FILE", dest="sizes_cache",
                   help="JSON cache of per-dump sample token totals; fetched "
                        f"from HuggingFace if missing (default: {DEFAULT_SIZES_CACHE.name}).")
    p.add_argument("--smooth", type=int, default=None, metavar="N",
                   help="Centered moving average over N dumps to damp per-dump "
                        "noise (default: 1 = off; 5 with --highlight-bands).")
    p.add_argument("--log", action="store_true", dest="log_scale",
                   help="Log-scale the y axis.")
    p.add_argument("--confidence", type=float, default=None, metavar="P",
                   help="Shade a Poisson sampling-uncertainty band at confidence "
                        "level P (e.g. 0.95). Each per-dump count is treated as a "
                        "Poisson draw from the 10BT sample, so bands are wide for "
                        "rare words and tight for common ones, and tighten with --smooth.")
    p.add_argument("--segment-by-peak", action="store_true", dest="segment_by_peak",
                   help="Emit one chart per cohort of words sharing a peak era "
                        "(sorted by peak dump, <= --per-chart words each). "
                        "Requires -o; writes <stem>_peakNN_<lo>_<hi><ext> files.")
    p.add_argument("--per-chart", type=int, default=10, metavar="N", dest="per_chart",
                   help="Max words per chart in --segment-by-peak mode (default: 10).")
    p.add_argument("--highlight-bands", action="store_true", dest="highlight_bands",
                   help="Emit the fixed ephemeral-word band figures "
                        "(highlight_<band>_count.png), with a broken y-axis for the "
                        "pre-2018 band. Pools prompt_scored + scenario_prompt_scored "
                        "and defaults to --smooth 5, 95%% CI; -o sets the output "
                        f"directory (default: {WRITING_DIR.name}/).")
    p.add_argument("--min-count", type=int, default=50, metavar="N", dest="min_count",
                   help="In --segment-by-peak mode, drop words with fewer than N "
                        "total above-threshold hits as noise (default: 50).")
    p.add_argument("-o", "--output", type=Path, metavar="FILE",
                   help="Save the plot here instead of displaying it.")
    return p


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    if args.scored_dir is None:
        args.scored_dir = ([DEFAULT_SCORED_DIR, SCENARIO_SCORED_DIR]
                           if args.highlight_bands else [DEFAULT_SCORED_DIR])
    for d in args.scored_dir:
        if not d.is_dir():
            parser.error(f"Scored directory not found: {d}")

    # --highlight-bands defaults: 5-dump smoothing and a 95% CI band.
    smooth = args.smooth if args.smooth is not None else (5 if args.highlight_bands else 1)
    confidence = args.confidence
    if confidence is None and args.highlight_bands:
        confidence = 0.95
    if args.no_ci:
        confidence = None
    if confidence is not None and not 0 < confidence < 1:
        parser.error("--confidence must be between 0 and 1 (e.g. 0.95).")

    counts: Dict[date, Counter] = defaultdict(Counter)
    for d in args.scored_dir:
        for dump_date, cnt in load_dump_counts(d, args.threshold).items():
            counts[dump_date].update(cnt)
    if not counts:
        parser.error("No crawl CSVs found in "
                     f"{', '.join(str(d) for d in args.scored_dir)}.")

    dump_tokens = None
    if not args.raw_counts:
        dump_tokens = fetch_dump_token_counts(args.sizes_cache)
        missing = [d for d in counts if d not in dump_tokens]
        if missing:
            sys.exit(f"No sample token totals for dump(s): "
                     f"{', '.join(d.isoformat() for d in sorted(missing))} — "
                     f"delete {args.sizes_cache} to re-fetch, or use --raw-counts.")

    note = (_settings_note(args.threshold, len(args.scored_dir), smooth, dump_tokens,
                           args.percent_of_peak, confidence, args.min_change, args.max_ci)
            if args.settings_note else None)

    _use_emoji_font()
    if args.highlight_bands:
        out_dir = args.output if args.output is not None else WRITING_DIR
        plot_highlight_bands(counts, dump_tokens, args.threshold, smooth,
                             out_dir, confidence, args.percent_of_peak,
                             args.min_change, args.max_ci, args.clean, note)
    elif args.segment_by_peak:
        if args.output is None:
            parser.error("--segment-by-peak writes multiple files; -o is required.")
        plot_segmented_by_peak(counts, dump_tokens, args.threshold, smooth,
                               args.log_scale, args.per_chart, args.min_count,
                               args.output, confidence, args.percent_of_peak,
                               args.min_change, args.max_ci, args.clean, note)
    else:
        plot_rates(counts, dump_tokens, args.threshold, args.words, args.top,
                   smooth, args.log_scale, args.output, confidence,
                   args.percent_of_peak, args.min_change, args.max_ci, args.clean, note)


if __name__ == "__main__":
    main()
