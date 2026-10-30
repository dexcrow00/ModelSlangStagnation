#!/usr/bin/env python3
"""
word_rate_plotter.py — Plot the slang occurrences of the words in HIGHLIGHT_BANDS per crawl dump from
scored context CSVs, one figure per band (highlight_<band>_count.png).

Reads the per-crawl CSVs produced by roberta_filter.py --score-all (columns:
target, uri, target_context, roberta_score; filenames carry the crawl id,
e.g. word_context_CC-MAIN-2019-35.csv), keeps rows with roberta_score >=
THRESHOLD, and plots one line per word with one data point per crawl dump (the
CC-MAIN year-week mapped to a calendar date), smoothed over SMOOTH dumps with a
95% confidence band by default. The word list is chosen upstream (see
plot_min_pct.py) and hard-coded in HIGHLIGHT_BANDS below.

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
    python word_rate_plotter.py                      # -> ../writing/highlight_*_count.png
    python word_rate_plotter.py --percent-of-peak    # -> ../writing/highlight_*_pct.png
    python word_rate_plotter.py --clean --settings-note -o figures/
    python word_rate_plotter.py --polynomial-fit     # -> ../writing/highlight_*_count_polynomial_fit.png
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
import numpy as np
from scipy.stats import chi2

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stderr)],
)
log = logging.getLogger(__name__)
logging.getLogger("httpx").setLevel(logging.WARNING)

ROOT = Path(__file__).resolve().parents[1]  # FineWebAnalysis/
DEFAULT_SCORED_DIR = ROOT / "prompt_scored"
SCENARIO_SCORED_DIR = ROOT / "scenario_prompt_scored"
DEFAULT_SIZES_CACHE = ROOT / "fineweb_10BT_dump_sizes.json"
WRITING_DIR = ROOT.parent / "writing"
HF_SAMPLE_DIR = "datasets/HuggingFaceFW/fineweb/sample/10BT"
CRAWL_ID_RE = re.compile(r"CC-MAIN-(\d{4})-(\d{2})")

# Min roberta_score for a scored row to count as a slang-sense occurrence. A
# constant rather than a flag so any change to it is tracked in git.
THRESHOLD = 0.99
# Centered moving-average window, in dumps, applied to every series (1 = off).
# Odd, so the window is symmetric about each dump.
SMOOTH = 9
# Degree of the least-squares polynomial fitted per word under --polynomial-fit.
POLY_DEGREE = 3

# Fixed cohorts of ephemeral words (pooled from both prompt_scored and
# scenario_prompt_scored), bucketed by peak year, each saved as
# highlight_<band>_count.png. The pre-2018 band gets a broken y-axis so lol
# (peak ~16/M) does not flatten the rest (peak <4/M).
HIGHLIGHT_BANDS = {
    "pre2018":    ["stfu", "lmfao", "griefing", "methinks", "omg", "u", "yapping",
                   "ick", "dude", "hella", "troll", "pumped", "rad", "lol", "meh",
                   "bro", "finna", "lmao", "thot"],
    "around2020": ["cucked", "shitposting", "struggle bus", "simp", "bruh"],
    "2022_24":    ["cuck", "doggo", "shitpost", "💀", "lowkey", "doomscrolling",
                   "glow-up", "vibing", "hooman", "situationship", "yeet",
                   "catfished", "red pill", "vibes", "idgaf", "legit"],
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
                "#e87ba4", "#008300", "#4a3aa7", "#e34948",
                # Extension for the large highlight bands: each picked greedily from
                # an OKLCH grid to maximise its minimum OKLab distance to the hues
                # before it. Not colourblind-separable at this count; the legend
                # carries identity.
                "#7c140a", "#144d44", "#7fb6ee", "#a73189", "#7f6342", "#ab79f5",
                "#a7976e", "#681c65", "#8ac738", "#8b769b", "#1590a1", "#5a4e72"]


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
    sys.path.insert(0, str(ROOT))  # FineWebAnalysis/Keys.py
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
) -> tuple[Dict[str, List[float]], Optional[Dict[str, tuple]]]:
    """Central series per word, plus the matching CI bands when ``confidence`` is set.

    Returns ``(series, bands)`` ready to hand to the renderers; ``bands`` is None
    when no confidence level was requested, which is how the renderers decide
    whether to shade at all. With ``percent_of_peak`` each word is rescaled to a
    percentage of its own maximum (see --percent-of-peak).
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
    return series, bands


def _build_fits(
    counts: Dict[date, Counter],
    dump_tokens: Optional[Dict[date, int]],
    dumps: List[date],
    targets: List[str],
    percent_of_peak: bool = False,
) -> tuple[Dict[str, List[float]], Dict[str, List[float]]]:
    """Least-squares POLY_DEGREE polynomial per word, as ``(fits, points)``.

    The fit is to the unsmoothed per-dump values (zero dumps included: a count of
    0 is a real observation), over dump dates as matplotlib date numbers, which
    Polynomial.fit rescales internally to keep the cubic well-conditioned. The
    fitted curve is clipped at 0. With ``percent_of_peak`` both the curve and its
    points are divided by the curve's own maximum, so every fit peaks at 100.
    """
    xs = mdates.date2num(dumps)
    raw, _ = _build_series(counts, dump_tokens, dumps, targets, 1, None)
    fits: Dict[str, List[float]] = {}
    points: Dict[str, List[float]] = {}
    for t, ys in raw.items():
        fit = np.polynomial.Polynomial.fit(xs, ys, POLY_DEGREE)(xs).clip(min=0.0)
        peak = fit.max()
        if percent_of_peak and peak > 0:
            fit = 100.0 * fit / peak
            ys = [100.0 * y / peak for y in ys]
        fits[t] = fit.tolist()
        points[t] = ys
    return fits, points

def _apply_clean_style(ax, n_series: int) -> None:
    """Recessive chrome for --clean: fixed hues, hairline y-grid, no boxed-in axes."""
    if n_series <= len(CLEAN_COLORS):
        ax.set_prop_cycle(color=CLEAN_COLORS[:n_series])
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.grid(True, axis="y", color="#e1e0d9", linewidth=0.8)
    ax.set_axisbelow(True)


def _draw_series(ax, dumps, series, bands=None, clean=False, points=None):
    """Draw one markered line (+ optional CI band) per word on ``ax``.

    Dumps where a word has zero occurrences are dropped rather than plotted at 0:
    a 0 is invalid on a log axis (and a misleading dive to the floor on a linear
    one), so we skip those points and let the line connect the neighboring dumps.

    With ``points`` (--polynomial-fit), each series is a fitted curve instead: it
    is drawn over every dump, including where it touches 0, and the word's
    non-zero per-dump values are scattered beneath it in the same colour. The y
    range is set by the curves, not the points: a rare word's noisy dumps can sit
    several times above its fit and would otherwise flatten every curve, so those
    points run off the top of the chart.
    """
    if points is not None:
        for target, ys in series.items():
            line, = ax.plot(dumps, ys, linewidth=1.6 if clean else 1, label=target)
            kept = [(d, y) for d, y in zip(dumps, points[target]) if y > 0]
            if kept:
                px, py = zip(*kept)
                ax.scatter(px, py, s=6, color=line.get_color(), alpha=0.35,
                           linewidths=0)
        top = 1.1 * max((max(ys) for ys in series.values()), default=0.0)
        if top > 0:
            ax.set_ylim(-0.03 * top, top)
        return
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
                   confidence: Optional[float], polynomial_fit: bool = False) -> str:
    """One-line summary of how the figure was produced, drawn in its bottom margin."""
    parts = [f"word_rate_plotter.py \u00b7 roberta_score \u2265 {threshold:g}",
             "target + scenario contexts pooled" if n_dirs > 1 else "target contexts",
             "occurrences per million sample tokens" if dump_tokens is not None
             else "raw per-dump counts"]
    if polynomial_fit:
        parts.append(f"degree-{POLY_DEGREE} least-squares fit to per-dump values")
    elif smooth > 1:
        parts.append(f"{smooth}-dump centred moving average")
    if percent_of_peak:
        parts.append("rescaled to % of each word's peak dump")
    if confidence:
        parts.append(f"{confidence:.0%} Poisson CI")
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
    points: Optional[Dict[str, List[float]]] = None,
) -> None:
    """Draw one chart from precomputed per-word series (with optional CI bands)."""
    fig, ax = plt.subplots(figsize=(12, 6))
    if clean:
        _apply_clean_style(ax, len(series))
    _draw_series(ax, dumps, series, bands, clean, points)

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
    points: Optional[Dict[str, List[float]]] = None,
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
    _draw_series(top, dumps, series, bands, clean, points)
    _draw_series(bot, dumps, series, bands, clean, points)
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
    clean: bool = False,
    note: Optional[str] = None,
) -> None:
    dumps = sorted(counts)
    totals: Counter = Counter()
    for dump_counts in counts.values():
        totals.update(dump_counts)

    targets = _select_targets(totals, words, top)
    series, bands = _build_series(counts, dump_tokens, dumps, targets, smooth,
                                  confidence, percent_of_peak)
    pct_note = ", % of each word's peak" if percent_of_peak else ""
    title = (f"Slang usage per crawl dump ({len(targets)} words, "
             f"roberta_score >= {threshold:g}"
             f"{f', {confidence:.0%} CI' if confidence else ''}"
             f"{pct_note})")
    _render_chart(dumps, series, dump_tokens, log_scale, title, output, bands,
                  percent_of_peak, clean, note)


def plot_highlight_bands(
    counts: Dict[date, Counter],
    dump_tokens: Optional[Dict[date, int]],
    smooth: int,
    out_dir: Path,
    confidence: Optional[float] = None,
    percent_of_peak: bool = False,
    clean: bool = False,
    note: Optional[str] = None,
    polynomial_fit: bool = False,
) -> None:
    """Emit the fixed ephemeral-word band figures, one PNG per band.

    Each band (HIGHLIGHT_BANDS) is a cohort of words on one linear-axis chart;
    bands listed in HIGHLIGHT_BROKEN get a split y-axis so a dominant word does
    not flatten the rest. Files are written as ``highlight_<band>_count.png``
    (``highlight_<band>_pct.png`` under --percent-of-peak, whose rescaling makes
    the fixed broken-axis limits meaningless, so the split is skipped).
    With ``polynomial_fit`` each word is drawn as its POLY_DEGREE fit over its
    per-dump points instead (no smoothing or CI), and ``_polynomial_fit`` is
    appended to the file name.
    """
    dumps = sorted(counts)
    for band, words in HIGHLIGHT_BANDS.items():
        present = [w for w in words if any(counts[d].get(w) for d in dumps)]
        missing = [w for w in words if w not in present]
        if missing:
            log.warning("Band %s: no above-threshold occurrences for %s",
                        band, ", ".join(missing))
        points = None
        if polynomial_fit:
            series, points = _build_fits(counts, dump_tokens, dumps, present,
                                         percent_of_peak)
            bands = None
        else:
            series, bands = _build_series(counts, dump_tokens, dumps, present, smooth,
                                          confidence, percent_of_peak)
        out = out_dir / (f"highlight_{band}_{'pct' if percent_of_peak else 'count'}"
                         f"{'_polynomial_fit' if polynomial_fit else ''}.png")
        title = HIGHLIGHT_TITLES[band] + (f" \u2014 degree-{POLY_DEGREE} polynomial fit"
                                          if polynomial_fit else "")
        if band in HIGHLIGHT_BROKEN and not percent_of_peak:
            _render_broken(dumps, series, dump_tokens, title, out, bands,
                           HIGHLIGHT_BROKEN[band], percent_of_peak, clean, note,
                           points)
        else:
            _render_chart(dumps, series, dump_tokens, False, title, out, bands,
                          percent_of_peak, clean, note, points)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Plot the HIGHLIGHT_BANDS words' slang occurrences per crawl "
                    "dump from scored CSVs, one figure per band.")
    p.add_argument("--scored-dir", type=Path, nargs="+", default=None,
                   metavar="DIR", dest="scored_dir",
                   help="One or more directories of scored crawl CSVs (pooled). "
                        f"Default: {DEFAULT_SCORED_DIR.name} and {SCENARIO_SCORED_DIR.name}.")
    p.add_argument("--raw-counts", action="store_true", dest="raw_counts",
                   help="Plot raw per-dump counts instead of occurrences per "
                        "million tokens of the FineWeb sample dump.")
    p.add_argument("--percent-of-peak", action="store_true", dest="percent_of_peak",
                   help="Plot each word as a percentage of its own peak dump value "
                        "instead of an absolute rate, so words of very different "
                        "frequency can be compared by shape. Confidence bands are "
                        "scaled by the same per-word peak.")
    p.add_argument("--polynomial-fit", action="store_true", dest="polynomial_fit",
                   help="Draw each word as a least-squares POLY_DEGREE polynomial "
                        "fitted to its unsmoothed per-dump values, over those points, "
                        "instead of the SMOOTH moving average (no CI band). Writes "
                        "highlight_<band>_<count|pct>_polynomial_fit.png.")
    p.add_argument("--no-ci", action="store_true", dest="no_ci",
                   help="Never shade confidence bands, overriding --confidence.")
    p.add_argument("--clean", action="store_true",
                   help="Recessive chart styling: no per-dump markers, thicker lines, "
                        "fixed categorical hues, hairline y-grid only, and no boxed-in "
                        "axes or legend frame.")
    p.add_argument("--settings-note", action="store_true", dest="settings_note",
                   help="Print a one-line summary of the settings used ("
                        "pooling, normalisation, smoothing) in the figure's "
                        "bottom margin, so the chart is self-documenting.")
    p.add_argument("--confidence", type=float, default=0.95, metavar="P",
                   help="Shade a Poisson sampling-uncertainty band at confidence "
                        "level P (default: 0.95). Each per-dump count is treated as a "
                        "Poisson draw from the 10BT sample, so bands are wide for "
                        "rare words and tight for common ones, and tighten with a wider SMOOTH.")
    p.add_argument("-o", "--output", type=Path, metavar="DIR",
                   help="Directory to write highlight_<band>_*.png into "
                        f"(default: {WRITING_DIR.name}/).")
    return p


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    if args.scored_dir is None:
        args.scored_dir = [DEFAULT_SCORED_DIR, SCENARIO_SCORED_DIR]
    for d in args.scored_dir:
        if not d.is_dir():
            parser.error(f"Scored directory not found: {d}")

    # --polynomial-fit draws no CI band (the fit is not a Poisson rate estimate).
    confidence = None if args.no_ci or args.polynomial_fit else args.confidence
    if confidence is not None and not 0 < confidence < 1:
        parser.error("--confidence must be between 0 and 1 (e.g. 0.95).")

    counts: Dict[date, Counter] = defaultdict(Counter)
    for d in args.scored_dir:
        for dump_date, cnt in load_dump_counts(d, THRESHOLD).items():
            counts[dump_date].update(cnt)
    if not counts:
        parser.error("No crawl CSVs found in "
                     f"{', '.join(str(d) for d in args.scored_dir)}.")

    dump_tokens = None
    if not args.raw_counts:
        dump_tokens = fetch_dump_token_counts(DEFAULT_SIZES_CACHE)
        missing = [d for d in counts if d not in dump_tokens]
        if missing:
            sys.exit(f"No sample token totals for dump(s): "
                     f"{', '.join(d.isoformat() for d in sorted(missing))} — "
                     f"delete {DEFAULT_SIZES_CACHE.name} to re-fetch, or use --raw-counts.")

    note = (_settings_note(THRESHOLD, len(args.scored_dir), SMOOTH, dump_tokens,
                           args.percent_of_peak, confidence, args.polynomial_fit)
            if args.settings_note else None)

    _use_emoji_font()
    out_dir = args.output if args.output is not None else WRITING_DIR
    plot_highlight_bands(counts, dump_tokens, SMOOTH, out_dir, confidence,
                         args.percent_of_peak, args.clean, note, args.polynomial_fit)


if __name__ == "__main__":
    main()
