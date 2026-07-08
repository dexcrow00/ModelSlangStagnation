#!/usr/bin/env python3
"""Effective-vintage statistic for LLM slang output.

    model_vintage  = Σ_w (response_freq_w × peak_year_w) / Σ_w response_freq_w
    corpus_vintage = Σ_w (corpus_freq_w   × peak_year_w) / Σ_w corpus_freq_w

where sums run over the slang vocabulary with reliable corpus statistics.

The corpus vintage is the frequency-weighted average peak year across the
vocabulary — the "expected year" of a randomly sampled token from FineWeb that
belongs to our slang vocabulary.  The model vintage is the same statistic but
weighted by how often each word appears in the LLM responses.  A negative lag
(model − corpus) means the model's slang skews older than the corpus.

Bootstrap for the model-vintage CI
───────────────────────────────────
Preferred: pass --responses to a directory or JSONL of individual response
records.  The script resamples N responses with replacement, recounts word
hits in each bootstrap sample, and recomputes the vintage.  This captures
both word-frequency variance and within-response word co-occurrence.

Fallback (default): parametric Poisson bootstrap.  Each word's observed total
count is treated as a Poisson rate; B independent draws simulate the
variability you'd get from resampling N responses where words occur
independently.  This is a good approximation when slang hits are sparse
(here ~0.9 hits/response) so within-response correlations are small.

Example output:
    Model vintage : 2014.8  [2014.1, 2015.6]  95% CI (B=5000, Poisson bootstrap)
    Corpus vintage: 2019.3
    Lag           : −4.5 years  (model skews older than corpus)

Usage (from PromptingSlang root):
    python stats/effective_vintage.py
    python stats/effective_vintage.py --responses data/responses/exp1_scenario_generation_20260512_182231.jsonl
    python stats/effective_vintage.py --min-corpus-hits 50 --bootstrap 10000
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import warnings
from collections import Counter
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = REPO_ROOT.parent

DEFAULT_PEAKS = PROJECT_ROOT / "FineWebAnalysis" / "peak_years.json"
DEFAULT_COUNTS = (REPO_ROOT / "experiments" / "scenario_based_prompting"
                  / "slang_counts_by_model.json")
DEFAULT_WORDS = [PROJECT_ROOT / "FineWebAnalysis" / "target_words.txt",
                 PROJECT_ROOT / "FineWebAnalysis" / "scenario_words.txt"]
DEFAULT_OUTPUT = Path(__file__).resolve().parent / "results" / "effective_vintage.txt"


# ── helpers ───────────────────────────────────────────────────────────────────

def _load_vocab(paths: list[Path]) -> list[str]:
    seen: set[str] = set()
    vocab: list[str] = []
    for p in paths:
        for line in p.read_text(encoding="utf-8").splitlines():
            w = line.strip().lower()
            if w and not w.startswith("#") and w not in seen:
                seen.add(w)
                vocab.append(w)
    return vocab


def _pattern(word: str) -> re.Pattern:
    return re.compile(rf"(?<![a-z]){re.escape(word)}(?![a-z])", re.IGNORECASE)


def _load_peak_data(path: Path, min_hits: int) -> dict[str, dict]:
    """Return {word: {peak_year, corpus_count}} for reliable words."""
    return {
        r["word"].lower(): {"peak_year": int(r["peak_year"]),
                            "corpus_count": int(r["total_hits"])}
        for r in json.loads(path.read_text(encoding="utf-8"))
        if "word" in r and "peak_year" in r and int(r.get("total_hits", 0)) >= min_hits
    }


def _load_response_counts(path: Path) -> tuple[Counter, int]:
    """Aggregate word counts and response count from slang_counts_by_model.json."""
    data = json.loads(path.read_text(encoding="utf-8"))
    totals: Counter = Counter()
    n_resp = 0
    for info in data["models"].values():
        n_resp += info.get("responses", 0)
        for w, c in info.get("words", {}).items():
            totals[w] += c
    return totals, n_resp


def _load_individual_responses(path: Path) -> list[str]:
    """Load response texts from a JSONL file or directory of JSONL files."""
    sys.path.insert(0, str(REPO_ROOT))
    from src.response_utils import read_responses  # noqa: E402
    records = read_responses(str(path))
    return [r["response"] for r in records if r.get("response")]


def _count_in_text(text: str, vocab_pats: dict[str, re.Pattern]) -> Counter:
    return Counter({w: len(pat.findall(text)) for w, pat in vocab_pats.items()
                    if pat.search(text)})


# ── vintage calculation ────────────────────────────────────────────────────────

def _vintage(word_counts: dict[str, float], peak_data: dict[str, dict]) -> float:
    """Frequency-weighted mean peak year over words with reliable corpus stats."""
    num = denom = 0.0
    for w, count in word_counts.items():
        if count > 0 and w in peak_data:
            num += count * peak_data[w]["peak_year"]
            denom += count
    return num / denom if denom > 0 else float("nan")


# ── bootstrap ─────────────────────────────────────────────────────────────────

def _bootstrap_responses(
    response_texts: list[str],
    vocab_pats: dict[str, re.Pattern],
    peak_data: dict[str, dict],
    B: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Resample N responses with replacement; recount words each iteration."""
    per_response: list[Counter] = [_count_in_text(t, vocab_pats) for t in response_texts]
    n = len(per_response)
    vintages = np.empty(B)
    for b in range(B):
        indices = rng.integers(0, n, size=n)
        agg: Counter = Counter()
        for i in indices:
            agg.update(per_response[i])
        vintages[b] = _vintage(dict(agg), peak_data)
    return vintages


def _bootstrap_poisson(
    word_counts: Counter,
    peak_data: dict[str, dict],
    B: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Parametric Poisson bootstrap: resample each word count ~ Poisson(observed).

    Equivalent to a response-level bootstrap when slang hits are sparse and
    approximately independent across words within a response.
    """
    words = [w for w in word_counts if w in peak_data]
    lambdas = np.array([float(word_counts[w]) for w in words])
    years = np.array([float(peak_data[w]["peak_year"]) for w in words])

    draws = rng.poisson(lambdas, size=(B, len(words)))   # (B, W)
    totals = draws.sum(axis=1)                            # (B,)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        vintages = np.where(totals > 0, (draws * years).sum(axis=1) / totals, np.nan)
    return vintages


# ── output ────────────────────────────────────────────────────────────────────

class _Tee:
    def __init__(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        self._file = path.open("w", encoding="utf-8")
        self._stdout = sys.stdout

    def write(self, text: str) -> None:
        self._stdout.write(text)
        self._file.write(text)

    def flush(self) -> None:
        self._stdout.flush()
        self._file.flush()

    def close(self) -> None:
        self._file.close()


# ── main ─────────────────────────────────────────────────────────────────────

def main() -> None:
    ap = argparse.ArgumentParser(
        description="Effective-vintage statistic: frequency-weighted mean peak year.")
    ap.add_argument("--peaks", type=Path, default=DEFAULT_PEAKS,
                    help=f"peak_years.json (default: {DEFAULT_PEAKS.name}).")
    ap.add_argument("--counts", type=Path, default=DEFAULT_COUNTS,
                    help=f"slang_counts_by_model.json for point estimates "
                         f"(default: {DEFAULT_COUNTS.name}).")
    ap.add_argument("--responses", type=Path, default=None, metavar="PATH",
                    help="JSONL file or directory of individual response records. "
                         "If given, uses proper response-level bootstrap instead "
                         "of the parametric Poisson fallback.")
    ap.add_argument("--words", type=Path, nargs="+", default=DEFAULT_WORDS,
                    metavar="FILE",
                    help="Word-list files whose union defines the slang vocabulary.")
    ap.add_argument("--min-corpus-hits", type=int, default=100, dest="min_hits",
                    help="Exclude words with fewer than N corpus hits (default: 100).")
    ap.add_argument("--bootstrap", type=int, default=5000, metavar="B",
                    help="Number of bootstrap iterations (default: 5000).")
    ap.add_argument("--ci", type=float, default=0.95,
                    help="Confidence level for the bootstrap CI (default: 0.95).")
    ap.add_argument("--seed", type=int, default=42,
                    help="Random seed for reproducibility (default: 42).")
    ap.add_argument("-o", "--output", type=Path, default=DEFAULT_OUTPUT,
                    help=f"Write results to this file in addition to stdout "
                         f"(default: {DEFAULT_OUTPUT.relative_to(REPO_ROOT)}).")
    args = ap.parse_args()

    tee = _Tee(args.output)
    sys.stdout = tee

    peak_data = _load_peak_data(args.peaks, args.min_hits)
    response_counts, n_responses = _load_response_counts(args.counts)
    rng = np.random.default_rng(args.seed)

    # ── point estimates ───────────────────────────────────────────────────────
    model_vintage = _vintage(dict(response_counts), peak_data)
    corpus_vintage = _vintage({w: d["corpus_count"] for w, d in peak_data.items()},
                               peak_data)
    lag = model_vintage - corpus_vintage

    n_vocab = len(peak_data)
    n_used = sum(1 for w in peak_data if response_counts.get(w, 0) > 0)
    total_hits = sum(response_counts.get(w, 0) for w in peak_data)

    print(f"\nEffective-vintage analysis")
    print(f"  Vocabulary : {n_vocab} words with ≥ {args.min_hits} corpus hits")
    print(f"               ({n_used} appear in responses, {total_hits} total hits "
          f"across {n_responses} responses)")

    # ── bootstrap ─────────────────────────────────────────────────────────────
    if args.responses is not None and args.responses.exists():
        vocab = _load_vocab(args.words)
        vocab_pats = {w: _pattern(w) for w in vocab if w in peak_data}
        response_texts = _load_individual_responses(args.responses)
        n_boot_resp = len(response_texts)
        boot_samples = _bootstrap_responses(
            response_texts, vocab_pats, peak_data, args.bootstrap, rng)
        boot_method = f"response-level bootstrap  (N={n_boot_resp} responses resampled)"
    else:
        boot_samples = _bootstrap_poisson(
            response_counts, peak_data, args.bootstrap, rng)
        boot_method = "parametric Poisson bootstrap  (use --responses for response-level)"

    alpha = (1.0 - args.ci) / 2.0
    ci_lo, ci_hi = np.nanpercentile(boot_samples, [100 * alpha, 100 * (1 - alpha)])

    # ── results ───────────────────────────────────────────────────────────────
    ci_pct = f"{args.ci:.0%}"
    print(f"\n{'─' * 62}")
    print(f"  Model vintage  : {model_vintage:.1f}  "
          f"[{ci_lo:.1f}, {ci_hi:.1f}]  {ci_pct} CI")
    print(f"  Corpus vintage : {corpus_vintage:.1f}")
    lag_dir = "older" if lag < 0 else "newer"
    print(f"  Lag            : {lag:+.1f} years  (model skews {lag_dir} than corpus)")
    print(f"{'─' * 62}")
    print(f"  Bootstrap      : B={args.bootstrap}, seed={args.seed}")
    print(f"  Method         : {boot_method}")

    # ── per-word breakdown ────────────────────────────────────────────────────
    print(f"\n{'─' * 62}")
    print(f"  Per-word contributions to model vintage  "
          f"(sorted by response weight)")
    print(f"{'─' * 62}")
    total_weight = float(sum(response_counts.get(w, 0) for w in peak_data))
    print(f"  {'word':<22} {'yr':>4} {'resp':>6} {'weight%':>8} {'contrib':>8}")
    rows = sorted(
        [(w, d) for w, d in peak_data.items() if response_counts.get(w, 0) > 0],
        key=lambda x: -response_counts[x[0]],
    )
    for w, d in rows:
        c = response_counts[w]
        wt = c / total_weight if total_weight > 0 else 0.0
        print(f"  {w:<22} {d['peak_year']:>4} {c:>6} {wt:>7.1%} "
              f"{wt * d['peak_year']:>8.2f}")

    tee.close()
    sys.stdout = tee._stdout
    print(f"Results written to {args.output}")


if __name__ == "__main__":
    main()
