#!/usr/bin/env python3
"""Temporal fingerprint via mixture fitting.

Treats the model's response distribution over slang words as a mixture of the
corpus's year-slice distributions and finds nonneg weights w_y (summing to 1)
that best explain what era the model's slang sounds like:

    P_model(word) ≈ Σ_y  w_y · P_corpus(word | year=y)

Two fitting methods:

  EM / KL  — minimises KL(P_model ‖ mixture) via the EM algorithm.
             The objective is the log-likelihood of the model distribution
             under the mixture; this is the statistically natural choice.

  L2       — minimises ‖mixture − P_model‖², solved via SLSQP.
             More sensitive to absolute probability differences; useful as
             a check that the KL solution is not an artefact.

Corpus year-slice distributions are built from two FineWeb scored-context
directories (merged):
  • FineWebAnalysis/prompt_scored/          (original target_words vocabulary)
  • FineWebAnalysis/scenario_prompt_scored/ (scenario words vocabulary)
Counts are aggregated per calendar year, then Laplace-smoothed.

Bootstrap (parametric Poisson on word counts) gives 95% CIs on each year's
weight, propagated through to the mixture-weighted effective vintage.

Usage (from PromptingSlang root):
    python stats/temporal_mixture.py
    python stats/temporal_mixture.py --method em
    python stats/temporal_mixture.py --method l2
    python stats/temporal_mixture.py --smooth 1.0 --bootstrap 2000
"""
from __future__ import annotations

import argparse
import csv
import json
import re
import sys
import warnings
from collections import Counter, defaultdict
from datetime import date
from pathlib import Path

import numpy as np
from scipy.optimize import minimize

REPO_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = REPO_ROOT.parent
FINEWEB = PROJECT_ROOT / "FineWebAnalysis"

DEFAULT_SCORED_DIRS = [FINEWEB / "prompt_scored", FINEWEB / "scenario_prompt_scored"]
DEFAULT_PEAKS = FINEWEB / "peak_years.json"
DEFAULT_COUNTS = (REPO_ROOT / "experiments" / "scenario_based_prompting"
                  / "slang_counts_by_model.json")
DEFAULT_OUTPUT = Path(__file__).resolve().parent / "results" / "temporal_mixture.txt"
THRESHOLD = 0.99
CRAWL_ID_RE = re.compile(r"CC-MAIN-(\d{4})-(\d{2})")


# ── data loading ──────────────────────────────────────────────────────────────

def _crawl_year(stem: str) -> int | None:
    m = CRAWL_ID_RE.search(stem)
    return int(m.group(1)) if m else None


def load_year_counts(scored_dirs: list[Path]) -> dict[int, Counter]:
    """Aggregate per-year word counts across all scored-context CSV directories."""
    year_word: dict[int, Counter] = defaultdict(Counter)
    for d in scored_dirs:
        for path in sorted(d.glob("*.csv")):
            year = _crawl_year(path.stem)
            if year is None:
                continue
            with path.open(newline="", encoding="utf-8-sig") as fh:
                for row in csv.DictReader(fh):
                    if float(row["roberta_score"]) >= THRESHOLD:
                        year_word[year][row["target"]] += 1
    return dict(year_word)


def load_peak_data(path: Path, min_hits: int) -> dict[str, dict]:
    return {
        r["word"].lower(): {"peak_year": int(r["peak_year"]),
                            "corpus_count": int(r["total_hits"])}
        for r in json.loads(path.read_text(encoding="utf-8"))
        if "word" in r and "peak_year" in r and int(r.get("total_hits", 0)) >= min_hits
    }


def load_response_counts(path: Path) -> tuple[Counter, int]:
    data = json.loads(path.read_text(encoding="utf-8"))
    totals: Counter = Counter()
    n_resp = 0
    for info in data["models"].values():
        n_resp += info.get("responses", 0)
        for w, c in info.get("words", {}).items():
            totals[w] += c
    return totals, n_resp


# ── distribution matrices ─────────────────────────────────────────────────────

def build_corpus_matrix(
    year_counts: dict[int, Counter],
    vocab: list[str],
    epsilon: float,
) -> tuple[list[int], np.ndarray]:
    """Return (years, P_corpus) where P_corpus[y, v] = P(word v | year y).

    Laplace-smoothed with pseudo-count epsilon so no entry is exactly zero.
    """
    years = sorted(year_counts)
    V = len(vocab)
    w2i = {w: i for i, w in enumerate(vocab)}
    C = np.zeros((len(years), V), dtype=float)
    for yi, y in enumerate(years):
        for w, cnt in year_counts[y].items():
            if w in w2i:
                C[yi, w2i[w]] = cnt
    C_smooth = C + epsilon
    P = C_smooth / C_smooth.sum(axis=1, keepdims=True)
    return years, P


def build_model_vec(response_counts: Counter, vocab: list[str]) -> np.ndarray:
    """P_model[v] = response_count[v] / total; 0 for unobserved words."""
    p = np.array([float(response_counts.get(w, 0)) for w in vocab])
    total = p.sum()
    if total == 0:
        raise ValueError("No response hits in vocabulary.")
    return p / total


# ── fitting ───────────────────────────────────────────────────────────────────

def fit_em(
    P_corpus: np.ndarray,
    p_model: np.ndarray,
    max_iter: int = 2000,
    tol: float = 1e-10,
) -> tuple[np.ndarray, float]:
    """EM algorithm: minimise KL(p_model ‖ Σ_y w_y P_corpus[y]).

    Returns (weights, final_kl).
    """
    Y = P_corpus.shape[0]
    w = np.ones(Y) / Y

    for _ in range(max_iter):
        Q = w @ P_corpus                                  # (V,) mixture
        # Soft assignments: R[y,v] = w[y]*P[y,v]/Q[v]
        with np.errstate(divide="ignore", invalid="ignore"):
            R = (w[:, None] * P_corpus) / np.where(Q > 0, Q, 1.0)
        w_new = R @ p_model                               # (Y,)
        w_new_sum = w_new.sum()
        if w_new_sum > 0:
            w_new /= w_new_sum
        if np.max(np.abs(w_new - w)) < tol:
            w = w_new
            break
        w = w_new

    Q_final = w @ P_corpus
    with np.errstate(divide="ignore", invalid="ignore"):
        kl = float(np.sum(np.where(p_model > 0,
                                   p_model * np.log(p_model / np.maximum(Q_final, 1e-300)),
                                   0.0)))
    return w, kl


def fit_l2(P_corpus: np.ndarray, p_model: np.ndarray) -> np.ndarray:
    """L2 fit: minimise ‖Σ_y w_y P_corpus[y] − p_model‖² via SLSQP."""
    Y = P_corpus.shape[0]
    A = P_corpus.T    # (V, Y)

    def obj(w):
        r = A @ w - p_model
        return float(np.dot(r, r))

    def grad(w):
        return 2.0 * (A.T @ (A @ w - p_model))

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res = minimize(
            obj, np.ones(Y) / Y, jac=grad, method="SLSQP",
            bounds=[(0.0, 1.0)] * Y,
            constraints={"type": "eq", "fun": lambda w: w.sum() - 1.0},
            options={"ftol": 1e-12, "maxiter": 2000},
        )
    w = np.maximum(res.x, 0.0)
    w /= w.sum()
    return w


# ── bootstrap ─────────────────────────────────────────────────────────────────

def bootstrap_weights(
    P_corpus: np.ndarray,
    response_counts: Counter,
    vocab: list[str],
    method: str,
    B: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Parametric Poisson bootstrap: resample word counts, refit mixture.

    Returns array of shape (B, Y).
    """
    lambdas = np.array([float(response_counts.get(w, 0)) for w in vocab])
    draws = rng.poisson(lambdas, size=(B, len(vocab)))     # (B, V)
    row_sums = draws.sum(axis=1)                            # (B,)
    valid = row_sums > 0
    boot = np.full((B, P_corpus.shape[0]), np.nan)
    p_models = np.where(row_sums[:, None] > 0, draws / np.maximum(row_sums[:, None], 1), 0.0)

    for b in range(B):
        if not valid[b]:
            continue
        if method == "em":
            boot[b], _ = fit_em(P_corpus, p_models[b])
        else:
            boot[b] = fit_l2(P_corpus, p_models[b])

    return boot


# ── diagnostics ───────────────────────────────────────────────────────────────

def _kl_null(p_model: np.ndarray, P_corpus: np.ndarray) -> float:
    """KL of p_model vs uniform-weight mixture (null model)."""
    Y = P_corpus.shape[0]
    Q_null = np.ones(Y) / Y @ P_corpus
    with np.errstate(divide="ignore", invalid="ignore"):
        return float(np.sum(np.where(p_model > 0,
                                     p_model * np.log(p_model / np.maximum(Q_null, 1e-300)),
                                     0.0)))


def _effective_vintage(weights: np.ndarray, years: list[int]) -> float:
    return float(np.dot(weights, years))


# ── output ────────────────────────────────────────────────────────────────────

class _Tee:
    def __init__(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        self._file = path.open("w", encoding="utf-8")
        self._stdout = sys.stdout

    def write(self, s: str) -> None:
        self._stdout.write(s)
        self._file.write(s)

    def flush(self) -> None:
        self._stdout.flush()
        self._file.flush()

    def close(self) -> None:
        self._file.close()


def _bar(value: float, max_value: float, width: int = 32) -> str:
    n = int(round(value / max_value * width)) if max_value > 0 else 0
    return "█" * n


# ── main ─────────────────────────────────────────────────────────────────────

def main() -> None:
    ap = argparse.ArgumentParser(
        description="Temporal fingerprint: mixture of corpus year-slices fitted to model output.")
    ap.add_argument("--peaks", type=Path, default=DEFAULT_PEAKS)
    ap.add_argument("--counts", type=Path, default=DEFAULT_COUNTS)
    ap.add_argument("--scored-dirs", type=Path, nargs="+", default=DEFAULT_SCORED_DIRS,
                    dest="scored_dirs",
                    help="Scored-context CSV directories to merge (default: both "
                         "prompt_scored and scenario_prompt_scored).")
    ap.add_argument("--min-corpus-hits", type=int, default=100, dest="min_hits",
                    help="Min sense-filtered corpus hits for a word to be included "
                         "(default: 100).")
    ap.add_argument("--smooth", type=float, default=0.5, metavar="EPS",
                    help="Laplace pseudo-count added to each (year, word) cell "
                         "(default: 0.5).  Prevents zero-probability years for "
                         "words that genuinely hadn't appeared yet.")
    ap.add_argument("--method", choices=["em", "l2", "both"], default="both",
                    help="Fitting method: em (KL minimisation), l2 (L2 minimisation), "
                         "or both (default: both).")
    ap.add_argument("--bootstrap", type=int, default=2000, metavar="B",
                    help="Bootstrap iterations for CIs (default: 2000).")
    ap.add_argument("--ci", type=float, default=0.95,
                    help="Confidence level (default: 0.95).")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("-o", "--output", type=Path, default=DEFAULT_OUTPUT)
    args = ap.parse_args()

    tee = _Tee(args.output)
    sys.stdout = tee

    # ── load data ─────────────────────────────────────────────────────────────
    peak_data = load_peak_data(args.peaks, args.min_hits)
    response_counts, n_responses = load_response_counts(args.counts)
    year_counts = load_year_counts(args.scored_dirs)

    # Vocabulary: words with reliable corpus stats that appear in per-year data
    covered = set(w for c in year_counts.values() for w in c)
    vocab = sorted(w for w in peak_data if w in covered)
    V = len(vocab)
    n_used = sum(1 for w in vocab if response_counts.get(w, 0) > 0)

    years, P_corpus = build_corpus_matrix(year_counts, vocab, args.smooth)
    p_model = build_model_vec(response_counts, vocab)
    Y = len(years)

    print(f"\nTemporal fingerprint — mixture model")
    print(f"  Vocabulary  : {V} words  ({n_used} with non-zero response counts)")
    print(f"  Years       : {years[0]}–{years[-1]}  ({Y} year-slices)")
    print(f"  Smoothing   : ε = {args.smooth}")
    print(f"  Responses   : {n_responses} total, {int(sum(response_counts.get(w,0) for w in vocab))} hits")

    rng = np.random.default_rng(args.seed)
    alpha = (1 - args.ci) / 2

    methods = ["em", "l2"] if args.method == "both" else [args.method]

    for method in methods:
        print(f"\n{'═' * 64}")
        if method == "em":
            w, kl_fit = fit_em(P_corpus, p_model)
            kl_null = _kl_null(p_model, P_corpus)
            print(f"  Method: EM  (minimises KL(P_model ‖ mixture))")
            print(f"  KL(P_model ‖ fitted mixture) = {kl_fit:.4f} nats")
            print(f"  KL(P_model ‖ uniform mixture) = {kl_null:.4f} nats  (null)")
        else:
            w = fit_l2(P_corpus, p_model)
            Q = w @ P_corpus
            l2_fit = float(np.sum((Q - p_model) ** 2))
            Q_null = np.ones(Y) / Y @ P_corpus
            l2_null = float(np.sum((Q_null - p_model) ** 2))
            print(f"  Method: L2  (minimises ‖mixture − P_model‖²)")
            print(f"  L2(fitted mixture)  = {l2_fit:.6f}")
            print(f"  L2(uniform mixture) = {l2_null:.6f}  (null)")

        eff_vintage = _effective_vintage(w, years)

        # ── bootstrap ─────────────────────────────────────────────────────────
        boot = bootstrap_weights(P_corpus, response_counts, vocab, method,
                                 args.bootstrap, rng)
        ci_lo = np.nanpercentile(boot, 100 * alpha, axis=0)       # (Y,)
        ci_hi = np.nanpercentile(boot, 100 * (1 - alpha), axis=0) # (Y,)
        boot_vintage = np.nansum(boot * np.array(years)[None, :], axis=1)
        v_lo, v_hi = np.nanpercentile(boot_vintage, [100 * alpha, 100 * (1 - alpha)])

        max_w = w.max()
        ci_pct = f"{args.ci:.0%}"

        print(f"\n  Year   Weight   {ci_pct} CI           histogram")
        print(f"  {'─'*57}")
        for y, wy, lo, hi in zip(years, w, ci_lo, ci_hi):
            bar = _bar(wy, max_w)
            print(f"  {y}   {wy:.4f}   [{lo:.3f}, {hi:.3f}]   {bar}")

        print(f"\n  Effective vintage: {eff_vintage:.1f}  "
              f"[{v_lo:.1f}, {v_hi:.1f}]  {ci_pct} CI  "
              f"(B={args.bootstrap}, seed={args.seed})")

    # ── per-word fit table (EM only for brevity) ───────────────────────────────
    w_em, _ = fit_em(P_corpus, p_model) if "em" not in methods else (None, None)
    if "em" in methods:
        w_em, _ = fit_em(P_corpus, p_model)
    elif "l2" in methods:
        w_em = fit_l2(P_corpus, p_model)

    Q_em = w_em @ P_corpus
    print(f"\n{'═' * 64}")
    print("  Per-word fit: model distribution vs fitted mixture (EM)")
    print(f"  {'─'*57}")
    print(f"  {'word':<22} {'p_model':>8} {'p_mix':>8} {'ratio':>7}  peak_yr")
    rows = sorted(
        [(w, p_model[i], Q_em[i], peak_data[w]["peak_year"])
         for i, w in enumerate(vocab) if p_model[i] > 0],
        key=lambda r: -r[1],
    )
    for word, pm, qm, py in rows:
        ratio = pm / qm if qm > 0 else float("inf")
        flag = "  ↑↑" if ratio > 3 else ("  ↓↓" if ratio < 0.33 else "")
        print(f"  {word:<22} {pm:8.4f} {qm:8.4f} {ratio:7.2f}x  {py}{flag}")

    print(f"\n  ↑↑ = model uses >3× more than mixture predicts (model over-indexes)")
    print(f"  ↓↓ = model uses >3× less than mixture predicts (model under-indexes)")

    tee.close()
    sys.stdout = tee._stdout
    print(f"Results written to {args.output}")


if __name__ == "__main__":
    main()
