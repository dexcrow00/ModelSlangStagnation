#!/usr/bin/env python3
"""Negative binomial regression: does corpus peak year predict LLM slang usage?

Model:
    response_count ~ NB(μ),   log(μ) = log(corpus_count) + β·peak_year + α

The log(corpus_count) offset absorbs each word's raw prevalence in FineWeb, so
β captures whether models over-/under-use a word *relative to its corpus share*
as a function of how recently it peaked.  A positive β means newer words are
used at a higher rate than their corpus frequency alone would predict.

Words with zero response counts are included (no selection bias from dropping
unsuccessful predictions).  Words without a reliable corpus count (< min_hits)
are excluded because their peak-year and corpus-count estimates are too noisy.

Fits both Poisson (baseline) and NB2, then:
  • reports β, IRR = exp(β), 95% CI, p-value for peak_year
  • reports AIC/BIC for model comparison
  • checks overdispersion via Pearson χ²/df
  • runs a likelihood-ratio test of NB vs Poisson (H₀: dispersion α = 0)

peak_year is mean-centred internally for numerical stability; the reported β
is on the original (year) scale and is unchanged by centring.

Data sources (resolved from project layout):
  • FineWebAnalysis/peak_years.json            → peak_year, corpus_count
  • experiments/…/slang_counts_by_model.json   → response_count per word

Usage (from PromptingSlang root):
    python stats/nb_regression.py
    python stats/nb_regression.py --min-corpus-hits 50
    python stats/nb_regression.py -o stats/results/custom.txt
"""
from __future__ import annotations

import argparse
import json
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats
from statsmodels.discrete.discrete_model import NegativeBinomial


class _Tee:
    """Write to both stdout and a file simultaneously."""
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

REPO_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = REPO_ROOT.parent

DEFAULT_PEAKS = PROJECT_ROOT / "FineWebAnalysis" / "peak_years.json"
DEFAULT_COUNTS = (REPO_ROOT / "experiments" / "scenario_based_prompting"
                  / "slang_counts_by_model.json")
DEFAULT_OUTPUT = Path(__file__).resolve().parent / "results" / "nb_regression.txt"


def load_data(peaks_path: Path, counts_path: Path, min_hits: int) -> pd.DataFrame:
    """Merge peak_years.json and slang_counts_by_model.json into a word-level DataFrame.

    All words meeting the corpus-hit threshold are included; response_count
    defaults to 0 for words absent from the LLM responses.
    """
    peaks = {
        r["word"].lower(): r
        for r in json.loads(peaks_path.read_text(encoding="utf-8"))
        if "word" in r and "peak_year" in r and int(r.get("total_hits", 0)) >= min_hits
    }

    counts_data = json.loads(counts_path.read_text(encoding="utf-8"))
    response_counts: dict[str, int] = {}
    for info in counts_data["models"].values():
        for w, c in info.get("words", {}).items():
            response_counts[w] = response_counts.get(w, 0) + c

    rows = [
        {
            "word": word,
            "response_count": int(response_counts.get(word, 0)),
            "corpus_count": int(pk["total_hits"]),
            "peak_year": int(pk["peak_year"]),
        }
        for word, pk in peaks.items()
    ]
    return pd.DataFrame(rows).sort_values("word").reset_index(drop=True)


def _fmt_p(p: float) -> str:
    if p < 0.001:
        return "< 0.001 ***"
    if p < 0.01:
        return f"{p:.4f}  **"
    if p < 0.05:
        return f"{p:.4f}  *"
    return f"{p:.4f}"


def _print_model(label: str, result, predictor: str, n: int) -> None:
    b = result.params[predictor]
    se = result.bse[predictor]
    p = result.pvalues[predictor]
    lo, hi = result.conf_int().loc[predictor]
    print(f"\n{'─' * 60}")
    print(f"  {label}  (n = {n})")
    print(f"{'─' * 60}")
    print(f"  β (peak_year)   = {b:+.4f}   SE {se:.4f}")
    print(f"  IRR = exp(β)    = {np.exp(b):.4f}   "
          f"95% CI [{np.exp(lo):.4f}, {np.exp(hi):.4f}]")
    print(f"  p-value         = {_fmt_p(p)}")
    print(f"  intercept       = {result.params['const']:+.4f}")
    print(f"  AIC / BIC       = {result.aic:.1f} / {result.bic:.1f}")
    print(f"  log-likelihood  = {result.llf:.3f}")


def main() -> None:
    ap = argparse.ArgumentParser(
        description="NB/Poisson regression of LLM slang usage on corpus peak year.")
    ap.add_argument("--peaks", type=Path, default=DEFAULT_PEAKS,
                    help=f"peak_years.json (default: {DEFAULT_PEAKS.name}).")
    ap.add_argument("--counts", type=Path, default=DEFAULT_COUNTS,
                    help=f"slang_counts_by_model.json (default: {DEFAULT_COUNTS.name}).")
    ap.add_argument("--min-corpus-hits", type=int, default=100, dest="min_hits",
                    help="Exclude words with fewer than N sense-filtered corpus hits "
                         "(default: 100).")
    ap.add_argument("-o", "--output", type=Path, default=DEFAULT_OUTPUT,
                    help=f"Write results to this file in addition to stdout "
                         f"(default: {DEFAULT_OUTPUT.relative_to(REPO_ROOT)}).")
    args = ap.parse_args()

    tee = _Tee(args.output)
    sys.stdout = tee

    df = load_data(args.peaks, args.counts, args.min_hits)
    n = len(df)
    if n < 5:
        tee.close()
        sys.stdout = tee._stdout
        sys.exit(f"Only {n} words after filtering — lower --min-corpus-hits.")

    n_zero = int((df["response_count"] == 0).sum())
    print(f"\nData: {n} words  (min_corpus_hits = {args.min_hits})")
    print(f"  peak_year     : {df['peak_year'].min()} – {df['peak_year'].max()}"
          f"  (mean {df['peak_year'].mean():.1f})")
    print(f"  corpus_count  : {df['corpus_count'].min():,} – {df['corpus_count'].max():,}")
    print(f"  response_count: 0 – {df['response_count'].max()}"
          f"  (total {df['response_count'].sum()}, {n_zero}/{n} zeros)")

    # Mean-centre peak_year for numerical stability; β is identical on original scale.
    yr_mean = df["peak_year"].mean()
    df["peak_year_c"] = df["peak_year"] - yr_mean
    predictor = "peak_year_c"

    offset = np.log(df["corpus_count"].astype(float))
    exog = sm.add_constant(df[[predictor]])
    endog = df["response_count"]

    # ── Poisson (baseline) ────────────────────────────────────────────────────
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        pois = sm.GLM(endog, exog, family=sm.families.Poisson(),
                      offset=offset).fit(disp=False)
    _print_model("Poisson regression (baseline)", pois, predictor, n)

    overdisp = pois.pearson_chi2 / pois.df_resid
    note = "overdispersed → prefer NB" if overdisp > 1.5 else "no strong overdispersion"
    print(f"\n  Pearson χ²/df = {overdisp:.2f}  ({note})")

    # ── Negative binomial NB2 ────────────────────────────────────────────────
    # Seed from Poisson params + α = 1 to avoid boundary solution at α → ∞.
    start_params = np.append(pois.params.values, 1.0)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        nb = NegativeBinomial(endog, exog, offset=offset).fit(
            start_params=start_params, method="bfgs", disp=False)

    if np.isnan(nb.bse[predictor]):
        print("\n  NB2 did not converge cleanly — falling back to Newton method.")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            nb = NegativeBinomial(endog, exog, offset=offset).fit(
                start_params=start_params, method="newton", disp=False)

    _print_model("Negative binomial NB2", nb, predictor, n)
    alpha_nb = float(nb.params.get("alpha", float("nan")))
    print(f"  dispersion α   = {alpha_nb:.4f}  (→ Poisson as α → 0)")

    # ── LR test: NB vs Poisson ────────────────────────────────────────────────
    lr_stat = 2.0 * (nb.llf - pois.llf)
    # α = 0 is on the boundary → one-sided p-value (divide chi² p by 2)
    lr_p = float(stats.chi2.sf(lr_stat, df=1)) / 2.0
    print(f"\n{'─' * 60}")
    print(f"  LR test NB vs Poisson  (H₀: α = 0)")
    print(f"  χ²(1) = {lr_stat:.2f},  p = {_fmt_p(lr_p)}")
    if lr_p < 0.05:
        print("  → NB significantly better fit (overdispersion confirmed)")
    else:
        print("  → Poisson not rejected at p < 0.05")

    # ── Per-word table (NB2 fitted values) ───────────────────────────────────
    print(f"\n{'─' * 60}")
    print("  Per-word fitted values — NB2  (sorted by response_count desc)")
    print(f"{'─' * 60}")
    print(f"  {'word':<22} {'yr':>4} {'corpus':>9} {'obs':>5} {'fit':>7}")
    fitted = nb.predict()
    for _, row in df.sort_values("response_count", ascending=False).iterrows():
        idx = int(df.index[df["word"] == row["word"]][0])
        print(f"  {row['word']:<22} {int(row['peak_year']):>4} "
              f"{int(row['corpus_count']):>9,} {int(row['response_count']):>5} "
              f"{fitted[idx]:>7.2f}")

    tee.close()
    sys.stdout = tee._stdout
    print(f"Results written to {args.output}")


if __name__ == "__main__":
    main()
