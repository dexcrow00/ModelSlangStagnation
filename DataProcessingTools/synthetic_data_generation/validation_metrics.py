#!/usr/bin/env python3
"""
validation_metrics.py — Compare the generated `is_slang` label against the
human-validated `manual_annotation_is_slang` column (the source of truth).

Reports overall agreement plus a confusion-matrix breakdown, treating
manual_annotation_is_slang as truth and is_slang as the prediction:
  - false positive : predicted slang (is_slang=1) but truth is not-slang (0)
  - false negative : predicted not-slang (is_slang=0) but truth is slang (1)

Usage:
    python validation_metrics.py [path/to/validation.csv]
"""

from __future__ import annotations

import csv
import sys
from pathlib import Path

DEFAULT_CSV = Path(__file__).resolve().parent / "Manual Synthetic Data_Annotation Validation - Sheet1.csv"

PRED_COL = "is_slang"
TRUTH_COL = "manual_annotation_is_slang"


def main() -> None:
    path = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_CSV
    if not path.is_file():
        sys.exit(f"File not found: {path}")

    tp = tn = fp = fn = 0
    skipped = 0  # rows with a missing/invalid label in either column

    with path.open(newline="", encoding="utf-8-sig") as fh:
        for row in csv.DictReader(fh):
            pred = (row.get(PRED_COL) or "").strip()
            truth = (row.get(TRUTH_COL) or "").strip()
            if pred not in ("0", "1") or truth not in ("0", "1"):
                skipped += 1
                continue
            pred_i, truth_i = int(pred), int(truth)
            if truth_i == 1 and pred_i == 1:
                tp += 1
            elif truth_i == 0 and pred_i == 0:
                tn += 1
            elif truth_i == 0 and pred_i == 1:
                fp += 1
            else:  # truth 1, pred 0
                fn += 1

    n = tp + tn + fp + fn
    if n == 0:
        sys.exit("No rows with valid labels in both columns.")

    agreement = (tp + tn) / n
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    fpr = fp / (fp + tn) if (fp + tn) else 0.0
    fnr = fn / (fn + tp) if (fn + tp) else 0.0

    print()
    print("=" * 52)
    print(f"  File      : {path.name}")
    print(f"  Evaluated : {n} rows" + (f"  ({skipped} skipped — blank/invalid)" if skipped else ""))
    print("-" * 52)
    print(f"  Agreement (accuracy) : {agreement:6.2%}  ({tp + tn}/{n})")
    print(f"  Precision            : {precision:6.2%}")
    print(f"  Recall (sensitivity) : {recall:6.2%}")
    print(f"  F1                   : {f1:6.2%}")
    print(f"  False positive rate  : {fpr:6.2%}")
    print(f"  False negative rate  : {fnr:6.2%}")
    print("-" * 52)
    print("  Confusion matrix  (truth = manual_annotation_is_slang)")
    print("                  pred not-slang   pred slang")
    print(f"  truth not-slang {tn:>13}   {fp:>10}")
    print(f"  truth slang     {fn:>13}   {tp:>10}")
    print("-" * 52)
    print(f"  False positives (gen said slang, truth not) : {fp}")
    print(f"  False negatives (gen said not, truth slang) : {fn}")
    print("=" * 52)
    print()


if __name__ == "__main__":
    main()
