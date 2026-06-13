#!/usr/bin/env python3
"""
Scores annotated examples with a model saved by finetune_roberta.py (the
prompt template stored next to the model is applied automatically) and
reports overall ROC AUC, a confusion matrix at the chosen threshold, a
per-word breakdown, and an optional ROC plot. With --train-dir, rows whose
target_context appears in the training annotations are counted and the
metrics are repeated for the unseen subset only.

The judged word comes from a per-row 'target' column when present, else from
the filename (same rule as finetune_roberta.py, so multi-word targets work).

Usage:
    python auc_eval.py --model-dir FineWebAnalysis/ft_model_roberta \\
        --annotations DataProcessingTools/fine_tuning_validation/fine_web_small_validation_set \\
        [--truth-col is_slang] [--train-dir DIR] [--plot roc.png]

"""

from __future__ import annotations

import argparse
import csv
import logging
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from sklearn.metrics import (
    confusion_matrix, f1_score, precision_score, recall_score,
    roc_auc_score, roc_curve,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stderr)],
)
log = logging.getLogger(__name__)

# Model/inference helpers live in FineWebAnalysis (no packaging in this repo).
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "FineWebAnalysis"))
from finetune_roberta import _target_from_filename  # pyright: ignore[reportMissingImports]
from roberta_filter import TransformerSlangClassifier  # pyright: ignore[reportMissingImports]


def load_annotations(
    path: Path, truth_col: str = "is_slang",
) -> Tuple[List[Dict[str, str]], List[int], List[str]]:
    """Load rows from a CSV file or a directory of CSVs (recursive).

    Returns parallel lists: classifier-input rows ('target'/'target_context'),
    0/1 labels from ``truth_col``, and the judged word per row.
    """
    rows: List[Dict[str, str]] = []
    labels: List[int] = []
    words: List[str] = []
    for csv_path in sorted(path.rglob("*.csv")) if path.is_dir() else [path]:
        file_target = _target_from_filename(csv_path.stem)
        with csv_path.open(newline="", encoding="utf-8-sig") as fh:
            for row in csv.DictReader(fh):
                a = (row.get(truth_col) or "").strip()
                if a not in ("0", "1"):
                    continue
                target = (row.get("target") or "").strip() or file_target
                rows.append({"target": target,
                             "target_context": row.get("target_context", "")})
                labels.append(int(a))
                words.append(target)
    log.info("Loaded %d examples from %s  (%d slang, %d not-slang)",
             len(rows), path, sum(labels), len(rows) - sum(labels))
    return rows, labels, words


def load_train_contexts(train_dir: Path) -> set:
    """Every target_context appearing in the training annotation CSVs."""
    contexts: set = set()
    for csv_path in train_dir.rglob("*.csv"):
        with csv_path.open(newline="", encoding="utf-8-sig") as fh:
            contexts.update((row.get("target_context") or "").strip()
                            for row in csv.DictReader(fh))
    contexts.discard("")
    return contexts


def compute_metrics(
    labels: List[int], probs: List[float], words: List[str], threshold: float,
) -> Dict:
    """Overall AUC, per-word breakdown, and confusion matrix at the threshold."""
    # threshold is on ft_score = 2p - 1 in [-1, 1], as used elsewhere
    preds = [1 if 2 * p - 1 >= threshold else 0 for p in probs]

    per_word: Dict[str, Dict] = {}
    for w in sorted(set(words)):
        idx = [i for i, word in enumerate(words) if word == w]
        w_labels = [labels[i] for i in idx]
        w_preds = [preds[i] for i in idx]
        per_word[w] = {
            "auc": (roc_auc_score(w_labels, [probs[i] for i in idx])
                    if len(set(w_labels)) == 2 else None),
            "acc": sum(t == p for t, p in zip(w_labels, w_preds)) / len(idx),
            "fp": sum(1 for t, p in zip(w_labels, w_preds) if (t, p) == (0, 1)),
            "fn": sum(1 for t, p in zip(w_labels, w_preds) if (t, p) == (1, 0)),
            "n": len(idx),
        }

    fpr, tpr, _ = roc_curve(labels, probs)
    return {
        "overall_auc": roc_auc_score(labels, probs),
        "fpr": fpr,
        "tpr": tpr,
        "confusion_matrix": confusion_matrix(labels, preds),
        "precision": precision_score(labels, preds, zero_division=0),
        "recall": recall_score(labels, preds, zero_division=0),
        "f1": f1_score(labels, preds, zero_division=0),
        "per_word": per_word,
        "threshold": threshold,
        "n_pos": sum(labels),
        "n_neg": len(labels) - sum(labels),
        "n_total": len(labels),
    }


def print_report(m: Dict, title: str = "all rows") -> None:
    tn, fp, fn, tp = m["confusion_matrix"].ravel()
    print()
    print("=" * 55)
    print(f"  Subset          : {title}")
    print(f"  Overall ROC AUC : {m['overall_auc']:.4f}")
    print(f"  Precision       : {m['precision']:.4f}")
    print(f"  Recall          : {m['recall']:.4f}")
    print(f"  F1              : {m['f1']:.4f}")
    print(f"  Threshold       : ft_score >= {m['threshold']:.2f}  "
          f"(P(slang) >= {(m['threshold'] + 1) / 2:.2f})")
    print(f"  Examples        : {m['n_total']}  "
          f"({m['n_pos']} slang / {m['n_neg']} not-slang)")
    print()
    print("  Confusion Matrix  (rows=actual, cols=predicted)")
    print("               not-slang   slang")
    print(f"  not-slang    {tn:>9}   {fp:>5}")
    print(f"  slang        {fn:>9}   {tp:>5}")
    print()
    print("  Per-word breakdown (AUC | acc | FP | FN | n):")
    by_auc = sorted(m["per_word"].items(),
                    key=lambda x: -1 if x[1]["auc"] is None else x[1]["auc"],
                    reverse=True)
    for w, pw in by_auc:
        auc = f"{pw['auc']:.4f}" if pw["auc"] is not None else "(single class)"
        bar = "#" * int(pw["auc"] * 30) if pw["auc"] is not None else ""
        print(f"    {w:<20} {auc:>14} | {pw['acc']:6.1%} | {pw['fp']:>3} | "
              f"{pw['fn']:>3} | {pw['n']:>4}  {bar}")
    print("=" * 55)
    print()


def plot_roc(m: Dict, model_dir: Path, annotations: Path, save_path: Optional[Path]) -> None:
    try:
        import matplotlib.pyplot as plt  # type: ignore
    except ImportError:
        log.warning("matplotlib not available — skipping plot.")
        return

    fig, ax = plt.subplots(figsize=(7, 6))
    ax.plot(m["fpr"], m["tpr"], color="steelblue", lw=2,
            label=f"ROC curve  (AUC = {m['overall_auc']:.4f})")
    ax.plot([0, 1], [0, 1], color="grey", lw=1, linestyle="--", label="Random")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title(f"ROC Curve — Fine-tuned Slang Classifier\n"
                 f"Model: {model_dir.name}   Annotations: {annotations.name}")
    ax.legend(loc="lower right")
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1.02])
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150)
        log.info("ROC curve saved to %s", save_path)
    else:
        plt.show()


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Compute ROC AUC for a fine-tuned slang classifier.")
    p.add_argument("--model-dir", required=True, type=Path, metavar="DIR",
                   dest="model_dir",
                   help="Directory containing the saved fine-tuned model.")
    p.add_argument("--annotations", required=True, type=Path, metavar="PATH",
                   help="Annotation CSV file or directory of CSVs (recursive).")
    p.add_argument("--truth-col", default="is_slang", metavar="COL", dest="truth_col",
                   help="Ground-truth column name (default: is_slang).")
    p.add_argument("--train-dir", type=Path, metavar="DIR", dest="train_dir",
                   help="Training annotation dir; if set, report training-data "
                        "overlap and metrics on the unseen subset only.")
    p.add_argument("--threshold", type=float, default=0.0, metavar="F",
                   help="ft_score threshold for binary predictions "
                        "(default: 0.0, i.e. P(slang) > 0.5).")
    p.add_argument("--batch-size", type=int, default=32, metavar="N", dest="batch_size",
                   help="Inference batch size (default: 32).")
    p.add_argument("--device", default=None, metavar="DEV",
                   help="PyTorch device, e.g. cpu, cuda, mps (default: auto-detect).")
    p.add_argument("--plot", type=Path, metavar="FILE",
                   help="Save the ROC curve here instead of displaying it.")
    return p


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    if not args.model_dir.exists():
        parser.error(f"Model directory not found: {args.model_dir}")
    if not args.annotations.exists():
        parser.error(f"Annotations path not found: {args.annotations}")
    if args.train_dir and not args.train_dir.exists():
        parser.error(f"Training annotation directory not found: {args.train_dir}")

    rows, labels, words = load_annotations(args.annotations, args.truth_col)
    if not rows:
        parser.error(f"No annotated examples found (truth column: {args.truth_col!r}).")

    clf = TransformerSlangClassifier(device=args.device)
    clf.load(args.model_dir)
    probs = [(s + 1) / 2 for s in clf.score_rows(rows, args.batch_size)]

    metrics = compute_metrics(labels, probs, words, args.threshold)
    print_report(metrics)

    if args.train_dir:
        train_contexts = load_train_contexts(args.train_dir)
        unseen = [i for i, r in enumerate(rows)
                  if r["target_context"].strip() not in train_contexts]
        print(f"  Overlap with training data: {len(rows) - len(unseen)}/{len(rows)} "
              f"rows have a target_context present in {args.train_dir}")
        if unseen and len({labels[i] for i in unseen}) == 2:
            print_report(
                compute_metrics([labels[i] for i in unseen],
                                [probs[i] for i in unseen],
                                [words[i] for i in unseen], args.threshold),
                title="unseen rows only (not in training data)",
            )
        elif unseen:
            log.warning("Unseen subset has a single class — skipping its report.")

    plot_roc(metrics, args.model_dir, args.annotations, args.plot)


if __name__ == "__main__":
    main()
