#!/usr/bin/env python3
"""
auc_eval.py — Evaluate a fine-tuned BERT slang classifier using ROC AUC.

Loads annotated examples (is_slang = 1/0), runs inference with a saved
fine-tuned model, and reports:
  - Overall ROC AUC
  - Per-word AUC breakdown
  - Confusion matrix at the default threshold (ft_score > 0, i.e. P(slang) > 0.5)
  - ROC curve plot (saved to --plot or displayed interactively)

NOTE: For the small (10BT) FineWeb sample, the evaluation is a bit biased, 
since we can't uniquely sample 50 lines for each word for both the training and validations sets.

Usage:
    # Evaluate on the same annotations used for training (in-sample)
    python auc_eval.py \\
        --model-dir CommonCrawlDiff/ft_model \\
        --annotations DataProcessingTools/completed_annotations/common_crawl

    # Cross-dataset evaluation (out-of-sample)
    python auc_eval.py \\
        --model-dir FineWebAnalysis/ft_model \\
        --annotations DataProcessingTools/completed_annotations/common_crawl \\
        --plot roc_curve.png

Requires: torch, transformers, scikit-learn, matplotlib
"""

from __future__ import annotations

import argparse
import csv
import logging
import re
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stderr)],
)
log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Load annotations
# ---------------------------------------------------------------------------

def load_annotations(annotation_dir: Path) -> Tuple[List[str], List[int], List[str]]:
    """Load (text, label, word) triples from all *_annotated.csv files.

    Returns three parallel lists:
      texts  — target_context strings
      labels — 1 (slang) or 0 (not slang)
      words  — source word inferred from the filename stem
    """
    texts: List[str]  = []
    labels: List[int] = []
    words: List[str]  = []

    for csv_path in sorted(annotation_dir.rglob("*.csv")):
        # Extract word from filenames like: alpha_validation, alpha_annotated,
        # alpha_annotations, or "Annotations V1 - FineWeb - alpha_annotations"
        m = re.search(r'(\b[a-zA-Z]+)(?:_validation|_annotated|_annotations)$',
                      csv_path.stem, re.IGNORECASE)
        word = m.group(1).lower() if m else csv_path.stem.lower()
        with csv_path.open(newline="", encoding="utf-8") as fh:
            for row in csv.DictReader(fh):
                a = (row.get("is_slang") or "").strip()
                if a == "1":
                    texts.append(row["target_context"])
                    labels.append(1)
                    words.append(word)
                elif a == "0":
                    texts.append(row["target_context"])
                    labels.append(0)
                    words.append(word)

    n_pos = sum(labels)
    log.info("Loaded %d examples from %s  (%d slang, %d not-slang)",
             len(texts), annotation_dir, n_pos, len(texts) - n_pos)
    return texts, labels, words


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------

def score_texts(
    texts: List[str],
    model_dir: Path,
    batch_size: int,
    device: Optional[str],
) -> List[float]:
    """Return P(slang) for each text using the saved fine-tuned model."""
    try:
        from transformers import AutoTokenizer, AutoModelForSequenceClassification  # type: ignore
    except ImportError:
        log.error("transformers is required. pip install transformers")
        sys.exit(1)

    if device is None:
        if torch.cuda.is_available():
            device = "cuda"
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"

    log.info("Loading model from %s (device=%s) ...", model_dir, device)
    tokenizer = AutoTokenizer.from_pretrained(str(model_dir))
    model = AutoModelForSequenceClassification.from_pretrained(str(model_dir)).to(device)
    model.eval()

    probs: List[float] = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        enc = tokenizer(
            batch, padding=True, truncation=True,
            max_length=256, return_tensors="pt",
        ).to(device)
        with torch.no_grad():
            p = torch.softmax(model(**enc).logits, dim=1)[:, 1].cpu().tolist()
        probs.extend(p)
        if (i // batch_size) % 10 == 0:
            log.info("  Scored %d / %d examples ...", min(i + batch_size, len(texts)), len(texts))

    return probs


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def _batched(items: list, size: int):
    for i in range(0, len(items), size):
        yield items[i : i + size]


def compute_metrics(
    labels: List[int],
    probs: List[float],
    words: List[str],
    threshold: float,
) -> Dict:
    """Compute overall AUC, per-word AUC, and confusion matrix."""
    try:
        from sklearn.metrics import (  # type: ignore
            roc_auc_score, roc_curve, confusion_matrix,
            precision_score, recall_score, f1_score,
        )
        import numpy as np
    except ImportError:
        log.error("scikit-learn is required. pip install scikit-learn")
        sys.exit(1)

    scores = [2 * p - 1 for p in probs]   # map to [-1, 1] as used elsewhere
    preds  = [1 if s >= threshold else 0 for s in scores]

    overall_auc = roc_auc_score(labels, probs)
    fpr, tpr, thresholds = roc_curve(labels, probs)

    cm = confusion_matrix(labels, preds)
    prec = precision_score(labels, preds, zero_division=0)
    rec  = recall_score(labels, preds, zero_division=0)
    f1   = f1_score(labels, preds, zero_division=0)

    # Per-word AUC
    unique_words = sorted(set(words))
    per_word: Dict[str, Optional[float]] = {}
    for w in unique_words:
        idx = [i for i, word in enumerate(words) if word == w]
        w_labels = [labels[i] for i in idx]
        w_probs  = [probs[i]  for i in idx]
        if len(set(w_labels)) < 2:
            per_word[w] = None   # can't compute AUC with only one class
        else:
            per_word[w] = roc_auc_score(w_labels, w_probs)

    return {
        "overall_auc": overall_auc,
        "fpr": fpr,
        "tpr": tpr,
        "thresholds": thresholds,
        "confusion_matrix": cm,
        "precision": prec,
        "recall": rec,
        "f1": f1,
        "per_word_auc": per_word,
        "threshold": threshold,
        "n_pos": sum(labels),
        "n_neg": len(labels) - sum(labels),
        "n_total": len(labels),
    }


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def print_report(metrics: Dict) -> None:
    cm = metrics["confusion_matrix"]
    tn, fp, fn, tp = cm.ravel() if cm.size == 4 else (0, 0, 0, 0)

    print()
    print("=" * 55)
    print(f"  Overall ROC AUC : {metrics['overall_auc']:.4f}")
    print(f"  Precision       : {metrics['precision']:.4f}")
    print(f"  Recall          : {metrics['recall']:.4f}")
    print(f"  F1              : {metrics['f1']:.4f}")
    print(f"  Threshold       : ft_score >= {metrics['threshold']:.2f}  (P(slang) >= {(metrics['threshold']+1)/2:.2f})")
    print(f"  Examples        : {metrics['n_total']}  ({metrics['n_pos']} slang / {metrics['n_neg']} not-slang)")
    print()
    print("  Confusion Matrix  (rows=actual, cols=predicted)")
    print(f"               not-slang   slang")
    print(f"  not-slang    {tn:>9}   {fp:>5}")
    print(f"  slang        {fn:>9}   {tp:>5}")
    print()
    print("  Per-word AUC:")
    per_word = metrics["per_word_auc"]
    valid = {w: v for w, v in per_word.items() if v is not None}
    skipped = [w for w, v in per_word.items() if v is None]
    for w, auc in sorted(valid.items(), key=lambda x: -x[1]):
        bar = "#" * int(auc * 30)
        print(f"    {w:<20} {auc:.4f}  {bar}")
    if skipped:
        print(f"    (skipped — single class only: {', '.join(skipped)})")
    print("=" * 55)
    print()


def plot_roc(metrics: Dict, model_dir: Path, annotation_dir: Path, save_path: Optional[Path]) -> None:
    try:
        import matplotlib.pyplot as plt  # type: ignore
    except ImportError:
        log.warning("matplotlib not available — skipping plot.")
        return

    fpr = metrics["fpr"]
    tpr = metrics["tpr"]
    auc = metrics["overall_auc"]

    fig, ax = plt.subplots(figsize=(7, 6))
    ax.plot(fpr, tpr, color="steelblue", lw=2,
            label=f"ROC curve  (AUC = {auc:.4f})")
    ax.plot([0, 1], [0, 1], color="grey", lw=1, linestyle="--", label="Random")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title(
        f"ROC Curve — Fine-tuned BERT Slang Classifier\n"
        f"Model: {model_dir.name}   Annotations: {annotation_dir.name}"
    )
    ax.legend(loc="lower right")
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1.02])
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150)
        log.info("ROC curve saved to %s", save_path)
    else:
        plt.show()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Compute ROC AUC for a fine-tuned BERT slang classifier.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # In-sample evaluation (same annotations used for training)
  python auc_eval.py \\
      --model-dir CommonCrawlDiff/ft_model \\
      --annotations DataProcessingTools/completed_annotations/common_crawl

  # Cross-dataset evaluation (out-of-sample)
  python auc_eval.py \\
      --model-dir FineWebAnalysis/ft_model \\
      --annotations DataProcessingTools/completed_annotations/common_crawl \\
      --plot roc_fineweb_model_on_cc_data.png
        """,
    )

    p.add_argument("--model-dir", required=True, metavar="DIR", dest="model_dir",
                   type=Path,
                   help="Directory containing the saved fine-tuned model.")
    p.add_argument("--annotations", required=True, metavar="DIR",
                   type=Path,
                   help="Directory of *_annotated.csv files with an 'is_slang' column.")
    p.add_argument("--threshold", type=float, default=0.0, metavar="F",
                   help="ft_score threshold for binary predictions in the confusion "
                        "matrix (default: 0.0, i.e. P(slang) > 0.5).")
    p.add_argument("--batch-size", type=int, default=32, metavar="N", dest="batch_size",
                   help="Inference batch size (default: 32).")
    p.add_argument("--device", default=None, metavar="DEV",
                   help="PyTorch device, e.g. cpu, cuda, mps (default: auto-detect).")
    p.add_argument("--plot", default=None, metavar="FILE", type=Path,
                   help="Save the ROC curve to this file instead of displaying it. "
                        "Omit to show interactively; pass /dev/null to suppress.")

    return p


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    if not args.model_dir.exists():
        parser.error(f"Model directory not found: {args.model_dir}")
    if not args.annotations.exists():
        parser.error(f"Annotations directory not found: {args.annotations}")

    texts, labels, words = load_annotations(args.annotations)
    if not texts:
        parser.error("No annotated examples found.")

    probs = score_texts(texts, args.model_dir, args.batch_size, args.device)
    metrics = compute_metrics(labels, probs, words, args.threshold)

    print_report(metrics)
    plot_roc(metrics, args.model_dir, args.annotations, args.plot)


if __name__ == "__main__":
    main()
