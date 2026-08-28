#!/usr/bin/env python3
"""
eval_scenario_model.py — Evaluate the fine-tuned scenario RoBERTa model on the
held-out validation set and write metrics to the model dir.

Scores every labelled row in ``scenario_finetune_validation`` with the model in
``scenario_ft_model_roberta`` (slang predicted when roberta_score >= 0, i.e.
P_slang >= 0.5), then reports accuracy / precision / recall / F1 overall, per
word, and split by source (synthetic vs. real). Writes
``scenario_ft_model_roberta/metrics.json``.
"""

from __future__ import annotations

import csv
import glob
import json
import sys
from collections import defaultdict
from pathlib import Path

from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_fscore_support

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from roberta_filter import TransformerSlangClassifier  # noqa: E402

MODEL_DIR = HERE / "scenario_ft_model_roberta"
VAL_DIR = HERE / "scenario_finetune_validation"


def load_labeled(d: Path) -> list[dict]:
    rows: list[dict] = []
    for f in sorted(glob.glob(str(d / "*.csv"))):
        with open(f, newline="", encoding="utf-8-sig") as fh:
            for r in csv.DictReader(fh):
                v = (r.get("is_slang") or "").strip()
                if v in ("0", "1"):
                    rows.append({
                        "target": (r.get("target") or "").strip(),
                        "target_context": r.get("target_context") or "",
                        "is_slang": int(v),
                        "uri": r.get("uri") or "",
                    })
    return rows


def metrics(y_true: list[int], y_pred: list[int]) -> dict:
    p, r, f, _ = precision_recall_fscore_support(
        y_true, y_pred, average="binary", pos_label=1, zero_division=0)
    return {"n": len(y_true), "n_slang": int(sum(y_true)),
            "accuracy": round(accuracy_score(y_true, y_pred), 4),
            "precision": round(float(p), 4), "recall": round(float(r), 4),
            "f1": round(float(f), 4)}


def main() -> None:
    if not MODEL_DIR.exists():
        sys.exit(f"Model dir not found: {MODEL_DIR}")
    rows = load_labeled(VAL_DIR)
    if not rows:
        sys.exit(f"No labelled validation rows in {VAL_DIR}.")

    clf = TransformerSlangClassifier(device=None)
    clf.load(MODEL_DIR)
    scores = clf.score_rows(rows, batch_size=64)

    y_true = [r["is_slang"] for r in rows]
    y_pred = [1 if s >= 0.0 else 0 for s in scores]  # score = 2P-1 >= 0  <=>  P >= 0.5

    overall = metrics(y_true, y_pred)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    overall["confusion"] = {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)}

    def by_mask(mask: list[bool]) -> dict | None:
        yt = [t for t, m in zip(y_true, mask) if m]
        yp = [p for p, m in zip(y_pred, mask) if m]
        return metrics(yt, yp) if yt else None

    is_syn = ["synthetic" in r["uri"] for r in rows]
    by_source = {"synthetic": by_mask(is_syn), "real": by_mask([not s for s in is_syn])}

    per_word: dict[str, dict] = {}
    buckets: dict[str, tuple[list, list]] = defaultdict(lambda: ([], []))
    for r, p in zip(rows, y_pred):
        buckets[r["target"]][0].append(r["is_slang"])
        buckets[r["target"]][1].append(p)
    for w, (yt, yp) in sorted(buckets.items()):
        per_word[w] = metrics(yt, yp)

    out = {
        "model_dir": MODEL_DIR.name,
        "base_model": "roberta-base",
        "eval_set": VAL_DIR.name,
        "decision_rule": "predict slang if roberta_score >= 0 (P_slang >= 0.5)",
        "overall": overall,
        "by_source": by_source,
        "per_word": per_word,
    }
    (MODEL_DIR / "metrics.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(json.dumps({"overall": overall, "by_source": by_source}, indent=2))
    print(f"\nWrote {MODEL_DIR / 'metrics.json'}")


if __name__ == "__main__":
    main()
