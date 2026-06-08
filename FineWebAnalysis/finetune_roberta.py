#!/usr/bin/env python3
"""
finetune_roberta.py — Fine-tune a RoBERTa (or any HuggingFace AutoModel)
sequence classifier on human-annotated slang examples.

Reads annotation CSVs (recursive scan; each must have an ``is_slang`` column
with 1 = slang / 0 = not-slang and a ``target_context`` column), fine-tunes a
binary classifier with a class-weighted loss, and saves the best val-accuracy
checkpoint to a model directory. That directory is then consumed by
roberta_filter.py for inference/scoring.

Requires: torch, transformers, scikit-learn

Usage:
    # Fine-tune roberta-base and save the best checkpoint
    python finetune_roberta.py --annotations annotations/ \\
        --model-dir ./roberta_model/

    # Larger base model, custom schedule
    python finetune_roberta.py --annotations annotations/ \\
        --model-dir ./roberta_model/ --base-model roberta-large \\
        --epochs 15 --lr 1e-5 --batch-size 16
"""

from __future__ import annotations

import argparse
import csv
import logging
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(line_buffering=True)  # type: ignore[attr-defined]

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stderr)],
)
log = logging.getLogger(__name__)

DEFAULT_BASE_MODEL = "roberta-base"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def pick_device(device: Optional[str]) -> str:
    """Resolve an explicit device string, else auto-detect cuda/mps/cpu."""
    if device:
        return device
    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


class _SlangDataset(Dataset):
    """Tokenised binary classification dataset stored in memory."""

    def __init__(self, texts: List[str], labels: List[int], tokenizer, max_len: int = 256) -> None:
        enc = tokenizer(texts, padding=True, truncation=True, max_length=max_len, return_tensors="pt")
        self.input_ids      = enc["input_ids"]
        self.attention_mask = enc["attention_mask"]
        self.labels         = torch.tensor(labels, dtype=torch.long)

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, i: int) -> Dict:
        return {
            "input_ids":      self.input_ids[i],
            "attention_mask": self.attention_mask[i],
            "labels":         self.labels[i],
        }


def load_annotation_examples(annotation_dir: Path) -> List[Tuple[str, int]]:
    """Recursively load (text, label) pairs from CSVs with an 'is_slang' column.

    Expects rows where is_slang == '1' (slang) or '0' (not slang).
    Rows with blank or missing values are silently skipped.
    """
    examples: List[Tuple[str, int]] = []
    for csv_path in sorted(annotation_dir.rglob("*.csv")):
        with csv_path.open(newline="", encoding="utf-8-sig") as fh:
            for row in csv.DictReader(fh):
                a = (row.get("is_slang") or "").strip()
                if a == "1":
                    examples.append((row["target_context"], 1))
                elif a == "0":
                    examples.append((row["target_context"], 0))
    n_pos = sum(e[1] for e in examples)
    log.info(
        "Loaded %d annotated examples from %s (%d slang, %d not-slang)",
        len(examples), annotation_dir, n_pos, len(examples) - n_pos,
    )
    return examples


# ---------------------------------------------------------------------------
# Fine-tuning
# ---------------------------------------------------------------------------

def finetune(
    examples: List[Tuple[str, int]],
    model_save_dir: Path,
    base_model: str = DEFAULT_BASE_MODEL,
    device: Optional[str] = None,
    epochs: int = 10,
    lr: float = 2e-5,
    batch_size: int = 16,
    label: str = "RoBERTa",
) -> float:
    """Fine-tune a sequence classifier and save the best val-acc checkpoint.

    Returns the best validation accuracy achieved. The saved directory (model +
    tokenizer) is what roberta_filter.py loads for inference.
    """
    try:
        from transformers import (  # type: ignore
            AutoModelForSequenceClassification,
            AutoTokenizer,
            get_linear_schedule_with_warmup,
        )
        from sklearn.model_selection import train_test_split  # type: ignore
    except ImportError as exc:
        log.error("Fine-tuning requires transformers and scikit-learn: %s", exc)
        sys.exit(1)

    if len(examples) < 10:
        log.error("[%s] Too few annotated examples (%d) for reliable fine-tuning.",
                  label, len(examples))
        sys.exit(1)

    device = pick_device(device)
    log.info("[%s] Using device: %s", label, device)

    texts  = [e[0] for e in examples]
    labels = [e[1] for e in examples]

    train_texts, val_texts, train_labels, val_labels = train_test_split(
        texts, labels, test_size=0.2, random_state=42, stratify=labels
    )
    log.info(
        "[%s] Fine-tune split — Train: %d (%d pos / %d neg) | Val: %d (%d pos / %d neg)",
        label,
        len(train_texts), sum(train_labels), len(train_labels) - sum(train_labels),
        len(val_texts),   sum(val_labels),   len(val_labels)   - sum(val_labels),
    )

    log.info("[%s] Loading base model: %s", label, base_model)
    tokenizer = AutoTokenizer.from_pretrained(base_model)
    model     = AutoModelForSequenceClassification.from_pretrained(
        base_model, num_labels=2
    ).to(device)

    n_pos = sum(labels)
    n_neg = len(labels) - n_pos
    class_weights = torch.tensor(
        [1.0, n_neg / max(n_pos, 1)], dtype=torch.float, device=device
    )
    log.info("[%s] Class weights: not-slang=%.2f, slang=%.2f",
             label, class_weights[0].item(), class_weights[1].item())

    train_ds     = _SlangDataset(train_texts, train_labels, tokenizer)
    val_ds       = _SlangDataset(val_texts,   val_labels,   tokenizer)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size)

    optimizer   = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    total_steps = len(train_loader) * epochs
    scheduler   = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=max(1, total_steps // 10),
        num_training_steps=total_steps,
    )

    model_save_dir.mkdir(parents=True, exist_ok=True)
    best_val_acc = -1.0

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        for batch in train_loader:
            input_ids      = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            batch_labels   = batch["labels"].to(device)
            logits = model(input_ids=input_ids, attention_mask=attention_mask).logits
            loss   = F.cross_entropy(logits, batch_labels, weight=class_weights)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            total_loss += loss.item()

        model.eval()
        correct = total = 0
        with torch.no_grad():
            for batch in val_loader:
                input_ids      = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                preds = (
                    model(input_ids=input_ids, attention_mask=attention_mask)
                    .logits.argmax(dim=1).cpu()
                )
                correct += (preds == batch["labels"]).sum().item()
                total   += len(batch["labels"])

        val_acc  = correct / total if total else 0.0
        avg_loss = total_loss / len(train_loader)
        log.info("[%s] Epoch %d/%d — loss: %.4f, val_acc: %.3f",
                 label, epoch, epochs, avg_loss, val_acc)

        if val_acc >= best_val_acc:
            best_val_acc = val_acc
            model.save_pretrained(str(model_save_dir))
            tokenizer.save_pretrained(str(model_save_dir))
            log.info("[%s]   Saved best model (val_acc=%.3f) -> %s",
                     label, val_acc, model_save_dir)

    log.info("[%s] Fine-tuning complete. Best val_acc: %.3f", label, best_val_acc)
    return best_val_acc


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Fine-tune a RoBERTa slang-sense classifier on annotated examples.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Fine-tune roberta-base and save the best checkpoint
  python finetune_roberta.py --annotations annotations/ --model-dir ./roberta_model/

  # Larger base model, custom schedule
  python finetune_roberta.py --annotations annotations/ --model-dir ./roberta_model/ \\
      --base-model roberta-large --epochs 15 --lr 1e-5 --batch-size 16
        """,
    )
    p.add_argument("--annotations", required=True, metavar="DIR",
                   help="Directory of annotated CSVs (recursive *.csv scan; each row "
                        "needs an 'is_slang' column and a 'target_context' column).")
    p.add_argument("--model-dir", required=True, metavar="DIR", dest="model_dir",
                   help="Directory to save the fine-tuned model + tokenizer.")
    p.add_argument("--base-model", default=DEFAULT_BASE_MODEL, metavar="ID",
                   dest="base_model",
                   help=f"Base HuggingFace model ID to fine-tune (default: {DEFAULT_BASE_MODEL}).")
    p.add_argument("--epochs", type=int, default=10, metavar="N",
                   help="Fine-tuning epochs (default: 10).")
    p.add_argument("--lr", type=float, default=2e-5, metavar="F",
                   help="Learning rate (default: 2e-5).")
    p.add_argument("--batch-size", type=int, default=16, metavar="N", dest="batch_size",
                   help="Training batch size (default: 16).")
    p.add_argument("--device", default=None, metavar="DEV",
                   help="PyTorch device, e.g. cpu, cuda, mps (default: auto-detect).")
    return p


def main() -> None:
    parser = build_parser()
    args   = parser.parse_args()

    ann_dir = Path(args.annotations)
    if not ann_dir.exists():
        parser.error(f"Annotation directory not found: {ann_dir}")

    examples = load_annotation_examples(ann_dir)
    if not examples:
        parser.error(f"No annotated examples found in {ann_dir}.")

    finetune(
        examples,
        model_save_dir = Path(args.model_dir),
        base_model     = args.base_model,
        device         = args.device,
        epochs         = args.epochs,
        lr             = args.lr,
        batch_size     = args.batch_size,
    )


if __name__ == "__main__":
    main()
