#!/usr/bin/env python3
"""
roberta_filter.py — Score/filter context rows with a fine-tuned RoBERTa slang
classifier.

  score = 2 * P(slang) - 1  (range [-1, 1]; positive = slang usage)

This is the inference/filtering half of the pipeline; fine-tune the model first
with finetune_roberta.py, which saves the model directory loaded here (model +
tokenizer + the prompt template applied at training time).

Usage:

    # Score every row in a directory of context CSVs
    python roberta_filter.py contexts/ --output-dir scored/ --score-all \\
        --roberta-model-dir ./roberta_model/

    # Filter above a score threshold into a single CSV
    python roberta_filter.py contexts/*.csv -o filtered.csv \\
        --roberta-model-dir ./roberta_model/ --threshold 0.1
"""

from __future__ import annotations

import argparse
import csv
import glob
import logging
import sys
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Tuple

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(line_buffering=True)  # type: ignore[attr-defined]

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stderr)],
)
log = logging.getLogger(__name__)

# Saved next to the model by finetune_roberta.py; inference must condition on
# the same target-word phrasing the model was trained with.
PROMPT_TEMPLATE_FILE = "prompt_template.txt"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _pick_device(device: Optional[str]) -> str:
    """Resolve an explicit device string, else auto-detect cuda/mps/cpu."""
    if device:
        return device
    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def _batched(items: list, size: int) -> Iterator[list]:
    for i in range(0, len(items), size):
        yield items[i : i + size]


def _read_rows(path: Path) -> List[Dict[str, str]]:
    """Read context rows (target / uri / target_context) from a CSV file."""
    with path.open(newline="", encoding="utf-8-sig") as fh:
        # Normalise quoted column names produced by Excel/PowerShell BOM exports
        return [{k.strip('"').strip(): v for k, v in row.items()}
                for row in csv.DictReader(fh)]


# ---------------------------------------------------------------------------
# Transformer classifier (inference only)
# ---------------------------------------------------------------------------

class TransformerSlangClassifier:
    """Inference-only binary classifier for slang-sense scoring.

    Loads a directory saved by finetune_roberta.py (model + tokenizer + prompt
    template) and maps P(slang) ∈ [0, 1] → [-1, 1] via ``2p - 1``. Works with
    any HuggingFace AutoModelForSequenceClassification checkpoint.
    """

    def __init__(self, device: Optional[str] = None) -> None:
        self.device = _pick_device(device)
        log.info("Using device: %s", self.device)
        self.tokenizer = None
        self.model = None
        self.prompt_template = ""

    def load(self, model_dir: Path) -> None:
        """Load the fine-tuned model, tokenizer, and prompt template."""
        log.info("Loading model from %s", model_dir)
        self.tokenizer = AutoTokenizer.from_pretrained(str(model_dir))
        self.model = (
            AutoModelForSequenceClassification.from_pretrained(str(model_dir))
            .to(self.device)
        )
        self.model.eval()

        template_path = Path(model_dir) / PROMPT_TEMPLATE_FILE
        if not template_path.is_file():
            log.error("%s not found in %s — retrain with finetune_roberta.py, "
                      "which saves the prompt template used at training time.",
                      PROMPT_TEMPLATE_FILE, model_dir)
            sys.exit(1)
        self.prompt_template = template_path.read_text(encoding="utf-8")
        log.info("Using saved prompt template: %s", self.prompt_template)

    def score_rows(self, rows: List[Dict[str, str]], batch_size: int) -> List[float]:
        """Return a score ∈ [-1, 1] per row, scoring its ``target_context``
        conditioned on its ``target`` word via the saved prompt template."""
        assert self.model is not None, "Call load() before score_rows()."

        # Only the template is parsed for {target}/{context} fields; braces in
        # the raw web text are inserted literally, never re-interpreted.
        texts = [self.prompt_template.format(target=r.get("target", ""),
                                             context=r.get("target_context", ""))
                 for r in rows]
        if not texts:
            return []
        log.info("  Scoring %d rows ...", len(texts))

        scores: List[float] = []
        for batch in _batched(texts, batch_size):
            enc = self.tokenizer(
                batch, padding=True, truncation=True,
                max_length=256, return_tensors="pt",
            ).to(self.device)
            with torch.no_grad():
                probs = torch.softmax(self.model(**enc).logits, dim=1)[:, 1].cpu().tolist()
            scores.extend(float(2 * p - 1) for p in probs)  # [0,1] -> [-1,1]
        return scores


# ---------------------------------------------------------------------------
# Scoring / output
# ---------------------------------------------------------------------------

_FIELDNAMES = ["target", "uri", "target_context", "roberta_score"]


def _score_file(
    path: Path,
    clf: TransformerSlangClassifier,
    args: argparse.Namespace,
) -> Tuple[List[Dict], int]:
    """Score one context file; returns (kept rows, total rows read)."""
    rows = _read_rows(path)
    if not rows:
        log.info("Skipping empty file: %s", path.name)
        return [], 0

    log.info("Processing %s (%d rows) ...", path.name, len(rows))
    scores = clf.score_rows(rows, args.batch_size)
    out_rows = [
        {"target": row.get("target", ""),
         "uri": row.get("uri", ""),
         "target_context": row.get("target_context", ""),
         "roberta_score": f"{score:.4f}"}
        for row, score in zip(rows, scores)
        if args.score_all or score >= args.threshold
    ]
    log.info("  %s: %d/%d rows kept (%.1f%%)",
             path.name, len(out_rows), len(rows), 100.0 * len(out_rows) / len(rows))
    return out_rows, len(rows)


def _write_csv(path: Path, rows: List[Dict]) -> None:
    rows.sort(key=lambda r: -float(r["roberta_score"]))
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=_FIELDNAMES, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Score/filter slang sense using a fine-tuned RoBERTa classifier.")
    p.add_argument("inputs", nargs="+", metavar="FILE",
                   help="Input context CSV file(s), glob patterns, or a directory "
                        "of CSVs.")
    out_group = p.add_mutually_exclusive_group(required=True)
    out_group.add_argument("-o", "--output", metavar="FILE",
                           help="Output CSV path (all inputs merged into one file).")
    out_group.add_argument("--output-dir", metavar="DIR", dest="output_dir",
                           help="Output directory (one output file per input).")
    p.add_argument("--score-all", action="store_true", dest="score_all",
                   help="Write every row with scores attached (no filtering).")
    p.add_argument("--threshold", type=float, default=0.0, metavar="F",
                   help="Min roberta_score to keep a row when filtering (default: 0.0).")
    p.add_argument("--batch-size", type=int, default=32, metavar="N", dest="batch_size",
                   help="Inference batch size (default: 32).")
    p.add_argument("--device", default=None, metavar="DEV",
                   help="PyTorch device, e.g. cpu, cuda, mps (default: auto-detect).")
    p.add_argument("--roberta-model-dir", default="FineWebAnalysis/ft_model_roberta",
                   metavar="DIR", dest="roberta_model_dir",
                   help="Directory of the fine-tuned model to load "
                        "(produced by finetune_roberta.py).")
    return p


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    input_paths: List[Path] = []
    for pattern in args.inputs:
        path = Path(pattern)
        if path.is_dir():
            input_paths.extend(sorted(path.glob("*.csv")))
        else:
            matched = sorted(glob.glob(pattern, recursive=True))
            input_paths.extend(Path(m) for m in matched or [pattern])

    missing = [p for p in input_paths if not p.exists()]
    if missing:
        parser.error(f"File(s) not found: {', '.join(str(p) for p in missing)}")
    if not input_paths:
        parser.error("No input files found.")
    if len(input_paths) > 1 and args.output_dir is None:
        log.warning("%d input files with -o set — all rows merged into one file. "
                    "Use --output-dir for per-file output.", len(input_paths))

    model_dir = Path(args.roberta_model_dir)
    if not model_dir.exists():
        parser.error(f"Model directory '{model_dir}' not found. "
                     f"Fine-tune one first with finetune_roberta.py.")
    clf = TransformerSlangClassifier(device=args.device)
    clf.load(model_dir)

    # Per-file mode: one output CSV per input, skipping existing outputs.
    if args.output_dir is not None:
        out_dir = Path(args.output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        grand_in = grand_out = skipped = 0
        for path in input_paths:
            out_path = out_dir / (path.stem + ".csv")
            if out_path.exists():
                log.info("Skipping %s — already exists in %s", path.name, out_dir)
                skipped += 1
                continue
            file_rows, n_in = _score_file(path, clf, args)
            _write_csv(out_path, file_rows)
            grand_in += n_in
            grand_out += len(file_rows)
        log.info("Done. %d/%d rows kept (%.1f%%) across %d file(s); %d skipped -> %s",
                 grand_out, grand_in, 100.0 * grand_out / grand_in if grand_in else 0.0,
                 len(input_paths) - skipped, skipped, out_dir)
        return

    # Single-file mode: all inputs merged, sorted by score.
    all_rows: List[Dict] = []
    grand_in = 0
    for path in input_paths:
        file_rows, n_in = _score_file(path, clf, args)
        all_rows.extend(file_rows)
        grand_in += n_in
    _write_csv(Path(args.output), all_rows)
    log.info("Done. %d/%d rows kept (%.1f%%), sorted by score -> %s",
             len(all_rows), grand_in,
             100.0 * len(all_rows) / grand_in if grand_in else 0.0, Path(args.output))


if __name__ == "__main__":
    main()
