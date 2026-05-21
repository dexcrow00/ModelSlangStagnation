#!/usr/bin/env python3
"""
bert_slang_filter.py — Slang sense scorer with two independent methods.

Scores each context row by how likely the target word is used in its slang
sense, running two methods in parallel for comparison:

  Method 1 — SBERT prototype matching (--sbert-model):
    Encodes the full context sentence with a sentence-transformer model trained
    for semantic similarity. Compares to per-word sense centroids built from
    slang.yaml examples.
    score = cos_sim(context, slang_proto) - cos_sim(context, standard_proto)
    For slang-only words (no standard examples): score = cos_sim(context, slang_proto)

  Method 2 — Zero-shot NLI (--nli-model):
    Frames sense classification as natural language inference.
    score = P(entailment | "X used as slang") - P(entailment | "X used literally")
    Needs no prototype examples — sense disambiguation is handled by the NLI model.

Both scores are written as separate columns for side-by-side comparison.
Rows containing no defined slang word get empty scores and are passed through.

Requires: sentence-transformers, pyyaml  (see requirements_bert_slang_filter.txt)

Usage:
    # Compare both methods — write all rows with scores
    python bert_slang_filter.py input.csv -o scored.csv \\
        --slang-defs slang.yaml --score-all

    # Filter using both scores
    python bert_slang_filter.py contexts/*.csv -o filtered.csv \\
        --slang-defs slang.yaml --sbert-threshold 0.1 --nli-threshold 0.1

    # SBERT only, lighter model
    python bert_slang_filter.py input.csv -o out.csv \\
        --slang-defs slang.yaml --no-nli --sbert-model all-MiniLM-L6-v2
"""

from __future__ import annotations

import argparse
import csv
import glob
import logging
import sys
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from sentence_transformers import CrossEncoder, SentenceTransformer

try:
    import yaml  # type: ignore
    HAS_YAML = True
except ImportError:
    HAS_YAML = False

try:
    from tqdm import tqdm  # type: ignore
    HAS_TQDM = True
except ImportError:
    HAS_TQDM = False

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

_SBERT_MODEL = "all-mpnet-base-v2"
_NLI_MODEL   = "cross-encoder/nli-deberta-v3-large"

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _batched(items: list, size: int) -> Iterator[list]:
    for i in range(0, len(items), size):
        yield items[i : i + size]


def _read_rows(path: Path) -> List[Dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def load_slang_definitions(path: Path) -> Dict:
    if not HAS_YAML:
        log.error("PyYAML is required. pip install pyyaml")
        sys.exit(1)
    with path.open(encoding="utf-8") as fh:
        data = yaml.safe_load(fh)
    if not isinstance(data, dict):
        log.error("Slang definitions file must be a YAML mapping of word -> senses.")
        sys.exit(1)
    return data

# ---------------------------------------------------------------------------
# Method 1 — SBERT prototype matching
# ---------------------------------------------------------------------------

class SBERTSlangClassifier:
    """Sentence-BERT prototype-based slang sense scorer.

    Encodes full context sentences and compares to per-word sense centroids
    built from slang.yaml example sentences.
    """

    def __init__(self, model: SentenceTransformer, definitions: Dict) -> None:
        self.model = model
        self.prototypes: Dict[str, Tuple[torch.Tensor, Optional[torch.Tensor]]] = {}
        self._build_prototypes(definitions)

    def _encode(self, sentences: List[str], batch_size: int = 32) -> torch.Tensor:
        embs = self.model.encode(
            sentences, batch_size=batch_size,
            show_progress_bar=False, convert_to_numpy=True,
        )
        return F.normalize(torch.tensor(embs), dim=1)

    def _build_prototypes(self, definitions: Dict) -> None:
        for word, senses in definitions.items():
            word_lc     = word.lower()
            slang_sents = senses.get("slang", [])
            std_sents   = senses.get("standard", [])

            if not slang_sents:
                log.warning("SBERT: skipping '%s' — no slang examples.", word)
                continue

            slang_proto = F.normalize(self._encode(slang_sents).mean(dim=0), dim=0)

            if std_sents:
                std_proto = F.normalize(self._encode(std_sents).mean(dim=0), dim=0)
                self.prototypes[word_lc] = (slang_proto, std_proto)
                log.info("  SBERT '%s' — %d slang / %d standard examples",
                         word_lc, len(slang_sents), len(std_sents))
            else:
                self.prototypes[word_lc] = (slang_proto, None)
                log.info("  SBERT '%s' — %d slang examples, slang-only mode",
                         word_lc, len(slang_sents))

    def score_rows(self, rows: List[Dict[str, str]], batch_size: int) -> List[Optional[float]]:
        texts  = [r["target_context"] for r in rows]
        scores: List[Optional[float]] = [None] * len(rows)

        for word, (slang_proto, std_proto) in self.prototypes.items():
            relevant = [i for i, t in enumerate(texts) if word in t.lower()]
            if not relevant:
                continue

            embs = self._encode([texts[i] for i in relevant], batch_size)  # (N, D), L2-normed
            slang_sims = (embs @ slang_proto).tolist()

            if std_proto is not None:
                std_sims    = (embs @ std_proto).tolist()
                word_scores = [float(s - t) for s, t in zip(slang_sims, std_sims)]
            else:
                word_scores = [float(s) for s in slang_sims]

            for idx, s in zip(relevant, word_scores):
                if scores[idx] is None or s > scores[idx]:
                    scores[idx] = s

        return scores

# ---------------------------------------------------------------------------
# Method 2 — Zero-shot NLI
# ---------------------------------------------------------------------------

class NLISlangClassifier:
    """Zero-shot NLI slang sense scorer.

    Uses a cross-encoder NLI model to score entailment for slang vs. standard
    sense hypotheses. No prototype examples needed.
    """

    _SLANG_TMPL = "In this text, the word '{word}' is used as internet slang or informal language."
    _STD_TMPL   = "In this text, the word '{word}' is used with its literal, standard dictionary meaning."

    def __init__(self, model: CrossEncoder, words: List[str]) -> None:
        self.model = model
        self.words = [w.lower() for w in words]
        label2id   = getattr(model.model.config, "label2id", {})
        label2id_lc = {k.lower(): v for k, v in label2id.items()}
        self.entailment_idx = label2id_lc.get("entailment", 1)
        log.info("  NLI entailment class index: %d  (label2id: %s)",
                 self.entailment_idx, label2id)

    def _entailment_probs(self, pairs: List[Tuple[str, str]], batch_size: int) -> np.ndarray:
        logits = self.model.predict(pairs, batch_size=batch_size,
                                    show_progress_bar=False)  # (N, 3)
        if logits.ndim == 1:
            logits = logits.reshape(1, -1)
        exp_x = np.exp(logits - logits.max(axis=1, keepdims=True))
        probs  = exp_x / exp_x.sum(axis=1, keepdims=True)
        return probs[:, self.entailment_idx]

    def score_rows(self, rows: List[Dict[str, str]], batch_size: int) -> List[Optional[float]]:
        texts  = [r["target_context"] for r in rows]
        scores: List[Optional[float]] = [None] * len(rows)

        for word in self.words:
            relevant = [i for i, t in enumerate(texts) if word in t.lower()]
            if not relevant:
                continue

            rel_texts   = [texts[i] for i in relevant]
            slang_pairs = [(t, self._SLANG_TMPL.format(word=word)) for t in rel_texts]
            std_pairs   = [(t, self._STD_TMPL.format(word=word))   for t in rel_texts]

            # Single predict call: slang pairs first, then standard
            log.info("  NLI scoring '%s': %d rows ...", word, len(rel_texts))
            all_probs   = self._entailment_probs(slang_pairs + std_pairs, batch_size)
            n           = len(rel_texts)
            slang_probs = all_probs[:n]
            std_probs   = all_probs[n:]

            for idx, sp, tp in zip(relevant, slang_probs, std_probs):
                s = float(sp - tp)
                if scores[idx] is None or s > scores[idx]:
                    scores[idx] = s

        return scores

# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Score slang sense using SBERT prototype matching and zero-shot NLI.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Compare both methods on all rows (recommended starting point)
  python bert_slang_filter.py input.csv -o scored.csv \\
      --slang-defs slang.yaml --score-all

  # Filter using both scores
  python bert_slang_filter.py contexts/*.csv -o filtered.csv \\
      --slang-defs slang.yaml --sbert-threshold 0.1 --nli-threshold 0.1

  # SBERT only, lighter model
  python bert_slang_filter.py input.csv -o out.csv \\
      --slang-defs slang.yaml --no-nli --sbert-model all-MiniLM-L6-v2

  # NLI only
  python bert_slang_filter.py input.csv -o out.csv \\
      --slang-defs slang.yaml --no-sbert
        """,
    )

    p.add_argument("inputs", nargs="+", metavar="CSV",
                   help="Input CSV file(s) with uri and target_context columns. "
                        "Glob patterns accepted.")
    p.add_argument("-o", "--output", required=True, metavar="FILE",
                   help="Output CSV path.")
    p.add_argument("--slang-defs", required=True, metavar="YAML", dest="slang_defs",
                   help="YAML file of slang word definitions (slang.yaml).")
    p.add_argument("--score-all", action="store_true", dest="score_all",
                   help="Write every row with scores attached, without filtering. "
                        "Recommended for comparing the two methods.")

    # Thresholds
    p.add_argument("--sbert-threshold", type=float, default=0.0, metavar="F",
                   dest="sbert_threshold",
                   help="Min SBERT score to keep a row (default: 0.0).")
    p.add_argument("--nli-threshold", type=float, default=0.0, metavar="F",
                   dest="nli_threshold",
                   help="Min NLI score to keep a row (default: 0.0).")

    # Method toggles
    p.add_argument("--no-sbert", action="store_true", dest="no_sbert",
                   help="Skip SBERT scoring.")
    p.add_argument("--no-nli", action="store_true", dest="no_nli",
                   help="Skip NLI scoring.")

    # Models
    p.add_argument("--sbert-model", default=_SBERT_MODEL, metavar="ID", dest="sbert_model",
                   help=f"Sentence-transformer model for SBERT scoring "
                        f"(default: {_SBERT_MODEL}). "
                        f"Lighter alternative: all-MiniLM-L6-v2 (~80 MB).")
    p.add_argument("--nli-model", default=_NLI_MODEL, metavar="ID", dest="nli_model",
                   help=f"Cross-encoder NLI model "
                        f"(default: {_NLI_MODEL}). "
                        f"Lighter alternative: cross-encoder/nli-MiniLM2-L6-H768 (~100 MB).")
    p.add_argument("--batch-size", type=int, default=32, metavar="N", dest="batch_size",
                   help="Inference batch size for both methods (default: 32).")

    return p


def main() -> None:
    parser = build_parser()
    args   = parser.parse_args()

    if args.no_sbert and args.no_nli:
        parser.error("Both methods are disabled — nothing to do.")

    # Expand glob patterns
    input_paths: List[Path] = []
    for pattern in args.inputs:
        matched = sorted(glob.glob(pattern, recursive=True))
        if matched:
            input_paths.extend(Path(p) for p in matched)
        else:
            input_paths.append(Path(pattern))

    missing = [p for p in input_paths if not p.exists()]
    if missing:
        parser.error(f"File(s) not found: {', '.join(str(p) for p in missing)}")

    definitions = load_slang_definitions(Path(args.slang_defs))

    # ── Load models ───────────────────────────────────────────────────────────
    sbert_clf: Optional[SBERTSlangClassifier] = None
    if not args.no_sbert:
        log.info("Loading SBERT model: %s", args.sbert_model)
        sbert_clf = SBERTSlangClassifier(
            SentenceTransformer(args.sbert_model), definitions
        )

    nli_clf: Optional[NLISlangClassifier] = None
    if not args.no_nli:
        log.info("Loading NLI model: %s", args.nli_model)
        nli_clf = NLISlangClassifier(
            CrossEncoder(args.nli_model, num_labels=3),
            list(definitions.keys()),
        )

    # ── Output setup ──────────────────────────────────────────────────────────
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = ["uri", "target_context"]
    if not args.no_sbert:
        fieldnames.append("sbert_score")
    if not args.no_nli:
        fieldnames.append("nli_score")

    grand_in = grand_out = 0

    with output_path.open("w", newline="", encoding="utf-8") as out_fh:
        writer = csv.DictWriter(out_fh, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()

        for path in input_paths:
            rows = _read_rows(path)
            if not rows:
                log.info("Skipping empty file: %s", path.name)
                continue

            log.info("Processing %s (%d rows) ...", path.name, len(rows))

            n = len(rows)
            sbert_scores = sbert_clf.score_rows(rows, args.batch_size) if sbert_clf else [None] * n
            nli_scores   = nli_clf.score_rows(rows, args.batch_size)   if nli_clf   else [None] * n

            kept = 0
            for row, ss, ns in zip(rows, sbert_scores, nli_scores):
                if not args.score_all:
                    if sbert_clf and ss is not None and ss < args.sbert_threshold:
                        continue
                    if nli_clf   and ns is not None and ns < args.nli_threshold:
                        continue

                out_row: Dict = {"uri": row["uri"], "target_context": row["target_context"]}
                if not args.no_sbert:
                    out_row["sbert_score"] = f"{ss:.4f}" if ss is not None else ""
                if not args.no_nli:
                    out_row["nli_score"] = f"{ns:.4f}" if ns is not None else ""
                writer.writerow(out_row)
                kept += 1

            grand_in  += len(rows)
            grand_out += kept
            log.info("  %s: %d/%d rows kept (%.1f%%)",
                     path.name, kept, len(rows),
                     100.0 * kept / len(rows) if rows else 0.0)

    log.info("Done. %d/%d rows kept (%.1f%%) -> %s",
             grand_out, grand_in,
             100.0 * grand_out / grand_in if grand_in else 0.0,
             output_path)


if __name__ == "__main__":
    main()
