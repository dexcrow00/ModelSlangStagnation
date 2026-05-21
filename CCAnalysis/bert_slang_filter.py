#!/usr/bin/env python3
"""
bert_slang_filter.py — BERT-based slang sense classifier.

Reads word_context.py CSV output (uri, target_context columns) and scores each
row by how likely the target word is being used in its slang sense.

Two scoring modes per word (configured via slang.yaml):
  Both senses defined (slang + standard examples):
    score = cos_sim(context, slang_proto) - cos_sim(context, standard_proto)
    Positive -> slang sense; negative -> standard sense.
  Slang examples only:
    score = cos_sim(context, slang_proto)  [0-1 absolute similarity]

Rows containing no defined slang word are passed through unfiltered.

Requires: torch, transformers, pyyaml
See requirements_bert_slang_filter.txt for installation instructions.

Usage:
    python bert_slang_filter.py contexts/*.csv -o scored.csv --slang-defs slang.yaml

    # Write all rows with scores attached (for threshold calibration)
    python bert_slang_filter.py input.csv -o scored.csv --slang-defs slang.yaml --score-all

    # Stricter gate
    python bert_slang_filter.py input.csv -o out.csv --slang-defs slang.yaml --slang-threshold 0.05
"""

from __future__ import annotations

import argparse
import csv
import glob
import logging
import os
import sys
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Tuple

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import torch
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer  # type: ignore

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

_BERT_MODEL = "bert-base-uncased"

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _batched(items: list, size: int) -> Iterator[list]:
    for i in range(0, len(items), size):
        yield items[i : i + size]


def _read_rows(path: Path) -> List[Dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def _find_word_positions(tokens: List[str], word_tokens: List[str]) -> List[int]:
    """Return token positions of the first full-word occurrence of word_tokens.

    Checks that the token immediately after the match is not a WordPiece
    continuation (## prefix), preventing 'fire' from matching inside 'fired'.
    Returns [] if not found.
    """
    n = len(word_tokens)
    for i in range(len(tokens) - n + 1):
        if tokens[i : i + n] != word_tokens:
            continue
        next_pos = i + n
        if next_pos < len(tokens) and tokens[next_pos].startswith("##"):
            continue
        return list(range(i, i + n))
    return []

# ---------------------------------------------------------------------------
# Slang classifier
# ---------------------------------------------------------------------------

class SlangClassifier:
    """Classify whether target words are used in their slang sense.

    Builds prototype embeddings for each defined word from example sentences.
    At inference, scores a context by comparing the word's contextual embedding
    to the prototypes.
    """

    def __init__(self, tokenizer, model, definitions: Dict) -> None:
        self.tokenizer = tokenizer
        self.model = model
        self.prototypes: Dict[str, Tuple[torch.Tensor, Optional[torch.Tensor]]] = {}
        self._build_prototypes(definitions)

    def _embed_word_in_sentence(self, sentence: str, word: str) -> Optional[torch.Tensor]:
        word_tokens = self.tokenizer.tokenize(word)
        if not word_tokens:
            return None
        enc = self.tokenizer(sentence, return_tensors="pt", truncation=True, max_length=512)
        tokens = self.tokenizer.convert_ids_to_tokens(enc["input_ids"][0])
        positions = _find_word_positions(tokens, word_tokens)
        if not positions:
            return None
        with torch.no_grad():
            hidden = self.model(**enc).last_hidden_state[0]  # (L, H)
        return hidden[positions].mean(dim=0)

    def _build_prototypes(self, definitions: Dict) -> None:
        for word, senses in definitions.items():
            word_lc = word.lower()
            slang_embeds, standard_embeds = [], []

            for sentence in senses.get("slang", []):
                emb = self._embed_word_in_sentence(sentence, word_lc)
                if emb is not None:
                    slang_embeds.append(emb)
                else:
                    log.warning("Slang def for '%s': word not found in: %r", word, sentence)

            for sentence in senses.get("standard", []):
                emb = self._embed_word_in_sentence(sentence, word_lc)
                if emb is not None:
                    standard_embeds.append(emb)
                else:
                    log.warning("Standard def for '%s': word not found in: %r", word, sentence)

            if not slang_embeds:
                log.warning("Skipping '%s': no usable slang examples.", word)
                continue

            slang_proto = F.normalize(torch.stack(slang_embeds).mean(0), dim=0)

            if not standard_embeds:
                log.info("  '%s' — %d slang examples, slang-only mode", word_lc, len(slang_embeds))
                self.prototypes[word_lc] = (slang_proto, None)
                continue

            standard_proto = F.normalize(torch.stack(standard_embeds).mean(0), dim=0)
            self.prototypes[word_lc] = (slang_proto, standard_proto)
            log.info("  '%s' — %d slang / %d standard examples",
                     word_lc, len(slang_embeds), len(standard_embeds))

    def score_batch(self, texts: List[str], word: str) -> List[Optional[float]]:
        slang_proto, standard_proto = self.prototypes[word]
        word_tokens = self.tokenizer.tokenize(word)

        enc = self.tokenizer(
            texts, return_tensors="pt", truncation=True, max_length=512, padding=True
        )
        with torch.no_grad():
            hidden = self.model(**enc).last_hidden_state  # (B, L, H)

        results: List[Optional[float]] = []
        for i in range(len(texts)):
            actual_len = int(enc["attention_mask"][i].sum().item())
            tokens = self.tokenizer.convert_ids_to_tokens(enc["input_ids"][i, :actual_len])
            positions = _find_word_positions(tokens, word_tokens)
            if not positions:
                results.append(None)
                continue
            emb = F.normalize(hidden[i, positions].mean(dim=0), dim=0)
            slang_sim = torch.dot(emb, slang_proto).item()
            score = (
                slang_sim - torch.dot(emb, standard_proto).item()
                if standard_proto is not None else slang_sim
            )
            results.append(score)

        return results

    def score_all_rows(self, rows: List[Dict[str, str]], batch_size: int) -> List[Optional[float]]:
        """Score every row. Returns None for rows with no defined slang word present."""
        n = len(rows)
        texts = [r["target_context"] for r in rows]
        scores: List[Optional[float]] = [None] * n

        for word in self.prototypes:
            relevant = [i for i, t in enumerate(texts) if word in t.lower()]
            if not relevant:
                continue
            relevant_texts = [texts[i] for i in relevant]
            batches = list(_batched(relevant_texts, batch_size))
            it = tqdm(batches, desc=f"Slang ({word})", unit="batch", leave=False) if HAS_TQDM else batches
            flat: List[Optional[float]] = []
            for batch in it:
                flat.extend(self.score_batch(batch, word))
            for idx, s in zip(relevant, flat):
                if s is not None and (scores[idx] is None or s > scores[idx]):
                    scores[idx] = s

        return scores

# ---------------------------------------------------------------------------
# YAML loader
# ---------------------------------------------------------------------------

def load_slang_definitions(path: Path) -> Dict:
    if not HAS_YAML:
        log.error("PyYAML is required. Install with: pip install pyyaml")
        sys.exit(1)
    with path.open(encoding="utf-8") as fh:
        data = yaml.safe_load(fh)
    if not isinstance(data, dict):
        log.error("Slang definitions file must be a YAML mapping of word -> senses.")
        sys.exit(1)
    return data

# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def build_classifier(
    slang_defs_path: Path,
    bert_model: str = _BERT_MODEL,
    quantize: bool = True,
) -> SlangClassifier:
    """Load model, optionally quantize, build prototypes, return a ready classifier."""
    log.info("Loading BERT model: %s", bert_model)
    tokenizer = AutoTokenizer.from_pretrained(bert_model)
    model = AutoModel.from_pretrained(bert_model).eval()

    if quantize:
        log.info("Applying int8 dynamic quantization ...")
        model = torch.quantization.quantize_dynamic(
            model, {torch.nn.Linear}, dtype=torch.qint8
        )

    definitions = load_slang_definitions(slang_defs_path)
    log.info("Building slang prototypes for %d word(s) ...", len(definitions))
    classifier = SlangClassifier(tokenizer, model, definitions)
    log.info("Slang classifier ready (%d word(s) with prototypes).", len(classifier.prototypes))
    return classifier

# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Score word_context.py CSV output for slang sense using BERT embeddings.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python bert_slang_filter.py contexts/*.csv -o scored.csv --slang-defs slang.yaml
  python bert_slang_filter.py input.csv -o scored.csv --slang-defs slang.yaml --score-all
  python bert_slang_filter.py input.csv -o out.csv --slang-defs slang.yaml --slang-threshold 0.05
        """,
    )

    parser.add_argument("inputs", nargs="+", metavar="CSV",
                        help="Input CSV file(s) with uri and target_context columns. "
                             "Glob patterns accepted.")
    parser.add_argument("-o", "--output", required=True, metavar="FILE",
                        help="Output CSV path.")
    parser.add_argument("--slang-defs", required=True, metavar="YAML", dest="slang_defs",
                        help="YAML file of slang word definitions (slang.yaml).")
    parser.add_argument("--slang-threshold", type=float, default=0.0, metavar="F",
                        dest="slang_threshold",
                        help="Min slang score to keep a row (default: 0.0). "
                             "For words with both senses: slang_sim - standard_sim. "
                             "For slang-only words: absolute cosine similarity (0-1). "
                             "Rows with no defined slang word are passed through.")
    parser.add_argument("--score-all", action="store_true", dest="score_all",
                        help="Write every row without filtering. Implies --keep-scores.")
    parser.add_argument("--keep-scores", action="store_true", dest="keep_scores",
                        help="Append a slang_score column to the output.")
    parser.add_argument("--batch-size", type=int, default=32, metavar="N", dest="batch_size",
                        help="BERT inference batch size (default: 32).")
    parser.add_argument("--bert-model", default=_BERT_MODEL, metavar="ID", dest="bert_model",
                        help=f"BERT checkpoint to use (default: {_BERT_MODEL}).")
    parser.add_argument("--no-quantize", action="store_true", dest="no_quantize",
                        help="Disable int8 dynamic quantization. "
                             "Quantization is on by default and reduces RAM ~4x.")

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    if args.score_all:
        args.keep_scores = True

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

    classifier = build_classifier(
        Path(args.slang_defs),
        bert_model=args.bert_model,
        quantize=not args.no_quantize,
    )

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = ["uri", "target_context"]
    if args.keep_scores:
        fieldnames.append("slang_score")

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
            slang_scores = classifier.score_all_rows(rows, args.batch_size)

            kept = 0
            for row, ss in zip(rows, slang_scores):
                if not args.score_all and ss is not None and ss < args.slang_threshold:
                    continue

                out_row: Dict = {"uri": row["uri"], "target_context": row["target_context"]}
                if args.keep_scores:
                    out_row["slang_score"] = f"{ss:.4f}" if ss is not None else ""
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
