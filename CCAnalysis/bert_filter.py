#!/usr/bin/env python3
"""
bert_filter.py — Filter word_context.py CSV output using BERT-based classifiers.

Three classifiers run on each target_context chunk:

  1. Language — papluca/xlm-roberta-base-language-detection
     Drops rows where English-class confidence < --lang-threshold (default 0.85).

  2. Writing quality — textattack/bert-base-uncased-CoLA
     BERT base fine-tuned on the Corpus of Linguistic Acceptability.
     Drops rows where "acceptable" confidence < --quality-threshold (default 0.70).
     Keyword strings, SEO spam, and nav menus score poorly here.

  3. Slang sense disambiguation — bert-base-uncased (contextual embeddings)
     For each word defined in --slang-defs, extracts the word's contextual
     BERT embedding and compares it to prototype embeddings built from
     user-supplied slang and standard example sentences.
     Score = cos_sim(context, slang_proto) - cos_sim(context, standard_proto).
     Drops rows where a defined slang word appears but scores below
     --slang-threshold (default 0.0, i.e. standard sense is more likely).
     Rows containing no defined slang word are not affected by this filter.

All three filters can be disabled independently. Use --score-all to write every
row with scores attached (useful for calibrating thresholds).

Slang definitions file format (YAML):

    fire:
      slang:
        - "That outfit is straight fire, you look amazing"
        - "The new album is fire, every track slaps"
      standard:
        - "The fire burned through the forest overnight"
        - "Firefighters arrived quickly to put out the blaze"
    lit:
      slang:
        - "The party was so lit, everyone had a great time"
      standard:
        - "She lit the candle and placed it on the table"

Usage:
    python bert_filter.py contexts/*.csv -o filtered.csv --slang-defs slang.yaml
    python bert_filter.py input.csv -o scored.csv --score-all --keep-scores
    python bert_filter.py input.csv -o out.csv --no-lang-filter --device cuda
    python bert_filter.py input.csv -o out.csv --slang-defs slang.yaml --slang-threshold 0.05
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

try:
    import torch
    import torch.nn.functional as F
    from transformers import (  # type: ignore
        AutoModel,
        AutoTokenizer,
        pipeline as hf_pipeline,
    )
except ImportError:
    print(
        "ERROR: transformers and torch are required.\n"
        "  pip install transformers torch",
        file=sys.stderr,
    )
    sys.exit(1)

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

# ---------------------------------------------------------------------------
# Default models
# ---------------------------------------------------------------------------

_LANG_MODEL    = "papluca/xlm-roberta-base-language-detection"
_QUALITY_MODEL = "textattack/bert-base-uncased-CoLA"
_SLANG_MODEL   = "bert-base-uncased"

# ---------------------------------------------------------------------------
# Device detection
# ---------------------------------------------------------------------------

def _detect_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"

# ---------------------------------------------------------------------------
# Pipeline score extraction helpers
# ---------------------------------------------------------------------------

def _english_score(label_scores: List[Dict]) -> float:
    for r in label_scores:
        if r["label"] == "en":
            return float(r["score"])
    return 0.0


def _acceptable_score(label_scores: List[Dict]) -> float:
    """LABEL_1 = acceptable in textattack/bert-base-uncased-CoLA."""
    for r in label_scores:
        if r["label"] in ("LABEL_1", "acceptable"):
            return float(r["score"])
    return 0.0

# ---------------------------------------------------------------------------
# CSV helpers
# ---------------------------------------------------------------------------

def _read_rows(path: Path) -> List[Dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def _batched(items: list, size: int) -> Iterator[list]:
    for i in range(0, len(items), size):
        yield items[i : i + size]

# ---------------------------------------------------------------------------
# Slang sense classifier
# ---------------------------------------------------------------------------

def _find_word_positions(tokens: List[str], word_tokens: List[str]) -> List[int]:
    """Return token positions of the first full-word occurrence of word_tokens.

    A full-word match requires that the token immediately after the match is
    not a WordPiece continuation (## prefix), preventing e.g. 'fire' from
    matching inside 'fired' when BERT splits it as ['fire', '##d'].
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


class SlangClassifier:
    """Classify whether target words are used in their slang sense.

    At construction, builds a prototype embedding for the slang and standard
    sense of each defined word by encoding the supplied example sentences with
    BERT and averaging the contextual embeddings of the target word across all
    examples for each sense.

    At inference, the score for a word in a given context is:
        cos_sim(word_embedding_in_context, slang_proto)
      - cos_sim(word_embedding_in_context, standard_proto)

    Positive score → slang sense more likely.
    """

    def __init__(self, model_name: str, definitions: Dict, device: str) -> None:
        log.info("Loading slang model: %s", model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name).to(device).eval()
        self.device = device
        self.prototypes: Dict[str, Tuple[torch.Tensor, torch.Tensor]] = {}
        self._build_prototypes(definitions)

    # ── Prototype construction ────────────────────────────────────────────────

    def _embed_word_in_sentence(
        self, sentence: str, word: str
    ) -> Optional[torch.Tensor]:
        """Return the mean hidden-state vector for word's tokens in sentence."""
        word_tokens = self.tokenizer.tokenize(word)
        if not word_tokens:
            return None

        enc = self.tokenizer(
            sentence, return_tensors="pt", truncation=True, max_length=512
        )
        enc = {k: v.to(self.device) for k, v in enc.items()}
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
                    log.warning(
                        "Slang definition for '%s': word not found in example: %r",
                        word, sentence,
                    )

            for sentence in senses.get("standard", []):
                emb = self._embed_word_in_sentence(sentence, word_lc)
                if emb is not None:
                    standard_embeds.append(emb)
                else:
                    log.warning(
                        "Standard definition for '%s': word not found in example: %r",
                        word, sentence,
                    )

            if not slang_embeds or not standard_embeds:
                log.warning(
                    "Skipping '%s': need at least one usable example per sense.", word
                )
                continue

            slang_proto    = F.normalize(torch.stack(slang_embeds).mean(0), dim=0)
            standard_proto = F.normalize(torch.stack(standard_embeds).mean(0), dim=0)
            self.prototypes[word_lc] = (slang_proto, standard_proto)
            log.info(
                "  '%s' — %d slang / %d standard examples",
                word_lc, len(slang_embeds), len(standard_embeds),
            )

    # ── Batch inference ───────────────────────────────────────────────────────

    def score_batch(
        self, texts: List[str], word: str
    ) -> List[Optional[float]]:
        """Score a list of texts for slang usage of word.

        Returns a list of floats (slang_sim - standard_sim) or None where the
        word's tokens could not be located in the context.
        """
        slang_proto, standard_proto = self.prototypes[word]
        word_tokens = self.tokenizer.tokenize(word)

        enc = self.tokenizer(
            texts,
            return_tensors="pt",
            truncation=True,
            max_length=512,
            padding=True,
        )
        enc = {k: v.to(self.device) for k, v in enc.items()}

        with torch.no_grad():
            hidden = self.model(**enc).last_hidden_state  # (B, L, H)

        results: List[Optional[float]] = []
        for i in range(len(texts)):
            # Use actual (non-padded) token length from attention mask.
            actual_len = int(enc["attention_mask"][i].sum().item())
            tokens = self.tokenizer.convert_ids_to_tokens(
                enc["input_ids"][i, :actual_len]
            )
            positions = _find_word_positions(tokens, word_tokens)
            if not positions:
                results.append(None)
                continue

            emb  = F.normalize(hidden[i, positions].mean(dim=0), dim=0)
            score = (
                torch.dot(emb, slang_proto).item()
                - torch.dot(emb, standard_proto).item()
            )
            results.append(score)

        return results

    def score_all_rows(
        self,
        rows: List[Dict[str, str]],
        batch_size: int,
    ) -> List[Optional[float]]:
        """Return a slang score for each row.

        For each defined word, runs batched inference over the rows that
        contain that word. If multiple defined words appear in one row, the
        highest score is kept (most slang-like interpretation wins).
        Returns None for rows containing no defined slang word.
        """
        n = len(rows)
        texts = [r["target_context"] for r in rows]
        scores: List[Optional[float]] = [None] * n

        for word in self.prototypes:
            relevant = [
                i for i, t in enumerate(texts) if word in t.lower()
            ]
            if not relevant:
                continue

            relevant_texts = [texts[i] for i in relevant]
            batches = list(_batched(relevant_texts, batch_size))
            desc = f"Slang ({word})"
            it = tqdm(batches, desc=desc, unit="batch", leave=False) if HAS_TQDM else batches

            flat: List[Optional[float]] = []
            for batch in it:
                flat.extend(self.score_batch(batch, word))

            for idx, s in zip(relevant, flat):
                if s is not None and (scores[idx] is None or s > scores[idx]):
                    scores[idx] = s

        return scores

# ---------------------------------------------------------------------------
# Core filtering
# ---------------------------------------------------------------------------

def score_rows(
    rows: List[Dict[str, str]],
    lang_pipe,
    quality_pipe,
    slang_classifier: Optional[SlangClassifier],
    batch_size: int,
    lang_threshold: float,
    quality_threshold: float,
) -> List[Tuple[Dict[str, str], float, float, Optional[float]]]:
    """Run all enabled classifiers. Returns (row, lang, quality, slang) per row.

    Slang is None for rows where no defined slang word appears in the context.
    Each later pass skips rows that already failed an earlier threshold.
    """
    n = len(rows)
    texts = [r["target_context"] for r in rows]
    lang_scores    = [1.0] * n
    quality_scores = [1.0] * n
    slang_scores: List[Optional[float]] = [None] * n

    # ── Language ──────────────────────────────────────────────────────────────
    if lang_pipe is not None:
        batches = list(_batched(texts, batch_size))
        it = tqdm(batches, desc="Language", unit="batch", leave=False) if HAS_TQDM else batches
        offset = 0
        for batch in it:
            for j, label_list in enumerate(lang_pipe(batch)):
                lang_scores[offset + j] = _english_score(label_list)
            offset += len(batch)

    # ── Quality (only on language-passing rows) ───────────────────────────────
    if quality_pipe is not None:
        idx_pass = [i for i, ls in enumerate(lang_scores) if ls >= lang_threshold]
        q_texts  = [texts[i] for i in idx_pass]
        q_flat: List[float] = []
        batches = list(_batched(q_texts, batch_size))
        it = tqdm(batches, desc="Quality", unit="batch", leave=False) if HAS_TQDM else batches
        for batch in it:
            for label_list in quality_pipe(batch):
                q_flat.append(_acceptable_score(label_list))
        for i, qs in zip(idx_pass, q_flat):
            quality_scores[i] = qs

    # ── Slang (only on rows passing both earlier filters) ─────────────────────
    if slang_classifier is not None:
        idx_pass = [
            i for i in range(n)
            if lang_scores[i] >= lang_threshold
            and quality_scores[i] >= quality_threshold
        ]
        pass_rows  = [rows[i] for i in idx_pass]
        pass_scores = slang_classifier.score_all_rows(pass_rows, batch_size)
        for i, ss in zip(idx_pass, pass_scores):
            slang_scores[i] = ss

    return [
        (row, ls, qs, ss)
        for row, ls, qs, ss in zip(rows, lang_scores, quality_scores, slang_scores)
    ]

# ---------------------------------------------------------------------------
# YAML loader
# ---------------------------------------------------------------------------

def load_slang_definitions(path: Path) -> Dict:
    if not HAS_YAML:
        log.error(
            "--slang-defs requires PyYAML. Install with: pip install pyyaml"
        )
        sys.exit(1)
    with path.open(encoding="utf-8") as fh:
        data = yaml.safe_load(fh)
    if not isinstance(data, dict):
        log.error("Slang definitions file must be a YAML mapping of word → senses.")
        sys.exit(1)
    return data

# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Filter word_context.py CSV output using BERT-based classifiers.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # All three filters, GPU
  python bert_filter.py contexts/*.csv -o filtered.csv \\
      --slang-defs slang.yaml --device cuda

  # Calibrate thresholds: write every row with all scores
  python bert_filter.py input.csv -o scored.csv --score-all --keep-scores \\
      --slang-defs slang.yaml

  # Language + quality only (no slang definitions available yet)
  python bert_filter.py contexts/*.csv -o filtered.csv

  # Stricter slang gate
  python bert_filter.py input.csv -o out.csv --slang-defs slang.yaml \\
      --slang-threshold 0.05
        """,
    )

    parser.add_argument("inputs", nargs="+", metavar="CSV",
                        help="Input CSV file(s) from word_context.py. Glob patterns accepted.")
    parser.add_argument("-o", "--output", required=True, metavar="FILE",
                        help="Output CSV path.")

    # Thresholds
    parser.add_argument("--lang-threshold", type=float, default=0.85, metavar="F",
                        dest="lang_threshold",
                        help="Min English confidence to keep a row (default: 0.85).")
    parser.add_argument("--quality-threshold", type=float, default=0.70, metavar="F",
                        dest="quality_threshold",
                        help="Min linguistic-acceptability confidence to keep (default: 0.70).")
    parser.add_argument("--slang-threshold", type=float, default=0.0, metavar="F",
                        dest="slang_threshold",
                        help="Min slang score (slang_sim - standard_sim) to keep a row "
                             "when a defined slang word is present (default: 0.0). "
                             "Rows with no defined slang word are unaffected.")

    # Filter toggles
    parser.add_argument("--no-lang-filter", action="store_true", dest="no_lang",
                        help="Skip language detection.")
    parser.add_argument("--no-quality-filter", action="store_true", dest="no_quality",
                        help="Skip writing-quality classification.")

    # Slang definitions
    parser.add_argument("--slang-defs", default=None, metavar="YAML", dest="slang_defs",
                        help="YAML file of slang word definitions (see script docstring). "
                             "Slang filter is skipped when omitted.")

    # Output options
    parser.add_argument("--keep-scores", action="store_true", dest="keep_scores",
                        help="Append lang_score, quality_score, slang_score columns.")
    parser.add_argument("--score-all", action="store_true", dest="score_all",
                        help="Write every row without filtering. Implies --keep-scores. "
                             "Useful for calibrating thresholds.")

    # Inference
    parser.add_argument("--batch-size", type=int, default=32, metavar="N", dest="batch_size",
                        help="Inference batch size (default: 32; increase for GPU).")
    parser.add_argument("--device", default=None,
                        help="cpu | cuda | cuda:N | mps. Auto-detected if omitted.")

    # Model overrides
    parser.add_argument("--lang-model", default=_LANG_MODEL, metavar="ID", dest="lang_model",
                        help=f"Language-detection model (default: {_LANG_MODEL}).")
    parser.add_argument("--quality-model", default=_QUALITY_MODEL, metavar="ID",
                        dest="quality_model",
                        help=f"Quality-classification model (default: {_QUALITY_MODEL}).")
    parser.add_argument("--slang-model", default=_SLANG_MODEL, metavar="ID", dest="slang_model",
                        help=f"BERT model for slang embedding (default: {_SLANG_MODEL}).")

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

    device = args.device or _detect_device()
    log.info("Device: %s", device)

    # ── Load pipeline models ──────────────────────────────────────────────────
    lang_pipe = quality_pipe = None

    if not args.no_lang:
        log.info("Loading language model: %s", args.lang_model)
        lang_pipe = hf_pipeline(
            "text-classification", model=args.lang_model, device=device, top_k=None,
        )

    if not args.no_quality:
        log.info("Loading quality model: %s", args.quality_model)
        quality_pipe = hf_pipeline(
            "text-classification", model=args.quality_model, device=device, top_k=None,
        )

    # ── Load slang classifier ─────────────────────────────────────────────────
    slang_classifier: Optional[SlangClassifier] = None
    if args.slang_defs:
        definitions = load_slang_definitions(Path(args.slang_defs))
        log.info("Building slang prototypes for %d word(s) …", len(definitions))
        slang_classifier = SlangClassifier(args.slang_model, definitions, device)
        log.info("Slang classifier ready (%d word(s) with prototypes).",
                 len(slang_classifier.prototypes))

    if lang_pipe is None and quality_pipe is None and slang_classifier is None:
        parser.error("All filters are disabled — nothing to do.")

    # ── Output setup ──────────────────────────────────────────────────────────
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = ["uri", "target_context"]
    if args.keep_scores:
        fieldnames += ["lang_score", "quality_score", "slang_score"]

    grand_in = grand_out = 0

    with output_path.open("w", newline="", encoding="utf-8") as out_fh:
        writer = csv.DictWriter(out_fh, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()

        for path in input_paths:
            rows = _read_rows(path)
            if not rows:
                log.info("Skipping empty file: %s", path.name)
                continue

            log.info("Processing %s (%d rows) …", path.name, len(rows))

            scored = score_rows(
                rows=rows,
                lang_pipe=lang_pipe,
                quality_pipe=quality_pipe,
                slang_classifier=slang_classifier,
                batch_size=args.batch_size,
                lang_threshold=args.lang_threshold,
                quality_threshold=args.quality_threshold,
            )

            kept = 0
            for row, ls, qs, ss in scored:
                if not args.score_all:
                    if ls < args.lang_threshold:
                        continue
                    if qs < args.quality_threshold:
                        continue
                    # Only apply slang threshold when the word was found and scored.
                    if ss is not None and ss < args.slang_threshold:
                        continue

                out_row: Dict = {
                    "uri": row["uri"],
                    "target_context": row["target_context"],
                }
                if args.keep_scores:
                    out_row["lang_score"]    = f"{ls:.4f}"
                    out_row["quality_score"] = f"{qs:.4f}"
                    out_row["slang_score"]   = f"{ss:.4f}" if ss is not None else ""
                writer.writerow(out_row)
                kept += 1

            grand_in  += len(rows)
            grand_out += kept
            log.info("  %s: %d/%d rows kept (%.1f%%)",
                     path.name, kept, len(rows),
                     100.0 * kept / len(rows) if rows else 0.0)

    log.info(
        "Done. %d/%d rows kept (%.1f%%) → %s",
        grand_out, grand_in,
        100.0 * grand_out / grand_in if grand_in else 0.0,
        output_path,
    )


if __name__ == "__main__":
    main()
