"""Shared helpers for the analysis, stats, and visualizer scripts.

Three operations are needed by nearly every downstream script: matching a slang
target in free text, loading the corpus peak-year table, and resolving a
``--model`` substring against the models present in a response set. They live
here so the scripts agree on the semantics rather than each carrying a copy.

Deliberately free of matplotlib/pandas/numpy so the analysis scripts can import
it without pulling in a plotting stack.
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path


def word_pattern(word: str) -> re.Pattern:
    """Case-insensitive matcher for one target word or phrase.

    Boundary-ish rather than ``\\b``: the lookarounds exclude only adjacent
    letters, so multi-word and punctuated targets ("red pill", "glow-up") match
    as written, as do targets that start or end with a non-letter (digits,
    emoji) where ``\\b`` would fail or match too eagerly.
    """
    return re.compile(rf"(?<![a-z]){re.escape(word)}(?![a-z])", re.IGNORECASE)


def load_vocab(paths: Path | list[Path]) -> list[str]:
    """Union of the words/phrases in one or more target-word files.

    Lowercased, blank and ``#`` comment lines dropped, first-seen order kept.
    """
    if isinstance(paths, Path):
        paths = [paths]
    seen: set[str] = set()
    vocab: list[str] = []
    for p in paths:
        for line in p.read_text(encoding="utf-8").splitlines():
            w = line.strip().lower()
            if w and not w.startswith("#") and w not in seen:
                seen.add(w)
                vocab.append(w)
    return vocab


def load_peak_records(path: Path, min_hits: int = 0) -> dict[str, dict]:
    """``{word: {"peak_year", "total_hits"}}`` from FineWebAnalysis/peak_year.py.

    ``total_hits`` gauges how trustworthy a peak is; words below ``min_hits`` are
    dropped. Returns ``{}`` when the file is absent so callers can treat the
    peak table as optional.
    """
    if not path.is_file():
        return {}
    return {
        r["word"].lower(): {"peak_year": int(r["peak_year"]),
                            "total_hits": int(r.get("total_hits", 0))}
        for r in json.loads(path.read_text(encoding="utf-8"))
        if "word" in r and "peak_year" in r and int(r.get("total_hits", 0)) >= min_hits
    }


def load_peak_years(path: Path, min_hits: int = 0) -> dict[str, int]:
    """``{word: peak year}`` — the common case of :func:`load_peak_records`."""
    return {w: rec["peak_year"] for w, rec in load_peak_records(path, min_hits).items()}


def pick_model(models, requested: str | None, empty_msg: str) -> str:
    """Resolve ``--model`` (a case-insensitive substring) against the models present.

    With no request, returns the first model and names the others so the caller
    knows it can narrow the selection. Exits on no match or an ambiguous one.
    """
    models = sorted(models)
    if not models:
        sys.exit(empty_msg)
    if requested is None:
        chosen = models[0]
        if len(models) > 1:
            print(f"Charting model '{chosen}'. Other models present "
                  f"(use --model to pick): {', '.join(m for m in models if m != chosen)}")
        return chosen
    matches = [m for m in models if requested.lower() in m.lower()]
    if not matches:
        sys.exit(f"No model matching '{requested}'. Available: {', '.join(models)}")
    if len(matches) > 1:
        sys.exit(f"'{requested}' matches several models: {', '.join(matches)}. Be more specific.")
    return matches[0]
