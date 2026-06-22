"""Shared helpers for reading experiment response JSONL and parsing logprobs.

Used by the per-experiment visualizers. Responses are written by
ResponseCollector as pretty-printed JSON blocks separated by blank lines, one
file per model (``<model>_<timestamp>.jsonl``).
"""

from __future__ import annotations

import glob
import json
import math
import os
import re
from pathlib import Path

_DECODER = json.JSONDecoder()
_TS_RE = re.compile(r"_(\d{8})_(\d{6})\.jsonl$")


def _file_timestamp(path: str) -> str:
    """Sort key from a ``<model>_<YYYYMMDD>_<HHMMSS>.jsonl`` filename.

    Falls back to mtime so undated filenames still order sensibly.
    """
    m = _TS_RE.search(os.path.basename(path))
    return f"{m.group(1)}{m.group(2)}" if m else f"mtime:{os.path.getmtime(path):020.0f}"


def _read_file(path: str) -> list[dict]:
    text = Path(path).read_text(encoding="utf-8")
    out: list[dict] = []
    i = 0
    while i < len(text):
        while i < len(text) and text[i] in " \t\r\n":
            i += 1
        if i >= len(text):
            break
        try:
            obj, i = _DECODER.raw_decode(text, i)
        except json.JSONDecodeError:
            break  # incomplete trailing record — stop on this file
        out.append(obj)
    return out


def read_responses(path: str | Path, latest_only: bool = True) -> list[dict]:
    """Read response records from a JSONL file, or every ``*.jsonl`` in a dir.

    Tolerates the pretty-printed record format and a trailing partial record
    (e.g. a file still being written). When *path* is a directory and
    ``latest_only`` is set (default), records are de-duplicated by
    ``(model, prompt_id, variables, sample)`` keeping the one from the
    newest-timestamped file — so re-runs in the same directory don't double
    count; only the most recent run's data survives for each prompt sample.
    """
    path = str(path)
    if not os.path.isdir(path):
        return _read_file(path)

    files = sorted(glob.glob(os.path.join(path, "*.jsonl")), key=_file_timestamp)
    if not latest_only:
        return [r for f in files for r in _read_file(f)]

    # Process oldest -> newest so later (newer) files overwrite duplicate keys.
    by_key: dict[tuple, dict] = {}
    for f in files:
        for rec in _read_file(f):
            key = (rec.get("model"), rec.get("prompt_id"),
                   json.dumps(rec.get("variables"), sort_keys=True), rec.get("sample"))
            by_key[key] = rec
    return list(by_key.values())


def model_short(model: str) -> str:
    """Short display name for a model id (drops the org/ prefix)."""
    return model.split("/")[-1]


def _strip_routing(tokens, lps, tops):
    """Drop leading <|special|> routing tokens (e.g. gpt-oss channel headers)."""
    i = 0
    while i < len(tokens) and isinstance(tokens[i], str) and tokens[i].startswith("<|"):
        i += 1
        if i < len(tokens) and not tokens[i].startswith("<|"):
            i += 1
    return tokens[i:], lps[i:], (tops[i:] if tops else [])


def normalize_logprobs(lp) -> tuple[list[str], list[float], list[dict]]:
    """Return (tokens, token_logprobs, top_logprobs) for Together or OpenAI formats.

    ``top_logprobs`` is a list (one per position) of ``{token: logprob}`` dicts.
    """
    if not isinstance(lp, dict):
        return [], [], []
    content = lp.get("content")
    if content and isinstance(content, list):  # OpenAI-compatible
        tokens = [c["token"] for c in content]
        lps = [float(c["logprob"]) for c in content]
        tops = [{a["token"]: a["logprob"] for a in (c.get("top_logprobs") or [])}
                for c in content]
    else:  # Together-native
        tokens = lp.get("tokens") or []
        lps = [float(x) if x is not None else 0.0 for x in (lp.get("token_logprobs") or [])]
        raw_tops = lp.get("top_logprobs") or []
        tops = []
        for t in raw_tops:
            if isinstance(t, dict):
                tops.append({k: float(v) for k, v in t.items()})
            else:
                tops.append({})
    return _strip_routing(tokens, lps, tops)


_STOP_CHARS = set(".!?,\n \t")


def responded_word(rec: dict) -> str:
    """The word/phrase a model produced for a single-word prompt.

    Uses ``response`` text when present (closed/sampled prompts); otherwise
    reconstructs it from the leading logprob tokens (open/logprob prompts).
    Lowercased and stripped of surrounding quotes/punctuation.
    """
    resp = rec.get("response")
    if resp:
        return resp.strip().strip('"\'.').lower()
    tokens, _lps, _tops = normalize_logprobs(rec.get("logprobs"))
    parts: list[str] = []
    for tok in tokens:
        if not isinstance(tok, str) or tok.startswith("<"):
            break
        if any(ch in tok for ch in ".!?,\n") or (tok.strip() == "" and parts):
            break
        parts.append(tok)
    return "".join(parts).strip().strip('"\'.').lower()


def joint_logprob(rec: dict) -> float | None:
    """Sum of token logprobs of the generated response (up to the first stop/EOS).

    None when the record carries no usable logprobs.
    """
    tokens, lps, _tops = normalize_logprobs(rec.get("logprobs"))
    total, got = 0.0, False
    for tok, lp in zip(tokens, lps):
        if not isinstance(tok, str) or tok.startswith("<"):
            break
        if any(ch in tok for ch in ".!?,\n") or (tok.strip() == "" and got):
            break
        total += lp
        got = True
    return total if got else None


def first_token_distribution(rec: dict) -> dict[str, float]:
    """Normalised probabilities over first-position candidates from top_logprobs[0].

    Empty dict when the record has no usable logprobs.
    """
    _tokens, _lps, tops = normalize_logprobs(rec.get("logprobs"))
    if not tops or not isinstance(tops[0], dict):
        return {}
    probs = {tok.strip().lower(): math.exp(lp)
             for tok, lp in tops[0].items() if tok.strip()}
    total = sum(probs.values())
    return {t: p / total for t, p in probs.items()} if total else probs
