#!/usr/bin/env python3
"""Experiment 3 analysis: forced-choice (A/B/C/D) variant of the semantic slots.

The free-response prompts ask the model to name a slang word outright; this
variant hands it four candidates and asks for a letter. Scoring is therefore a
lookup rather than a text match: parse the chosen letter, resolve it back to the
word it stood for in that prompt, and check whether it was the slot's target.

Because each slot is asked once per calendar year, the useful view is the target
rate *per year* — if the "The year is {year}." cue steers vocabulary at all, the
target rate should track the word's real-world lifecycle rather than sit flat.

Usage (run from the PromptingSlang root):
    python experiments/semantic_slot_queries/analysis/exp3_choice_freq.py
    python experiments/semantic_slot_queries/analysis/exp3_choice_freq.py --slot exp3_aura
    python experiments/semantic_slot_queries/analysis/exp3_choice_freq.py --csv choices.csv
"""

from __future__ import annotations

import argparse
import csv
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]           # PromptingSlang
sys.path.insert(0, str(REPO_ROOT))
from src.response_utils import read_responses  # noqa: E402

DEFAULT_RESPONSES = Path(__file__).resolve().parents[1] / "results_from_list"

# "... multiple choice: A) lmao B) lol C) haha D) rofl" -> the option block.
_OPTS_TAIL = re.compile(r"multiple choice:\s*(.*)$", re.IGNORECASE | re.DOTALL)
# One option: letter, then everything up to the next letter marker or the end.
# Non-greedy with a lookahead so multi-word values ("red pill", "solo dolo",
# "friends with benefits") stay intact.
_OPT = re.compile(r"\b([ABCD])\)\s*(.*?)(?=\s+\b[ABCD]\)|$)", re.DOTALL)
# A bare letter, optionally wrapped: "B", "B)", "**B) rad**".
_ANSWER = re.compile(r"^\W*([ABCD])\b")


def normalize(word: str) -> str:
    """Fold the spellings that differ between a prompt id and its option text.

    ``exp3_glow_up`` has to match the option ``glow-up``, so underscores and
    hyphens both collapse to spaces.
    """
    return re.sub(r"[\s_-]+", " ", word.strip().lower())


def parse_options(prompt_text: str) -> dict[str, str]:
    """``{letter: word}`` for one rendered prompt, or ``{}`` if unparseable."""
    tail = _OPTS_TAIL.search(prompt_text or "")
    if not tail:
        return {}
    return {letter: word.strip() for letter, word in _OPT.findall(tail.group(1))}


def parse_answer(response: str | None) -> tuple[str | None, str]:
    """``(letter, status)`` for one response.

    Status is ``clean`` for a bare letter, ``lenient`` when a letter was
    recovered from surrounding prose or markup, and otherwise a reason the
    record was dropped. ``enumeration`` catches the degenerate "A, B, C, D"
    reply — a model echoing the instruction's letter list instead of choosing —
    which would otherwise be silently scored as a vote for A.
    """
    s = (response or "").strip().strip("*").strip()
    if not s:
        return None, "empty"
    if len({m for m in re.findall(r"\b([ABCD])\b", s)}) >= 3:
        return None, "enumeration"
    m = _ANSWER.match(s)
    if not m:
        return None, "unparsed"
    return m.group(1), "clean" if re.fullmatch(r"[ABCD][).]?", s) else "lenient"


def target_letter(slot: str, options: dict[str, str]) -> str | None:
    """Which option holds the slot's own word (``exp3_aura`` -> the ``aura`` option)."""
    want = normalize(slot.removeprefix("exp3_"))
    for letter, word in options.items():
        if normalize(word) == want:
            return letter
    return None


def score(records: list[dict]) -> tuple[list[dict], Counter]:
    """Flatten records into scored rows, plus a tally of parse outcomes."""
    rows: list[dict] = []
    health: Counter = Counter()
    for rec in records:
        slot = rec.get("prompt_id", "")
        if not slot.startswith("exp3_"):
            continue
        options = parse_options(rec.get("prompt_text", ""))
        target = target_letter(slot, options)
        if not options or target is None:
            health["no_options"] += 1
            continue
        letter, status = parse_answer(rec.get("response"))
        health[status] += 1
        if letter is None:
            continue
        rows.append({
            "model": rec.get("model", "?"),
            "slot": slot,
            "target": options[target],
            "year": rec.get("variables", {}).get("year", "?"),
            "choice": options.get(letter, letter),
            "is_target": int(letter == target),
        })
    return rows, health


def report(rows: list[dict], health: Counter) -> None:
    parsed = health["clean"] + health["lenient"]
    total = sum(health.values())
    print(f"\nExperiment 3 — forced choice (A/B/C/D)")
    print(f"  {total} records  ·  {parsed} scored  ·  {len(rows)} rows")
    if total:
        for reason in ("clean", "lenient", "enumeration", "unparsed", "empty", "no_options"):
            if health[reason]:
                print(f"    {reason:12s} {health[reason]:6d}  ({100 * health[reason] / total:5.1f}%)")

    by_model_slot: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for r in rows:
        by_model_slot[(r["model"], r["slot"])].append(r)

    for model in sorted({r["model"] for r in rows}):
        print(f"\n{'=' * 72}\n  {model}\n{'=' * 72}")
        slots = sorted(s for m, s in by_model_slot if m == model)
        for slot in slots:
            group = by_model_slot[(model, slot)]
            rate = sum(r["is_target"] for r in group) / len(group)
            target = group[0]["target"]
            print(f"\n  {slot:22s} target={target!r}  overall={rate:6.1%}  (n={len(group)})")

            per_year: dict[str, list[int]] = defaultdict(list)
            for r in group:
                per_year[r["year"]].append(r["is_target"])
            years = sorted(per_year)
            print("    year  " + " ".join(f"{y[-2:]:>5s}" for y in years))
            print("    rate  " + " ".join(
                f"{100 * sum(v) / len(v):4.0f}%" for v in (per_year[y] for y in years)))

            distractors = Counter(r["choice"] for r in group if not r["is_target"])
            if distractors:
                top = ", ".join(f"{w} x{n}" for w, n in distractors.most_common(3))
                print(f"    picked instead: {top}")


def write_csv(rows: list[dict], path: Path) -> None:
    """Tidy long format: one row per model x slot x year x chosen word."""
    counts: Counter = Counter()
    n_valid: Counter = Counter()
    meta: dict[tuple, str] = {}
    for r in rows:
        key = (r["model"], r["slot"], r["year"])
        counts[key + (r["choice"], r["is_target"])] += 1
        n_valid[key] += 1
        meta[key] = r["target"]

    with path.open("w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["model", "slot", "target", "year", "choice", "is_target", "count", "n_valid"])
        for (model, slot, year, choice, is_target), n in sorted(counts.items()):
            key = (model, slot, year)
            w.writerow([model, slot, meta[key], year, choice, is_target, n, n_valid[key]])
    print(f"\nWrote {path}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Score Experiment 3 forced-choice (A/B/C/D) responses.")
    parser.add_argument("responses", nargs="*", default=[str(DEFAULT_RESPONSES)],
                        help=f"Response file(s) or directory (default: {DEFAULT_RESPONSES}).")
    parser.add_argument("--slot", help="Filter to one slot (e.g. exp3_aura).")
    parser.add_argument("--model", help="Filter to models matching this substring.")
    parser.add_argument("--csv", type=Path, help="Also write scored rows to this CSV.")
    args = parser.parse_args()

    records: list[dict] = []
    for p in args.responses:
        if not Path(p).exists():
            sys.exit(f"Not found: {p}")
        records.extend(read_responses(p))
    if not records:
        sys.exit("No response records found.")

    rows, health = score(records)
    if args.slot:
        rows = [r for r in rows if r["slot"] == args.slot]
    if args.model:
        rows = [r for r in rows if args.model.lower() in r["model"].lower()]
    if not rows:
        sys.exit("No rows left after filtering.")

    report(rows, health)
    if args.csv:
        write_csv(rows, args.csv)
    print()


if __name__ == "__main__":
    main()
