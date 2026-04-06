"""Prompt template definitions and loaders."""

from __future__ import annotations

import itertools
import json
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class PromptTemplate:
    id: str
    system: str
    user: str
    variables: dict = field(default_factory=dict)

    def expand(self) -> list[tuple[dict, str, str]]:
        """Return one (variables_used, system, user) tuple per variable combination.

        Variable values that are lists are expanded via cartesian product so that
        each combination produces a separate rendered prompt. Scalar values are
        shared across all combinations unchanged.

        Example — given variables {"year": [2020, 2022], "register": "casual"}
        this yields two tuples: one with year=2020 and one with year=2022.
        """
        list_vars = {k: v for k, v in self.variables.items() if isinstance(v, list)}
        scalar_vars = {k: v for k, v in self.variables.items() if not isinstance(v, list)}

        if not list_vars:
            ctx = scalar_vars
            return [(ctx, self.system.format_map(ctx), self.user.format_map(ctx))]

        keys = list(list_vars.keys())
        results = []
        for combo in itertools.product(*[list_vars[k] for k in keys]):
            ctx = {**scalar_vars, **dict(zip(keys, combo))}
            results.append((ctx, self.system.format_map(ctx), self.user.format_map(ctx)))
        return results


def load_prompts(path: str | Path) -> list[tuple[PromptTemplate, int | None, bool | None]]:
    """Load prompt templates from a JSONL file.

    Supports both compact (one object per line) and pretty-printed (multi-line)
    formats. Each record must have at least 'id', 'system', 'user'.

    Returns a list of (template, logprobs, echo) tuples. logprobs and echo are
    read from the corresponding JSON fields, or None if not specified.
    """
    results: list[tuple[PromptTemplate, int | None, bool | None]] = []
    decoder = json.JSONDecoder()
    text = Path(path).read_text(encoding="utf-8")
    pos = 0
    while pos < len(text):
        # Skip whitespace (including blank lines between records).
        while pos < len(text) and text[pos] in " \t\n\r":
            pos += 1
        if pos >= len(text):
            break
        try:
            obj, end = decoder.raw_decode(text, pos)
        except json.JSONDecodeError as exc:
            raise ValueError(f"Invalid JSON in {path} at position {pos}: {exc}") from exc
        template = PromptTemplate(
            id=obj["id"],
            system=obj["system"],
            user=obj["user"],
            variables=obj.get("variables", {}),
        )
        results.append((template, obj.get("logprobs"), obj.get("echo")))
        pos = end
    return results
