"""Writes model responses to per-model JSONL files in an output directory."""

from __future__ import annotations

import json
import re
from datetime import datetime, timezone
from pathlib import Path
from types import TracebackType
from typing import IO


class _ModelEncoder(json.JSONEncoder):
    """Extends the default encoder to handle Pydantic model objects returned by
    the Together SDK (e.g. the logprobs field), converting them via model_dump()."""

    def default(self, obj):
        if hasattr(obj, "model_dump"):
            return obj.model_dump()
        return super().default(obj)


def _safe_filename(model: str) -> str:
    """Turn a model id (may contain '/', ':', etc.) into a safe filename stem."""
    return re.sub(r"[^A-Za-z0-9._-]+", "_", model).strip("_") or "model"


class ResponseCollector:
    """Context manager that appends response records to per-model JSONL files.

    Each record is routed by its ``model`` field to
    ``<output_dir>/<model>_<timestamp>.jsonl`` (the model id sanitised into a safe
    filename); file handles are opened lazily on first write per model. The
    timestamp is fixed for the lifetime of the collector (one per run), so each
    run writes a fresh set of files rather than appending to or overwriting prior
    runs.
    """

    def __init__(self, output_dir: str | Path, timestamp: str | None = None):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.timestamp = timestamp or datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        self._handles: dict[str, IO[str]] = {}
        self._active = False

    def __enter__(self) -> "ResponseCollector":
        self._active = True
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> None:
        for fh in self._handles.values():
            fh.close()
        self._handles.clear()
        self._active = False

    def path_for(self, model: str) -> Path:
        """Output file path for *model* (includes the collector's run timestamp)."""
        return self.output_dir / f"{_safe_filename(model)}_{self.timestamp}.jsonl"

    def save(self, record: dict) -> None:
        """Append *record* as a pretty-printed JSON block to its model's file."""
        if not self._active:
            raise RuntimeError("ResponseCollector must be used as a context manager.")
        model = record.get("model") or "unknown"
        fh = self._handles.get(model)
        if fh is None:
            fh = open(self.path_for(model), "a", encoding="utf-8")
            self._handles[model] = fh
        fh.write(json.dumps(record, ensure_ascii=False, indent=2, cls=_ModelEncoder) + "\n\n")
        fh.flush()
