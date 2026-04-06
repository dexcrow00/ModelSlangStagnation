"""Writes model responses to a JSONL file."""

from __future__ import annotations

import json
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


class ResponseCollector:
    """Context manager that appends one JSON record per line to *output_path*."""

    def __init__(self, output_path: str | Path):
        self.output_path = Path(output_path)
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        self._fh: IO[str] | None = None

    def __enter__(self) -> "ResponseCollector":
        self._fh = open(self.output_path, "a", encoding="utf-8")
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> None:
        if self._fh is not None:
            self._fh.close()
            self._fh = None

    def save(self, record: dict) -> None:
        """Append *record* as a pretty-printed JSON block and flush immediately."""
        if self._fh is None:
            raise RuntimeError("ResponseCollector must be used as a context manager.")
        self._fh.write(json.dumps(record, ensure_ascii=False, indent=2, cls=_ModelEncoder) + "\n\n")
        self._fh.flush()
