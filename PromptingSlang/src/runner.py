"""Orchestrates model × prompt combinations with retry logic."""

from __future__ import annotations

import logging
import uuid
from datetime import datetime, timezone
from typing import Any

from tenacity import (
    retry,
    retry_if_exception,
    stop_after_attempt,
    wait_exponential,
)
from tqdm import tqdm

from .client import RouterClient, TogetherClient
from .collector import ResponseCollector
from .prompts import PromptTemplate

logger = logging.getLogger(__name__)

# Retry on Together rate-limit (429) or server errors (5xx).
_RETRYABLE_STATUS = {429, 500, 502, 503, 504}


def _is_retryable(exc: BaseException) -> bool:
    status = getattr(exc, "status_code", None) or getattr(exc, "status", None)
    return status in _RETRYABLE_STATUS


class Runner:
    def __init__(
        self,
        client: RouterClient | TogetherClient,
        collector: ResponseCollector,
        models: list[str],
        gen_kwargs: dict[str, Any] | None = None,
        run_id: str | None = None,
    ):
        self.client = client
        self.collector = collector
        self.models = models
        self.gen_kwargs = gen_kwargs or {}
        self.run_id = run_id or uuid.uuid4().hex

    def run(self, prompts: list[tuple[PromptTemplate, int | None, bool | None]]) -> None:
        """Iterate over every model × prompt expansion and collect responses.

        Each prompt is expanded into one variant per variable combination before
        the model loop, so list-valued variables produce separate requests.
        """
        # Expand prompts first so tqdm shows the true total request count.
        expanded = [
            (template.id, logprobs, echo, variables, system_text, user_text)
            for template, logprobs, echo in prompts
            for variables, system_text, user_text in template.expand()
        ]
        combos = [(model, *exp) for model in self.models for exp in expanded]
        logger.info(
            "Starting run %s — %d model(s) × %d prompt variant(s) = %d requests",
            self.run_id,
            len(self.models),
            len(expanded),
            len(combos),
        )

        for model, prompt_id, logprobs, echo, variables, system_text, user_text in tqdm(combos, desc="Prompting", unit="req"):
            self._process(model, prompt_id, logprobs, echo, variables, system_text, user_text)

    @retry(
        retry=retry_if_exception(_is_retryable),
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=30),
        reraise=True,
    )
    def _call(self, model: str, messages: list[dict], **extra_kwargs) -> dict:
        return self.client.complete(model, messages, **self.gen_kwargs, **extra_kwargs)

    def _process(
        self,
        model: str,
        prompt_id: str,
        logprobs: int | None,
        echo: bool | None,
        variables: dict,
        system_text: str,
        user_text: str,
    ) -> None:
        messages = [
            {"role": "system", "content": system_text},
            {"role": "user", "content": user_text},
        ]
        extra = {}
        if logprobs is not None:
            extra["logprobs"] = logprobs
        if echo is not None:
            extra["echo"] = echo
        try:
            result = self._call(model, messages, **extra)
        except Exception as exc:
            logger.error("Failed model=%s prompt=%s: %s", model, prompt_id, exc)
            return

        record = {
            "run_id": self.run_id,
            "model": model,
            "prompt_id": prompt_id,
            "variables": variables,
            "prompt_text": user_text,
            "system_text": system_text,
            "response": result["text"] if logprobs is None else None,
            "logprobs": result.get("logprobs") if logprobs is not None else None,
            "finish_reason": result.get("finish_reason"),
            "usage": result.get("usage"),
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
        self.collector.save(record)
