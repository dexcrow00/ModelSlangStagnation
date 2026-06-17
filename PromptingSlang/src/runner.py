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

from .client import RouterClient, TogetherClient, _infer_provider
from .collector import ResponseCollector
from .prompts import PromptTemplate

logger = logging.getLogger(__name__)

# Retry on Together rate-limit (429) or server errors (5xx).
_RETRYABLE_STATUS = {429, 500, 502, 503, 504}


def _is_retryable(exc: BaseException) -> bool:
    status = getattr(exc, "status_code", None) or getattr(exc, "status", None)
    return status in _RETRYABLE_STATUS


def model_class(model: str) -> str:
    """Classify a model as 'open' (Together-hosted) or 'closed' (OpenAI/Anthropic).

    A prompt's 'model_type' field is matched against this so open prompts (which
    request logprobs) run only on open models, and closed prompts run on the
    closed APIs where logprobs aren't available.
    """
    return "open" if _infer_provider(model) == "together" else "closed"


class Runner:
    def __init__(
        self,
        client: RouterClient | TogetherClient,
        collector: ResponseCollector,
        models: list[str],
        gen_kwargs: dict[str, Any] | None = None,
        run_id: str | None = None,
        closed_samples: int = 1,
        samples: int = 1,
    ):
        self.client = client
        self.collector = collector
        self.models = models
        self.gen_kwargs = gen_kwargs or {}
        self.run_id = run_id or uuid.uuid4().hex
        # Closed (non-logprob) prompts have no token distribution, so sample them
        # repeatedly to approximate one from response frequencies.
        self.closed_samples = max(1, closed_samples)
        # Global multiplier applied to every prompt's sample count (closed prompts
        # therefore get samples * closed_samples).
        self.samples = max(1, samples)

    def run(self, prompts: list[tuple[PromptTemplate, int | None, bool | None]]) -> None:
        """Iterate over every model × prompt expansion and collect responses.

        Each prompt is expanded into one variant per variable combination before
        the model loop, so list-valued variables produce separate requests.
        """
        # Expand prompts first so tqdm shows the true total request count.
        expanded = [
            (template.id, template.model_type, template.temperature,
             logprobs, echo, variables, system_text, user_text)
            for template, logprobs, echo in prompts
            for variables, system_text, user_text in template.expand()
        ]
        # Route by model_type: a prompt tagged 'open'/'closed' runs only on models
        # of that class; an untagged prompt (model_type=None) runs on every model.
        combos = []
        skipped = 0
        for model in self.models:
            mclass = model_class(model)
            for exp in expanded:
                prompt_model_type = exp[1]
                if prompt_model_type is not None and prompt_model_type != mclass:
                    skipped += 1
                    continue
                # Every prompt is sampled `samples` times; closed prompts get an
                # extra closed_samples factor (no logprobs -> approximate by counts).
                n = self.samples * (self.closed_samples if prompt_model_type == "closed" else 1)
                for sample_idx in range(n):
                    combos.append((model, sample_idx, *exp))
        logger.info(
            "Starting run %s — %d model(s), %d prompt variant(s) -> %d requests "
            "(%d skipped by model_type; samples x%d, closed prompts x%d more)",
            self.run_id, len(self.models), len(expanded), len(combos), skipped,
            self.samples, self.closed_samples,
        )

        for (model, sample_idx, prompt_id, _model_type, temperature, logprobs, echo,
             variables, system_text, user_text) in tqdm(combos, desc="Prompting", unit="req"):
            self._process(model, prompt_id, logprobs, echo, variables,
                          system_text, user_text, temperature, sample_idx)

    @retry(
        retry=retry_if_exception(_is_retryable),
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=30),
        reraise=True,
    )
    def _call(self, model: str, messages: list[dict], **extra_kwargs) -> dict:
        # extra_kwargs (e.g. a per-prompt temperature) override the run defaults.
        return self.client.complete(model, messages, **{**self.gen_kwargs, **extra_kwargs})

    def _process(
        self,
        model: str,
        prompt_id: str,
        logprobs: int | None,
        echo: bool | None,
        variables: dict,
        system_text: str,
        user_text: str,
        temperature: float | None = None,
        sample: int = 0,
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
        if temperature is not None:
            extra["temperature"] = temperature
        try:
            result = self._call(model, messages, **extra)
        except Exception as exc:
            logger.error("Failed model=%s prompt=%s: %s", model, prompt_id, exc)
            return

        record = {
            "run_id": self.run_id,
            "model": model,
            "prompt_id": prompt_id,
            "sample": sample,
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
