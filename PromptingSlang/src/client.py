"""LLM API clients for Together, OpenAI, and Anthropic."""

import importlib.util
import os
import re
from pathlib import Path
from typing import Any


def _load_keys():
    """Load the project's Keys.py (it lives at the repo root, two levels up)."""
    for base in (Path(__file__).resolve().parents[2], Path.cwd()):
        candidate = base / "Keys.py"
        if candidate.is_file():
            spec = importlib.util.spec_from_file_location("Keys", candidate)
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            return module
    return None


_Keys = _load_keys()


def _get_key(name: str) -> str | None:
    """Value of *name* from Keys.py, falling back to the environment."""
    from_keys = getattr(_Keys, name, None) if _Keys else None
    return from_keys or os.environ.get(name)


# ---------------------------------------------------------------------------
# Provider clients
# ---------------------------------------------------------------------------

# Per-request timeout (seconds). Without it a stalled connection hangs the whole
# run indefinitely; with it a hung request raises after this long, the SDK
# retries a couple of times, and the runner then logs+skips it and moves on.
REQUEST_TIMEOUT = 90.0


class TogetherClient:
    def __init__(self, api_key: str | None = None):
        try:
            from together import Together
        except ImportError:
            raise ImportError("pip install together")
        key = api_key or _get_key("TOGETHER_API_KEY")
        if not key:
            raise ValueError("TOGETHER_API_KEY not set.")
        self._client = Together(api_key=key, timeout=REQUEST_TIMEOUT)

    def complete(self, model: str, messages: list[dict], **gen_kwargs) -> dict:
        response = self._client.chat.completions.create(
            model=model, messages=messages, **gen_kwargs,
        )
        choice = response.choices[0]
        return {
            "text": choice.message.content,
            "model": response.model,
            "finish_reason": choice.finish_reason,
            "usage": {
                "prompt_tokens": response.usage.prompt_tokens,
                "completion_tokens": response.usage.completion_tokens,
                "total_tokens": response.usage.total_tokens,
            },
            "logprobs": getattr(choice, "logprobs", None),
        }


class OpenAIClient:
    def __init__(self, api_key: str | None = None):
        try:
            import openai as _openai
            self._openai = _openai
        except ImportError:
            raise ImportError("pip install openai")
        key = api_key or _get_key("OPENAI_API_KEY")
        if not key:
            raise ValueError("OPENAI_API_KEY not set.")
        self._client = _openai.OpenAI(api_key=key, timeout=REQUEST_TIMEOUT)

    def complete(self, model: str, messages: list[dict], **gen_kwargs) -> dict:
        # OpenAI uses logprobs=True + top_logprobs=N rather than a bare integer.
        n_logprobs = gen_kwargs.pop("logprobs", None)
        if n_logprobs is not None:
            gen_kwargs["logprobs"] = True
            gen_kwargs["top_logprobs"] = n_logprobs
        gen_kwargs.pop("echo", None)  # not supported by chat completions

        response = self._client.chat.completions.create(
            model=model, messages=messages, **gen_kwargs,
        )
        choice = response.choices[0]
        return {
            "text": choice.message.content,
            "model": response.model,
            "finish_reason": choice.finish_reason,
            "usage": {
                "prompt_tokens": response.usage.prompt_tokens,
                "completion_tokens": response.usage.completion_tokens,
                "total_tokens": response.usage.total_tokens,
            },
            "logprobs": choice.logprobs,
        }


class AnthropicClient:
    def __init__(self, api_key: str | None = None):
        try:
            import anthropic as _anthropic
            self._anthropic = _anthropic
        except ImportError:
            raise ImportError("pip install anthropic")
        key = api_key or _get_key("ANTHROPIC_API_KEY")
        if not key:
            raise ValueError("ANTHROPIC_API_KEY not set.")
        self._client = _anthropic.Anthropic(api_key=key, timeout=REQUEST_TIMEOUT)

    def complete(self, model: str, messages: list[dict], **gen_kwargs) -> dict:
        # Anthropic takes the system prompt as a top-level parameter.
        system = next((m["content"] for m in messages if m["role"] == "system"), None)
        chat = [m for m in messages if m["role"] != "system"]

        gen_kwargs.pop("logprobs", None)  # not available in standard API
        gen_kwargs.pop("echo", None)
        gen_kwargs.setdefault("max_tokens", 512)

        kwargs: dict[str, Any] = {"model": model, "messages": chat, **gen_kwargs}
        if system:
            kwargs["system"] = system

        response = self._client.messages.create(**kwargs)
        text = response.content[0].text if response.content else ""
        return {
            "text": text,
            "model": response.model,
            "finish_reason": response.stop_reason,
            "usage": {
                "prompt_tokens": response.usage.input_tokens,
                "completion_tokens": response.usage.output_tokens,
                "total_tokens": response.usage.input_tokens + response.usage.output_tokens,
            },
            "logprobs": None,
        }


# ---------------------------------------------------------------------------
# Router
# ---------------------------------------------------------------------------

def _infer_provider(model: str) -> str:
    """Infer the API provider from the model identifier.

    Together models use 'org/model' format.  OpenAI and Anthropic models use
    plain names (gpt-*, o1/o3/o4-*, claude-*).
    """
    if "/" in model:
        return "together"
    if model.startswith("claude-"):
        return "anthropic"
    if re.match(r"^(gpt-|o\d)", model):
        return "openai"
    return "together"


class RouterClient:
    """Dispatches each request to the correct provider based on model ID."""

    def __init__(
        self,
        together: TogetherClient | None = None,
        openai: OpenAIClient | None = None,
        anthropic: AnthropicClient | None = None,
    ):
        self._clients: dict[str, Any] = {}
        if together is not None:
            self._clients["together"] = together
        if openai is not None:
            self._clients["openai"] = openai
        if anthropic is not None:
            self._clients["anthropic"] = anthropic

    def complete(self, model: str, messages: list[dict], **gen_kwargs) -> dict:
        provider = _infer_provider(model)
        client = self._clients.get(provider)
        if client is None:
            raise ValueError(
                f"No client configured for provider '{provider}' (model: {model}). "
                f"Available: {list(self._clients)}"
            )
        return client.complete(model, messages, **gen_kwargs)
