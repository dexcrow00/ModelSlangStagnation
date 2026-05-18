"""LLM API clients for Together, OpenAI, and Anthropic."""

import os
import re
from typing import Any

try:
    import Keys as _Keys
except ImportError:
    _Keys = None


def _get_key(env_var: str) -> str | None:
    from_keys = getattr(_Keys, env_var, None) if _Keys else None
    return from_keys or os.environ.get(env_var)


# ---------------------------------------------------------------------------
# Provider clients
# ---------------------------------------------------------------------------

class TogetherClient:
    def __init__(self, api_key: str | None = None):
        try:
            from together import Together
        except ImportError:
            raise ImportError("pip install together")
        key = api_key or _get_key("TOGETHER_API_KEY")
        if not key:
            raise ValueError("TOGETHER_API_KEY not set.")
        self._client = Together(api_key=key)

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
        self._client = _openai.OpenAI(api_key=key)

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
        self._client = _anthropic.Anthropic(api_key=key)

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
