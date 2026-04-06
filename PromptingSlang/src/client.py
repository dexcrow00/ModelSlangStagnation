"""TogetherAI API wrapper."""

import os
from together import Together
import Keys


class TogetherClient:
    def __init__(self, api_key: str | None = None):
        key = api_key or Keys.TOGETHER_API_KEY or os.environ.get("TOGETHER_API_KEY")
        if not key:
            raise ValueError("TOGETHER_API_KEY not set in environment or passed explicitly.")
        self._client = Together(api_key=key)

    def complete(self, model: str, messages: list[dict], **gen_kwargs) -> dict:
        """Call chat completions and return the full response as a dict.

        Args:
            model: Together model ID (e.g. 'meta-llama/Llama-3-8b-chat-hf').
            messages: List of {'role': ..., 'content': ...} dicts.
            **gen_kwargs: Generation params forwarded to the API
                          (temperature, max_tokens, top_p, etc.).

        Returns:
            Dict with keys: 'text', 'model', 'usage', 'finish_reason'.
        """
        response = self._client.chat.completions.create(
            model=model,
            messages=messages,
            **gen_kwargs,
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
