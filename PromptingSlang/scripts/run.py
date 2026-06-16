#!/usr/bin/env python3
"""CLI entry point for PromptingSlang."""

from __future__ import annotations

import argparse
import logging
import os
import sys
import uuid
from pathlib import Path

# Allow running from repo root without installing the package.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dotenv import load_dotenv

from src.client import AnthropicClient, OpenAIClient, RouterClient, TogetherClient
from src.collector import ResponseCollector
from src.prompts import load_prompts
from src.runner import Runner, model_class

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    datefmt="%H:%M:%S",
)

# Default models span three providers; RouterClient picks the API per model ID
# ('org/model' -> Together, 'claude-*' -> Anthropic, 'gpt-*'/'o#-*' -> OpenAI).
DEFAULT_MODELS = [
    # Together (org/model format)
    "meta-llama/Llama-3.3-70B-Instruct-Turbo",
    "Qwen/Qwen3.7-Max",
    "deepseek-ai/DeepSeek-V4-Pro",
    "google/gemma-4-31B-it",
    "moonshotai/Kimi-K2.6",
    # Anthropic (Claude)
    "claude-sonnet-4-6",
    # OpenAI (GPT)
    "gpt-4o",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prompt LLMs via Together, OpenAI, or Anthropic and collect responses."
    )
    parser.add_argument(
        "--prompts",
        default="data/prompts/example.jsonl",
        help="Path to a JSONL file of prompt templates.",
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=DEFAULT_MODELS,
        help=(
            "Model IDs to query (space-separated). Together models use 'org/model' format. "
            "OpenAI models use 'gpt-*'/'o*' names. Anthropic models use 'claude-*' names. "
            "The correct API is chosen automatically per model."
        ),
    )
    parser.add_argument(
        "--output-dir",
        dest="output_dir",
        help=(
            "Directory for the per-model response files (one <model>.jsonl each). "
            "Defaults to a 'responses' directory next to the prompt file's directory."
        ),
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="Sampling temperature (default: 0.0).",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=512,
        dest="max_tokens",
        help="Maximum tokens to generate per response (default: 512).",
    )
    parser.add_argument(
        "--run-id",
        default=None,
        dest="run_id",
        help="Explicit run identifier; auto-generated if omitted.",
    )
    return parser.parse_args()


def main() -> None:
    load_dotenv()
    args = parse_args()

    run_id = args.run_id or uuid.uuid4().hex

    prompts = load_prompts(args.prompts)
    if not prompts:
        print("No prompts found — check your JSONL file.", file=sys.stderr)
        sys.exit(1)

    # Per-model response files land in a 'responses' directory next to the
    # directory holding the prompt file (e.g. <experiment>/prompts/foo.jsonl ->
    # <experiment>/responses/), unless overridden with --output-dir.
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        output_dir = Path(args.prompts).resolve().parent.parent / "responses"

    n_variants = sum(len(template.expand()) for template, _lp, _echo in prompts)
    n_models = len(args.models)
    # Requests honor model_type routing: a tagged prompt only runs on its model class.
    n_requests = sum(
        len(template.expand())
        for model in args.models
        for template, _lp, _echo in prompts
        if template.model_type is None or template.model_type == model_class(model)
    )

    print(f"  Prompt file : {args.prompts}")
    print(f"  Models      : {n_models}")
    print(f"  Variants    : {n_variants}  ({len(prompts)} template(s), expanded across variables)")
    print(f"  Total calls : {n_requests}")
    print(f"  Output dir  : {output_dir}  (one <model>_<timestamp>.jsonl per model)")
    print()
    try:
        confirm = input("Proceed? [y/N] ").strip().lower()
    except (EOFError, KeyboardInterrupt):
        print("\nAborted.")
        sys.exit(0)
    if confirm != "y":
        print("Aborted.")
        sys.exit(0)
    print()

    # Build whichever provider clients have keys configured.
    def _try(cls, **kw):
        try:
            return cls(**kw)
        except (ValueError, ImportError):
            return None

    router = RouterClient(
        together=_try(TogetherClient),
        openai=_try(OpenAIClient),
        anthropic=_try(AnthropicClient),
    )
    if not router._clients:
        print(
            "No API keys found. Set TOGETHER_API_KEY, OPENAI_API_KEY, or ANTHROPIC_API_KEY.",
            file=sys.stderr,
        )
        sys.exit(1)

    gen_kwargs = {"temperature": args.temperature, "max_tokens": args.max_tokens}

    with ResponseCollector(output_dir) as collector:
        runner = Runner(
            client=router,
            collector=collector,
            models=args.models,
            gen_kwargs=gen_kwargs,
            run_id=run_id,
        )
        runner.run(prompts)

    print(f"\nDone. Responses written to: {output_dir}  (one timestamped file per model)")


if __name__ == "__main__":
    main()
