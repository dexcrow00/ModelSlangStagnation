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

from src.client import TogetherClient
from src.collector import ResponseCollector
from src.prompts import load_prompts
from src.runner import Runner

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    datefmt="%H:%M:%S",
)

DEFAULT_MODELS = [
    "meta-llama/Llama-3.2-3B-Instruct-Turbo",
    "Qwen/Qwen2.5-7B-Instruct-Turbo",
    "deepseek-ai/DeepSeek-V4-Pro",
    #"openai/gpt-oss-120b",
    "google/gemma-4-31B-it",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prompt open-source LLMs via TogetherAI and collect responses."
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
        help="Together model IDs to query (space-separated). Defaults to a curated list.",
    )
    parser.add_argument(
        "--output",
        help=(
            "Output JSONL path. Defaults to data/responses/<run_id>.jsonl "
            "relative to repo root."
        ),
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="Sampling temperature (default: 0.8).",
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
    prompt_stem = Path(args.prompts).stem
    repo_root = Path(__file__).resolve().parents[1]

    prompts = load_prompts(args.prompts)
    if not prompts:
        print("No prompts found — check your JSONL file.", file=sys.stderr)
        sys.exit(1)

    using_logprobs = any(logprobs is not None for _, logprobs, _echo in prompts)
    if args.output:
        output_path = Path(args.output)
    elif using_logprobs:
        output_path = repo_root / "data" / "responses" / f"{prompt_stem}_logprobs_{run_id}.jsonl"
    else:
        output_path = repo_root / "data" / "responses" / f"{prompt_stem}_{run_id}.jsonl"

    n_variants = sum(len(template.expand()) for template, _lp, _echo in prompts)
    n_models = len(args.models)
    n_requests = n_models * n_variants

    print(f"  Prompt file : {args.prompts}")
    print(f"  Models      : {n_models}")
    print(f"  Variants    : {n_variants}  ({len(prompts)} template(s), expanded across variables)")
    print(f"  Total calls : {n_requests}")
    print(f"  Output      : {output_path}")
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

    client = TogetherClient()
    gen_kwargs = {"temperature": args.temperature, "max_tokens": args.max_tokens}

    with ResponseCollector(output_path) as collector:
        runner = Runner(
            client=client,
            collector=collector,
            models=args.models,
            gen_kwargs=gen_kwargs,
            run_id=run_id,
        )
        runner.run(prompts)

    print(f"\nDone. Responses written to: {output_path}")


if __name__ == "__main__":
    main()
