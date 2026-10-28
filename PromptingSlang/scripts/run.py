#!/usr/bin/env python3
"""CLI entry point for PromptingSlang."""

from __future__ import annotations

import argparse
import logging
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

# Models every experiment runs against. Edit this list (not a CLI flag) to change
# them, so the model set is tracked in git. RouterClient picks the API per model ID
# ('org/model' -> Together, 'claude-*' -> Anthropic, 'gpt-*'/'o#-*' -> OpenAI).
MODELS = [
    # Together (org/model format)
    "moonshotai/Kimi-K3",
    "meta-models/Muse-Spark-Open-816B-A42B",
    "deepseek-ai/DeepSeek-V4.1-Flash",
    # Anthropic (Claude)
    "claude-sonnet-5-5",
    # OpenAI (GPT)
    "gpt-6-luna",
]

# Generation settings for every run. A prompt record's own "temperature" field
# still overrides TEMPERATURE for that prompt.
TEMPERATURE = 0.0
MAX_TOKENS = 512

# Directory for the per-model response files (one <model>.jsonl each). None puts
# them in a 'responses' directory next to the directory holding the prompt file
# (e.g. <experiment>/prompts/foo.jsonl -> <experiment>/responses/).
OUTPUT_DIR: Path | None = None


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
        "--concurrency",
        type=int,
        default=1,
        help=(
            "Requests in flight at once (default: 1, strictly serial). The calls are "
            "HTTP waits, so this scales throughput almost linearly until the provider "
            "starts rate-limiting; 8-16 is a sane range for a dedicated endpoint."
        ),
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help=(
            "Skip (model, prompt, variables, sample) combos already saved in the "
            "output directory, so an interrupted run continues where it left off "
            "instead of re-querying everything."
        ),
    )
    return parser.parse_args()


def main() -> None:
    load_dotenv()
    args = parse_args()

    run_id = uuid.uuid4().hex

    prompts = load_prompts(args.prompts)
    if not prompts:
        print("No prompts found — check your JSONL file.", file=sys.stderr)
        sys.exit(1)

    output_dir = OUTPUT_DIR or Path(args.prompts).resolve().parent.parent / "responses"

    n_variants = sum(len(template.expand()) for template, _lp, _echo in prompts)
    # Requests honor model_type routing (a tagged prompt only runs on its model
    # class) and each record's "samples" count.
    n_requests = sum(
        len(template.expand()) * template.samples
        for model in MODELS
        for template, _lp, _echo in prompts
        if template.model_type is None or template.model_type == model_class(model)
    )

    print(f"  Prompt file    : {args.prompts}")
    print(f"  Models         : {len(MODELS)}  ({', '.join(MODELS)})")
    print(f"  Variants       : {n_variants}  ({len(prompts)} template(s), expanded across variables)")
    sample_counts = sorted({template.samples for template, _lp, _echo in prompts})
    print(f"  Samples        : {', '.join(map(str, sample_counts))}  (per prompt record)")
    print(f"  Generation     : temperature {TEMPERATURE}, max tokens {MAX_TOKENS}")
    print(f"  Total calls    : {n_requests}")
    print(f"  Concurrency    : {args.concurrency}")
    print(f"  Output dir     : {output_dir}  (one <model>_<timestamp>.jsonl per model)")
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
    if not router.providers:
        print(
            "No API keys found. Set TOGETHER_API_KEY, OPENAI_API_KEY, or ANTHROPIC_API_KEY.",
            file=sys.stderr,
        )
        sys.exit(1)

    gen_kwargs = {"temperature": TEMPERATURE, "max_tokens": MAX_TOKENS}

    with ResponseCollector(output_dir) as collector:
        runner = Runner(
            client=router,
            collector=collector,
            models=MODELS,
            gen_kwargs=gen_kwargs,
            run_id=run_id,
            resume=args.resume,
            concurrency=args.concurrency,
        )
        runner.run(prompts)

    print(f"\nDone. Responses written to: {output_dir}  (one timestamped file per model)")


if __name__ == "__main__":
    main()
