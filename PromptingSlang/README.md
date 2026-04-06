# PromptingSlang

A framework for systematically prompting open-source LLMs via [TogetherAI](https://www.together.ai/) to produce slang and colloquial language responses. Part of the SlangShift project.

## Setup

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Configure your API key

The client looks for your TogetherAI key in two places, in order:

1. A `Keys.py` file in the project root (a module-level `TOGETHER_API_KEY` variable)
2. The `TOGETHER_API_KEY` environment variable (loaded automatically from a `.env` file if present)

**Option A — `Keys.py` (recommended for local dev):**

Create `Keys.py` in the project root:

```python
TOGETHER_API_KEY = "your_together_api_key_here"
```

**Option B — `.env` file:**

```bash
cp .env.example .env
# then edit .env and fill in your key
```

---

## Running a collection

All commands are run from the **project root**.

### Basic run (default model, example prompts)

```bash
python scripts/run.py --prompts data/prompts/example.jsonl
```

This uses the default model (`ServiceNow-AI/Apriel-1.6-15b-Thinker`) and writes output to `data/responses/<run_id>.jsonl`.

### Specify one or more models

```bash
python scripts/run.py \
  --prompts data/prompts/example.jsonl \
  --models meta-llama/Llama-3.3-70B-Instruct-Turbo mistralai/Mistral-7B-Instruct-v0.3
```

### Control generation parameters

```bash
python scripts/run.py \
  --prompts data/prompts/example.jsonl \
  --temperature 1.0 \
  --max-tokens 256
```

| Flag | Default | Description |
|---|---|---|
| `--temperature` | `0.8` | Sampling temperature |
| `--max-tokens` | `512` | Max tokens generated per response |

### Set a custom output path

```bash
python scripts/run.py \
  --prompts data/prompts/example.jsonl \
  --output data/responses/my_run.jsonl
```

### Set a fixed run ID (for reproducibility / resuming)

```bash
python scripts/run.py \
  --prompts data/prompts/example.jsonl \
  --run-id slang_batch_01
```

---

## Prompt file format

Prompt files are JSONL — one JSON object per record, separated by blank lines. Each record requires `id`, `system`, and `user` fields.

| Field | Required | Description |
|---|---|---|
| `id` | Yes | Unique identifier for the template |
| `system` | Yes | System prompt text |
| `user` | Yes | User prompt text; supports `{placeholder}` interpolation |
| `variables` | No | Default values for placeholders; list values expand into separate requests |
| `logprobs` | No | Integer — requests token log probabilities from the API instead of a text response |

```json
{
  "id": "unique_prompt_id",
  "system": "You are a young adult who uses a lot of internet slang...",
  "user": "Hey, what do you think about {topic}? Give me your honest take.",
  "variables": {
    "topic": ["AI taking over creative jobs", "social media addiction"]
  },
  "logprobs": 10
}
```

**Variable expansion:** if a variable's value is a list, one request is made per value. Multiple list variables are combined via cartesian product.

**logprobs:** when set, the API returns token-level log probabilities instead of a plain text response. The output filename gains a `_logprobs_` infix: `<prompt>_logprobs_<run_id>.jsonl`.

Place prompt files in `data/prompts/`. Example files are provided at `data/prompts/example.jsonl` and `data/prompts/single_word.jsonl`.

---

## Output format

Each run produces a JSONL file at `data/responses/<prompt>_<run_id>.jsonl` (or `<prompt>_logprobs_<run_id>.jsonl` when logprobs are requested). One record is written per model × prompt variant:

**Text response record:**

```json
{
  "run_id": "a3f9c1...",
  "model": "ServiceNow-AI/Apriel-1.6-15b-Thinker",
  "prompt_id": "slang_casual_convo",
  "variables": {"topic": "AI taking over creative jobs"},
  "prompt_text": "Hey, what do you think about AI taking over creative jobs?",
  "system_text": "You are a young adult who uses a lot of internet slang...",
  "response": "ok so honestly...",
  "logprobs": null,
  "finish_reason": "stop",
  "usage": {"prompt_tokens": 74, "completion_tokens": 183, "total_tokens": 257},
  "timestamp": "2026-02-28T10:34:12.001Z"
}
```

**Logprobs record** (when `"logprobs"` is set in the prompt file):

```json
{
  "run_id": "b7d2e4...",
  "model": "ServiceNow-AI/Apriel-1.6-15b-Thinker",
  "prompt_id": "GiveTopSlang",
  "variables": {"year": "2021"},
  "prompt_text": "What is the most unique and popular slang word associated with the year 2021?",
  "system_text": "Give the response as a single returned token or word.",
  "response": null,
  "logprobs": {
    "content": [
      {
        "token": "no",
        "logprob": -0.412,
        "top_logprobs": [
          {"token": "no", "logprob": -0.412},
          {"token": "on", "logprob": -1.204},
          {"token": "the", "logprob": -2.913}
        ]
      }
    ]
  },
  "finish_reason": "stop",
  "usage": {"prompt_tokens": 31, "completion_tokens": 2, "total_tokens": 33},
  "timestamp": "2026-02-28T10:35:01.220Z"
}
```

---

## Visualizing logprob results

The `visualizer/visualize.py` script renders a logprob response file as a heatmap matrix. Each row is one record (labeled with model and variable values), each column is a token position, and cell colour encodes the log probability (red = low, green = high).

### Display interactively

```bash
python visualizer/visualize.py data/responses/single_word_logprobs_<run_id>.jsonl
```

### Save to an image file

```bash
python visualizer/visualize.py data/responses/single_word_logprobs_<run_id>.jsonl -o out.png
```

PNG, PDF, and SVG are all supported — the format is inferred from the file extension.

---

## Project structure

```
PromptingSlang/
├── src/
│   ├── client.py      # TogetherAI API wrapper
│   ├── prompts.py     # PromptTemplate dataclass + JSONL loader
│   ├── collector.py   # Writes responses to JSONL
│   └── runner.py      # Orchestrates model × prompt loop with retries
├── data/
│   ├── prompts/       # Input prompt JSONL files
│   └── responses/     # Output — one file per run
├── scripts/
│   └── run.py         # CLI entry point
├── visualizer/
│   └── visualize.py   # Logprob heatmap renderer
├── requirements.txt
└── .env.example
```
