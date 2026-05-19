# CCAnalysis

Tools for collecting and analyzing slang word usage across Common Crawl WET dumps. Designed to run on AWS EC2 with S3 access (the CC bucket is public — no credentials needed).

## Setup

```bash
pip install -r requirements.txt
```

Dependencies: `warcio`, `transformers`, `torch`, `pyyaml`, `boto3`, `plotly`, `beautifulsoup4`, `lxml`.

---

## Scripts

### word_context.py — context extraction

Reservoir-samples URIs from CC WET files and extracts +-K token windows around every occurrence of a target word. Outputs one CSV per crawl dump.

```bash
# All crawls since 2019, 200 URIs per dump, 8 workers
python word_context.py \
    --file-list s3://commoncrawl/crawl-data/CC-MAIN-2024-10/wet.paths.gz \
    --targets target_words.txt \
    --since 2019 \
    --n-uris 200 \
    --output-dir contexts/ \
    --workers 8

# Specific crawl range
python word_context.py --targets target_words.txt \
    --from-crawl CC-MAIN-2020-05 --to-crawl CC-MAIN-2022-49 \
    --n-uris 100 --output-dir contexts/

# Single WET file
python word_context.py /data/segment.wet.gz --targets target_words.txt \
    --n-uris 50 --output contexts/CC-MAIN-2024-10.csv
```

Key flags:

| Flag | Default | Description |
|---|---|---|
| `--targets` | required | Plain-text file of target words, one per line |
| `--n-uris` | `200` | URIs to sample per dump |
| `--context-window` | `10` | Tokens of context on each side of the match |
| `--n-files` | `50` | WET files to sample per dump |
| `--workers` | `4` | Parallel workers |
| `--since` | — | Only process crawls from this year onward |
| `--from-crawl` / `--to-crawl` | — | Crawl ID range filter |
| `--force` | off | Re-collect even if output already exists |
| `--seed` | `42` | RNG seed for reproducibility |

Without `--force`, existing output files are checked for how many unique URIs they contain and are top-filled to reach `--n-uris` rather than re-run from scratch.

Output CSV columns: `uri`, `target_context` (context window with the matched word wrapped in `[WORD]` markers).

---

### bert_filter.py — BERT-based filtering

Filters `word_context.py` CSV output through three classifiers in cascade. Each later pass only scores rows that cleared the earlier ones.

```bash
# All three filters
python bert_filter.py contexts/*.csv -o filtered.csv --slang-defs slang.yaml

# GPU, stricter slang gate
python bert_filter.py contexts/*.csv -o filtered.csv \
    --slang-defs slang.yaml --slang-threshold 0.05 --device cuda

# Calibrate thresholds: write all rows with scores attached
python bert_filter.py input.csv -o scored.csv --score-all --keep-scores \
    --slang-defs slang.yaml

# Language + quality only (no slang definitions)
python bert_filter.py contexts/*.csv -o filtered.csv
```

**Classifier 1 — Language** (`papluca/xlm-roberta-base-language-detection`):
Drops rows where English-class confidence < `--lang-threshold` (default 0.85). Useful for pre-2021 dumps that lack the WARC language header.

**Classifier 2 — Writing quality** (`textattack/bert-base-uncased-CoLA`):
BERT fine-tuned on the Corpus of Linguistic Acceptability. Drops keyword strings, SEO spam, and nav menus that score below `--quality-threshold` (default 0.70).

**Classifier 3 — Slang sense** (`bert-base-uncased`):
Builds prototype embeddings from `slang.yaml` examples (one per word per sense), then compares each occurrence's contextual BERT embedding to the prototypes.
- Words with both slang + standard examples: `score = cos_sim(slang_proto) - cos_sim(standard_proto)` — positive means slang sense.
- Words with only slang examples (pure internet slang): `score = cos_sim(slang_proto)` — absolute similarity, 0–1.
Drops rows scoring below `--slang-threshold` (default 0.0). Rows containing no defined slang word are not filtered.

Key flags:

| Flag | Default | Description |
|---|---|---|
| `--slang-defs` | — | Path to `slang.yaml`; slang filter disabled if omitted |
| `--lang-threshold` | `0.85` | Min English confidence |
| `--quality-threshold` | `0.70` | Min CoLA acceptability score |
| `--slang-threshold` | `0.0` | Min slang score (meaning differs by mode — see above) |
| `--keep-scores` | off | Append score columns to output |
| `--score-all` | off | Write every row unfiltered (implies `--keep-scores`) |
| `--batch-size` | `32` | Inference batch size; increase for GPU |
| `--device` | auto | `cpu`, `cuda`, `cuda:N`, `mps` |
| `--no-lang-filter` | off | Skip language classifier |
| `--no-quality-filter` | off | Skip quality classifier |

---

### slang.yaml — sense prototype definitions

YAML file used by `bert_filter.py` to build word-sense prototype embeddings. Contains 5-6 example sentences per word per sense.

```yaml
fire:
  slang:
    - "That new track is straight fire, I have had it on repeat all week"
    - "His fit was fire, everyone turned to look when he walked in"
  standard:
    - "The fire spread quickly through the dry forest before crews arrived"
    - "She warmed her hands by the fire after coming in from the cold"
yolo:
  slang:
    - "Just booked a last-minute flight, yolo, life is too short"
  # no standard key -- word has no literal meaning outside slang
```

Words without a `standard` key are scored in slang-only mode (absolute cosine similarity).

---

### word_sample.py — word frequency sampling

Reservoir-samples individual words from CC WET files. Used as input to `word_variation.py`. Usually invoked via `run_all_crawls.py`.

```bash
python word_sample.py --file-list wet.paths.gz --n 100000 --seed 42
python word_sample.py /data/*.wet.gz --n 50000 --seed 42 --max-records 50
```

Output: `word_sample_CC-MAIN-YYYY-WW_<datetime>.csv` with a single `word` column.

---

### run_all_crawls.py — multi-crawl orchestration

Runs `word_sample.py` across every CC-MAIN-* dump on S3. Fetches the crawl list from the public CC collinfo API, checks that `wet.paths.gz` exists on S3, then invokes the child script once per crawl. Already-completed output files are skipped, making it safe to interrupt and resume.

```bash
python run_all_crawls.py --n 100000 --seed 42 --output-dir samples/
python run_all_crawls.py --n 100000 --seed 42 --since 2018 --output-dir samples/
python run_all_crawls.py --n 100000 --seed 42 --crawls CC-MAIN-2024-10 CC-MAIN-2023-40
python run_all_crawls.py --n 100000 --seed 42 --dry-run   # list crawls, no execution
```

---

### word_variation.py — frequency variation plots

Loads per-crawl word-sample CSVs, scores each word by `log2(max_rate / min_rate)` across crawls, and plots the top-N most variable words.

```bash
python word_variation.py samples/ --top 25 -o variation.html
python word_variation.py samples/ --top 30 --min-crawls 10 --bucket-by-year -o variation.png
python word_variation.py samples/ --words target_words.txt --log -o slang_rates.html
```

| Flag | Default | Description |
|---|---|---|
| `--top` | `20` | Words to plot |
| `--min-crawls` | `5` | Minimum crawls a word must appear in |
| `--min-avg-rate` | `2e-5` | Minimum average sampling rate (filters noise) |
| `--words` | — | Plot only these words (plain-text file); ignores `--top`/`--min-*` |
| `--bucket-by-year` | off | Average per-crawl values into yearly means before plotting |
| `--log` | off | Log-scale y-axis (sampling rate instead of count) |

---

### RawCounting/word_count.py — raw word counts

Counts all word/phrase occurrences in WET/WARC files. See `RawCounting/README.md` for usage.

---

## EC2 deployment

```bash
# Upload scripts
scp -i DexcrowLogin.pem word_context.py bert_filter.py slang.yaml target_words.txt \
    run_all_crawls.py word_sample.py ec2-user@<IPV4>:~/

# Download results
scp -i DexcrowLogin.pem -r ec2-user@<IPV4>:~/contexts/ ~/Documents/Projects/SlangShift/CCAnalysis/
```

The EC2 instance IAM role needs `s3:GetObject` on `s3://commoncrawl/*` (the bucket is public, so anonymous access works without any credentials).
