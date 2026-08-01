# CommonCrawlAnalysis

Project code for the LLM slang stagnation paper ("LLMs Can't Keep Up with Language
That's on Fleek"). The pipeline has two halves: measure when each slang term
actually peaked on the web, then measure which slang the models reach for.

## Directories

**`FineWebAnalysis/`** — the corpus side (Experiment 1), and the active half of
the project. Pulls target-word contexts out of the FineWeb 10BT sample
(`fineweb_context.py`), scores each occurrence for slang vs. standard sense with a
fine-tuned RoBERTa classifier (`finetune_roberta.py`, `roberta_filter.py`,
`eval_scenario_model.py`), and turns the survivors into per-dump frequency curves
and peak years (`word_rate_plotter.py`, `peak_year.py` → `peak_years.json`).
`peak_years.json` is the corpus ground truth every downstream analysis reads.

**`PromptingSlang/`** — the model side (Experiments 2–4). `src/` holds the
provider-routing client, runner, and shared response/analysis helpers;
`experiments/<name>/` each hold prompts, responses, analysis, and visualizers for
one experiment; `stats/` holds the paper's statistics (`effective_vintage.py`,
`temporal_mixture.py`, `nb_regression.py`), whose committed outputs live in
`stats/results/`.

**`DataProcessingTools/`** — annotation sets used to fine-tune and validate the
sense classifier, plus `auc_eval.py` and the synthetic-annotation generator. The
FineWeb-era successors to these scripts now live in `FineWebAnalysis/`.

**`DatasetAnalysis/`** — one-off notebooks from the initial survey of candidate
corpora (C4, FineWeb, FineWeb-deduplicated, Reddit comments). Kept for the record;
not part of the pipeline.

**`CCAnalysis/`** — the original raw Common Crawl (WET/S3) pipeline, superseded by
`FineWebAnalysis/`. Nothing in the current pipeline imports it.

**`writing/`** — the LaTeX paper, its figures, and `references.bib`.

## Reproducing

Most scripts are standalone and self-documenting via `--help`. The usual order is:

```bash
python FineWebAnalysis/fineweb_context.py --output-dir contexts/ --words FineWebAnalysis/target_words.txt
python FineWebAnalysis/roberta_filter.py contexts/*.parquet -o prompt_scored/ --score-all
python FineWebAnalysis/peak_year.py
python FineWebAnalysis/word_rate_plotter.py --threshold 0.99 --highlight-bands
```

Then the model side, per experiment:

```bash
python PromptingSlang/scripts/run.py --prompts PromptingSlang/experiments/<name>/prompts/<file>.jsonl
```

Note that scripts touching the corpus expect a `Keys.py` at the repo root (and in
`FineWebAnalysis/`) supplying `HF_TOKEN` and the model API keys; both are
gitignored, as are the large model directories and raw data.

## Target words

`FineWebAnalysis/target_words.txt` (and `scenario_words.txt`) list slang manually
curated from Urban Dictionary, Word-of-the-Year lists, and Google Search Trends.
Computational and linguistic limits mean we operate on a subset of internet slang;
curation aimed at a mix of millennial (2010–2020) and Gen-Z (2020+) terms, treating
slang as an indicator of in-group belonging to a generation (Citera 2020,
"Differences in emotional word use across generations in the United States"; Earl
1972, "Semantic Influence and Concept Attainment of Slang..."; Barbieri 2008,
"Patterns of age-based linguistic variation in American English"). Curation also
avoided hate speech and highly offensive terms — though a good deal of internet
slang is mildly offensive regardless.
