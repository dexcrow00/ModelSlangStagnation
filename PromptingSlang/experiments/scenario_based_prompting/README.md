# Experiment 1 — Scenario Based Prompting

This set of experiments uses a prompt to try to get the model to respond in a
colloquial manner that may leverage slang usage in its response. This should be
the more "natural" way of sampling vocabulary from the models, but may be
susceptible to all the usual prompt related variance.

## Contents

- `prompts/exp1_scenario_generation.jsonl` — scenario prompts (text message, Reddit
  comment, group chat, reaction post) fed to the models via `scripts/run.py`.
- `analysis/exp1_word_freq.py` — counts target slang word frequencies in the free
  generated responses, grouped by era of peak usage.
- `visualizer/viz_exp1.py` — renders the slang-frequency figure (per-word bars by
  era + per-scenario breakdown).

Runs are executed from the PromptingSlang root; responses land in the shared
`data/responses/` directory.
