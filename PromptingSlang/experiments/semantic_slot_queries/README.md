# Experiment 3 — Semantic Slot Queries

Prompt the models with a prompt designed to evoke a specific slang word.
Essentially measure the reluctance of the model to use the intended slang word.
This is also an easy way to detect words that may just be out of vocabulary for
the model.

## Contents

- `prompts/exp3_semantic_neighbor.jsonl` — semantic-slot prompts (e.g. "cool /
  impressive", "charisma", "suspicious") designed to evoke a specific slang term.
- `analysis/exp3_response_freq.py` — distribution of words produced per slot, via
  the generated word and the first-token alternatives.
- `visualizer/viz_exp3.py` — per-slot word-tile grid coloured by era, plus a
  per-era probability-mass summary strip.

Runs are executed from the PromptingSlang root; responses land in the shared
`data/responses/` directory.
