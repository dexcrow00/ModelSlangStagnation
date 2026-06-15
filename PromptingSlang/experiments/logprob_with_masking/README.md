# Experiment 2 — Logprob with Masking

Directly measure the logprob (this is only possible on some open models). Using a
sentence with the slang word masked, check and see how likely the model thinks the
slang word is to go into the masked slot. A very direct measurement, but may not
capture other factors of output conditioning, especially for closed models.

## Contents

- `prompts/exp2_logprob_probing.jsonl` — fill-in-blank syntactic frames (e.g.
  "That was absolutely ___") requesting token logprobs.
- `analysis/exp2_logprob_compare.py` — reconstructs the generated word + its joint
  logprob and the top first-token alternatives across frames.
- `visualizer/viz_exp2.py` — per-model bar charts of generated-word probability,
  coloured by slang era, with a first-token alternatives heatmap.

Runs are executed from the PromptingSlang root; responses land in the shared
`data/responses/` directory.
