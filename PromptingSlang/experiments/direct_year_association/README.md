# Experiment 4 — Direct Year Association (bonus)

**Placeholder — no code yet.** Drop the analysis/visualizer code into the
`analysis/` and `visualizer/` subfolders here when it's written.

A look at meta-level awareness. This is maybe just more of an interest piece. If
we ask a model when a given slang word was most popular, does it line up with its
occurrence in training data? We'd expect it may not in some cases, since the model
is learning from content in its data, not from the structure of the data itself.
Could be a good way of highlighting some of the larger problems (lack of
self-awareness) this paper is hinting at.

## Related prompts (in shared `data/prompts/`)

The year/word-of-the-year prompt sets that feed this experiment currently live in
the shared `data/prompts/` directory rather than here:

- `dated_prompts.jsonl` — "list some popular slang from {year}" / topic prompts.
- `single_word.jsonl` — "most popular slang associated with {year}? single word".
- `single_word_echo.jsonl` — echo + logprobs variant scoring a candidate word per year.
- `woty_echo.jsonl` — Oxford word-of-the-year echo/logprob probes by year.
- `dictionary_com_woty_echo.jsonl` — same, using the Dictionary.com word-of-the-year list.
