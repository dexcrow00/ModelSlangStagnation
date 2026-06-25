# Scenario-word held-out evaluation set (UNLABELED)

Held-out contexts for evaluating the RoBERTa slang-sense classifier on the
new scenario words, sampled from `10BT_scenario_merged` by
`prepare_finetune_data.py` and kept **disjoint** from the training set
(scenario_finetune_annotations). One CSV per target word; columns:
`target, is_slang, target_context, uri`.

## To use
1. Fill the `is_slang` column: `1` if the target word is used in its slang
   sense in that context, `0` if it is the ordinary sense. Leave blank to skip.
2. Score with the fine-tuned model and compute metrics:
   `python roberta_filter.py <this-dir> --score-all \
       --output-dir scored_eval/ --roberta-model-dir ./ft_model_roberta_scenario/`
   then treat `roberta_score >= 0` as predicted-slang and compare against
   `is_slang` for precision / recall / F1 (per word and overall).

## Contents (15 words, 600 contexts, up to 40 each)

| word | contexts |
| --- | --- |
| fresh | 40 |
| u | 40 |
| cancelled | 40 |
| extra | 40 |
| goals | 40 |
| era | 40 |
| squad | 40 |
| pumped | 40 |
| tight | 40 |
| woke | 40 |
| shaking | 40 |
| clout | 40 |
| dude | 40 |
| rent free | 40 |
| vibing | 40 |
