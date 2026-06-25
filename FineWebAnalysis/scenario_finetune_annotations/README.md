# Scenario-word fine-tuning annotations (UNLABELED)

Annotation-ready contexts for the RoBERTa slang-sense classifier, sampled
from `10BT_scenario_merged` by `prepare_finetune_data.py`. One CSV per
target word; columns: `target, is_slang, target_context, uri`.

## To use
1. Fill the `is_slang` column in each CSV: `1` if the target word is used
   in its slang sense in that context, `0` if it is the ordinary sense
   (e.g. `extra` = dramatic -> 1, `extra` = additional -> 0). Leave a row
   blank to skip it.
2. Fine-tune:
   `python finetune_roberta.py --annotations scenario_finetune_annotations \
       --model-dir ./ft_model_roberta_scenario/`

## Contents (26 words, 2640 contexts, up to 120 each)

| word | contexts |
| --- | --- |
| fresh | 120 |
| u | 120 |
| cancelled | 120 |
| extra | 120 |
| goals | 120 |
| era | 120 |
| squad | 120 |
| pumped | 120 |
| legit | 120 |
| tight | 120 |
| woke | 120 |
| shaking | 120 |
| clout | 120 |
| omg | 120 |
| dude | 120 |
| rent free | 120 |
| lmfao | 120 |
| vibing | 120 |
| 💀 | 120 |
| situationship | 120 |
| hits different | 72 |
| lmaooo | 62 |
| omgggg | 48 |
| lmaooooo | 28 |
| lmaoooo | 28 |
| left no crumbs | 2 |
