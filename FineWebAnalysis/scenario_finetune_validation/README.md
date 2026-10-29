# Scenario-word held-out evaluation set (UNLABELED)

Held-out contexts for evaluating the RoBERTa slang-sense classifier on the
new scenario words, sampled from `10BT_scenario_merged` by
`prepare_finetune_data.py` and kept **disjoint** from the training set
(FineWebAnalysis\scenario_finetune_annotations). One CSV per target word; columns:
`target, is_slang, target_context, uri`.

## To use
1. Fill the `is_slang` column: `1` if the target word is used in its slang
   sense in that context, `0` if it is the ordinary sense. Leave blank to skip.
2. Score with the fine-tuned model and compute metrics:
   `python roberta_filter.py <this-dir> --score-all \
       --output-dir scored_eval/ --roberta-model-dir ./ft_model_roberta_scenario/`
   then treat `roberta_score >= 0` as predicted-slang and compare against
   `is_slang` for precision / recall / F1 (per word and overall).

Spelling variants that differ only in punctuation share one CSV; each row's
`target` keeps the spelling it matched.

## Contents (65 words, 2566 contexts, up to 40 each)

| word | contexts |
| --- | --- |
| til | 40 |
| fr | 40 |
| peak | 40 |
| frying | 40 |
| tilted | 40 |
| clutch | 40 |
| bc | 40 |
| cracked | 40 |
| carry | 40 |
| btw | 40 |
| trashed | 40 |
| goat | 40 |
| legend | 40 |
| idc | 40 |
| elite | 40 |
| uninstall | 40 |
| peaked | 40 |
| top tier / top-tier | 40 |
| imo | 40 |
| iconic | 40 |
| threw | 40 |
| clutched | 40 |
| deleted | 40 |
| choked | 40 |
| ate it | 40 |
| not me | 40 |
| flick | 40 |
| fumbled | 40 |
| wtf | 40 |
| nerf | 40 |
| feral | 40 |
| banger | 40 |
| the audacity | 40 |
| def | 40 |
| rn | 40 |
| demure | 40 |
| ppl | 40 |
| unhinged | 40 |
| delusion | 40 |
| pls | 40 |
| whiffed | 40 |
| dupe | 40 |
| tbh | 40 |
| bangers | 40 |
| smurf | 40 |
| choker | 40 |
| gg | 40 |
| flamed | 40 |
| no notes | 40 |
| ngl | 40 |
| very mindful | 40 |
| tryna | 40 |
| clip it | 40 |
| wayyyy | 40 |
| fyp | 40 |
| ggs | 40 |
| wyd | 40 |
| wayyyyy | 40 |
| tmrw | 40 |
| dunked on | 40 |
| heyyy | 40 |
| absolute legend | 40 |
| clowned | 40 |
| parasocial | 33 |
| goooo | 13 |

## Skipped: fewer than 50 contexts in the corpus (14 words)

| word | contexts |
| --- | --- |
| gooooooo | 49 |
| understood the assignment | 33 |
| secondhand embarrassment | 33 |
| very demure | 25 |
| dogpiled | 24 |
| huge w | 17 |
| diffed | 17 |
| he's him / he’s him | 16 |
| comfort show | 15 |
| touch grass | 11 |
| w stream | 4 |
| letsgooooo | 1 |
| fit ate | 1 |
| w chat | 1 |
