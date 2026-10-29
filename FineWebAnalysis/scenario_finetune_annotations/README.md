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

Words with no standard sense (`slang_constants.ALWAYS_SLANG`: brooo, broooo, brooooo, brooooooo, cappin, circlejerked, cluttercore, gatekeeping, kekw, legit, lmaooo, lmaoooo, lmaooooo, lmfao, lul, meme'd, memed, meme’d, nahhh, nahhhh, nahhhhh, omegalul, omg, omggg, omgggg, peepoparty, pogchamp, ragequit, situationship, snoozefest, speedran, speedrun, speedrunning, stunlocked, tl;dr, yooo, yoooo, yooooo, yoooooo, 💀, 😭) are **excluded** --- `roberta_filter.py` always passes them, so they need no classifier training.

Spelling variants that differ only in punctuation share one CSV; each row's
`target` keeps the spelling it matched.

## Contents (73 words, 8495 contexts, up to 120 each)

| word | contexts |
| --- | --- |
| til | 120 |
| fr | 120 |
| peak | 120 |
| frying | 120 |
| tilted | 120 |
| clutch | 120 |
| bc | 120 |
| cracked | 120 |
| carry | 120 |
| btw | 120 |
| trashed | 120 |
| goat | 120 |
| legend | 120 |
| idc | 120 |
| elite | 120 |
| uninstall | 120 |
| peaked | 120 |
| top tier / top-tier | 120 |
| imo | 120 |
| iconic | 120 |
| threw | 120 |
| clutched | 120 |
| deleted | 120 |
| choked | 120 |
| ate it | 120 |
| not me | 120 |
| flick | 120 |
| fumbled | 120 |
| wtf | 120 |
| nerf | 120 |
| fyp | 120 |
| feral | 120 |
| banger | 120 |
| the audacity | 120 |
| def | 120 |
| rn | 120 |
| demure | 120 |
| ppl | 120 |
| unhinged | 120 |
| delusion | 120 |
| pls | 120 |
| whiffed | 120 |
| wayyyy | 120 |
| dupe | 120 |
| tbh | 120 |
| bangers | 120 |
| flamed | 120 |
| smurf | 120 |
| choker | 120 |
| gg | 120 |
| no notes | 120 |
| ngl | 120 |
| very mindful | 120 |
| tryna | 120 |
| clip it | 120 |
| ggs | 120 |
| wyd | 120 |
| wayyyyy | 120 |
| tmrw | 120 |
| dunked on | 120 |
| heyyy | 120 |
| absolute legend | 120 |
| clowned | 120 |
| parasocial | 120 |
| goooo | 120 |
| gooo | 112 |
| chef's kiss / chef’s kiss | 111 |
| diffing | 94 |
| gooooo | 92 |
| heyyyy | 92 |
| built different | 68 |
| goooooo | 65 |
| idek | 61 |

## Skipped: fewer than 50 contexts in the corpus (14 words)

| word | contexts |
| --- | --- |
| gooooooo | 49 |
| secondhand embarrassment | 33 |
| understood the assignment | 33 |
| very demure | 25 |
| dogpiled | 24 |
| diffed | 17 |
| huge w | 17 |
| he's him / he’s him | 16 |
| comfort show | 15 |
| touch grass | 11 |
| w stream | 4 |
| letsgooooo | 1 |
| w chat | 1 |
| fit ate | 1 |
