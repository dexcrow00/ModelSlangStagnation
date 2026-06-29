"""Generate linear scenario-frequency charts in groups of 6 words, ordered by
total response usage from slang_counts_by_model.json (most-used words first)."""
import importlib.util
import json
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("word_rate_plotter", HERE / "word_rate_plotter.py")
wrp = importlib.util.module_from_spec(spec); spec.loader.exec_module(wrp)

SCORED = HERE / "scenario_prompt_scored"
CACHE = HERE / "fineweb_10BT_dump_sizes.json"
COUNTS_JSON = (HERE.parent / "PromptingSlang" / "experiments" / "scenario_based_prompting"
              / "slang_counts_by_model.json")
SCENARIO_WORDS = HERE / "scenario_words.txt"
OUT_DIR = HERE / "scenario_charts"
THRESHOLD, SMOOTH, CONFIDENCE, GROUP = 0.99, 5, 0.95, 6

# Total response usage per word (summed over models).
data = json.loads(COUNTS_JSON.read_text(encoding="utf-8"))
total = Counter()
for info in data["models"].values():
    for w, c in info["words"].items():
        total[w] += c

# Scenario words ranked by response usage (desc), unseen words last (alpha tie-break).
words = [l.strip().lower() for l in SCENARIO_WORDS.read_text(encoding="utf-8").splitlines()
         if l.strip() and not l.startswith("#")]
ranked = sorted(words, key=lambda w: (-total.get(w, 0), w))

counts = wrp.load_dump_counts(SCORED, THRESHOLD)
dump_tokens = wrp.fetch_dump_token_counts(CACHE)
wrp._use_emoji_font()

for i in range(0, len(ranked), GROUP):
    grp = ranked[i:i + GROUP]
    n = i // GROUP + 1
    out = OUT_DIR / f"scenario_freq_count_group{n:02d}.png"
    print(f"group {n}: {grp}")
    wrp.plot_rates(counts, dump_tokens, THRESHOLD, grp, None, SMOOTH, False, out, CONFIDENCE)
