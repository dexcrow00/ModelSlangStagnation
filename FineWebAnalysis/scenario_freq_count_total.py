"""Single aggregated frequency chart for all scenario slang words combined."""
import sys
from collections import Counter, defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import word_rate_plotter as wrp  # noqa: E402

SCORED = HERE / "scenario_prompt_scored"
CACHE = HERE / "fineweb_10BT_dump_sizes.json"
SCENARIO_WORDS = HERE / "scenario_words.txt"
OUT = HERE / "scenario_charts" / "scenario_freq_count.png"
THRESHOLD, SMOOTH, CONFIDENCE = 0.99, 5, 0.95

LABEL = "all scenario words"

scenario_words = {
    l.strip().lower()
    for l in SCENARIO_WORDS.read_text(encoding="utf-8").splitlines()
    if l.strip() and not l.startswith("#")
}

raw_counts = wrp.load_dump_counts(SCORED, THRESHOLD)
dump_tokens = wrp.fetch_dump_token_counts(CACHE)
wrp._use_emoji_font()

# Aggregate counts across all scenario words per dump into a single series.
agg_counts: dict = defaultdict(Counter)
for d, counter in raw_counts.items():
    agg_counts[d][LABEL] = sum(v for w, v in counter.items() if w in scenario_words)

wrp.plot_rates(agg_counts, dump_tokens, THRESHOLD, [LABEL], None, SMOOTH, False, OUT, CONFIDENCE)
