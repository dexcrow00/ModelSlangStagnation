"""Shared constants for the FineWeb RoBERTa slang pipeline.

Kept dependency-free so both ``roberta_filter.py`` (inference) and
``prepare_finetune_data.py`` (data prep) can import it without pulling in torch.
"""

from __future__ import annotations

# Target words with no standard (non-slang) sense: every occurrence is slang.
# Consequences of membership here:
#   - roberta_filter.py assigns score 1.0 and always passes the word (bypasses
#     the classifier).
# Compared case-folded.
ALWAYS_SLANG = frozenset({
    "omg", "omgggg", "lmfao", "lmaooo", "lmaoooo", "lmaooooo",
    "legit", "situationship", "💀", "circlejerked", "meme'd", "meme’d", "memed",
    "cappin", "snoozefest", "speedrun", "speedran", "speedrunning", "cluttercore",
    "tl;dr", "kekw", "omegalul", "lul", "peepoparty", "pogchamp", "ragequit", "stunlocked",
    "gatekeeping", "😭", "brooo", "broooo", "brooooo", "brooooooo", "yooo", "yoooo", "yooooo", 
    "yoooooo", "nahhh", "nahhhh", "nahhhhh", "omggg"
})
