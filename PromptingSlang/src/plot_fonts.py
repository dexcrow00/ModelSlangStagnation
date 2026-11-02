"""Matplotlib font setup so emoji target words (e.g. 💀) render in chart labels."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib import font_manager

# Monochrome Noto Emoji (SIL OFL, see fonts/OFL.txt), shared with FineWebAnalysis.
EMOJI_FONT = Path(__file__).resolve().parents[2] / "fonts" / "NotoEmoji.ttf"


def use_emoji_font() -> None:
    """Fall back to Noto Emoji for any glyph DejaVu Sans lacks.

    Colour emoji fonts (Apple Color Emoji, Segoe UI Emoji, Noto Color Emoji)
    store bitmap glyphs that matplotlib's renderer cannot draw, so a bundled
    outline font is used instead. Both families go straight into font.family,
    not font.sans-serif: matplotlib only falls back glyph-by-glyph across fonts
    listed there, and a generic family resolves to a single font.
    """
    font_manager.fontManager.addfont(EMOJI_FONT)
    plt.rcParams["font.family"] = ["DejaVu Sans", "Noto Emoji"]
