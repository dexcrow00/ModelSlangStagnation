"""Ridgeline chart shared by the year-swept visualizers.

One ridge per target word (earliest true corpus peak at the top) tracing a rate
across prompt years. Ridge height and fill colour both encode the rate; the
colour is a diverging red <-> blue scale centred on a reference rate (chance,
an even split, ...), so cells below the reference read warm and above it cool.
A dark vertical line through each ridge marks the word's true corpus peak year.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import patheffects
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm
from matplotlib.path import Path as MplPath

# Diverging red <-> blue around a neutral gray: below the centre reads warm, above cool.
RATE_CMAP = LinearSegmentedColormap.from_list(
    "below_above_center",
    ["#a82a2b", "#e34948", "#f5c4c2", "#f0efec", "#b7d3f6", "#5598e7", "#1c5cab"])


def mean_rates(per_model: dict[str, dict[str, dict[str, float]]]) -> dict[str, dict[str, float]]:
    """``{model: {word: {year: rate}}}`` -> ``{word: {year: rate}}``, each model weighted equally.

    A mean of per-model rates rather than pooled counts, so a model with more
    responses in a cell does not outvote the others. NaN rates (no usable
    responses) are left out of that cell's mean.
    """
    cells: dict[str, dict[str, list[float]]] = {}
    for words in per_model.values():
        for w, years in words.items():
            for y, r in years.items():
                if np.isfinite(r):
                    cells.setdefault(w, {}).setdefault(y, []).append(r)
    return {w: {y: float(np.mean(v)) for y, v in years.items()} for w, years in cells.items()}


def render_ridgeline(rates: dict[str, dict[str, float]], peaks: dict[str, int], *,
                     center: float, center_label: str, title: str, xlabel: str,
                     cbar_label: str, output: Path | None) -> None:
    """Draw ``{word: {year: rate}}`` as a ridgeline and save it (or show it if ``output`` is None)."""
    words = sorted(rates, key=lambda w: (peaks.get(w, 9999), w))  # earliest peak first (top)
    years = sorted({int(y) for w in words for y in rates[w]})
    if not words or not years:
        raise SystemExit("No year-resolved records to chart.")

    # Ridge height at 100%, in row spacings. Most rates sit near the ceiling, so
    # ridges taller than a row would hide the one above; keep each in its own band.
    overlap = 0.9
    edge, peak_color = "#52514e", "#0b0b0b"
    norm = TwoSlopeNorm(vmin=0.0, vcenter=center, vmax=1.0)
    x = np.array(years)
    xs = np.linspace(x[0], x[-1], 600)
    n = len(words)
    fig, ax = plt.subplots(figsize=(max(9, len(years) * 0.75) + 1.5, 1.5 + n * 0.5))

    im = None
    for i, w in enumerate(words):
        base = n - 1 - i
        y = np.array([rates[w].get(str(yr), np.nan) for yr in years])
        ok = np.isfinite(y)
        if not ok.any():
            continue
        top = base + overlap * y
        # Draw top to bottom so each ridge sits in front of the one above it.
        z = 3 * i
        # Colour the area under the ridge by the rate at each x: a one-row gradient
        # image, clipped to the ridge's outline (gaps for missing years stay empty).
        grad = np.interp(xs, x[ok], y[ok])[np.newaxis, :]
        im = ax.imshow(grad, cmap=RATE_CMAP, norm=norm, aspect="auto", origin="lower",
                       interpolation="bilinear", extent=(x[0], x[-1], base, base + overlap),
                       zorder=z)
        outline = ax.fill_between(x, base, top, where=ok, facecolor="none", linewidth=0)
        im.set_clip_path(MplPath.make_compound_path(*outline.get_paths()), ax.transData)
        outline.remove()
        ax.plot(x, top, color=edge, linewidth=1.5, zorder=z + 2)
        ax.plot([x[0], x[-1]], [base, base], color="#c3c2b7", linewidth=0.8, zorder=z + 2)
        pk = peaks.get(w)
        if pk is not None and years[0] <= pk <= years[-1]:
            ax.vlines(pk, base, base + overlap, color=peak_color, linewidth=2, zorder=z + 1,
                      path_effects=[patheffects.withStroke(linewidth=4, foreground="white")])

    # Scale key for ridge height, beside the bottom ridge.
    kx = x[-1] + 0.6
    ax.plot([kx, kx], [0, overlap], color="#52514e", linewidth=1, clip_on=False)
    for frac, label in ((0, "0%"), (center, f"{center_label} {100 * center:.0f}%"), (1, "100%")):
        ax.plot([kx - 0.08, kx], [overlap * frac] * 2, color="#52514e", linewidth=1, clip_on=False)
        ax.text(kx + 0.12, overlap * frac, label, va="center", fontsize=7, color="#52514e")

    ax.set_xticks(years)
    ax.set_xticklabels([str(yr) for yr in years], rotation=45, ha="right", fontsize=8)
    ax.set_xlim(x[0] - 0.3, x[-1] + 0.3)
    ax.set_yticks(np.arange(n) + overlap / 2)   # label the middle of each ridge's band
    ax.set_yticklabels([f"{w} ({peaks[w]})" if w in peaks else w for w in reversed(words)],
                       fontsize=8)
    ax.set_ylim(-0.3, n - 1 + overlap + 0.2)
    ax.tick_params(axis="y", length=0)
    for side in ("left", "right", "top"):
        ax.spines[side].set_visible(False)
    ax.grid(axis="x", color="#e1e0d9", linewidth=0.6, zorder=-1)
    ax.set_axisbelow(True)

    ax.set_xlabel(xlabel, fontsize=9)
    ax.set_ylabel("target word (true corpus peak year)", fontsize=9)
    ax.set_title(title, fontsize=11, loc="left")
    ax.legend(handles=[plt.Line2D([], [], color=peak_color, linewidth=2,
                                  label="true corpus peak year")],
              loc="lower right", bbox_to_anchor=(1.0, 1.0), fontsize=8, frameon=False)
    if im is not None:
        cbar = fig.colorbar(im, ax=ax, shrink=0.35, anchor=(0.0, 1.0), pad=0.1)
        ticks = sorted([t for t in (0.0, 0.25, 0.5, 0.75, 1.0) if abs(t - center) > 0.04]
                       + [center])
        cbar.set_ticks(ticks, labels=[f"{100 * t:.0f}%" + (f" ({center_label})" if t == center
                                                           else "") for t in ticks])
        cbar.ax.tick_params(labelsize=7)
        cbar.set_label(cbar_label, fontsize=8)
        cbar.outline.set_visible(False)
    fig.tight_layout()

    if output:
        output.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output, dpi=120, bbox_inches="tight")
        print(f"Saved: {output}")
    else:
        plt.show()
    plt.close(fig)
