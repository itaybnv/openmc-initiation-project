"""Matplotlib style for slide figures (house-style.md, figures.md).

    import sys; sys.path.insert(0, "<repo>/.claude/skills/building-decks/scripts")
    import slide_style as ss
    ss.use()
    fig, ax = plt.subplots(figsize=ss.BOX_A)       # the box the figure will fill, in inches
    ...
    ss.save(fig, "shared/cartoons/<name>.png")

Colours by meaning: BLUE assembly / detector, CRIMSON pairs / fission / brute-force truth,
GREY references and point kinetics, INK everything else, ORANGE and ROSE for a third and
fourth series.
"""
from __future__ import annotations

INK, GREY, GREY_SOFT = "#30363D", "#8A939B", "#C4CACF"
BLUE, CRIMSON, ORANGE, ROSE = "#1F77B4", "#C0392B", "#C4620E", "#C2558B"
PALETTE = [BLUE, CRIMSON, ORANGE, ROSE, GREY, INK]

# figure boxes of the house layouts, inches (width, height)
BOX_A = (5.6, 3.75)        # figure right / left, one figure
BOX_ROW2 = (4.4, 2.5)      # two figures in a row under a band
BOX_ROW3 = (2.9, 2.3)      # three in a row
BOX_FULL = (9.0, 3.6)      # one figure across the content area


def use(base_pt: float = 11.0):
    import matplotlib as mpl
    mpl.rcParams.update({
        "font.family": ["Carlito", "Calibri", "DejaVu Sans"],
        "font.size": base_pt, "axes.titlesize": base_pt, "axes.labelsize": base_pt,
        "xtick.labelsize": base_pt - 1, "ytick.labelsize": base_pt - 1, "legend.fontsize": base_pt - 1,
        "axes.edgecolor": INK, "axes.labelcolor": INK, "xtick.color": INK, "ytick.color": INK,
        "text.color": INK, "axes.spines.top": False, "axes.spines.right": False,
        "axes.prop_cycle": mpl.cycler(color=PALETTE),
        "axes.unicode_minus": True, "legend.frameon": False,
        "lines.linewidth": 1.4, "lines.markersize": 5, "errorbar.capsize": 2,
        "figure.dpi": 100, "savefig.dpi": 200, "savefig.facecolor": "white",
        "axes.grid": False,
    })


def save(fig, path, dpi: int = 200):
    from pathlib import Path
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=dpi, bbox_inches="tight", pad_inches=0.04, facecolor="white")
    return path


def band(ax, x, lo, hi, color=BLUE, alpha=0.18):
    """Translucent fit band behind data (draw before the points)."""
    return ax.fill_between(x, lo, hi, color=color, alpha=alpha, lw=0, zorder=0)
