"""Shared figure style.

One palette for the whole report so that a colour means the same thing in
every panel: slot 1 is always the uncorrected Gaussian, slot 2 the calibrated
DRAG pulse, slot 3 the numerically optimised pulse.  Ideal or target values are
drawn in neutral ink rather than a fourth hue, keeping the categorical set to
three mutually distinguishable colours under normal and deficient colour
vision.  Sequential maps use a single blue ramp (light = small) and signed
quantities a blue/red pair with a neutral midpoint.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib as mpl
import numpy as np
from matplotlib.colors import LinearSegmentedColormap

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK2 = "#52514e"
MUTED = "#8f8e88"
GRID = "#e3e2dd"

#: Fixed meaning, never cycled.
PLAIN, DRAG, OPT = "#2a78d6", "#eb6834", "#1baf7a"
SERIES = (PLAIN, DRAG, OPT)

#: Second sequential context (leakage), one hue of its own so it is never
#: confused with the P(|1>) maps.
SEQ2 = LinearSegmentedColormap.from_list("seq_orange", [
    "#fdf1ea", "#fbd9c6", "#f7b491", "#f28e5c", "#eb6834", "#c14f22", "#943a17", "#66270f"])

SEQ = LinearSegmentedColormap.from_list("seq_blue", [
    "#f4f8fe", "#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"])
SEQ_R = SEQ.reversed()
DIV = LinearSegmentedColormap.from_list("div_br", [
    "#0d366b", "#2a78d6", "#9ec5f4", "#f0efec", "#f4a5a4", "#e34948", "#8f1f1e"])


def use_style() -> None:
    """Install the house rcParams; call once at the top of every script."""
    mpl.rcParams.update({
        "figure.facecolor": SURFACE, "savefig.facecolor": SURFACE,
        "axes.facecolor": SURFACE, "axes.edgecolor": MUTED,
        "axes.labelcolor": INK, "axes.titlecolor": INK,
        "axes.linewidth": 0.8, "axes.grid": True, "axes.axisbelow": True,
        "axes.titlesize": 10.5, "axes.labelsize": 9.5, "axes.titleweight": "semibold",
        "axes.titlelocation": "left", "axes.titlepad": 8,
        "grid.color": GRID, "grid.linewidth": 0.7,
        "xtick.color": INK2, "ytick.color": INK2, "xtick.labelsize": 8.5,
        "ytick.labelsize": 8.5, "xtick.direction": "out", "ytick.direction": "out",
        "legend.frameon": False, "legend.fontsize": 8.5, "legend.labelcolor": INK2,
        "lines.linewidth": 1.8, "lines.markersize": 4.5,
        "font.size": 9.5, "font.family": "DejaVu Sans",
        "figure.dpi": 140, "savefig.dpi": 220, "savefig.bbox": "tight",
    })


def despine(*axes) -> None:
    for ax in axes:
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)


def annotate(ax, text: str, xy, xytext, color: str = INK2, **kw) -> None:
    """Direct label with a hairline leader; used instead of dense legends."""
    ax.annotate(text, xy=xy, xytext=xytext, color=color, fontsize=8.5,
                arrowprops=dict(arrowstyle="-", color=color, lw=0.7,
                                shrinkA=0, shrinkB=3), **kw)


def caption(fig, text: str, width: int = 118) -> None:
    """Footnote under the figure, wrapped so it never widens the saved bounding box."""
    import textwrap

    fig.text(0.008, -0.012, "\n".join(textwrap.wrap(text, width)), fontsize=8, color=MUTED,
             va="top", ha="left")


def save(fig, path: Path | str) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path)
    return path


def bloch_frame(ax, r: float = 1.0) -> None:
    """Draw a recessive Bloch sphere: equator, meridians and labelled poles."""
    th = np.linspace(0, 2 * np.pi, 200)
    ax.plot(r * np.cos(th), r * np.sin(th), 0, color=MUTED, lw=0.6, alpha=0.6)
    for phi in (0, np.pi / 2):
        ax.plot(r * np.cos(th) * np.cos(phi), r * np.cos(th) * np.sin(phi),
                r * np.sin(th), color=MUTED, lw=0.5, alpha=0.35)
    for vec, lab in (((1, 0, 0), "x"), ((0, 1, 0), "y"), ((0, 0, 1), r"$|0\rangle$")):
        v = np.array(vec) * r
        ax.plot(*zip((0, 0, 0), v), color=MUTED, lw=0.6, alpha=0.7)
        ax.text(*v * 1.16, lab, color=INK2, fontsize=8, ha="center", va="center")
    ax.text(0, 0, -1.2 * r, r"$|1\rangle$", color=INK2, fontsize=8, ha="center")
    ax.set_axis_off()
    ax.set_box_aspect((1, 1, 1))
    ax.set_xlim(-0.78 * r, 0.78 * r)
    ax.set_ylim(-0.78 * r, 0.78 * r)
    ax.set_zlim(-0.78 * r, 0.78 * r)


def colored_path(ax, xyz: np.ndarray, values: np.ndarray, cmap, norm, lw: float = 2.2):
    """3-D polyline whose colour tracks a per-point scalar (here, leakage)."""
    from mpl_toolkits.mplot3d.art3d import Line3DCollection

    segs = np.stack([xyz[:-1], xyz[1:]], axis=1)
    lc = Line3DCollection(segs, cmap=cmap, norm=norm, linewidths=lw)
    lc.set_array(0.5 * (values[:-1] + values[1:]))
    ax.add_collection3d(lc)
    return lc
