"""The Phase 09 figures, in the house style `evaluation/plots.py` defines.

Its own module rather than functions added there, because that module is a
prerequisite of stages whose records are frozen: editing it would mark them stale for
a figure none of them draws. It borrows the palette and chrome instead of copying them.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.lines import Line2D

from fraud_engine.evaluation.plots import (
    AXIS,
    GRIDLINE,
    INK,
    INK_MUTED,
    SERIES_COLOURS,
    SURFACE,
    style_axes,
)

# Fixed order, never cycled: one slot per slice, in the order they happened.
SLICES = {
    "val_fit": ("VAL-FIT", SERIES_COLOURS[0]),
    "val_cal": ("VAL-CAL", SERIES_COLOURS[1]),
    "test": ("test", SERIES_COLOURS[2]),
}
# What a reader needs to know about a slice before trusting its points.
CAVEATS = {"val_fit": "VAL-FIT (early-stopped on)"}


def _points(axes, table: pd.DataFrame, column: str, *, interval: bool = False) -> None:
    """One slice at a time, with the partial window drawn hollow."""
    for split, (_, colour) in SLICES.items():
        part = table[table["slice"] == split]
        for partial, face in ((False, colour), (True, SURFACE)):
            rows = part[part["partial"] == partial]
            if rows.empty:
                continue
            error = None
            if interval:
                error = [
                    rows[column] - rows[f"{column}_low"],
                    rows[f"{column}_high"] - rows[column],
                ]
            axes.errorbar(
                rows["days_since_training"],
                rows[column],
                yerr=error,
                fmt="o",
                color=colour,
                markerfacecolor=face,
                markeredgecolor=colour,
                markersize=5,
                elinewidth=1.5,
                capsize=0,
            )


def _baseline(axes, value: float, text: str) -> None:
    axes.axhline(value, color=INK_MUTED, linewidth=1, linestyle="--")
    axes.annotate(
        text,
        xy=(1.0, value),
        xycoords=("axes fraction", "data"),
        xytext=(4, 0),
        textcoords="offset points",
        va="center",
        ha="left",
        fontsize=8,
        color=INK_MUTED,
    )


def plot_decay(table: pd.DataFrame, baseline: dict, bound: dict) -> plt.Figure:
    """PR-AUC by window, with the three things it cannot be read without beneath it.

    Four panels on one horizontal axis, never a second y-scale: PR-AUC with its bar and
    the `VAL-CAL` baseline; ROC-AUC, which prevalence does not move; the base rate,
    which does; and the identity share, against the range validation already spans.

    Args:
        table: `decay.read_windows` output.
        baseline: `{"pr_auc", "roc_auc"}` of the pooled `VAL-CAL` slice.
        bound: `decay.composition_bound` output, for the identity band.

    Returns:
        The figure, unsaved. Use `save_figure`.
    """
    figure, (pr, roc, rate, mix) = plt.subplots(
        4, 1, figsize=(7.0, 9.0), dpi=150, sharex=True, height_ratios=(3, 2, 1.4, 1.4)
    )
    figure.patch.set_facecolor(SURFACE)
    for axes in (pr, roc, rate, mix):
        style_axes(axes)

    _points(pr, table, "pr_auc", interval=True)
    _baseline(pr, baseline["pr_auc"], "VAL-CAL\npooled")
    pr.set_ylabel("PR-AUC", color=INK_MUTED, fontsize=9)
    pr.set_title(
        "Decay by five-day window, with 95% day-bootstrap intervals",
        color=INK,
        fontsize=11,
        loc="left",
        pad=12,
    )
    low, high = pr.get_ylim()
    pr.set_ylim(low, high + (high - low) * 0.12)
    for split, (label, _) in SLICES.items():
        centre = table.loc[table["slice"] == split, "days_since_training"].mean()
        pr.annotate(
            label,
            xy=(centre, 0.97),
            xycoords=("data", "axes fraction"),
            ha="center",
            va="top",
            fontsize=9,
            color=INK_MUTED,
        )

    _points(roc, table, "roc_auc", interval=True)
    _baseline(roc, baseline["roc_auc"], "VAL-CAL\npooled")
    roc.set_ylabel("ROC-AUC", color=INK_MUTED, fontsize=9)

    _points(rate, table.assign(base_rate=100 * table["base_rate"]), "base_rate")
    rate.set_ylabel("fraud rate, %", color=INK_MUTED, fontsize=9)

    mix.axhspan(100 * bound["low"], 100 * bound["high"], color=GRIDLINE, linewidth=0)
    mix.annotate(
        "spanned by\nvalidation",
        xy=(1.0, 100 * (bound["low"] + bound["high"]) / 2),
        xycoords=("axes fraction", "data"),
        xytext=(4, 0),
        textcoords="offset points",
        va="center",
        ha="left",
        fontsize=8,
        color=INK_MUTED,
    )
    _points(mix, table.assign(identity_share=100 * table["identity_share"]), "identity_share")
    mix.set_ylabel("with identity, %", color=INK_MUTED, fontsize=9)
    mix.set_xlabel("days since the training window ended", color=INK_MUTED, fontsize=9)

    for axes in (pr, roc, rate, mix):
        axes.spines["bottom"].set_color(AXIS)

    handles = [
        Line2D([], [], marker="o", linestyle="", color=colour, markersize=5)
        for _, colour in SLICES.values()
    ] + [
        Line2D(
            [], [], marker="o", linestyle="", markerfacecolor=SURFACE,
            markeredgecolor=INK_MUTED, markersize=5,
        )
    ]  # fmt: skip
    labels = [CAVEATS.get(split, label) for split, (label, _) in SLICES.items()]
    figure.tight_layout(rect=(0, 0.035, 1, 1))
    figure.legend(
        handles,
        [*labels, "partial window"],
        loc="lower center",
        ncol=len(handles),
        fontsize=8,
        frameon=False,
        labelcolor=INK_MUTED,
    )
    return figure
