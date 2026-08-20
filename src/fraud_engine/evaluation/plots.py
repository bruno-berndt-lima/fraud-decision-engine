"""Figures for the evaluation harness.

The only module that imports matplotlib. Keeping it here means ``report.py`` and
``metrics.py`` stay importable in the Phase 08 container, which has no reason to
carry a plotting library — and an accidental import there fails loudly instead of
quietly bloating the image.

Figures are written to ``reports/figures/``, which is tracked: they are
deliverables, not debugging output.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path

import matplotlib

# Report generation, never interactive. Without this matplotlib may select a GUI
# backend on a developer machine and none at all in a container.
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import precision_recall_curve

from fraud_engine.evaluation.metrics import recall_at_capacity

# Categorical slots 1-3, assigned in fixed order and never cycled. Documented as
# passing all-pairs colour-vision separation in both modes. A fourth series folds
# into "other" or becomes a second figure rather than borrowing slot 4, which
# puts orange and yellow on screen together.
SERIES_COLOURS = ("#2a78d6", "#eb6834", "#1baf7a")

# Chart chrome. Text never wears a series colour — the line beside it carries
# identity.
INK = "#0b0b0b"
INK_MUTED = "#52514e"
GRIDLINE = "#e1e0d9"
AXIS = "#c3c2b7"
SURFACE = "#fcfcfb"

# Slot 3 sits below 3:1 contrast on this surface, so every series carries a
# direct label rather than relying on the legend swatch alone.
_MAX_SERIES = len(SERIES_COLOURS)


def _new_axes(figsize=(7.0, 4.5)):
    """A styled figure and axes: recessive chrome, no chartjunk."""
    figure, axes = plt.subplots(figsize=figsize, dpi=150)
    figure.patch.set_facecolor(SURFACE)
    axes.set_facecolor(SURFACE)

    axes.grid(True, color=GRIDLINE, linewidth=1, linestyle="-")
    axes.set_axisbelow(True)
    for side in ("top", "right"):
        axes.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        axes.spines[side].set_color(AXIS)
        axes.spines[side].set_linewidth(1)

    axes.tick_params(colors=INK_MUTED, labelsize=9, length=0)
    return figure, axes


def _check_series(scores: Mapping[str, pd.Series]) -> None:
    if not scores:
        raise ValueError("no series to plot.")
    if len(scores) > _MAX_SERIES:
        raise ValueError(
            f"{len(scores)} series, but only {_MAX_SERIES} categorical slots are "
            f"validated for this palette. Fold the extras together or use a second "
            f"figure rather than adding a colour."
        )


def plot_pr_curve(
    y_true: pd.Series,
    scores: Mapping[str, pd.Series],
    title: str = "Precision-recall",
) -> plt.Figure:
    """Precision-recall curves, with the base-rate floor drawn.

    **The floor is not decoration.** A PR-AUC of 0.30 is excellent against a
    0.035 base rate and worthless against a 0.30 one, so a PR curve without its
    floor cannot be read. It is drawn as a reference line rather than a series:
    it is a property of the data, not a competitor.

    Curves are drawn as **steps, not lines**. The straight path between two PR
    points is not reachable by any threshold — which is exactly why ``pr_auc``
    is average precision rather than a trapezoid. Drawing a smooth curve would
    picture area the metric deliberately refuses to count.

    Args:
        y_true: Binary labels, shared by every series.
        scores: ``{label: risk scores}``, aligned to ``y_true``. At most three.
        title: Figure title.

    Returns:
        The figure, unsaved. Use ``save_figure``.

    Raises:
        ValueError: If there are no series, or more than the palette validates.
    """
    _check_series(scores)
    figure, axes = _new_axes()

    base_rate = float(y_true.mean())
    peak = 0.0
    for colour, (label, score) in zip(SERIES_COLOURS, scores.items(), strict=False):
        precision, recall, _ = precision_recall_curve(y_true, score)
        # sklearn appends a synthetic (precision=1, recall=0) endpoint that no
        # threshold produces. Left in, it draws a spike to the top of the axis
        # and dominates the scale. Dropped.
        precision, recall = precision[:-1], recall[:-1]
        # Scale on precision at meaningful recall only. Below ~5% recall the
        # precision is computed over a handful of predictions and swings wildly;
        # one such spike would set the axis and flatten every real curve. The
        # spike is still drawn, it just runs off the top.
        meaningful = precision[recall >= 0.05]
        if meaningful.size:
            peak = max(peak, float(meaningful.max()))
        # sklearn returns recall descending; reverse so the step runs left to
        # right. steps-pre puts each precision over the recall interval ending
        # at it, which is the interval average precision actually sums.
        axes.plot(
            recall[::-1],
            precision[::-1],
            drawstyle="steps-pre",
            color=colour,
            linewidth=2,
            solid_joinstyle="round",
            solid_capstyle="round",
            label=label,
        )

    axes.axhline(base_rate, color=AXIS, linewidth=1)
    axes.annotate(
        f"base rate {base_rate:.3f} — a random ranker scores here",
        xy=(0.99, base_rate),
        # Below the floor, not above it: every curve converges onto the base
        # rate at high recall, so the space above is the one place it collides.
        xytext=(0, -5),
        textcoords="offset points",
        ha="right",
        va="top",
        fontsize=8,
        color=INK_MUTED,
    )

    axes.set_xlim(0, 1)
    # Not (0, 1). At a 3.5% base rate every curve lives in the bottom tenth of a
    # full axis, and the shape - which is the whole point of the figure - becomes
    # unreadable. Scaled to the data, with the floor always in frame.
    # Capped at 1: precision cannot exceed it, so headroom above is dead space.
    axes.set_ylim(0, min(1.0, max(peak * 1.3, base_rate * 3)))
    axes.set_xlabel("Recall", color=INK_MUTED, fontsize=9)
    axes.set_ylabel("Precision", color=INK_MUTED, fontsize=9)
    axes.set_title(title, color=INK, fontsize=11, loc="left", pad=12)

    if len(scores) > 1:
        axes.legend(frameon=False, fontsize=9, labelcolor=INK_MUTED, loc="upper right")

    figure.tight_layout()
    return figure


def plot_recall_at_capacity(
    y_true: pd.Series,
    scores: Mapping[str, pd.Series],
    days: pd.Series,
    capacities: Sequence[float] | None = None,
    operating_capacity: float = 0.01,
    title: str = "Recall at review capacity",
) -> plt.Figure:
    """Recall as a function of the daily review budget.

    The operational curve. Recall alone is a free parameter — review everything
    and it reaches 1.0 — so the honest reading is recall *against what it cost*,
    and this figure is that trade drawn out.

    A **perfect ranker** is drawn as a ceiling. At a small capacity the limit is
    the number of review seats, not the model: on this data an oracle catches
    only about 28% of fraud at 1% of daily volume. Without that line a reader
    has no way to tell a weak model from a tight constraint.

    Args:
        y_true: Binary labels, shared by every series.
        scores: ``{label: risk scores}``, aligned to ``y_true``. At most three.
        days: The day each transaction falls on — capacity is per day.
        capacities: Points to evaluate. Defaults to 25 points spanning
            0.2%-5% of daily volume.
        operating_capacity: The committed constraint, marked with a rule.
        title: Figure title.

    Returns:
        The figure, unsaved. Use ``save_figure``.

    Raises:
        ValueError: If there are no series, or more than the palette validates.
    """
    _check_series(scores)
    if capacities is None:
        capacities = np.linspace(0.002, 0.05, 25)
    capacities = sorted(float(capacity) for capacity in capacities)

    figure, axes = _new_axes()

    def recall_curve(score: pd.Series) -> list[float]:
        return [recall_at_capacity(y_true, score, days, c).recall for c in capacities]

    # The ceiling first, so the model curves draw over it.
    oracle = recall_curve(y_true.astype(float))
    axes.plot(capacities, oracle, color=AXIS, linewidth=1)
    # Labelled below its own curve, two thirds along: the right edge clips, and
    # the top band belongs to the capacity rule.
    anchor = int(len(capacities) * 0.62)
    axes.annotate(
        "perfect ranker — the ceiling is the seat count, not the model",
        xy=(capacities[anchor], oracle[anchor]),
        xytext=(4, -12),
        textcoords="offset points",
        ha="left",
        va="top",
        fontsize=8,
        color=INK_MUTED,
    )

    for colour, (label, score) in zip(SERIES_COLOURS, scores.items(), strict=False):
        curve = recall_curve(score)
        axes.plot(
            capacities,
            curve,
            color=colour,
            linewidth=2,
            solid_joinstyle="round",
            solid_capstyle="round",
        )
        # Direct labels rather than a legend box. Identity sits beside the mark
        # it names instead of in a corner, and the top-left stays free for the
        # capacity rule, which a legend was colliding with.
        axes.annotate(
            label,
            xy=(capacities[-1], curve[-1]),
            xytext=(6, 0),
            textcoords="offset points",
            va="center",
            ha="left",
            fontsize=9,
            color=INK_MUTED,
        )

    axes.axvline(operating_capacity, color=AXIS, linewidth=1)
    axes.annotate(
        f"committed capacity {operating_capacity:.1%} of daily volume",
        # Axes-fraction y so the anchor does not depend on set_ylim running first.
        xy=(operating_capacity, 1.0),
        xycoords=("data", "axes fraction"),
        xytext=(4, -8),
        textcoords="offset points",
        ha="left",
        va="top",
        fontsize=8,
        color=INK_MUTED,
    )

    # Right margin for the direct labels, with ticks kept inside the data range
    # so no gridline runs through the label gutter.
    axes.set_xlim(capacities[0], capacities[-1] + (capacities[-1] - capacities[0]) * 0.26)
    axes.set_xticks([tick for tick in axes.get_xticks() if capacities[0] <= tick <= capacities[-1]])
    axes.set_xlim(capacities[0], capacities[-1] + (capacities[-1] - capacities[0]) * 0.26)
    axes.set_ylim(0, None)
    axes.xaxis.set_major_formatter(lambda value, _: f"{value:.0%}")
    axes.yaxis.set_major_formatter(lambda value, _: f"{value:.0%}")
    axes.set_xlabel("Share of daily transactions reviewed", color=INK_MUTED, fontsize=9)
    axes.set_ylabel("Fraud caught", color=INK_MUTED, fontsize=9)
    axes.set_title(title, color=INK, fontsize=11, loc="left", pad=12)

    figure.tight_layout()
    return figure


def save_figure(figure: plt.Figure, path: Path | str) -> Path:
    """Write a figure and release it.

    Closing matters: a report run producing a dozen figures otherwise holds
    every one open until the process exits.

    Args:
        figure: From one of the plot functions.
        path: Destination. Parent directories are created.

    Returns:
        The path written.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, facecolor=figure.get_facecolor(), bbox_inches="tight")
    plt.close(figure)
    return path
