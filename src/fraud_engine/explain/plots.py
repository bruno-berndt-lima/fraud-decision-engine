"""The two Phase 07 figures the house style draws, rather than `shap`.

`shap` draws the beeswarm and the waterfalls: those are canonical forms a reader
recognises on sight, and reimplementing a waterfall's cumulative bars and tail
aggregation is work with nothing to show for it. These two it cannot draw.

**The tier ranking**, because serving tier is not something `shap` knows about. It is
also the figure this phase actually needs: the beeswarm's top rows are `C*` and `D*`
columns whose definitions were never published, so the conventional global view is
close to mute here, and what a reader has to see is how much of the model can be put
into a sentence at all.

**The amount dependence**, because H1 is a claim about *shape* — rising with amount
within a product and flattening at the top, rather than turning over — and reading a
shape means one panel per product, not five series in one.

Both borrow `evaluation/plots.py`'s palette and chrome. Neither is importable from the
serving path, which is the rule that module's docstring actually states.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import LogLocator, NullFormatter, ScalarFormatter

from fraud_engine.evaluation.plots import (
    GRIDLINE,
    INK,
    INK_MUTED,
    SERIES_COLOURS,
    SURFACE,
    new_axes,
    style_axes,
)

# Four serving tiers, three colour slots. `plots.py` refuses a fourth series and gives
# the reason — slot 4 puts orange and yellow on screen together — so the two tiers that
# answer the same serving question are drawn as one group. Nothing is lost: `features.md`
# and §4's table carry the four-way split, and what this figure is for is the line
# between what can be named and what cannot.
TIER_GROUPS = {
    "tier_0": ("inherited, unreproducible", SERIES_COLOURS[1]),
    "tier_1": ("request-only", SERIES_COLOURS[0]),
    "tier_2": ("fitted table or entity state", SERIES_COLOURS[2]),
    "tier_3": ("fitted table or entity state", SERIES_COLOURS[2]),
}


def plot_tier_ranking(
    ranked: pd.DataFrame,
    top_n: int = 20,
    title: str = "What moves the model, and what can be named",
) -> plt.Figure:
    """The strongest contributors as bars, coloured by whether a sentence exists for them.

    Args:
        ranked: As `contributions.ranking` returned it — `feature`, `tier`, `mean_abs`,
            ordered by `mean_abs` descending.
        top_n: How many features to draw, from the top.
        title: Figure title.

    Returns:
        The figure, unsaved. Use `evaluation.plots.save_figure`.

    Raises:
        ValueError: If a tier has no group — a new serving tier needs a deliberate
            colour decision, not whatever the palette happens to have left.
    """
    unknown = sorted(set(ranked["tier"]) - set(TIER_GROUPS))
    if unknown:
        raise ValueError(f"no colour group for serving tier(s) {unknown}")

    shown = ranked.head(top_n).iloc[::-1]
    figure, axes = new_axes(figsize=(7.5, 0.32 * len(shown) + 1.4))

    positions = np.arange(len(shown))
    axes.barh(
        positions,
        shown["mean_abs"],
        color=[TIER_GROUPS[tier][1] for tier in shown["tier"]],
        height=0.72,
    )
    axes.set_yticks(positions, shown["feature"], fontsize=9)
    axes.set_xlabel("mean |contribution| to the log-odds", color=INK_MUTED, fontsize=9)
    axes.set_title(title, color=INK, fontsize=11, loc="left", pad=12)
    axes.grid(axis="y", visible=False)

    # One handle per group, not per tier, so the two that share a colour share a row.
    seen: dict[str, str] = {}
    for tier in ranked["tier"]:
        label, colour = TIER_GROUPS[tier]
        seen.setdefault(label, colour)
    axes.legend(
        handles=[plt.Rectangle((0, 0), 1, 1, color=colour) for colour in seen.values()],
        labels=list(seen),
        loc="lower right",
        frameon=False,
        fontsize=9,
        labelcolor=INK_MUTED,
    )

    return figure


def binned_median(x: np.ndarray, y: np.ndarray, bins: int) -> tuple[np.ndarray, np.ndarray]:
    """`y`'s median within quantile bins of `x`, and each bin's centre.

    Quantile bins, not equal width: transaction amounts are heavily skewed, and equal
    width would put almost every row in the first bin and read a shape off the handful
    of rows in the rest.

    Args:
        x: The axis to bin along.
        y: The values to summarise.
        bins: How many bins to aim for. Fewer are returned when `x` has ties at its
            quantiles, which is what a popular round amount looks like.

    Returns:
        `(centres, medians)`, empty when there is nothing to bin.
    """
    if len(x) == 0:
        return np.empty(0), np.empty(0)

    edges = np.unique(np.quantile(x, np.linspace(0, 1, bins + 1)))
    if len(edges) < 2:
        return np.array([float(np.median(x))]), np.array([float(np.median(y))])

    index = np.clip(np.digitize(x, edges[1:-1]), 0, len(edges) - 2)
    frame = pd.DataFrame({"bin": index, "x": x, "y": y}).groupby("bin")
    return frame["x"].median().to_numpy(), frame["y"].median().to_numpy()


def plot_amount_dependence(
    frame: pd.DataFrame,
    bins: int = 12,
    title: str = "What the amount contributes, within each product",
) -> plt.Figure:
    """One panel per product: the amount's contribution against the amount.

    Small multiples rather than five series in one axes, and one colour rather than
    five. The palette validates three categorical slots, and the question is whether
    each curve rises and flattens or turns over — which is read panel against panel,
    not by tracing one line out of five.

    A shared y axis is what makes that comparison legal: panels on their own scales
    would show five identical shapes regardless of what the model does.

    Args:
        frame: Columns `product`, `amount` and `contribution`, one row per explained
            transaction.
        bins: Quantile bins of amount, within each product.
        title: Figure title.

    Returns:
        The figure, unsaved.

    Raises:
        ValueError: If the frame is empty — an empty grid of panels is not a figure
            that says nothing, it is a figure that looks like a result.
    """
    products = sorted(frame["product"].dropna().unique())
    if not products:
        raise ValueError("no products to draw: the dependence frame is empty")

    figure, panels = plt.subplots(
        1, len(products), figsize=(2.6 * len(products), 3.4), dpi=150, sharey=True
    )
    figure.patch.set_facecolor(SURFACE)

    for axes, product in zip(np.atleast_1d(panels), products, strict=True):
        style_axes(axes)
        rows = frame[frame["product"] == product]
        centres, medians = binned_median(
            rows["amount"].to_numpy(dtype="float64"),
            rows["contribution"].to_numpy(dtype="float64"),
            bins,
        )

        axes.axhline(0, color=GRIDLINE, linewidth=1.5, zorder=1)
        axes.plot(
            centres,
            medians,
            color=SERIES_COLOURS[0],
            linewidth=1.8,
            marker="o",
            markersize=3.5,
            zorder=3,
        )
        axes.set_xscale("log")
        # Decades only. Matplotlib labels log minor ticks by default, and at this width
        # five panels of them overlap into an unreadable band — the axis stops being an
        # axis and becomes texture.
        axes.xaxis.set_major_locator(LogLocator(base=10.0))
        axes.xaxis.set_major_formatter(ScalarFormatter())
        axes.xaxis.set_minor_formatter(NullFormatter())
        axes.set_title(f"{product}  ({len(rows):,})", color=INK_MUTED, fontsize=9, pad=6)
        axes.set_xlabel("amount, USD", color=INK_MUTED, fontsize=8)

    np.atleast_1d(panels)[0].set_ylabel("contribution to the log-odds", color=INK_MUTED, fontsize=9)
    figure.suptitle(title, color=INK, fontsize=11, x=0.02, ha="left", y=1.04)

    return figure
