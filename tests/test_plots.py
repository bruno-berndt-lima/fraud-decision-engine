"""Tests for the evaluation figures.

Structural, not pixel-level: that the right marks exist, that the reference
lines are drawn, and that the palette cap is enforced. Whether a figure *reads*
well is settled by rendering one and looking at it, which no assertion replaces.
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import precision_recall_curve

from fraud_engine.evaluation.plots import (
    SERIES_COLOURS,
    plot_pr_curve,
    plot_recall_at_capacity,
    save_figure,
)


@pytest.fixture
def data():
    """Labels, days and one weak-signal score over four days.

    A normal latent, not label + uniform noise: the additive-uniform trick
    leaves a band at the top of the range that only positives can reach, so
    precision hits 1.0 and every scaling assertion becomes vacuous. Real
    scorers overlap everywhere.
    """
    rng = np.random.default_rng(0)
    n = 800
    y_true = pd.Series((rng.random(n) < 0.05).astype(int))
    score = pd.Series(y_true * 0.8 + rng.normal(size=n), index=y_true.index)
    days = pd.Series(np.arange(n) % 4, index=y_true.index)
    return y_true, score, days


@pytest.fixture(autouse=True)
def _close_figures():
    yield
    plt.close("all")


# ---- plot_pr_curve -----------------------------------------------------------


def test_pr_curve_draws_one_line_per_series(data):
    y_true, score, _ = data
    axes = plot_pr_curve(y_true, {"a": score, "b": score * 2}).axes[0]
    # Two series plus the base-rate reference line.
    assert len(axes.lines) == 3


def test_pr_curve_draws_the_base_rate_floor(data):
    """Without the floor the curve cannot be read at all."""
    y_true, score, _ = data
    axes = plot_pr_curve(y_true, {"model": score}).axes[0]

    floors = [line for line in axes.lines if np.allclose(line.get_ydata(), y_true.mean())]
    assert floors, "no horizontal line at the base rate"
    assert any(f"{y_true.mean():.3f}" in text.get_text() for text in axes.texts)


def test_pr_curve_is_drawn_as_steps(data):
    """The straight path between PR points is unreachable by any threshold —
    the same reason pr_auc is average precision and not a trapezoid."""
    y_true, score, _ = data
    axes = plot_pr_curve(y_true, {"model": score}).axes[0]
    series = [line for line in axes.lines if line.get_color() == SERIES_COLOURS[0]]
    assert series and series[0].get_drawstyle().startswith("steps")


def test_pr_curve_scales_to_the_data(data):
    """A fixed 0-1 axis flattens every curve at a 3.5% base rate.

    Asserted as a relation rather than an absolute: a weaker scorer must get a
    tighter axis than a stronger one. Pinning a number here would just restate
    the formula.
    """
    y_true, weak, _ = data
    strong = pd.Series(y_true * 5.0 + weak, index=y_true.index)

    weak_top = plot_pr_curve(y_true, {"model": weak}).axes[0].get_ylim()[1]
    strong_top = plot_pr_curve(y_true, {"model": strong}).axes[0].get_ylim()[1]

    assert weak_top < strong_top
    assert strong_top <= 1.0, "precision cannot exceed 1, so neither should the axis"


def test_pr_curve_drops_sklearns_synthetic_endpoint(data):
    """sklearn appends (precision=1, recall=0), which no threshold produces.

    Asserted by count, not by absence of a 1.0: precision genuinely reaches 1
    at low recall when the top-ranked rows are all fraud.
    """
    y_true, score, _ = data
    raw_precision, raw_recall, _ = precision_recall_curve(y_true, score)
    assert (raw_precision[-1], raw_recall[-1]) == (1.0, 0.0)

    axes = plot_pr_curve(y_true, {"model": score}).axes[0]
    series = next(line for line in axes.lines if line.get_color() == SERIES_COLOURS[0])
    assert len(series.get_ydata()) == len(raw_precision) - 1


def test_pr_curve_labels_its_axes(data):
    y_true, score, _ = data
    axes = plot_pr_curve(y_true, {"model": score}, title="A title").axes[0]
    assert axes.get_xlabel() == "Recall"
    assert axes.get_ylabel() == "Precision"
    assert axes.get_title(loc="left") == "A title"


# ---- plot_recall_at_capacity -------------------------------------------------


def test_capacity_curve_draws_the_perfect_ranker(data):
    """Without the ceiling a reader cannot tell a weak model from a tight
    constraint."""
    y_true, score, days = data
    axes = plot_recall_at_capacity(y_true, {"model": score}, days).axes[0]

    # One series, plus the oracle ceiling and the capacity rule (axvline is a
    # Line2D as well).
    assert len(axes.lines) == 3
    assert any("perfect ranker" in text.get_text() for text in axes.texts)


def test_capacity_curve_marks_the_committed_capacity(data):
    y_true, score, days = data
    axes = plot_recall_at_capacity(y_true, {"model": score}, days, operating_capacity=0.02).axes[0]
    assert any("2.0%" in text.get_text() for text in axes.texts)


def test_capacity_curve_labels_series_directly_rather_than_in_a_legend(data):
    """Identity beside the mark it names, and it keeps the top-left free."""
    y_true, score, days = data
    axes = plot_recall_at_capacity(y_true, {"alpha": score, "beta": score * 2}, days).axes[0]

    assert axes.get_legend() is None
    labelled = {text.get_text() for text in axes.texts}
    assert {"alpha", "beta"} <= labelled


def test_capacity_curve_never_exceeds_the_perfect_ranker(data):
    """A model catching more than an oracle would mean the metric is broken."""
    y_true, score, days = data
    axes = plot_recall_at_capacity(y_true, {"model": score}, days).axes[0]

    oracle, model = (line.get_ydata() for line in axes.lines[:2])
    assert np.all(model <= oracle + 1e-12)


def test_capacity_curve_keeps_ticks_inside_the_data_range(data):
    """The right margin holds direct labels; a gridline there cuts through them."""
    y_true, score, days = data
    capacities = [0.01, 0.02, 0.03]
    axes = plot_recall_at_capacity(y_true, {"model": score}, days, capacities=capacities).axes[0]

    assert max(axes.get_xticks()) <= max(capacities)
    assert axes.get_xlim()[1] > max(capacities)


# ---- shared guards -----------------------------------------------------------


@pytest.mark.parametrize("plot", [plot_pr_curve, plot_recall_at_capacity])
def test_rejects_more_series_than_the_palette_validates(data, plot):
    """A fourth slot puts orange and yellow on screen together, which fails
    colour-vision separation. Fold or facet instead of adding a colour."""
    y_true, score, days = data
    too_many = dict.fromkeys(("a", "b", "c", "d"), score)

    with pytest.raises(ValueError, match="categorical slots"):
        plot(y_true, too_many, days) if plot is plot_recall_at_capacity else plot(y_true, too_many)


@pytest.mark.parametrize("plot", [plot_pr_curve, plot_recall_at_capacity])
def test_rejects_no_series(data, plot):
    y_true, _, days = data
    with pytest.raises(ValueError, match="no series"):
        plot(y_true, {}, days) if plot is plot_recall_at_capacity else plot(y_true, {})


def test_series_take_palette_slots_in_fixed_order(data):
    """Never cycled — colour follows the entity, not its position in a filter."""
    y_true, score, _ = data
    axes = plot_pr_curve(y_true, {"a": score, "b": score * 2, "c": score * 3}).axes[0]

    used = [line.get_color() for line in axes.lines if line.get_color() in SERIES_COLOURS]
    assert used == list(SERIES_COLOURS)


# ---- save_figure -------------------------------------------------------------


def test_save_figure_writes_and_creates_parents(data, tmp_path):
    y_true, score, _ = data
    path = save_figure(plot_pr_curve(y_true, {"model": score}), tmp_path / "nested" / "pr.png")

    assert path.exists()
    assert path.stat().st_size > 0


def test_save_figure_closes_the_figure(data, tmp_path):
    """A report run producing a dozen figures otherwise holds them all open."""
    y_true, score, _ = data
    figure = plot_pr_curve(y_true, {"model": score})
    save_figure(figure, tmp_path / "pr.png")

    assert not plt.fignum_exists(figure.number)
