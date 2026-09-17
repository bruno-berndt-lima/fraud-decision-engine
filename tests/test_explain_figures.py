"""Tests for the Phase 07 figures: the selection rules and the frames they read.

Nothing here asserts what a figure looks like. What is asserted is that a case is
selected by its rule rather than by whichever row happens to be first, that a frame
misaligned against its contributions is refused, and that a serving tier with no colour
stops the run instead of borrowing one.
"""

import matplotlib
import numpy as np
import pandas as pd
import pytest

matplotlib.use("Agg")

from fraud_engine.evaluation.cost import ALLOW, BLOCK
from fraud_engine.explain import figures, plots
from fraud_engine.explain.contributions import BASE_VALUE

COLUMNS = ["amount_feature", "card_feature"]


def contributions_frame(rows: int = 4) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "TransactionID": range(rows),
            "day": 1,
            "isFraud": 0,
            "amount_feature": np.linspace(-1, 1, rows),
            "card_feature": np.linspace(1, -1, rows),
            BASE_VALUE: -2.0,
        }
    )


def values_frame(rows: int = 4) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "amount_feature": np.arange(float(rows)),
            "card_feature": pd.Categorical(["visa", "amex"] * (rows // 2)),
        }
    )


# ------------------------------------------------------------------------------
# as_numeric
# ------------------------------------------------------------------------------


def test_categoricals_become_codes_so_the_colour_axis_has_something_to_read():
    numeric = figures.as_numeric(values_frame())

    assert numeric.dtype == np.float64
    assert set(np.unique(numeric[:, 1])) == {0.0, 1.0}


# ------------------------------------------------------------------------------
# build_explanation
# ------------------------------------------------------------------------------


def test_the_explanation_carries_the_base_value_apart_from_the_features():
    explanation = figures.build_explanation(contributions_frame(), values_frame(), COLUMNS)

    # `.values` here is shap's Explanation attribute, not a pandas frame's.
    assert explanation.values.shape == (4, 2)  # noqa: PD011
    assert (explanation.base_values == -2.0).all()
    assert list(explanation.feature_names) == COLUMNS


def test_rows_that_are_not_the_same_rows_are_refused():
    with pytest.raises(ValueError, match="same rows, in order"):
        figures.build_explanation(contributions_frame(4), values_frame(2), COLUMNS)


# ------------------------------------------------------------------------------
# select_cases
# ------------------------------------------------------------------------------


def decided_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "TransactionID": [1, 2, 3, 4, 5],
            "isFraud": [1, 1, 1, 0, 1],
            "amount": [10.0, 50.0, 900.0, 40.0, 20.0],
            "fallback": [BLOCK, BLOCK, BLOCK, BLOCK, ALLOW],
        }
    )


def test_the_typical_case_is_the_median_amount_not_the_most_convincing():
    cases = figures.select_cases(decided_frame(), pd.Index([1, 2, 3, 4, 5]))

    assert cases["true_positive"] == 2


def test_the_high_value_catch_is_the_largest_blocked_fraud():
    cases = figures.select_cases(decided_frame(), pd.Index([1, 2, 3, 4, 5]))

    assert cases["high_value_catch"] == 3


def test_an_allowed_fraud_is_not_a_catch():
    """Row 5 is fraud and was allowed; nothing about it is a decision to explain."""
    cases = figures.select_cases(decided_frame(), pd.Index([1, 2, 3, 4, 5]))

    assert 5 not in cases.values()


def test_only_explained_rows_can_be_drawn():
    cases = figures.select_cases(decided_frame(), pd.Index([1, 2, 4]))

    assert cases["high_value_catch"] == 2


def test_a_case_with_no_eligible_row_stops_the_run():
    frame = decided_frame()
    frame["isFraud"] = 1

    with pytest.raises(ValueError, match="false_positive"):
        figures.select_cases(frame, pd.Index([1, 2, 3, 4, 5]))


# ------------------------------------------------------------------------------
# binned_median
# ------------------------------------------------------------------------------


def test_bins_are_quantiles_so_a_skewed_axis_is_not_one_bin():
    x = np.array([1.0, 2.0, 3.0, 1000.0])
    centres, medians = plots.binned_median(x, np.array([0.0, 1.0, 2.0, 3.0]), bins=2)

    assert len(centres) == 2
    assert medians[0] < medians[-1]


def test_an_axis_of_one_repeated_value_collapses_to_a_single_point():
    """A round amount everybody uses has no quantiles to split on."""
    centres, _ = plots.binned_median(np.full(5, 100.0), np.arange(5.0), bins=4)

    assert len(centres) == 1


def test_nothing_to_bin_returns_nothing():
    centres, medians = plots.binned_median(np.empty(0), np.empty(0), bins=4)

    assert len(centres) == 0 and len(medians) == 0


# ------------------------------------------------------------------------------
# plot_tier_ranking
# ------------------------------------------------------------------------------


def ranked_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "feature": ["C13", "TransactionAmt", "freq_card1"],
            "tier": ["tier_0", "tier_1", "tier_2"],
            "mean_abs": [0.8, 0.3, 0.2],
        }
    )


def test_a_bar_per_feature_shown():
    figure = plots.plot_tier_ranking(ranked_frame(), top_n=2)

    assert len(figure.axes[0].patches) == 2


def test_the_two_tiers_that_answer_one_serving_question_share_a_colour():
    groups = plots.TIER_GROUPS

    assert groups["tier_2"][1] == groups["tier_3"][1]


def test_a_tier_with_no_colour_group_stops_the_run():
    frame = ranked_frame()
    frame.loc[0, "tier"] = "tier_9"

    with pytest.raises(ValueError, match="no colour group"):
        plots.plot_tier_ranking(frame)


# ------------------------------------------------------------------------------
# plot_amount_dependence
# ------------------------------------------------------------------------------


def test_one_panel_per_product():
    frame = pd.DataFrame(
        {
            "product": ["C", "C", "W", "W"],
            "amount": [10.0, 20.0, 30.0, 40.0],
            "contribution": [0.1, 0.2, -0.1, -0.2],
        }
    )

    assert len(plots.plot_amount_dependence(frame, bins=2).axes) == 2


def test_the_panels_share_a_y_axis_so_the_shapes_can_be_compared():
    frame = pd.DataFrame(
        {
            "product": ["C", "C", "W", "W"],
            "amount": [10.0, 20.0, 30.0, 40.0],
            "contribution": [0.1, 0.2, -5.0, -6.0],
        }
    )

    panels = plots.plot_amount_dependence(frame, bins=2).axes
    assert panels[0].get_ylim() == panels[1].get_ylim()


def test_an_empty_frame_is_refused_rather_than_drawn():
    empty = pd.DataFrame({"product": [], "amount": [], "contribution": []})

    with pytest.raises(ValueError, match="dependence frame is empty"):
        plots.plot_amount_dependence(empty)
