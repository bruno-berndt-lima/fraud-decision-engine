"""Tests for the population stability index and its bins.

Hand-checkable counts for the index itself, where the formula is; small columns for
the binning, where the registered decisions are — ties, missing, and the categorical
path through `apply_categories`.
"""

import math

import numpy as np
import pandas as pd
import pytest

from fraud_engine.features.encoders import MISSING
from fraud_engine.models.train import OTHER, apply_categories
from fraud_engine.monitoring.psi import categorical_counts, numeric_counts, psi, quantile_edges

EPSILON = 1e-6


# ---- the index ------------------------------------------------------------------


def test_identical_distributions_score_zero():
    counts = np.array([10, 30, 60, 0])
    assert psi(counts, counts, EPSILON) == 0.0


def test_a_hand_computed_shift():
    """50/50 against 80/20: 0.3·ln(1.6) + 0.3·ln(2.5), which is 0.3·ln(4)."""
    assert psi([50, 50], [80, 20], 1e-12) == pytest.approx(0.3 * math.log(4), abs=1e-9)


def test_only_shares_matter_not_volume():
    assert psi([50, 50], [80, 20], EPSILON) == pytest.approx(psi([5, 5], [8000, 2000], EPSILON))


def test_it_is_symmetric():
    assert psi([10, 20, 70], [40, 40, 20], EPSILON) == pytest.approx(
        psi([40, 40, 20], [10, 20, 70], EPSILON)
    )


@pytest.mark.parametrize("seed", range(20))
def test_it_is_never_negative(seed):
    rng = np.random.default_rng(seed)
    assert psi(rng.integers(0, 100, 12), rng.integers(0, 100, 12), EPSILON) >= 0


def test_a_larger_shift_scores_higher():
    shifted = [psi([50, 50], [50 + k, 50 - k], EPSILON) for k in (5, 15, 30, 45)]
    assert shifted == sorted(shifted)
    assert len(set(shifted)) == len(shifted)


def test_a_bin_empty_on_both_sides_adds_nothing():
    assert psi([50, 50, 0], [80, 20, 0], EPSILON) == pytest.approx(psi([50, 50], [80, 20], EPSILON))


def test_a_bin_empty_on_one_side_is_finite():
    assert math.isfinite(psi([50, 50], [100, 0], EPSILON))


def test_misaligned_bins_are_refused():
    with pytest.raises(ValueError, match="line up"):
        psi([1, 2, 3], [1, 2], EPSILON)


def test_a_negative_count_is_refused():
    with pytest.raises(ValueError, match="negative"):
        psi([1, -1], [1, 1], EPSILON)


def test_an_empty_side_is_refused():
    with pytest.raises(ValueError, match="no rows"):
        psi([0, 0], [1, 1], EPSILON)


@pytest.mark.parametrize("epsilon", [0.0, -1e-6])
def test_an_epsilon_that_is_not_positive_is_refused(epsilon):
    with pytest.raises(ValueError, match="epsilon"):
        psi([1, 1], [1, 1], epsilon)


# ---- numeric bins -----------------------------------------------------------------


def test_ten_bins_on_continuous_values_hold_a_tenth_each():
    reference = pd.Series(np.random.default_rng(0).normal(size=10_000))
    edges = quantile_edges(reference, 10)
    counts = numeric_counts(reference, edges)

    assert len(edges) == 9
    assert counts[:-1].tolist() == [1_000] * 10


def test_ties_merge_edges_and_leave_fewer_bins():
    """A count column with most of its mass at zero cannot hold ten quantile bins."""
    reference = pd.Series([0.0] * 60 + [1.0] * 20 + [2.0] * 10 + [5.0] * 10)
    edges = quantile_edges(reference, 10)

    assert edges.tolist() == [0.0, 1.0, 2.0]
    assert numeric_counts(reference, edges).tolist() == [60, 20, 10, 10, 0]


def test_no_edge_falls_between_two_observed_values():
    """An interpolated edge between tied runs opens a bin nothing can land in."""
    reference = pd.Series([0.0] * 55 + [1.0] * 45)
    edges = quantile_edges(reference, 10)

    assert set(edges) <= set(reference)
    assert (numeric_counts(reference, edges)[: len(edges)] > 0).all()


def test_a_value_on_an_edge_falls_in_the_bin_it_closes():
    edges = np.array([1.0, 2.0])
    assert numeric_counts(pd.Series([1.0, 2.0, 2.5]), edges).tolist() == [1, 1, 1, 0]


def test_values_beyond_the_reference_fall_in_the_end_bins():
    edges = np.array([1.0, 2.0])
    assert numeric_counts(pd.Series([-50.0, 99.0, np.inf]), edges).tolist() == [1, 0, 2, 0]


def test_missing_is_the_last_bin_and_is_there_even_when_empty():
    edges = np.array([1.0])
    assert numeric_counts(pd.Series([0.0, 3.0]), edges).tolist() == [1, 1, 0]
    assert numeric_counts(pd.Series([0.0, np.nan, np.nan]), edges).tolist() == [1, 0, 2]


def test_a_field_that_stops_arriving_is_drift():
    """The reason missing is never imputed: the same values, a third of them gone."""
    rng = np.random.default_rng(0)
    reference = pd.Series(rng.normal(size=9_000))
    window = reference.copy()
    window.iloc[::3] = np.nan

    edges = quantile_edges(reference, 10)
    shifted = psi(numeric_counts(reference, edges), numeric_counts(window, edges), EPSILON)
    assert shifted > 0.25


def test_the_reference_is_cut_on_its_values_alone():
    with_nulls = pd.Series([1.0, 2.0, 3.0, 4.0, np.nan, np.nan])
    without = pd.Series([1.0, 2.0, 3.0, 4.0])
    assert np.array_equal(quantile_edges(with_nulls, 4), quantile_edges(without, 4))


def test_a_constant_reference_leaves_two_value_bins():
    edges = quantile_edges(pd.Series([7.0] * 50), 10)
    assert edges.tolist() == [7.0]
    assert numeric_counts(pd.Series([7.0, 8.0]), edges).tolist() == [1, 1, 0]


def test_an_all_null_reference_leaves_one_value_bin():
    edges = quantile_edges(pd.Series([np.nan] * 5), 10)
    assert edges.size == 0
    assert numeric_counts(pd.Series([3.0, np.nan]), edges).tolist() == [1, 1]


def test_a_boolean_column_is_binned_as_numbers():
    reference = pd.Series([True] * 30 + [False] * 70)
    counts = numeric_counts(reference, quantile_edges(reference, 10))
    assert counts[0] == 70
    assert counts[:-1].sum() == 100


def test_an_infinite_reference_is_refused():
    with pytest.raises(ValueError, match="infinity"):
        quantile_edges(pd.Series([1.0, np.inf], name="ratio"), 10)


def test_bins_below_one_are_refused():
    with pytest.raises(ValueError, match="at least one bin"):
        quantile_edges(pd.Series([1.0, 2.0]), 0)


def test_the_numeric_path_refuses_a_categorical():
    column = pd.Series(["a", "b"], dtype="category", name="ProductCD")
    with pytest.raises(ValueError, match="apply_categories"):
        quantile_edges(column, 10)
    with pytest.raises(ValueError, match="apply_categories"):
        numeric_counts(column, np.array([]))


# ---- categorical bins ---------------------------------------------------------------

VOCABULARY = pd.Index(["C", "H", "W", OTHER, MISSING], dtype=object)


def routed(values):
    frame = pd.DataFrame({"ProductCD": pd.Series(values, dtype="category")})
    return apply_categories(frame, {"ProductCD": VOCABULARY})["ProductCD"]


def test_counts_follow_the_vocabulary_order_with_zeros():
    assert categorical_counts(routed(["W", "W", "C"])).tolist() == [1, 0, 2, 0, 0]


def test_nulls_and_unseen_levels_land_in_different_bins():
    """An influx of new values must not read as a field going silent, or the reverse."""
    counts = categorical_counts(routed(["W", "Z", "Z", None]))
    assert counts[VOCABULARY.get_loc(OTHER)] == 2
    assert counts[VOCABULARY.get_loc(MISSING)] == 1


def test_an_unrouted_column_with_nulls_is_refused():
    raw = pd.Series(["W", None], dtype="category", name="ProductCD")
    with pytest.raises(ValueError, match="apply_categories"):
        categorical_counts(raw)


def test_a_non_categorical_is_refused():
    with pytest.raises(ValueError, match="not categorical"):
        categorical_counts(pd.Series([1.0, 2.0], name="C1"))


@pytest.mark.parametrize("bins", [2, 4, 8])
@pytest.mark.parametrize("seed", range(5))
def test_the_edges_are_the_empirical_quantiles(bins, seed):
    """Against numpy's own definition, on levels a float holds exactly."""
    values = np.random.default_rng(seed).integers(0, 30, size=997).astype("float64")
    levels = np.arange(1, bins) / bins
    expected = np.unique(np.quantile(values, levels, method="inverted_cdf"))
    assert np.array_equal(quantile_edges(pd.Series(values), bins), expected)


def test_fewer_rows_than_bins_still_cuts_inside_the_data():
    edges = quantile_edges(pd.Series([3.0, 1.0, 2.0]), 10)
    assert edges.tolist() == [1.0, 2.0, 3.0]
