"""Tests for the decomposition everything in Phase 07 is read from.

Boosters here are trained on a few hundred synthetic rows, as in
`test_train_model`. The point is never the contribution — it is that the array is
the shape the module claims, that the guards fire on the failures they name, and
that the one property the phase rests on is checked rather than assumed.
"""

from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
import pytest

from fraud_engine.explain import contributions as explain
from fraud_engine.models.train import CONTRACT_PARAMS, apply_categories, fit_categories, to_dataset

FEATURES = ["signal", "noise", "brand"]
EXPLAIN_CFG = {"sample_rows": 50, "seed": 0}


def make_matrix(rows: int = 300, seed: int = 0) -> pd.DataFrame:
    """A matrix with a planted signal, so a booster has something to decompose."""
    rng = np.random.default_rng(seed)
    signal = rng.random(rows)
    return pd.DataFrame(
        {
            "TransactionID": range(rows),
            "TransactionDT": np.arange(rows) * 60,
            "isFraud": (signal > 0.8).astype(int),
            "day": 1,
            "signal": signal,
            "noise": rng.random(rows),
            "brand": pd.Categorical(rng.choice(["visa", "amex"], rows)),
        }
    )


@pytest.fixture
def matrix() -> pd.DataFrame:
    frame = make_matrix()
    return apply_categories(frame, fit_categories(frame, min_rows=1))


@pytest.fixture
def booster(matrix: pd.DataFrame) -> lgb.Booster:
    dataset = to_dataset(matrix, FEATURES)
    return lgb.train({**CONTRACT_PARAMS, "seed": 0, "num_leaves": 4}, dataset, num_boost_round=20)


TIERS = {"signal": "tier_1", "noise": "tier_0", "brand": "tier_2"}


# ------------------------------------------------------------------------------
# contributions
# ------------------------------------------------------------------------------


def test_one_column_per_feature_plus_the_base_value(booster, matrix):
    contribs = explain.contributions(booster, matrix[FEATURES])

    assert contribs.shape == (len(matrix), len(FEATURES) + 1)


def test_the_base_value_is_the_same_for_every_row(booster, matrix):
    """The property that catches a misalignment, which additivity cannot."""
    base = explain.contributions(booster, matrix[FEATURES])[:, -1]

    assert (base == base[0]).all()


def test_contributions_decompose_the_raw_margin_not_the_probability(booster, matrix):
    contribs = explain.contributions(booster, matrix[FEATURES])
    margin = booster.predict(matrix[FEATURES], raw_score=True)
    probability = booster.predict(matrix[FEATURES])

    total = contribs.sum(axis=1)
    assert np.abs(total - margin).max() < explain.ADDITIVITY_TOLERANCE
    assert np.abs(total - probability).max() > explain.ADDITIVITY_TOLERANCE


# ------------------------------------------------------------------------------
# check_additive
# ------------------------------------------------------------------------------


def test_a_faithful_decomposition_passes_and_reports_its_worst_row():
    contribs = np.array([[1.0, -2.0, 0.5], [0.25, 0.25, -1.0]])
    margin = contribs.sum(axis=1)

    assert explain.check_additive(contribs, margin) == 0.0


def test_a_decomposition_of_something_else_is_refused():
    contribs = np.array([[1.0, -2.0, 0.5]])

    with pytest.raises(ValueError, match="do not sum to the raw margin"):
        explain.check_additive(contribs, np.array([7.0]))


def test_the_deviation_is_measured_against_the_contribution_mass():
    """Two rows off by the same absolute amount are not equally wrong."""
    contribs = np.array([[1.0, 1.0], [100.0, 100.0]])
    margin = contribs.sum(axis=1) + 0.01

    worst = explain.check_additive(contribs, margin)
    assert worst == pytest.approx(0.01 / 2.0)


# ------------------------------------------------------------------------------
# check_base_value
# ------------------------------------------------------------------------------


def test_the_base_value_is_handed_back_to_be_recorded():
    contribs = np.array([[1.0, 2.0, -0.5], [3.0, -1.0, -0.5]])

    assert explain.check_base_value(contribs) == -0.5


def test_a_decomposition_named_one_place_out_is_refused():
    """The shift moves between two columns, so the rows still sum to the same margin.

    That is what a misalignment actually looks like, and it is invisible to
    `check_additive` — which is the whole reason this second guard exists.
    """
    contribs = np.array([[1.0, 2.0, -0.5], [3.0, -1.0, -0.5]])
    shift = np.array([0.0, 1.0])
    contribs[:, 0] -= shift
    contribs[:, -1] += shift

    assert explain.check_additive(contribs, contribs.sum(axis=1)) == 0.0
    with pytest.raises(ValueError, match="base value column is not constant"):
        explain.check_base_value(contribs)


# ------------------------------------------------------------------------------
# take_sample
# ------------------------------------------------------------------------------


def test_a_split_no_larger_than_the_sample_is_taken_whole(matrix):
    assert explain.take_sample(matrix, len(matrix) + 1, seed=0) is matrix


def test_the_sample_keeps_the_order_the_split_already_had(matrix):
    sample = explain.take_sample(matrix, 50, seed=0)

    assert sample["TransactionDT"].is_monotonic_increasing


def test_the_same_seed_draws_the_same_rows(matrix):
    first = explain.take_sample(matrix, 50, seed=0)["TransactionID"].tolist()
    second = explain.take_sample(matrix, 50, seed=0)["TransactionID"].tolist()

    assert first == second


def test_a_different_seed_draws_different_rows(matrix):
    first = explain.take_sample(matrix, 50, seed=0)["TransactionID"].tolist()
    other = explain.take_sample(matrix, 50, seed=1)["TransactionID"].tolist()

    assert first != other


# ------------------------------------------------------------------------------
# ranking
# ------------------------------------------------------------------------------


def test_features_are_ordered_by_how_much_of_the_margin_they_move():
    contribs = np.array([[0.1, -3.0, 1.0, 0.0], [0.1, 3.0, -1.0, 0.0]])

    ranked = explain.ranking(contribs, FEATURES, TIERS)

    assert ranked["feature"].tolist() == ["noise", "brand", "signal"]


def test_a_feature_that_pushes_both_ways_is_still_used():
    """`mean_signed` cancelling while `mean_abs` does not is the threshold signature."""
    contribs = np.array([[0.0, -3.0, 0.0, 0.0], [0.0, 3.0, 0.0, 0.0]])

    ranked = explain.ranking(contribs, FEATURES, TIERS).set_index("feature")

    assert ranked.loc["noise", "mean_abs"] == 3.0
    assert ranked.loc["noise", "mean_signed"] == 0.0


def test_an_array_without_a_base_value_column_is_refused():
    with pytest.raises(ValueError, match="base value column"):
        explain.ranking(np.zeros((2, len(FEATURES))), FEATURES, TIERS)


# ------------------------------------------------------------------------------
# tier_shares
# ------------------------------------------------------------------------------


def test_the_tier_shares_account_for_all_of_the_contribution():
    contribs = np.array([[1.0, 2.0, 1.0, 0.0]])

    shares = explain.tier_shares(explain.ranking(contribs, FEATURES, TIERS))

    assert sum(shares.values()) == pytest.approx(1.0)
    assert shares["tier_0"] == pytest.approx(0.5)


# ------------------------------------------------------------------------------
# feature_tiers
# ------------------------------------------------------------------------------


def test_a_feature_in_no_serving_tier_stops_the_run(monkeypatch):
    """§4's shares would otherwise omit a column without saying so."""
    monkeypatch.setattr(
        explain, "resolve_tiers", lambda _: {"tier_1": ("signal",), "keys": ("isFraud",)}
    )

    with pytest.raises(ValueError, match="features in no serving tier"):
        explain.feature_tiers("unused", FEATURES)


def test_the_keys_are_not_offered_as_a_tier(monkeypatch):
    monkeypatch.setattr(
        explain,
        "resolve_tiers",
        lambda _: {"tier_1": tuple(FEATURES), "keys": ("isFraud", "signal")},
    )

    assert explain.feature_tiers("unused", FEATURES) == dict.fromkeys(FEATURES, "tier_1")


# ------------------------------------------------------------------------------
# write_contributions
# ------------------------------------------------------------------------------


def test_the_written_file_carries_the_keys_beside_every_feature(tmp_path: Path):
    contribs = np.arange(8, dtype="float64").reshape(2, 4)
    keys = pd.DataFrame({"TransactionID": [10, 11], "day": [1, 1], "isFraud": [0, 1]})
    path = tmp_path / "split.parquet"

    explain.write_contributions(contribs, keys, FEATURES, path)

    written = pd.read_parquet(path)
    assert written.columns.tolist() == [*keys.columns, *FEATURES, explain.BASE_VALUE]
    assert written["TransactionID"].tolist() == [10, 11]


def test_the_contributions_survive_the_round_trip_undiminished(tmp_path: Path):
    """Float64 or the additivity the phase rests on cannot be re-checked."""
    contribs = np.array([[0.1234567890123456, -1.0, 2.0, 0.5]])
    keys = pd.DataFrame({"TransactionID": [1], "day": [1], "isFraud": [0]})
    path = tmp_path / "split.parquet"

    explain.write_contributions(contribs, keys, FEATURES, path)

    assert pd.read_parquet(path)[FEATURES[0]].iloc[0] == contribs[0, 0]


# ------------------------------------------------------------------------------
# explain_split
# ------------------------------------------------------------------------------


def test_a_split_is_explained_only_as_far_as_the_sample(booster, matrix):
    explained = explain.explain_split(booster, matrix, FEATURES, EXPLAIN_CFG)

    assert len(explained.rows) == EXPLAIN_CFG["sample_rows"]
    assert explained.contributions.shape == (EXPLAIN_CFG["sample_rows"], len(FEATURES) + 1)


def test_the_decomposition_carries_what_the_record_states(booster, matrix):
    explained = explain.explain_split(booster, matrix, FEATURES, EXPLAIN_CFG)

    assert explained.base_value == explained.contributions[0, -1]
    assert explained.deviation < explain.ADDITIVITY_TOLERANCE
