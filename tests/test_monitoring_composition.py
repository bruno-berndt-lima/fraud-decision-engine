"""Tests for E5, the identity strata.

The decomposition is checked against the two cases where its answer is known exactly:
a validation set that differs from train only in mix, where composition must explain
the whole gap, and one that differs only in the model's ranking, where it must explain
none of it.
"""

import numpy as np
import pandas as pd
import pytest

from fraud_engine.evaluation.metrics import pr_auc
from fraud_engine.models.train import LABEL
from fraud_engine.monitoring.composition import (
    POOLED,
    SPLITS,
    STRATUM,
    cells,
    decompose,
    identity_share,
    mix_weights,
    weighted_pr_auc,
)


def stratum(n, identity, fraud_rate, signal, seed):
    """Rows of one stratum: labels, and a score that separates them by `signal`."""
    rng = np.random.default_rng(seed)
    y = (rng.random(n) < fraud_rate).astype(int)
    return pd.DataFrame(
        {
            LABEL: y,
            "score": rng.random(n) + signal * y,
            STRATUM: identity,
            "day": rng.integers(1, 21, n),
        }
    )


def population(identity_rows=3_000, other_rows=7_000, seed=0):
    """Two strata that the model ranks differently, as it does in this data."""
    return pd.concat(
        [
            stratum(identity_rows, True, 0.07, 0.6, seed),
            stratum(other_rows, False, 0.02, 0.3, seed + 1),
        ],
        ignore_index=True,
    )


# ---- the weighted metric --------------------------------------------------------


@pytest.mark.parametrize("seed", range(5))
def test_at_unit_weights_it_is_the_harness_pr_auc(seed):
    frame = population(seed=seed)
    ones = pd.Series(1.0, index=frame.index)
    assert weighted_pr_auc(frame[LABEL], frame["score"], ones) == pr_auc(
        frame[LABEL], frame["score"]
    )


def test_an_integer_weight_counts_a_row_that_many_times():
    frame = population(identity_rows=300, other_rows=700)
    weight = pd.Series(np.where(frame[STRATUM], 3.0, 1.0), index=frame.index)
    tripled = pd.concat([frame] + [frame[frame[STRATUM]]] * 2, ignore_index=True)

    assert weighted_pr_auc(frame[LABEL], frame["score"], weight) == pytest.approx(
        pr_auc(tripled[LABEL], tripled["score"])
    )


@pytest.mark.parametrize("bad", [0.0, -1.0, np.inf, np.nan])
def test_a_weight_that_is_not_finite_and_positive_is_refused(bad):
    frame = population(identity_rows=50, other_rows=50)
    weight = pd.Series(1.0, index=frame.index)
    weight.iloc[0] = bad
    with pytest.raises(ValueError, match="finite and positive"):
        weighted_pr_auc(frame[LABEL], frame["score"], weight)


def test_a_single_class_is_refused():
    frame = population(identity_rows=50, other_rows=50).assign(**{LABEL: 0})
    with pytest.raises(ValueError, match="both classes"):
        weighted_pr_auc(frame[LABEL], frame["score"], pd.Series(1.0, index=frame.index))


# ---- the mix --------------------------------------------------------------------


def test_the_weights_give_validation_train_s_identity_share():
    validation = population(identity_rows=1_700, other_rows=8_300)
    weights = mix_weights(validation, train_share=0.29)

    assert np.average(validation[STRATUM], weights=weights) == pytest.approx(0.29)


def test_the_weights_are_constant_within_a_stratum():
    validation = population(identity_rows=1_700, other_rows=8_300)
    weights = mix_weights(validation, train_share=0.29)
    assert weights.groupby(validation[STRATUM]).nunique().eq(1).all()


def test_an_equal_mix_leaves_every_weight_at_one():
    validation = population()
    assert np.allclose(mix_weights(validation, identity_share(validation)), 1.0)


@pytest.mark.parametrize("share", [0.0, 1.0])
def test_a_side_holding_one_stratum_is_refused(share):
    with pytest.raises(ValueError, match="one stratum"):
        mix_weights(population(), train_share=share)


def test_a_validation_holding_one_stratum_is_refused():
    only_identity = population()[lambda frame: frame[STRATUM]]
    with pytest.raises(ValueError, match="one stratum"):
        mix_weights(only_identity, train_share=0.3)


# ---- the decomposition ------------------------------------------------------------


def test_a_mix_shift_alone_is_all_composition():
    """Validation is train with the no-identity rows repeated: same model, new mix."""
    train = population()
    validation = pd.concat([train] + [train[~train[STRATUM]]] * 2, ignore_index=True)

    result = decompose(train, validation)

    assert result["pooled_gap"] > 0
    assert result["reweighted_pr_auc"] == pytest.approx(result["train_pr_auc"])
    assert result["share_of_gap"] == pytest.approx(1.0)
    assert result["remainder"] == pytest.approx(0.0, abs=1e-12)


def test_a_ranking_change_alone_is_no_composition():
    """Same mix on both sides; the model separates validation less well in each stratum."""
    train = population(seed=0)
    validation = pd.concat(
        [stratum(3_000, True, 0.07, 0.2, 10), stratum(7_000, False, 0.02, 0.1, 11)],
        ignore_index=True,
    )

    result = decompose(train, validation)

    assert result["pooled_gap"] > 0
    assert result["compositional"] == pytest.approx(0.0, abs=1e-12)
    assert result["share_of_gap"] == pytest.approx(0.0, abs=1e-12)


def test_the_parts_add_up_to_the_gap():
    result = decompose(population(seed=0), population(identity_rows=1_500, seed=5))
    assert result["compositional"] + result["remainder"] == pytest.approx(result["pooled_gap"])


def test_no_gap_leaves_no_share():
    train = population(seed=0)
    result = decompose(train, train)
    assert result["pooled_gap"] == 0
    assert result["share_of_gap"] is None


# ---- the table --------------------------------------------------------------------


def scored_splits():
    parts = [
        population(seed=0).assign(split="train"),
        population(identity_rows=1_800, seed=3).assign(split="val_fit"),
        population(identity_rows=1_700, seed=6).assign(split="val_cal"),
    ]
    return pd.concat(parts, ignore_index=True)


def test_the_table_holds_every_side_and_stratum():
    table = cells(scored_splits(), capacities=[0.01])

    assert set(table) == {*SPLITS, POOLED}
    for side in table.values():
        assert set(side) == {"all", "identity", "no_identity"}


def test_the_strata_partition_each_side():
    table = cells(scored_splits(), capacities=[0.01])
    for side in table.values():
        assert side["identity"]["n"] + side["no_identity"]["n"] == side["all"]["n"]


def test_the_pooled_side_is_both_validation_slices():
    table = cells(scored_splits(), capacities=[0.01])
    assert table[POOLED]["all"]["n"] == table["val_fit"]["all"]["n"] + table["val_cal"]["all"]["n"]


def test_test_is_never_among_the_splits():
    """E5's first constraint: test's coverage stays out of every figure."""
    assert "test" not in SPLITS
