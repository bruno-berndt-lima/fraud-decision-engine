"""Tests for entity aggregates.

The arithmetic is a groupby and a division. What needs pinning is the shrinkage
— which exists to stop two degenerate cases that would otherwise produce
plausible-looking garbage — and the same fit/apply boundary `encoders` has.

The degenerate cases, both of which shrinkage has to make impossible:
a card seen once has a mean equal to its only amount, so its z-score is exactly
zero and it looks perfectly typical on no evidence; and a card whose amounts are
all identical has zero spread, so every z-score against it is a division by zero.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from fraud_engine.features.aggregations import (
    COLUMNS,
    ENTITY_COLUMNS,
    GLOBAL,
    add_amount_stats,
    apply_amount_stats,
    entity_column,
    fit_amount_stats,
    write_tables,
)
from fraud_engine.features.encoders import MISSING

CFG = {"prior_strength": 10}


def frame(card1, amounts, split=None) -> pd.DataFrame:
    """Rows carrying every entity column; only `card1` is given structure."""
    built = pd.DataFrame(
        {
            "card1": card1,
            "TransactionAmt": [float(amount) for amount in amounts],
            "split": split if split is not None else ["train"] * len(card1),
        }
    )
    for column in ENTITY_COLUMNS:
        if column not in built:
            built[column] = "x"
    return built


# ------------------------------------------------------------------------------
# fit_amount_stats — shrinkage
# ------------------------------------------------------------------------------
def test_a_well_evidenced_entity_keeps_its_own_mean():
    """Fifty transactions against a prior worth ten: the entity should dominate."""
    rows = frame(["a"] * 50 + ["b"] * 50, [10.0] * 50 + [1000.0] * 50)

    table = fit_amount_stats(rows, ("card1",), 10)["card1"]

    assert table["mean"]["a"] == pytest.approx(10 + (505 - 10) * 10 / 60, rel=1e-6)
    assert table["mean"]["a"] < 100


def test_an_entity_seen_once_is_pulled_almost_to_the_global_mean():
    """Otherwise its mean is its own amount, its z-score is exactly zero, and it
    reads as perfectly typical on a single observation."""
    rows = frame(["a"] * 50 + ["lonely"], [100.0] * 50 + [5000.0])

    table = fit_amount_stats(rows, ("card1",), 10)["card1"]
    prior = rows["TransactionAmt"].mean()

    assert abs(table["mean"]["lonely"] - prior) < abs(table["mean"]["lonely"] - 5000.0)


def test_an_entity_with_no_spread_still_gets_a_positive_one():
    """The division-by-zero this family would otherwise carry: identical amounts
    give an observed standard deviation of zero."""
    rows = frame(["flat"] * 20 + ["varied"] * 20, [50.0] * 20 + list(range(20)))

    table = fit_amount_stats(rows, ("card1",), 10)["card1"]

    assert table["std"]["flat"] > 0


def test_null_is_an_entity_of_its_own():
    rows = frame([None, None, "a", "a"], [10.0, 20.0, 500.0, 600.0])

    assert MISSING in fit_amount_stats(rows, ("card1",), 10)["card1"].index


def test_the_global_row_carries_the_unshrunk_training_statistics():
    """It is the fallback for entities train never saw, so it must describe the
    window itself rather than a shrunken version of it."""
    rows = frame(["a", "a", "b", "b"], [10.0, 20.0, 30.0, 40.0])

    table = fit_amount_stats(rows, ("card1",), 10)["card1"]

    assert table["mean"][GLOBAL] == pytest.approx(25.0)
    assert table["std"][GLOBAL] == pytest.approx(rows["TransactionAmt"].std())


# ------------------------------------------------------------------------------
# apply_amount_stats — the fit/apply boundary
# ------------------------------------------------------------------------------
def test_validation_rows_do_not_move_the_fitted_statistics():
    """The fit is confined to the training window; the transform is not."""
    small = frame(
        ["a"] * 6 + ["a", "a"],
        [90.0, 95.0, 100.0, 105.0, 110.0, 115.0, 1.0, 9000.0],
        ["train"] * 6 + ["val_fit"] * 2,
    )
    padded = pd.concat(
        [small, frame(["a"] * 200, [9000.0] * 200, ["val_fit"] * 200)], ignore_index=True
    )

    assert add_amount_stats(small, CFG)["amt_z_card1"].iloc[0] == pytest.approx(
        add_amount_stats(padded, CFG)["amt_z_card1"].iloc[0]
    )


def test_an_unseen_entity_is_scored_against_the_training_window():
    rows = frame(
        ["a"] * 6 + ["stranger"],
        [90.0, 95.0, 100.0, 105.0, 110.0, 115.0, 100.0],
        ["train"] * 6 + ["val_fit"],
    )

    scored = add_amount_stats(rows, CFG)
    tables = fit_amount_stats(rows[rows["split"] == "train"], ("card1",), 10)["card1"]

    expected = (100.0 - tables["mean"][GLOBAL]) / tables["std"][GLOBAL]
    assert scored["amt_z_card1"].iloc[-1] == pytest.approx(expected, rel=1e-5)


def test_the_absolute_deviation_discards_only_the_sign():
    rows = frame(["a"] * 10, [10, 20, 30, 40, 50, 60, 70, 80, 90, 1000])

    scored = add_amount_stats(rows, CFG)

    assert (scored["amt_absz_card1"] == scored["amt_z_card1"].abs()).all()
    assert (scored["amt_z_card1"] < 0).any(), "a signed z-score that never goes negative is not one"


def test_the_family_is_null_free_and_finite():
    """`build_features` promises families make their own fill decisions, and an
    infinity would survive StandardScaler to poison the whole fit."""
    # 'a' has zero spread of its own, and one row's entity is null: both paths
    # that would otherwise reach a division by zero or a missing lookup.
    rows = frame(["a", "a", None, "b", "b", "b"], [10.0, 10.0, 10.0, 10.0, 50.0, 900.0])

    scored = add_amount_stats(rows, CFG)

    assert scored[list(COLUMNS)].notna().all().all()
    assert np.isfinite(scored[list(COLUMNS)].to_numpy()).all()
    assert all(scored[column].dtype == "float32" for column in COLUMNS)


def test_a_training_window_with_no_spread_is_refused():
    """The one input shrinkage cannot rescue: with a prior of zero, every entity
    spread is zero too and every z-score is 0/0. This family promises finite,
    null-free columns, so it says so rather than emitting NaN."""
    rows = frame(["a", "a", "b", "b"], [10.0, 10.0, 10.0, 10.0])

    with pytest.raises(ValueError, match="no spread"):
        fit_amount_stats(rows, ("card1",), 10)


def test_the_input_frame_is_not_mutated():
    original = frame(["a", "a", "b"], [10.0, 20.0, 300.0])
    apply_amount_stats(original, fit_amount_stats(original, ("card1",), 10))

    assert not set(COLUMNS) & set(original.columns)


# ------------------------------------------------------------------------------
# names and persistence
# ------------------------------------------------------------------------------
def test_the_declared_columns_are_the_ones_produced():
    rows = frame(["a", "a", "b"], [10.0, 20.0, 300.0])

    assert set(COLUMNS) <= set(add_amount_stats(rows, CFG).columns)
    assert entity_column("card1", "absz") in COLUMNS


def test_the_fitted_tables_round_trip_carrying_their_own_fallback(tmp_path: Path):
    """A serving process reads one file and gets both the lookup and what to do
    when the lookup misses."""
    tables = fit_amount_stats(frame(["a", "a", "b"], [10.0, 20.0, 30.0]), ("card1",), 10)
    write_tables(tables, tmp_path / "amount_stats.parquet")

    written = pd.read_parquet(tmp_path / "amount_stats.parquet")

    assert list(written.columns) == ["entity", "level", "count", "mean", "std"]
    assert GLOBAL in set(written["level"])
