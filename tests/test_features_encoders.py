"""Tests for frequency encoding — the first family with fitted state.

The arithmetic is one line of `value_counts`. What needs pinning is everything
around it: that the fit sees only training rows, that the transform sees all of
them, that a level train never saw lands somewhere defined, and that a level is
spelled the same way on both sides of the join.

That last one is the quiet killer. `card1` arrives as a float, so `13926.0` has
to be the same key at fit time and at apply time, or every row misses the lookup
and the whole column silently becomes UNSEEN_FREQUENCY.
"""

from pathlib import Path

import pandas as pd
import pytest

from fraud_engine.features.encoders import (
    COLUMNS,
    FREQUENCY_COLUMNS,
    MISSING,
    UNSEEN_FREQUENCY,
    add_frequency_features,
    apply_frequencies,
    encoded_name,
    fit_frequencies,
    write_tables,
)


def frame(**overrides) -> pd.DataFrame:
    """Eight rows: six train, two validation, with one level unique to each."""
    base = {
        "split": ["train"] * 6 + ["val_fit"] * 2,
        # 'a' three times, 'b' twice, 'c' once in train; 'z' only in validation.
        "card1": ["a", "a", "a", "b", "b", "c", "a", "z"],
    }
    built = pd.DataFrame({**base, **overrides})

    # The family encodes every FREQUENCY_COLUMN, so they all have to exist. Only
    # the one a test names carries any structure.
    for column in FREQUENCY_COLUMNS:
        if column not in built:
            built[column] = "x"
    return built


# ------------------------------------------------------------------------------
# fit_frequencies
# ------------------------------------------------------------------------------
def test_a_level_is_valued_as_its_share_of_the_training_rows():
    table = fit_frequencies(frame().head(6), ("card1",))["card1"]

    assert table["a"] == 0.5  # 3 of 6
    assert table["b"] == pytest.approx(2 / 6)
    assert table["c"] == pytest.approx(1 / 6)


def test_the_rates_of_one_column_sum_to_one():
    """A share, not a count — the property that survives a refit on a different
    window, which E1 will do with a training set 31% longer."""
    table = fit_frequencies(frame().head(6), ("card1",))["card1"]

    assert table.sum() == pytest.approx(1.0)


def test_null_is_a_level_rather_than_an_absence():
    train = frame(card1=[None, None, "a", "a", "a", "a", "a", "a"]).head(6)

    assert fit_frequencies(train, ("card1",))["card1"][MISSING] == pytest.approx(2 / 6)


# ------------------------------------------------------------------------------
# apply_frequencies — the leakage claims
# ------------------------------------------------------------------------------
def test_validation_rows_do_not_influence_the_rates():
    """The fit is confined to the training window; the transform is not. Piling
    a hundred extra 'b' rows into validation must not change what 'b' is worth."""
    small = frame()
    padded = pd.concat(
        [small, pd.DataFrame({"split": ["val_fit"] * 100, "card1": ["b"] * 100})],
        ignore_index=True,
    )

    a = add_frequency_features(small)
    b = add_frequency_features(padded)

    assert a["freq_card1"].iloc[0] == b["freq_card1"].iloc[0]


def test_every_row_is_encoded_not_only_the_ones_fitted_on():
    encoded = add_frequency_features(frame())

    assert encoded["freq_card1"].notna().all()
    assert encoded.loc[encoded["split"] == "val_fit", "freq_card1"].iloc[0] == 0.5


def test_a_level_train_never_saw_takes_the_unseen_value():
    encoded = add_frequency_features(frame())

    assert encoded["freq_card1"].iloc[-1] == UNSEEN_FREQUENCY


def test_numeric_levels_survive_the_round_trip():
    """`card1` is a float in the real data. If the fit keys on 13926.0 and the
    apply keys on '13926.0', every row misses and the column silently flatlines
    at UNSEEN_FREQUENCY — a bug that raises nothing and shows as a dead feature."""
    numeric = frame(card1=[13926.0, 13926.0, 13926.0, 4.0, 4.0, 7.0, 13926.0, 99.0])

    encoded = add_frequency_features(numeric)

    assert encoded["freq_card1"].iloc[0] == 0.5
    assert (encoded["freq_card1"] == UNSEEN_FREQUENCY).sum() == 1


def test_the_output_is_null_free_float32():
    """`build_features` promises families make their own fill decisions."""
    encoded = add_frequency_features(frame(card1=["a", None, "a", "b", "b", "c", None, "z"]))

    assert encoded["freq_card1"].notna().all()
    assert encoded["freq_card1"].dtype == "float32"


def test_the_input_frame_is_not_mutated():
    original = frame()
    apply_frequencies(original, fit_frequencies(original.head(6), ("card1",)))

    assert "freq_card1" not in original.columns


# ------------------------------------------------------------------------------
# names and persistence
# ------------------------------------------------------------------------------
def test_the_declared_columns_are_the_ones_produced():
    """`COLUMNS` is imported by the family registry, so a rename that missed one
    would leave `evaluate.py` asking the probe for a column nobody writes."""
    assert tuple(encoded_name(column) for column in FREQUENCY_COLUMNS) == COLUMNS


def test_the_fitted_tables_round_trip_through_disk(tmp_path: Path):
    """The artifact a served model carries. It has to describe itself: which
    column, which level, what rate — with no knowledge of the fitting code."""
    tables = fit_frequencies(frame().head(6), ("card1",))
    write_tables(tables, tmp_path / "encoders.parquet")

    written = pd.read_parquet(tmp_path / "encoders.parquet")

    assert list(written.columns) == ["column", "level", "frequency"]
    assert set(written["column"]) == {"card1"}
    assert written.set_index("level")["frequency"]["a"] == 0.5
