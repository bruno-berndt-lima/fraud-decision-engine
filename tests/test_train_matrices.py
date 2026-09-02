"""Tests for what LightGBM is handed: the feature set and the category vocabulary.

Synthetic frames throughout — CI has no data, and the cases that matter are ones
the real matrices only happen to contain.

Several fixtures give a categorical a level that appears in its dtype but in no
row. That is not contrived: it is the shape the split files actually arrive in,
since ``partition`` sliced one frame and the vocabulary travelled with it.
"""

from pathlib import Path

import pandas as pd
import pytest

from fraud_engine.models.train import (
    EXCLUDED_COLUMNS,
    MISSING,
    OTHER,
    TRAINING_SPLITS,
    apply_categories,
    feature_columns,
    fit_categories,
    load_split_matrices,
)

MIN_ROWS = 3

LEVELS = ["common", "rare", "elsewhere"]


def categorical(values: list, levels: list[str] = LEVELS) -> pd.Categorical:
    """A column whose dtype knows more levels than its rows use."""
    return pd.Categorical(values, categories=levels)


def make_train() -> pd.DataFrame:
    """``common`` clears the floor, ``rare`` does not, ``elsewhere`` has no rows."""
    return pd.DataFrame(
        {
            "browser": categorical(["common"] * MIN_ROWS + ["rare", None]),
            "amount": [1.0, 2.0, 3.0, 4.0, 5.0],
        }
    )


def make_matrix(split: str, rows: int = 4) -> pd.DataFrame:
    """A matrix carrying every column the deny-list names."""
    return pd.DataFrame(
        {
            "TransactionID": range(rows),
            "TransactionDT": range(rows),
            "isFraud": [0] * rows,
            "day": [1] * rows,
            "amount": [1.0] * rows,
            "split_marker": [split] * rows,
        }
    )


# ------------------------------------------------------------------------------
# fit_categories
# ------------------------------------------------------------------------------


def test_only_levels_train_saw_often_enough_are_learned():
    vocabulary = fit_categories(make_train(), MIN_ROWS)

    assert "common" in vocabulary["browser"]
    assert "rare" not in vocabulary["browser"]


def test_a_level_no_training_row_uses_is_not_in_the_vocabulary():
    vocabulary = fit_categories(make_train(), MIN_ROWS)

    assert "elsewhere" in make_train()["browser"].cat.categories
    assert "elsewhere" not in vocabulary["browser"]


def test_both_sentinels_are_present_even_when_train_needed_neither():
    train = pd.DataFrame({"browser": categorical(["common"] * MIN_ROWS, ["common"])})

    vocabulary = fit_categories(train, MIN_ROWS)

    assert list(vocabulary["browser"]) == ["common", OTHER, MISSING]


def test_the_vocabulary_does_not_depend_on_row_order():
    train = make_train()

    assert list(fit_categories(train, MIN_ROWS)["browser"]) == list(
        fit_categories(train.sample(frac=1.0, random_state=1), MIN_ROWS)["browser"]
    )


def test_only_categorical_columns_are_fitted():
    assert set(fit_categories(make_train(), MIN_ROWS)) == {"browser"}


def test_a_column_whose_every_level_is_rare_keeps_only_the_sentinels():
    train = pd.DataFrame({"browser": categorical(["rare", None], ["rare"])})

    assert list(fit_categories(train, MIN_ROWS)["browser"]) == [OTHER, MISSING]


# ------------------------------------------------------------------------------
# apply_categories
# ------------------------------------------------------------------------------


def test_a_level_train_never_saw_becomes_other():
    vocabulary = fit_categories(make_train(), MIN_ROWS)
    scored = pd.DataFrame({"browser": categorical(["elsewhere"])})

    assert apply_categories(scored, vocabulary)["browser"].tolist() == [OTHER]


def test_a_rare_training_level_becomes_other():
    vocabulary = fit_categories(make_train(), MIN_ROWS)

    applied = apply_categories(make_train(), vocabulary)["browser"]
    assert applied.tolist() == ["common"] * MIN_ROWS + [OTHER, MISSING]


def test_a_null_becomes_missing_rather_than_other():
    vocabulary = fit_categories(make_train(), MIN_ROWS)
    scored = pd.DataFrame({"browser": categorical([None])})

    assert apply_categories(scored, vocabulary)["browser"].tolist() == [MISSING]


def test_two_frames_end_up_with_the_same_categories_in_the_same_order():
    vocabulary = fit_categories(make_train(), MIN_ROWS)
    other = pd.DataFrame({"browser": categorical(["elsewhere", None])})

    first = apply_categories(make_train(), vocabulary)["browser"]
    second = apply_categories(other, vocabulary)["browser"]
    assert list(first.cat.categories) == list(second.cat.categories)


def test_fitted_columns_come_back_null_free():
    vocabulary = fit_categories(make_train(), MIN_ROWS)

    assert not apply_categories(make_train(), vocabulary)["browser"].isna().any()


def test_the_input_frame_is_not_modified():
    train = make_train()
    apply_categories(train, fit_categories(train, MIN_ROWS))

    assert train["browser"].isna().any()
    assert list(train["browser"].cat.categories) == LEVELS


def test_a_categorical_with_no_fitted_vocabulary_is_refused():
    scored = make_train()
    scored["device"] = categorical(["common"] * 5)

    with pytest.raises(ValueError, match="no fitted vocabulary"):
        apply_categories(scored, fit_categories(make_train(), MIN_ROWS))


# ------------------------------------------------------------------------------
# feature_columns
# ------------------------------------------------------------------------------


def test_the_label_and_the_timeline_are_not_features():
    assert feature_columns(make_matrix("train")) == ["amount", "split_marker"]


def test_the_columns_keep_the_order_the_matrix_had():
    matrix = make_matrix("train")[
        ["amount", "isFraud", "split_marker", "day", *EXCLUDED_COLUMNS[:2]]
    ]

    assert feature_columns(matrix) == ["amount", "split_marker"]


@pytest.mark.parametrize("excluded", EXCLUDED_COLUMNS)
def test_an_excluded_column_missing_from_the_matrix_is_refused(excluded: str):
    with pytest.raises(ValueError, match="no longer describes this table"):
        feature_columns(make_matrix("train").drop(columns=excluded))


# ------------------------------------------------------------------------------
# load_split_matrices
# ------------------------------------------------------------------------------


@pytest.fixture
def features_dir(tmp_path: Path) -> Path:
    """One parquet per split, each knowing which split it is."""
    for split in (*TRAINING_SPLITS, "test"):
        make_matrix(split).to_parquet(tmp_path / f"{split}.parquet", index=False)
    return tmp_path


def test_test_is_not_read_unless_it_is_named(features_dir: Path):
    assert set(load_split_matrices(features_dir)) == set(TRAINING_SPLITS)


def test_naming_test_reads_it(features_dir: Path):
    matrices = load_split_matrices(features_dir, ("test",))

    assert matrices["test"]["split_marker"].unique().tolist() == ["test"]


def test_each_matrix_is_the_file_of_that_name(features_dir: Path):
    matrices = load_split_matrices(features_dir)

    assert all(
        matrix["split_marker"].unique().tolist() == [split] for split, matrix in matrices.items()
    )


def test_a_name_that_is_not_a_split_is_refused(features_dir: Path):
    with pytest.raises(ValueError, match="not splits"):
        load_split_matrices(features_dir, ("train", "holdout"))
