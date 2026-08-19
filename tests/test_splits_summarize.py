"""Tests for the per-section counts.

Same compact layout as test_splits_validate, one row per day, so every count in
the expected tables can be checked by eye.
"""

import pandas as pd
import pytest

from fraud_engine.data.splits import GAP_LABEL, SECTION_ORDER, assign_splits, summarize

#  day   1  2  3 | 4  5 | 6  7 | 8  9 | 10 11
#        train   | gap  | v_fit| v_cal|  test
BOUNDARIES = {
    "train": (1, 3),
    "val_fit": (6, 7),
    "val_cal": (8, 9),
    "test": (10, 11),
}

# One positive in each section, including the gap.
DEFAULT_FRAUD_DAYS = (1, 4, 6, 8, 10)


def make_inputs(fraud_days=DEFAULT_FRAUD_DAYS, boundaries=None):
    days = pd.Series(range(1, 12), name="day")
    is_fraud = days.isin(fraud_days).astype("int8").rename("isFraud")
    return days, assign_splits(days, boundaries or BOUNDARIES), is_fraud


def test_counts_rows_and_positives_per_section():
    result = summarize(*make_inputs())
    assert result["rows"].tolist() == [3, 2, 2, 2, 2]
    assert result["frauds"].tolist() == [1, 1, 1, 1, 1]


def test_reports_the_day_span_of_each_section():
    result = summarize(*make_inputs())
    assert result["first_day"].tolist() == [1, 4, 6, 8, 10]
    assert result["last_day"].tolist() == [3, 5, 7, 9, 11]
    assert result["n_days"].tolist() == [3, 2, 2, 2, 2]


def test_fraud_rate_is_positives_over_rows():
    result = summarize(*make_inputs())
    assert result.loc["train", "fraud_rate"] == pytest.approx(1 / 3)
    assert result.loc["test", "fraud_rate"] == pytest.approx(1 / 2)


def test_every_row_is_accounted_for():
    days, split, is_fraud = make_inputs()
    assert summarize(days, split, is_fraud)["rows"].sum() == len(days)


# ---- Shape of the artifact --------------------------------------------------
# It is written to reports/ and compared row by row across the two E1 runs, so
# the index, the column order and the absence of nulls in the counts are part
# of the contract rather than incidental.


def test_index_is_chronological_with_the_gap_in_place():
    assert tuple(summarize(*make_inputs()).index) == SECTION_ORDER
    assert SECTION_ORDER == ("train", GAP_LABEL, "val_fit", "val_cal", "test")


def test_columns_are_in_report_order():
    assert list(summarize(*make_inputs()).columns) == [
        "first_day",
        "last_day",
        "n_days",
        "rows",
        "frauds",
        "fraud_rate",
    ]


def test_counts_are_never_null():
    result = summarize(*make_inputs())
    assert not result["rows"].isna().any()
    assert not result["frauds"].isna().any()


def test_an_empty_gap_is_a_row_of_zeros_not_a_missing_row():
    """gap_days: 0 must not change the artifact's shape — E1 compares the two."""
    contiguous = {**BOUNDARIES, "train": (1, 5)}
    result = summarize(*make_inputs(fraud_days=(1, 6, 8, 10), boundaries=contiguous))

    assert tuple(result.index) == SECTION_ORDER
    assert result.loc[GAP_LABEL, "rows"] == 0
    assert result.loc[GAP_LABEL, "frauds"] == 0
    assert pd.isna(result.loc[GAP_LABEL, "first_day"])
    assert pd.isna(result.loc[GAP_LABEL, "n_days"])


def test_the_whole_gap_moves_into_train_when_there_is_none():
    """The E1 contrast, in counts: train gains exactly what the gap held."""
    purged = summarize(*make_inputs())
    unpurged = summarize(*make_inputs(boundaries={**BOUNDARIES, "train": (1, 5)}))

    assert (
        unpurged.loc["train", "rows"] == purged.loc["train", "rows"] + purged.loc[GAP_LABEL, "rows"]
    )
    for section in ("val_fit", "val_cal", "test"):
        assert unpurged.loc[section, "rows"] == purged.loc[section, "rows"]


# ---- Guards -----------------------------------------------------------------


@pytest.mark.parametrize(
    ("label", "fraud_days", "barren"),
    [
        ("train has none", (4, 6, 8, 10), "train"),
        ("val_fit has none", (1, 4, 8, 10), "val_fit"),
        ("val_cal has none", (1, 4, 6, 10), "val_cal"),
        ("test has none", (1, 4, 6, 8), "test"),
    ],
)
def test_raises_when_a_split_holds_no_positives(label, fraud_days, barren):
    """PR-AUC is undefined on a slice with no fraud; fail here, not in Phase 03."""
    with pytest.raises(ValueError, match=barren):
        summarize(*make_inputs(fraud_days=fraud_days))


def test_names_every_barren_split_not_just_the_first():
    with pytest.raises(ValueError, match=r"\['val_cal', 'test'\]"):
        summarize(*make_inputs(fraud_days=(1, 6)))


def test_a_gap_with_no_positives_is_not_an_error():
    """The gap is discarded, so it needs none."""
    result = summarize(*make_inputs(fraud_days=(1, 6, 8, 10)))
    assert result.loc[GAP_LABEL, "frauds"] == 0


def test_rejects_misaligned_inputs():
    days, split, is_fraud = make_inputs()
    with pytest.raises(ValueError, match="share an index"):
        summarize(days, split, is_fraud.iloc[:-1])
