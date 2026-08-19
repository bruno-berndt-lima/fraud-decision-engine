"""Tests for checking an assignment against its data.

A compact layout, not Layout B: these checks are layout-independent, and an
11-day span makes every case inspectable. No parquet is read — CI has no data.
"""

import pandas as pd
import pytest

from fraud_engine.data.splits import assign_splits, validate_splits

#  day   1  2  3 | 4  5 | 6  7 | 8  9 | 10 11
#        train   | gap  | v_fit| v_cal|  test
BOUNDARIES = {
    "train": (1, 3),
    "val_fit": (6, 7),
    "val_cal": (8, 9),
    "test": (10, 11),
}
GAP_DAYS = (4, 5)


def make_days(values=None) -> pd.Series:
    """One row per day, spanning the boundaries exactly."""
    return pd.Series(list(values if values is not None else range(1, 12)), name="day")


def valid_pair() -> tuple[pd.Series, pd.Series]:
    days = make_days()
    return days, assign_splits(days, BOUNDARIES)


def test_a_valid_assignment_passes():
    days, split = valid_pair()
    assert len(validate_splits(days, split, BOUNDARIES)) == len(days)


def test_returns_the_labels_unchanged():
    """It composes into a pipeline, so it must not alter what it validates."""
    days, split = valid_pair()
    pd.testing.assert_series_equal(validate_splits(days, split, BOUNDARIES), split)


def test_gap_rows_are_not_a_violation():
    days, split = valid_pair()
    assert split[days.isin(GAP_DAYS)].isna().all()
    validate_splits(days, split, BOUNDARIES)


def test_passes_when_there_is_no_gap_at_all():
    """The E1 case: gap_days 0 leaves nothing unclaimed, which is not an error."""
    contiguous = {**BOUNDARIES, "train": (1, 5)}
    days = make_days()
    validate_splits(days, assign_splits(days, contiguous), contiguous)


# ---- Inputs -----------------------------------------------------------------


def test_rejects_a_misaligned_index():
    days, split = valid_pair()
    with pytest.raises(ValueError, match="share an index"):
        validate_splits(days, split.iloc[:-1], BOUNDARIES)


def test_rejects_empty_input():
    empty = pd.Series([], dtype="int64")
    with pytest.raises(ValueError, match="no rows"):
        validate_splits(empty, assign_splits(empty, BOUNDARIES), BOUNDARIES)


# ---- Span -------------------------------------------------------------------


@pytest.mark.parametrize(
    ("label", "values"),
    [
        ("data starts before train_start", range(0, 12)),
        ("data extends past test_end", range(1, 13)),
        ("data starts after train_start", range(2, 12)),
        ("data ends before test_end", range(1, 11)),
    ],
)
def test_rejects_a_span_mismatch(label, values):
    """Data wider than the span drops rows; span wider names days that do not exist."""
    days = make_days(values)
    with pytest.raises(ValueError, match="span"):
        validate_splits(days, assign_splits(days, BOUNDARIES), BOUNDARIES)


# ---- The assignment itself --------------------------------------------------


def test_rejects_an_empty_middle_split():
    """Only val_fit and val_cal can be empty once the span matches.

    A matching span means a row exists on train_start and on test_end, and each
    sits inside its own split — so those two are never empty here.
    """
    days = make_days([day for day in range(1, 12) if day not in (8, 9)])
    with pytest.raises(ValueError, match="val_cal is empty"):
        validate_splits(days, assign_splits(days, BOUNDARIES), BOUNDARIES)


def test_rejects_a_row_labelled_outside_its_own_range():
    """Checks the assignment against the boundaries rather than trusting it."""
    days, split = valid_pair()
    tampered = split.copy()
    tampered[days == 1] = "test"

    with pytest.raises(ValueError, match="test contains days"):
        validate_splits(days, tampered, BOUNDARIES)


@pytest.mark.parametrize(
    ("label", "day"),
    [
        ("inside train's range", 2),
        ("inside val_cal's range", 8),
        ("on the last day", 11),
        # The two days where the gap's own bounds sit. An unclaimed row here is
        # still a violation, and a `<` where the code needs `<=` misses exactly
        # these and nothing else.
        ("on train_end, the day before the gap", 3),
        ("on val_fit_start, the day after the gap", 6),
    ],
)
def test_rejects_an_unclaimed_row_outside_the_gap(label, day):
    """The check nothing upstream can make — null means gap OR out of span."""
    days, split = valid_pair()
    tampered = split.copy()
    tampered[days == day] = None

    with pytest.raises(ValueError, match="outside the purge gap"):
        validate_splits(days, tampered, BOUNDARIES)
