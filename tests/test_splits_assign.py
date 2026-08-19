"""Tests for labelling rows with their split.

Boundaries are literal here rather than built by `resolve_boundaries`, so a bug
in that function cannot mask one here. A single test at the bottom checks the
two compose.
"""

from pathlib import Path

import pandas as pd
import pytest
import yaml

from fraud_engine.data.splits import SPLIT_NAMES, assign_splits, resolve_boundaries

REPO_ROOT = Path(__file__).resolve().parents[1]

# Layout B. The gap is days 91-120, claimed by nothing.
BOUNDARIES = {
    "train": (1, 90),
    "val_fit": (121, 140),
    "val_cal": (141, 160),
    "test": (161, 182),
}

EMPTY = pd.Series([], dtype="int64")


@pytest.mark.parametrize(
    ("day", "expected"),
    [
        (0, None),  # before train_start
        (1, "train"),
        (90, "train"),  # inclusive end
        (91, None),  # first gap day
        (120, None),  # last gap day
        (121, "val_fit"),  # inclusive start
        (140, "val_fit"),
        (141, "val_cal"),
        (160, "val_cal"),
        (161, "test"),
        (182, "test"),
        (183, None),  # past test_end
    ],
)
def test_boundary_days_land_in_the_right_split(day, expected):
    """Every bug in this function lives on one of these days."""
    result = assign_splits(pd.Series([day]), BOUNDARIES).iloc[0]
    if expected is None:
        assert pd.isna(result)
    else:
        assert result == expected


def test_the_gap_is_exactly_the_unclaimed_days():
    span = pd.Series(range(1, 183))
    unclaimed = span[assign_splits(span, BOUNDARIES).isna()]
    assert list(unclaimed) == list(range(91, 121))


def test_gap_days_zero_leaves_nothing_unclaimed():
    """Ties this function to the E1 contract pinned in test_splits_boundaries."""
    boundaries = resolve_boundaries(
        {
            "gap_days": 0,
            "train_start": 1,
            "val_fit_start": 121,
            "val_cal_start": 141,
            "test_start": 161,
            "test_end": 182,
        }
    )
    span = pd.Series(range(1, 183))
    assert assign_splits(span, boundaries).notna().all()


# ---- The returned Series ----------------------------------------------------


def test_returns_an_ordered_categorical_over_split_names():
    dtype = assign_splits(pd.Series([1, 121]), BOUNDARIES).dtype
    assert isinstance(dtype, pd.CategoricalDtype)
    assert tuple(dtype.categories) == SPLIT_NAMES
    assert dtype.ordered


def test_groupby_sorts_chronologically_without_a_key():
    """Why the dtype is ordered — summarize() and the reports rely on it."""
    days = pd.Series([161, 1, 141, 121])
    grouped = days.groupby(assign_splits(days, BOUNDARIES), observed=True).size()
    assert tuple(grouped.index) == SPLIT_NAMES


def test_is_named_split():
    assert assign_splits(pd.Series([1]), BOUNDARIES).name == "split"


def test_preserves_the_index():
    days = pd.Series([90, 121], index=[77, 12])
    assert list(assign_splits(days, BOUNDARIES).index) == [77, 12]


def test_preserves_length_including_unclaimed_rows():
    assert len(assign_splits(pd.Series([1, 91, 121, 183]), BOUNDARIES)) == 4


def test_an_empty_series_returns_an_empty_labelled_series():
    result = assign_splits(EMPTY, BOUNDARIES)
    assert len(result) == 0
    assert tuple(result.dtype.categories) == SPLIT_NAMES


# ---- Guards -----------------------------------------------------------------


@pytest.mark.parametrize("days", [EMPTY, pd.Series([1])], ids=["empty", "one_row"])
@pytest.mark.parametrize(
    ("label", "boundaries"),
    [
        ("unknown key", {"nope": (1, 2)}),
        ("missing a split", {k: v for k, v in BOUNDARIES.items() if k != "test"}),
        ("keys out of order", {k: BOUNDARIES[k] for k in ("val_fit", "train", "val_cal", "test")}),
        ("overlapping ranges", {**BOUNDARIES, "val_fit": (85, 140)}),
        ("ranges touching at one day", {**BOUNDARIES, "val_cal": (140, 160)}),
        ("inverted range", {**BOUNDARIES, "val_fit": (140, 121)}),
    ],
)
def test_rejects_bad_boundaries_regardless_of_the_data(label, boundaries, days):
    """Both parametrised over the data on purpose.

    Overlap was once checked by tracking which *rows* were claimed twice, which
    passed whenever no row fell inside the overlap — including on an empty
    Series. The `empty` case is the regression test for that.
    """
    with pytest.raises(ValueError):
        assign_splits(days, boundaries)


def test_accepts_the_output_of_resolve_boundaries():
    """The only test coupling the two functions."""
    config = yaml.safe_load((REPO_ROOT / "config" / "config.yaml").read_text())
    boundaries = resolve_boundaries(config["splits"])
    labelled = assign_splits(pd.Series([90, 91, 121, 182]), boundaries)
    assert list(labelled.dropna()) == ["train", "val_fit", "test"]
    assert labelled.isna().tolist() == [False, True, False, False]
