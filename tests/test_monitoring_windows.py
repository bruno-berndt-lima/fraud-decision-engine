"""Tests for the monitoring window.

The two layouts `docs/monitoring.md` §2 registers are pinned exactly: a change here
moves every PSI and PR-AUC point in the phase.
"""

import numpy as np
import pandas as pd
import pytest

from fraud_engine.monitoring.windows import describe, window_index


def every_day(first, last):
    return pd.Series(np.arange(first, last + 1))


# ---- the registered layouts ---------------------------------------------------


def test_the_labelled_layout_is_the_registered_one():
    table = describe(every_day(121, 182), 5)

    assert list(zip(table["first_day"], table["last_day"], strict=True)) == [
        (121, 125), (126, 130), (131, 135), (136, 140),
        (141, 145), (146, 150), (151, 155), (156, 160),
        (161, 165), (166, 170), (171, 175), (176, 180),
        (181, 182),
    ]  # fmt: skip
    assert table["partial"].tolist() == [False] * 12 + [True]


def test_no_labelled_window_straddles_a_split_boundary():
    """Five divides VAL-FIT and VAL-CAL, so 141 and 161 both open a window."""
    starts = set(describe(every_day(121, 182), 5)["first_day"])
    assert {121, 141, 161} <= starts


def test_the_horizon_layout_is_the_registered_one():
    table = describe(every_day(213, 395), 28)

    assert list(zip(table["first_day"], table["last_day"], strict=True)) == [
        (213, 240), (241, 268), (269, 296), (297, 324), (325, 352), (353, 380),
        (381, 395),
    ]  # fmt: skip
    assert table["days"].iloc[-1] == 15
    assert table["partial"].tolist() == [False] * 6 + [True]


# ---- assignment ---------------------------------------------------------------


def test_every_row_lands_in_exactly_one_window():
    rng = np.random.default_rng(0)
    days = pd.Series(rng.integers(121, 183, size=5_000))

    index = window_index(days, 5)
    table = describe(days, 5)

    assert table["rows"].sum() == len(days)
    assert np.array_equal(np.bincount(index, minlength=len(table)), table["rows"])


def test_a_row_sits_in_the_window_whose_span_holds_its_day():
    days = pd.Series([121, 125, 126, 182])
    table = describe(days, 5)

    for day, window in zip(days, window_index(days, 5), strict=True):
        row = table.iloc[window]
        assert row["first_day"] <= day <= row["last_day"]


def test_the_order_of_rows_is_kept():
    assert window_index(pd.Series([182, 121, 140]), 5).tolist() == [12, 0, 3]


def test_an_exact_fit_has_no_partial_window():
    table = describe(every_day(141, 160), 5)
    assert len(table) == 4
    assert not table["partial"].any()


def test_a_day_with_no_rows_does_not_shorten_its_window():
    """Windows are calendar spans; missing days are reported as missing, not skipped."""
    table = describe(pd.Series([1, 2, 11, 12]), 5)

    assert table["rows"].tolist() == [2, 0, 2]
    assert table["days"].tolist() == [5, 5, 2]


def test_the_anchor_defaults_to_the_first_day_present():
    days = pd.Series([150, 151, 160])
    assert window_index(days, 5).tolist() == [0, 0, 2]


def test_an_earlier_anchor_is_honoured():
    days = pd.Series([123, 126])
    assert window_index(days, 5, anchor=121).tolist() == [0, 1]
    assert describe(days, 5, anchor=121)["first_day"].tolist() == [121, 126]


# ---- refusals -----------------------------------------------------------------


def test_an_anchor_after_the_first_day_is_refused():
    with pytest.raises(ValueError, match="after the first day"):
        window_index(pd.Series([121, 130]), 5, anchor=125)


@pytest.mark.parametrize("width", [0, -5])
def test_a_width_below_one_is_refused(width):
    with pytest.raises(ValueError, match="width"):
        describe(every_day(1, 10), width)


def test_no_days_is_refused():
    with pytest.raises(ValueError, match="no days"):
        describe(pd.Series([], dtype="int32"), 5)


def test_a_row_with_no_day_is_refused():
    with pytest.raises(ValueError, match="no day"):
        window_index(pd.Series([121.0, np.nan]), 5)


def test_a_fractional_day_is_refused():
    with pytest.raises(ValueError, match="whole"):
        window_index(pd.Series([121.0, 121.5]), 5)
