"""Tests for load.add_time_columns.

The inputs are chosen so every expected value is verifiable by hand, which
matters more here than in most places: these columns feed the temporal split in
Phase 02, so an off-by-one in the arithmetic would corrupt the split boundaries
themselves — silently, and with no downstream error to reveal it.
"""

import pandas as pd
import pytest

from fraud_engine.data.load import add_time_columns

HOUR = 3_600
DAY = 86_400

# (seconds, day, hour, weekday) — all hand-computed.
#   86400   the dataset's first transaction: exactly one day past the reference
#   90000   +1h,  same day
#  176400   +25h from the first: the day rolls over, hour wraps back to 1
#  777600   day 9; 9 % 7 == 2, so the weekday cycle has wrapped
#  691200   day 8; 8 % 7 == 1, same weekday slot as day 1
CASES = [
    (DAY, 1, 0, 1),
    (DAY + HOUR, 1, 1, 1),
    (DAY + 25 * HOUR, 2, 1, 2),
    (9 * DAY, 9, 0, 2),
    (8 * DAY, 8, 0, 1),
]


@pytest.fixture
def frame() -> pd.DataFrame:
    return pd.DataFrame({"TransactionDT": [seconds for seconds, *_ in CASES]})


@pytest.mark.parametrize(("seconds", "day", "hour", "weekday"), CASES)
def test_derives_expected_values(seconds, day, hour, weekday):
    row = add_time_columns(pd.DataFrame({"TransactionDT": [seconds]})).iloc[0]
    assert (row["day"], row["hour"], row["weekday"]) == (day, hour, weekday)


def test_all_three_columns_are_present(frame):
    """Regression test: `day` was once assigned twice, silently replacing itself.

    That produced a `day` column holding weekday values and no weekday column at
    all — no error, wrong data, and the temporal split built on top of it.
    """
    assert {"day", "hour", "weekday"} <= set(add_time_columns(frame).columns)


def test_columns_are_not_aliases_of_each_other(frame):
    """`day` and `weekday` diverge once the 7-day cycle wraps."""
    result = add_time_columns(frame)
    assert result["day"].tolist() != result["weekday"].tolist()


def test_input_frame_is_not_modified(frame):
    """The module returns new frames throughout; this one must not mutate."""
    before = list(frame.columns)
    add_time_columns(frame)
    assert list(frame.columns) == before


def test_existing_columns_are_preserved():
    original = pd.DataFrame({"TransactionID": [1, 2], "TransactionDT": [DAY, 2 * DAY]})
    result = add_time_columns(original)
    assert result["TransactionID"].tolist() == [1, 2]


def test_hour_wraps_at_24():
    seconds = [h * HOUR for h in range(26)]
    hours = add_time_columns(pd.DataFrame({"TransactionDT": seconds}))["hour"]
    assert hours.tolist() == [*range(24), 0, 1]


def test_weekday_wraps_at_7():
    seconds = [d * DAY for d in range(9)]
    weekdays = add_time_columns(pd.DataFrame({"TransactionDT": seconds}))["weekday"]
    assert weekdays.tolist() == [0, 1, 2, 3, 4, 5, 6, 0, 1]


def test_derived_columns_are_integers(frame):
    """Integer division, never a datetime conversion — TransactionDT is an
    offset in seconds, and parsing it as a timestamp yields plausible 1970 dates."""
    result = add_time_columns(frame)
    for column in ("day", "hour", "weekday"):
        assert pd.api.types.is_integer_dtype(result[column]), column
