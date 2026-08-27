"""Tests for the velocity family.

This is the trickiest code in the phase and the only family with no fitted state,
so the tests are about *ordering* rather than about a fit/apply boundary. Three
properties carry the weight:

- no row may see its own future;
- a row's window must reach across split boundaries, because a card's history
  does not stop existing because the labels after it were purged;
- and rows sharing a `TransactionDT` must not scramble, which is what a
  time-indexed rolling window would do to the many such rows in the data.

The last one is not hypothetical. The first implementation of this module used
`groupby(...).rolling(window, on=...)`, whose result is indexed by timestamp, and
it silently misaligned.
"""

import numpy as np
import pandas as pd
import pytest

from fraud_engine.features.velocity import (
    COLUMNS,
    ENTITY,
    RECENCY,
    WINDOWS,
    add_velocity_features,
    trailing_counts,
)

HOUR = 3_600
DAY = 86_400
CFG = {"first_seen_gap_days": 30}


def frame(seconds, cards=None, splits=None) -> pd.DataFrame:
    """Rows in causal order, one card unless told otherwise."""
    seconds = list(seconds)
    built = pd.DataFrame(
        {
            "TransactionID": range(1, len(seconds) + 1),
            "TransactionDT": seconds,
            ENTITY: cards if cards is not None else ["a"] * len(seconds),
            "split": splits if splits is not None else ["train"] * len(seconds),
        }
    )
    return built.sort_values(["TransactionDT", "TransactionID"], ignore_index=True)


def counts(rows, window="1h"):
    return trailing_counts(rows, {window: WINDOWS[window]})[f"vel_n{window}_{ENTITY}"]


# ------------------------------------------------------------------------------
# window semantics
# ------------------------------------------------------------------------------
def test_a_transaction_counts_itself():
    """A lone transaction is one in its own window, not zero. The column is a
    count of the window, and the row is in it."""
    assert list(counts(frame([0]))) == [1]


def test_the_window_is_half_open_at_the_far_edge():
    """`(t - 3600, t]`. A transaction exactly an hour earlier is outside; one a
    second later than that is inside. Off-by-one here changes every count."""
    assert list(counts(frame([0, HOUR]))) == [1, 1]
    assert list(counts(frame([0, HOUR - 1]))) == [1, 2]


def test_longer_windows_never_count_less():
    rows = frame([0, HOUR * 2, DAY * 2, DAY * 6])
    built = trailing_counts(rows, WINDOWS)

    assert (built["vel_n1h_card1"] <= built["vel_n24h_card1"]).all()
    assert (built["vel_n24h_card1"] <= built["vel_n7d_card1"]).all()


def test_another_card_does_not_count():
    rows = frame([0, 1, 2], cards=["a", "b", "a"])

    assert list(counts(rows)) == [1, 1, 2]


# ------------------------------------------------------------------------------
# the ordering invariants
# ------------------------------------------------------------------------------
def test_no_row_can_see_its_own_future():
    """The definition of a causal feature, as an experiment: delete every row
    after the third and the third's answer must not move."""
    full = frame([0, 60, 120, 180, 240])
    truncated = full.iloc[:3].copy()

    assert counts(full).iloc[2] == counts(truncated).iloc[2]


def test_the_window_reaches_across_a_split_boundary():
    """A validation row's trailing window legitimately includes training and
    purged-gap rows: those transactions happened, and production would count
    them. Computing per split would reset every card's history at the boundary
    and invent a train/serve skew that production does not have."""
    rows = frame([0, 60, 120], splits=["train", None, "val_fit"])

    assert list(counts(rows)) == [1, 2, 3]


def test_rows_sharing_a_timestamp_do_not_scramble():
    """The bug the first implementation had. Three transactions on one second
    for card 'a' and one for card 'b': if the counts were unwound by timestamp
    rather than by position, 'b' would collect 'a''s history."""
    rows = frame([0, 0, 0, 0], cards=["a", "a", "a", "b"])

    built = counts(rows)

    assert list(built[rows[ENTITY] == "a"]) == [1, 2, 3]
    assert list(built[rows[ENTITY] == "b"]) == [1]


def test_the_counts_agree_with_a_time_indexed_rolling_window():
    """Cross-check of the binary search against pandas, on data with no ties —
    the case where the two are supposed to agree exactly."""
    rows = frame([0, 100, 1_000, 3_000, 3_599, 3_601, 7_000])

    expected = (
        rows.assign(_at=pd.to_datetime(rows["TransactionDT"], unit="s"))
        .set_index("_at")["TransactionID"]
        .rolling("1h")
        .count()
    )

    assert list(counts(rows)) == list(expected)


# ------------------------------------------------------------------------------
# recency
# ------------------------------------------------------------------------------
def test_recency_measures_the_gap_to_the_previous_transaction():
    built = add_velocity_features(frame([0, HOUR]), CFG)

    assert built[RECENCY].iloc[1] == pytest.approx(np.log1p(HOUR), rel=1e-5)


def test_a_first_sighting_lands_at_the_slow_end_rather_than_null():
    """It has no predecessor. Measured on train its lift sits on the slowest
    quintile's, so it belongs at that end of the scale — not as a null, and not
    as a flag of its own."""
    built = add_velocity_features(frame([0, HOUR]), CFG)

    assert built[RECENCY].iloc[0] == pytest.approx(np.log1p(30 * DAY), rel=1e-5)
    assert built[RECENCY].iloc[0] > built[RECENCY].iloc[1]


def test_recency_is_measured_per_card():
    built = add_velocity_features(frame([0, HOUR], cards=["a", "b"]), CFG)

    assert built[RECENCY].iloc[0] == built[RECENCY].iloc[1], "both are first sightings"


# ------------------------------------------------------------------------------
# the family contract
# ------------------------------------------------------------------------------
def test_a_null_card_is_one_entity_rather_than_dropped():
    """`levels()` is shared with the fitted families so every one of them groups
    a null the same way. A plain groupby would drop these rows and leave nulls
    in a family that promises none."""
    rows = frame([0, 60, 120], cards=[None, None, "a"])

    built = add_velocity_features(rows, CFG)

    assert built[list(COLUMNS)].notna().all().all()
    assert built["vel_n1h_card1"].iloc[1] == 2


def test_the_family_is_null_free_float32():
    built = add_velocity_features(frame([0, 60, DAY * 10]), CFG)

    assert built[list(COLUMNS)].notna().all().all()
    assert all(built[column].dtype == "float32" for column in COLUMNS)


def test_the_input_frame_is_not_mutated():
    original = frame([0, 60])
    add_velocity_features(original, CFG)

    assert not set(COLUMNS) & set(original.columns)
