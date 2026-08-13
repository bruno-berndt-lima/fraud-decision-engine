"""Tests for load.join_identity.

The load-bearing test here is `test_matched_row_with_all_null_fields_is_still
_flagged`. Every other property could survive an implementation that derived
`has_identity` from a null check instead of the join indicator — that one
cannot, and the distinction it protects is the reason the flag exists.
"""

import pandas as pd
import pytest
from pandas.errors import MergeError

from fraud_engine.data.load import join_identity


@pytest.fixture
def transactions() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "TransactionID": [1, 2, 3, 4],
            "TransactionAmt": [10.0, 20.0, 30.0, 40.0],
        }
    )


@pytest.fixture
def identity() -> pd.DataFrame:
    """Covers 2 of 4 transactions. Transaction 4's record is entirely null.

    That last row is the point: it exists, so it matched, but every attribute
    in it is empty — indistinguishable from an unmatched row if you look only
    at the values.
    """
    return pd.DataFrame(
        {
            "TransactionID": [2, 4],
            "DeviceType": ["mobile", None],
            "DeviceInfo": ["iOS", None],
        }
    )


def test_row_count_is_preserved(transactions, identity):
    assert len(join_identity(transactions, identity)) == len(transactions)


def test_every_transaction_survives(transactions, identity):
    joined = join_identity(transactions, identity)
    assert joined["TransactionID"].tolist() == transactions["TransactionID"].tolist()


def test_identity_columns_are_attached(transactions, identity):
    joined = join_identity(transactions, identity)
    assert {"DeviceType", "DeviceInfo"} <= set(joined.columns)


def test_flag_is_true_for_matched_rows_only(transactions, identity):
    joined = join_identity(transactions, identity)
    flagged = set(joined.loc[joined["has_identity"], "TransactionID"])
    assert flagged == {2, 4}


def test_matched_row_with_all_null_fields_is_still_flagged(transactions, identity):
    """Transaction 4 matched, but every identity attribute is null.

    Deriving the flag from `DeviceInfo.notna()` would mark this False and
    conflate "no identity record" with "a record that happened to be empty".
    Only the join indicator can tell them apart.
    """
    joined = join_identity(transactions, identity).set_index("TransactionID")
    assert joined.loc[4, "DeviceInfo"] is None or pd.isna(joined.loc[4, "DeviceInfo"])
    assert bool(joined.loc[4, "has_identity"]) is True


def test_unmatched_rows_are_not_filled(transactions, identity):
    """Missingness is signal; nothing here may impute it away."""
    joined = join_identity(transactions, identity).set_index("TransactionID")
    assert pd.isna(joined.loc[1, "DeviceType"])
    assert pd.isna(joined.loc[3, "DeviceType"])


def test_merge_indicator_does_not_reach_the_output(transactions, identity):
    """`_merge` is scaffolding — it must not end up in the parquet."""
    assert "_merge" not in join_identity(transactions, identity).columns


def test_flag_is_boolean(transactions, identity):
    assert join_identity(transactions, identity)["has_identity"].dtype == bool


def test_duplicate_identity_keys_are_rejected(transactions):
    """A LEFT join with duplicate right-hand keys multiplies rows.

    Without the validate= check this would return 5 rows for 4 transactions —
    fabricating a transaction rather than failing.
    """
    dupes = pd.DataFrame(
        {"TransactionID": [2, 2], "DeviceType": ["mobile", "desktop"], "DeviceInfo": ["iOS", "Win"]}
    )
    with pytest.raises(MergeError):
        join_identity(transactions, dupes)


def test_duplicate_transaction_keys_are_rejected(identity):
    """The same guard protects the left side, where duplicates mean a bad read."""
    dupes = pd.DataFrame({"TransactionID": [2, 2], "TransactionAmt": [10.0, 20.0]})
    with pytest.raises(MergeError):
        join_identity(dupes, identity)
