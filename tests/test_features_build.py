"""Tests for the feature-matrix build stage.

Every frame here is synthetic. Nothing in this suite has ever needed `data/`,
and the stage's contract is row bookkeeping — which twelve rows state more
clearly than 590,540 do.

The exception is the split boundaries. The `calendar` fixture drives the
committed config through the whole stage at one transaction per day, so a
boundary edit this stage would mishandle fails here rather than two phases on.
"""

from pathlib import Path

import pandas as pd
import pytest
import yaml

from fraud_engine.data.splits import SPLIT_NAMES, assign_splits, resolve_boundaries
from fraud_engine.features.build import (
    attach_splits,
    build_features,
    order_by_time,
    partition,
    write_matrices,
)

REPO_ROOT = Path(__file__).resolve().parents[1]


def interim(ids, dts) -> pd.DataFrame:
    """An interim-like frame: the two columns this stage actually reads."""
    return pd.DataFrame({"TransactionID": list(ids), "TransactionDT": list(dts)})


def labelled(split_labels) -> pd.DataFrame:
    """A frame as it reaches `partition` — already joined, already sorted."""
    n = len(split_labels)
    return pd.DataFrame(
        {
            "TransactionID": range(1, n + 1),
            "TransactionDT": range(100, 100 + n),
            "day": range(1, n + 1),
            "split": split_labels,
        }
    )


# ------------------------------------------------------------------------------
# attach_splits
# ------------------------------------------------------------------------------
def test_each_label_lands_on_its_own_transaction():
    frame = interim([3, 1, 2], [30, 10, 20])
    splits = pd.DataFrame({"TransactionID": [1, 2, 3], "split": ["train", None, "test"]})

    by_id = attach_splits(frame, splits).set_index("TransactionID")["split"]

    assert by_id[1] == "train"
    assert by_id[3] == "test"
    assert pd.isna(by_id[2])


def test_a_splits_file_missing_a_transaction_is_rejected():
    """The case a left join would hide: the unmatched row would come back with a
    null label, which is exactly what a legitimately purged gap row looks like."""
    with pytest.raises(ValueError, match="different transactions"):
        attach_splits(
            interim([1, 2, 3], [10, 20, 30]),
            pd.DataFrame({"TransactionID": [1, 2], "split": ["train", "train"]}),
        )


def test_a_splits_file_naming_an_unknown_transaction_is_rejected():
    """The other direction: a splits.parquet left over from a different interim."""
    with pytest.raises(ValueError, match="different transactions"):
        attach_splits(
            interim([1, 2], [10, 20]),
            pd.DataFrame({"TransactionID": [1, 2, 3], "split": ["train", "train", "test"]}),
        )


def test_duplicate_transaction_ids_are_rejected():
    """Raised by `validate="one_to_one"` as pandas' MergeError, a ValueError."""
    with pytest.raises(ValueError):
        attach_splits(
            interim([1, 2], [10, 20]),
            pd.DataFrame({"TransactionID": [1, 1, 2], "split": ["train", "train", "test"]}),
        )


# ------------------------------------------------------------------------------
# order_by_time
# ------------------------------------------------------------------------------
def test_rows_come_back_in_transaction_dt_order():
    assert list(order_by_time(interim([1, 2, 3], [300, 100, 200]))["TransactionDT"]) == [
        100,
        200,
        300,
    ]


def test_transaction_dt_ties_break_by_transaction_id():
    """5.7% of the dataset shares a TransactionDT with another row, up to eight at
    once. Without the second key those rows' trailing-window counts would depend
    on whatever order the interim table happened to arrive in."""
    assert list(order_by_time(interim([9, 4, 7], [100, 100, 100]))["TransactionID"]) == [4, 7, 9]


def test_the_order_does_not_depend_on_the_input_order():
    frame = interim([1, 2, 3, 4], [100, 100, 200, 200])
    assert order_by_time(frame).equals(order_by_time(frame.iloc[[3, 0, 2, 1]]))


def test_the_index_is_reset_to_a_row_number():
    assert list(order_by_time(interim([1, 2, 3], [300, 100, 200])).index) == [0, 1, 2]


# ------------------------------------------------------------------------------
# build_features
# ------------------------------------------------------------------------------
def test_the_row_sequence_survives_the_seam():
    """Trivially true while the seam is empty, and the point of writing it now: it
    stops being trivial the moment a family reindexes or reorders, which would
    misalign the `split` mask that fitted families select train rows with."""
    frame = order_by_time(interim([1, 2, 3], [300, 100, 200]))
    assert list(build_features(frame)["TransactionID"]) == list(frame["TransactionID"])


# ------------------------------------------------------------------------------
# partition
# ------------------------------------------------------------------------------
def test_gap_rows_do_not_reach_any_matrix():
    matrices = partition(labelled(["train", None, "val_fit", None, "test"]))

    assert sum(len(matrix) for matrix in matrices.values()) == 3
    assert list(matrices["train"]["TransactionID"]) == [1]


def test_every_split_gets_a_matrix_even_when_empty():
    """An empty split still gets a key, and so still gets a file — a downstream
    stage reading val_cal.parquet should fail loudly, not on a missing path."""
    matrices = partition(labelled(["train"]))

    assert list(matrices) == list(SPLIT_NAMES)
    assert len(matrices["test"]) == 0


def test_the_split_column_does_not_survive():
    matrices = partition(labelled(["train", "test"]))
    assert all("split" not in matrix.columns for matrix in matrices.values())


def test_each_matrix_is_indexed_from_zero():
    assert list(partition(labelled(["train", "test", "train"]))["train"].index) == [0, 1]


def test_an_unrecognised_label_is_rejected_rather_than_dropped():
    with pytest.raises(ValueError, match="unrecognised split labels"):
        partition(labelled(["train", "validation"]))


# ------------------------------------------------------------------------------
# write_matrices
# ------------------------------------------------------------------------------
def test_one_parquet_is_written_per_split(tmp_path):
    write_matrices(partition(labelled(["train", "val_fit", "val_cal", "test"])), tmp_path / "out")

    written = sorted(path.name for path in (tmp_path / "out").glob("*.parquet"))
    assert written == sorted(f"{name}.parquet" for name in SPLIT_NAMES)


def test_a_matrix_round_trips_without_gaining_an_index_column(tmp_path):
    matrices = partition(labelled(["train", "test"]))
    write_matrices(matrices, tmp_path)

    assert pd.read_parquet(tmp_path / "train.parquet").equals(matrices["train"])


# ------------------------------------------------------------------------------
# The whole stage, against the committed boundaries
# ------------------------------------------------------------------------------
@pytest.fixture(scope="module")
def calendar():
    """The stage run end to end on one transaction per day of the 182-day span.

    Labels come from `assign_splits` against the committed config rather than
    from literals, so this asserts the stage honours whatever boundaries ship —
    `test_splits_assign.py` is what pins the boundaries themselves.
    """
    config = yaml.safe_load((REPO_ROOT / "config" / "config.yaml").read_text())
    boundaries = resolve_boundaries(config["splits"])

    days = pd.Series(range(1, 183))
    frame = pd.DataFrame({"TransactionID": days, "TransactionDT": days * 86_400, "day": days})
    splits = pd.DataFrame({"TransactionID": days, "split": assign_splits(days, boundaries)})

    return boundaries, partition(build_features(order_by_time(attach_splits(frame, splits))))


def test_each_matrix_holds_exactly_the_days_its_boundary_claims(calendar):
    boundaries, matrices = calendar

    for name, (first, last) in boundaries.items():
        assert list(matrices[name]["day"]) == list(range(first, last + 1))


def test_no_day_reaches_two_matrices(calendar):
    _, matrices = calendar

    held = [day for matrix in matrices.values() for day in matrix["day"]]
    assert len(held) == len(set(held))


def test_the_days_dropped_are_exactly_the_ones_no_boundary_claims(calendar):
    boundaries, matrices = calendar

    held = {day for matrix in matrices.values() for day in matrix["day"]}
    claimed = {day for first, last in boundaries.values() for day in range(first, last + 1)}
    assert held == claimed
