"""Tests for the feature-family evaluation stage and the noise floor.

Synthetic frames again, and small ones: a probe fit on 800 rows takes
milliseconds where the real one takes 26 seconds, and none of the properties
worth pinning here are properties of the dataset.

The centre of gravity is the two leakage tests. `score_with` claims one `.fit()`
that only ever sees the train slice, and the Pipeline is supposed to make that
structural rather than careful — these are what would notice if it stopped being
true.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from fraud_engine.data.splits import SPLIT_NAMES
from fraud_engine.features.evaluate import load_matrices, score_with
from fraud_engine.features.floor import noise_floor
from fraud_engine.models.logistic import (
    COUNT_COLUMNS,
    DELTA_COLUMNS,
    MATCH_COLUMNS,
    prepare,
)

# Literal rather than read from config.yaml: these tests are about the probe's
# behaviour, and should not start failing because a baseline was retuned.
LOGISTIC_CFG = {"C": 1.0, "max_iter": 1000}
CAPACITIES = [0.1]

# Wide enough that a per-day capacity of 10% reviews whole transactions rather
# than fractions of one.
ROWS_PER_DAY = 100
DAYS = {"train": (1, 4), "val_fit": (5, 6), "val_cal": (7, 7), "test": (8, 8)}


def synthetic(seed: int = 0) -> pd.DataFrame:
    """A frame carrying every column the probe reads, with learnable signal.

    `isFraud` depends on amount, so the fit has something to find and a scored
    column is not constant — several tests below compare score vectors, which
    says nothing if every score is identical.
    """
    rng = np.random.default_rng(seed)
    split = [name for name, (first, last) in DAYS.items() for _ in range(first, last + 1)]
    day = [d for _, (first, last) in DAYS.items() for d in range(first, last + 1)]

    n = len(day) * ROWS_PER_DAY
    frame = pd.DataFrame(
        {
            "TransactionID": range(1, n + 1),
            "day": np.repeat(day, ROWS_PER_DAY),
            "split": np.repeat(split, ROWS_PER_DAY),
            "TransactionAmt": rng.lognormal(4.0, 1.0, n),
            "hour": rng.integers(0, 24, n),
            "has_identity": rng.integers(0, 2, n),
            "ProductCD": rng.choice(["W", "C", "R"], n),
            "card4": rng.choice(["visa", "mastercard"], n),
            "card6": rng.choice(["debit", "credit"], n),
            "P_emaildomain": rng.choice(["gmail.com", "yahoo.com"], n),
            "weekday": rng.integers(0, 7, n),
        }
    )
    for column in (*COUNT_COLUMNS, *DELTA_COLUMNS):
        frame[column] = rng.normal(size=n)
    for column in MATCH_COLUMNS:
        frame[column] = rng.choice(["T", "F"], n)

    odds = (np.log(frame["TransactionAmt"]) - 4.0) / 2.0
    frame["isFraud"] = (rng.random(n) < 1 / (1 + np.exp(-odds))).astype(int)

    return prepare(frame)


def probe(frame: pd.DataFrame, extra_numeric: tuple[str, ...] = ()) -> pd.Series:
    """`score_with` with the fixtures' fixed config, returning just the scores."""
    return score_with(frame, extra_numeric, list(DELTA_COLUMNS), LOGISTIC_CFG)["score"]


# ------------------------------------------------------------------------------
# load_matrices
# ------------------------------------------------------------------------------
@pytest.fixture
def written(tmp_path: Path) -> Path:
    """Four matrices on disk, each holding its own split's rows and nothing else."""
    for index, name in enumerate(SPLIT_NAMES):
        frame = pd.DataFrame(
            {"TransactionID": [index * 10 + 1, index * 10 + 2], "day": [index, index]}
        )
        frame.to_parquet(tmp_path / f"{name}.parquet", index=False)
    return tmp_path


def test_the_split_label_is_restored_from_the_filename(written):
    frame = load_matrices(written, ["TransactionID", "day"])

    assert list(frame["split"]) == [name for name in SPLIT_NAMES for _ in range(2)]


def test_the_split_column_is_categorical_over_every_split(written):
    """Object dtype would cost tens of megabytes on the real matrices, and a
    concat of four object columns cannot be relied on to stay comparable."""
    split = load_matrices(written, ["TransactionID", "day"])["split"]

    assert split.dtype == "category"
    assert list(split.cat.categories) == list(SPLIT_NAMES)


def test_only_the_requested_columns_are_read(written):
    frame = load_matrices(written, ["TransactionID"])

    assert list(frame.columns) == ["TransactionID", "split"]


def test_naming_split_among_the_columns_is_rejected(written):
    """It is added here, not read — and the matrices genuinely do not carry it,
    so without this the caller gets a pyarrow stack trace instead of a sentence."""
    with pytest.raises(ValueError, match="restored from the filenames"):
        load_matrices(written, ["TransactionID", "split"])


def test_a_missing_matrix_is_not_silently_skipped(written):
    """Three matrices and a gap is not three quarters of a run — it is a run
    whose baseline is measured on different rows than the families' will be."""
    (written / "val_cal.parquet").unlink()

    with pytest.raises(FileNotFoundError):
        load_matrices(written, ["TransactionID"])


# ------------------------------------------------------------------------------
# score_with — the leakage claims
# ------------------------------------------------------------------------------
def test_validation_labels_cannot_reach_the_fit():
    """The claim is one `.fit()` on the train slice. If the pipeline ever fitted
    on the whole frame, flipping every validation label would move the scores."""
    frame = synthetic()
    flipped = frame.copy()
    held_out = flipped["split"] != "train"
    flipped.loc[held_out, "isFraud"] = 1 - flipped.loc[held_out, "isFraud"]

    pd.testing.assert_series_equal(probe(frame), probe(flipped))


def test_validation_feature_values_cannot_reach_the_fit():
    """The other half, and the one a Pipeline is there to guarantee: the median
    that fills a family's nulls and the scale it is divided by come from train.
    Multiplying the held-out rows by a thousand must not move a train score."""
    frame = synthetic()
    frame["extra"] = np.arange(len(frame), dtype=float)

    perturbed = frame.copy()
    held_out = perturbed["split"] != "train"
    perturbed.loc[held_out, "extra"] *= 1000

    on_train = frame["split"] == "train"
    pd.testing.assert_series_equal(
        probe(frame, ("extra",))[on_train], probe(perturbed, ("extra",))[on_train]
    )


def test_extra_columns_reach_the_model():
    """Guards the guard above: both leakage tests would pass vacuously if
    `extra_numeric` were quietly dropped on the way to the ColumnTransformer."""
    frame = synthetic()
    frame["extra"] = frame["isFraud"] * 3.0

    assert not probe(frame).equals(probe(frame, ("extra",)))


def test_every_row_is_scored_including_the_splits_not_fitted_on():
    frame = synthetic()
    scored = probe(frame)

    assert len(scored) == len(frame)
    assert scored.between(0, 1).all()


# ------------------------------------------------------------------------------
# noise_floor
# ------------------------------------------------------------------------------
def test_one_row_per_seed_at_the_requested_width():
    measured = noise_floor(synthetic(), list(DELTA_COLUMNS), LOGISTIC_CFG, CAPACITIES, range(3), 2)

    assert list(measured["seed"]) == [0, 1, 2]
    assert list(measured["n_columns"]) == [2, 2, 2]


def test_the_same_seed_measures_the_same_thing():
    """The floor is a claim about chance, so it has to be reproducible chance —
    otherwise re-running the stage moves the bar every family is judged against."""
    args = (synthetic(), list(DELTA_COLUMNS), LOGISTIC_CFG, CAPACITIES, [7])

    assert noise_floor(*args)["pr_auc"][0] == noise_floor(*args)["pr_auc"][0]


def test_width_changes_the_measurement():
    one = noise_floor(synthetic(), list(DELTA_COLUMNS), LOGISTIC_CFG, CAPACITIES, [0], 1)
    six = noise_floor(synthetic(), list(DELTA_COLUMNS), LOGISTIC_CFG, CAPACITIES, [0], 6)

    assert one["pr_auc"][0] != six["pr_auc"][0]


def test_the_callers_frame_does_not_come_back_carrying_noise():
    """`noise_floor` works on a copy. If it did not, the family scored after it
    would silently inherit the last seed's noise columns."""
    frame = synthetic()
    before = list(frame.columns)

    noise_floor(frame, list(DELTA_COLUMNS), LOGISTIC_CFG, CAPACITIES, range(2))

    assert list(frame.columns) == before
