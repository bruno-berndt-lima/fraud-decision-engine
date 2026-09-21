"""Tests for what serving without the card's history costs — `docs/serving.md` §2.

The measurement itself needs the shipped booster and the built matrices, so what runs
everywhere is the part a wrong answer would hide in: that the neutralised arm really is
neutralised, that the share qualifying it counts the right rows, and that the difference
is reported in the direction the doc reads it.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from fraud_engine.data.load import DEFAULT_CONFIG_PATH, load_config
from fraud_engine.evaluation.cost import load_costs
from fraud_engine.features import velocity
from fraud_engine.models.calibrate import SPLIT
from fraud_engine.models.train import LABEL
from fraud_engine.serving.neutralisation import (
    INTACT,
    NEUTRALISED,
    first_sighting_share,
    measure,
    neutralise,
)

CONFIG = load_config(DEFAULT_CONFIG_PATH)
FEATURES = CONFIG["features"]
COLUMNS = list(velocity.COLUMNS)
COSTS = load_costs(load_config(Path(CONFIG["paths"]["cost_matrix"])))

ROWS, DAYS, SEED = 400, 8, 0


@pytest.fixture
def matrix() -> pd.DataFrame:
    """A slice carrying the velocity family and a column that must survive untouched."""
    generator = np.random.default_rng(SEED)

    return pd.DataFrame(
        {
            "TransactionID": np.arange(ROWS),
            "TransactionAmt": generator.uniform(10, 500, ROWS).astype("float32"),
            **{column: generator.uniform(2, 40, ROWS).astype("float32") for column in COLUMNS},
        }
    )


def defaults(features_cfg: dict) -> dict[str, float]:
    """The no-history values, read off an empty frame rather than written out again."""
    return neutralise(pd.DataFrame(index=[0], columns=COLUMNS, dtype="float32"), features_cfg)[
        COLUMNS
    ].iloc[0]


def test_every_velocity_column_becomes_the_no_history_value(matrix):
    neutralised = neutralise(matrix, FEATURES)

    for column, value in defaults(FEATURES).items():
        assert (neutralised[column] == value).all()


def test_history_the_slice_carried_does_not_survive(matrix):
    """The failure this exists for: `fill_history` keeps supplied columns.

    Filling without dropping first would return the matrix unchanged, both arms would
    score identically, and the measurement would report that the store costs nothing.
    """
    neutralised = neutralise(matrix, FEATURES)

    assert not neutralised[COLUMNS].equals(matrix[COLUMNS])


def test_nothing_outside_the_family_is_touched(matrix):
    neutralised = neutralise(matrix, FEATURES)

    pd.testing.assert_frame_equal(neutralised.drop(columns=COLUMNS), matrix.drop(columns=COLUMNS))


def test_the_columns_stay_the_type_the_family_writes(matrix):
    """A float64 velocity column reaches the booster as a different number than float32."""
    neutralised = neutralise(matrix, FEATURES)

    assert (neutralised[COLUMNS].dtypes == "float32").all()


def test_a_slice_of_first_sightings_is_reported_as_one(matrix):
    already = neutralise(matrix, FEATURES)

    assert first_sighting_share(already, neutralise(already, FEATURES)) == 1.0


def test_a_slice_with_history_is_reported_as_none(matrix):
    assert first_sighting_share(matrix, neutralise(matrix, FEATURES)) == 0.0


def test_a_row_counts_only_when_every_column_already_held_the_default(matrix):
    """Three of four matching is a card with history, not a first sighting."""
    partial = matrix.copy()
    partial.loc[0, COLUMNS[:-1]] = defaults(FEATURES)[COLUMNS[:-1]].to_numpy()

    assert first_sighting_share(partial, neutralise(partial, FEATURES)) == 0.0


# ---- the difference, and which way round it is -------------------------------


def arm(frame: pd.DataFrame, score: np.ndarray) -> pd.DataFrame:
    """One arm's scores, shaped as `train.score` shapes them."""
    return pd.DataFrame(
        {
            "TransactionID": frame["TransactionID"],
            "split": SPLIT,
            "day": frame["day"],
            LABEL: frame[LABEL],
            "score": score,
        }
    )


@pytest.fixture
def arms() -> tuple[pd.DataFrame, dict[str, pd.DataFrame]]:
    """A slice, and two arms of which the second ranks fraud strictly worse.

    The worse arm is the intact one's scores with noise added, so the comparison has a
    known sign: whatever the policy costs, the arm that separates fraud less well cannot
    separate it better.
    """
    generator = np.random.default_rng(SEED)
    y = (generator.random(ROWS) < 0.2).astype("int8")

    frame = pd.DataFrame(
        {
            "TransactionID": np.arange(ROWS),
            "day": np.repeat(np.arange(DAYS), ROWS // DAYS),
            LABEL: y,
            "amount": generator.uniform(20, 400, ROWS),
        }
    )

    clean = np.clip(0.05 + 0.6 * y + generator.normal(0, 0.05, ROWS), 1e-6, 1 - 1e-6)
    noisy = np.clip(0.05 + 0.6 * y + generator.normal(0, 0.45, ROWS), 1e-6, 1 - 1e-6)

    return frame, {INTACT: arm(frame, clean), NEUTRALISED: arm(frame, noisy)}


def test_the_delta_is_the_neutralised_arm_minus_the_intact_one(arms):
    """The sign is the whole reading. Reversed, a cost would be reported as a saving."""
    frame, scored = arms
    results = measure(
        frame, scored, COSTS, 0.01, CONFIG["calibration"], "platt", CONFIG["neutralisation"]
    )

    assert results["delta"]["pr_auc"] < 0, "the arm that ranks fraud worse cannot score higher"
    assert results["delta"]["pr_auc"] == pytest.approx(
        results["arms"][NEUTRALISED]["pr_auc"] - results["arms"][INTACT]["pr_auc"]
    )
    assert results["delta"]["usd_per_1000"] == pytest.approx(
        results["arms"][NEUTRALISED]["usd_per_1000"] - results["arms"][INTACT]["usd_per_1000"]
    )


def test_the_interval_is_on_the_same_difference(arms):
    frame, scored = arms
    results = measure(
        frame, scored, COSTS, 0.01, CONFIG["calibration"], "platt", CONFIG["neutralisation"]
    )

    low, high = results["delta"]["usd_interval"]
    assert low < high
    assert results["delta"]["interval"] == CONFIG["neutralisation"]["interval"]
