"""Tests for the unlabelled horizon.

Two tiers, as `docs/monitoring.md` §8 registers them. Above the gate, the pieces the
proof composes, against the synthetic deployment every serving test uses: the identity
rename, the refusal of a horizon missing an input, the matrix with its medians left out,
the score, and the two checks — each shown to refuse what it exists to refuse. The gate
itself runs the stage's proof against the shipped artifacts.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

from fraud_engine.features import velocity
from fraud_engine.features.build import build_features, order_by_time
from fraud_engine.models.calibrate import apply_calibrator
from fraud_engine.monitoring.horizon import (
    build,
    check_inputs,
    check_matrix,
    check_scores,
    identity_names,
    load_horizon,
    prove,
    score,
)
from fraud_engine.serving.artifacts import load_model
from fraud_engine.serving.transform import transform

CONFIG = yaml.safe_load(Path("config/config.yaml").read_text())


# ---- the identity block ---------------------------------------------------------


def test_the_test_file_s_identity_block_gets_the_model_s_spelling():
    names = identity_names(pd.Index(["TransactionID", "id-01", "id-38", "DeviceType"]))
    assert names == {"id-01": "id_01", "id-38": "id_38"}


def test_the_pair_loads_renamed_joined_and_dated(tmp_path):
    pd.DataFrame(
        {
            "TransactionID": [1, 2, 3],
            "TransactionDT": [18_403_224, 18_403_300, 18_490_000],
            "TransactionAmt": [10.0, 20.0, 30.0],
            "ProductCD": ["W", "C", "H"],
        }
    ).to_csv(tmp_path / "transactions.csv", index=False)
    pd.DataFrame({"TransactionID": [2], "id-01": [-5.0], "DeviceType": ["mobile"]}).to_csv(
        tmp_path / "identity.csv", index=False
    )

    frame = load_horizon(tmp_path / "transactions.csv", tmp_path / "identity.csv", CONFIG["load"])

    assert "id_01" in frame.columns and "id-01" not in frame.columns
    assert frame["has_identity"].tolist() == [False, True, False]
    assert frame["day"].tolist() == [213, 213, 214]


# ---- the refusal of a horizon missing an input ----------------------------------


@pytest.fixture(scope="module")
def shipped(deployment):
    return load_model(deployment["config"]["paths"], impute=True, threads=1)


def test_a_complete_horizon_passes(deployment, shipped):
    check_inputs(deployment["raw"], shipped.columns)


def test_a_missing_input_is_refused_rather_than_scored_as_null(deployment, shipped):
    """The misspelled identity block, in general form: the transform would null it."""
    incomplete = deployment["raw"].drop(columns=["card1"])
    with pytest.raises(ValueError, match="absent from the horizon"):
        check_inputs(incomplete, shipped.columns)


def test_a_column_still_in_the_test_file_s_spelling_is_refused(deployment, shipped):
    stray = deployment["raw"].assign(**{"id-01": 1.0})
    with pytest.raises(ValueError, match="still spelled"):
        check_inputs(stray, shipped.columns)


# ---- the matrix and the score ---------------------------------------------------


def as_loaded(raw):
    """The synthetic interim table in the dtypes `data/load.py` gives the real one."""
    wide = [name for name in raw.select_dtypes("float64").columns if name != "TransactionAmt"]
    return raw.astype(dict.fromkeys(wide, CONFIG["load"]["default_float_dtype"]))


def test_the_matrix_is_the_transform_with_only_the_fill_left_out(deployment, shipped):
    raw = deployment["raw"].copy()
    raw.loc[raw.index[::4], "card2"] = np.nan
    unfilled = build(raw, shipped.tables, CONFIG["load"], CONFIG["features"], shipped.columns)
    served = transform(raw, shipped.tables, CONFIG["load"], CONFIG["features"], shipped.columns)

    assert unfilled.isna().to_numpy().sum() > 0
    assert served.isna().to_numpy().sum() == 0
    filled = unfilled.fillna(shipped.tables.medians.reindex(unfilled.columns).dropna())
    pd.testing.assert_frame_equal(filled, served, check_exact=True)


def test_the_score_is_the_booster_s_and_the_probability_the_calibrator_s(deployment, shipped):
    raw = deployment["raw"]
    unfilled = build(raw, shipped.tables, CONFIG["load"], CONFIG["features"], shipped.columns)
    served = transform(raw, shipped.tables, CONFIG["load"], CONFIG["features"], shipped.columns)

    uncalibrated, calibrated = score(unfilled, shipped)

    booster = shipped.booster
    assert np.array_equal(
        uncalibrated, booster.predict(served, num_iteration=booster.best_iteration)
    )
    assert np.array_equal(calibrated, apply_calibrator(shipped.calibrator, uncalibrated))


# ---- the checks the proof is made of --------------------------------------------


@pytest.fixture(scope="module")
def labelled(deployment, shipped):
    """Labelled rows carrying the history the training matrix built, and that matrix."""
    raw = as_loaded(deployment["raw"])
    expected, _ = build_features(order_by_time(raw), CONFIG["features"])
    history = raw.merge(expected[["TransactionID", *velocity.COLUMNS]], on="TransactionID")
    unfilled = build(history, shipped.tables, CONFIG["load"], CONFIG["features"], shipped.columns)
    return unfilled, expected


def test_rows_carrying_their_history_reproduce_the_training_matrix(labelled, shipped):
    unfilled, expected = labelled
    check_matrix(unfilled, expected, shipped.tables)


def test_a_matrix_that_differs_in_one_value_is_refused(labelled, shipped):
    unfilled, expected = labelled
    changed = expected.copy()
    changed.loc[changed.index[0], "C1"] += 1
    with pytest.raises(ValueError, match="does not reproduce"):
        check_matrix(unfilled, changed, shipped.tables)


def test_scores_that_agree_to_the_bit_pass():
    values = np.array([0.1, 0.2])
    check_scores(
        values, values / 2, pd.DataFrame({"uncalibrated": values, "calibrated": values / 2})
    )


def test_a_score_that_differs_at_all_is_refused():
    values = np.array([0.1, 0.2])
    persisted = pd.DataFrame({"uncalibrated": values, "calibrated": np.nextafter(values / 2, 1)})
    with pytest.raises(ValueError, match="calibrated scores differ"):
        check_scores(values, values / 2, persisted)


# ---- the gate -------------------------------------------------------------------

PATHS = CONFIG["paths"]
REQUIRED = (
    PATHS["model"],
    PATHS["calibrator"],
    PATHS["categories"],
    PATHS["medians"],
    PATHS["encoders"],
    PATHS["amount_stats"],
    PATHS["vblock"],
    PATHS["interim"],
    f"{PATHS['features_dir']}/test.parquet",
    f"{PATHS['predictions_dir']}/headline_test.parquet",
)
gated = pytest.mark.skipif(
    not all(Path(path).exists() for path in REQUIRED),
    reason=(
        "the shipped artifacts, the matrices and the headline's scores are not in this "
        "checkout; docs/monitoring.md §8 registers this tier as artifact-gated"
    ),
)


@pytest.mark.artifacts
@gated
def test_the_horizon_path_reproduces_the_single_touch():
    """The stage's own proof: every test row, every column, and every score to the bit."""
    model = load_model(PATHS, CONFIG["model"]["impute"], CONFIG["horizon"]["threads"])
    proof = prove(model, PATHS, CONFIG["load"], CONFIG["features"])
    assert proof["rows"] > 0
