"""Tests for E2's arms: what each one changes, and what it must leave alone.

`arm_params` is pure and gets pinned exactly — including that the two weighting
arms ask for the same thing by different routes, which E2 predicted and which
would stop being demonstrated if either drifted.

`resample` is run for real on a small frame. SMOTE-NC is the arm this project
runs precisely so it can report the reflex losing, and the properties that make
that report honest — training rows only, categoricals kept to real levels — are
not things a stub would check.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from fraud_engine.models.imbalance import (
    IMPUTED_ARMS,
    WEIGHT_ARMS,
    arm_params,
    measure,
    resample,
)
from fraud_engine.models.train import LABEL, apply_categories, fit_categories

CAPACITIES = [0.1]
MODEL_CFG = {"tuned": {}, "seed": 0, "early_stopping_rounds": 5, "num_boost_round": 60}

FEATURES = ["signal", "noise", "brand"]


# Train and validation carry deliberately different base rates. A ratio read off
# the wrong slice would otherwise land within rounding of the right one, and
# every test here would pass on a module measuring imbalance where it must not.
TRAIN_CUTOFF = 0.9
VALIDATION_CUTOFF = 0.7


def make_matrix(rows: int = 400, seed: int = 0, cutoff: float = TRAIN_CUTOFF) -> pd.DataFrame:
    """An imbalanced matrix: a planted signal, and positives above `cutoff`."""
    rng = np.random.default_rng(seed)
    signal = rng.random(rows)

    return pd.DataFrame(
        {
            "TransactionID": range(rows),
            "TransactionDT": np.arange(rows) * 60,
            LABEL: (signal > cutoff).astype(int),
            "day": 1,
            "signal": signal,
            "noise": rng.random(rows),
            "brand": pd.Categorical(rng.choice(["visa", "amex", "elo"], rows)),
        }
    )


@pytest.fixture
def matrices() -> dict[str, pd.DataFrame]:
    raw = {
        "train": make_matrix(seed=0, cutoff=TRAIN_CUTOFF),
        "val_fit": make_matrix(seed=1, cutoff=VALIDATION_CUTOFF),
        "val_cal": make_matrix(seed=2, cutoff=VALIDATION_CUTOFF),
    }
    vocabulary = fit_categories(raw["train"], min_rows=1)
    return {name: apply_categories(frame, vocabulary) for name, frame in raw.items()}


@pytest.fixture
def paths(tmp_path: Path) -> dict:
    return {"metrics_dir": str(tmp_path / "metrics"), "predictions_dir": str(tmp_path / "preds")}


# ------------------------------------------------------------------------------
# arm_params
# ------------------------------------------------------------------------------


def test_the_reference_arm_changes_nothing():
    assert arm_params("none", ratio=27.5) == {}


def test_scale_pos_weight_takes_the_measured_ratio():
    """Balanced is a fact about the training window, not a preference.

    A number in config would silently stop being balanced the moment the window
    moved.
    """
    assert arm_params("scale_pos_weight", ratio=27.5) == {"scale_pos_weight": 27.5}


def test_is_unbalance_asks_lightgbm_to_derive_it():
    assert arm_params("is_unbalance", ratio=27.5) == {"is_unbalance": True}


def test_the_two_weighting_arms_do_not_set_the_same_key():
    """They are expected to agree in result, by different routes.

    Setting both would make the demonstration circular — E2's point is that
    LightGBM derives the ratio this module measured, not that one arm copies
    the other.
    """
    by_hand = arm_params("scale_pos_weight", ratio=27.5)
    derived = arm_params("is_unbalance", ratio=27.5)

    assert not set(by_hand) & set(derived)


def test_the_data_arms_add_no_parameters():
    """What they change is upstream of the model."""
    for arm in IMPUTED_ARMS:
        assert arm_params(arm, ratio=27.5) == {}


def test_an_unknown_arm_falls_through_to_the_reference():
    assert arm_params("typo", ratio=27.5) == {}


def test_no_arm_reaches_a_contract_parameter():
    from fraud_engine.models.train import CONTRACT_PARAMS

    for arm in (*WEIGHT_ARMS, *IMPUTED_ARMS):
        assert not set(arm_params(arm, ratio=27.5)) & set(CONTRACT_PARAMS)


# ------------------------------------------------------------------------------
# resample
# ------------------------------------------------------------------------------


@pytest.fixture
def resampled(matrices) -> pd.DataFrame:
    return resample(matrices["train"], FEATURES, seed=0)


def test_the_positive_class_reaches_parity(resampled):
    """Parity is the reflex under test, not a gentler ratio someone preferred."""
    counts = resampled[LABEL].value_counts()

    assert counts[0] == counts[1]


def test_the_majority_class_is_untouched(matrices, resampled):
    """Oversampling adds positives; it must not drop negatives."""
    before = (matrices["train"][LABEL] == 0).sum()

    assert (resampled[LABEL] == 0).sum() == before


def test_synthetic_rows_carry_no_transaction_identity(resampled):
    """A synthetic row is not a transaction, and inventing an id would say it was."""
    assert set(resampled.columns) == {*FEATURES, LABEL}


def test_categoricals_stay_on_real_levels(matrices, resampled):
    """There is no midpoint between two card brands.

    SMOTE-NC takes the majority level among neighbours instead, and this is what
    would notice if plain SMOTE were substituted and started interpolating codes.
    """
    levels = set(matrices["train"]["brand"].cat.categories)

    assert set(pd.Series(resampled["brand"]).unique()) <= levels


def test_resampling_is_reproducible(matrices):
    """The synthetic rows are the same on every rerun, so an arm can be re-measured."""
    first = resample(matrices["train"], FEATURES, seed=0)
    second = resample(matrices["train"], FEATURES, seed=0)

    pd.testing.assert_frame_equal(first, second)


def test_a_different_seed_synthesises_different_rows(matrices):
    first = resample(matrices["train"], FEATURES, seed=0)
    second = resample(matrices["train"], FEATURES, seed=1)

    assert not first.equals(second)


def test_validation_is_never_resampled(matrices, resampled):
    """Synthesising validation rows scores the model on transactions that never happened."""
    assert len(resampled) > len(matrices["train"])
    assert len(matrices["val_fit"]) == len(make_matrix(seed=1, cutoff=VALIDATION_CUTOFF))


# ------------------------------------------------------------------------------
# measure
# ------------------------------------------------------------------------------


@pytest.fixture
def comparison(matrices, paths) -> pd.DataFrame:
    return measure(matrices, FEATURES, MODEL_CFG, CAPACITIES, paths)


def test_every_arm_is_measured(comparison):
    assert list(comparison["arm"]) == [*WEIGHT_ARMS, *IMPUTED_ARMS]


def test_a_record_is_written_per_arm(comparison, paths):
    written = {path.stem for path in Path(paths["metrics_dir"]).glob("*.json")}

    assert written == {f"imbalance_{arm}" for arm in (*WEIGHT_ARMS, *IMPUTED_ARMS)}


def test_only_the_resampling_arm_trains_on_more_rows(comparison, matrices):
    """`imputed` exists to isolate the resampling from the imputation it requires.

    If it grew too, the SMOTE arm's row count would no longer say what SMOTE
    alone did.
    """
    rows = comparison.set_index("arm")["train_rows"]

    assert rows["smote"] > rows["imputed"]
    assert set(rows.drop("smote")) == {len(matrices["train"])}


def test_the_two_weighting_arms_land_on_the_same_number(comparison):
    """E2's registered prediction, demonstrated once rather than asserted.

    It is also what notices a ratio read off the wrong slice: `is_unbalance`
    makes LightGBM derive it from the training data it was handed, so the two
    agree only while this module measures it there too.
    """
    scores = comparison.set_index("arm")["pr_auc"]

    assert scores["scale_pos_weight"] == pytest.approx(scores["is_unbalance"])


def test_the_ratio_is_measured_from_the_training_window(
    matrices, paths, caplog: pytest.LogCaptureFixture
):
    """Not from config, and not from validation.

    The ratio never leaves the module as a value, so the log line it reports is
    what there is to read. Asserting it against a recomputation here would be
    circular; asserting it against the slice it must *not* come from is not.
    """
    train, validation = matrices["train"], matrices["val_fit"]
    from_train = (len(train) - train[LABEL].sum()) / train[LABEL].sum()
    from_validation = (len(validation) - validation[LABEL].sum()) / validation[LABEL].sum()

    assert abs(from_train - from_validation) > 1.0, "the fixture cannot tell the two apart"

    with caplog.at_level("INFO"):
        measure(matrices, FEATURES, MODEL_CFG, CAPACITIES, paths)

    reported = next(line for line in caplog.messages if "neg/pos" in line)

    assert f"{from_train:.2f}" in reported
    assert f"{from_validation:.2f}" not in reported


def test_records_carry_the_splits_they_were_given(comparison, paths):
    for path in Path(paths["metrics_dir"]).glob("*.json"):
        assert "test" not in json.loads(path.read_text())["splits"]
