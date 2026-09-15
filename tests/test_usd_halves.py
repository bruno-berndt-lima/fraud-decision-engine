"""Tests for the USD halves of E1 and E3.

The reproduction guard is the one that matters. Without it, a refit on a stale matrix
or a changed configuration would still produce a USD figure, and it would be filed
next to a PR-AUC measured on a different model.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

from fraud_engine.evaluation.policy import load_rehearsal_frame
from fraud_engine.models import usd_halves
from fraud_engine.models.ablation import fit_without
from fraud_engine.models.train import apply_categories, fit_categories
from fraud_engine.models.usd_halves import (
    check_reproduces,
    cost_arm,
    fit_arm,
    paired_day_bootstrap,
    resolve_experiment_arms,
)
from test_ablation import MODEL_CFG, make_matrix
from test_policy import COSTS, write_inputs

CALIBRATION = {"n_folds": 4, "ece_bins": 10, "score_clip": 1e-15}


# ---- the bootstrap ---------------------------------------------------------------


def test_identical_costs_give_a_zero_interval():
    rng = np.random.default_rng(0)
    day = np.repeat(np.arange(20), 50)
    cost = rng.exponential(5, size=day.size)

    assert paired_day_bootstrap(day, cost, cost, 500, 0.95, 0) == (0.0, 0.0)


def test_a_constant_difference_is_recovered_exactly():
    rng = np.random.default_rng(0)
    day = np.repeat(np.arange(20), 50)
    reference = rng.exponential(5, size=day.size)

    low, high = paired_day_bootstrap(day, reference + 0.25, reference, 500, 0.95, 0)

    assert low == pytest.approx(250.0)
    assert high == pytest.approx(250.0)


def test_day_level_noise_shared_by_both_arms_cancels():
    """A costly day raises both policies; pairing is what keeps it out of the interval."""
    rng = np.random.default_rng(0)
    day = np.repeat(np.arange(20), 100)
    shared = np.repeat(rng.exponential(200, size=20), 100)
    reference = shared + rng.normal(0, 1, size=day.size)
    arm = shared + 0.1 + rng.normal(0, 1, size=day.size)

    low, high = paired_day_bootstrap(day, arm, reference, 2_000, 0.95, 0)

    assert low < 100 < high
    assert high - low < 200


def test_the_bootstrap_is_reproducible_under_its_seed():
    rng = np.random.default_rng(0)
    day = np.repeat(np.arange(20), 50)
    arm, reference = rng.exponential(5, size=(2, day.size))

    first = paired_day_bootstrap(day, arm, reference, 500, 0.95, 7)

    assert paired_day_bootstrap(day, arm, reference, 500, 0.95, 7) == first
    assert paired_day_bootstrap(day, arm, reference, 500, 0.95, 8) != first


# ---- the reproduction guard ------------------------------------------------------


def scores(split: str, ids: range, values: np.ndarray) -> pd.DataFrame:
    return pd.DataFrame({"TransactionID": list(ids), "split": split, "score": values})


def test_an_identical_refit_passes():
    values = np.linspace(0, 1, 10)
    scored = pd.concat(
        [scores("val_fit", range(10), values), scores("val_cal", range(10, 20), values)]
    )

    check_reproduces(scored, scores("val_fit", range(10), values), "arm")


def test_a_single_differing_score_is_refused():
    values = np.linspace(0, 1, 10)
    changed = values.copy()
    changed[3] += 1e-12

    with pytest.raises(ValueError, match="1 VAL-FIT scores differ"):
        check_reproduces(
            scores("val_fit", range(10), changed), scores("val_fit", range(10), values), "arm"
        )


def test_a_refit_over_different_rows_is_refused():
    values = np.linspace(0, 1, 10)

    with pytest.raises(ValueError, match="different VAL-FIT rows"):
        check_reproduces(
            scores("val_fit", range(1, 11), values), scores("val_fit", range(10), values), "arm"
        )


def test_row_order_is_not_a_difference():
    values = np.linspace(0, 1, 10)
    recorded = scores("val_fit", range(10), values)

    check_reproduces(recorded.iloc[::-1], recorded, "arm")


# ---- fitting ------------------------------------------------------------------------


@pytest.fixture
def matrices() -> dict[str, pd.DataFrame]:
    raw = {name: make_matrix(seed=i) for i, name in enumerate(("train", "val_fit", "val_cal"))}
    raw["val_cal"]["TransactionID"] += 10_000
    vocabulary = fit_categories(raw["train"], min_rows=1)
    return {name: apply_categories(frame, vocabulary) for name, frame in raw.items()}


def test_the_refit_is_the_phase_05_fit(matrices):
    """Same VAL-FIT scores as `ablation.fit_without`, so the guard can pass on real arms."""
    scored, iteration = fit_arm(matrices, ("vel_a",), MODEL_CFG)
    phase_05, phase_05_iteration, _ = fit_without(matrices, ("vel_a",), MODEL_CFG)

    assert iteration == phase_05_iteration
    check_reproduces(scored, phase_05, "arm")


def test_the_refit_scores_the_calibration_slice(matrices):
    scored, _ = fit_arm(matrices, (), MODEL_CFG)

    counts = scored["split"].value_counts()
    assert counts[counts > 0].to_dict() == {"val_fit": 400, "val_cal": 400}


# ---- costing ------------------------------------------------------------------------


@pytest.fixture
def frame(tmp_path: Path) -> pd.DataFrame:
    predictions, interim = write_inputs(tmp_path)
    return load_rehearsal_frame(predictions, interim, "platt")


def arm_scores(frame: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "TransactionID": frame["TransactionID"].to_numpy()[::-1],
            "split": "val_cal",
            "isFraud": frame["isFraud"].to_numpy()[::-1],
            "score": np.clip(frame["calibrated"].to_numpy()[::-1] ** 2, 1e-9, 1 - 1e-9),
        }
    )


def test_an_arm_is_costed_per_transaction_of_the_frame(frame):
    summary, cost = cost_arm(frame, arm_scores(frame), COSTS, 0.01, CALIBRATION, "platt")

    assert set(summary) == {"usd_per_1000", "reviews_per_day", "block_rate"}
    assert cost.shape == (len(frame),)
    assert summary["usd_per_1000"] == pytest.approx(cost.mean() * 1_000)


def test_an_arm_over_different_rows_is_refused(frame):
    with pytest.raises(ValueError, match="different VAL-CAL rows"):
        cost_arm(frame, arm_scores(frame).iloc[1:], COSTS, 0.01, CALIBRATION, "platt")


def test_an_arm_that_disagrees_on_labels_is_refused(frame):
    scored = arm_scores(frame)
    scored.loc[0, "isFraud"] = 1 - scored.loc[0, "isFraud"]

    with pytest.raises(ValueError, match="disagree on VAL-CAL labels"):
        cost_arm(frame, scored, COSTS, 0.01, CALIBRATION, "platt")


# ---- the arms ------------------------------------------------------------------------


def test_the_arms_are_phase_05s(tmp_path, monkeypatch):
    purge_dir = tmp_path / "e1"
    for name in ("purged", "recent", "unpurged"):
        (purge_dir / name).mkdir(parents=True)
        (purge_dir / name / "config.yaml").write_text(
            yaml.safe_dump({"paths": {"features_dir": f"data/e1/{name}/features"}})
        )
    monkeypatch.setattr(
        usd_halves,
        "resolve_arms",
        lambda _: {"full": (), "velocity": ("vel_a", "vel_b"), "amount": ("x",)},
    )

    arms = resolve_experiment_arms({"purge_dir": str(purge_dir), "features_dir": "data/features"})

    assert [(a.name, a.recorded) for a in arms["E1"]] == [
        ("purged", "purge_purged"),
        ("recent", "purge_recent"),
        ("unpurged", "purge_unpurged"),
    ]
    assert arms["E1"][1].features_dir == Path("data/e1/recent/features")
    assert [(a.name, a.dropped, a.recorded) for a in arms["E3"]] == [
        ("full", (), "ablation_full"),
        ("velocity", ("vel_a", "vel_b"), "ablation_velocity"),
    ]
