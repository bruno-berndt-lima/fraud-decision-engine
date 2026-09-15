"""Tests for the VAL-CAL rehearsal.

The joins are the guard that matters: a rules run and a calibration output drawn from
different rows, or an amount matched to the wrong transaction, would still produce
four USD figures, and nothing about them would look wrong.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import mlflow
import numpy as np
import pandas as pd
import pytest
import yaml

from fraud_engine.evaluation import policy
from fraud_engine.evaluation.cost import Costs
from fraud_engine.evaluation.policy import load_rehearsal_frame, rehearse

REPO_ROOT = Path(__file__).resolve().parents[1]

COSTS = Costs(chargeback_fee=25.0, false_positive=15.0, review=1.5, review_friction=3.0, version=1)


def write_inputs(directory: Path, n: int = 4_000, seed: int = 0) -> tuple[Path, Path]:
    """Calibration output, rules predictions and amounts for `n` fake VAL-CAL rows."""
    rng = np.random.default_rng(seed)
    ids = np.arange(n) + 1_000
    day = np.repeat(np.arange(141, 161), n // 20)
    platt = np.clip(rng.beta(0.3, 8, size=n), 0, 1)
    y = rng.binomial(1, platt)

    predictions = directory / "predictions"
    predictions.mkdir(parents=True)

    pd.DataFrame(
        {
            "TransactionID": ids,
            "day": day,
            "fold": (day - 141) // 5,
            "isFraud": y,
            "score": platt**3,
            "platt": platt,
            "isotonic": np.round(platt, 2),
        }
    ).to_parquet(predictions / "calibration.parquet")

    pd.DataFrame(
        {
            "TransactionID": ids[::-1],
            "split": "val_cal",
            "day": day[::-1],
            "isFraud": y[::-1],
            "score": rng.integers(0, 6, size=n) + rng.uniform(0, 0.5, size=n),
        }
    ).to_parquet(predictions / "rules_baseline.parquet")

    interim = directory / "transactions.parquet"
    pd.DataFrame(
        {
            "TransactionID": np.r_[ids, ids.max() + 1 + np.arange(100)],
            "TransactionAmt": rng.lognormal(4, 1.2, size=n + 100),
        }
    ).to_parquet(interim)

    return predictions, interim


def test_the_frame_joins_every_score_to_its_own_transaction(tmp_path):
    predictions, interim = write_inputs(tmp_path)

    frame = load_rehearsal_frame(predictions, interim, "platt").set_index("TransactionID")

    oof = pd.read_parquet(predictions / "calibration.parquet").set_index("TransactionID")
    rules = pd.read_parquet(predictions / "rules_baseline.parquet").set_index("TransactionID")
    amounts = pd.read_parquet(interim).set_index("TransactionID")

    assert len(frame) == len(oof)
    assert np.array_equal(frame["calibrated"], oof.loc[frame.index, "platt"])
    assert np.array_equal(frame["uncalibrated"], oof.loc[frame.index, "score"])
    assert np.array_equal(frame["rules_score"], rules.loc[frame.index, "score"])
    assert np.array_equal(frame["amount"], amounts.loc[frame.index, "TransactionAmt"])


def test_the_selected_method_supplies_the_calibrated_column(tmp_path):
    predictions, interim = write_inputs(tmp_path)

    frame = load_rehearsal_frame(predictions, interim, "isotonic")
    oof = pd.read_parquet(predictions / "calibration.parquet")

    assert np.array_equal(
        frame.set_index("TransactionID")["calibrated"],
        oof.set_index("TransactionID").loc[frame["TransactionID"], "isotonic"],
    )


def test_rules_over_different_rows_are_refused(tmp_path):
    predictions, interim = write_inputs(tmp_path)
    path = predictions / "rules_baseline.parquet"
    pd.read_parquet(path).iloc[1:].to_parquet(path)

    with pytest.raises(ValueError, match="do not cover the same"):
        load_rehearsal_frame(predictions, interim, "platt")


def test_rules_that_disagree_on_labels_are_refused(tmp_path):
    predictions, interim = write_inputs(tmp_path)
    path = predictions / "rules_baseline.parquet"
    rules = pd.read_parquet(path)
    rules.loc[0, "isFraud"] = 1 - rules.loc[0, "isFraud"]
    rules.to_parquet(path)

    with pytest.raises(ValueError, match="disagree on labels"):
        load_rehearsal_frame(predictions, interim, "platt")


def test_a_transaction_without_an_amount_is_refused(tmp_path):
    predictions, interim = write_inputs(tmp_path)
    pd.read_parquet(interim).iloc[5:].to_parquet(interim)

    with pytest.raises(ValueError, match="have no TransactionAmt"):
        load_rehearsal_frame(predictions, interim, "platt")


def test_the_rehearsal_reports_the_four_rows_against_the_rules(tmp_path):
    predictions, interim = write_inputs(tmp_path)
    frame = load_rehearsal_frame(predictions, interim, "platt")

    rows = rehearse(frame, COSTS, capacity=0.01)

    assert list(rows) == ["rules", "naive", "ev_uncalibrated", "ev"]
    assert rows["rules"]["reduction_vs_rules"] == 0
    assert rows["rules"]["block_rate"] == 0
    assert rows["naive"]["reviews_per_day"] == 0
    for summary in rows.values():
        assert summary["reviews_per_day"] <= 0.01 * len(frame) / 20
        assert summary["reduction_vs_rules"] == pytest.approx(
            1 - summary["usd_per_1000"] / rows["rules"]["usd_per_1000"]
        )


def test_calibration_is_the_only_difference_between_the_two_ev_rows(tmp_path):
    predictions, interim = write_inputs(tmp_path)
    frame = load_rehearsal_frame(predictions, interim, "platt")

    same = rehearse(frame.assign(uncalibrated=frame["calibrated"]), COSTS, 0.01)

    assert same["ev"] == same["ev_uncalibrated"]


@pytest.fixture
def stage_config(tmp_path: Path) -> Path:
    predictions, interim = write_inputs(tmp_path)

    cost_matrix = tmp_path / "cost_matrix.yaml"
    shutil.copy(REPO_ROOT / "config" / "cost_matrix.yaml", cost_matrix)

    calibration = tmp_path / "calibration.json"
    calibration.write_text(json.dumps({"selected": "platt"}))

    config = {
        "paths": {
            "predictions_dir": str(predictions),
            "interim": str(interim),
            "calibration": str(calibration),
            "rehearsal": str(tmp_path / "policy_val_cal.json"),
            "cost_matrix": str(cost_matrix),
        },
        "splits": {"val_cal_start": 141, "test_start": 161},
        "tracking": {"store": str(tmp_path / "mlruns"), "experiment_name": "test-policy"},
    }
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(config))
    return path


def test_the_stage_writes_its_record_and_run(stage_config):
    policy.main(stage_config)

    paths = yaml.safe_load(stage_config.read_text())["paths"]
    record = json.loads(Path(paths["rehearsal"]).read_text())

    assert record["split"] == "val_cal"
    assert record["probabilities"] == "platt, out-of-fold"
    assert record["cost_matrix_version"] == 1
    assert list(record["policies"]) == ["rules", "naive", "ev_uncalibrated", "ev"]

    [run] = mlflow.search_runs(experiment_names=["test-policy"], output_format="list")
    assert "ev.usd_per_1000" in run.data.metrics
