"""Tests for the cost-assumption sweeps.

What matters is that every point is a full recomputation: a sweep that held the
rules engine's cost fixed, or quietly missed the headline's own value, would draw a
margin that nothing in the rehearsal agrees with.
"""

from __future__ import annotations

import json
import shutil
from dataclasses import replace
from pathlib import Path

import mlflow
import pandas as pd
import pytest
import yaml

from fraud_engine.evaluation import sensitivity
from fraud_engine.evaluation.policy import load_rehearsal_frame, rehearse
from fraud_engine.evaluation.sensitivity import (
    false_positive_points,
    sweep,
    sweep_values,
    verdict,
)
from test_policy import COSTS, REPO_ROOT, write_inputs

SENSITIVITY = {
    "false_positive": {"min": 5.0, "max": 100.0, "steps": 20},
    "chargeback_fee": {"min": 15.0, "max": 100.0, "steps": 10},
    "review_capacity": {"values": [0.005, 0.01, 0.02]},
}


def test_the_false_positive_grid_is_every_five_dollars():
    values = sweep_values(SENSITIVITY, COSTS, 0.01)

    assert values["false_positive"] == [float(v) for v in range(5, 101, 5)]


def test_a_grid_that_misses_the_headline_value_gains_it():
    """15 to 100 in ten points skips $25, the fee the headline was computed at."""
    fees = sweep_values(SENSITIVITY, COSTS, 0.01)["chargeback_fee"]

    assert len(fees) == 11
    assert 25.0 in fees
    assert fees[0] == 15.0 and fees[-1] == 100.0


def test_capacities_come_from_the_matrix_and_must_include_the_headline():
    assert sweep_values(SENSITIVITY, COSTS, 0.01)["review_capacity"] == [0.005, 0.01, 0.02]

    with pytest.raises(ValueError, match="not among the sweep values"):
        sweep_values(SENSITIVITY, COSTS, 0.03)


@pytest.fixture
def frame(tmp_path: Path) -> pd.DataFrame:
    predictions, interim = write_inputs(tmp_path)
    return load_rehearsal_frame(predictions, interim, "platt")


@pytest.fixture
def small_values() -> dict:
    return {
        "false_positive": [5.0, 15.0, 100.0],
        "chargeback_fee": [15.0, 25.0, 100.0],
        "review_capacity": [0.005, 0.01, 0.02],
    }


def test_every_row_is_costed_at_every_point(frame, small_values):
    points = sweep(frame, COSTS, 0.01, small_values)

    assert len(points) == 9 * 4
    assert (points.groupby(["assumption", "policy"])["is_base"].sum() == 1).all()


def test_the_base_point_reproduces_the_rehearsal(frame, small_values):
    points = sweep(frame, COSTS, 0.01, small_values)
    rehearsal = rehearse(frame, COSTS, 0.01)

    for _, row in points[points["is_base"]].iterrows():
        assert row["usd_per_1000"] == pytest.approx(rehearsal[row["policy"]]["usd_per_1000"])


def test_each_point_moves_only_its_own_assumption(frame, small_values):
    points = sweep(frame, COSTS, 0.01, small_values)
    at_100 = points[(points["assumption"] == "false_positive") & (points["value"] == 100.0)]

    expected = rehearse(frame, replace(COSTS, false_positive=100.0), 0.01)

    for _, row in at_100.iterrows():
        assert row["usd_per_1000"] == pytest.approx(expected[row["policy"]]["usd_per_1000"])


def test_the_rules_engine_is_recomputed_where_its_cost_can_move(frame, small_values):
    """It never blocks, so the false-positive cost cannot touch it; the fee and the
    capacity both do."""
    points = sweep(frame, COSTS, 0.01, small_values)
    rules = points[points["policy"] == "rules"]

    def spread(assumption: str) -> int:
        return rules[rules["assumption"] == assumption]["usd_per_1000"].round(6).nunique()

    assert spread("false_positive") == 1
    assert spread("chargeback_fee") == 3
    assert spread("review_capacity") == 3


def test_a_higher_false_positive_cost_makes_the_ev_policy_block_less(frame, small_values):
    points = sweep(frame, COSTS, 0.01, small_values)
    ev = points[(points["policy"] == "ev") & (points["assumption"] == "false_positive")]

    assert ev.sort_values("value")["block_rate"].is_monotonic_decreasing


def test_the_verdict_reports_the_worst_point_against_both_bars():
    points = pd.DataFrame(
        {
            "assumption": ["false_positive"] * 3 + ["chargeback_fee"] * 2,
            "value": [5.0, 15.0, 100.0, 15.0, 100.0],
            "policy": ["ev"] * 5,
            "reduction_vs_rules": [0.6, 0.5, 0.08, 0.3, 0.2],
        }
    )

    verdicts = verdict(points)

    assert verdicts["false_positive"]["min_reduction"] == 0.08
    assert verdicts["false_positive"]["at_value"] == 100.0
    assert not verdicts["false_positive"]["above_win_everywhere"]
    assert verdicts["false_positive"]["above_noise_everywhere"]
    assert verdicts["chargeback_fee"]["above_win_everywhere"]


def test_the_chart_input_pairs_rules_and_ev_by_value(frame, small_values):
    points = sweep(frame, COSTS, 0.01, small_values)

    chart = false_positive_points(points)

    assert chart["value"].tolist() == [5.0, 15.0, 100.0]
    at_15 = chart[chart["value"] == 15.0].iloc[0]
    rehearsal = rehearse(frame, COSTS, 0.01)
    assert at_15["rules_usd"] == pytest.approx(rehearsal["rules"]["usd_per_1000"])
    assert at_15["ev_usd"] == pytest.approx(rehearsal["ev"]["usd_per_1000"])
    assert at_15["reduction"] == pytest.approx(rehearsal["ev"]["reduction_vs_rules"])


def test_the_stage_writes_its_record_run_and_chart(tmp_path):
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
            "sensitivity": str(tmp_path / "sensitivity.json"),
            "figures_dir": str(tmp_path / "figures"),
            "cost_matrix": str(cost_matrix),
        },
        "splits": {"val_cal_start": 141, "test_start": 161},
        "tracking": {"store": str(tmp_path / "mlruns"), "experiment_name": "test-sensitivity"},
    }
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.safe_dump(config))

    sensitivity.main(config_path)

    record = json.loads((tmp_path / "sensitivity.json").read_text())
    assert set(record["verdicts"]) == {"false_positive", "chargeback_fee", "review_capacity"}
    assert len(record["points"]) == (20 + 11 + 3) * 4
    assert (tmp_path / "figures" / sensitivity.FIGURE).stat().st_size > 0

    [run] = mlflow.search_runs(experiment_names=["test-sensitivity"], output_format="list")
    assert "false_positive.min_reduction" in run.data.metrics
