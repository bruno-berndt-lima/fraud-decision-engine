"""Tests for the calibration of the shipped booster.

The folds are the guard that matters: a fold that shared a day with another, or
quietly came up short, would let the cross-fit score a calibrator on rows it was
fitted on, and every out-of-fold figure would flatter it without a symptom.
"""

from __future__ import annotations

import json
from pathlib import Path

import mlflow
import numpy as np
import pandas as pd
import pytest
import yaml
from scipy.special import expit
from sklearn.isotonic import IsotonicRegression

from fraud_engine.evaluation import reliability
from fraud_engine.models import calibrate as calibration_stage
from fraud_engine.models.calibrate import (
    METHODS,
    apply_calibrator,
    apply_isotonic,
    apply_platt,
    assign_folds,
    calibrate,
    calibration_metrics,
    check_clip,
    cross_fit,
    expected_calibration_error,
    fit_calibrator,
    fit_isotonic,
    fit_platt,
    logit,
    reliability_bins,
    select_method,
)

CLIP = 1e-15


def days(first: int, last: int, rows_per_day: int = 3) -> pd.Series:
    return pd.Series(np.repeat(np.arange(first, last + 1), rows_per_day), name="day")


def test_folds_are_contiguous_blocks_in_calendar_order():
    day = days(141, 160)
    fold = assign_folds(day, 4)

    spans = day.groupby(fold).agg(["min", "max"])

    assert spans.to_dict("index") == {
        0: {"min": 141, "max": 145},
        1: {"min": 146, "max": 150},
        2: {"min": 151, "max": 155},
        3: {"min": 156, "max": 160},
    }


def test_no_day_is_split_across_folds():
    day = days(141, 160).sample(frac=1, random_state=0)
    fold = assign_folds(day, 4)

    assert fold.groupby(day).nunique().max() == 1


def test_folds_align_to_the_input_index():
    day = days(141, 160).sample(frac=1, random_state=0)

    assert assign_folds(day, 4).index.equals(day.index)


def test_days_that_do_not_divide_are_refused():
    with pytest.raises(ValueError, match="do not divide"):
        assign_folds(days(141, 160), 3)


def test_a_missing_day_is_refused():
    day = days(141, 160)

    with pytest.raises(ValueError, match=r"missing \[150\]"):
        assign_folds(day[day != 150], 4)


def test_a_single_fold_is_refused():
    with pytest.raises(ValueError, match="at least two"):
        assign_folds(days(141, 160), 1)


def test_logit_is_finite_at_zero_and_one():
    z = logit(np.array([0.0, 1.0]), CLIP)

    assert np.isfinite(z).all()


def test_the_clip_leaves_ordinary_scores_untouched():
    score = np.array([1e-13, 0.3, 1 - 1e-13])

    assert np.allclose(expit(logit(score, CLIP)), score, rtol=1e-6)


def test_platt_recovers_the_identity_on_calibrated_scores():
    rng = np.random.default_rng(0)
    score = expit(rng.normal(-3, 2, size=200_000))
    y = rng.binomial(1, score)

    a, b = fit_platt(score, y, CLIP)

    assert a == pytest.approx(1, abs=0.05)
    assert b == pytest.approx(0, abs=0.05)


def test_platt_recovers_a_known_distortion():
    """The shape it exists to fix: a model whose log-odds are too extreme."""
    rng = np.random.default_rng(0)
    true_logit = rng.normal(-3, 1, size=200_000)
    y = rng.binomial(1, expit(true_logit))
    overconfident = expit(3 * true_logit + 1)

    a, b = fit_platt(overconfident, y, CLIP)

    assert a == pytest.approx(1 / 3, abs=0.02)
    assert b == pytest.approx(-1 / 3, abs=0.05)


def test_platt_is_the_unpenalised_maximum_likelihood_fit():
    """Both score equations sit at zero, to solver tolerance. The default L2
    penalty would leave the slope's off by roughly the slope itself."""
    rng = np.random.default_rng(1)
    score = expit(rng.normal(-1, 1.5, size=500))
    y = rng.binomial(1, score)

    a, b = fit_platt(score, y, CLIP)

    z = logit(score, CLIP)
    residual = y - expit(a * z + b)

    assert abs(residual.sum()) < 1e-2
    assert abs((residual * z).sum()) < 1e-2


def test_platt_preserves_the_ranking():
    rng = np.random.default_rng(0)
    score = expit(rng.normal(-3, 2, size=10_000))
    y = rng.binomial(1, score)

    calibrated = apply_platt(score, *fit_platt(score, y, CLIP), CLIP)

    order = np.argsort(score)
    assert (np.diff(calibrated[order]) >= 0).all()


def test_a_reversed_score_is_refused():
    rng = np.random.default_rng(0)
    score = expit(rng.normal(-3, 2, size=10_000))
    y = rng.binomial(1, score)

    with pytest.raises(ValueError, match="reverse the ranking"):
        fit_platt(1 - score, y, CLIP)


def calibrated_sample(size: int, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    score = expit(rng.normal(-3, 2, size=size))
    return score, rng.binomial(1, score)


def test_isotonic_breakpoints_reproduce_predict():
    score, y = calibrated_sample(5_000)
    new = np.r_[0.0, np.linspace(0, 1, 1_001), 1.0]

    model = IsotonicRegression(y_min=0.0, y_max=1.0, out_of_bounds="clip").fit(score, y)

    assert np.allclose(apply_isotonic(new, fit_isotonic(score, y)), model.predict(new))


def test_isotonic_is_bounded_and_non_decreasing_beyond_its_range():
    score, y = calibrated_sample(5_000)
    table = fit_isotonic(score, y)
    new = np.linspace(-0.5, 1.5, 2_001)

    p = apply_isotonic(new, table)

    assert np.isfinite(p).all()
    assert ((p >= 0) & (p <= 1)).all()
    assert (np.diff(p) >= 0).all()


def test_isotonic_fits_a_shape_platt_cannot():
    """A step in the true probability: no sigmoid of the logit bends that sharply."""
    rng = np.random.default_rng(0)
    score = rng.uniform(0, 1, size=100_000)
    truth = np.where(score < 0.5, 0.02, 0.6)
    y = rng.binomial(1, truth)

    platt = calibrate("platt", score, y, score, CLIP)
    isotonic = calibrate("isotonic", score, y, score, CLIP)

    assert np.mean((isotonic - truth) ** 2) < np.mean((platt - truth) ** 2) / 10


def test_an_unknown_method_is_refused():
    score, y = calibrated_sample(1_000)

    with pytest.raises(ValueError, match="unknown calibration method"):
        calibrate("beta", score, y, score, CLIP)


def test_ece_is_zero_when_every_bin_matches_its_rate():
    y = np.r_[np.zeros(90), np.ones(10), np.zeros(50), np.ones(50)]
    p = np.r_[np.full(100, 0.1), np.full(100, 0.5)]

    assert expected_calibration_error(y, p, bins=10) == pytest.approx(0)


def test_ece_is_the_row_weighted_gap():
    y = np.r_[np.zeros(70), np.ones(30), np.zeros(300)]
    p = np.r_[np.full(100, 0.1), np.full(300, 0.1)]

    # One tied value, so one bin: predicted 0.1 against a rate of 30 / 400.
    assert expected_calibration_error(y, p, bins=10) == pytest.approx(abs(0.1 - 30 / 400))


def test_tied_probabilities_never_split_across_bins():
    p = np.r_[np.full(500, 0.01), np.linspace(0.02, 0.9, 500)]
    y = np.zeros_like(p)

    table = reliability_bins(y, p, bins=10)

    assert table["rows"].iloc[0] == 500
    assert table["rows"].sum() == len(p)


def test_a_single_probability_is_one_bin_not_an_error():
    table = reliability_bins(np.r_[0, 1, 0, 0], np.full(4, 0.25), bins=10)

    assert len(table) == 1
    assert table["rows"].iloc[0] == 4


def test_log_loss_is_finite_when_isotonic_predicts_zero():
    metrics = calibration_metrics(np.array([1, 0, 0]), np.array([0.0, 0.0, 1.0]), bins=2)

    assert np.isfinite(list(metrics.values())).all()


@pytest.fixture
def folded() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    score, y = calibrated_sample(40_000)
    fold = np.repeat(np.arange(4), 10_000)
    return score, y, fold


@pytest.mark.parametrize("method", METHODS)
def test_every_row_is_predicted_exactly_once(folded, method):
    score, y, fold = folded

    probabilities, per_fold = cross_fit(score, y, fold, method, CLIP, bins=10)

    assert np.isfinite(probabilities).all()
    assert per_fold["fold"].tolist() == [0, 1, 2, 3]
    assert per_fold["rows"].sum() == len(score)
    assert per_fold["positives"].sum() == y.sum()


@pytest.mark.parametrize("method", METHODS)
def test_a_folds_own_labels_never_reach_its_predictions(folded, method):
    """The leak this function exists to prevent: flip every label in fold 0 and
    its out-of-fold probabilities must not move."""
    score, y, fold = folded
    flipped = np.where(fold == 0, 1 - y, y)

    before, _ = cross_fit(score, y, fold, method, CLIP, bins=10)
    after, _ = cross_fit(score, flipped, fold, method, CLIP, bins=10)

    assert np.array_equal(before[fold == 0], after[fold == 0])
    assert not np.array_equal(before[fold != 0], after[fold != 0])


@pytest.mark.parametrize("method", METHODS)
def test_per_fold_metrics_are_those_of_the_out_of_fold_probabilities(folded, method):
    score, y, fold = folded

    probabilities, per_fold = cross_fit(score, y, fold, method, CLIP, bins=10)

    held = fold == 2
    expected = calibration_metrics(y[held], probabilities[held], bins=10)
    row = per_fold.set_index("fold").loc[2]

    assert {name: row[name] for name in expected} == pytest.approx(expected)


def briers(values: list[float]) -> pd.DataFrame:
    return pd.DataFrame({"fold": range(len(values)), "brier": values})


def test_isotonic_is_selected_only_when_it_wins_every_fold():
    platt = briers([0.030, 0.030, 0.030, 0.030])

    assert select_method({"platt": platt, "isotonic": briers([0.029] * 4)}) == "isotonic"
    assert select_method({"platt": platt, "isotonic": briers([0.020] * 3 + [0.031])}) == "platt"


def test_a_tie_goes_to_platt():
    table = briers([0.030] * 4)

    assert select_method({"platt": table, "isotonic": table.copy()}) == "platt"


def test_selection_refuses_tables_over_different_folds():
    with pytest.raises(ValueError, match="folds differ"):
        select_method({"platt": briers([0.03] * 4), "isotonic": briers([0.02] * 3)})


def test_a_clip_that_reaches_real_scores_is_refused():
    with pytest.raises(ValueError, match="reaches 2 scores"):
        check_clip(np.array([1e-9, 0.5, 1 - 1e-9]), 1e-7)


def test_a_clip_below_every_score_passes():
    check_clip(np.array([1e-13, 0.5, 1 - 1e-13]), CLIP)


@pytest.mark.parametrize("method", METHODS)
def test_a_calibrator_survives_a_json_round_trip(method):
    """What ships is the JSON, so the JSON must reproduce the fit exactly."""
    score, y = calibrated_sample(5_000)
    fitted = fit_calibrator(method, score, y, CLIP)
    loaded = json.loads(json.dumps(fitted))

    assert np.array_equal(apply_calibrator(loaded, score), apply_calibrator(fitted, score))


def test_a_calibrator_naming_an_unknown_method_is_refused():
    with pytest.raises(ValueError, match="unknown calibration method"):
        apply_calibrator({"method": "beta"}, np.array([0.5]))


# ---- the stage, end to end on synthetic predictions ---------------------------


@pytest.fixture
def stage_config(tmp_path: Path) -> Path:
    """A config whose every path is under `tmp_path`, over four folds of fake VAL-CAL."""
    rng = np.random.default_rng(0)
    day = np.repeat(np.arange(141, 161), 500)
    score = expit(rng.normal(-5, 2, size=len(day)))
    y = rng.binomial(1, expit(0.5 * logit(score, CLIP) + 0.5))

    predictions = tmp_path / "predictions"
    predictions.mkdir()
    pd.DataFrame(
        {
            "TransactionID": np.arange(len(day)),
            "split": "val_cal",
            "day": day,
            "isFraud": y,
            "score": score,
        }
    ).to_parquet(predictions / "lightgbm_tuned.parquet")

    cost_matrix = tmp_path / "cost_matrix.yaml"
    cost_matrix.write_text("version: 1\n")

    config = {
        "paths": {
            "predictions_dir": str(predictions),
            "calibrator": str(tmp_path / "models" / "calibrator.json"),
            "calibration": str(tmp_path / "calibration.json"),
            "figures_dir": str(tmp_path / "figures"),
            "cost_matrix": str(cost_matrix),
        },
        "splits": {"val_cal_start": 141, "test_start": 161},
        "tracking": {"store": str(tmp_path / "mlruns"), "experiment_name": "test-calibration"},
        "model": {"tuned": {"num_leaves": 31}},
        "calibration": {"n_folds": 4, "ece_bins": 10, "score_clip": CLIP},
    }
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(config))
    return path


def test_the_stage_ships_what_its_record_selected(stage_config):
    calibration_stage.main(stage_config)
    config = yaml.safe_load(stage_config.read_text())
    paths = config["paths"]

    record = json.loads(Path(paths["calibration"]).read_text())
    calibrator = json.loads(Path(paths["calibrator"]).read_text())

    assert calibrator["method"] == record["selected"]
    assert calibrator["fitted_on"] == "val_cal"
    assert record["rows"] == 10_000
    assert set(record["per_fold"]) == {"uncalibrated", *METHODS}
    assert all(len(rows) == 4 for rows in record["per_fold"].values())

    out_of_fold = pd.read_parquet(Path(paths["predictions_dir"]) / "calibration.parquet")
    assert len(out_of_fold) == 10_000
    assert out_of_fold[list(METHODS)].notna().all().all()


def test_the_stage_is_a_tracked_run(stage_config):
    calibration_stage.main(stage_config)

    [run] = mlflow.search_runs(experiment_names=["test-calibration"], output_format="list")

    assert run.data.params["selected"] in METHODS
    assert "platt.brier" in run.data.metrics


def test_the_diagram_is_drawn_from_the_stage_output(stage_config):
    calibration_stage.main(stage_config)
    reliability.main(stage_config)

    figures_dir = Path(yaml.safe_load(stage_config.read_text())["paths"]["figures_dir"])
    assert (figures_dir / reliability.FIGURE).stat().st_size > 0
