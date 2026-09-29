"""Tests for the retraining trigger's replay.

Each clarification `docs/monitoring.md` §7 registered before the run, against a record
whose answer is known: the clock, which windows each condition may read, a partial window
that reaches a band and does not fire, ties going to the signal, and the retrain
arithmetic, shown to give back Phase 02's own layout on the day the shipped model's
calibration labels mature.
"""

import pytest

from fraud_engine.monitoring.trigger import (
    check_drift_record,
    clock,
    firings,
    first,
    read_windows,
    retrain,
    thresholds,
)

# Phase 02's layout: train 1-90, purge 91-120, VAL-FIT 121-140, VAL-CAL 141-160, test 161-182.
SPLITS = {
    "gap_days": 30,
    "train_start": 1,
    "val_fit_start": 121,
    "val_cal_start": 141,
    "test_start": 161,
    "test_end": 182,
}
PSI = {"bins": 10, "epsilon": 1e-6, "moderate": 0.1, "significant": 0.25}
TRIGGER = {
    "top_k": 2,
    "feature_psi": "moderate",
    "score_psi": "significant",
    "pr_auc_drop": 0.10,
    "cadence_days": 90,
}
CFG = {"psi": PSI, "trigger": TRIGGER}
LIMITS = thresholds(TRIGGER, PSI)
DAYS = clock(SPLITS, TRIGGER["cadence_days"])


def labelled(first_day, last_day, weighted, partial=False):
    return {"first_day": first_day, "last_day": last_day, "partial": partial,
            "weighted_psi": weighted}  # fmt: skip


def horizon(first_day, last_day, weighted, score, partial=False):
    return labelled(first_day, last_day, weighted, partial) | {"prediction": {"score_psi": score}}


def drift(labelled_windows=(), horizon_windows=()):
    return {
        "psi": PSI,
        "weighted_columns": [{"feature": "C13"}, {"feature": "C1"}],
        "labelled": list(labelled_windows),
        "horizon": list(horizon_windows),
    }


QUIET = drift(
    [labelled(156, 160, 0.9), labelled(161, 165, 0.02), labelled(181, 182, 0.01, partial=True)],
    [horizon(213, 240, 0.02, 0.02), horizon(241, 268, 0.03, 0.03), horizon(381, 395, 0.12, 0.02, partial=True)],
)  # fmt: skip


# ---- the rule's values ------------------------------------------------------------


def test_the_psi_conditions_are_the_convention_s_named_bands():
    assert LIMITS == {"feature_psi": 0.1, "score_psi": 0.25}


@pytest.mark.parametrize("name", ["stable", 0.1])
def test_a_threshold_that_is_not_a_band_is_refused(name):
    with pytest.raises(ValueError, match="must name a band"):
        thresholds(TRIGGER | {"feature_psi": name}, PSI)


def test_a_drift_record_measured_as_the_rule_reads_passes():
    check_drift_record(QUIET, CFG)


def test_a_drift_record_weighing_other_columns_is_refused():
    with pytest.raises(ValueError, match="top_k"):
        check_drift_record(QUIET, CFG | {"trigger": TRIGGER | {"top_k": 10}})


def test_a_drift_record_read_against_other_bands_is_refused():
    with pytest.raises(ValueError, match="moderate"):
        check_drift_record(QUIET, CFG | {"psi": PSI | {"moderate": 0.2}})


# ---- the clock --------------------------------------------------------------------


def test_the_clock_is_the_split_layout():
    assert DAYS == {
        "training_ends": 90,
        "deployment": 161,
        "maturity_days": 30,
        "calibration_labels_mature_on": 190,
        "baseline_labels_mature_on": 212,
        "cadence_fires_on": 251,
    }


# ---- which windows each condition reads -------------------------------------------


def test_windows_before_deployment_are_not_read():
    """Validation's 0.9 would reach any band; the model was not live yet."""
    windows = read_windows(QUIET, LIMITS, DAYS)
    assert min(w["first_day"] for w in windows) == 161


def test_a_window_straddling_deployment_is_refused():
    with pytest.raises(ValueError, match="straddles"):
        read_windows(drift([labelled(158, 162, 0.01)]), LIMITS, DAYS)


def test_the_score_condition_reads_the_horizon_only():
    windows = read_windows(QUIET, LIMITS, DAYS)
    assert all(("score_psi" in w) == (w["horizon"] == "horizon") for w in windows)


def test_the_boundary_value_reaches_the_band():
    (window,) = read_windows(drift([], [horizon(213, 240, 0.1, 0.25)]), LIMITS, DAYS)
    assert window["feature_psi"]["fires"] and window["score_psi"]["fires"]


def test_a_partial_window_reaches_a_band_and_never_fires():
    partial = read_windows(QUIET, LIMITS, DAYS)[-1]
    assert partial["partial"]
    assert partial["feature_psi"] == {"value": 0.12, "reaches": True, "fires": False}


def test_the_labelled_condition_never_fires_and_says_when_it_could_have_been_read():
    by_day = {w["first_day"]: w["pr_auc"] for w in read_windows(QUIET, LIMITS, DAYS)}
    assert not any(entry["fires"] for entry in by_day.values())
    assert by_day[161] == {
        "fires": False,
        "labels_mature_on": 195,
        "why_not": "inside the baseline",
    }
    assert by_day[213] == {"fires": False, "labels_mature_on": 270, "why_not": "no labels"}


# ---- which fires first ------------------------------------------------------------


def test_with_every_window_quiet_the_cadence_fires_first():
    windows = read_windows(QUIET, LIMITS, DAYS)
    fired = firings(windows, DAYS)
    assert fired == {1: None, 2: None, 3: None, 4: 251}
    assert first(fired, windows) == {
        "condition": 4,
        "name": "cadence",
        "day": 251,
        "window": {"horizon": "horizon", "first_day": 241, "last_day": 268, "partial": False},
    }


def test_a_window_is_read_on_its_last_day():
    windows = read_windows(drift([], [horizon(213, 240, 0.3, 0.02)]), LIMITS, DAYS)
    assert firings(windows, DAYS)[1] == 240
    assert first(firings(windows, DAYS), windows)["condition"] == 1


def test_a_tie_with_the_cadence_goes_to_the_signal():
    windows = read_windows(drift([], [horizon(224, 251, 0.02, 0.3)]), LIMITS, DAYS)
    assert first(firings(windows, DAYS), windows)["condition"] == 2


def test_a_firing_day_no_window_covers_names_no_window():
    windows = read_windows(drift([labelled(161, 165, 0.02)]), LIMITS, DAYS)
    assert first(firings(windows, DAYS), windows)["window"] is None


# ---- the retrain ------------------------------------------------------------------


def test_the_retrain_arithmetic_gives_back_phase_02_s_layout():
    """Day 190 is when VAL-CAL's labels mature: exactly the data the shipped model used."""
    layout = retrain(DAYS["calibration_labels_mature_on"], SPLITS)
    assert layout["labelled_through"] == 160
    assert layout["under_the_layout"] == {
        "training_ends": 90,
        "val_fit": [121, 140],
        "val_cal": [141, 160],
        "days_stale_on_the_trigger_day": 100,
    }


def test_a_later_trigger_moves_the_whole_layout_with_it():
    early, late = retrain(251, SPLITS), retrain(281, SPLITS)
    assert late["labelled_through"] - early["labelled_through"] == 30
    assert (
        late["under_the_layout"]["training_ends"] - early["under_the_layout"]["training_ends"] == 30
    )
