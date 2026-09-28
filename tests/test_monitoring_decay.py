"""Tests for the decay chart.

What the stage reads, the proof that it reads what the records measured, and the
registered reading rules — each window's verdict is decided in code, so the rules are
pinned here rather than applied by eye to a figure.
"""

import inspect

import numpy as np
import pandas as pd
import pytest

from fraud_engine.evaluation.metrics import pr_auc, roc_auc
from fraud_engine.models.train import LABEL
from fraud_engine.monitoring import decay
from fraud_engine.monitoring.decay import (
    attach_identity,
    check_records,
    composition_bound,
    day_bootstrap,
    load_scores,
    read_windows,
    window_table,
)

BOOTSTRAP = {"resamples": 200, "interval": 0.95, "seed": 0}


def labelled(days, fraud_rate=0.05, signal=0.8, seed=0):
    rng = np.random.default_rng(seed)
    y = (rng.random(len(days)) < fraud_rate).astype(int)
    return pd.DataFrame(
        {
            "TransactionID": np.arange(len(days)) + 10_000 * seed,
            "day": days,
            LABEL: y,
            "score": rng.random(len(days)) + signal * y,
        }
    )


# ---- what is read ---------------------------------------------------------------


@pytest.fixture
def predictions(tmp_path):
    rng = np.random.default_rng(0)
    validation = pd.concat(
        [
            labelled(rng.integers(1, 91, 300), seed=1).assign(split="train"),
            labelled(rng.integers(121, 141, 300), seed=2).assign(split="val_fit"),
            labelled(rng.integers(141, 161, 300), seed=3).assign(split="val_cal"),
        ],
        ignore_index=True,
    )
    validation.to_parquet(tmp_path / "lightgbm_tuned.parquet", index=False)

    test = labelled(rng.integers(161, 183, 300), seed=4)
    test.drop(columns="score").assign(
        uncalibrated=test["score"], calibrated=test["score"] / 3, amount=10.0
    ).to_parquet(tmp_path / "headline_test.parquet", index=False)
    return tmp_path, validation, test


def test_the_three_slices_are_read_and_train_is_not(predictions):
    directory, _, _ = predictions
    scores = load_scores(directory, "lightgbm_tuned")
    assert sorted(scores["split"].unique()) == ["test", "val_cal", "val_fit"]


def test_test_is_read_as_the_raw_score_not_the_probability(predictions):
    directory, _, test = predictions
    scores = load_scores(directory, "lightgbm_tuned")
    read = scores[scores["split"] == "test"].set_index("TransactionID")["score"]
    assert read.sort_index().equals(test.set_index("TransactionID")["score"].sort_index())


def test_an_untouched_test_says_what_to_run(predictions):
    directory, _, _ = predictions
    (directory / "headline_test.parquet").unlink()
    with pytest.raises(FileNotFoundError, match="make headline"):
        load_scores(directory, "lightgbm_tuned")


def test_a_scored_row_without_identity_is_refused(tmp_path):
    scores = labelled(np.full(4, 121))
    pd.DataFrame({"TransactionID": scores["TransactionID"][:3], "has_identity": True}).to_parquet(
        tmp_path / "interim.parquet", index=False
    )
    with pytest.raises(ValueError, match="no has_identity"):
        attach_identity(scores, tmp_path / "interim.parquet")


def test_the_stage_never_loads_a_model():
    """A figure is a view of what a run said; a booster here could say something new."""
    source = inspect.getsource(decay)
    assert "Booster" not in source
    assert "load_model" not in source


# ---- the proof against the records --------------------------------------------------


def records_for(scores):
    block = {
        split: {
            "pr_auc": pr_auc(part[LABEL], part["score"]),
            "roc_auc": roc_auc(part[LABEL], part["score"]),
        }
        for split, part in scores.groupby("split")
    }
    return {"splits": block}, {"measured": {"model": block["test"]}}


def test_vectors_that_match_their_records_pass(predictions):
    directory, _, _ = predictions
    scores = load_scores(directory, "lightgbm_tuned")
    check_records(scores, *records_for(scores))


def test_a_vector_that_differs_from_its_record_at_all_is_refused(predictions):
    directory, _, _ = predictions
    scores = load_scores(directory, "lightgbm_tuned")
    model, headline = records_for(scores)
    headline["measured"]["model"]["pr_auc"] += 1e-12
    with pytest.raises(ValueError, match="test pr_auc"):
        check_records(scores, model, headline)


# ---- the bar ----------------------------------------------------------------------


def test_the_bootstrap_is_a_rerunnable_number():
    frame = labelled(np.repeat(np.arange(121, 126), 400))
    assert day_bootstrap(frame, **BOOTSTRAP) == day_bootstrap(frame, **BOOTSTRAP)


def test_one_day_leaves_nothing_to_resample():
    """Days are drawn whole: a single day can only ever redraw itself."""
    frame = labelled(np.full(1_000, 130))
    intervals = day_bootstrap(frame, **BOOTSTRAP)
    point = pr_auc(frame[LABEL], frame["score"])
    assert intervals["pr_auc"] == pytest.approx((point, point))


def test_the_interval_holds_the_point_on_a_stable_window():
    frame = labelled(np.repeat(np.arange(121, 126), 2_000))
    low, high = day_bootstrap(frame, **BOOTSTRAP)["pr_auc"]
    assert low <= pr_auc(frame[LABEL], frame["score"]) <= high


# ---- the table --------------------------------------------------------------------


def slices(identity_share=0.18):
    rng = np.random.default_rng(0)
    parts = []
    for split, first, last, seed in (
        ("val_fit", 121, 140, 1),
        ("val_cal", 141, 160, 2),
        ("test", 161, 182, 3),
    ):
        days = np.repeat(np.arange(first, last + 1), 150)
        parts.append(labelled(days, seed=seed).assign(split=split))
    frame = pd.concat(parts, ignore_index=True)
    return frame.assign(has_identity=rng.random(len(frame)) < identity_share)


def test_the_registered_layout_comes_out_of_the_table():
    table = window_table(slices(), 5, 121, 90, BOOTSTRAP)

    assert len(table) == 13
    assert table["partial"].tolist() == [False] * 12 + [True]
    assert table["slice"].tolist() == ["val_fit"] * 4 + ["val_cal"] * 4 + ["test"] * 5
    assert table["optimistic"].tolist() == [True] * 4 + [False] * 9


def test_days_since_training_is_the_window_midpoint():
    table = window_table(slices(), 5, 121, 90, BOOTSTRAP)
    assert table["days_since_training"].iloc[0] == 123 - 90
    assert table["days_since_training"].iloc[-1] == 181.5 - 90


def test_a_window_straddling_two_slices_is_refused():
    with pytest.raises(ValueError, match="straddles"):
        window_table(slices(), 7, 121, 90, BOOTSTRAP)


# ---- the reading rules ------------------------------------------------------------

BASELINE = {"pr_auc": 0.52, "roc_auc": 0.89}
BOUND = {"per_point": 0.0064, "low": 0.1729, "high": 0.1787}


def window(pr, pr_high, roc_high, share=0.175):
    return {
        "pr_auc": pr,
        "pr_auc_high": pr_high,
        "roc_auc_high": roc_high,
        "identity_share": share,
    }


def verdicts(*windows):
    return read_windows(pd.DataFrame(windows), BASELINE, BOUND)


def test_an_interval_that_reaches_the_baseline_is_no_decline():
    assert verdicts(window(0.45, 0.53, 0.85))["reading"].tolist() == ["no decline"]


def test_a_fall_roc_auc_does_not_share_is_prevalence():
    assert verdicts(window(0.45, 0.50, 0.90))["reading"].tolist() == ["prevalence"]


def test_a_fall_both_metrics_share_is_decline():
    assert verdicts(window(0.45, 0.50, 0.85))["reading"].tolist() == ["decline"]


def test_a_share_inside_the_validated_range_can_explain_nothing():
    table = verdicts(window(0.45, 0.50, 0.85, share=0.175))
    assert table["mix_bound"].iloc[0] == 0
    assert table["beyond_mix"].iloc[0]


def test_a_share_outside_the_range_explains_its_distance_times_the_rate():
    table = verdicts(window(0.45, 0.50, 0.85, share=0.1629))
    assert table["mix_bound"].iloc[0] == pytest.approx(1.0 * BOUND["per_point"])


def test_more_identity_than_validation_cannot_explain_a_fall():
    """Identity rows are the ones ranked best: more of them would raise PR-AUC."""
    table = verdicts(window(0.45, 0.50, 0.85, share=0.2829))
    assert table["mix_bound"].iloc[0] == 0
    assert table["beyond_mix"].iloc[0]


def test_a_decline_mix_could_cover_is_not_beyond_it():
    """Ten points below the range could explain 0.064; this window is 0.05 short."""
    table = verdicts(window(0.47, 0.50, 0.85, share=0.0729))
    assert table["mix_bound"].iloc[0] == pytest.approx(0.064)
    assert not table["beyond_mix"].iloc[0]


def test_only_a_decline_is_ever_beyond_mix():
    table = verdicts(window(0.45, 0.50, 0.90), window(0.51, 0.55, 0.85))
    assert not table["beyond_mix"].any()


def test_the_bound_is_e5_s_part_per_point_of_shift():
    record = {
        "identity_share": {"train": 0.29, "validation": 0.18, "val_fit": 0.181, "val_cal": 0.179},
        "decomposition": {"compositional": 0.055},
    }
    bound = composition_bound(record)
    assert bound["per_point"] == pytest.approx(0.055 / 11)
    assert (bound["low"], bound["high"]) == (0.179, 0.181)
