"""Tests for the baseline comparison figure stage.

Synthetic predictions throughout — no parquet from the pipeline is read, since
CI has no data. What is under test is the selection and the alignment check, not
how a curve looks; ``test_plots.py`` covers the drawing.
"""

import numpy as np
import pandas as pd
import pytest

from fraud_engine.evaluation.figures import (
    BASELINE_RUNS,
    FIGURE_SPLIT,
    PR_CURVE_FIGURE,
    RECALL_FIGURE,
    align,
    load_predictions,
)
from fraud_engine.evaluation.report import PREDICTION_COLUMNS, write_predictions

RUNS = tuple(BASELINE_RUNS)


def make_predictions(n: int = 240, seed: int = 0, first_id: int = 1) -> pd.DataFrame:
    """A scored frame in the shape ``write_predictions`` emits."""
    rng = np.random.default_rng(seed)
    is_fraud = np.zeros(n, dtype=int)
    is_fraud[: n // 10] = 1
    rng.shuffle(is_fraud)
    return pd.DataFrame(
        {
            "TransactionID": range(first_id, first_id + n),
            "split": np.where(np.arange(n) < n // 2, FIGURE_SPLIT, "val_cal"),
            "day": np.repeat(np.arange(1, 5), n // 4),
            "isFraud": is_fraud,
            "score": rng.random(n),
        }
    )[list(PREDICTION_COLUMNS)]


@pytest.fixture
def predictions_dir(tmp_path):
    """Both baseline runs on disk, scoring the same rows."""
    for offset, name in enumerate(RUNS):
        write_predictions(name, make_predictions(seed=offset), tmp_path, splits=(FIGURE_SPLIT,))
    return tmp_path


# ---- what the figure is about ------------------------------------------------


def test_the_e2_variant_is_not_a_series():
    """The comparison is incumbent vs reference. `logistic_balanced` is the E2
    loser and belongs in a table, not in a third colour slot here."""
    assert "logistic_balanced" not in BASELINE_RUNS


def test_the_figure_names_match_what_the_readme_promises():
    assert PR_CURVE_FIGURE == "pr_curve_baselines.png"
    assert RECALL_FIGURE == "recall_at_capacity_baselines.png"


# ---- load_predictions --------------------------------------------------------


def test_predictions_load_indexed_by_the_join_key(predictions_dir):
    frames = load_predictions(RUNS, predictions_dir)
    assert set(frames) == set(RUNS)
    for frame in frames.values():
        assert frame.index.name == "TransactionID"
        assert frame.index.is_monotonic_increasing


def test_a_missing_run_names_the_stage_that_writes_it(tmp_path):
    with pytest.raises(FileNotFoundError, match="make baselines"):
        load_predictions(RUNS, tmp_path)


def test_asking_for_a_split_the_run_never_scored_raises(predictions_dir):
    with pytest.raises(ValueError, match="holds no 'test' rows"):
        load_predictions(RUNS, predictions_dir, split="test")


# ---- align -------------------------------------------------------------------


def test_align_returns_the_shared_labels_and_one_series_per_run(predictions_dir):
    y_true, days, scores = align(load_predictions(RUNS, predictions_dir))

    assert list(scores) == [BASELINE_RUNS[name] for name in RUNS]
    for series in scores.values():
        assert series.index.equals(y_true.index)
    assert days.index.equals(y_true.index)


def test_runs_that_scored_different_rows_cannot_be_overlaid(tmp_path):
    """Two plausible curves on one pair of axes, drawn on different
    populations, is the failure that does not announce itself."""
    first, second = RUNS
    write_predictions(first, make_predictions(), tmp_path, splits=(FIGURE_SPLIT,))
    write_predictions(second, make_predictions(first_id=9001), tmp_path, splits=(FIGURE_SPLIT,))

    with pytest.raises(ValueError, match="scored different rows"):
        align(load_predictions(RUNS, tmp_path))


def test_align_rejects_an_empty_set_of_runs():
    with pytest.raises(ValueError, match="nothing to draw"):
        align({})
