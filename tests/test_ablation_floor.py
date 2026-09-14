"""Tests for the tree ablation's bar: what a draw removes, and how the bar is read.

`draw` and `summarize` are pure. `measure` runs for real on a few hundred rows,
because the property that makes this bar different from the probe's — each draw
early-stops at its own round — only exists when boosters actually train.
"""

from __future__ import annotations

import mlflow
import numpy as np
import pandas as pd
import pytest

from conftest import children
from fraud_engine.models.ablation import REFERENCE
from fraud_engine.models.floor import draw, measure, summarize
from fraud_engine.models.train import apply_categories, feature_columns, fit_categories

CAPACITIES = [0.1]
MODEL_CFG = {"tuned": {}, "seed": 0, "early_stopping_rounds": 5, "num_boost_round": 60}
COLUMNS = [f"c{i}" for i in range(20)]


def make_matrix(rows: int = 400, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    signal = rng.random(rows)
    frame = pd.DataFrame(
        {
            "TransactionID": range(rows),
            "TransactionDT": np.arange(rows) * 60,
            "isFraud": (signal > 0.8).astype(int),
            "day": 1,
            "signal": signal,
            "brand": pd.Categorical(rng.choice(["visa", "amex"], rows)),
        }
    )
    for index in range(6):
        frame[f"x{index}"] = rng.random(rows)
    return frame


@pytest.fixture
def matrices() -> dict[str, pd.DataFrame]:
    raw = {name: make_matrix(seed=i) for i, name in enumerate(("train", "val_fit"))}
    vocabulary = fit_categories(raw["train"], min_rows=1)
    return {name: apply_categories(frame, vocabulary) for name, frame in raw.items()}


@pytest.fixture
def floor(matrices, experiment_run) -> pd.DataFrame:
    return measure(matrices, [1, 3], 2, MODEL_CFG, CAPACITIES)


# ------------------------------------------------------------------------------
# draw
# ------------------------------------------------------------------------------


def test_a_draw_has_the_requested_width_and_no_repeats():
    drawn = draw(COLUMNS, 7, np.random.default_rng(0))

    assert len(drawn) == 7
    assert len(set(drawn)) == 7
    assert set(drawn) <= set(COLUMNS)


def test_a_draw_is_reproducible_from_its_seed():
    """Draw 7 at width 6 is the same set whether or not the draws before it ran."""
    assert draw(COLUMNS, 6, np.random.default_rng([0, 6, 7])) == draw(
        COLUMNS, 6, np.random.default_rng([0, 6, 7])
    )


def test_different_draws_remove_different_columns():
    assert draw(COLUMNS, 6, np.random.default_rng([0, 6, 1])) != draw(
        COLUMNS, 6, np.random.default_rng([0, 6, 2])
    )


def test_a_draw_wider_than_the_matrix_raises():
    with pytest.raises(ValueError, match="nothing would be left"):
        draw(COLUMNS, len(COLUMNS) + 1, np.random.default_rng(0))


def test_names_come_back_as_plain_strings():
    """numpy strings would not serialise into the run's artifact."""
    assert all(type(name) is str for name in draw(COLUMNS, 3, np.random.default_rng(0)))


# ------------------------------------------------------------------------------
# summarize
# ------------------------------------------------------------------------------


def test_the_bar_is_the_largest_absolute_delta():
    """Two-sided: removing columns moves the metric either way."""
    frame = pd.DataFrame({"width": [3, 3, 3], "delta": [0.01, -0.02, 0.005]})

    assert summarize(frame).set_index("width").loc[3, "bar"] == pytest.approx(0.02)


def test_one_row_per_width_with_its_draw_count_and_spread():
    frame = pd.DataFrame({"width": [3, 3, 7, 7, 7], "delta": [0.1, 0.3, 0.0, 0.1, 0.2]})
    result = summarize(frame).set_index("width")

    assert list(result.index) == [3, 7]
    assert list(result["draws"]) == [2, 3]
    assert result.loc[7, "sd"] == pytest.approx(np.std([0.0, 0.1, 0.2], ddof=1))


def test_widths_never_share_a_bar():
    frame = pd.DataFrame({"width": [3, 7], "delta": [0.5, 0.01]})
    result = summarize(frame).set_index("width")

    assert result.loc[7, "bar"] == pytest.approx(0.01)


# ------------------------------------------------------------------------------
# measure
# ------------------------------------------------------------------------------


def test_one_row_per_width_and_draw(floor):
    assert list(zip(floor["width"], floor["draw"], strict=True)) == [(1, 0), (1, 1), (3, 0), (3, 1)]


def test_features_shrink_by_the_width(floor, matrices):
    full = len(feature_columns(matrices["train"]))

    assert list(floor["n_features"]) == [full - width for width in floor["width"]]


def test_deltas_are_measured_from_one_reference_fit(floor, experiment_run):
    runs = {run.info.run_name: run for run in children(experiment_run)}
    reference = runs[f"floor_{REFERENCE}"].data.metrics["val_fit.pr_auc"]

    assert list(floor["delta"]) == pytest.approx(list(floor["pr_auc"] - reference))


def test_the_reference_and_every_draw_are_child_runs(floor, experiment_run):
    names = [run.info.run_name for run in children(experiment_run)]

    assert names == [
        f"floor_{REFERENCE}",
        "floor_w1_d0",
        "floor_w1_d1",
        "floor_w3_d0",
        "floor_w3_d1",
    ]


def test_each_draw_records_the_columns_it_removed(floor, experiment_run):
    for run in children(experiment_run):
        if run.info.run_name == f"floor_{REFERENCE}":
            continue
        removed = mlflow.artifacts.load_dict(f"{run.info.artifact_uri}/removed_columns.json")
        assert len(removed) == int(run.data.params["width"])


def test_measuring_twice_draws_the_same_sets(matrices, experiment_run):
    first = measure(matrices, [3], 2, MODEL_CFG, CAPACITIES)
    second = measure(matrices, [3], 2, MODEL_CFG, CAPACITIES)

    pd.testing.assert_frame_equal(first, second)
