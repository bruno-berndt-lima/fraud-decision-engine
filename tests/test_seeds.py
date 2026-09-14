"""Tests for the seed spread — the bar E6 judges a tuned candidate against.

The property that carries the phase is the contrast between the two kinds of
point: a configuration that samples nothing returns one number repeated, and one
that subsamples does not. Everything E6 concludes rests on that being measured
rather than assumed, so it is measured here too, on small data.

`summarize` is pure and gets its statistic pinned exactly. A population standard
deviation would report a narrower bar than the seeds justify, and every verdict
downstream would be slightly too generous.
"""

from __future__ import annotations

import lightgbm as lgb
import numpy as np
import pandas as pd
import pytest

from conftest import children
from fraud_engine.models.seeds import measure, summarize
from fraud_engine.models.train import (
    CONTRACT_PARAMS,
    apply_categories,
    fit_categories,
    to_dataset,
)

CAPACITIES = [0.1]
FEATURES = ["signal", "noise", "extra", "brand"]

# One point that samples nothing and one that samples both rows and columns.
# The names mirror config's; what matters is that only the second can move.
POINTS = {
    "untuned": {},
    "subsampled": {"feature_fraction": 0.5, "bagging_fraction": 0.5, "bagging_freq": 1},
}

MODEL_CFG = {
    "tuned": {},
    "seed": 0,
    "early_stopping_rounds": 5,
    "num_boost_round": 80,
    "spread": {"points": POINTS, "seeds": 3},
}

SEEDS = range(3)


def make_matrix(rows: int = 600, seed: int = 0) -> pd.DataFrame:
    """A matrix with a weak planted signal, so subsampling has room to matter."""
    rng = np.random.default_rng(seed)
    signal = rng.random(rows)

    return pd.DataFrame(
        {
            "TransactionID": range(rows),
            "isFraud": (signal + rng.normal(0, 0.25, rows) > 0.8).astype(int),
            "day": 1,
            "signal": signal,
            "noise": rng.random(rows),
            "extra": rng.random(rows),
            "brand": pd.Categorical(rng.choice(["visa", "amex", "elo"], rows)),
        }
    )


@pytest.fixture
def matrices() -> dict[str, pd.DataFrame]:
    raw = {name: make_matrix(seed=i) for i, name in enumerate(("train", "val_fit"))}
    vocabulary = fit_categories(raw["train"], min_rows=1)
    return {name: apply_categories(frame, vocabulary) for name, frame in raw.items()}


@pytest.fixture
def datasets(matrices) -> tuple[lgb.Dataset, lgb.Dataset]:
    train = to_dataset(matrices["train"], FEATURES)
    return train, to_dataset(matrices["val_fit"], FEATURES, reference=train)


@pytest.fixture
def spread(datasets, matrices, experiment_run) -> pd.DataFrame:
    train, val_fit = datasets
    return measure(train, val_fit, matrices, FEATURES, MODEL_CFG, CAPACITIES, SEEDS)


def frame(values: dict[str, list[float]], iterations: dict[str, list[int]] | None = None):
    rows = []
    for point, scores in values.items():
        for seed, score in enumerate(scores):
            iteration = 100 if iterations is None else iterations[point][seed]
            rows.append(
                {"point": point, "seed": seed, "pr_auc": score, "best_iteration": iteration}
            )
    return pd.DataFrame(rows)


# ------------------------------------------------------------------------------
# measure
# ------------------------------------------------------------------------------


def test_one_row_per_point_and_seed(spread):
    assert len(spread) == len(POINTS) * len(SEEDS)
    assert list(spread["point"].unique()) == list(POINTS)


def test_the_points_come_from_config(datasets, matrices, experiment_run):
    """Nothing about which configurations are measured is fixed in this module."""
    train, val_fit = datasets
    renamed = {**MODEL_CFG, "spread": {"points": {"only_one": {}}, "seeds": 2}}

    result = measure(train, val_fit, matrices, FEATURES, renamed, CAPACITIES, range(2))

    assert list(result["point"].unique()) == ["only_one"]


def test_a_configuration_that_samples_nothing_returns_one_number(spread):
    """The finding the whole phase rests on, measured rather than argued.

    LightGBM's defaults draw neither rows nor columns, so the seed has nothing to
    act on. If this ever stopped holding, the untuned reference would need a
    spread behind every comparison it appears in.
    """
    untuned = spread.loc[spread["point"] == "untuned"]

    assert len(set(untuned["pr_auc"])) == 1
    assert len(set(untuned["best_iteration"])) == 1


def test_a_subsampling_configuration_moves_with_the_seed(spread):
    """And the contrast is the point: a bar of zero would judge nothing."""
    subsampled = spread.loc[spread["point"] == "subsampled"]

    assert len(set(subsampled["pr_auc"])) > 1


def test_the_stopping_round_is_recorded_beside_the_metric(spread):
    """Where the mechanism is visible.

    The seed moves the subsample, which moves the validation curve, which moves
    where early stopping lands — so a point whose iteration count swings is not
    producing the same model twice.
    """
    assert spread["best_iteration"].gt(0).all()
    assert set(spread.columns) == {"point", "seed", "pr_auc", "best_iteration"}


def test_scores_are_a_metric_and_not_a_count(spread):
    assert spread["pr_auc"].between(0, 1).all()


def test_only_the_seed_varies_within_a_point(datasets, matrices, experiment_run):
    """Re-measuring a point returns the same rows, so the bar is reproducible."""
    train, val_fit = datasets

    first = measure(train, val_fit, matrices, FEATURES, MODEL_CFG, CAPACITIES, SEEDS)
    second = measure(train, val_fit, matrices, FEATURES, MODEL_CFG, CAPACITIES, SEEDS)

    pd.testing.assert_frame_equal(first, second)


def test_no_point_may_override_the_contract(datasets, matrices, experiment_run):
    """A point that moved `metric` would change what every other point measured."""
    train, val_fit = datasets
    tampered = {
        **MODEL_CFG,
        "spread": {"points": {"bad": {"metric": "auc"}}, "seeds": 1},
    }

    with pytest.raises(ValueError, match="contract parameters cannot be set"):
        measure(train, val_fit, matrices, FEATURES, tampered, CAPACITIES, range(1))


def test_the_contract_metric_is_what_the_bar_is_measured_on():
    """PR-AUC, not log loss. Early stopping on the wrong metric stops elsewhere."""
    assert CONTRACT_PARAMS["metric"] == "average_precision"


# ------------------------------------------------------------------------------
# summarize
# ------------------------------------------------------------------------------


def test_the_spread_is_a_sample_standard_deviation():
    """`ddof=1`, because these seeds are a sample of the draws, not all of them.

    A population deviation reports a narrower bar than the seeds justify, and
    every verdict downstream comes out slightly too generous.
    """
    values = [0.50, 0.54, 0.58, 0.61]
    result = summarize(frame({"point": values}))

    assert result.loc["point", "pr_auc_std"] == pytest.approx(np.std(values, ddof=1))
    assert result.loc["point", "pr_auc_std"] != pytest.approx(np.std(values))


def test_a_point_that_never_moved_has_zero_spread():
    result = summarize(frame({"flat": [0.5, 0.5, 0.5]}))

    assert result.loc["flat", "pr_auc_std"] == 0.0
    assert result.loc["flat", "pr_auc_min"] == result.loc["flat", "pr_auc_max"]


def test_points_keep_the_order_they_were_measured_in():
    """Alphabetical would reorder the table against the config that produced it."""
    result = summarize(frame({"zulu": [0.5, 0.6], "alpha": [0.4, 0.7]}))

    assert list(result.index) == ["zulu", "alpha"]


def test_the_mean_and_the_extremes_are_reported():
    values = [0.40, 0.50, 0.90]
    result = summarize(frame({"point": values})).loc["point"]

    assert result["pr_auc_mean"] == pytest.approx(np.mean(values))
    assert (result["pr_auc_min"], result["pr_auc_max"]) == (min(values), max(values))


def test_the_iteration_range_is_reported_per_point():
    result = summarize(frame({"point": [0.5, 0.6, 0.7]}, {"point": [140, 900, 300]}))

    assert (result.loc["point", "iteration_min"], result.loc["point", "iteration_max"]) == (
        140,
        900,
    )


def test_a_single_seed_reports_no_spread_rather_than_zero():
    """One draw is not a spread of zero, and E6's bar must not read it as one."""
    result = summarize(frame({"lonely": [0.5]}))

    assert pd.isna(result.loc["lonely", "pr_auc_std"])


# ------------------------------------------------------------------------------
# what reaches MLflow
# ------------------------------------------------------------------------------


def test_every_point_and_seed_is_a_child_run(spread, experiment_run):
    names = [run.info.run_name for run in children(experiment_run)]

    assert names == [f"spread_{point}_seed{seed}" for point in POINTS for seed in SEEDS]


def test_each_child_records_the_seed_it_trained_with(spread, experiment_run):
    """The seed is the only thing that varies within a point, so it must be a column."""
    for run in children(experiment_run):
        seed = int(run.info.run_name.rsplit("seed", 1)[1])
        assert run.data.params["seed"] == str(seed)


def test_child_metrics_match_the_returned_spread(spread, experiment_run):
    logged = [run.data.metrics["val_fit.pr_auc"] for run in children(experiment_run)]

    assert logged == pytest.approx(list(spread["pr_auc"]))
