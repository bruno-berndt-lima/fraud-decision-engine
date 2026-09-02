"""Tests for what reaches LightGBM and what comes back.

Boosters here are trained on a few hundred synthetic rows with a planted signal.
The point is never the score — it is that the parameters, the datasets and the
scored frame are assembled the way the pipeline claims.
"""

from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
import pytest

from fraud_engine.data.splits import SPLIT_NAMES
from fraud_engine.evaluation.report import PREDICTION_COLUMNS, REQUIRED_COLUMNS
from fraud_engine.models.train import (
    CONTRACT_PARAMS,
    apply_categories,
    feature_columns,
    fit,
    fit_categories,
    resolve_params,
    score,
    to_dataset,
    write_categories,
)

MODEL_CFG = {"tuned": {}, "seed": 7}

FEATURES = ["signal", "noise", "brand", "device"]


def make_matrix(rows: int = 400, seed: int = 0) -> pd.DataFrame:
    """A matrix with a planted signal, so a booster has something to find."""
    rng = np.random.default_rng(seed)
    signal = rng.random(rows)
    return pd.DataFrame(
        {
            "TransactionID": range(rows),
            "TransactionDT": np.arange(rows) * 60,
            "isFraud": (signal > 0.8).astype(int),
            "day": 1,
            "signal": signal,
            "noise": rng.random(rows),
            "brand": pd.Categorical(rng.choice(["visa", "amex"], rows)),
            "device": pd.Categorical(rng.choice(["ios", "android", "web"], rows)),
        }
    )


@pytest.fixture
def matrices() -> dict[str, pd.DataFrame]:
    """Prepared matrices sharing one fitted vocabulary."""
    raw = {name: make_matrix(seed=index) for index, name in enumerate(SPLIT_NAMES[:3])}
    vocabulary = fit_categories(raw["train"], min_rows=1)
    return {name: apply_categories(frame, vocabulary) for name, frame in raw.items()}


@pytest.fixture
def booster(matrices: dict[str, pd.DataFrame]) -> lgb.Booster:
    train = to_dataset(matrices["train"], FEATURES)
    return lgb.train({**CONTRACT_PARAMS, "seed": 0, "num_leaves": 4}, train, num_boost_round=20)


# ------------------------------------------------------------------------------
# resolve_params
# ------------------------------------------------------------------------------


def test_the_contract_reaches_lightgbm():
    params = resolve_params(MODEL_CFG)

    assert all(params[key] == value for key, value in CONTRACT_PARAMS.items())


def test_the_seed_is_passed_through():
    assert resolve_params(MODEL_CFG)["seed"] == 7


def test_tuned_knobs_are_merged_over_the_contract():
    params = resolve_params({**MODEL_CFG, "tuned": {"num_leaves": 63}})

    assert params["num_leaves"] == 63
    assert params["objective"] == CONTRACT_PARAMS["objective"]


@pytest.mark.parametrize("key", sorted(CONTRACT_PARAMS))
def test_config_cannot_set_a_contract_parameter(key: str):
    with pytest.raises(ValueError, match="contract parameters"):
        resolve_params({**MODEL_CFG, "tuned": {key: "anything"}})


def test_nothing_is_set_that_was_not_asked_for():
    assert set(resolve_params(MODEL_CFG)) == {*CONTRACT_PARAMS, "seed"}


# ------------------------------------------------------------------------------
# to_dataset
# ------------------------------------------------------------------------------


def test_the_dataset_carries_the_named_features(matrices: dict[str, pd.DataFrame]):
    dataset = to_dataset(matrices["train"], FEATURES)
    dataset.construct()

    assert dataset.num_feature() == len(FEATURES)
    assert dataset.num_data() == len(matrices["train"])


def test_the_label_is_read_from_the_frame(matrices: dict[str, pd.DataFrame]):
    dataset = to_dataset(matrices["train"], FEATURES)
    dataset.construct()

    assert np.array_equal(dataset.get_label(), matrices["train"]["isFraud"].to_numpy())


def test_categoricals_are_named_rather_than_inferred(matrices: dict[str, pd.DataFrame]):
    assert to_dataset(matrices["train"], FEATURES).categorical_feature == ["brand", "device"]


def test_a_validation_set_reuses_the_training_bins(matrices: dict[str, pd.DataFrame]):
    train = to_dataset(matrices["train"], FEATURES)
    val = to_dataset(matrices["val_fit"], FEATURES, reference=train)
    val.construct()

    assert train in val.get_ref_chain()


# ------------------------------------------------------------------------------
# score
# ------------------------------------------------------------------------------


def test_every_row_of_every_split_is_scored(
    booster: lgb.Booster, matrices: dict[str, pd.DataFrame]
):
    scored = score(booster, matrices, FEATURES)

    assert len(scored) == sum(len(frame) for frame in matrices.values())


def test_the_frame_carries_what_the_harness_requires(
    booster: lgb.Booster, matrices: dict[str, pd.DataFrame]
):
    scored = score(booster, matrices, FEATURES)

    assert set(REQUIRED_COLUMNS) <= set(scored.columns)
    assert set(PREDICTION_COLUMNS) <= set(scored.columns)


def test_the_split_comes_from_the_key_not_from_a_column(
    booster: lgb.Booster, matrices: dict[str, pd.DataFrame]
):
    scored = score(booster, matrices, FEATURES)

    assert list(scored["split"].cat.categories) == list(SPLIT_NAMES)
    assert set(scored["split"].unique()) == set(matrices)


def test_scores_are_probabilities(booster: lgb.Booster, matrices: dict[str, pd.DataFrame]):
    scored = score(booster, matrices, FEATURES)

    assert scored["score"].between(0.0, 1.0).all()


def test_the_best_iteration_is_used_rather_than_every_tree(
    booster: lgb.Booster, matrices: dict[str, pd.DataFrame]
):
    booster.best_iteration = 5
    one = matrices["train"]

    scored = score(booster, {"train": one}, FEATURES)
    assert np.allclose(scored["score"], booster.predict(one[FEATURES], num_iteration=5))
    assert not np.allclose(scored["score"], booster.predict(one[FEATURES], num_iteration=20))


# ------------------------------------------------------------------------------
# write_categories
# ------------------------------------------------------------------------------


def test_the_vocabulary_round_trips(tmp_path: Path, matrices: dict[str, pd.DataFrame]):
    vocabulary = fit_categories(matrices["train"], min_rows=1)
    write_categories(vocabulary, tmp_path / "categories.parquet")

    back = pd.read_parquet(tmp_path / "categories.parquet")
    rebuilt = {
        column: list(group.sort_values("code")["level"]) for column, group in back.groupby("column")
    }
    assert rebuilt == {column: list(levels) for column, levels in vocabulary.items()}


def test_the_code_is_written_rather_than_left_to_row_order(
    tmp_path: Path, matrices: dict[str, pd.DataFrame]
):
    vocabulary = fit_categories(matrices["train"], min_rows=1)
    write_categories(vocabulary, tmp_path / "categories.parquet")

    back = pd.read_parquet(tmp_path / "categories.parquet")
    assert "code" in back.columns
    assert all(
        list(group["code"]) == list(range(len(group))) for _, group in back.groupby("column")
    )


def test_the_parent_directory_is_created(tmp_path: Path, matrices: dict[str, pd.DataFrame]):
    path = tmp_path / "nested" / "categories.parquet"
    write_categories(fit_categories(matrices["train"], min_rows=1), path)

    assert path.exists()


# ------------------------------------------------------------------------------
# feature_columns, against a matrix a booster actually trains on
# ------------------------------------------------------------------------------


def test_the_features_are_what_the_booster_was_given(matrices: dict[str, pd.DataFrame]):
    assert feature_columns(matrices["train"]) == FEATURES


# ------------------------------------------------------------------------------
# fit — the ceiling
# ------------------------------------------------------------------------------


def datasets(matrices: dict[str, pd.DataFrame]) -> tuple[lgb.Dataset, lgb.Dataset]:
    train = to_dataset(matrices["train"], FEATURES)
    return train, to_dataset(matrices["val_fit"], FEATURES, reference=train)


def test_a_run_that_exhausts_its_round_budget_is_refused(matrices: dict[str, pd.DataFrame]):
    train, val_fit = datasets(matrices)
    cfg = {**MODEL_CFG, "num_boost_round": 5, "early_stopping_rounds": 50}

    with pytest.raises(ValueError, match="bound before early stopping"):
        fit(train, val_fit, cfg)


def test_a_run_with_room_to_stop_is_returned(matrices: dict[str, pd.DataFrame]):
    train, val_fit = datasets(matrices)
    cfg = {**MODEL_CFG, "num_boost_round": 500, "early_stopping_rounds": 10}

    booster = fit(train, val_fit, cfg)

    assert booster.best_iteration + cfg["early_stopping_rounds"] <= cfg["num_boost_round"]
