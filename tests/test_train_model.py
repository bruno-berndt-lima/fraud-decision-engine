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
    TRAINING_SPLITS,
    apply_categories,
    apply_medians,
    feature_columns,
    fit,
    fit_categories,
    fit_medians,
    prepare_matrices,
    resolve_params,
    run_name,
    score,
    to_dataset,
    write_categories,
    write_medians,
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


# ------------------------------------------------------------------------------
# imputation, and the artifact it has to ship with
# ------------------------------------------------------------------------------


def with_gaps() -> pd.DataFrame:
    """A matrix whose numeric columns carry nulls, including one all-null column."""
    frame = make_matrix(rows=6)
    frame.loc[[0, 1], "signal"] = np.nan
    frame["dead"] = np.nan
    return frame


def test_medians_come_from_the_training_window_only():
    train = with_gaps()
    medians = fit_medians(train, ["signal", "noise", "dead"])

    assert medians["signal"] == train["signal"].median()


def test_only_numeric_columns_get_a_median():
    medians = fit_medians(with_gaps(), ["signal", "brand", "device"])

    assert set(medians.index) == {"signal"}


def test_applying_medians_leaves_no_numeric_nulls():
    train = with_gaps()
    medians = fit_medians(train, ["signal", "noise"])

    assert not apply_medians(train, medians)["signal"].isna().any()


def test_a_column_with_no_training_values_keeps_its_nulls():
    train = with_gaps()
    filled = apply_medians(train, fit_medians(train, ["signal", "dead"]))

    assert filled["dead"].isna().all()


def test_applying_medians_does_not_modify_the_input():
    train = with_gaps()
    apply_medians(train, fit_medians(train, ["signal"]))

    assert train["signal"].isna().any()


def test_the_fill_values_round_trip(tmp_path: Path):
    medians = fit_medians(with_gaps(), ["signal", "noise"])
    write_medians(medians, tmp_path / "medians.parquet")

    back = pd.read_parquet(tmp_path / "medians.parquet").set_index("column")["median"]
    assert back.to_dict() == medians.to_dict()


# ------------------------------------------------------------------------------
# run_name
# ------------------------------------------------------------------------------


def test_an_empty_tuned_block_names_the_untuned_reference():
    assert run_name({"tuned": {}}) == "lightgbm_untuned"


def test_tuned_parameters_name_a_tuned_run():
    assert run_name({"tuned": {"num_leaves": 251}}) == "lightgbm_tuned"


# ------------------------------------------------------------------------------
# prepare_matrices
# ------------------------------------------------------------------------------


def gappy_matrix(seed: int) -> pd.DataFrame:
    """A matrix carrying nulls, so imputation has something to do.

    `make_matrix` has none, and against a frame with no gaps "fill every split"
    and "fill train alone" produce the same answer — which is a test that passes
    on a function that leaks.
    """
    rng = np.random.default_rng(seed)
    frame = make_matrix(seed=seed)
    frame["gappy"] = np.where(rng.random(len(frame)) < 0.3, np.nan, rng.random(len(frame)))
    return frame


@pytest.fixture
def features_dir(tmp_path: Path) -> Path:
    for index, name in enumerate(SPLIT_NAMES[:3]):
        gappy_matrix(seed=index).to_parquet(tmp_path / f"{name}.parquet")
    return tmp_path


PREPARE_CFG = {"min_category_rows": 1, "impute": True}


def test_preparation_returns_the_tables_that_shaped_it(features_dir: Path):
    """A caller writing them as artifacts needs the fits, not just the frames."""
    matrices, vocabulary, medians = prepare_matrices(features_dir, PREPARE_CFG)

    assert set(matrices) == set(TRAINING_SPLITS)
    assert set(vocabulary) == {"brand", "device"}
    assert medians is not None


def test_every_split_is_re_levelled_against_the_training_vocabulary(features_dir: Path):
    """A code must stand for the same level either side of the boundary."""
    matrices, vocabulary, _ = prepare_matrices(features_dir, PREPARE_CFG)

    for frame in matrices.values():
        for column, levels in vocabulary.items():
            assert list(frame[column].cat.categories) == list(levels)


def test_validation_is_filled_too(features_dir: Path):
    """A model fitted on filled data and scored on nulls meets a distribution it never saw."""
    matrices, _, medians = prepare_matrices(features_dir, PREPARE_CFG)

    assert pd.read_parquet(features_dir / "val_fit.parquet")["gappy"].isna().any(), (
        "the fixture carries no nulls; this would pass on a function that fills train alone"
    )

    for frame in matrices.values():
        assert frame[medians.index].isna().to_numpy().sum() == 0


def test_the_fills_come_from_train_alone(features_dir: Path):
    _, _, medians = prepare_matrices(features_dir, PREPARE_CFG)
    raw = pd.read_parquet(features_dir / "train.parquet")

    pd.testing.assert_series_equal(medians, raw[medians.index].median(), check_names=False)


def test_imputation_off_returns_no_medians(features_dir: Path):
    """`None` rather than an empty table: a caller has to tell the two apart."""
    _, _, medians = prepare_matrices(features_dir, {**PREPARE_CFG, "impute": False})

    assert medians is None


def test_only_the_named_splits_are_read(features_dir: Path):
    """Naming test has to be deliberate, and no experiment names it."""
    matrices, _, _ = prepare_matrices(features_dir, PREPARE_CFG, ("train", "val_fit"))

    assert set(matrices) == {"train", "val_fit"}


def test_splits_without_train_raise(features_dir: Path):
    """The fits come from train; without it they would come from validation."""
    with pytest.raises(ValueError, match="there is nothing to fit on"):
        prepare_matrices(features_dir, PREPARE_CFG, ("val_fit", "val_cal"))
