"""Tests for the search space, the acceptance rule, and what a trial trains on.

`verdict` is pure and gets the arithmetic pinned exactly — it is the rule that
decides whether tuning shipped, and the one place in this phase where a wrong
inequality would be invisible in the output.

`search_space` is driven by a stub trial rather than an Optuna study. What
matters is which knobs are asked for, on which scale, and against which bounds;
running a real sampler would test Optuna.
"""

from __future__ import annotations

import lightgbm as lgb
import numpy as np
import pandas as pd
import pytest

from conftest import children
from fraud_engine.models.tune import Variant, build_variants, confirm, search_space, verdict

SPACE = {
    "num_leaves": [15, 255],
    "min_child_samples": [20, 500],
    "learning_rate": [0.03, 0.15],
    "feature_fraction": [0.4, 1.0],
    "bagging_fraction": [0.4, 1.0],
    "min_data_per_group": [50, 500],
    "cat_smooth": [1.0, 100.0],
}

FEATURES = ["signal", "gappy", "brand"]


class StubTrial:
    """Records every suggestion instead of sampling one.

    Returns the low bound so the values are predictable; the tests here are
    about what was asked for, not what came back.
    """

    def __init__(self, categorical=None):
        self.asked: dict[str, dict] = {}
        self._categorical = {} if categorical is None else categorical

    def suggest_int(self, name, low, high, log=False):
        self.asked[name] = {"kind": "int", "low": low, "high": high, "log": log}
        return low

    def suggest_float(self, name, low, high, log=False):
        self.asked[name] = {"kind": "float", "low": low, "high": high, "log": log}
        return low

    def suggest_categorical(self, name, choices):
        self.asked[name] = {"kind": "categorical", "choices": choices}
        return self._categorical.get(name, choices[0])


def runs(candidate: list[float], reference: list[float]) -> pd.DataFrame:
    return pd.DataFrame(
        [
            *({"config": "candidate", "seed": i, "pr_auc": v} for i, v in enumerate(candidate)),
            *({"config": "reference", "seed": i, "pr_auc": v} for i, v in enumerate(reference)),
        ]
    )


def make_matrix(rows: int = 300, seed: int = 0) -> pd.DataFrame:
    """A matrix with a column that carries nulls, so imputation has work to do."""
    rng = np.random.default_rng(seed)
    signal = rng.random(rows)
    gappy = rng.random(rows)
    gappy[rng.random(rows) < 0.3] = np.nan

    return pd.DataFrame(
        {
            "TransactionID": range(rows),
            "isFraud": (signal > 0.8).astype(int),
            "day": 1,
            "signal": signal,
            "gappy": gappy,
            "brand": pd.Categorical(rng.choice(["visa", "amex"], rows)),
        }
    )


# ------------------------------------------------------------------------------
# search_space
# ------------------------------------------------------------------------------


def test_every_configured_knob_is_asked_for():
    trial = StubTrial()

    search_space(trial, SPACE)

    assert set(trial.asked) == {*SPACE, "impute"}


def test_bounds_come_from_config_and_are_not_hardcoded():
    trial = StubTrial()
    narrowed = {**SPACE, "num_leaves": [7, 31]}

    search_space(trial, narrowed)

    assert (trial.asked["num_leaves"]["low"], trial.asked["num_leaves"]["high"]) == (7, 31)


def test_multiplicative_knobs_are_log_scaled():
    """Uniform sampling would spend most trials among large values barely apart."""
    trial = StubTrial()

    search_space(trial, SPACE)

    for knob in ("num_leaves", "min_child_samples", "learning_rate", "min_data_per_group"):
        assert trial.asked[knob]["log"], knob


def test_fractions_are_sampled_uniformly():
    trial = StubTrial()

    search_space(trial, SPACE)

    for knob in ("feature_fraction", "bagging_fraction"):
        assert not trial.asked[knob]["log"], knob


def test_bagging_frequency_is_fixed_rather_than_searched():
    """A second way to express what bagging_fraction already controls."""
    trial = StubTrial()

    params, _ = search_space(trial, SPACE)

    assert params["bagging_freq"] == 1
    assert "bagging_freq" not in trial.asked


def test_imputation_is_returned_separately_and_never_as_a_parameter():
    """LightGBM warns about unknown keys rather than refusing them.

    Left in the mapping, `impute` would be silently ignored, every trial would
    train on the same data, and the study would record that they had not.
    """
    trial = StubTrial(categorical={"impute": True})

    params, impute = search_space(trial, SPACE)

    assert impute is True
    assert "impute" not in params


def test_both_imputation_choices_are_offered():
    trial = StubTrial()

    search_space(trial, SPACE)

    assert sorted(trial.asked["impute"]["choices"]) == [False, True]


def test_no_contract_parameter_is_reachable_from_the_space():
    """The space must not touch what defines the comparison.

    `objective` or `metric` moving between trials would silently change what
    "better" means, and every earlier number with it.
    """
    from fraud_engine.models.train import CONTRACT_PARAMS

    params, _ = search_space(StubTrial(), SPACE)

    assert not set(params) & set(CONTRACT_PARAMS)


# ------------------------------------------------------------------------------
# verdict
# ------------------------------------------------------------------------------


def test_a_gap_wider_than_both_spreads_is_accepted():
    result = verdict(runs(candidate=[0.60, 0.61, 0.62], reference=[0.50, 0.51, 0.52]))

    assert result["accepted"] is True


def test_a_gap_inside_the_spreads_is_refused():
    result = verdict(runs(candidate=[0.50, 0.60, 0.70], reference=[0.49, 0.59, 0.69]))

    assert result["accepted"] is False


def test_the_bar_is_the_sum_of_spreads_not_the_standard_error():
    """The strictness is the price of the candidate being a maximum over trials.

    A standard error would shrink with the seed count and let a bigger budget
    buy significance. The sum does not move, so this pins the arithmetic rather
    than the outcome.
    """
    candidate, reference = [0.60, 0.62, 0.64], [0.50, 0.54, 0.58]
    result = verdict(runs(candidate, reference))

    expected = pd.Series(candidate).std() + pd.Series(reference).std()

    assert result["bar"] == pytest.approx(expected)


def test_the_gap_is_the_difference_of_means():
    candidate, reference = [0.60, 0.62, 0.64], [0.50, 0.54, 0.58]
    result = verdict(runs(candidate, reference))

    assert result["gap"] == pytest.approx(np.mean(candidate) - np.mean(reference))


def test_a_candidate_that_merely_ties_is_refused():
    """`>` and not `>=`: a gap of exactly the bar has not cleared it."""
    result = verdict(runs(candidate=[0.5, 0.5], reference=[0.5, 0.5]))

    assert result["gap"] == 0.0
    assert result["bar"] == 0.0
    assert result["accepted"] is False


def test_a_worse_candidate_is_refused():
    assert verdict(runs(candidate=[0.40, 0.41], reference=[0.60, 0.61]))["accepted"] is False


def test_the_verdict_is_json_serialisable():
    """It is written to a tracked record; a numpy bool would not survive."""
    import json

    result = verdict(runs(candidate=[0.60, 0.61], reference=[0.50, 0.51]))

    assert json.loads(json.dumps(result))["accepted"] is True
    assert isinstance(result["accepted"], bool)


# ------------------------------------------------------------------------------
# build_variants
# ------------------------------------------------------------------------------


@pytest.fixture
def variants() -> dict[bool, Variant]:
    matrices = {name: make_matrix(seed=i) for i, name in enumerate(("train", "val_fit"))}
    return build_variants(matrices, FEATURES)


def test_both_variants_are_built(variants):
    assert set(variants) == {False, True}
    assert all(isinstance(variant.train, lgb.Dataset) for variant in variants.values())


def test_only_the_imputed_variant_has_its_nulls_filled(variants):
    assert variants[True].matrices["train"]["gappy"].isna().sum() == 0
    assert variants[False].matrices["train"]["gappy"].isna().sum() > 0


def test_validation_is_filled_too(variants):
    """A model fitted on filled data and scored on nulls meets a distribution it never saw."""
    assert variants[True].matrices["val_fit"]["gappy"].isna().sum() == 0


def test_the_medians_come_from_train_alone(variants):
    """Fitting them on both slices would leak validation into the fill value."""
    train_median = variants[False].matrices["train"]["gappy"].median()
    filled = variants[True].matrices["val_fit"]["gappy"]
    was_null = variants[False].matrices["val_fit"]["gappy"].isna()

    assert filled[was_null].eq(train_median).all()


def test_validation_references_the_training_bins(variants):
    """Bin edges are what a learned threshold means; a self-binned set moves them."""
    for variant in variants.values():
        assert variant.val_fit.reference is variant.train


def test_the_unimputed_variant_is_the_untouched_matrix(variants):
    """The reference configuration trains on nulls as they arrive."""
    matrices = {name: make_matrix(seed=i) for i, name in enumerate(("train", "val_fit"))}

    pd.testing.assert_frame_equal(variants[False].matrices["train"], matrices["train"])


# ------------------------------------------------------------------------------
# confirm
# ------------------------------------------------------------------------------

CONFIRM_CFG = {"tuned": {}, "seed": 0, "early_stopping_rounds": 5, "num_boost_round": 60}
CANDIDATE = ({"num_leaves": 7, "bagging_fraction": 0.6, "bagging_freq": 1}, True)


@pytest.fixture
def confirmation(variants, experiment_run) -> pd.DataFrame:
    return confirm(CANDIDATE, variants, FEATURES, CONFIRM_CFG, [0.1], range(2))


def test_both_configurations_run_at_every_seed(confirmation):
    assert list(zip(confirmation["config"], confirmation["seed"], strict=True)) == [
        ("candidate", 0),
        ("candidate", 1),
        ("reference", 0),
        ("reference", 1),
    ]


def test_every_confirmation_fit_is_a_child_run(confirmation, experiment_run):
    """The verdict is only reproducible if the numbers behind it are recorded."""
    names = [run.info.run_name for run in children(experiment_run)]

    assert names == [
        "confirm_candidate_seed0",
        "confirm_candidate_seed1",
        "confirm_reference_seed0",
        "confirm_reference_seed1",
    ]


def test_children_carry_what_the_verdict_was_computed_from(confirmation, experiment_run):
    logged = [run.data.metrics["val_fit.pr_auc"] for run in children(experiment_run)]

    assert logged == pytest.approx(list(confirmation["pr_auc"]))


def test_the_candidate_children_record_its_parameters(confirmation, experiment_run):
    candidate = [r for r in children(experiment_run) if r.data.params["config"] == "candidate"]

    assert {r.data.params["num_leaves"] for r in candidate} == {"7"}
    assert {r.data.params["impute"] for r in candidate} == {"True"}
