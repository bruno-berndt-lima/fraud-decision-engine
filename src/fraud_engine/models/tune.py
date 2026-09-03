"""Hyperparameter search — deliberate, capped, and judged by E6's rule.

The trial cap is a compute budget and nothing more. A run picks the best of many
trials and the maximum of noise is biased upward, but that inflation grows with
the square root of the log of the trial count — going from thirty trials to two
hundred barely moves it. What removes the bias is re-measuring the winner
without selecting on it, which is what E6 registers and what this module ends
with.

No pruning. The space reaches down to small learning rates, and a model with a
small learning rate is slow by construction — that is the point of it. A pruner
judging at round fifty would kill exactly the configurations the theory favours,
and would feed the sampler a biased view of the space on top of that.
"""

from __future__ import annotations

from typing import NamedTuple

import lightgbm as lgb
import mlflow
import optuna
import pandas as pd

from fraud_engine.evaluation.report import evaluate_splits
from fraud_engine.models.train import fit, score


class Variant(NamedTuple):
    """One version of the data, built once and reused by every trial that picks it.

    Two exist: nulls as they arrive, and nulls filled. E2 measured that filling
    them helps and deliberately left the choice open, because the mechanism
    proposed for *why* it helps — near-constant columns becoming inert — is
    regularisation, and the search moves two other regularisers. Whether the
    benefit survives them is exactly what a search dimension is for.

    Datasets are constructed once. Rebuilding them per trial would re-bin three
    hundred thousand rows sixty times for no change, and reuse across differing
    ``min_child_samples`` was measured safe on these matrices rather than assumed.
    """

    train: lgb.Dataset
    val_fit: lgb.Dataset
    matrices: dict[str, pd.DataFrame]


def search_space(trial: optuna.Trial, space: dict) -> tuple[dict, bool]:
    """One trial's parameters, and whether it trains on imputed data.

    **Seven knobs, each with one job.** ``bagging_freq`` is fixed at 1 rather
    than searched: with ``bagging_fraction`` already controlling how much
    randomisation there is, frequency is a second way to express the same effect,
    and two knobs for one effect spends trials without buying resolution.

    **Log scales where the effect is multiplicative.** Going from 31 leaves to 63
    changes a model about as much as going from 63 to 127, so a uniform scale
    would spend most of its trials among large values that differ from each other
    hardly at all. Fractions are uniform, because there 0.4 to 0.5 and 0.9 to 1.0
    are comparable steps.

    **Imputation is returned separately, not as a parameter.** It selects which
    matrices the trial trains on; it is not something LightGBM understands. Put
    in the parameter mapping it would be *silently ignored* — LightGBM warns
    about unknown keys rather than refusing them — and every trial would train on
    the same data while the study recorded that they had not.

    Args:
        trial: The trial being suggested for.
        space: The ``model.tune.space`` config block — inclusive bounds per knob.

    Returns:
        ``(params, impute)``: the parameters to merge over the contract, and
        whether this trial's data has its nulls filled.
    """
    params = {
        # The complexity knob for leaf-wise growth. Not max_depth — LightGBM
        # splits the single most promising leaf rather than a whole level, so
        # depth is an outcome and this is the cause.
        "num_leaves": trial.suggest_int("num_leaves", *space["num_leaves"], log=True),
        # The floor under a leaf. At this base rate a twenty-row leaf holds well
        # under one fraud on average, so the low end of this range is where a
        # split stops being evidence and starts being memory.
        "min_child_samples": trial.suggest_int(
            "min_child_samples", *space["min_child_samples"], log=True
        ),
        # Floored well above zero because early_stopping_rounds is fixed. A
        # smaller step makes the validation curve flatter, and a fixed patience
        # closes on a flat curve sooner — measured, and the opposite of the usual
        # assumption that low rates need more rounds. The floor is set by that
        # interaction, not by compute.
        "learning_rate": trial.suggest_float("learning_rate", *space["learning_rate"], log=True),
        # Where stochastic regularisation enters at all: at 1.0 the seed does
        # nothing, which the seed-spread measurement showed exactly.
        "feature_fraction": trial.suggest_float("feature_fraction", *space["feature_fraction"]),
        "bagging_fraction": trial.suggest_float("bagging_fraction", *space["bagging_fraction"]),
        "bagging_freq": 1,
        # The two categorical knobs Decision B deferred to here. DeviceInfo keeps
        # seventy levels after the vocabulary floor, and a categorical split can
        # still carve a set out of them on very little evidence; these are what
        # hold that back.
        "min_data_per_group": trial.suggest_int(
            "min_data_per_group", *space["min_data_per_group"], log=True
        ),
        "cat_smooth": trial.suggest_float("cat_smooth", *space["cat_smooth"], log=True),
    }

    return params, trial.suggest_categorical("impute", [False, True])


def objective(
    trial: optuna.Trial,
    variants: dict[bool, Variant],
    columns: list[str],
    model_cfg: dict,
    capacities: list[float],
) -> float:
    """One trial: fit a configuration and report what it scores on ``VAL-FIT``.

    **The seed is fixed across trials**, per E6. Letting it vary would mean two
    trials could differ by nothing at all and the sampler would learn from the
    difference — it would be modelling the random number generator alongside the
    hyperparameters. The winner is re-measured across seeds afterwards, which is
    where seed variation belongs.

    **Scored through the harness**, not read off LightGBM's own metric, so a
    trial's number is the same quantity as the reference's and as every other run
    in ``reports/metrics/``.

    **A trial that exhausts the round budget is not caught.** ``fit`` raises, and
    the study stops. Swallowing it would silently drop the slowest
    configurations, which is the bias this module rejected pruning to avoid — and
    an aborted study says the ceiling needs raising, which is worth knowing.

    Args:
        trial: The trial being evaluated.
        variants: Both data versions, keyed by whether nulls are filled.
        columns: Feature names.
        model_cfg: The ``model:`` config block.
        capacities: Review capacities, so ``evaluate`` has its full input.

    Returns:
        ``VAL-FIT`` PR-AUC, which is what the study maximises.
    """
    params, impute = search_space(trial, model_cfg["tune"]["space"])
    variant = variants[impute]

    with mlflow.start_run(run_name=f"trial-{trial.number:03d}", nested=True):
        mlflow.log_params({**params, "impute": impute})

        booster = fit(
            variant.train,
            variant.val_fit,
            {**model_cfg, "tuned": params, "num_boost_round": model_cfg["tune"]["num_boost_round"]},
        )

        scored = score(booster, {"val_fit": variant.matrices["val_fit"]}, columns)
        pr_auc = evaluate_splits(scored, capacities, ("val_fit",))["val_fit"]["pr_auc"]

        mlflow.log_metrics({"val_fit.pr_auc": pr_auc, "best_iteration": booster.best_iteration})

    return pr_auc
