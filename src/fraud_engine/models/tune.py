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

import optuna


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
