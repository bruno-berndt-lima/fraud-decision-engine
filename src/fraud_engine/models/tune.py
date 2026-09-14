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

import json
import logging
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import NamedTuple

import lightgbm as lgb
import mlflow
import optuna
import pandas as pd

from fraud_engine.data.load import DEFAULT_CONFIG_PATH, load_config
from fraud_engine.evaluation.report import evaluate_splits, git_revision, load_capacities
from fraud_engine.evaluation.tracking import (
    configure_tracking,
    flatten_metrics,
    tracked_child,
    tracked_run,
)
from fraud_engine.models.train import (
    apply_categories,
    apply_medians,
    feature_columns,
    fit,
    fit_categories,
    fit_medians,
    load_split_matrices,
    resolve_params,
    score,
    to_dataset,
)

log = logging.getLogger(__name__)


class Variant(NamedTuple):
    """One version of the data, built once and reused by every trial that picks it.

    Two exist: nulls as they arrive, and nulls filled. E2 measured that filling
    them helps and deliberately left the choice open, because the mechanism
    proposed for *why* it helps — near-constant columns becoming inert — is
    regularisation, and the search moves two other regularisers. Whether the
    benefit survives them is exactly what a search dimension is for.

    Datasets are constructed once. Rebuilding them per trial would re-bin three
    hundred thousand rows sixty times for no change. What makes reuse legal at
    all is ``to_dataset`` disabling ``feature_pre_filter`` — without it LightGBM
    refuses to train a constructed dataset with a *smaller* ``min_data_in_leaf``
    than it was built under, and a search that moves that knob downward does
    exactly that.
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

    # What the trial actually trained on, stored rather than reconstructed.
    # `study.best_params` carries only the *suggested* values, and bagging_freq
    # is set rather than suggested — rebuilding the winner from best_params
    # would drop it, and the confirmation run would not be the winning
    # configuration.
    trial.set_user_attr("params", params)
    trial.set_user_attr("impute", impute)

    trial_cfg = {**model_cfg, "tuned": params}

    with tracked_child(
        f"trial-{trial.number:03d}", {**resolve_params(trial_cfg), "impute": impute}
    ):
        booster = fit(variant.train, variant.val_fit, trial_cfg)

        scored = score(booster, {"val_fit": variant.matrices["val_fit"]}, columns)
        evaluated = evaluate_splits(scored, capacities, ("val_fit",))
        mlflow.log_metrics(
            {**flatten_metrics({"splits": evaluated}), "best_iteration": booster.best_iteration}
        )

    return evaluated["val_fit"]["pr_auc"]


def confirm(
    candidate: tuple[dict, bool],
    variants: dict[bool, Variant],
    columns: list[str],
    model_cfg: dict,
    capacities: list[float],
    seeds: Sequence[int],
) -> pd.DataFrame:
    """Re-run the winner and the untuned reference across seeds, per E6.

    The search maximised over sixty trials, so the winner's own number is the
    maximum of a noisy sample and is biased upward by construction. These fits
    select nothing: the configuration is already fixed, and every seed's result
    counts. That is what makes the mean unbiased where the search's number was
    not.

    **The reference is re-run too, not assumed.** It samples neither rows nor
    columns, so its seeds should return one number repeated — and running them
    is what turns that from an expectation into a check. Ten identical fits are
    cheap; a reference that turned out not to be deterministic would invalidate
    every comparison in this phase, and finding that out here is the point.

    Both configurations get the same round ceiling — the one the shipped model
    trains under — so the only difference between them is the parameters, and
    what is confirmed here is what `make train` produces.

    Args:
        candidate: The winning ``(params, impute)``, from the trial's own record.
        variants: Both data versions.
        columns: Feature names.
        model_cfg: The ``model:`` config block.
        capacities: Review capacities.
        seeds: One fit per seed, per configuration.

    Returns:
        One row per configuration and seed: ``config``, ``seed``, ``pr_auc``,
        ``best_iteration``.
    """
    winner_params, winner_impute = candidate

    # The reference is the shipped untuned model: no tuned knobs, nulls as they
    # arrive. It is what Phase 06 calibrates if the candidate does not clear E6.
    configurations = {
        "candidate": (winner_params, winner_impute),
        "reference": ({}, False),
    }

    measured = []

    for name, (params, impute) in configurations.items():
        variant = variants[impute]
        for seed in seeds:
            seed_cfg = {**model_cfg, "tuned": params, "seed": seed}

            # The runs E6's verdict is computed from, so each is recorded: the
            # decision is only reproducible if the twenty numbers behind it are.
            with tracked_child(
                f"confirm_{name}_seed{seed}",
                {**resolve_params(seed_cfg), "config": name, "impute": impute},
            ):
                booster = fit(variant.train, variant.val_fit, seed_cfg)
                scored = score(booster, {"val_fit": variant.matrices["val_fit"]}, columns)
                evaluated = evaluate_splits(scored, capacities, ("val_fit",))
                mlflow.log_metrics(
                    {
                        **flatten_metrics({"splits": evaluated}),
                        "best_iteration": booster.best_iteration,
                    }
                )
            pr_auc = evaluated["val_fit"]["pr_auc"]

            measured.append(
                {
                    "config": name,
                    "seed": seed,
                    "pr_auc": pr_auc,
                    "best_iteration": booster.best_iteration,
                }
            )
            log.info(
                "%-10s seed=%-3d val_fit pr_auc=%.5f  best_iteration=%d",
                name,
                seed,
                pr_auc,
                booster.best_iteration,
            )

    return pd.DataFrame(measured)


def verdict(runs: pd.DataFrame) -> dict:
    """E6's rule, applied. Pure, so it is testable without fitting anything.

    ``mean(candidate) - mean(reference) > std(candidate) + std(reference)``

    The sum of the standard deviations, not the standard error of their
    difference. The standard error shrinks with the number of seeds and would
    let a larger seed budget buy significance; the sum does not move. That
    strictness is the price of the candidate having been selected as the maximum
    over many trials, which no single-comparison test accounts for.

    Args:
        runs: As ``confirm`` returned them.

    Returns:
        Both means, both spreads, the gap, the bar it had to clear, and whether
        it did.
    """
    stats = runs.groupby("config")["pr_auc"].agg(["mean", "std"])

    gap = stats.loc["candidate", "mean"] - stats.loc["reference", "mean"]
    bar = stats.loc["candidate", "std"] + stats.loc["reference", "std"]

    return {
        "candidate_mean": stats.loc["candidate", "mean"],
        "candidate_std": stats.loc["candidate", "std"],
        "reference_mean": stats.loc["reference", "mean"],
        "reference_std": stats.loc["reference", "std"],
        "gap": gap,
        "bar": bar,
        "accepted": bool(gap > bar),
    }


def build_variants(matrices: dict[str, pd.DataFrame], columns: list[str]) -> dict[bool, Variant]:
    """Both data versions, binned once.

    The medians come from train and reach validation too, exactly as E2's
    ``imputed`` arm did — a model fitted on filled data and scored on data still
    carrying nulls would be measured on a distribution it never saw.
    """
    medians = fit_medians(matrices["train"], columns)
    filled = {split: apply_medians(frame, medians) for split, frame in matrices.items()}

    variants = {}
    for impute, frames in ((False, matrices), (True, filled)):
        train = to_dataset(frames["train"], columns)
        variants[impute] = Variant(
            train, to_dataset(frames["val_fit"], columns, reference=train), frames
        )
    return variants


def main(config_path: Path = DEFAULT_CONFIG_PATH) -> None:
    """Run the study, apply E6's rule, and write the verdict.

    Wiring only. Invoked by ``make tune`` as ``python -m fraud_engine.models.tune``.

    **Nothing here changes the shipped model.** The winning parameters are
    written to a record and printed as the config block that would adopt them.
    Adopting is a separate, committed edit to ``config.yaml``, so the history
    shows when tuning changed the model and `make train` remains the one stage
    that writes it.

    Args:
        config_path: Path to ``config.yaml``.
    """
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    config = load_config(config_path)
    paths, model_cfg = config["paths"], config["model"]
    tune_cfg = model_cfg["tune"]

    configure_tracking(config["tracking"])

    matrices = load_split_matrices(paths["features_dir"], ("train", "val_fit"))
    vocabulary = fit_categories(matrices["train"], model_cfg["min_category_rows"])
    matrices = {split: apply_categories(frame, vocabulary) for split, frame in matrices.items()}

    columns = feature_columns(matrices["train"])
    variants = build_variants(matrices, columns)
    capacities = load_capacities(load_config(Path(paths["cost_matrix"])))

    study = optuna.create_study(
        direction="maximize",
        # Seeded, so the study is a rerunnable stage rather than a one-off. The
        # sampler's own randomness is not the seed E6 measures — that one is
        # fixed inside every trial.
        sampler=optuna.samplers.TPESampler(seed=model_cfg["seed"]),
    )

    with tracked_run("tuning", {"trials": tune_cfg["trials"]}, config_path):
        study.optimize(
            lambda trial: objective(trial, variants, columns, model_cfg, capacities),
            n_trials=tune_cfg["trials"],
        )

        best = study.best_trial
        candidate = (best.user_attrs["params"], best.user_attrs["impute"])
        log.info("\nbest trial #%d: %.5f", best.number, best.value)

        runs = confirm(
            candidate, variants, columns, model_cfg, capacities, range(model_cfg["spread"]["seeds"])
        )
        decision = verdict(runs)

        mlflow.log_metrics({key: value for key, value in decision.items() if key != "accepted"})
        mlflow.log_param("accepted", decision["accepted"])

    record = {
        "name": "tuning",
        "created": datetime.now(UTC).isoformat(timespec="seconds"),
        "git_revision": git_revision(),
        "trials": tune_cfg["trials"],
        "winner": {"params": candidate[0], "impute": candidate[1], "search_pr_auc": best.value},
        "verdict": decision,
        "confirmation": runs.to_dict(orient="records"),
        "history": [
            {"number": trial.number, "pr_auc": trial.value, **trial.params}
            for trial in study.trials
        ],
    }

    path = Path(paths["tuning"])
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(record, indent=2, default=float) + "\n")

    log.info(
        "\ncandidate %.5f +- %.5f   reference %.5f +- %.5f\ngap %.5f against a bar of %.5f -> %s",
        decision["candidate_mean"],
        decision["candidate_std"],
        decision["reference_mean"],
        decision["reference_std"],
        decision["gap"],
        decision["bar"],
        "ACCEPTED" if decision["accepted"] else "REJECTED, the untuned reference ships",
    )
    if decision["accepted"]:
        log.info(
            "\nto adopt, set model.tuned in config.yaml:\n%s", json.dumps(candidate[0], indent=2)
        )
    log.info("wrote %s", path)


if __name__ == "__main__":
    main()
