"""Seed spread — how far the same configuration moves on its own.

Its own command, for the reason ``features/floor.py`` is: this is a bar, not a
result. It changes when the data or the probe changes, not when a trial does, so
folding it into a tuning run would pay for it once per trial.

E6 registers what it is for and the rule it feeds. The short version: a tuning
run picks the best of many trials, and the maximum of noise is biased upward, so
a candidate has to beat the reference by more than either of them moves on its
own.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from pathlib import Path

import lightgbm as lgb
import mlflow
import pandas as pd

from fraud_engine.data.load import DEFAULT_CONFIG_PATH, load_config
from fraud_engine.evaluation.report import evaluate_splits, load_capacities
from fraud_engine.evaluation.tracking import (
    configure_tracking,
    flatten_metrics,
    tracked_child,
    tracked_run,
)
from fraud_engine.models.train import (
    apply_categories,
    feature_columns,
    fit,
    fit_categories,
    load_split_matrices,
    resolve_params,
    score,
    to_dataset,
)

log = logging.getLogger(__name__)


def measure(
    train: lgb.Dataset,
    val_fit: lgb.Dataset,
    matrices: dict[str, pd.DataFrame],
    columns: list[str],
    model_cfg: dict,
    capacities: list[float],
    seeds: Sequence[int],
) -> pd.DataFrame:
    """``VAL-FIT`` PR-AUC at each configured point, once per seed.

    Only the seed varies within a point. What moves is the row and column
    subsample each round draws, which moves the validation curve, which moves
    where early stopping lands — so ``best_iteration`` is recorded beside the
    metric. It is where the mechanism is visible, and a point whose iteration
    count swings by hundreds is not producing the same model twice.

    Measured through ``evaluate_splits`` rather than LightGBM's own metric, so
    the bar and the runs it judges are the same number computed the same way.

    Args:
        train: The training dataset.
        val_fit: The early-stopping dataset, referencing ``train``'s bins.
        matrices: Prepared matrices, for scoring.
        columns: Feature names.
        model_cfg: The ``model:`` config block, including ``spread.points``.
        capacities: Review capacities, so ``evaluate`` has its full input.
        seeds: One fit per seed, at every point.

    Returns:
        One row per point and seed: ``point``, ``seed``, ``pr_auc``,
        ``best_iteration``.
    """
    measured = []

    for point, tuned in model_cfg["spread"]["points"].items():
        for seed in seeds:
            point_cfg = {**model_cfg, "tuned": tuned, "seed": seed}

            with tracked_child(
                f"spread_{point}_seed{seed}", {**resolve_params(point_cfg), "point": point}
            ):
                booster = fit(train, val_fit, point_cfg)
                scored = score(booster, {"val_fit": matrices["val_fit"]}, columns)
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
                    "point": point,
                    "seed": seed,
                    "pr_auc": pr_auc,
                    "best_iteration": booster.best_iteration,
                }
            )
            log.info(
                "point=%-12s seed=%-3d val_fit pr_auc=%.5f  best_iteration=%d",
                point,
                seed,
                pr_auc,
                booster.best_iteration,
            )

    return pd.DataFrame(measured)


def summarize(spread: pd.DataFrame) -> pd.DataFrame:
    """Mean and spread per point — the shape E6's rule reads.

    Standard deviation with ``ddof=1``: these seeds are a sample of the draws the
    configuration could have made, not the population of them.
    """
    return spread.groupby("point", sort=False).agg(
        pr_auc_mean=("pr_auc", "mean"),
        pr_auc_std=("pr_auc", lambda column: column.std(ddof=1)),
        pr_auc_min=("pr_auc", "min"),
        pr_auc_max=("pr_auc", "max"),
        iteration_min=("best_iteration", "min"),
        iteration_max=("best_iteration", "max"),
    )


def main(config_path: Path = DEFAULT_CONFIG_PATH) -> None:
    """Measure the spread at every configured point and write it to reports/.

    Wiring only. Invoked by ``make spread`` as
    ``python -m fraud_engine.models.seeds``.
    """
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    config = load_config(config_path)
    paths, model_cfg = config["paths"], config["model"]

    matrices = load_split_matrices(paths["features_dir"], ("train", "val_fit"))
    vocabulary = fit_categories(matrices["train"], model_cfg["min_category_rows"])
    matrices = {split: apply_categories(frame, vocabulary) for split, frame in matrices.items()}

    columns = feature_columns(matrices["train"])
    train = to_dataset(matrices["train"], columns)
    val_fit = to_dataset(matrices["val_fit"], columns, reference=train)

    capacities = load_capacities(load_config(Path(paths["cost_matrix"])))
    seeds = range(model_cfg["spread"]["seeds"])

    configure_tracking(config["tracking"])
    points = ",".join(model_cfg["spread"]["points"])
    with tracked_run("seed_spread", {"seeds": len(seeds), "points": points}, config_path):
        spread = measure(train, val_fit, matrices, columns, model_cfg, capacities, seeds)

    path = Path(paths["seed_spread"])
    path.parent.mkdir(parents=True, exist_ok=True)
    spread.to_csv(path, index=False)

    log.info("\n%s", summarize(spread).to_string())
    log.info("wrote %s", path)


if __name__ == "__main__":
    main()
