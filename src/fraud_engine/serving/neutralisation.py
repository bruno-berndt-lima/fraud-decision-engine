"""What serving without the live-entity store costs, measured on VAL-CAL.

Registered in `docs/serving.md` §2. The service defaults the four tier-3 columns to the
values the velocity family itself assigns a card it has never seen, because no caller in
this scenario can populate them and E3 twice failed to show the store pays. That default
shifts the distribution the model meets, and this stage prices the shift rather than
asserting it is small.

**Both arms are the same booster on the same transactions.** Nothing is refitted: the
shipped model scores VAL-CAL twice, once on the matrix as built and once with the four
columns replaced. So the difference carries no fit variation at all — which is what E3's
own reading rule said its arms could not claim.

**Nothing here may change anything** (§9). VAL-CAL rather than test, because choosing a
serving default is a decision and decisions do not touch test; out-of-fold probabilities
throughout, per Phase 06's rule for everything measured on that slice.
"""

from __future__ import annotations

import json
import logging
from datetime import UTC, datetime
from pathlib import Path

import lightgbm as lgb
import mlflow
import numpy as np
import pandas as pd

from fraud_engine.data.load import DEFAULT_CONFIG_PATH, load_config
from fraud_engine.evaluation.arms import cost_arm, paired_day_bootstrap
from fraud_engine.evaluation.cost import load_costs
from fraud_engine.evaluation.metrics import pr_auc
from fraud_engine.evaluation.policy import load_rehearsal_frame
from fraud_engine.evaluation.report import git_revision, load_operating_capacity
from fraud_engine.evaluation.reproduce import check_reproduces
from fraud_engine.evaluation.tracking import configure_tracking, tracked_child, tracked_run
from fraud_engine.features import velocity
from fraud_engine.models.calibrate import SPLIT
from fraud_engine.models.train import LABEL, feature_columns, prepare_matrices, run_name, score
from fraud_engine.serving.transform import fill_history

log = logging.getLogger(__name__)

NAME = "neutralisation"

# `train` is not scored; `prepare_matrices` fits the vocabulary and the medians on it, and
# a slice re-levelled against its own categories would give the booster codes standing for
# different levels than the ones it was fitted on.
SPLITS = ("train", SPLIT)

INTACT, NEUTRALISED = "with_history", "neutralised"


def neutralise(matrix: pd.DataFrame, features_cfg: dict) -> pd.DataFrame:
    """Every row as the card nobody has seen before — what a request without a store is.

    Dropped before it is filled, because `fill_history` keeps tier-3 columns a caller
    supplied and would therefore keep all four. Going through that function rather than
    writing the values here is the point: what serving does to a request is what is
    measured, and a second definition of the defaults could drift from it silently.

    Args:
        matrix: A prepared matrix, carrying the velocity family.
        features_cfg: The `features:` config block.

    Returns:
        The same matrix with the four velocity columns replaced by their no-history
        values. Column order is not preserved; callers index by feature name.
    """
    return fill_history(matrix.drop(columns=list(velocity.COLUMNS)), features_cfg["velocity"])


def first_sighting_share(matrix: pd.DataFrame, neutralised: pd.DataFrame) -> float:
    """Share of rows the default was already true of.

    The delta cannot be read without this. A slice whose cards are mostly new to the
    window is barely moved by neutralising the family, and that would be a fact about
    VAL-CAL rather than about the serving default.
    """
    columns = list(velocity.COLUMNS)
    unchanged = matrix[columns].to_numpy() == neutralised.loc[matrix.index, columns].to_numpy()
    return float(unchanged.all(axis=1).mean())


def scored_arms(
    booster: lgb.Booster, matrix: pd.DataFrame, columns: list[str], features_cfg: dict
) -> tuple[dict[str, pd.DataFrame], float]:
    """Both arms' scores, and how much of the slice the default already described.

    Returns:
        `({arm: scored}, first-sighting share)`, each scored frame as `score` shapes it.
    """
    without = neutralise(matrix, features_cfg)

    return {
        INTACT: score(booster, {SPLIT: matrix}, columns),
        NEUTRALISED: score(booster, {SPLIT: without}, columns),
    }, first_sighting_share(matrix, without)


def measure(
    frame: pd.DataFrame,
    arms: dict[str, pd.DataFrame],
    costs,
    capacity: float,
    calibration: dict,
    method: str,
    bootstrap: dict,
) -> dict:
    """Each arm's PR-AUC and USD, and the interval on the difference between them.

    PR-AUC is read off the raw scores, which is what the metric has meant everywhere else
    in this project; the USD comes from out-of-fold calibrated probabilities, which is
    what Phase 06 requires of anything measured on this slice. The two therefore answer
    slightly different questions about the same arm, and the record says which is which.

    Args:
        frame: From `policy.load_rehearsal_frame` — labels, days, amounts.
        arms: `{arm: scored}`, both covering the same VAL-CAL rows.
        costs: The version-1 costs.
        capacity: Review capacity, as a share of daily volume.
        calibration: The `calibration:` config block.
        method: The calibration method §1 of `decision-policy.md` selected.
        bootstrap: The `neutralisation:` config block.

    Returns:
        `{arm: {pr_auc, usd_per_1000, ...}}` under `arms`, plus the deltas and the
        paired interval under `delta`.
    """
    y = frame[LABEL]
    results, per_transaction = {}, {}

    for arm, scored in arms.items():
        summary, cost = cost_arm(frame, scored, costs, capacity, calibration, method)

        aligned = scored.set_index("TransactionID").loc[frame["TransactionID"]]
        summary["pr_auc"] = pr_auc(y, aligned["score"].reset_index(drop=True))

        results[arm], per_transaction[arm] = summary, cost

    low, high = paired_day_bootstrap(
        frame["day"].to_numpy(),
        per_transaction[NEUTRALISED],
        per_transaction[INTACT],
        bootstrap["bootstrap_resamples"],
        bootstrap["interval"],
        bootstrap["seed"],
    )

    return {
        "arms": results,
        "delta": {
            "pr_auc": results[NEUTRALISED]["pr_auc"] - results[INTACT]["pr_auc"],
            "usd_per_1000": results[NEUTRALISED]["usd_per_1000"] - results[INTACT]["usd_per_1000"],
            "usd_interval": [low, high],
            "interval": bootstrap["interval"],
        },
    }


def main(config_path: Path = DEFAULT_CONFIG_PATH) -> None:
    """Score VAL-CAL with and without the card's history, and write the record.

    Wiring only. Invoked by `make neutralisation` as
    `python -m fraud_engine.serving.neutralisation`.

    Args:
        config_path: Path to `config.yaml`.
    """
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    config = load_config(config_path)
    paths, model_cfg = config["paths"], config["model"]

    configure_tracking(config["tracking"])

    cost_matrix = load_config(Path(paths["cost_matrix"]))
    costs = load_costs(cost_matrix)
    capacity = load_operating_capacity(cost_matrix)
    method = json.loads(Path(paths["calibration"]).read_text())["selected"]

    frame = load_rehearsal_frame(paths["predictions_dir"], paths["interim"], method)

    matrices, _, _ = prepare_matrices(paths["features_dir"], model_cfg, SPLITS)
    matrix = matrices[SPLIT]
    columns = feature_columns(matrices["train"])

    booster = lgb.Booster(model_file=paths["model"])
    arms, first_sightings = scored_arms(booster, matrix, columns, config["features"])

    # Before anything new is attached to it: the booster reloaded from disk must reproduce
    # the scores its own record was made from. The intact arm is that rescoring, so the
    # proof costs nothing beyond the comparison.
    recorded = pd.read_parquet(Path(paths["predictions_dir"]) / f"{run_name(model_cfg)}.parquet")
    check_reproduces(arms[INTACT], recorded, NAME, split=SPLIT)

    params = {
        "split": SPLIT,
        "calibration": method,
        "review_capacity": capacity,
        "cost_matrix_version": costs.version,
        "neutralised": ",".join(velocity.COLUMNS),
        **config["neutralisation"],
    }

    with tracked_run(NAME, params, config_path):
        results = measure(
            frame, arms, costs, capacity, config["calibration"], method, config["neutralisation"]
        )

        for arm, summary in results["arms"].items():
            with tracked_child(arm, {"history": arm == INTACT}):
                mlflow.log_metrics(summary)

        mlflow.log_metrics(
            {
                "delta.pr_auc": results["delta"]["pr_auc"],
                "delta.usd_per_1000": results["delta"]["usd_per_1000"],
                "first_sighting_share": first_sightings,
            }
        )

    for arm, summary in results["arms"].items():
        log.info(
            "%-14s PR-AUC %.5f   $%9.2f per 1,000   block %.4f",
            arm,
            summary["pr_auc"],
            summary["usd_per_1000"],
            summary["block_rate"],
        )
    delta = results["delta"]
    log.info(
        "serving without the store: PR-AUC %+.5f, $%+.2f per 1,000 [%+.2f, %+.2f]",
        delta["pr_auc"],
        delta["usd_per_1000"],
        *delta["usd_interval"],
    )

    record = {
        "name": NAME,
        "created": datetime.now(UTC).isoformat(timespec="seconds"),
        "git_revision": git_revision(),
        "split": SPLIT,
        "rows": len(frame),
        "positives": int(frame[LABEL].sum()),
        "days": int(np.unique(frame["day"]).size),
        "model": run_name(model_cfg),
        "neutralised": list(velocity.COLUMNS),
        "first_sighting_share": first_sightings,
        "pr_auc_from": "raw scores",
        "usd_from": f"{method}, out-of-fold",
        "review_capacity": capacity,
        "cost_matrix_version": costs.version,
        **results,
    }
    path = Path(paths["neutralisation"])
    path.write_text(json.dumps(record, indent=2) + "\n")
    log.info("wrote %s", path)


if __name__ == "__main__":
    main()
