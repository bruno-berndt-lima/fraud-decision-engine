"""E5 — is the train/validation gap the model, or the data it was given?

Registered in `docs/experiments.md` E5 and `docs/monitoring.md` §5. Identity coverage
falls across the split boundary, and this stratifies the shipped model's scores by
`has_identity` on both sides of it. Nothing is fitted or tuned.

**Train is scored here, and its scores are in-sample.** They are the one vector no
stage persisted, so the reloaded booster first reproduces its recorded `VAL-FIT` and
`VAL-CAL` scores exactly, then scores train. The booster memorised those rows: no train
figure is read as a generalisation estimate.

**The compositional part is measured on validation alone.** PR-AUC does not add across
strata, so validation is reweighted to train's identity mix and scored again. The
difference between that and validation as it is holds only rows the model never saw.

**Test is not read.** E5's first constraint, and the splits below are why.
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
from sklearn.metrics import average_precision_score

from fraud_engine.data.load import DEFAULT_CONFIG_PATH, load_config
from fraud_engine.evaluation.metrics import evaluate
from fraud_engine.evaluation.report import git_revision, load_capacities
from fraud_engine.evaluation.reproduce import check_reproduces
from fraud_engine.evaluation.tracking import configure_tracking, tracked_run
from fraud_engine.models.train import (
    LABEL,
    feature_columns,
    prepare_matrices,
    run_name,
    score,
)

log = logging.getLogger(__name__)

NAME = "composition"
STRATUM = "has_identity"
TRAIN = "train"
VALIDATION = ("val_fit", "val_cal")
SPLITS = (TRAIN, *VALIDATION)
POOLED = "validation"
STRATA = {True: "identity", False: "no_identity"}


def weighted_pr_auc(y_true: pd.Series, y_score: pd.Series, weight: pd.Series) -> float:
    """Average precision with each row counted `weight` times.

    `metrics.pr_auc` with weights, and nothing else: at unit weights the two agree
    exactly, which the tests hold. It is not a parameter on `pr_auc` itself because
    every stage that scores a baseline names `metrics.py` as a prerequisite.

    Raises:
        ValueError: On misaligned inputs, a single class, or a weight that is not
            finite and positive.
    """
    if not (len(y_true) == len(y_score) == len(weight)):
        raise ValueError("labels, scores and weights are not the same length")
    if y_true.nunique() != 2:
        raise ValueError("average precision needs both classes")
    if not np.isfinite(weight).all() or (weight <= 0).any():
        raise ValueError("every weight must be finite and positive")
    return float(average_precision_score(y_true, y_score, sample_weight=weight))


def identity_share(frame: pd.DataFrame) -> float:
    """The share of rows that arrived with an identity block."""
    return float(frame[STRATUM].mean())


def mix_weights(validation: pd.DataFrame, train_share: float) -> pd.Series:
    """Per-row weights that give validation train's identity mix.

    A row's weight is its stratum's share in train over its share in validation, so
    the weighted identity share is train's and nothing inside a stratum changes.

    Raises:
        ValueError: If either stratum is empty on either side; its weight is undefined.
    """
    share = identity_share(validation)
    for side, value in (("train", train_share), ("validation", share)):
        if not 0 < value < 1:
            raise ValueError(f"{side} holds only one stratum; there is no mix to reweight")

    identity = validation[STRATUM].astype(bool)
    weights = np.where(identity, train_share / share, (1 - train_share) / (1 - share))
    return pd.Series(weights, index=validation.index)


def decompose(train: pd.DataFrame, validation: pd.DataFrame) -> dict:
    """The pooled gap, and the part of it a change of identity mix accounts for.

    Returns:
        `train_pr_auc` (in-sample), `validation_pr_auc`, `reweighted_pr_auc`,
        `pooled_gap`, `compositional` (reweighted less raw validation — out of
        sample), `remainder`, and `share_of_gap`, which is `None` when there is
        no gap to share.
    """
    ones = pd.Series(1.0, index=validation.index)
    raw = weighted_pr_auc(validation[LABEL], validation["score"], ones)
    reweighted = weighted_pr_auc(
        validation[LABEL], validation["score"], mix_weights(validation, identity_share(train))
    )
    fitted = weighted_pr_auc(train[LABEL], train["score"], pd.Series(1.0, index=train.index))

    gap = fitted - raw
    compositional = reweighted - raw
    return {
        "train_pr_auc": fitted,
        "validation_pr_auc": raw,
        "reweighted_pr_auc": reweighted,
        "pooled_gap": gap,
        "compositional": compositional,
        "remainder": gap - compositional,
        "share_of_gap": compositional / gap if gap > 0 else None,
    }


def cells(scored: pd.DataFrame, capacities: list[float]) -> dict:
    """E5's table: every side, pooled and by stratum, through the harness's `evaluate`.

    Recall is taken at each capacity *within* the cell — the top share of that
    stratum's own daily rows — so it measures ranking inside the stratum, not what
    the policy would send to review.

    Returns:
        `{side: {"all" | "identity" | "no_identity": evaluate(...)}}`, sides being
        each split and the pooled validation.
    """
    sides = {split: scored[scored["split"] == split] for split in SPLITS}
    sides[POOLED] = scored[scored["split"].isin(VALIDATION)]

    table = {}
    for side, frame in sides.items():
        parts = {"all": frame} | {
            name: frame[frame[STRATUM] == flag] for flag, name in STRATA.items()
        }
        table[side] = {
            name: evaluate(part[LABEL], part["score"], part["day"], capacities)
            for name, part in parts.items()
        }
    return table


def score_splits(features_dir: Path | str, model_path: Path | str, model_cfg: dict) -> pd.DataFrame:
    """The shipped booster, reloaded, scoring train and both validation slices.

    Matrices are prepared as `make train` prepared them; the proofs in `main` show that
    path still yields the model's inputs.

    Returns:
        The harness's scored frame, with each row's `has_identity` beside it.

    Raises:
        ValueError: If the booster's features are not the matrices', in order.
    """
    matrices, _, _ = prepare_matrices(features_dir, model_cfg, SPLITS)
    booster = lgb.Booster(model_file=str(model_path))

    columns = booster.feature_name()
    if columns != feature_columns(matrices[TRAIN]):
        raise ValueError("the booster's features are not the matrices' features, in order")

    scored = score(booster, matrices, columns)
    identity = pd.concat(
        [frame[["TransactionID", STRATUM]] for frame in matrices.values()], ignore_index=True
    )
    return scored.merge(identity, on="TransactionID", how="left", validate="one_to_one").assign(
        **{STRATUM: lambda frame: frame[STRATUM].astype(bool)}
    )


def main(config_path: Path = DEFAULT_CONFIG_PATH) -> None:
    """Prove, score, stratify and record E5.

    Wiring only. Invoked by `make composition` as
    `python -m fraud_engine.monitoring.composition`.

    Args:
        config_path: Path to `config.yaml`.
    """
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    config = load_config(config_path)
    paths, model_cfg = config["paths"], config["model"]
    configure_tracking(config["tracking"])
    capacities = load_capacities(load_config(Path(paths["cost_matrix"])))

    model_name = run_name(model_cfg)
    scored = score_splits(paths["features_dir"], paths["model"], model_cfg)
    recorded = pd.read_parquet(Path(paths["predictions_dir"]) / f"{model_name}.parquet")
    for split in VALIDATION:
        check_reproduces(scored, recorded, model_name, split)
    log.info("proofs passed: %s reproduces its VAL-FIT and VAL-CAL records", model_name)

    train = scored[scored["split"] == TRAIN]
    validation = scored[scored["split"].isin(VALIDATION)]
    mix = {split: identity_share(scored[scored["split"] == split]) for split in SPLITS}
    mix[POOLED] = identity_share(validation)

    params = {"model": model_name, "stratum": STRATUM, "splits": ",".join(SPLITS)}
    with tracked_run(NAME, params, config_path):
        table = cells(scored, capacities)
        decomposition = decompose(train, validation)
        mlflow.log_metrics(
            {f"mix.{side}": share for side, share in mix.items()}
            | {k: v for k, v in decomposition.items() if v is not None}
            | {
                f"{side}.{stratum}.pr_auc": block["pr_auc"]
                for side, strata in table.items()
                for stratum, block in strata.items()
            }
        )

    record = {
        "name": NAME,
        "created": datetime.now(UTC).isoformat(timespec="seconds"),
        "git_revision": git_revision(),
        "model": model_name,
        "stratum": STRATUM,
        "splits": list(SPLITS),
        "in_sample": [TRAIN],
        "recall_capacity_within": "stratum",
        "proofs": {f"{model_name} reproduces {split}": True for split in VALIDATION},
        "identity_share": mix,
        "cells": table,
        "decomposition": decomposition,
    }
    record_path = Path(paths["composition"])
    record_path.write_text(json.dumps(record, indent=2) + "\n")

    log.info(
        "compositional %+.5f of a pooled gap %.5f; validation %.5f, reweighted %.5f",
        decomposition["compositional"],
        decomposition["pooled_gap"],
        decomposition["validation_pr_auc"],
        decomposition["reweighted_pr_auc"],
    )
    log.info("wrote %s", record_path)


if __name__ == "__main__":
    main()
