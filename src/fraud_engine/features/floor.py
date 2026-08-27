"""The noise floor — how far chance alone moves the metric.

Its own command rather than part of a family run, because it is *deterministic*:
the same seeds produce the same numbers, so folding it into `evaluate` would
spend eighteen minutes per run recomputing a constant. Measure it when the probe
changes, not when the families do.

E4 records what it is for, and the correction that followed from measuring it
twice: the floor did not rise between width 1 and width 3, so families are
compared against one pooled bar rather than a per-width maximum.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import pandas as pd

from fraud_engine.data.load import DEFAULT_CONFIG_PATH, load_config
from fraud_engine.evaluation.report import evaluate_splits, load_capacities
from fraud_engine.features.evaluate import NOISE_COLUMN, load_matrices, score_with
from fraud_engine.models.logistic import SOURCE_COLUMNS, prepare, select_delta_columns

log = logging.getLogger(__name__)


def noise_floor(
    frame: pd.DataFrame,
    delta_columns: list[str],
    logistic_cfg: dict,
    capacities: list[float],
    seeds: Sequence[int],
    n_columns: int = 1,
) -> pd.DataFrame:
    """VAL-FIT PR-AUC with ``n_columns`` meaningless columns added, once per seed.

    The probe is deterministic given its data, so this is not run-to-run
    variance. It is how far columns carrying no information at all can move the
    metric by chance — and a family that beats the baseline by less than this
    spread has demonstrated nothing.

    Measured at a chosen width because the floor rises with it: more columns are
    more chances for the fit to read signal into noise. A floor taken at one
    column understates what a six-column family has to clear.

    Args:
        frame: Every non-gap row, carrying ``split``.
        delta_columns: From ``select_delta_columns``, measured on train.
        logistic_cfg: The ``baselines.logistic`` config block.
        capacities: Review capacities, only so ``evaluate`` has its full input.
        seeds: One run per seed. The spread across them is the result; a single
            seed is one draw and says nothing.
        n_columns: How many noise columns to add, matching the width of the
            family being judged.

    Returns:
        One row per seed: ``seed``, ``n_columns``, ``pr_auc``.
    """
    noise_columns = tuple(f"{NOISE_COLUMN}{index}" for index in range(n_columns))

    # One copy, reused: each seed overwrites the noise columns rather than
    # rebuilding a half-million-row frame per draw.
    working = frame.copy()
    measured = []

    for seed in seeds:
        rng = np.random.default_rng(seed)
        for column in noise_columns:
            working[column] = rng.standard_normal(len(working))

        scored = score_with(working, noise_columns, delta_columns, logistic_cfg)
        pr_auc = evaluate_splits(scored, capacities, ("val_fit",))["val_fit"]["pr_auc"]

        measured.append({"seed": seed, "n_columns": n_columns, "pr_auc": pr_auc})
        log.info("noise seed=%-4d width=%d  val_fit pr_auc=%.5f", seed, n_columns, pr_auc)

    return pd.DataFrame(measured)


def main(config_path: Path = DEFAULT_CONFIG_PATH) -> None:
    """Measure the floor at every configured width and write it to reports/.

    Wiring only. Invoked by ``make floor`` as
    ``python -m fraud_engine.features.floor``.
    """
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    config = load_config(config_path)
    paths = config["paths"]
    logistic_cfg = config["baselines"]["logistic"]
    evaluation_cfg = config["features"]["evaluation"]
    capacities = load_capacities(load_config(Path(paths["cost_matrix"])))

    frame = prepare(load_matrices(paths["features_dir"], SOURCE_COLUMNS))
    train = frame[frame["split"] == "train"]
    delta_columns = select_delta_columns(train, logistic_cfg["d_max_null_frac"])

    baseline = evaluate_splits(
        score_with(frame, (), delta_columns, logistic_cfg), capacities, ("val_fit",)
    )["val_fit"]["pr_auc"]
    log.info("bare probe val_fit pr_auc=%.5f", baseline)

    measured = pd.concat(
        [
            noise_floor(
                frame,
                delta_columns,
                logistic_cfg,
                capacities,
                range(evaluation_cfg["noise_seeds"]),
                width,
            )
            for width in evaluation_cfg["noise_widths"]
        ],
        ignore_index=True,
    )
    measured["delta"] = measured["pr_auc"] - baseline

    path = Path(paths["noise_floor"])
    path.parent.mkdir(parents=True, exist_ok=True)
    measured.to_csv(path, index=False)

    for width, group in measured.groupby("n_columns"):
        log.info(
            "noise floor width=%d over %d seeds: mean %+.5f, max %+.5f",
            width,
            len(group),
            group["delta"].mean(),
            group["delta"].max(),
        )
    log.info("wrote %s", path)


if __name__ == "__main__":
    main()
