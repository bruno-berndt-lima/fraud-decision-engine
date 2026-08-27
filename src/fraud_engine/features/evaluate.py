from __future__ import annotations

import logging
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import pandas as pd

from fraud_engine.data.load import DEFAULT_CONFIG_PATH, load_config
from fraud_engine.data.splits import SPLIT_NAMES
from fraud_engine.evaluation.report import evaluate_splits, load_capacities, write_run
from fraud_engine.features import amounts
from fraud_engine.models.logistic import (
    FEATURE_COLUMNS,
    SOURCE_COLUMNS,
    build_pipeline,
    prepare,
    select_delta_columns,
)

log = logging.getLogger(__name__)

# Name -> the engineered columns it contributes. "none" is the bare probe, the
# reference every delta and the noise floor itself are measured against.
FAMILIES: dict[str, tuple[str, ...]] = {
    "none": (),
    "amount": amounts.COLUMNS,
}

NOISE_COLUMN = "_noise"


def load_matrices(features_dir: Path | str, columns: Sequence[str]) -> pd.DataFrame:
    """The four matrices as one frame, with ``split`` restored from the filenames.

    ``partition`` drops the label because the filename already carries it. This
    is the other half of that trade, paid once here instead of in every consumer.

    Only ``columns`` are read. The matrices are self-contained and 438 columns
    wide, so naming what a run needs is the difference between a narrow frame
    and four full copies of the dataset in memory.

    Concatenating in ``SPLIT_NAMES`` order leaves the result in causal order,
    since the splits are chronological and each matrix is sorted. Nothing here
    relies on that — fitting and per-split metrics are both order-blind — but a
    caller that needs it does not have to re-sort. Gap rows are absent; they
    have no matured label to fit or score on.

    Args:
        features_dir: Directory holding ``{split}.parquet``.
        columns: Columns to read from every matrix. ``split`` is added here and
            must not be named.

    Returns:
        Every non-gap row, carrying ``columns`` plus a categorical ``split``.

    Raises:
        ValueError: If ``columns`` names ``split``.
        FileNotFoundError: If a split's matrix is absent.
        pyarrow.lib.ArrowInvalid: If a named column is absent from a matrix.
    """
    if "split" in columns:
        raise ValueError("`split` is restored from the filenames, not read; drop it from columns.")

    features_dir = Path(features_dir)
    matrices = []

    for name in SPLIT_NAMES:
        matrix = pd.read_parquet(features_dir / f"{name}.parquet", columns=list(columns))
        # Categorical rather than object: 590,540 repeated short strings cost
        # tens of megabytes as objects, and the categories are a closed set.
        matrix["split"] = pd.Categorical([name] * len(matrix), categories=SPLIT_NAMES)
        matrices.append(matrix)

    return pd.concat(matrices, ignore_index=True)


def score_with(
    frame: pd.DataFrame,
    extra_numeric: tuple[str, ...],
    delta_columns: list[str],
    logistic_cfg: dict,
) -> pd.DataFrame:
    """Fit the probe on train with these extra columns, then score every row.

    ``class_weight`` is fixed at ``None``. E2 measured that variant the stronger
    of the two on VAL-FIT PR-AUC — 0.322 against 0.294 — and a probe whose
    baseline moved between families would be comparing weightings rather than
    features.

    One ``.fit()``, on the train slice alone. Every learned parameter a family
    introduces — the median that fills its nulls, the scale it is divided by —
    is fitted inside the same Pipeline as the rest, so a family cannot leak
    validation statistics into the model that is meant to be judging it.

    Args:
        frame: Every non-gap row, carrying ``split`` and the extra columns.
        extra_numeric: The family's columns. Empty scores the bare baseline.
        delta_columns: From ``select_delta_columns``, measured on train.
        logistic_cfg: The ``baselines.logistic`` config block.

    Returns:
        A copy of ``frame`` with a ``score`` column.
    """
    features = [*FEATURE_COLUMNS, *extra_numeric]

    pipeline = build_pipeline(
        delta_columns, logistic_cfg, class_weight=None, extra_numeric=extra_numeric
    )
    train = frame[frame["split"] == "train"]
    pipeline.fit(train[features], train["isFraud"])

    scored = frame.copy()
    scored["score"] = pipeline.predict_proba(frame[features])[:, 1]
    return scored


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
    """Score every feature family through the Phase 02 harness, on VAL-FIT alone.

    Wiring only. Invoked by ``make families`` as
    ``python -m fraud_engine.features.evaluate``.

    VAL-CAL is not scored, and must not be. Choosing which features ship is
    tuning, and VAL-CAL is held back precisely so the calibrator and the
    decision threshold meet data no tuning decision has touched. ``write_run``
    would score it by default, so every call here names its splits instead.

    Args:
        config_path: Path to ``config.yaml``. Defaults to a repo-root-relative
            location, which is where the Makefile runs from.
    """
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    config = load_config(config_path)
    paths = config["paths"]
    logistic_cfg = config["baselines"]["logistic"]
    evaluation_cfg = config["features"]["evaluation"]
    capacities = load_capacities(load_config(Path(paths["cost_matrix"])))

    # Only what the probe and the families need. The matrices are 438 columns
    # wide, and every column read is carried through every fit in this run.
    family_columns = [column for columns in FAMILIES.values() for column in columns]
    columns = list(dict.fromkeys([*SOURCE_COLUMNS, *family_columns]))

    frame = prepare(load_matrices(paths["features_dir"], columns))
    train = frame[frame["split"] == "train"]
    delta_columns = select_delta_columns(train, logistic_cfg["d_max_null_frac"])

    pr_auc = {}
    for name, extra_numeric in FAMILIES.items():
        scored = score_with(frame, extra_numeric, delta_columns, logistic_cfg)
        metrics_path, _ = write_run(
            f"family_{name}",
            scored,
            capacities,
            paths["metrics_dir"],
            paths["predictions_dir"],
            splits=("val_fit",),
        )
        pr_auc[name] = evaluate_splits(scored, capacities, ("val_fit",))["val_fit"]["pr_auc"]
        log.info("family %-14s val_fit pr_auc=%.5f -> %s", name, pr_auc[name], metrics_path)

    # "none" is the bare probe: every delta below, and the noise floor itself,
    # is measured against it.
    for name, value in pr_auc.items():
        if name != "none":
            log.info("family %-14s delta vs none: %+.5f", name, value - pr_auc["none"])

    floor = pd.concat(
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
    floor["delta"] = floor["pr_auc"] - pr_auc["none"]

    floor_path = Path(paths["noise_floor"])
    floor_path.parent.mkdir(parents=True, exist_ok=True)
    floor.to_csv(floor_path, index=False)

    for width, measured in floor.groupby("n_columns"):
        log.info(
            "noise floor width=%d over %d seeds: mean %+.5f, max %+.5f",
            width,
            len(measured),
            measured["delta"].mean(),
            measured["delta"].max(),
        )
    log.info("wrote %s", floor_path)


if __name__ == "__main__":
    main()
