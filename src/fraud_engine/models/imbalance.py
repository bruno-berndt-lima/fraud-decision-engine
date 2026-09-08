"""E2's second run — what class imbalance handling buys a tree ensemble.

Its own command rather than part of ``train``, for the reason ``seeds.py`` is:
the shipped model is one arm of this, and rerunning four others every time the
model is retrained would pay for an answer that does not change.

E2 registers the arms and the objections. The short version: at this base rate
the reflex is to resample, and this project does not resample by reflex — it
runs the comparison and reports what happened, including when the reflex loses.

Every arm is deterministic. The untuned configuration samples neither rows nor
columns, so each is one number rather than a distribution and E6's bar does not
apply.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import pandas as pd
from imblearn.over_sampling import SMOTENC

from fraud_engine.data.load import DEFAULT_CONFIG_PATH, load_config
from fraud_engine.evaluation.report import load_capacities, write_run
from fraud_engine.models.train import (
    LABEL,
    apply_categories,
    apply_medians,
    feature_columns,
    fit,
    fit_categories,
    fit_medians,
    load_split_matrices,
    score,
    to_dataset,
)

log = logging.getLogger(__name__)

# Arms that only change parameters, and the value each sets. `None` marks the
# one that changes nothing — the untuned reference, rerun here so the comparison
# reads off one table rather than two files written on different days.
WEIGHT_ARMS = ("none", "scale_pos_weight", "is_unbalance")

# Arms that change the training data instead. Both impute, so `imputed` is what
# isolates the resampling from the imputation it requires.
IMPUTED_ARMS = ("imputed", "smote")


def resample(train: pd.DataFrame, columns: list[str], seed: int) -> pd.DataFrame:
    """Oversample the positive class to parity with SMOTE-NC.

    Parity rather than a gentler ratio because parity is the reflex under test.
    ``SMOTENC`` rather than ``SMOTE`` because a third of these columns are
    categorical and there is no midpoint between two card brands; it takes the
    majority level among the neighbours instead, which is the least bad answer
    to a question that should not have been asked of a category.

    Only training rows are resampled. Synthesising validation rows would be
    scoring the model on transactions that never happened.

    Args:
        train: Prepared, imputed training rows.
        columns: Feature names.
        seed: Fixed, so the synthetic rows are the same on every rerun.

    Returns:
        The training frame with synthetic positives appended, carrying only
        ``columns`` and the label — a synthetic row has no transaction id and no
        day, and inventing them would make it look like a transaction.
    """
    categorical = [
        index
        for index, column in enumerate(columns)
        if isinstance(train[column].dtype, pd.CategoricalDtype)
    ]

    features, labels = SMOTENC(categorical_features=categorical, random_state=seed).fit_resample(
        train[columns], train[LABEL]
    )
    # concat rather than assign: inserting one column into a 349-column frame
    # copies it block by block and pandas warns about the fragmentation.
    return pd.concat([features, labels.rename(LABEL)], axis=1)


def arm_params(arm: str, ratio: float) -> dict:
    """The parameters an arm adds, on top of the untuned reference.

    ``scale_pos_weight`` takes the measured ``neg/pos`` rather than a config
    value: "balanced" is a fact about the training window, not a preference, and
    a number typed into config would silently stop being balanced when the window
    moved. ``is_unbalance`` asks LightGBM to derive the same ratio itself, which
    is why the two are expected to land on the same result.

    The data arms add nothing here — what they change is upstream of the model.
    """
    if arm == "scale_pos_weight":
        return {"scale_pos_weight": ratio}
    if arm == "is_unbalance":
        return {"is_unbalance": True}
    return {}


def measure(
    matrices: dict[str, pd.DataFrame],
    columns: list[str],
    model_cfg: dict,
    capacities: list[float],
    paths: dict,
) -> pd.DataFrame:
    """Fit and score every arm, writing each through the Phase 02 harness.

    One record per arm, so the comparison is a read across
    ``reports/metrics/imbalance_*.json`` rather than a table assembled by hand.

    Args:
        matrices: Prepared matrices, as ``train.py`` produces them.
        columns: Feature names.
        model_cfg: The ``model:`` config block.
        capacities: Review capacities.
        paths: The ``paths:`` config block.

    Returns:
        One row per arm: ``arm``, ``pr_auc``, ``best_iteration``, ``train_rows``.
    """
    positives = int(matrices["train"][LABEL].sum())
    ratio = (len(matrices["train"]) - positives) / positives
    log.info(
        "train %d rows, %d positive — neg/pos = %.2f", len(matrices["train"]), positives, ratio
    )

    medians = fit_medians(matrices["train"], columns)
    imputed = {split: apply_medians(frame, medians) for split, frame in matrices.items()}

    measured = []

    for arm in (*WEIGHT_ARMS, *IMPUTED_ARMS):
        source = imputed if arm in IMPUTED_ARMS else matrices
        train_frame = source["train"]

        if arm == "smote":
            train_frame = resample(train_frame, columns, model_cfg["seed"])

        train = to_dataset(train_frame, columns)
        val_fit = to_dataset(source["val_fit"], columns, reference=train)

        booster = fit(train, val_fit, {**model_cfg, "tuned": arm_params(arm, ratio)})

        scored = score(booster, {split: source[split] for split in ("val_fit", "val_cal")}, columns)
        metrics_path, _ = write_run(
            f"imbalance_{arm}", scored, capacities, paths["metrics_dir"], paths["predictions_dir"]
        )

        pr_auc = json.loads(Path(metrics_path).read_text())["splits"]["val_fit"]["pr_auc"]
        measured.append(
            {
                "arm": arm,
                "pr_auc": pr_auc,
                "best_iteration": booster.best_iteration,
                "train_rows": len(train_frame),
            }
        )
        log.info("arm=%-17s val_fit pr_auc=%.5f -> %s", arm, pr_auc, metrics_path)

    return pd.DataFrame(measured)


def main(config_path: Path = DEFAULT_CONFIG_PATH) -> None:
    """Run every arm and write the comparison.

    Wiring only. Invoked by ``make imbalance`` as
    ``python -m fraud_engine.models.imbalance``.
    """
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    config = load_config(config_path)
    paths, model_cfg = config["paths"], config["model"]

    matrices = load_split_matrices(paths["features_dir"], ("train", "val_fit", "val_cal"))
    vocabulary = fit_categories(matrices["train"], model_cfg["min_category_rows"])
    matrices = {split: apply_categories(frame, vocabulary) for split, frame in matrices.items()}

    columns = feature_columns(matrices["train"])
    capacities = load_capacities(load_config(Path(paths["cost_matrix"])))

    comparison = measure(matrices, columns, model_cfg, capacities, paths)

    path = Path(paths["imbalance"])
    path.parent.mkdir(parents=True, exist_ok=True)
    comparison.to_csv(path, index=False)

    log.info("\n%s", comparison.to_string(index=False))
    log.info("wrote %s", path)


if __name__ == "__main__":
    main()
