"""The unlabelled horizon — days 213 to 395, built and scored as the service would.

`docs/monitoring.md` §6 and §8. Kaggle's test files carry no `isFraud` and never will,
which makes them the honest stand-in for what a live monitor sees: real transactions,
scored by the shipped model, whose outcomes are not known yet.

**Built by the service's own transform, in batch.** Every family is applied with the
tables already shipped, so nothing is refitted on the rows being watched — a builder
that fitted on them would encode the drift it exists to measure. Tier 3 is filled as
the service fills it: the file starts thirty days after the labelled data, with no card
history behind it.

**Proven before it measures.** Test's labelled rows go through the same path first,
carrying the history the training matrix built for them, and must reproduce that matrix
on every column and the single touch's persisted scores to the bit. Only then is a
horizon row read.

**Two outputs.** The matrix §3's PSI reads — vocabulary applied, medians not, because a
fill would erase the null-rate drift PSI is there to see — and the scores §6 reads.
"""

from __future__ import annotations

import json
import logging
from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd

from fraud_engine.data.load import (
    DEFAULT_CONFIG_PATH,
    add_time_columns,
    join_identity,
    load_config,
    load_typed_csv,
)
from fraud_engine.evaluation.report import git_revision
from fraud_engine.features import velocity
from fraud_engine.models.calibrate import apply_calibrator
from fraud_engine.models.train import apply_categories, apply_medians, run_name
from fraud_engine.serving.artifacts import Model, Tables, load_model
from fraud_engine.serving.transform import raw_inputs, transform

log = logging.getLogger(__name__)

FEATURES = "features.parquet"
SCORES = "scores.parquet"
MANIFEST = "build.json"
PROOF_SPLIT = "test"
HEADLINE_PREDICTIONS = "headline_test"
# `test_identity.csv` spells the identity block `id-01`; the training pair and the
# model spell it `id_01`.
RAW_IDENTITY_PREFIX = "id-"
IDENTITY_PREFIX = "id_"
KEEP = ("TransactionID", "day", "TransactionAmt", "has_identity")


def identity_names(columns: pd.Index) -> dict[str, str]:
    """The rename that gives the test file's identity block the model's spelling."""
    return {
        name: IDENTITY_PREFIX + name[len(RAW_IDENTITY_PREFIX) :]
        for name in columns
        if name.startswith(RAW_IDENTITY_PREFIX)
    }


def load_horizon(
    transactions_path: Path | str, identity_path: Path | str, load_cfg: dict
) -> pd.DataFrame:
    """The unlabelled pair, typed, joined and dated as `data/load.py` treats the labelled one.

    Raises:
        ValueError: If the join changed the row count, or per `join_identity`, which
            refuses a pair of files that do not belong together.
    """
    transactions = load_typed_csv(Path(transactions_path), load_cfg)
    identity = load_typed_csv(Path(identity_path), load_cfg)
    identity = identity.rename(columns=identity_names(identity.columns))

    frame = add_time_columns(join_identity(transactions, identity))
    if len(frame) != len(transactions):
        raise ValueError(
            f"the join changed the row count: {len(transactions)} in, {len(frame)} out"
        )
    return frame


def check_inputs(frame: pd.DataFrame, columns: list[str]) -> None:
    """Refuse a horizon that lacks any input the model reads.

    The transform treats an omitted field as a null, which is right for a request and
    wrong for a file: a whole column missing from the horizon would be scored as absent
    on every row and read by PSI as a feed gone silent. The identity block misspelled
    is the case this exists for, and nothing downstream would look wrong.

    Raises:
        ValueError: If an input is absent, or a column still carries the test file's
            spelling.
    """
    stray = [name for name in frame.columns if name.startswith(RAW_IDENTITY_PREFIX)]
    if stray:
        raise ValueError(
            f"{len(stray)} columns still spelled as the test file spells them: {stray[:3]}"
        )

    absent = [name for name in raw_inputs(columns) if name not in frame.columns]
    if absent:
        raise ValueError(
            f"{len(absent)} inputs the model reads are absent from the horizon: {absent[:5]}; "
            "the transform would score them as nulls on every row"
        )


def build(
    raw: pd.DataFrame, tables: Tables, load_cfg: dict, features_cfg: dict, columns: list[str]
) -> pd.DataFrame:
    """The matrix PSI reads: the service's transform, with the medians left out.

    The transform skips the fill when a model ships without medians, so turning them off
    here yields the same matrix one step earlier rather than a second implementation of
    it. Rows keep the input's index.
    """
    return transform(raw, replace(tables, medians=None), load_cfg, features_cfg, columns)


def score(unfilled: pd.DataFrame, model: Model) -> tuple[np.ndarray, np.ndarray]:
    """The booster's score and the calibrated probability, as the headline produced them.

    Returns:
        `(uncalibrated, calibrated)`, one of each per row.
    """
    matrix = unfilled
    if model.tables.medians is not None:
        matrix = apply_medians(unfilled, model.tables.medians)

    booster = model.booster
    raw = booster.predict(
        matrix[model.columns], num_iteration=booster.best_iteration, num_threads=model.threads
    )
    return raw, apply_calibrator(model.calibrator, raw)


def check_matrix(unfilled: pd.DataFrame, expected: pd.DataFrame, tables: Tables) -> None:
    """Refuse a build that differs from the training matrix in any column, value or dtype.

    Raises:
        ValueError: With the first difference `pandas` finds.
    """
    reference = apply_categories(expected[list(unfilled.columns)], tables.vocabulary)
    try:
        pd.testing.assert_frame_equal(
            unfilled.reset_index(drop=True),
            reference.reset_index(drop=True),
            check_exact=True,
            check_dtype=True,
            check_categorical=True,
        )
    except AssertionError as difference:
        raise ValueError(
            f"the horizon build does not reproduce the training matrix: {difference}"
        ) from difference


def check_scores(uncalibrated: np.ndarray, calibrated: np.ndarray, persisted: pd.DataFrame) -> None:
    """Refuse scores that differ at all from the ones the single touch persisted.

    Raises:
        ValueError: If either vector differs.
    """
    for name, values in (("uncalibrated", uncalibrated), ("calibrated", calibrated)):
        if not np.array_equal(values, persisted[name].to_numpy()):
            worst = float(np.max(np.abs(values - persisted[name].to_numpy())))
            raise ValueError(f"{name} scores differ from the headline's, by up to {worst:.3g}")


def prove(model: Model, paths: dict, load_cfg: dict, features_cfg: dict) -> dict:
    """Test's labelled rows through the horizon's own path, against what is on record.

    The rows carry the tier-3 history the training matrix built for them, which the
    transform keeps, so the whole matrix is comparable. What this proves is the path;
    the horizon itself is scored with the service's defaults, which nothing built from
    history could be compared against.

    Returns:
        What was proven, for the manifest.

    Raises:
        FileNotFoundError: If test was never touched.
        ValueError: Per `check_matrix` and `check_scores`.
    """
    headline = Path(paths["predictions_dir"]) / f"{HEADLINE_PREDICTIONS}.parquet"
    if not headline.exists():
        raise FileNotFoundError(f"{headline} is absent: `make headline` is the one test touch")

    expected = pd.read_parquet(Path(paths["features_dir"]) / f"{PROOF_SPLIT}.parquet")
    identifiers = expected["TransactionID"].tolist()

    raw = pd.read_parquet(paths["interim"], filters=[("TransactionID", "in", identifiers)])
    raw = raw.set_index("TransactionID").loc[identifiers].reset_index()
    raw = raw.merge(
        expected[["TransactionID", *velocity.COLUMNS]],
        on="TransactionID",
        how="left",
        validate="one_to_one",
    )

    unfilled = build(raw, model.tables, load_cfg, features_cfg, model.columns)
    check_matrix(unfilled, expected, model.tables)

    persisted = pd.read_parquet(headline).set_index("TransactionID").loc[identifiers]
    check_scores(*score(unfilled, model), persisted)

    return {
        "split": PROOF_SPLIT,
        "rows": len(raw),
        "matrix reproduces data/features": True,
        "scores reproduce the headline's": True,
    }


def main(config_path: Path = DEFAULT_CONFIG_PATH) -> None:
    """Prove the path, then build and score the horizon.

    Wiring only. Invoked by `make horizon` as `python -m fraud_engine.monitoring.horizon`.

    Args:
        config_path: Path to `config.yaml`.
    """
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    config = load_config(config_path)
    paths, load_cfg, features_cfg = config["paths"], config["load"], config["features"]
    model = load_model(paths, config["model"]["impute"], config["horizon"]["threads"])

    proof = prove(model, paths, load_cfg, features_cfg)
    log.info(
        "proof passed: %d test rows reproduce the matrix and the headline's scores", proof["rows"]
    )

    raw = load_horizon(
        paths["horizon_raw"]["transactions"], paths["horizon_raw"]["identity"], load_cfg
    )
    check_inputs(raw, model.columns)
    unfilled = build(raw, model.tables, load_cfg, features_cfg, model.columns)
    uncalibrated, calibrated = score(unfilled, model)

    directory = Path(paths["horizon_dir"])
    directory.mkdir(parents=True, exist_ok=True)
    pd.concat([raw[["TransactionID", "day"]], unfilled], axis=1).to_parquet(
        directory / FEATURES, index=False
    )

    manifest = {
        "created": datetime.now(UTC).isoformat(timespec="seconds"),
        "git_revision": git_revision(),
        "model": run_name(config["model"]),
        "calibrator": model.calibrator["method"],
        "rows": len(raw),
        "first_day": int(raw["day"].min()),
        "last_day": int(raw["day"].max()),
        "with_identity": int(raw["has_identity"].sum()),
        "tier_3": "filled as the service fills it (serving/transform.fill_history)",
        "medians_in_features": False,
        "proof": proof,
    }
    (directory / MANIFEST).write_text(json.dumps(manifest, indent=2) + "\n")

    # Written last: it is the stage's sentinel, so its presence means everything else is.
    scores = raw[list(KEEP)].assign(uncalibrated=uncalibrated, calibrated=calibrated)
    scores.to_parquet(directory / SCORES, index=False)

    log.info(
        "horizon: %d rows, days %d-%d, %.1f%% with identity; wrote %s",
        len(raw),
        manifest["first_day"],
        manifest["last_day"],
        100 * raw["has_identity"].mean(),
        directory,
    )


if __name__ == "__main__":
    main()
