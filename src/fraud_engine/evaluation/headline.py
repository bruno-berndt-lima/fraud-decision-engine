"""The test touch: the headline, measured once, with everything it depends on frozen.

Registered in `docs/decision-policy.md` §7. The model, the calibrator, the costs, the
capacity and the policy are the ones already committed; nothing here fits, tunes or
chooses. The test number is a measurement, not a target.

**Once.** The stage refuses a dirty tree, and refuses to run if the record already
exists: a second touch would mean deleting a tracked file, which the history shows.

**Proven before it is used.** The reloaded model must reproduce its recorded VAL-CAL
scores exactly, and the refitted rules engine its own, before either scores a test row.
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
from fraud_engine.evaluation.cost import ALLOW, Decision, load_costs
from fraud_engine.evaluation.plots import plot_reliability, save_figure
from fraud_engine.evaluation.policy import AMOUNT, REFERENCE, RULES_RUN, rehearse
from fraud_engine.evaluation.report import (
    evaluate_splits,
    git_revision,
    load_capacities,
    load_operating_capacity,
)
from fraud_engine.evaluation.reproduce import check_reproduces
from fraud_engine.evaluation.tracking import configure_tracking, tracked_run
from fraud_engine.models import rules
from fraud_engine.models.calibrate import (
    apply_calibrator,
    calibration_metrics,
    reliability_bins,
)
from fraud_engine.models.train import (
    LABEL,
    feature_columns,
    prepare_matrices,
    run_name,
    score,
)

log = logging.getLogger(__name__)

NAME = "policy_test"
SPLIT = "test"
PROOF_SPLIT = "val_cal"
ALLOW_EVERYTHING = "allow_everything"
PREDICTIONS = "headline_test"
FIGURE = "reliability_test.png"


def check_single_touch(record_path: Path, revision: str | None) -> None:
    """Refuse a touch that could not be reproduced from history, or a second one.

    Raises:
        RuntimeError: Outside git, on a dirty tree, or if the record already exists.
    """
    if revision is None:
        raise RuntimeError("not in a git checkout; the test touch must name the commit it ran from")
    if revision.endswith("-dirty"):
        raise RuntimeError(f"the tree is dirty ({revision}); commit before touching test")
    if record_path.exists():
        raise RuntimeError(
            f"{record_path} exists: test has been touched. A second touch is not a rerun; "
            "if one is truly needed, delete the record in a commit that says why."
        )


def score_model(features_dir: Path | str, model_path: Path | str, model_cfg: dict) -> pd.DataFrame:
    """The shipped booster, reloaded from disk, scoring VAL-CAL and test.

    Matrices are prepared from train exactly as `make train` prepared them; the VAL-CAL
    proof in `main` is what shows that path still yields the model's inputs.

    Raises:
        ValueError: If the booster's features are not the matrices' features, in order.
    """
    matrices, _, _ = prepare_matrices(features_dir, model_cfg, ("train", PROOF_SPLIT, SPLIT))
    booster = lgb.Booster(model_file=str(model_path))

    columns = booster.feature_name()
    if columns != feature_columns(matrices[SPLIT]):
        raise ValueError("the booster's features are not the test matrix's features, in order")

    return score(booster, {s: matrices[s] for s in (PROOF_SPLIT, SPLIT)}, columns)


def score_rules(interim_path: Path | str, splits_path: Path | str, rules_cfg: dict) -> pd.DataFrame:
    """The rules engine as `rules.main` builds it — constants from train — on VAL-CAL and test."""
    frame = pd.read_parquet(interim_path, columns=list(rules.INTERIM_COLUMNS))
    assignment = pd.read_parquet(splits_path, columns=["TransactionID", "split"])
    frame = frame.merge(assignment, on="TransactionID", how="inner", validate="one_to_one")

    engine = rules.build_rules(rules_cfg)
    constants = rules.fit(frame[frame["split"] == "train"], rules_cfg)

    frame = frame[frame["split"].isin((PROOF_SPLIT, SPLIT))].copy()
    frame["score"] = rules.score(frame, engine, constants)
    return frame


def build_test_frame(model: pd.DataFrame, engine: pd.DataFrame, calibrator: dict) -> pd.DataFrame:
    """One row per test transaction: label, day, amount, and every score a row needs.

    Returns:
        Columns `TransactionID`, `day`, `isFraud`, `amount`, `uncalibrated`,
        `calibrated` and `rules_score`, the shape `policy.rehearse` reads.

    Raises:
        ValueError: If the model and the rules engine scored different test rows or
            disagree on a label.
    """
    model = model[model["split"] == SPLIT]
    engine = engine[engine["split"] == SPLIT]

    if set(model["TransactionID"]) != set(engine["TransactionID"]):
        raise ValueError("the model and the rules engine scored different test rows")

    frame = model[["TransactionID", "day", LABEL, "score"]].merge(
        engine[["TransactionID", LABEL, "score", AMOUNT]].rename(
            columns={LABEL: "rules_label", "score": "rules_score"}
        ),
        on="TransactionID",
        validate="one_to_one",
    )
    if (frame["rules_label"] != frame[LABEL]).any():
        raise ValueError("the model and the rules engine disagree on test labels")

    raw = frame["score"].to_numpy(dtype="float64")
    return (
        frame.drop(columns=["rules_label", "score"])
        .assign(uncalibrated=raw, calibrated=apply_calibrator(calibrator, raw))
        .rename(columns={AMOUNT: "amount"})
        .reset_index(drop=True)
    )


def headline_rows(frame: pd.DataFrame, costs, capacity: float) -> tuple[dict, dict]:
    """The four §4 rows, and allowing everything as a reference beside them.

    Returns:
        `(policies, references)`, each `{row: summary with reduction_vs_rules}`.
    """
    policies = rehearse(frame, costs, capacity)

    y = frame[LABEL].to_numpy()
    amount = frame["amount"].to_numpy(dtype="float64")
    day = frame["day"].to_numpy()
    allow = Decision(np.full(len(frame), ALLOW), np.zeros(len(frame))).summary(
        y, amount, day, costs
    )
    allow["reduction_vs_rules"] = float(
        1 - allow["usd_per_1000"] / policies[REFERENCE]["usd_per_1000"]
    )

    return policies, {ALLOW_EVERYTHING: allow}


def main(config_path: Path = DEFAULT_CONFIG_PATH) -> None:
    """Prove, score, cost and record the test touch.

    Wiring only. Invoked by `make headline` as `python -m fraud_engine.evaluation.headline`.

    Args:
        config_path: Path to `config.yaml`.
    """
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    config = load_config(config_path)
    paths, model_cfg = config["paths"], config["model"]
    record_path = Path(paths["headline"])

    revision = git_revision()
    check_single_touch(record_path, revision)

    configure_tracking(config["tracking"])

    cost_matrix = load_config(Path(paths["cost_matrix"]))
    costs = load_costs(cost_matrix)
    capacity = load_operating_capacity(cost_matrix)
    capacities = load_capacities(cost_matrix)
    bins = config["calibration"]["ece_bins"]

    model_name = run_name(model_cfg)
    calibrator = json.loads(Path(paths["calibrator"]).read_text())
    if calibrator.get("model") != model_name or calibrator.get("fitted_on") != PROOF_SPLIT:
        raise ValueError(
            f"calibrator was fitted for {calibrator.get('model')!r} on "
            f"{calibrator.get('fitted_on')!r}, not {model_name!r} on {PROOF_SPLIT!r}"
        )

    predictions_dir = Path(paths["predictions_dir"])
    model = score_model(paths["features_dir"], paths["model"], model_cfg)
    check_reproduces(
        model, pd.read_parquet(predictions_dir / f"{model_name}.parquet"), model_name, PROOF_SPLIT
    )

    engine = score_rules(paths["interim"], paths["splits"], config["baselines"]["rules"])
    check_reproduces(
        engine, pd.read_parquet(predictions_dir / f"{RULES_RUN}.parquet"), RULES_RUN, PROOF_SPLIT
    )
    log.info(
        "proofs passed: %s and %s reproduce their %s records", model_name, RULES_RUN, PROOF_SPLIT
    )

    frame = build_test_frame(model, engine, calibrator)
    y = frame[LABEL].to_numpy()

    params = {
        "split": SPLIT,
        "model": model_name,
        "calibration": calibrator["method"],
        "review_capacity": capacity,
        "cost_matrix_version": costs.version,
    }

    with tracked_run(NAME, params, config_path):
        policies, references = headline_rows(frame, costs, capacity)
        calibration = {
            "uncalibrated": calibration_metrics(y, frame["uncalibrated"].to_numpy(), bins),
            "calibrated": calibration_metrics(y, frame["calibrated"].to_numpy(), bins),
        }
        measured = {
            name: evaluate_splits(
                frame.assign(split=SPLIT, score=frame[column]), capacities, (SPLIT,)
            )[SPLIT]
            for name, column in (("model", "uncalibrated"), ("rules", "rules_score"))
        }
        mlflow.log_metrics(
            {
                f"{row}.{metric}": value
                for row, summary in {**policies, **references}.items()
                for metric, value in summary.items()
            }
            | {f"calibration.{k}.{m}": v for k, ms in calibration.items() for m, v in ms.items()}
            | {f"{name}.pr_auc": block["pr_auc"] for name, block in measured.items()}
        )

    for row, summary in {**policies, **references}.items():
        log.info(
            "%-16s $%9.2f per 1,000   %5.1f reviews/day   block %.4f   vs rules %+.1f%%",
            row,
            summary["usd_per_1000"],
            summary["reviews_per_day"],
            summary["block_rate"],
            100 * summary["reduction_vs_rules"],
        )

    predictions_path = predictions_dir / f"{PREDICTIONS}.parquet"
    frame.to_parquet(predictions_path, index=False)

    stored = pd.read_parquet(predictions_path)
    tables = {
        "uncalibrated": reliability_bins(stored[LABEL], stored["uncalibrated"], bins),
        f"{calibrator['method']}, fitted on VAL-CAL": reliability_bins(
            stored[LABEL], stored["calibrated"], bins
        ),
    }
    figure_path = save_figure(
        plot_reliability(tables, title="Reliability — TEST"), Path(paths["figures_dir"]) / FIGURE
    )

    record = {
        "name": NAME,
        "created": datetime.now(UTC).isoformat(timespec="seconds"),
        "git_revision": revision,
        "split": SPLIT,
        "rows": len(frame),
        "positives": int(y.sum()),
        "days": int(frame["day"].nunique()),
        "frozen": {
            "model": model_name,
            "calibrator": {
                k: calibrator[k] for k in ("method", "a", "b", "git_revision") if k in calibrator
            },
            "cost_matrix_version": costs.version,
            "costs": {k: v for k, v in vars(costs).items() if k != "version"},
            "review_capacity": capacity,
        },
        "proofs": {
            f"{model_name} reproduces {PROOF_SPLIT}": True,
            f"{RULES_RUN} reproduces {PROOF_SPLIT}": True,
        },
        "policies": policies,
        "references": references,
        "calibration": calibration,
        "measured": measured,
    }
    record_path.write_text(json.dumps(record, indent=2) + "\n")

    for path in (record_path, figure_path, predictions_path):
        log.info("wrote %s", path)


if __name__ == "__main__":
    main()
