"""The rehearsal: the four rows of decision-policy.md §4, costed on VAL-CAL.

Run before the test touch so that a model which does not beat the rules engine is
found out on validation. Nothing may change because of what this shows (§4).

Probabilities are the out-of-fold ones `calibrate` wrote, never the shipped
calibrator applied to the rows it was fitted on — that would grade the calibrator
on its own training data and make this rehearsal optimistic where it should warn.
"""

from __future__ import annotations

import json
import logging
from datetime import UTC, datetime
from pathlib import Path

import mlflow
import numpy as np
import pandas as pd

from fraud_engine.data.load import DEFAULT_CONFIG_PATH, load_config
from fraud_engine.evaluation.cost import (
    Costs,
    ev_policy,
    load_costs,
    naive_policy,
    rules_policy,
)
from fraud_engine.evaluation.report import git_revision, load_operating_capacity
from fraud_engine.evaluation.tracking import configure_tracking, tracked_run
from fraud_engine.models.calibrate import OUT_OF_FOLD, SPLIT
from fraud_engine.models.train import LABEL

log = logging.getLogger(__name__)

RULES_RUN = "rules_baseline"
NAME = "policy_val_cal"
REFERENCE = "rules"
AMOUNT = "TransactionAmt"


def load_rehearsal_frame(
    predictions_dir: Path | str, interim_path: Path | str, method: str
) -> pd.DataFrame:
    """One row per VAL-CAL transaction: label, day, amount, and every score a row needs.

    Returns:
        Columns `TransactionID`, `day`, `isFraud`, `amount`, `uncalibrated`,
        `calibrated` (the out-of-fold probability of `method`) and `rules_score`.

    Raises:
        ValueError: If the rules run and the calibration output do not cover the same
            transactions with the same labels, or an amount is missing.
    """
    predictions_dir = Path(predictions_dir)

    oof = pd.read_parquet(predictions_dir / f"{OUT_OF_FOLD}.parquet")
    frame = oof[["TransactionID", "day", LABEL]].assign(
        uncalibrated=oof["score"], calibrated=oof[method]
    )

    rules = pd.read_parquet(predictions_dir / f"{RULES_RUN}.parquet")
    rules = rules.loc[rules["split"] == SPLIT, ["TransactionID", LABEL, "score"]]

    if set(rules["TransactionID"]) != set(frame["TransactionID"]):
        raise ValueError(f"{RULES_RUN} and {OUT_OF_FOLD} do not cover the same {SPLIT} rows")

    frame = frame.merge(
        rules.rename(columns={"score": "rules_score", LABEL: "rules_label"}),
        on="TransactionID",
        validate="one_to_one",
    )
    if (frame["rules_label"] != frame[LABEL]).any():
        raise ValueError(f"{RULES_RUN} and {OUT_OF_FOLD} disagree on labels")

    amounts = pd.read_parquet(interim_path, columns=["TransactionID", AMOUNT])
    frame = frame.merge(amounts, on="TransactionID", how="left", validate="one_to_one")
    if frame[AMOUNT].isna().any():
        raise ValueError(f"{int(frame[AMOUNT].isna().sum())} {SPLIT} rows have no {AMOUNT}")

    return frame.drop(columns="rules_label").rename(columns={AMOUNT: "amount"})


def rehearse(frame: pd.DataFrame, costs: Costs, capacity: float) -> dict[str, dict]:
    """Each §4 row's summary, plus its reduction in USD against the rules engine.

    Returns:
        `{row: {usd_per_1000, reviews_per_day, block_rate, reduction_vs_rules}}`,
        in the order §4 lists them. The rules row's reduction is 0 by definition.
    """
    y = frame[LABEL].to_numpy()
    amount = frame["amount"].to_numpy(dtype="float64")
    day = frame["day"].to_numpy()

    decisions = {
        REFERENCE: rules_policy(frame["rules_score"].to_numpy(), day, capacity),
        "naive": naive_policy(frame["calibrated"].to_numpy()),
        "ev_uncalibrated": ev_policy(
            frame["uncalibrated"].to_numpy(), amount, day, costs, capacity
        ),
        "ev": ev_policy(frame["calibrated"].to_numpy(), amount, day, costs, capacity),
    }

    rows = {name: decision.summary(y, amount, day, costs) for name, decision in decisions.items()}

    reference = rows[REFERENCE]["usd_per_1000"]
    for summary in rows.values():
        summary["reduction_vs_rules"] = float(1 - summary["usd_per_1000"] / reference)

    return rows


def main(config_path: Path = DEFAULT_CONFIG_PATH) -> None:
    """Cost the four rows on VAL-CAL and write the rehearsal record.

    Wiring only. Invoked by `make rehearsal` as `python -m fraud_engine.evaluation.policy`.

    Args:
        config_path: Path to `config.yaml`.
    """
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    config = load_config(config_path)
    paths = config["paths"]

    configure_tracking(config["tracking"])

    cost_matrix = load_config(Path(paths["cost_matrix"]))
    costs = load_costs(cost_matrix)
    capacity = load_operating_capacity(cost_matrix)
    method = json.loads(Path(paths["calibration"]).read_text())["selected"]

    frame = load_rehearsal_frame(paths["predictions_dir"], paths["interim"], method)

    params = {
        "split": SPLIT,
        "calibration": method,
        "review_capacity": capacity,
        "cost_matrix_version": costs.version,
    }

    with tracked_run(NAME, params, config_path):
        rows = rehearse(frame, costs, capacity)
        mlflow.log_metrics(
            {
                f"{row}.{metric}": value
                for row, summary in rows.items()
                for metric, value in summary.items()
            }
        )

    for row, summary in rows.items():
        log.info(
            "%-16s $%9.2f per 1,000   %5.1f reviews/day   block %.4f   vs rules %+.1f%%",
            row,
            summary["usd_per_1000"],
            summary["reviews_per_day"],
            summary["block_rate"],
            100 * summary["reduction_vs_rules"],
        )

    record = {
        "name": NAME,
        "created": datetime.now(UTC).isoformat(timespec="seconds"),
        "git_revision": git_revision(),
        "split": SPLIT,
        "rows": len(frame),
        "positives": int(frame[LABEL].sum()),
        "days": int(np.unique(frame["day"]).size),
        "probabilities": f"{method}, out-of-fold",
        "review_capacity": capacity,
        "cost_matrix_version": costs.version,
        "costs": {name: value for name, value in vars(costs).items() if name != "version"},
        "policies": rows,
    }
    path = Path(paths["rehearsal"])
    path.write_text(json.dumps(record, indent=2) + "\n")
    log.info("wrote %s", path)


if __name__ == "__main__":
    main()
