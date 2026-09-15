"""The USD halves E1 and E3 owe: each arm refitted, calibrated and costed on VAL-CAL.

Registered in `docs/decision-policy.md` §6. Nothing here ships and nothing may change
because of it; measuring a field of candidates on VAL-CAL is allowed only on that
condition.

**An arm is proven to be Phase 05's arm before it is costed.** Both experiments ran on
the untuned configuration, which samples nothing, so a refit returns identical digits.
Its VAL-FIT scores must equal the predictions Phase 05 recorded, or the stage stops:
a USD figure for a different fit would be attached to a PR-AUC it never earned.

VAL-CAL is scored here and not by `ablation.fit_without`, which keeps the calibration
slice unreachable from a Phase 05 arm by construction.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

import mlflow
import numpy as np
import pandas as pd
import yaml

from fraud_engine.data.load import DEFAULT_CONFIG_PATH, load_config
from fraud_engine.evaluation.cost import Costs, ev_policy, load_costs, rules_policy
from fraud_engine.evaluation.policy import load_rehearsal_frame
from fraud_engine.evaluation.report import git_revision, load_operating_capacity
from fraud_engine.evaluation.tracking import configure_tracking, tracked_child, tracked_run
from fraud_engine.models import purge
from fraud_engine.models.ablation import REFERENCE as ABLATION_REFERENCE
from fraud_engine.models.ablation import drop_columns, resolve_arms
from fraud_engine.models.calibrate import assign_folds, check_clip, cross_fit
from fraud_engine.models.train import (
    LABEL,
    feature_columns,
    fit,
    prepare_matrices,
    resolve_params,
    score,
    to_dataset,
)

log = logging.getLogger(__name__)

NAME = "usd_halves"
SPLITS = ("train", "val_fit", "val_cal")
E3_ARM = "velocity"


@dataclass(frozen=True)
class Arm:
    """Where an arm's matrices live, what it removes, and the record it must reproduce."""

    experiment: str
    name: str
    features_dir: Path
    dropped: tuple[str, ...]
    recorded: str


def resolve_experiment_arms(paths: dict) -> dict[str, list[Arm]]:
    """E1's three training windows and E3's two feature sets, reference first.

    E1's matrices are the ones `make purge` built, found through the config each arm
    wrote beside them. E3's removed columns come from `ablation.resolve_arms`, so they
    are the velocity family exactly as Phase 05 removed it.
    """
    e1 = []
    for name in purge.ARMS:
        arm_config = yaml.safe_load((Path(paths["purge_dir"]) / name / "config.yaml").read_text())
        e1.append(Arm("E1", name, Path(arm_config["paths"]["features_dir"]), (), f"purge_{name}"))

    ablation_arms = resolve_arms(paths["features_dir"])
    e3 = [
        Arm("E3", name, Path(paths["features_dir"]), ablation_arms[name], f"ablation_{name}")
        for name in (ABLATION_REFERENCE, E3_ARM)
    ]

    return {"E1": e1, "E3": e3}


def fit_arm(
    matrices: dict[str, pd.DataFrame], dropped: tuple[str, ...], model_cfg: dict
) -> tuple[pd.DataFrame, int]:
    """Fit as Phase 05 did — early-stopped on VAL-FIT — and score VAL-FIT and VAL-CAL."""
    frames = {split: drop_columns(matrices[split], dropped) for split in SPLITS}
    columns = feature_columns(frames["train"])

    train = to_dataset(frames["train"], columns)
    val_fit = to_dataset(frames["val_fit"], columns, reference=train)
    booster = fit(train, val_fit, model_cfg)

    scored = score(booster, {s: frames[s] for s in ("val_fit", "val_cal")}, columns)
    return scored, booster.best_iteration


def check_reproduces(scored: pd.DataFrame, recorded: pd.DataFrame, name: str) -> None:
    """Refuse a refit whose VAL-FIT scores differ from what Phase 05 recorded.

    Raises:
        ValueError: If the transactions differ or any score differs at all.
    """
    new = scored[scored["split"] == "val_fit"].set_index("TransactionID")["score"]
    old = recorded[recorded["split"] == "val_fit"].set_index("TransactionID")["score"]

    if set(new.index) != set(old.index):
        raise ValueError(f"{name}: the refit scored different VAL-FIT rows than the record")

    differ = int((new.loc[old.index] != old).sum())
    if differ:
        raise ValueError(
            f"{name}: {differ} VAL-FIT scores differ from the recorded run; this is not the "
            "arm Phase 05 measured"
        )


def paired_day_bootstrap(
    day: np.ndarray,
    arm_cost: np.ndarray,
    reference_cost: np.ndarray,
    resamples: int,
    interval: float,
    seed: int,
) -> tuple[float, float]:
    """Interval for the arm's USD per 1,000 minus the reference's, resampling whole days.

    Days, because review capacity is per day; paired, because both policies decide the
    same transactions, so a costly day raises both and cancels from the difference.
    It captures which days VAL-CAL happened to hold, not how a refit would move.

    Returns:
        `(low, high)`, the central `interval` of the resampled differences.
    """
    days, index = np.unique(np.asarray(day), return_inverse=True)
    rows = np.bincount(index)
    delta = np.bincount(index, weights=arm_cost) - np.bincount(index, weights=reference_cost)

    draws = np.random.default_rng(seed).integers(0, days.size, size=(resamples, days.size))
    resampled = delta[draws].sum(axis=1) / rows[draws].sum(axis=1) * 1_000

    tail = (1 - interval) / 2
    low, high = np.quantile(resampled, [tail, 1 - tail])
    return float(low), float(high)


def cost_arm(
    frame: pd.DataFrame,
    scored: pd.DataFrame,
    costs: Costs,
    capacity: float,
    calibration: dict,
    method: str,
) -> tuple[dict, np.ndarray]:
    """Calibrate an arm's VAL-CAL scores out-of-fold and cost them under the EV policy.

    Args:
        frame: From `policy.load_rehearsal_frame` — labels, days, amounts.
        scored: The arm's scores, VAL-CAL rows among them.
        costs: The version-1 costs.
        capacity: Review capacity.
        calibration: The `calibration:` config block.
        method: The method §1 selected, applied unchanged.

    Returns:
        `(summary, cost per transaction aligned to frame)`.

    Raises:
        ValueError: If the arm's VAL-CAL rows or labels differ from the frame's.
    """
    val_cal = scored[scored["split"] == "val_cal"].set_index("TransactionID")

    if set(val_cal.index) != set(frame["TransactionID"]):
        raise ValueError("the arm scored different VAL-CAL rows than the rehearsal")

    val_cal = val_cal.loc[frame["TransactionID"]]
    if (val_cal[LABEL].to_numpy() != frame[LABEL].to_numpy()).any():
        raise ValueError("the arm and the rehearsal disagree on VAL-CAL labels")

    raw = val_cal["score"].to_numpy(dtype="float64")
    y = frame[LABEL].to_numpy()
    day = frame["day"].to_numpy()
    amount = frame["amount"].to_numpy(dtype="float64")

    check_clip(raw, calibration["score_clip"])
    fold = assign_folds(frame["day"], calibration["n_folds"]).to_numpy()
    p, _ = cross_fit(raw, y, fold, method, calibration["score_clip"], calibration["ece_bins"])

    decision = ev_policy(p, amount, day, costs, capacity)
    return decision.summary(y, amount, day, costs), decision.cost(y, amount, costs)


def main(config_path: Path = DEFAULT_CONFIG_PATH) -> None:
    """Refit, verify, calibrate and cost every E1 and E3 arm; write one record.

    Wiring only. Invoked by `make usd-halves` as `python -m fraud_engine.models.usd_halves`.

    Args:
        config_path: Path to `config.yaml`.
    """
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    config = load_config(config_path)
    paths, halves_cfg = config["paths"], config["usd_halves"]
    model_cfg = {**config["model"], "tuned": {}}

    configure_tracking(config["tracking"])

    cost_matrix = load_config(Path(paths["cost_matrix"]))
    costs = load_costs(cost_matrix)
    capacity = load_operating_capacity(cost_matrix)
    method = json.loads(Path(paths["calibration"]).read_text())["selected"]

    frame = load_rehearsal_frame(paths["predictions_dir"], paths["interim"], method)
    y, amount, day = frame[LABEL].to_numpy(), frame["amount"].to_numpy(), frame["day"].to_numpy()

    rules = rules_policy(frame["rules_score"].to_numpy(), day, capacity)
    rules_usd = rules.summary(y, amount, day, costs)["usd_per_1000"]

    experiments = {}
    params = {"calibration": method, "cost_matrix_version": costs.version, **halves_cfg}

    with tracked_run(NAME, params, config_path):
        prepared: dict[Path, dict] = {}

        for experiment, arms in resolve_experiment_arms(paths).items():
            results, arm_costs = {}, {}

            for arm in arms:
                if arm.features_dir not in prepared:
                    prepared[arm.features_dir], _, _ = prepare_matrices(
                        arm.features_dir, model_cfg, SPLITS
                    )
                matrices = prepared[arm.features_dir]

                with tracked_child(
                    f"{NAME}_{experiment}_{arm.name}",
                    {**resolve_params(model_cfg), "experiment": experiment, "arm": arm.name},
                    artifacts={"removed_columns.json": list(arm.dropped)},
                ):
                    scored, best_iteration = fit_arm(matrices, arm.dropped, model_cfg)
                    recorded = pd.read_parquet(
                        Path(paths["predictions_dir"]) / f"{arm.recorded}.parquet"
                    )
                    check_reproduces(scored, recorded, arm.recorded)

                    summary, arm_costs[arm.name] = cost_arm(
                        frame, scored, costs, capacity, config["calibration"], method
                    )
                    summary["reduction_vs_rules"] = float(1 - summary["usd_per_1000"] / rules_usd)
                    mlflow.log_metrics({**summary, "best_iteration": best_iteration})

                results[arm.name] = {
                    "best_iteration": best_iteration,
                    "removed": len(arm.dropped),
                    **summary,
                }
                log.info(
                    "%s %-10s $%8.2f per 1,000  block %.4f  reviews/day %.2f",
                    experiment,
                    arm.name,
                    summary["usd_per_1000"],
                    summary["block_rate"],
                    summary["reviews_per_day"],
                )

            reference = arms[0].name
            for arm in arms[1:]:
                low, high = paired_day_bootstrap(
                    day,
                    arm_costs[arm.name],
                    arm_costs[reference],
                    halves_cfg["bootstrap_resamples"],
                    halves_cfg["interval"],
                    halves_cfg["seed"],
                )
                results[arm.name]["delta_vs_reference"] = (
                    results[arm.name]["usd_per_1000"] - results[reference]["usd_per_1000"]
                )
                results[arm.name]["delta_interval"] = [low, high]
                results[arm.name]["interval_excludes_zero"] = bool(low > 0 or high < 0)

            experiments[experiment] = {"reference": reference, "arms": results}

    record = {
        "name": NAME,
        "created": datetime.now(UTC).isoformat(timespec="seconds"),
        "git_revision": git_revision(),
        "split": "val_cal",
        "instrument": "untuned",
        "probabilities": f"{method}, out-of-fold, refitted per arm",
        "review_capacity": capacity,
        "cost_matrix_version": costs.version,
        "rules_usd_per_1000": rules_usd,
        "bootstrap": halves_cfg,
        "experiments": experiments,
    }
    path = Path(paths["usd_halves"])
    path.write_text(json.dumps(record, indent=2) + "\n")
    log.info("wrote %s", path)


if __name__ == "__main__":
    main()
