"""How one arm is costed on VAL-CAL, and how far the difference between two could be.

Its own module for the reason `reproduce.py` gives, in mirror image. Both functions were
written in `usd_halves`, and a second stage wanting them had two bad options: import that
module, dragging an experiment's whole graph — the purge arms, the ablation, the rules
engine — and the config sections those read but the caller does not, into a make rule
that would then restage on edits to sections it never opens; or hold a second copy of a
numeric routine and let the two records drift apart.

They stay one definition, in a module that reads no config section of its own.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from fraud_engine.evaluation.cost import Costs, ev_policy
from fraud_engine.models.calibrate import SPLIT, assign_folds, check_clip, cross_fit
from fraud_engine.models.train import LABEL


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
        method: The method `decision-policy.md` §1 selected, applied unchanged.

    Returns:
        `(summary, cost per transaction aligned to frame)`.

    Raises:
        ValueError: If the arm's VAL-CAL rows or labels differ from the frame's.
    """
    val_cal = scored[scored["split"] == SPLIT].set_index("TransactionID")

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
    It captures which days the slice happened to hold, not how a refit would move.

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
