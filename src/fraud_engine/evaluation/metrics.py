"""Evaluation metrics.

PR-AUC and recall@capacity are the primary metrics; ROC-AUC is secondary, for
comparability. Accuracy is deliberately absent — 96.5% by always answering "not
fraud" — and a module that cannot compute it cannot report it.

The capacity constraint these are measured against is in
``docs/problem-statement.md`` §3.2: 1% of **daily** transaction volume.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import NamedTuple

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score


def _validate(y_true: pd.Series, y_score: pd.Series) -> None:
    """Shared checks for the threshold-free metrics.

    Both are ranking measures over the whole vector, so a misaligned pair does
    not fail — sklearn sees two arrays and scores the wrong pairing.
    """
    if not y_true.index.equals(y_score.index):
        raise ValueError("y_true and y_score must share an index.")

    classes = {int(value) for value in pd.unique(y_true)}
    if not classes <= {0, 1}:
        raise ValueError(f"y_true must be binary 0/1, found {sorted(classes)}.")
    if len(classes) < 2:
        raise ValueError(
            f"y_true holds only class {classes.pop()}; both PR-AUC and ROC-AUC are "
            "undefined without positives and negatives."
        )


class CapacityResult(NamedTuple):
    """Recall at a review capacity, with the numbers behind it.

    ``ambiguous_days`` is not decoration. When scores are coarse — a rules
    baseline emits a handful of distinct values — the capacity cut lands inside
    a block of equal scores, and which of them gets reviewed is decided by row
    order rather than by the model. The recall is then an artifact. This counts
    the days where that happened so the number can be trusted or discarded on
    evidence.
    """

    recall: float
    reviewed: int
    caught: int
    positives: int
    capacity: float
    ambiguous_days: int


def recall_at_capacity(
    y_true: pd.Series,
    y_score: pd.Series,
    days: pd.Series,
    capacity: float,
) -> CapacityResult:
    """Share of fraud caught by reviewing the riskiest transactions each day.

    "Of all fraud, what do we catch if the team can review 1% of volume?" —
    the metric a risk team actually argues about, and a hard constraint in
    Phase 06 rather than a preference.

    **Ranked within each day, not across the slice.** ``§3.2`` commits to 1% of
    *daily* volume. Ranking globally lets the model spend its whole budget on
    the worst days, which a team with fixed daily capacity cannot do — the two
    select the same *number* of transactions and different ones, and the global
    version reports a recall the operation cannot deliver.

    **The daily quota is floored.** At 2,846 transactions 1% is 28.46 reviews;
    reviewing 29 exceeds a limit §3.2 calls hard. A stated convention, not a
    deep truth — it differs by at most one review per day.

    Ties are broken by original row order, which is deterministic but arbitrary.
    See ``ambiguous_days`` on the return.

    Args:
        y_true: Binary labels, 1 for fraud.
        y_score: Risk scores, higher meaning riskier. Any monotone scale — only
            the ordering within a day is used, so this works on probabilities,
            log-odds or a rules engine's points.
        days: The day each transaction falls on, defining the review periods.
        capacity: Fraction of each day's volume that can be reviewed, in (0, 1].

    Returns:
        A ``CapacityResult``. Check ``ambiguous_days`` before quoting ``recall``.

    Raises:
        ValueError: If the inputs are misaligned, ``capacity`` is outside
            (0, 1], or there are no positives to recall.
    """
    if not 0 < capacity <= 1:
        raise ValueError(f"capacity must be a fraction in (0, 1], got {capacity}.")
    if not (y_true.index.equals(y_score.index) and y_true.index.equals(days.index)):
        raise ValueError("y_true, y_score and days must share an index.")

    positives = int(y_true.sum())
    if positives == 0:
        raise ValueError("no positives: recall is undefined on a slice with no fraud.")

    frame = pd.DataFrame(
        {"y": y_true.to_numpy(), "score": y_score.to_numpy(), "day": days.to_numpy()}
    )

    # Sorting the negated score ascending rather than passing ascending=False:
    # both are stable in pandas today, but only this one is stable by
    # construction rather than by implementation detail.
    ranked = frame.iloc[np.argsort(-frame["score"].to_numpy(), kind="stable")]

    grouped = ranked.groupby("day", sort=False)
    position = grouped.cumcount().to_numpy()
    quota = np.floor(grouped["y"].transform("size").to_numpy() * capacity).astype(int)
    selected = position < quota

    caught = int(ranked["y"].to_numpy()[selected].sum())

    # The cut is ambiguous on a day when the last reviewed score equals the
    # first unreviewed one — the boundary falls inside a tied block.
    boundary = pd.DataFrame(
        {"day": ranked["day"].to_numpy(), "score": ranked["score"].to_numpy()},
        index=pd.RangeIndex(len(ranked)),
    )
    last_in = boundary[position == quota - 1].set_index("day")["score"]
    first_out = boundary[position == quota].set_index("day")["score"]
    shared = last_in.index.intersection(first_out.index)

    return CapacityResult(
        recall=caught / positives,
        reviewed=int(selected.sum()),
        caught=caught,
        positives=positives,
        capacity=capacity,
        ambiguous_days=int((last_in.loc[shared] == first_out.loc[shared]).sum()),
    )


def pr_auc(y_true: pd.Series, y_score: pd.Series) -> float:
    """Area under the precision-recall curve, as average precision.

    The primary metric. At a 3.5% positive rate the negatives dominate every
    ROC calculation, so precision-recall is where the signal is.

    **Average precision, not** ``auc(recall, precision)``. Both get called "the
    area under the PR curve" and they disagree: ``auc`` interpolates linearly
    between operating points, but the straight line between two PR points is not
    reachable — the achievable path is a staircase — so the trapezoid counts area
    no threshold delivers. Average precision sums ``(Rn - Rn-1) * Pn``, which is
    only what a real cut-off achieves.

    The size of the error tracks the number of *distinct scores*, not the base
    rate. On continuous scores the two agree to the fourth decimal. On coarse
    ones they diverge violently — measured at a 3.5% base rate, a two-valued
    score gives 0.056 here against 0.466 by trapezoid, an eightfold inflation.
    The Phase 03 rules baseline emits a handful of distinct values, so that is
    its ordinary case rather than a corner one.

    Args:
        y_true: Binary labels, 1 for fraud.
        y_score: Risk scores, higher meaning riskier.

    Returns:
        Average precision. A random ranker scores about the base rate, which is
        the floor to compare against — not 0.5.

    Raises:
        ValueError: If the inputs are misaligned, not binary, or single-class.
    """
    _validate(y_true, y_score)
    return float(average_precision_score(y_true, y_score))


def roc_auc(y_true: pd.Series, y_score: pd.Series) -> float:
    """Area under the ROC curve. Secondary — reported for comparability only.

    Every published number on this dataset is an ROC-AUC, so omitting it makes
    the work harder to place. It is not steered by: at a 3.5% positive rate a
    model can move a great deal of precision while barely moving ROC-AUC,
    because the false-positive rate is measured against a huge denominator.

    Args:
        y_true: Binary labels, 1 for fraud.
        y_score: Risk scores, higher meaning riskier.

    Returns:
        ROC-AUC. A random ranker scores 0.5 regardless of the base rate.

    Raises:
        ValueError: If the inputs are misaligned, not binary, or single-class.
    """
    _validate(y_true, y_score)
    return float(roc_auc_score(y_true, y_score))


def recall_ceiling(y_true: pd.Series, days: pd.Series, capacity: float) -> float:
    """The best recall *any* ranker could reach at this capacity.

    Capacity binds long before recall reaches 1.0: if a day carries more fraud
    than the review budget holds, no ranking can catch all of it. So a recall of
    5% is uninterpretable on its own — against a ceiling of 29% it is a fifth of
    what was available, against a ceiling of 6% it is nearly everything.

    Measured, not derived, because the ceiling depends on how fraud is
    distributed across days: the same fraud count concentrated on a few days
    yields a lower ceiling than the same count spread evenly.

    Implemented by ranking on the labels themselves — the oracle scorer.

    Args:
        y_true: Binary labels, 1 for fraud.
        days: The day each transaction falls on.
        capacity: Fraction of each day's volume that can be reviewed.

    Returns:
        The oracle's recall at this capacity, in [0, 1].
    """
    return recall_at_capacity(y_true, y_true.astype(float), days, capacity).recall


def _capacity_entry(
    y_true: pd.Series,
    y_score: pd.Series,
    days: pd.Series,
    capacity: float,
) -> dict:
    """One capacity's result, carrying the ceiling it should be read against."""
    entry = recall_at_capacity(y_true, y_score, days, capacity)._asdict()
    ceiling = recall_ceiling(y_true, days, capacity)
    entry["recall_ceiling"] = ceiling
    # Guarded rather than assumed positive: a capacity small enough to floor
    # every day's quota to zero catches nothing, and the ceiling is then 0 too.
    entry["share_of_ceiling"] = entry["recall"] / ceiling if ceiling else float("nan")
    return entry


def evaluate(
    y_true: pd.Series,
    y_score: pd.Series,
    days: pd.Series,
    capacities: Sequence[float],
) -> dict:
    """Every metric for one prediction vector, in one serialisable dict.

    The shape report.py writes and Phase 09 compares across time windows, so it
    stays JSON-friendly: plain floats and ints, no numpy scalars, capacities as
    a list rather than dict keys.

    ``base_rate`` is included because it is the reference for ``pr_auc`` — a
    PR-AUC of 0.30 means nothing until you know whether the floor was 0.035 or
    0.30. Accuracy is not included, here or anywhere.

    Args:
        y_true: Binary labels, 1 for fraud.
        y_score: Risk scores, higher meaning riskier.
        days: The day each transaction falls on.
        capacities: Review capacities to report, as fractions of daily volume.

    Returns:
        ``{n, positives, base_rate, pr_auc, roc_auc, recall_at_capacity}``,
        where the last is a list of ``CapacityResult`` dicts, one per capacity,
        each extended with ``recall_ceiling`` and ``share_of_ceiling`` — see
        ``recall_ceiling`` for why a recall figure is unreadable without them.

    Raises:
        ValueError: If the inputs are misaligned, not binary, or single-class.
    """
    _validate(y_true, y_score)

    return {
        "n": len(y_true),
        "positives": int(y_true.sum()),
        "base_rate": float(y_true.mean()),
        "pr_auc": pr_auc(y_true, y_score),
        "roc_auc": roc_auc(y_true, y_score),
        "recall_at_capacity": [
            _capacity_entry(y_true, y_score, days, capacity) for capacity in capacities
        ],
    }
