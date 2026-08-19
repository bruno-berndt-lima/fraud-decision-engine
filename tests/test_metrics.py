"""Tests for the evaluation metrics.

Small hand-checkable cases for recall@capacity, where the design decisions are;
synthetic populations for the threshold-free metrics, where the properties are
statistical.
"""

import json

import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import auc, average_precision_score, precision_recall_curve

from fraud_engine.evaluation.metrics import evaluate, pr_auc, recall_at_capacity, roc_auc

BASE_RATE = 0.035


def series(*columns):
    """Align several plain lists onto one index."""
    index = pd.RangeIndex(len(columns[0]))
    return tuple(pd.Series(column, index=index) for column in columns)


def population(n=20_000, base_rate=BASE_RATE, seed=0):
    """Labels and a random score, with the score independent of the label."""
    rng = np.random.default_rng(seed)
    y_true = pd.Series((rng.random(n) < base_rate).astype(int))
    return y_true, pd.Series(rng.random(n), index=y_true.index)


# ---- recall_at_capacity: the counting ---------------------------------------


def test_counts_fraud_caught_in_each_days_top_slice():
    """Day 1's two frauds rank top; day 2 hides one at the bottom of the day."""
    y, score, day = series(
        [*[1, 1, 0, 0, 0, 0, 0, 0, 0, 0], *[1, 0, 0, 0, 0, 0, 0, 0, 0, 1]],
        list(range(9, -1, -1)) * 2,
        [*[1] * 10, *[2] * 10],
    )
    result = recall_at_capacity(y, score, day, 0.2)

    assert result.reviewed == 4  # two per day
    assert result.caught == 3
    assert result.positives == 4
    assert result.recall == pytest.approx(0.75)


def test_ranks_within_each_day_not_across_the_slice():
    """The whole reason this metric groups by day.

    Day 1 holds four frauds and the highest scores. Spending the same budget
    globally puts every review on day 1, which a team with fixed daily capacity
    cannot do — and reports a recall the operation cannot deliver.
    """
    y, score, day = series(
        [*[1, 1, 1, 1, 0, 0, 0, 0, 0, 0], *[1, 0, 0, 0, 0, 0, 0, 0, 0, 0]],
        [*[99, 98, 97, 96, 5, 4, 3, 2, 1, 0], *[50, 8, 7, 6, 5, 4, 3, 2, 1, 0]],
        [*[1] * 10, *[2] * 10],
    )
    result = recall_at_capacity(y, score, day, 0.2)
    assert result.recall == pytest.approx(0.6)

    globally_ranked = y.iloc[np.argsort(-score.to_numpy(), kind="stable")[: result.reviewed]]
    assert globally_ranked.sum() / y.sum() == pytest.approx(0.8)


def test_the_daily_quota_is_floored():
    """7 transactions at capacity 0.5 is 3.5 seats; a team cannot review half a case."""
    y, score, day = series([1, 0, 0, 0, 0, 0, 0], [7, 6, 5, 4, 3, 2, 1], [1] * 7)
    assert recall_at_capacity(y, score, day, 0.5).reviewed == 3


def test_days_are_quotaed_independently():
    """A big day gets more seats than a small one — 1% of *its own* volume."""
    y, score, day = series(
        [*[1] * 4, *[0] * 16],
        [*range(16, 0, -1), 4, 3, 2, 1],
        [*[1] * 16, *[2] * 4],
    )
    assert recall_at_capacity(y, score, day, 0.25).reviewed == 5  # floor(4) + floor(1)


def test_capacity_of_one_reviews_everything():
    y, score, day = series([1, 0, 0, 1], [4, 3, 2, 1], [1] * 4)
    result = recall_at_capacity(y, score, day, 1.0)
    assert result.reviewed == 4
    assert result.recall == 1.0


def test_only_the_ordering_of_scores_matters():
    """Any monotone scale works — probabilities, log-odds, or rules points."""
    y, score, day = series([1, 1, 0, 0, 0, 0, 0, 0, 0, 0], list(range(10, 0, -1)), [1] * 10)
    baseline = recall_at_capacity(y, score, day, 0.2)
    rescaled = recall_at_capacity(y, score * 1000 - 7, day, 0.2)
    assert baseline.recall == rescaled.recall


# ---- recall_at_capacity: the tie diagnostic ---------------------------------


def test_flags_a_day_whose_cut_falls_inside_a_tied_block():
    y, score, day = series([1, 0, 0, 0, 0], [1, 1, 1, 1, 1], [1] * 5)
    assert recall_at_capacity(y, score, day, 0.4).ambiguous_days == 1


def test_no_ambiguity_when_scores_are_distinct():
    y, score, day = series([1, 0, 0, 0, 0], [5, 4, 3, 2, 1], [1] * 5)
    assert recall_at_capacity(y, score, day, 0.4).ambiguous_days == 0


def test_counts_ambiguous_days_not_ambiguous_rows():
    """One tied day among two — the diagnostic is per review period."""
    y, score, day = series(
        [*[1, 0, 0, 0, 0], *[1, 0, 0, 0, 0]],
        [*[1, 1, 1, 1, 1], *[5, 4, 3, 2, 1]],
        [*[1] * 5, *[2] * 5],
    )
    assert recall_at_capacity(y, score, day, 0.4).ambiguous_days == 1


# ---- recall_at_capacity: guards ---------------------------------------------


@pytest.mark.parametrize("capacity", [0, -0.1, 1.5])
def test_rejects_a_capacity_outside_the_unit_interval(capacity):
    y, score, day = series([1, 0], [2, 1], [1, 1])
    with pytest.raises(ValueError, match="capacity"):
        recall_at_capacity(y, score, day, capacity)


def test_rejects_a_slice_with_no_positives():
    y, score, day = series([0, 0], [2, 1], [1, 1])
    with pytest.raises(ValueError, match="no positives"):
        recall_at_capacity(y, score, day, 0.5)


def test_recall_rejects_misaligned_inputs():
    y, score, day = series([1, 0], [2, 1], [1, 1])
    with pytest.raises(ValueError, match="share an index"):
        recall_at_capacity(y, score, day.iloc[:-1], 0.5)


# ---- pr_auc ------------------------------------------------------------------


@pytest.mark.parametrize("seed", range(5))
def test_random_scores_give_a_pr_auc_near_the_base_rate(seed):
    """The DoD sanity check: if this drifts, the harness is lying."""
    y_true, y_score = population(seed=seed)
    assert pr_auc(y_true, y_score) == pytest.approx(y_true.mean(), abs=0.01)


def test_a_perfect_ranker_scores_one():
    y_true, _ = population()
    assert pr_auc(y_true, y_true.astype(float)) == pytest.approx(1.0)


def test_pr_auc_is_average_precision_and_not_the_trapezoid():
    """Pins the choice against a 'simplification' to auc(recall, precision).

    The trapezoid interpolates along a path no threshold can reach. How badly
    depends on how many distinct scores there are — and the Phase 03 rules
    baseline emits a handful, so this is its ordinary case.
    """
    rng = np.random.default_rng(0)
    n = 20_000
    y_true = pd.Series((rng.random(n) < BASE_RATE).astype(int))
    latent = y_true * 1.2 + rng.normal(size=n)
    two_valued = pd.Series((latent > latent.median()).astype(int), index=y_true.index)

    measured = pr_auc(y_true, two_valued)
    precision, recall, _ = precision_recall_curve(y_true, two_valued)

    assert measured == pytest.approx(average_precision_score(y_true, two_valued))
    # Guards the guard: if the two agreed here, the test would prove nothing.
    assert auc(recall, precision) - measured > 0.3


# ---- roc_auc -----------------------------------------------------------------


@pytest.mark.parametrize("seed", range(5))
def test_random_scores_give_a_roc_auc_near_a_half(seed):
    y_true, y_score = population(seed=seed)
    assert roc_auc(y_true, y_score) == pytest.approx(0.5, abs=0.05)


def test_an_inverted_ranker_scores_zero():
    y_true, _ = population()
    assert roc_auc(y_true, -y_true.astype(float)) == pytest.approx(0.0)


# ---- shared guards -----------------------------------------------------------


@pytest.mark.parametrize("metric", [pr_auc, roc_auc])
def test_threshold_free_metrics_reject_misaligned_inputs(metric):
    """Neither would fail on its own — sklearn scores the wrong pairing."""
    y_true, y_score = population(n=100)
    with pytest.raises(ValueError, match="share an index"):
        metric(y_true, y_score.iloc[:-1])


@pytest.mark.parametrize("metric", [pr_auc, roc_auc])
@pytest.mark.parametrize(
    ("label", "labels"),
    [("all negative", [0, 0, 0]), ("all positive", [1, 1, 1]), ("not binary", [0, 1, 2])],
)
def test_threshold_free_metrics_reject_unusable_labels(metric, label, labels):
    y_true, y_score = series(labels, [3.0, 2.0, 1.0])
    with pytest.raises(ValueError):
        metric(y_true, y_score)


# ---- evaluate ----------------------------------------------------------------


def test_evaluate_reports_every_metric_and_capacity():
    y_true, y_score = population()
    days = pd.Series(np.arange(len(y_true)) % 20, index=y_true.index)

    result = evaluate(y_true, y_score, days, capacities=[0.005, 0.01, 0.02])

    assert result["n"] == len(y_true)
    assert result["positives"] == int(y_true.sum())
    assert result["base_rate"] == pytest.approx(y_true.mean())
    assert result["pr_auc"] == pr_auc(y_true, y_score)
    assert result["roc_auc"] == roc_auc(y_true, y_score)
    assert [row["capacity"] for row in result["recall_at_capacity"]] == [0.005, 0.01, 0.02]


def test_evaluate_output_is_json_serialisable():
    """It is written to reports/ and diffed across runs, so no numpy scalars."""
    y_true, y_score = population(n=2_000)
    days = pd.Series(np.arange(len(y_true)) % 10, index=y_true.index)

    round_tripped = json.loads(json.dumps(evaluate(y_true, y_score, days, capacities=[0.1])))
    assert round_tripped["recall_at_capacity"][0]["capacity"] == 0.1


def test_evaluate_reports_no_accuracy():
    """The invariant, enforced: 96.5% by always answering 'not fraud'."""
    y_true, y_score = population(n=2_000)
    days = pd.Series(np.arange(len(y_true)) % 10, index=y_true.index)

    assert "accuracy" not in evaluate(y_true, y_score, days, capacities=[0.1])
