"""Tests for the cost of a decision.

The currency guard and the break-even are the ones that matter: a wrong unit or a
dropped term does not raise anywhere downstream, it just blocks the wrong
transactions and reports a USD figure for them.
"""

from __future__ import annotations

import copy
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
import yaml

from fraud_engine.evaluation.cost import (
    ALLOW,
    BLOCK,
    REVIEW,
    Costs,
    Decision,
    allow_or_block,
    break_even,
    daily_capacity,
    ev_policy,
    expected_costs,
    load_costs,
    naive_policy,
    realised_cost,
    review_shares,
    rules_policy,
    usd_per_thousand,
)

REPO_ROOT = Path(__file__).resolve().parents[1]

COSTS = Costs(chargeback_fee=25.0, false_positive=15.0, review=1.5, review_friction=3.0, version=1)


@pytest.fixture
def matrix() -> dict:
    return yaml.safe_load((REPO_ROOT / "config" / "cost_matrix.yaml").read_text())


def test_the_committed_matrix_loads_as_version_1(matrix):
    assert load_costs(matrix) == COSTS


def test_another_currency_is_refused(matrix):
    brl = {**matrix, "currency": "BRL"}

    with pytest.raises(ValueError, match="not 'USD'"):
        load_costs(brl)


def test_a_changed_false_negative_formula_is_refused(matrix):
    changed = copy.deepcopy(matrix)
    changed["costs"]["false_negative"]["formula"] = "amount"

    with pytest.raises(ValueError, match="false_negative formula"):
        load_costs(changed)


@pytest.mark.parametrize("name", ["true_positive", "true_negative"])
def test_a_nonzero_cost_the_policy_assumes_zero_is_refused(matrix, name):
    changed = copy.deepcopy(matrix)
    changed["costs"][name]["value"] = 1.0

    with pytest.raises(ValueError, match=f"{name} is 1.0"):
        load_costs(changed)


def test_a_negative_cost_is_refused(matrix):
    changed = copy.deepcopy(matrix)
    changed["costs"]["review"]["value"] = -1.0

    with pytest.raises(ValueError, match="negative costs"):
        load_costs(changed)


def test_break_even_matches_the_worked_examples_in_the_matrix():
    """cost_matrix.yaml: $20 blocks above 25.0%, $8,000 above 0.187%."""
    p_star = break_even(np.array([20.0, 8_000.0]), COSTS)

    assert p_star[0] == pytest.approx(0.25)
    assert p_star[1] == pytest.approx(15 / 8_040)
    assert round(p_star[1] * 100, 3) == 0.187


def test_break_even_falls_as_the_amount_rises():
    assert (np.diff(break_even(np.linspace(0, 10_000, 1_001), COSTS)) < 0).all()


def test_allow_and_block_cost_the_same_at_break_even():
    amount = np.array([20.0, 150.0, 8_000.0])
    expected = expected_costs(break_even(amount, COSTS), amount, COSTS)

    assert np.allclose(expected[ALLOW], expected[BLOCK])


def test_the_decision_flips_exactly_at_break_even():
    amount = np.array([20.0, 20.0, 20.0])
    p = np.array([0.2499, 0.25, 0.2501])

    assert allow_or_block(p, amount, COSTS).tolist() == [ALLOW, ALLOW, BLOCK]


def test_the_decision_is_the_cheaper_action_in_expectation():
    rng = np.random.default_rng(0)
    p = rng.uniform(0, 1, size=10_000)
    amount = rng.lognormal(4, 1.5, size=10_000)

    action = allow_or_block(p, amount, COSTS)
    expected = expected_costs(p, amount, COSTS)
    chosen = np.where(action == BLOCK, expected[BLOCK], expected[ALLOW])

    assert (chosen <= np.minimum(expected[ALLOW], expected[BLOCK]) + 1e-12).all()


def test_the_same_probability_blocks_a_large_amount_and_allows_a_small_one():
    action = allow_or_block(np.array([0.05, 0.05]), np.array([20.0, 8_000.0]), COSTS)

    assert action.tolist() == [ALLOW, BLOCK]


def test_a_higher_false_positive_cost_raises_the_bar():
    amount = np.array([100.0])

    assert break_even(amount, replace(COSTS, false_positive=100.0)) > break_even(amount, COSTS)


@pytest.mark.parametrize(
    ("p", "amount", "match"),
    [
        ([np.nan], [10.0], "finite"),
        ([0.5], [np.nan], "finite"),
        ([1.5], [10.0], r"\[0, 1\]"),
        ([0.5], [-1.0], "non-negative"),
        ([0.5, 0.5], [10.0], "2 probabilities for 1 amounts"),
    ],
)
def test_inputs_that_would_decide_silently_are_refused(p, amount, match):
    with pytest.raises(ValueError, match=match):
        allow_or_block(np.array(p), np.array(amount), COSTS)


def test_realised_cost_of_every_outcome():
    action = np.array([ALLOW, ALLOW, BLOCK, BLOCK, REVIEW, REVIEW])
    y = np.array([1, 0, 1, 0, 1, 0])
    amount = np.full(6, 100.0)

    cost = realised_cost(action, y, amount, COSTS)

    assert cost.tolist() == [125.0, 0.0, 0.0, 15.0, 1.5, 4.5]


def test_realised_cost_refuses_an_unknown_action():
    with pytest.raises(ValueError, match="unknown actions"):
        realised_cost(np.array(["hold"]), np.array([1]), np.array([10.0]), COSTS)


def test_realised_cost_refuses_a_label_that_is_not_binary():
    with pytest.raises(ValueError, match="0 or 1"):
        realised_cost(np.array([ALLOW]), np.array([2]), np.array([10.0]), COSTS)


def test_expected_cost_is_the_mean_realised_cost_under_calibrated_labels():
    """What makes the policy sound: with honest probabilities, expectation and
    outcome agree on average."""
    rng = np.random.default_rng(0)
    n = 400_000
    p = rng.uniform(0, 0.2, size=n)
    amount = rng.lognormal(4, 1, size=n)
    y = rng.binomial(1, p)

    action = allow_or_block(p, amount, COSTS)
    expected = expected_costs(p, amount, COSTS)
    chosen = np.where(action == BLOCK, expected[BLOCK], expected[ALLOW])

    realised = realised_cost(action, y, amount, COSTS)

    assert realised.mean() == pytest.approx(chosen.mean(), rel=0.02)


def test_usd_per_thousand_scales_the_mean():
    assert usd_per_thousand(np.array([10.0, 0.0, 5.0, 1.0])) == pytest.approx(4_000.0)


def test_usd_per_thousand_refuses_an_empty_slice():
    with pytest.raises(ValueError, match="no transactions"):
        usd_per_thousand(np.array([]))


# ---- review under a daily capacity ---------------------------------------------


def test_capacity_is_floored_per_day():
    day = np.repeat([1, 2, 3], [250, 199, 99])

    assert daily_capacity(day, 0.01) == {1: 2, 2: 1, 3: 0}


def test_float_error_does_not_take_a_review_away():
    """0.29 * 100 is 28.999... in floating point; the day has room for 29."""
    assert daily_capacity(np.ones(100, dtype=int), 0.29) == {1: 29}


def test_the_highest_priorities_fill_the_capacity():
    priority = np.array([5.0, 1.0, 4.0, 2.0, 3.0])

    share = review_shares(priority, np.ones(5, dtype=int), fraction=0.4)

    assert share.tolist() == [1.0, 0.0, 1.0, 0.0, 0.0]


def test_ties_at_the_cut_share_the_remaining_capacity():
    """Capacity 3: one clear winner, then four tied for the last two slots."""
    priority = np.array([9.0, 5.0, 5.0, 5.0, 5.0, 1.0, 1.0, 1.0, 1.0, 1.0])

    share = review_shares(priority, np.ones(10, dtype=int), fraction=0.3)

    assert share.tolist() == [1.0, 0.5, 0.5, 0.5, 0.5, 0, 0, 0, 0, 0]
    assert share.sum() == pytest.approx(3)


def test_shares_do_not_depend_on_row_order():
    rng = np.random.default_rng(0)
    priority = rng.integers(0, 4, size=1_000).astype(float)
    day = rng.integers(0, 5, size=1_000)
    order = rng.permutation(1_000)

    share = review_shares(priority, day, 0.05)
    shuffled = review_shares(priority[order], day[order], 0.05)

    assert np.allclose(shuffled, share[order])


def test_no_day_exceeds_its_capacity():
    rng = np.random.default_rng(0)
    day = rng.integers(0, 20, size=20_000)
    priority = rng.integers(0, 6, size=20_000).astype(float)

    share = review_shares(priority, day, 0.01)
    capacity = daily_capacity(day, 0.01)

    for d, allowed in capacity.items():
        assert share[day == d].sum() == pytest.approx(allowed)


def test_capacity_is_not_pooled_across_days():
    """A day full of high priorities cannot borrow an empty day's reviews."""
    day = np.repeat([1, 2], 100)
    priority = np.r_[np.full(100, 10.0), np.zeros(100)]

    share = review_shares(priority, day, 0.05)

    assert share[day == 1].sum() == pytest.approx(5)
    assert share[day == 2].sum() == pytest.approx(5)


def test_ineligible_transactions_are_never_reviewed_even_with_room():
    priority = np.array([5.0, 4.0, 3.0, 2.0])
    eligible = np.array([False, True, False, True])

    share = review_shares(priority, np.ones(4, dtype=int), 0.75, eligible=eligible)

    assert share.tolist() == [0.0, 1.0, 0.0, 1.0]


def test_zero_capacity_reviews_nothing():
    assert review_shares(np.arange(50.0), np.ones(50, dtype=int), 0.01).sum() == 0


# ---- the policies ----------------------------------------------------------------


@pytest.fixture
def slice_() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Twenty days of calibrated probabilities over a skewed amount distribution."""
    rng = np.random.default_rng(0)
    n = 40_000
    day = np.repeat(np.arange(141, 161), n // 20)
    p = np.clip(rng.beta(0.3, 8, size=n), 0, 1)
    amount = rng.lognormal(4, 1.2, size=n)
    y = rng.binomial(1, p)
    return p, amount, day, y


def test_the_ev_policy_without_capacity_is_allow_or_block(slice_):
    p, amount, day, _ = slice_

    decision = ev_policy(p, amount, day, COSTS, capacity=0.0)

    assert decision.review_share.sum() == 0
    assert (decision.fallback == allow_or_block(p, amount, COSTS)).all()


def test_the_ev_policy_reviews_only_where_review_is_cheaper(slice_):
    p, amount, day, _ = slice_
    expected = expected_costs(p, amount, COSTS)
    gain = np.minimum(expected[ALLOW], expected[BLOCK]) - expected[REVIEW]

    decision = ev_policy(p, amount, day, COSTS, capacity=0.5)

    assert (gain[decision.review_share > 0] > 0).all()


def test_the_ev_policy_reviews_the_largest_savings_first(slice_):
    p, amount, day, _ = slice_
    expected = expected_costs(p, amount, COSTS)
    gain = np.minimum(expected[ALLOW], expected[BLOCK]) - expected[REVIEW]

    decision = ev_policy(p, amount, day, COSTS, capacity=0.01)

    for d in np.unique(day):
        today = day == d
        reviewed = gain[today & (decision.review_share == 1)]
        passed_over = gain[today & (decision.review_share == 0) & (gain > 0)]
        if reviewed.size and passed_over.size:
            assert reviewed.min() >= passed_over.max()


def test_review_never_raises_the_expected_cost(slice_):
    p, amount, day, _ = slice_
    expected = expected_costs(p, amount, COSTS)

    def expected_total(decision: Decision) -> float:
        unreviewed = np.where(decision.fallback == BLOCK, expected[BLOCK], expected[ALLOW])
        share = decision.review_share
        return float((share * expected[REVIEW] + (1 - share) * unreviewed).sum())

    without = expected_total(ev_policy(p, amount, day, COSTS, capacity=0.0))
    with_review = expected_total(ev_policy(p, amount, day, COSTS, capacity=0.01))

    assert with_review < without


def test_the_rules_engine_never_blocks_and_fills_its_capacity(slice_):
    _, _, day, _ = slice_
    points = np.random.default_rng(1).integers(0, 8, size=len(day)).astype(float)

    decision = rules_policy(points, day, 0.01)

    assert (decision.fallback == ALLOW).all()
    for d, allowed in daily_capacity(day, 0.01).items():
        assert decision.review_share[day == d].sum() == pytest.approx(allowed)


def test_the_naive_policy_blocks_at_and_above_its_threshold():
    decision = naive_policy(np.array([0.49, 0.5, 0.9]))

    assert decision.fallback.tolist() == [ALLOW, BLOCK, BLOCK]
    assert decision.review_share.sum() == 0


def test_a_prorated_transaction_is_costed_as_its_share_of_each_action():
    decision = Decision(fallback=np.array([ALLOW]), review_share=np.array([0.25]))

    cost = decision.cost(np.array([1]), np.array([100.0]), COSTS)

    assert cost[0] == pytest.approx(0.25 * 1.5 + 0.75 * 125.0)


def test_the_summary_reports_what_the_headline_was_bought_with():
    decision = Decision(
        fallback=np.array([BLOCK, BLOCK, ALLOW, ALLOW]),
        review_share=np.array([0.0, 0.5, 1.0, 0.0]),
    )
    y = np.array([0, 1, 0, 0])
    amount = np.full(4, 100.0)
    day = np.array([1, 1, 2, 2])

    summary = decision.summary(y, amount, day, COSTS)

    # Costs: 15 + (0.5 * 1.5 + 0.5 * 0) + 4.5 + 0 = 20.25 over 4 transactions.
    assert summary["usd_per_1000"] == pytest.approx(20.25 / 4 * 1_000)
    assert summary["reviews_per_day"] == pytest.approx(0.75)
    assert summary["block_rate"] == pytest.approx(1.5 / 4)
