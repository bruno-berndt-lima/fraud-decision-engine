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
    allow_or_block,
    break_even,
    expected_costs,
    load_costs,
    realised_cost,
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
