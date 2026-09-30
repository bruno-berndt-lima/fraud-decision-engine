"""Tests for what the headline was missing (decision-policy.md §8).

The cut search against brute force and against the policy it names, the ladder's rows
against the ones §4 already defines, the reduction interval shown to share its draws with
the USD one, and the proof shown to refuse a record that differs at all.
"""

from itertools import pairwise
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

from fraud_engine.evaluation.arms import paired_day_bootstrap
from fraud_engine.evaluation.attribution import (
    LADDER,
    best_global_cut,
    break_even_only,
    check_rows,
    ladder,
    measure,
    reduction_interval,
)
from fraud_engine.evaluation.cost import ALLOW, ev_policy, load_costs, naive_policy
from fraud_engine.evaluation.policy import rehearse
from fraud_engine.evaluation.report import load_operating_capacity

CONFIG = yaml.safe_load(Path("config/config.yaml").read_text())
COST_MATRIX = yaml.safe_load(Path(CONFIG["paths"]["cost_matrix"]).read_text())
COSTS = load_costs(COST_MATRIX)
CAPACITY = load_operating_capacity(COST_MATRIX)
BOOTSTRAP = CONFIG["attribution"] | {"bootstrap_resamples": 200}


def frame(n=3_000, seed=0):
    rng = np.random.default_rng(seed)
    p = rng.beta(0.4, 10, n)
    return pd.DataFrame(
        {
            "isFraud": (rng.random(n) < p).astype(int),
            "amount": rng.lognormal(4, 1.2, n).round(2),
            "day": rng.integers(161, 183, n),
            "calibrated": p,
            "uncalibrated": np.clip(p * 1.5, 0, 1),
            "rules_score": rng.integers(0, 6, n).astype(float),
        }
    )


def usd(decision, data):
    y, amount, day = data["isFraud"].to_numpy(), data["amount"].to_numpy(), data["day"].to_numpy()
    return decision.summary(y, amount, day, COSTS)["usd_per_1000"]


def search(data):
    return best_global_cut(
        data["calibrated"].to_numpy(), data["isFraud"].to_numpy(), data["amount"].to_numpy(), COSTS
    )


# ---- the cut ----------------------------------------------------------------------


def test_the_cut_is_the_cheapest_of_every_distinct_probability():
    data = frame(n=400)
    p = data["calibrated"].to_numpy()
    brute = {t: usd(naive_policy(p, t), data) for t in np.unique(p)}
    assert usd(naive_policy(p, search(data)), data) == pytest.approx(min(brute.values()))


def test_equal_cost_cuts_resolve_to_the_highest():
    """At 0.4, allowing a $5 fraud costs $30 and blocking two good customers costs $30."""
    data = pd.DataFrame(
        {
            "isFraud": [1, 1, 0, 0, 0],
            "amount": [500.0, 5.0, 10.0, 10.0, 10.0],
            "calibrated": [0.6, 0.4, 0.4, 0.4, 0.1],
            "day": 161,
        }
    )
    p = data["calibrated"].to_numpy()
    assert usd(naive_policy(p, 0.6), data) == usd(naive_policy(p, 0.4), data)
    assert search(data) == 0.6


def test_a_cheaper_lower_cut_is_taken():
    data = pd.DataFrame(
        {"isFraud": [1, 1, 0], "amount": [500.0, 500.0, 10.0], "calibrated": [0.6, 0.4, 0.1]}
    )
    assert search(data) == 0.4


def test_a_cut_is_refused_when_blocking_nothing_is_cheapest():
    data = frame(n=200).assign(isFraud=0)
    with pytest.raises(ValueError, match="blocking nothing"):
        search(data)


# ---- the ladder -------------------------------------------------------------------


def test_the_break_even_row_is_the_ev_policy_less_its_reviews():
    data = frame()
    p, amount, day = data["calibrated"].to_numpy(), data["amount"].to_numpy(), data["day"]
    alone, ev = break_even_only(p, amount, COSTS), ev_policy(p, amount, day, COSTS, CAPACITY)
    assert np.array_equal(alone.fallback, ev.fallback)
    assert not alone.review_share.any() and ev.review_share.any()


def test_the_rows_section_4_already_defines_are_unchanged():
    data = frame()
    rows = measure(data, ladder(data, COSTS, CAPACITY, 0.5), COSTS, BOOTSTRAP)["rows"]
    known = rehearse(data, COSTS, CAPACITY)
    for name in ("rules", "naive", "ev"):
        assert rows[name] == known[name]


def test_a_cut_of_one_half_is_the_naive_row():
    data = frame()
    decisions = ladder(data, COSTS, CAPACITY, 0.5)
    assert np.array_equal(decisions["global_cut"].fallback, decisions["naive"].fallback)


def test_the_steps_add_up_to_the_whole_ladder():
    data = frame()
    measured = measure(data, ladder(data, COSTS, CAPACITY, search(data)), COSTS, BOOTSTRAP)
    rows, steps = measured["rows"], measured["steps"]
    assert [(s["from"], s["to"]) for s in steps] == list(pairwise(LADDER))
    whole = rows["ev"]["reduction_vs_rules"] - rows["naive"]["reduction_vs_rules"]
    assert sum(s["points"] for s in steps) == pytest.approx(100 * whole)


def test_a_step_is_shown_only_when_its_interval_excludes_zero():
    data = frame()
    for step in measure(data, ladder(data, COSTS, CAPACITY, 0.5), COSTS, BOOTSTRAP)["steps"]:
        low, high = step["usd_interval"]
        assert step["interval_excludes_zero"] == (high < 0 or low > 0)
        if step["from"] == "naive":  # a cut of 0.5 is the naive row: nothing to show
            assert step["usd_interval"] == [0.0, 0.0]
            assert not step["interval_excludes_zero"]


# ---- the intervals ----------------------------------------------------------------


def costs_for(data):
    decisions = ladder(data, COSTS, CAPACITY, 0.5)
    y, amount = data["isFraud"].to_numpy(), data["amount"].to_numpy()
    return decisions["ev"].cost(y, amount, COSTS), decisions["rules"].cost(y, amount, COSTS)


def test_the_reduction_interval_is_drawn_on_the_usd_interval_s_days():
    """Rebuilt from the same draws, the USD difference matches `paired_day_bootstrap`."""
    data = frame()
    ev, rules = costs_for(data)
    day = data["day"].to_numpy()
    resamples, interval, seed = 300, 0.95, 7

    days, index = np.unique(day, return_inverse=True)
    rows = np.bincount(index)
    delta = np.bincount(index, weights=ev) - np.bincount(index, weights=rules)
    draws = np.random.default_rng(seed).integers(0, days.size, size=(resamples, days.size))
    rebuilt = np.quantile(delta[draws].sum(axis=1) / rows[draws].sum(axis=1) * 1_000,
                          [0.025, 0.975])  # fmt: skip
    assert tuple(rebuilt) == paired_day_bootstrap(day, ev, rules, resamples, interval, seed)

    ratio = 1 - np.bincount(index, weights=ev)[draws].sum(axis=1) / np.bincount(
        index, weights=rules
    )[draws].sum(axis=1)
    assert reduction_interval(day, ev, rules, resamples, interval, seed) == tuple(
        np.quantile(ratio, [0.025, 0.975])
    )


def test_a_policy_against_itself_reduces_nothing():
    data = frame()
    _, rules = costs_for(data)
    assert reduction_interval(data["day"].to_numpy(), rules, rules, 100, 0.95, 0) == (0.0, 0.0)


def test_the_headline_interval_brackets_its_own_estimate():
    data = frame()
    head = measure(data, ladder(data, COSTS, CAPACITY, 0.5), COSTS, BOOTSTRAP)["headline"]
    low, high = head["reduction_interval"]
    assert low <= head["reduction"] <= high


# ---- the proof --------------------------------------------------------------------


def recorded(data):
    return {"policies": rehearse(data, COSTS, CAPACITY)}


def test_rows_that_reproduce_the_record_pass():
    data = frame()
    check_rows(rehearse(data, COSTS, CAPACITY), recorded(data), "the rehearsal")


def test_a_row_that_differs_at_all_is_refused():
    data = frame()
    record = recorded(data)
    record["policies"]["ev"]["usd_per_1000"] += 1e-9
    with pytest.raises(ValueError, match=r"ev\.usd_per_1000"):
        check_rows(rehearse(data, COSTS, CAPACITY), record, "the rehearsal")


def test_allowing_everything_is_never_a_ladder_row():
    """The ladder is the model's; the reference stays the rules engine."""
    decisions = ladder(frame(), COSTS, CAPACITY, 0.5)
    assert set(decisions) == {"rules", *LADDER}
    assert (decisions["rules"].fallback == ALLOW).all()
