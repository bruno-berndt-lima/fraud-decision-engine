"""Tests for the rules baseline engine.

No parquet is read — CI has no data. Every frame here is built by hand and
small enough to verify by inspection.

The rules' *evidence* is not tested; that lives in docs/rules-baseline.md and
is a claim about the dataset, not about this code. What is tested is that the
machinery awards the points it says it awards, that fit() cannot reach beyond
what it was given, and that the weighting is coherent.
"""

import numpy as np
import pandas as pd
import pytest

from fraud_engine.models.rules import (
    Rule,
    amount_percentile,
    build_rules,
    contributions,
    fit,
    score,
)

RULES_CFG = {
    "round_amount": {"step": 50, "min": 150, "max": 500, "weight": 2},
    "product_tier": {"C": 3, "S": 2, "H": 1, "R": 0, "W": 0},
    "amount_percentile": {"quantile": 0.99, "weight": 1},
    "new_card": {"d1_max": 3, "weight": 3},
    "w_m4_m2": {"weight": 1},
    "amount_tiebreaker": 0.5,
    "amount_ecdf_points": 101,
}


def make_frame(rows: list[dict]) -> pd.DataFrame:
    """A frame carrying exactly the columns the rules require."""
    frame = pd.DataFrame(rows)
    frame["ProductCD"] = frame["ProductCD"].astype("category")
    return frame


def base_row(**overrides) -> dict:
    """A transaction that fires nothing: product W, plain amount, old card."""
    row = {"TransactionAmt": 77.0, "ProductCD": "W", "D1": 400.0, "M4": "M0"}
    row.update(overrides)
    return row


# Amounts run to $1,000 per product, so the fitted p99 lands near $990 and the
# band amounts used below ($100-$550) sit safely under it. Otherwise a test
# aimed at one rule would silently fire amount_percentile as well.
TRAIN_MAX_AMOUNT = 1_000


@pytest.fixture
def constants():
    """Constants fitted on a spread of amounts across every product."""
    train = make_frame(
        [
            base_row(TransactionAmt=float(amount), ProductCD=product)
            for product in "CHRSW"
            for amount in range(1, TRAIN_MAX_AMOUNT + 1)
        ]
    )
    return fit(train, RULES_CFG)


@pytest.fixture
def rules():
    return build_rules(RULES_CFG)


# ---- fit ---------------------------------------------------------------------


def test_fit_produces_a_cut_point_per_product(constants):
    assert set(constants["amount_p99"]) == set("CHRSW")


def test_fit_reaches_nothing_beyond_the_frame_it_was_given(rules):
    """The train-only boundary, as a test rather than a convention.

    Cut points fitted on cheap rows must not move when expensive rows exist
    elsewhere — if they did, fit() would be reading beyond its window.
    """
    cheap = make_frame([base_row(TransactionAmt=float(a), ProductCD="W") for a in range(1, 101)])
    fitted = fit(cheap, RULES_CFG)

    expensive = make_frame([base_row(TransactionAmt=99_999.0, ProductCD="W")])
    refitted = fit(pd.concat([cheap, expensive], ignore_index=True), RULES_CFG)

    assert fitted["amount_p99"]["W"] < refitted["amount_p99"]["W"]


def test_fit_rejects_an_empty_frame():
    empty = pd.DataFrame({"TransactionAmt": pd.Series(dtype=float), "ProductCD": []})
    with pytest.raises(ValueError, match="empty"):
        fit(empty, RULES_CFG)


def test_fit_rejects_a_frame_missing_a_column():
    with pytest.raises(ValueError, match="TransactionAmt"):
        fit(pd.DataFrame({"ProductCD": ["W"]}), RULES_CFG)


# ---- predicates: each rule awards its own weight and nothing else ------------


@pytest.mark.parametrize(
    ("overrides", "expected"),
    [
        ({}, {}),
        ({"TransactionAmt": 300.0}, {"round_amount": 2}),
        ({"ProductCD": "C"}, {"product_tier": 3}),
        ({"ProductCD": "S"}, {"product_tier": 2}),
        ({"D1": 2.0}, {"new_card": 3}),
        ({"M4": "M2"}, {"w_m4_m2": 1}),
    ],
)
def test_each_rule_awards_its_weight_in_isolation(rules, constants, overrides, expected):
    frame = make_frame([base_row(**overrides)])
    awarded = contributions(frame, rules, constants).iloc[0].to_dict()
    assert awarded == {rule.name: expected.get(rule.name, 0) for rule in rules}


def test_round_amount_respects_the_band_edges(rules, constants):
    """$100 is a $50 multiple and must NOT fire — the band is the hypothesis."""
    frame = make_frame([base_row(TransactionAmt=a) for a in (100.0, 150.0, 500.0, 550.0)])
    fired = contributions(frame, rules, constants)["round_amount"].to_list()
    assert fired == [0, 2, 2, 0]


def test_new_card_excludes_d1_zero(rules, constants):
    """D1 == 0 covers half the data at barely any lift; the rule starts at 1."""
    frame = make_frame([base_row(D1=d) for d in (0.0, 1.0, 3.0, 4.0)])
    assert contributions(frame, rules, constants)["new_card"].to_list() == [0, 3, 3, 0]


def test_w_m4_m2_does_not_fire_outside_product_w(rules, constants):
    """Unscoped, M4=='M2' is 94% product C and restates product_tier."""
    frame = make_frame([base_row(ProductCD=p, M4="M2") for p in ("W", "C")])
    assert contributions(frame, rules, constants)["w_m4_m2"].to_list() == [1, 0]


def test_a_product_unseen_in_train_scores_zero_rather_than_raising(rules, constants):
    """At serving time a new product code is ordinary, not a reason to fail."""
    frame = make_frame([base_row(ProductCD="Z", TransactionAmt=99_999.0)])
    awarded = contributions(frame, rules, constants).iloc[0]
    assert awarded["product_tier"] == 0
    assert awarded["amount_percentile"] == 0


def test_nulls_do_not_leak_into_the_score(rules, constants):
    """A null must mean 'did not fire', never NaN — NaN poisons the ranking."""
    frame = make_frame([base_row(D1=np.nan, M4=None, TransactionAmt=77.0)])
    assert score(frame, rules, constants).notna().all()


# ---- the machinery -----------------------------------------------------------


def test_score_is_the_weighted_sum_plus_the_tiebreaker(rules, constants):
    frame = make_frame([base_row(TransactionAmt=300.0, ProductCD="C", D1=2.0)])
    awarded = contributions(frame, rules, constants)
    expected_points = 2 + 3 + 3  # round_amount + product_tier + new_card

    assert awarded.sum(axis=1).iloc[0] == expected_points
    tiebreak = score(frame, rules, constants).iloc[0] - expected_points
    assert 0 <= tiebreak <= RULES_CFG["amount_tiebreaker"]


def test_the_tiebreaker_can_never_outweigh_a_rule(rules, constants):
    """A one-point rule must beat any amount, or the score stops being rules."""
    # Below the fitted p99, so the amount fires no rule of its own and the
    # comparison isolates the tiebreaker.
    plain_but_huge = make_frame([base_row(TransactionAmt=TRAIN_MAX_AMOUNT - 50.0)])
    cheap_but_flagged = make_frame([base_row(TransactionAmt=1.0, M4="M2")])
    assert (
        score(cheap_but_flagged, rules, constants).iloc[0]
        > score(plain_but_huge, rules, constants).iloc[0]
    )


def test_scoring_one_row_matches_scoring_it_in_a_batch(rules, constants):
    """The serving case. A fitted ECDF is why this holds; a batch rank would not."""
    rows = [base_row(TransactionAmt=a, ProductCD=p) for a, p in ((300.0, "C"), (77.0, "W"))]
    batch = score(make_frame(rows), rules, constants)
    alone = [score(make_frame([row]), rules, constants).iloc[0] for row in rows]
    assert batch.to_list() == pytest.approx(alone)


def test_contributions_rejects_an_empty_rule_set(constants):
    with pytest.raises(ValueError, match="no rules"):
        contributions(make_frame([base_row()]), (), constants)


def test_contributions_rejects_a_frame_missing_a_required_column(rules, constants):
    frame = make_frame([base_row()]).drop(columns=["D1"])
    with pytest.raises(ValueError, match="D1"):
        contributions(frame, rules, constants)


def test_a_misaligned_predicate_fails_loudly(constants):
    """A rule returning the wrong length would otherwise produce a NaN report."""
    broken = Rule(
        name="broken",
        provenance="search",
        rationale="returns fewer rows than it was given",
        points=lambda frame, constants: pd.Series([1], index=[0]),
    )
    frame = make_frame([base_row(), base_row()])
    with pytest.raises(ValueError, match="index does not match"):
        contributions(frame, (broken,), constants)


def test_amount_percentile_is_monotone_and_bounded(constants):
    amounts = pd.Series([0.0, 10.0, 50.0, 150.0, 1e6])
    percentiles = amount_percentile(amounts, constants)
    assert percentiles.is_monotonic_increasing
    assert ((percentiles >= 0) & (percentiles <= 1)).all()


# ---- coherence ---------------------------------------------------------------


def test_every_rule_declares_a_provenance_and_a_rationale(rules):
    """A rule cannot enter the engine without saying where it came from."""
    assert len(rules) == 5
    for rule in rules:
        assert rule.provenance in ("hypothesis", "search")
        assert len(rule.rationale) > 40


def test_score_is_monotone_in_the_points_awarded(rules, constants):
    """Weight coherence: more rules fired must never mean a lower score.

    Cheap here, and the thing that catches an incoherent weighting before it
    reaches a report. The equivalent check against fraud RATE needs data and
    belongs in the report, not in CI.
    """
    frame = make_frame(
        [
            base_row(),
            base_row(M4="M2"),
            base_row(D1=2.0),
            base_row(D1=2.0, TransactionAmt=300.0),
            base_row(D1=2.0, TransactionAmt=300.0, ProductCD="C"),
        ]
    )
    awarded = contributions(frame, rules, constants).sum(axis=1)
    scored = score(frame, rules, constants)
    assert awarded.is_monotonic_increasing
    assert scored.is_monotonic_increasing
