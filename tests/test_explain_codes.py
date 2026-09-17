"""Tests for what a declined customer is told.

The guards here are the ones that would otherwise fail silently and in public: a
dictionary that has drifted from the model, a notice repeating one sentence three
times, and "this looked safe because..." appearing in a decline.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from fraud_engine.evaluation.cost import ALLOW, BLOCK, Costs
from fraud_engine.explain.codes import (
    Dictionary,
    check_covered,
    load_dictionary,
    reason_codes,
    summarise,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
COSTS = Costs(chargeback_fee=25.0, false_positive=15.0, review=1.5, review_friction=3.0, version=1)

COLUMNS = ["TransactionAmt", "C13", "freq_card1", "C1"]
DICTIONARY = Dictionary(
    version=1,
    generic="an indicator the provider does not define",
    generic_many="several indicators the provider does not define",
    features={"TransactionAmt": "the amount is unusual", "freq_card1": "this card is new here"},
)


# ------------------------------------------------------------------------------
# the committed dictionary
# ------------------------------------------------------------------------------


def test_the_committed_dictionary_describes_every_nameable_column():
    """The registry is the guard, and it lives here because serving has no matrices."""
    from fraud_engine.features.registry import TIER_1, TIER_2, TIER_3

    dictionary = load_dictionary(REPO_ROOT / "config" / "reason_codes.yaml")
    nameable = set(TIER_1) | set(TIER_2) | set(TIER_3)

    assert set(dictionary.features) == nameable


def test_the_generic_has_a_form_for_more_than_one():
    dictionary = load_dictionary(REPO_ROOT / "config" / "reason_codes.yaml")

    assert dictionary.generic and dictionary.generic_many
    assert dictionary.generic != dictionary.generic_many


def test_the_sentences_are_readable_as_reasons_not_as_column_names():
    """A phrase that is just the column back again explains nothing."""
    dictionary = load_dictionary(REPO_ROOT / "config" / "reason_codes.yaml")

    for column, phrase in dictionary.features.items():
        assert column.lower() not in phrase.lower()
        assert len(phrase.split()) >= 4


# ------------------------------------------------------------------------------
# check_covered
# ------------------------------------------------------------------------------


def test_a_nameable_column_with_no_sentence_stops_the_run():
    with pytest.raises(ValueError, match="without a sentence"):
        check_covered(COLUMNS, {"TransactionAmt", "freq_card1", "card6"}, DICTIONARY)


def test_a_sentence_for_a_column_the_model_lost_stops_the_run():
    with pytest.raises(ValueError, match="does not have"):
        check_covered(["C13", "C1"], set(), DICTIONARY)


# ------------------------------------------------------------------------------
# reason_codes
# ------------------------------------------------------------------------------


def codes_for(contributions, probability, amount, top_k=3):
    return reason_codes(
        np.array(contributions, dtype="float64"),
        COLUMNS,
        probability,
        amount,
        COSTS,
        DICTIONARY,
        top_k,
    )


def test_a_decline_reports_the_bar_it_was_taken_against():
    codes = codes_for([1.0, 2.0, 0.5, -3.0], probability=0.5, amount=100.0)

    assert codes.decision == BLOCK
    assert codes.adverse
    assert codes.break_even == pytest.approx(15.0 / (100.0 + 25.0 + 15.0))


def test_a_small_amount_faces_a_higher_bar_than_a_large_one():
    """The whole reason a single threshold was not used."""
    small = codes_for([1.0, 0.0, 0.0, 0.0], probability=0.2, amount=20.0)
    large = codes_for([1.0, 0.0, 0.0, 0.0], probability=0.2, amount=5000.0)

    assert small.break_even > large.break_even
    assert small.decision == ALLOW and large.decision == BLOCK


def test_an_allowed_transaction_is_not_owed_an_explanation():
    codes = codes_for([1.0, 0.0, 0.0, 0.0], probability=0.001, amount=20.0)

    assert codes.decision == ALLOW
    assert not codes.adverse


def test_only_the_contributors_arguing_for_the_decline_are_listed():
    """Ranking by absolute value would put "this looked safe because" in a decline."""
    codes = codes_for([0.5, -9.0, 0.2, -0.1], probability=0.5, amount=100.0)

    assert [reason.feature for reason in codes.reasons] == ["TransactionAmt", "freq_card1"]


def test_at_most_top_k_reasons():
    codes = codes_for([1.0, 2.0, 3.0, 4.0], probability=0.5, amount=100.0, top_k=2)

    assert len(codes.reasons) == 2


def test_an_unnameable_contributor_gets_the_generic_and_is_marked():
    codes = codes_for([0.1, 5.0, 0.0, 0.0], probability=0.5, amount=100.0)
    leading = codes.reasons[0]

    assert leading.feature == "C13"
    assert leading.phrase == DICTIONARY.generic
    assert not leading.named


def test_the_cost_matrix_version_rides_along():
    """A later matrix silently changes what customers are told; the notice says which."""
    assert codes_for([1.0, 0.0, 0.0, 0.0], 0.5, 100.0).cost_matrix_version == 1


def test_review_eligibility_is_reported_not_the_review():
    codes = codes_for([1.0, 0.0, 0.0, 0.0], probability=0.5, amount=100.0)

    assert isinstance(codes.review_eligible, bool)
    assert not hasattr(codes, "reviewed")


def test_more_than_one_row_is_refused():
    with pytest.raises(ValueError, match="one transaction at a time"):
        reason_codes(np.zeros((2, 4)), COLUMNS, 0.5, 100.0, COSTS, DICTIONARY)


# ------------------------------------------------------------------------------
# statements
# ------------------------------------------------------------------------------


def test_one_unnameable_contributor_reads_in_the_singular():
    codes = codes_for([2.0, 1.0, 0.0, 0.0], probability=0.5, amount=100.0)

    assert codes.statements(DICTIONARY) == ("the amount is unusual", DICTIONARY.generic)


def test_several_unnameable_contributors_collapse_into_one_line():
    """A notice printing the same sentence three times reads as a fault, not a limit."""
    codes = codes_for([0.1, 3.0, 0.0, 2.0], probability=0.5, amount=100.0)

    assert codes.statements(DICTIONARY) == (DICTIONARY.generic_many, "the amount is unusual")


def test_the_collapsed_line_keeps_the_place_the_first_one_held():
    codes = codes_for([5.0, 3.0, 0.0, 2.0], probability=0.5, amount=100.0)

    assert codes.statements(DICTIONARY)[0] == "the amount is unusual"


def test_the_audit_trail_keeps_every_contributor_the_notice_collapsed():
    """Three contributors argued for this decline; the customer reads two lines."""
    codes = codes_for([0.1, 3.0, 0.0, 2.0], probability=0.5, amount=100.0)

    assert [reason.feature for reason in codes.reasons] == ["C13", "C1", "TransactionAmt"]
    assert len(codes.statements(DICTIONARY)) == 2


# ------------------------------------------------------------------------------
# summarise
# ------------------------------------------------------------------------------


def test_the_share_of_declines_with_nothing_but_the_generic():
    generic = codes_for([0.0, 3.0, 0.0, 2.0], probability=0.5, amount=100.0)
    partly = codes_for([3.0, 2.0, 0.0, 0.0], probability=0.5, amount=100.0)
    tiers = {"TransactionAmt": "tier_1", "C13": "tier_0", "freq_card1": "tier_2", "C1": "tier_0"}

    summary = summarise([generic, partly], tiers)

    assert summary["declines"] == 2
    assert summary["fully_generic"] == 0.5
    assert summary["at_least_one_named"] == 0.5
    assert summary["leading_reason_unnameable"] == 0.5
