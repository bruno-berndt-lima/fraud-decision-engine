"""Tests for the amount family.

Two kinds of test here, and they are doing different jobs.

Most of them pin float behaviour. Every column in this family is a divisibility
question, floats cannot answer those, and the failure mode is silent: an amount
that should count as round quietly does not, the feature loses a few percent of
its rows, and nothing raises.

The rest pin H2's *shape* rather than its strength. The falsifier and the claim
have to remain distinguishable — a band that admits everything, or an
interaction that ignores the product, would leave the hypothesis untestable
while every arithmetic test still passed.
"""

from pathlib import Path

import pandas as pd
import pytest
import yaml

from fraud_engine.features.amounts import (
    COLUMNS,
    ROUND_BAND,
    ROUND_BAND_PRODUCT,
    WHOLE_DOLLAR,
    add_amount_features,
    is_round_in_band,
    is_whole_dollar,
    thousandths,
)

REPO_ROOT = Path(__file__).resolve().parents[1]

CFG = {"round_step": 50, "round_min": 150, "round_max": 500, "round_product": "H"}


def amounts(*values) -> pd.Series:
    return pd.Series(values, dtype="float64")


# ------------------------------------------------------------------------------
# thousandths — the float safety net
# ------------------------------------------------------------------------------
@pytest.mark.parametrize(
    ("amount", "expected"),
    [
        (1.0, 1000),
        (49.99, 49990),
        # The same number, spelled the way a float actually stores it.
        (49.989999999999995, 49990),
        (0.1 + 0.2, 300),
        (150.0, 150000),
        (0.001, 1),
    ],
)
def test_an_amount_becomes_whole_thousandths(amount, expected):
    assert thousandths(amounts(amount)).iloc[0] == expected


def test_the_result_is_an_integer_type():
    """Divisibility on a float dtype is the bug this module exists to avoid, so
    the conversion has to actually leave the float domain."""
    assert thousandths(amounts(1.0, 49.99)).dtype == "int64"


# ------------------------------------------------------------------------------
# is_whole_dollar — H2's falsifier
# ------------------------------------------------------------------------------
@pytest.mark.parametrize(
    ("amount", "expected"),
    [(1.0, True), (150.0, True), (49.99, False), (0.1 + 0.2, False), (49.999, False)],
)
def test_whole_dollars_are_recognised_through_float_error(amount, expected):
    assert bool(is_whole_dollar(amounts(amount)).iloc[0]) is expected


# ------------------------------------------------------------------------------
# is_round_in_band — H2's claim
# ------------------------------------------------------------------------------
@pytest.mark.parametrize(
    ("amount", "expected"),
    [
        (150.0, True),  # lower bound, inclusive
        (300.0, True),
        (500.0, True),  # upper bound, inclusive
        (100.0, False),  # a $50 multiple below the band
        (550.0, False),  # a $50 multiple above the band
        (175.0, False),  # inside the band, not a multiple
        (150.01, False),  # inside the band, not whole
    ],
)
def test_the_band_admits_exactly_its_own_multiples(amount, expected):
    assert bool(is_round_in_band(amounts(amount), 50, 150, 500).iloc[0]) is expected


def test_the_band_is_narrower_than_plain_roundness():
    """H2 stands or falls on these being different populations. If the band ever
    admitted every whole dollar, the falsifier and the claim would be the same
    feature and the hypothesis could not fail."""
    values = amounts(100.0, 150.0, 175.0, 300.0, 550.0, 49.99)

    band = is_round_in_band(values, 50, 150, 500)
    assert (band <= is_whole_dollar(values)).all()
    assert band.sum() < is_whole_dollar(values).sum()


# ------------------------------------------------------------------------------
# add_amount_features
# ------------------------------------------------------------------------------
def frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "TransactionAmt": [300.0, 300.0, 100.0, 49.99],
            "ProductCD": ["H", "W", "H", "H"],
        }
    )


def test_the_family_arrives_null_free_as_int8():
    """`build_features` promises families make their own fill decisions, so the
    probe's median imputer never fires on them."""
    built = add_amount_features(frame(), CFG)

    assert not built[list(COLUMNS)].isna().to_numpy().any()
    assert all(built[column].dtype == "int8" for column in COLUMNS)


def test_the_interaction_needs_both_the_band_and_the_product():
    """A linear probe reads roundness and product additively, so this column is
    the only place their conjunction exists. It has to actually be a
    conjunction."""
    built = add_amount_features(frame(), CFG)

    assert list(built[ROUND_BAND]) == [1, 1, 0, 0]
    assert list(built[ROUND_BAND_PRODUCT]) == [1, 0, 0, 0]


def test_the_input_frame_is_not_mutated():
    """Families are chained inside `build_features`; one that edits in place
    would make the order they run in matter."""
    original = frame()
    add_amount_features(original, CFG)

    assert not set(COLUMNS) & set(original.columns)


def test_the_product_is_configured_not_assumed():
    built = add_amount_features(frame(), {**CFG, "round_product": "W"})

    assert list(built[ROUND_BAND_PRODUCT]) == [0, 1, 0, 0]


def test_the_committed_config_satisfies_the_family():
    """The literal config the other test modules use is a convenience. This is
    the one place that checks config.yaml actually carries what build.py will
    hand this family."""
    config = yaml.safe_load((REPO_ROOT / "config" / "config.yaml").read_text())

    built = add_amount_features(frame(), config["features"]["amounts"])

    assert set(COLUMNS) <= set(built.columns)
    assert built[WHOLE_DOLLAR].sum() > 0
