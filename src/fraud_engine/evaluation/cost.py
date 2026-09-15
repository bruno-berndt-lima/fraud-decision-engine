"""What a decision costs, in USD, under `config/cost_matrix.yaml`.

The policy registered in `docs/decision-policy.md` §2, starting with its two-action
core: allow or block, whichever has the lower expected cost. Review and its daily
capacity build on these functions.

One unit of account. `TransactionAmt` is USD and so is every cost, and the break-even
probability is a fixed cost over a variable amount — mixed units would rescale it and
block the wrong transactions, so the currency is asserted at load.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

CURRENCY = "USD"

ALLOW = "allow"
BLOCK = "block"
REVIEW = "review"

# The derivation in decision-policy.md §2 assumes these. A matrix that changed them
# would be silently ignored rather than applied, so a change is refused at load.
FALSE_NEGATIVE_FORMULA = "amount + chargeback_fee"
ZERO_COSTS = ("true_positive", "true_negative")


@dataclass(frozen=True)
class Costs:
    """The version-1 cost matrix as numbers. `dataclasses.replace` varies one for a sweep."""

    chargeback_fee: float
    false_positive: float
    review: float
    review_friction: float
    version: int


def load_costs(cost_matrix: dict) -> Costs:
    """Costs from the parsed `cost_matrix.yaml`.

    Takes the parsed mapping rather than a path, like `report.load_capacities`.

    Raises:
        ValueError: If the currency is not USD, the false-negative formula or a zero
            cost differs from what the policy was derived under, or a cost is negative.
    """
    if cost_matrix.get("currency") != CURRENCY:
        raise ValueError(
            f"cost matrix currency is {cost_matrix.get('currency')!r}, not {CURRENCY!r}. "
            "TransactionAmt is USD; a threshold built from mixed units blocks the wrong "
            "transactions."
        )

    costs = cost_matrix["costs"]

    formula = costs["false_negative"]["formula"]
    if formula != FALSE_NEGATIVE_FORMULA:
        raise ValueError(
            f"false_negative formula is {formula!r}; the policy is derived for "
            f"{FALSE_NEGATIVE_FORMULA!r}"
        )

    for name in ZERO_COSTS:
        if costs[name]["value"] != 0:
            raise ValueError(f"{name} is {costs[name]['value']}; the policy assumes 0")

    loaded = Costs(
        chargeback_fee=float(costs["chargeback_fee"]["value"]),
        false_positive=float(costs["false_positive"]["value"]),
        review=float(costs["review"]["value"]),
        review_friction=float(costs["review_friction"]["value"]),
        version=int(cost_matrix["version"]),
    )

    negative = [name for name, value in vars(loaded).items() if value < 0]
    if negative:
        raise ValueError(f"negative costs: {negative}")

    return loaded


def check_inputs(p: np.ndarray, amount: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Probabilities in [0, 1] and non-negative amounts, as float arrays of one length.

    A NaN would compare false against every threshold and quietly allow the
    transaction, so it is refused rather than decided.

    Raises:
        ValueError: On a length mismatch, a non-finite value, a probability outside
            [0, 1] or a negative amount.
    """
    p = np.asarray(p, dtype="float64")
    amount = np.asarray(amount, dtype="float64")

    if p.shape != amount.shape:
        raise ValueError(f"{p.shape[0]} probabilities for {amount.shape[0]} amounts")
    if not (np.isfinite(p).all() and np.isfinite(amount).all()):
        raise ValueError("probabilities and amounts must be finite")
    if ((p < 0) | (p > 1)).any():
        raise ValueError("probabilities must lie in [0, 1]")
    if (amount < 0).any():
        raise ValueError("amounts must be non-negative")

    return p, amount


def break_even(amount: np.ndarray, costs: Costs) -> np.ndarray:
    """The probability above which blocking is cheaper than allowing.

    `false_positive / (amount + chargeback_fee + false_positive)`. The trailing
    `false_positive` is there because a block only costs anything when the
    transaction is legitimate.
    """
    amount = np.asarray(amount, dtype="float64")
    return costs.false_positive / (amount + costs.chargeback_fee + costs.false_positive)


def expected_costs(p: np.ndarray, amount: np.ndarray, costs: Costs) -> dict[str, np.ndarray]:
    """Expected USD cost of each action, per transaction, given probability `p`."""
    p, amount = check_inputs(p, amount)

    return {
        ALLOW: p * (amount + costs.chargeback_fee),
        BLOCK: (1 - p) * costs.false_positive,
        REVIEW: costs.review + (1 - p) * costs.review_friction,
    }


def allow_or_block(p: np.ndarray, amount: np.ndarray, costs: Costs) -> np.ndarray:
    """The cheaper of allow and block in expectation. No review.

    A tie allows: at exactly break-even the two cost the same, and declining a
    customer for nothing is the outcome the cost matrix calls expensive.
    """
    expected = expected_costs(p, amount, costs)
    return np.where(expected[BLOCK] < expected[ALLOW], BLOCK, ALLOW)


def realised_cost(
    action: np.ndarray, y: np.ndarray, amount: np.ndarray, costs: Costs
) -> np.ndarray:
    """USD each decision actually cost, given the label.

    Raises:
        ValueError: If an action is not allow, block or review, or a label is not 0/1.
    """
    action = np.asarray(action)
    y = np.asarray(y)
    amount = np.asarray(amount, dtype="float64")

    unknown = sorted(set(np.unique(action)) - {ALLOW, BLOCK, REVIEW})
    if unknown:
        raise ValueError(f"unknown actions: {unknown}")
    if not np.isin(y, (0, 1)).all():
        raise ValueError("labels must be 0 or 1")

    return np.select(
        [action == ALLOW, action == BLOCK],
        [y * (amount + costs.chargeback_fee), (1 - y) * costs.false_positive],
        default=costs.review + (1 - y) * costs.review_friction,
    )


def usd_per_thousand(cost: np.ndarray) -> float:
    """Total realised cost scaled to 1,000 transactions — the headline unit."""
    cost = np.asarray(cost, dtype="float64")
    if cost.size == 0:
        raise ValueError("no transactions to cost")
    return float(cost.sum() / cost.size * 1_000)
