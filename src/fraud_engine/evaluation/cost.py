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


# ---- review under a daily capacity ---------------------------------------------

# Absorbs float error in `fraction * rows` — 0.29 * 100 is 28.999…, and a bare floor
# would take a review away from a day that has room for it.
_CAPACITY_TOLERANCE = 1e-9


def daily_capacity(day: np.ndarray, fraction: float) -> dict[int, int]:
    """Reviews each day allows: `floor(fraction * that day's transactions)`.

    Floor, not round, so the constraint is never exceeded (decision-policy.md §2).
    """
    days, rows = np.unique(np.asarray(day), return_counts=True)
    return {
        int(d): int(np.floor(fraction * n + _CAPACITY_TOLERANCE))
        for d, n in zip(days, rows, strict=True)
    }


def review_shares(
    priority: np.ndarray, day: np.ndarray, fraction: float, eligible: np.ndarray | None = None
) -> np.ndarray:
    """How much of each transaction is reviewed, filling each day's capacity by priority.

    1 above the day's cut and 0 below it. The transactions tied at the cut share what
    capacity is left, `r / m` each, so a day reviews exactly its capacity whenever it
    has that many eligible transactions, and the result never depends on row order.

    Args:
        priority: Higher is reviewed first.
        day: Day of each transaction.
        fraction: Review capacity as a fraction of daily volume.
        eligible: Transactions allowed into review at all. Default: every one.

    Returns:
        A share in [0, 1] per transaction.
    """
    priority = np.asarray(priority, dtype="float64")
    day = np.asarray(day)
    eligible = np.ones(len(priority), dtype=bool) if eligible is None else np.asarray(eligible)

    share = np.zeros(len(priority))

    for d, capacity in daily_capacity(day, fraction).items():
        candidates = np.flatnonzero((day == d) & eligible)

        if capacity == 0 or candidates.size == 0:
            continue
        if candidates.size <= capacity:
            share[candidates] = 1.0
            continue

        values = priority[candidates]
        cut = np.sort(values)[::-1][capacity - 1]
        above = values > cut
        tied = values == cut

        share[candidates[above]] = 1.0
        share[candidates[tied]] = (capacity - above.sum()) / tied.sum()

    return share


@dataclass(frozen=True)
class Decision:
    """A policy's output: the action a transaction gets if not reviewed, and how much of it is.

    A share rather than a fourth action because of prorated ties: a transaction on the
    cut is costed as that fraction of a review and the rest of its fallback.
    """

    fallback: np.ndarray
    review_share: np.ndarray

    def cost(self, y: np.ndarray, amount: np.ndarray, costs: Costs) -> np.ndarray:
        """Realised USD per transaction."""
        reviewed = realised_cost(np.full(len(self.fallback), REVIEW), y, amount, costs)
        unreviewed = realised_cost(self.fallback, y, amount, costs)
        return self.review_share * reviewed + (1 - self.review_share) * unreviewed

    def summary(self, y: np.ndarray, amount: np.ndarray, day: np.ndarray, costs: Costs) -> dict:
        """The headline unit, and what it was bought with: review volume and block rate."""
        blocked = (1 - self.review_share) * (self.fallback == BLOCK)
        return {
            "usd_per_1000": usd_per_thousand(self.cost(y, amount, costs)),
            "reviews_per_day": float(self.review_share.sum() / np.unique(day).size),
            "block_rate": float(blocked.mean()),
        }


def ev_policy(
    p: np.ndarray, amount: np.ndarray, day: np.ndarray, costs: Costs, capacity: float
) -> Decision:
    """decision-policy.md §2: allow or block by expected cost, review where it saves most.

    A transaction is eligible for review only if reviewing it is cheaper in expectation
    than the better of allow and block; each day, the largest savings are reviewed.
    """
    expected = expected_costs(p, amount, costs)
    gain = np.minimum(expected[ALLOW], expected[BLOCK]) - expected[REVIEW]

    return Decision(
        fallback=allow_or_block(p, amount, costs),
        review_share=review_shares(gain, day, capacity, eligible=gain > 0),
    )


def rules_policy(points: np.ndarray, day: np.ndarray, capacity: float) -> Decision:
    """decision-policy.md §3: the highest-scoring transactions each day are reviewed.

    The engine never blocks, so everything it does not review is allowed.
    """
    points = np.asarray(points, dtype="float64")
    return Decision(
        fallback=np.full(len(points), ALLOW),
        review_share=review_shares(points, day, capacity),
    )


def naive_policy(p: np.ndarray, threshold: float = 0.5) -> Decision:
    """decision-policy.md §4: block at or above one fixed probability; no review."""
    p = np.asarray(p, dtype="float64")
    return Decision(
        fallback=np.where(p >= threshold, BLOCK, ALLOW),
        review_share=np.zeros(len(p)),
    )
