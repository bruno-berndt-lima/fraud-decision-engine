"""One transaction, from request fields to an action.

`docs/serving.md` §6, over the policy `decision-policy.md` §2 registered and Phase 06
froze. Nothing here chooses anything: the expected costs, the break-even and the review
gain are `evaluation/cost.py`'s, applied to one row instead of to a slice.

**The row is assembled by `fast.py`, which `serving.md` §5's amendment permits and §8's
gate polices.** `transform.py` is still the definition of what the row *is*; what runs on
the request path is the arithmetic that produces the same one in a tenth of the time.

**The probability is calibrated before it is priced.** The policy multiplies a
probability by money, and this booster's raw output is not a frequency — `calibrator.json`
is what makes it one, and a service that skipped it would block transactions the cost
matrix says to allow, silently.

**Review is eligibility and a priority, never an outcome.** Whether a transaction reaches
an analyst depends on the other transactions that day, and a service deciding one request
cannot see them. So the response carries what a queue needs to rank by and stops there.

**The fallback decides differently, and says so.** The rules engine emits points, not a
probability, so the per-transaction threshold cannot be applied to it at all
(`cost.rules_policy`): it never blocks, and everything it scores is a review candidate
ranked by its points. A caller reading `mode` knows which of the two it was given.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

import numpy as np
import pandas as pd

from fraud_engine.evaluation.cost import (
    ALLOW,
    BLOCK,
    REVIEW,
    Costs,
    break_even,
    expected_costs,
)
from fraud_engine.models.calibrate import apply_calibrator
from fraud_engine.models.rules import REQUIRED_COLUMNS
from fraud_engine.models.rules import score as rules_score
from fraud_engine.serving.artifacts import Fallback, Model
from fraud_engine.serving.fast import Layout, carried, coverage, history_supplied, row

MODEL, RULES = "model", "rules"

# The fallback allows everything it does not hand to an analyst — `decision-policy.md` §3.
# That asymmetry is the accepted cost of fail-open, not an oversight.
FALLBACK_DECISION = ALLOW


@dataclass(frozen=True)
class Verdict:
    """What one request is answered with.

    `probability` is `None` under the rules engine rather than a number: points are not a
    frequency, and reporting them in a field a caller reads as one would be the unit
    confusion the project's cost model is built to avoid.
    """

    decision: str
    probability: float | None
    break_even: float
    amount: float
    review_eligible: bool
    review_priority: float
    mode: str
    inherited_present: int
    inherited_expected: int
    history_supplied: tuple[str, ...]


def calibrated(model: Model, features: pd.DataFrame) -> np.ndarray:
    """The booster's score, made a probability.

    `num_iteration` is not passed: `model.txt` was written truncated at the peak, so the
    file holds those trees and no others — `explain/contributions.py` records the same.
    `num_threads` is, for the reason `Model` gives.
    """
    return apply_calibrator(
        model.calibrator, model.booster.predict(features, num_threads=model.threads)
    )


def decide(
    model: Model,
    layout: Layout,
    values: Mapping,
    costs: Costs,
    load_cfg: dict,
    features_cfg: dict,
) -> Verdict:
    """Score one transaction and price the actions against each other.

    Args:
        model: From `artifacts.load_model`.
        layout: From `fast.build_layout`, for this model.
        values: The request's fields. An absent key and a null are the same thing (§1).
        costs: The loaded cost matrix.
        load_cfg: The `load` block of `config.yaml`.
        features_cfg: The `features` block.

    Returns:
        The action, the bar it was taken against, and what it was decided on.
    """
    features = row(values, model, layout, load_cfg, features_cfg)

    probability = calibrated(model, features)
    amount = np.array([float(values["TransactionAmt"])], dtype="float64")

    expected = expected_costs(probability, amount, costs)
    blocked = bool(expected[BLOCK][0] < expected[ALLOW][0])
    gain = float(min(expected[ALLOW][0], expected[BLOCK][0]) - expected[REVIEW][0])

    present, total = coverage(values, model.columns)

    return Verdict(
        decision=BLOCK if blocked else ALLOW,
        probability=float(probability[0]),
        break_even=float(break_even(amount, costs)[0]),
        amount=float(amount[0]),
        review_eligible=gain > 0,
        review_priority=gain,
        mode=MODEL,
        inherited_present=present,
        inherited_expected=total,
        history_supplied=history_supplied(values),
    )


def fallback_frame(values: Mapping) -> pd.DataFrame:
    """The four fields the incumbent reads, typed as its predicates expect them.

    Built here rather than through `transform.typed`: the engine needs no vocabulary, and
    a frame of four columns is not worth the machinery that types four hundred.
    """
    numeric = {
        name: [float(values[name]) if carried(values, name) else np.nan]
        for name in ("TransactionAmt", "D1")
    }
    strings = {name: [values.get(name)] for name in ("ProductCD", "M4")}

    return pd.DataFrame(numeric | strings)


def decide_with_rules(fallback: Fallback, values: Mapping, costs: Costs) -> Verdict:
    """The incumbent's answer, for when the model cannot give one.

    Everything is a review candidate, ranked by the engine's points: `rules_policy` masks
    nothing, because an engine that cannot price a transaction cannot say which ones are
    not worth an analyst's time. The break-even is still reported — it is a property of
    the amount and the cost matrix, not of the model — so a caller's downstream logic does
    not have to branch on which mode answered.

    Args:
        fallback: From `artifacts.load_fallback`.
        values: The request's fields.
        costs: The loaded cost matrix.

    Returns:
        A verdict in `RULES` mode, carrying no probability.
    """
    raw = fallback_frame(values)
    amount = raw["TransactionAmt"].to_numpy(dtype="float64")
    points = rules_score(raw, fallback.rules, fallback.constants)

    # The coverage reported is of what *this* path read. Reporting the model's own would
    # describe inputs nothing in this mode looks at, and zero would read as "nothing
    # arrived" rather than "four fields were needed".
    present = sum(carried(values, name) for name in REQUIRED_COLUMNS)

    return Verdict(
        decision=FALLBACK_DECISION,
        probability=None,
        break_even=float(break_even(amount, costs)[0]),
        amount=float(amount[0]),
        review_eligible=True,
        review_priority=float(points.iloc[0]),
        mode=RULES,
        inherited_present=present,
        inherited_expected=len(REQUIRED_COLUMNS),
        history_supplied=(),
    )
