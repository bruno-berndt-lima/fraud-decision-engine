"""Why a transaction was declined, produced off the request path.

`docs/serving.md` §3, and the arithmetic that closed it: the EV policy blocks 6.45% of
transactions and a contribution call costs around 900 ms, so explaining declines inside
`/score` would put a 900 ms call on more than one request in twenty — and a p95 is exactly
the percentile that lands inside a 6.45% tail. The explanation is therefore its own
endpoint, with its own budget, which is **not** `problem-statement.md` §3.1's: §3.1 prices
an authorisation, and an adverse-action notice is not one.

**Stateless, like everything else here.** The caller sends the record again rather than a
reference to a decision the service kept, because this process holds no store — the same
reason it reports review *eligibility* rather than promising a review. A notice is owed
when it is asked for, and whoever asks has the record.

**No fallback, and the reason is not resignation.** A decision that cannot be made by the
model is better made by the incumbent than not made at all; an explanation that cannot be
produced has no such substitute, because an invented one is a false statement to a
customer. There is also nothing to explain in the degraded mode: the rules engine never
blocks (`decision-policy.md` §3), so a service in fallback issues no adverse decisions.

**The serving path still never imports `shap`.** Contributions come from the booster's own
`pred_contrib`, exactly as `explain/contributions.py` takes them — that module cannot be
imported here because it reaches for the feature registry and MLflow, so the one call it
would have lent is made directly.
"""

from __future__ import annotations

import logging
from collections.abc import Mapping

import numpy as np

from fraud_engine.evaluation.cost import Costs
from fraud_engine.explain.codes import Dictionary, ReasonCodes, reason_codes
from fraud_engine.models.calibrate import apply_calibrator
from fraud_engine.serving.artifacts import Model
from fraud_engine.serving.fast import Layout, row

log = logging.getLogger(__name__)

# `explainability.md` §2's guard, restated rather than imported: the module holding it
# reaches for the feature registry, which the serving path may not. Relative to the row's
# contribution mass, and loose on purpose — it detects a wrong object, not a precision
# claim, and `shap`'s own bound is the same order.
ADDITIVITY_TOLERANCE = 1e-2


def check_additive(contributions: np.ndarray, margin: float) -> None:
    """Refuse contributions that do not add up to the prediction they decompose.

    What this catches is an explanation attached to the wrong row or the wrong column
    order — which would put a true sentence about someone else's transaction into a
    decline notice, and raise nothing.

    Raises:
        ValueError: If the contributions and the base value miss the margin by more than
            the tolerance, relative to the mass being summed.
    """
    total = float(contributions.sum())
    mass = float(np.abs(contributions).sum())
    drift = abs(total - margin) / max(mass, 1.0)

    if drift > ADDITIVITY_TOLERANCE:
        raise ValueError(
            f"contributions sum to {total:.6f} against a margin of {margin:.6f} "
            f"({drift:.2%} of the mass); this is not an explanation of this prediction"
        )


def explain(
    model: Model,
    layout: Layout,
    values: Mapping,
    costs: Costs,
    dictionary: Dictionary,
    top_k: int,
    load_cfg: dict,
    features_cfg: dict,
) -> ReasonCodes:
    """Everything a person can be told about one decision.

    Three predictions, and each is a different question: the calibrated probability the
    policy decided on, the raw margin the contributions have to add up to, and the
    contributions themselves. The first two cost about twenty milliseconds each against
    the third's nine hundred, which is what makes the check worth making.

    Args:
        model: From `artifacts.load_model`.
        layout: From `fast.build_layout`, for this model.
        values: The request's fields, as `/score` takes them.
        costs: The loaded cost matrix.
        dictionary: From `explain.codes.load_dictionary`.
        top_k: Contributors to report, from `explain.top_k`.
        load_cfg: The `load` block of `config.yaml`.
        features_cfg: The `features` block.

    Returns:
        The decision, the bar it was taken against, and what argued for it.

    Raises:
        ValueError: If the contributions do not decompose the prediction.
    """
    features = row(values, model, layout, load_cfg, features_cfg)

    probability = apply_calibrator(
        model.calibrator, model.booster.predict(features, num_threads=model.threads)
    )
    margin = float(model.booster.predict(features, raw_score=True, num_threads=model.threads)[0])
    decomposed = model.booster.predict(features, pred_contrib=True, num_threads=model.threads)[0]

    check_additive(decomposed, margin)

    return reason_codes(
        contributions=decomposed[:-1],
        columns=model.columns,
        probability=float(probability[0]),
        amount=float(values["TransactionAmt"]),
        costs=costs,
        dictionary=dictionary,
        top_k=top_k,
    )
