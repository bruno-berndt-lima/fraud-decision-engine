"""The request contract, and the answer it gets back.

`docs/serving.md` §1 and §6. Pydantic is the contract: a malformed request is a 422 with
a message naming the field, never a 500 from somewhere deeper where the value was finally
used.

**The request model is derived from the model, not typed out.** The fields a request must
be able to carry are the booster's own feature names less what the families build, plus
the V block the reduction consumes — around four hundred of them, and a hand-written list
would drift from the model the first time either changed. `transform.raw_inputs` computes
that set and this builds the contract from it, so the two cannot disagree.

**Omitted is null; misspelled is a 422.** §1 registers the pair and why it is a pair: if a
field may be omitted, then a field whose name is wrong has simply been omitted, and
`C_13` would be scored as a null `C13` that never arrived. Refusing unknown names is what
keeps a caller's typo visible.

**Four fields are required and three of those may not be null**, per §1. The amount is
also refused below zero here rather than by `cost.check_inputs` three calls later, because
a contract that catches it returns a 422 and one that does not returns a 500.
"""

from __future__ import annotations

from collections.abc import Sequence

from pydantic import BaseModel, ConfigDict, Field, create_model

from fraud_engine.serving.transform import NON_NULL_INPUTS, REQUIRED_INPUTS

# What a field that did not arrive looks like in the built model: present, null, and not
# an error. The required ones are given `...` instead, which is Pydantic's "no default".
OMITTED = None

REQUEST_MODEL_NAME = "ScoreRequest"


def field_type(name: str, vocabulary: Sequence[str]) -> tuple[object, object]:
    """The annotation and default for one input.

    Args:
        name: The input's name.
        vocabulary: The columns the category vocabulary covers — those arrive as strings
            and are levelled against it; everything else is a number to the pipeline.

    Returns:
        `(annotation, default)`, in the shape `pydantic.create_model` takes.
    """
    required = name in REQUIRED_INPUTS
    nullable = name not in NON_NULL_INPUTS

    if name == "TransactionDT":
        annotation = int
    elif name == "TransactionAmt":
        annotation = float
    elif name == "has_identity":
        annotation = bool
    elif name in vocabulary:
        annotation = str
    else:
        annotation = float

    if nullable:
        annotation = annotation | None

    if name == "TransactionAmt":
        # Refused here rather than by check_inputs three calls later: the break-even is a
        # fixed cost over an amount, and a negative one is a 422 rather than a 500.
        return annotation, Field(..., ge=0)

    return annotation, (... if required else OMITTED)


def request_model(
    inputs: Sequence[str], vocabulary: Sequence[str], forbid_unknown: bool = True
) -> type[BaseModel]:
    """The Pydantic model for one transaction, built from what the booster reads.

    Args:
        inputs: From `transform.raw_inputs`.
        vocabulary: The category vocabulary's columns.
        forbid_unknown: Whether a field the contract does not have is a 422. True
            whenever the model loaded, which is what makes §1's permissive omission safe.

    **False only in the degraded mode of §4**, where the booster is unavailable and the
    inputs are the four the rules engine reads. A service in fallback cannot know which
    of the model's four hundred names are real, and refusing the ones it does not
    recognise would decline the whole record a caller is still entitled to a decision on
    — which is the revenue outage fail-open exists to prevent. The cost is that a typo
    goes unreported for the duration, and `mode` in the response says that the duration
    is now.

    Returns:
        A model requiring the four §1 names and treating every other absence as a null.
    """
    fields = {name: field_type(name, vocabulary) for name in inputs}

    return create_model(
        REQUEST_MODEL_NAME,
        __config__=ConfigDict(extra="forbid" if forbid_unknown else "ignore"),
        **fields,
    )


class ScoreResponse(BaseModel):
    """What the caller is told, and what it may not read into it.

    `decision` is the action; `review_eligible` and `review_priority` are what a queue
    needs and are never a promise that a review will happen — whether a transaction
    reaches an analyst depends on the other transactions that day, which is queue state
    this service does not hold (§6).

    `inherited_present` and `history_supplied` say what the decision was made *on*. §1
    registers the first: a caller whose integration stops sending the inherited block
    would otherwise get plausible scores and no signal that anything changed.
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
    model_version: str
    cost_matrix_version: int

    # `model_version` starts with the one prefix Pydantic reserves for itself; the field
    # is named for the reader of the response rather than for the library.
    model_config = ConfigDict(protected_namespaces=())


class HealthResponse(BaseModel):
    """Live, ready, and what is loaded.

    Three different facts, per §4: the process being up, the artifacts being loaded, and
    which mode is currently being served. An orchestrator that cannot tell them apart
    either restarts a healthy degraded service or sends traffic to one holding nothing.

    `fallback_decisions` is the fourth, and the one nothing else reports: a service whose
    model loaded but answers every request from the incumbent is indistinguishable from a
    healthy one at the status code, and this is the number that is not.
    """

    status: str
    mode: str
    model_loaded: bool
    fallback_loaded: bool
    fail_open: bool
    budget_ms: float
    fallback_decisions: int
    artifacts: dict[str, str]
    features: int | None
    trees: int | None
    calibration: str | None
    cost_matrix_version: int | None

    model_config = ConfigDict(protected_namespaces=())


class ReasonResponse(BaseModel):
    """One contributor, as the audit trail records it."""

    feature: str
    contribution: float
    phrase: str
    named: bool


class ExplainResponse(BaseModel):
    """What a declined customer can be told, and what the file has to show a regulator.

    Two objects, deliberately. `statements` is the notice — the lines a person is read,
    with the unnameable contributors collapsed into one honest sentence rather than the
    same sentence three times. `reasons` is the audit trail, one entry per contributor,
    which is what `explainability.md` §7 measures and what a reviewer needs.

    `adverse` is whether an explanation is owed at all. Reasons are produced for an allowed
    transaction too — they are useful internally — and the flag is what keeps them from
    being read as a notice nobody was sent.
    """

    decision: str
    adverse: bool
    review_eligible: bool
    probability: float
    break_even: float
    amount: float
    statements: tuple[str, ...]
    reasons: tuple[ReasonResponse, ...]
    dictionary_version: int
    cost_matrix_version: int
    model_version: str

    model_config = ConfigDict(protected_namespaces=())
