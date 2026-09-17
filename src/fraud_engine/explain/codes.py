"""Why a transaction was declined, in sentences a person can be read.

`docs/explainability.md` §7. The module Phase 08 imports; nothing here touches a
matrix, a figure or a plotting library.

**A decision does not decompose into one list.** The score says where the probability
came from; the bar says what that probability had to clear, and it falls as the amount
rises. A code that showed only the first would explain the model rather than the
decision — the §6 high-value catch was blocked *because of* an amount its own
contribution called reassuring, and the contributions alone say the opposite.

**Review is reported as eligibility, never as an outcome.** Whether a transaction
reaches an analyst depends on the other transactions that day, which is queue state at
serving time. Promising a review the capacity cannot honour would be a false statement
to a customer.

**Only the contributors that argue for the decision that was taken.** For a block, the
largest *positive* ones. Ranking by absolute value would put "this looked safe
because..." into a decline notice, which explains nothing about the decline.

**The dictionary is the boundary.** A feature with an entry gets its sentence; anything
else gets the tier-0 generic, because no sentence exists for it. That keeps the serving
path free of the feature registry — the registry guards the dictionary in the test
suite, where the built matrices are available, rather than at request time.
"""

from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from fraud_engine.data.load import DEFAULT_CONFIG_PATH, load_config
from fraud_engine.evaluation.cost import (
    ALLOW,
    BLOCK,
    REVIEW,
    Costs,
    break_even,
    expected_costs,
    load_costs,
)
from fraud_engine.evaluation.report import git_revision, load_operating_capacity

log = logging.getLogger(__name__)

NAME = "reason_codes"
SPLIT = "test"
HEADLINE_PREDICTIONS = "headline_test"
GENERIC_TIER = "tier_0"


@dataclass(frozen=True)
class Dictionary:
    """The customer-facing sentences, and the version they were written at.

    The generic has two forms. A notice listing three contributors none of which can be
    named would otherwise print one sentence three times, which reads as a fault rather
    than as the limit it is.
    """

    version: int
    generic: str
    generic_many: str
    features: dict[str, str]


def load_dictionary(path: Path | str) -> Dictionary:
    """Read `config/reason_codes.yaml`.

    Raises:
        KeyError: If the file is missing the generic sentence — a dictionary without
            one cannot answer for the columns that dominate this model.
    """
    content = yaml.safe_load(Path(path).read_text())
    generic = content["generic"][GENERIC_TIER]
    return Dictionary(
        version=int(content["version"]),
        generic=generic["one"].strip(),
        generic_many=generic["many"].strip(),
        features={name: phrase.strip() for name, phrase in content["features"].items()},
    )


@dataclass(frozen=True)
class Reason:
    """One contributor, as a customer would be told it."""

    feature: str
    contribution: float
    phrase: str
    named: bool


@dataclass(frozen=True)
class ReasonCodes:
    """A decision, the bar it was taken against, and what argued for it.

    `adverse` is whether an explanation is owed at all. Reasons are produced for an
    allowed transaction too — they are useful internally — and the flag is what keeps
    them from being read as a notice nobody was sent.
    """

    decision: str
    adverse: bool
    review_eligible: bool
    probability: float
    break_even: float
    amount: float
    reasons: tuple[Reason, ...]
    cost_matrix_version: int

    def statements(self, dictionary: Dictionary) -> tuple[str, ...]:
        """The lines a person is actually shown.

        `reasons` is the audit trail — one entry per contributor, which is what §7's
        measurement counts. This is the notice: the contributors with no sentence of
        their own collapse into a single line, in the position the first of them held,
        because printing one sentence three times says nothing three times.
        """
        unnamed = [reason for reason in self.reasons if not reason.named]
        lines: list[str] = []

        for reason in self.reasons:
            if reason.named:
                lines.append(reason.phrase)
            elif reason is unnamed[0]:
                lines.append(dictionary.generic if len(unnamed) == 1 else dictionary.generic_many)

        return tuple(lines)


def phrase_for(feature: str, dictionary: Dictionary) -> tuple[str, bool]:
    """The sentence for a feature, and whether one exists.

    Returns:
        `(phrase, named)`. Unnamed features get the generic, which is an honest
        statement of the limit rather than a filler: §4 measures how much of this model
        it has to cover.
    """
    phrase = dictionary.features.get(feature)
    return (phrase, True) if phrase is not None else (dictionary.generic, False)


def check_covered(columns: list[str], nameable: set[str], dictionary: Dictionary) -> None:
    """Refuse a dictionary that has drifted from the columns the model can split on.

    Args:
        columns: Every feature the booster holds.
        nameable: Those whose serving tier means a sentence should exist — tiers 1 to 3.
        dictionary: As `load_dictionary` returned it.

    Raises:
        ValueError: If a nameable column has no entry, or an entry names a column the
            model does not have. The first ships the generic to a customer for a feature
            someone could have described; the second is a sentence nobody will ever see.
    """
    described = set(dictionary.features)
    missing = sorted(nameable - described)
    stray = sorted(described - set(columns))

    if missing or stray:
        raise ValueError(
            f"the reason-code dictionary has drifted: {len(missing)} nameable columns "
            f"without a sentence {missing}, {len(stray)} sentences for columns the model "
            f"does not have {stray}"
        )


def reason_codes(
    contributions: np.ndarray,
    columns: list[str],
    probability: float,
    amount: float,
    costs: Costs,
    dictionary: Dictionary,
    top_k: int = 3,
) -> ReasonCodes:
    """Everything a person can be told about one decision.

    Args:
        contributions: One row's per-feature contributions, in `columns` order. The base
            value is not one of them.
        columns: Feature names, in the booster's order.
        probability: The **calibrated** probability the policy decided on.
        amount: The transaction amount, USD.
        costs: The loaded cost matrix.
        dictionary: As `load_dictionary` returned it.
        top_k: How many contributors to report.

    Returns:
        The decision, the bar, review eligibility and the reasons that argue for it.

    Raises:
        ValueError: If the contributions are not one row of `columns`.
    """
    if contributions.shape != (len(columns),):
        raise ValueError(
            f"{contributions.shape} contributions for {len(columns)} features; "
            "reason codes explain one transaction at a time"
        )

    p = np.array([probability], dtype="float64")
    value = np.array([amount], dtype="float64")
    expected = expected_costs(p, value, costs)

    blocked = bool(expected[BLOCK][0] < expected[ALLOW][0])
    gain = float(min(expected[ALLOW][0], expected[BLOCK][0]) - expected[REVIEW][0])

    order = np.argsort(-contributions)[:top_k]
    reasons = tuple(
        Reason(columns[index], float(contributions[index]), *phrase_for(columns[index], dictionary))
        for index in order
        if contributions[index] > 0
    )

    return ReasonCodes(
        decision=BLOCK if blocked else ALLOW,
        adverse=blocked,
        review_eligible=gain > 0,
        probability=float(probability),
        break_even=float(break_even(value, costs)[0]),
        amount=float(amount),
        reasons=reasons,
        cost_matrix_version=costs.version,
    )


def summarise(coded: list[ReasonCodes], tiers: dict[str, str]) -> dict:
    """What share of declines can be explained, and with what.

    §4 deferred this: the global ranking says how much of the *model* is nameable, and
    that is not the same as how often a *decision* can be put into sentences. A tail of
    295 columns can dominate the total while the head of any one row is nameable.

    Args:
        coded: Reason codes for the adverse decisions.
        tiers: `{feature: serving tier}`, for the composition table.

    Returns:
        The per-row composition, and the share of declines with nothing but the generic.
    """
    leading = [codes.reasons[0] for codes in coded if codes.reasons]
    every = [reason for codes in coded for reason in codes.reasons]
    fully_generic = [
        codes for codes in coded if codes.reasons and not any(r.named for r in codes.reasons)
    ]

    composition = pd.Series([tiers[reason.feature] for reason in every]).value_counts(
        normalize=True
    )

    return {
        "declines": len(coded),
        "tier_of_every_reason": {tier: float(share) for tier, share in composition.items()},
        "leading_reason_unnameable": float(np.mean([not r.named for r in leading])),
        "fully_generic": len(fully_generic) / len(coded),
        "at_least_one_named": float(np.mean([any(r.named for r in c.reasons) for c in coded])),
    }


def main(config_path: Path = DEFAULT_CONFIG_PATH) -> None:
    """Code every declined test transaction, and record how explainable they were.

    Wiring only. Invoked by `make reason-codes` as
    `python -m fraud_engine.explain.codes`.

    Args:
        config_path: Path to `config.yaml`.
    """
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    from fraud_engine.features.registry import TIER_1, TIER_2, TIER_3, resolve_tiers

    config = load_config(config_path)
    paths, explain_cfg = config["paths"], config["explain"]

    cost_matrix = load_config(Path(paths["cost_matrix"]))
    costs = load_costs(cost_matrix)
    load_operating_capacity(cost_matrix)
    dictionary = load_dictionary(paths["reason_dictionary"])

    contributions = pd.read_parquet(Path(paths["explain_dir"]) / f"{SPLIT}.parquet")
    resolved = resolve_tiers(paths["features_dir"])
    tiers = {c: t for t, members in resolved.items() for c in members if t != "keys"}
    columns = [c for c in contributions.columns if c in tiers]

    check_covered(columns, set(TIER_1) | set(TIER_2) | set(TIER_3), dictionary)
    log.info("dictionary covers every nameable column the model holds")

    predictions = Path(paths["predictions_dir"]) / f"{HEADLINE_PREDICTIONS}.parquet"
    if not predictions.exists():
        raise FileNotFoundError(f"{predictions} is absent; run `make headline` first")

    decided = pd.read_parquet(predictions).set_index("TransactionID")
    rows = contributions[contributions["TransactionID"].isin(decided.index)]
    values = rows[columns].to_numpy(dtype="float64")

    coded = [
        reason_codes(
            values[position],
            columns,
            float(decided.loc[transaction, "calibrated"]),
            float(decided.loc[transaction, "amount"]),
            costs,
            dictionary,
            explain_cfg["top_k"],
        )
        for position, transaction in enumerate(rows["TransactionID"])
    ]
    declines = [codes for codes in coded if codes.adverse]

    summary = summarise(declines, tiers)
    for key, value in summary.items():
        log.info("%-28s %s", key, value)

    record = {
        "name": NAME,
        "created": datetime.now(UTC).isoformat(timespec="seconds"),
        "git_revision": git_revision(),
        "split": SPLIT,
        "dictionary_version": dictionary.version,
        "cost_matrix_version": costs.version,
        "top_k": explain_cfg["top_k"],
        "explained": len(coded),
        "summary": summary,
        "examples": {
            name: asdict(codes) | {"statements": list(codes.statements(dictionary))}
            for name, codes in {
                "largest_decline": max(declines, key=lambda c: c.amount),
                "fully_generic": next(
                    (c for c in declines if c.reasons and not any(r.named for r in c.reasons)),
                    declines[0],
                ),
            }.items()
        },
    }
    path = Path(paths["reason_codes"])
    path.write_text(json.dumps(record, indent=2) + "\n")
    log.info("wrote %s", path)


if __name__ == "__main__":
    main()
