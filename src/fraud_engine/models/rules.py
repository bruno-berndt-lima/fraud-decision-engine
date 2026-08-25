"""The rules baseline — the incumbent the model has to beat.

Hand-written heuristics of the kind a fraud team actually starts with. Two
roles, and both constrain the design:

1. **The comparison.** The Phase 06 headline is USD saved against this engine,
   so a strawman here inflates the result without earning it. Rules were chosen
   on measured evidence and weighted from measured lift.
2. **The fail-open path.** If the model is unavailable or exceeds its latency
   budget, this decides (``docs/problem-statement.md`` §2). So every predicate
   must be answerable from a single request, and every fitted value must be a
   stored constant rather than something computed over a batch.

Evidence, provenance and the rejected rules: ``docs/rules-baseline.md``.

Nothing here is fitted on, tuned against, or evaluated on validation.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd

from fraud_engine.data.load import DEFAULT_CONFIG_PATH, load_config
from fraud_engine.evaluation.report import load_capacities, write_run

log = logging.getLogger(__name__)

# Config choices merged with the values fit() derives from train. One object, so
# serving loads one artifact and a predicate cannot reach a config value that
# was not present when the constants were fitted.
Constants = dict[str, object]

# Where a rule came from. A Literal rather than a free string: two different
# processes produced these rules and they carry different risks, so a rule
# cannot be added without declaring which.
#   hypothesis — formed in Phase 01 EDA, written down before any modelling
#   search     — found by scanning a column family on train during Phase 03,
#                and therefore selection-biased
Provenance = Literal["hypothesis", "search"]

REQUIRED_COLUMNS = ("TransactionAmt", "ProductCD", "D1", "M4")

# Read from interim; TransactionID joins to the split assignment.
INTERIM_COLUMNS = ("TransactionID", "isFraud", "day", *REQUIRED_COLUMNS)

REPORT_NAME = "rules_baseline"


@dataclass(frozen=True)
class Rule:
    """One rule: a name, where it came from, why, and the points it awards.

    ``points`` rather than a boolean predicate because ``product_tier`` awards
    3/2/1/0 by product and is not boolean. Unifying on points removes the
    special case, and "did it fire" stays available as ``points > 0`` — which is
    what Phase 07 reason codes need.

    Attributes:
        name: Column name in ``contributions``. Stable — it is the join key
            between this code, ``config/config.yaml`` and the docs.
        provenance: ``hypothesis`` or ``search``. See ``Provenance``.
        rationale: Why this predicts fraud, in one line. Carried at runtime
            rather than left in a comment because the engine is the fail-open
            path and its decisions have to be explainable at 2am.
        points: ``(frame, constants) -> Series[int64]``. Vectorised over the
            frame; a one-row frame is the serving case and must work.
    """

    name: str
    provenance: Provenance
    rationale: str
    points: Callable[[pd.DataFrame, Constants], pd.Series]


def _binary(
    name: str,
    weight: int,
    provenance: Provenance,
    rationale: str,
    predicate: Callable[[pd.DataFrame, Constants], pd.Series],
) -> Rule:
    """A rule that awards ``weight`` points when ``predicate`` holds, else 0.

    ``astype(bool)`` before the multiply is load-bearing: a nullable ``boolean``
    Series propagates ``pd.NA`` into the score, which poisons the ranking
    silently. Any null in a predicate means "did not fire" — decided per
    predicate below, not inherited by accident.
    """

    def points(frame: pd.DataFrame, constants: Constants) -> pd.Series:
        fired = predicate(frame, constants).fillna(False).astype(bool)
        return fired.astype("int64") * weight

    return Rule(name=name, provenance=provenance, rationale=rationale, points=points)


# ---- predicates --------------------------------------------------------------
# Each takes the whole frame and the fitted constants, and returns a boolean
# Series aligned to it. Pure: same frame in, same result out.


def _round_amount(frame: pd.DataFrame, constants: Constants) -> pd.Series:
    cfg = constants["round_amount"]
    amount = frame["TransactionAmt"]
    return amount.mod(cfg["step"]).eq(0) & amount.between(cfg["min"], cfg["max"])


def _amount_over_product_percentile(frame: pd.DataFrame, constants: Constants) -> pd.Series:
    # ProductCD is a category, so .map() returns a categorical and the
    # comparison below raises. astype(str) first.
    #
    # A product unseen in train maps to NaN, and `amount > NaN` is False — the
    # rule does not fire on a product it has no threshold for. Conservative and
    # deliberate: the alternative is inventing a cut point at serving time.
    cuts = constants["amount_p99"]
    threshold = frame["ProductCD"].astype(str).map(cuts).astype("float64")
    return frame["TransactionAmt"] > threshold


def _new_card(frame: pd.DataFrame, constants: Constants) -> pd.Series:
    # D1 == 0 is excluded deliberately: it covers ~51% of rows at 1.15x, so
    # including it turns a precise rule into a broad one. Null D1 does not fire.
    d1 = frame["D1"]
    return d1.gt(0) & d1.le(constants["new_card"]["d1_max"])


def _w_and_m4_m2(frame: pd.DataFrame, constants: Constants) -> pd.Series:  # noqa: ARG001
    # The ProductCD == "W" scope is load-bearing, not decoration. Pooled,
    # M4 == "M2" looks like a 3.05x signal, but 94% of its rows are product C
    # and within C its lift is 1.01x. Unscoped this rule is product_tier
    # restated. M4 holds strings ("M0"/"M1"/"M2"), not booleans.
    product = frame["ProductCD"].astype(str)
    m4 = frame["M4"].astype("object")
    return product.eq("W") & m4.eq("M2")


def _product_tier(frame: pd.DataFrame, constants: Constants) -> pd.Series:
    # The one non-boolean rule: a points lookup, not a predicate. An unknown
    # product scores 0 rather than raising — at serving time a new product code
    # is an ordinary event, and refusing to score it would fail the transaction.
    tiers = constants["product_tier"]
    mapped = frame["ProductCD"].astype(str).map(tiers)
    return mapped.fillna(0).astype("int64")


def build_rules(rules_cfg: dict) -> tuple[Rule, ...]:
    """The rule set, with weights read from config.

    Weights live in ``config/config.yaml`` so they are visible and versioned
    rather than buried in code — but see the comment there: they were set from
    train lift and are not free parameters to nudge.

    Args:
        rules_cfg: The ``baselines.rules`` block of ``config/config.yaml``.

    Returns:
        The rules, in a fixed order.
    """
    return (
        _binary(
            "round_amount",
            rules_cfg["round_amount"]["weight"],
            "hypothesis",
            "H2: fraud clusters on $50 multiples between $150 and $500, an effect that "
            "strengthens rather than dissolves under a ProductCD control.",
            _round_amount,
        ),
        Rule(
            name="product_tier",
            provenance="hypothesis",
            rationale=(
                "H3: ProductCD separates risk as sharply as any single field available at "
                "scoring time — 11.2% fraud in C against 2.1% in W — and costs nothing to "
                "obtain."
            ),
            points=_product_tier,
        ),
        _binary(
            "amount_percentile",
            rules_cfg["amount_percentile"]["weight"],
            "hypothesis",
            "H1 in the form that survived: fraud amounts run high WITHIN product. The "
            "pooled version was falsified, so the threshold is per-product.",
            _amount_over_product_percentile,
        ),
        _binary(
            "new_card",
            rules_cfg["new_card"]["weight"],
            "search",
            "Servable proxy for a newly seen card. D1 is monotone in fraud rate: 10.5% at "
            "D1<=3 falling to 1.3% past 180 days. Found by scanning the D* family on train.",
            _new_card,
        ),
        _binary(
            "w_m4_m2",
            rules_cfg["w_m4_m2"]["weight"],
            "search",
            "The only rule off the amount/product axis. Within W — the product holding most "
            "fraud USD — M4=='M2' carries independent signal. Scanned from the M* family, "
            "so selection-biased; the weight reflects the weaker half of the split check.",
            _w_and_m4_m2,
        ),
    )


def fit(train: pd.DataFrame, rules_cfg: dict) -> Constants:
    """Derive every value the rules need, from the training window alone.

    This is the only function that sees train, and the only place a number is
    learned from data. Everything downstream is a lookup — which is what makes
    "fit on train, apply everywhere" structural here rather than remembered.

    Two things are fitted:

    - **Per-product amount cut points**, at the configured quantile.
    - **The amount distribution**, as a quantile grid. The tiebreaker needs a
      percentile for a *single* transaction at serving time, so ranking within
      the scored batch is not available: a one-row frame has no distribution.
      The grid makes the percentile a stored constant, and keeps the scored
      order identical whether a row arrives alone or in a batch of 300,000.

    Args:
        train: Training rows. Must carry ``TransactionAmt`` and ``ProductCD``.
        rules_cfg: The ``baselines.rules`` block. Merged into the result so
            predicates take one object.

    Returns:
        The constants, ready to hand to ``contributions`` or ``score``.

    Raises:
        ValueError: If ``train`` is empty or a required column is missing.
    """
    missing = [column for column in ("TransactionAmt", "ProductCD") if column not in train.columns]
    if missing:
        raise ValueError(f"train is missing {missing}; fit() needs TransactionAmt and ProductCD.")
    if train.empty:
        raise ValueError("train is empty: there is nothing to fit the cut points on.")

    quantile = rules_cfg["amount_percentile"]["quantile"]
    cuts = train.groupby(train["ProductCD"].astype(str), observed=True)["TransactionAmt"].quantile(
        quantile
    )

    n_points = rules_cfg["amount_ecdf_points"]
    grid = np.quantile(train["TransactionAmt"].to_numpy(), np.linspace(0.0, 1.0, n_points))

    constants: Constants = dict(rules_cfg)
    constants["amount_p99"] = cuts.to_dict()
    constants["amount_grid"] = grid
    return constants


def amount_percentile(amounts: pd.Series, constants: Constants) -> pd.Series:
    """Where each amount sits in the *training* amount distribution, in [0, 1].

    A lookup against the fitted grid, not a rank over ``amounts`` — see ``fit``.

    Args:
        amounts: Transaction amounts.
        constants: From ``fit``.

    Returns:
        Percentiles aligned to ``amounts``.
    """
    grid = constants["amount_grid"]
    positions = np.searchsorted(grid, amounts.to_numpy(), side="right")
    return pd.Series(positions / len(grid), index=amounts.index)


def contributions(
    frame: pd.DataFrame,
    rules: tuple[Rule, ...],
    constants: Constants,
) -> pd.DataFrame:
    """Points awarded by each rule, one column per rule.

    Kept separate from ``score`` because the per-rule breakdown is what Phase 07
    turns into reason codes, and what makes a decision auditable. A rule "fired"
    where its column is greater than zero.

    Args:
        frame: Transactions to score.
        rules: From ``build_rules``.
        constants: From ``fit``.

    Returns:
        Integer points, columns in rule order, index aligned to ``frame``.

    Raises:
        ValueError: If ``rules`` is empty, a required column is missing, or a
            rule returns a Series that does not align with ``frame``.
    """
    if not rules:
        raise ValueError("no rules: the engine would score every transaction zero.")

    missing = [column for column in REQUIRED_COLUMNS if column not in frame.columns]
    if missing:
        raise ValueError(f"frame is missing {missing}; needs {list(REQUIRED_COLUMNS)}.")

    awarded = {}
    for rule in rules:
        points = rule.points(frame, constants)
        # A predicate that quietly returns the wrong length would produce a
        # score frame full of NaN on assembly, and a report about nothing.
        if not points.index.equals(frame.index):
            raise ValueError(
                f"rule {rule.name!r} returned a Series whose index does not match the frame "
                f"({len(points)} values against {len(frame)} rows)."
            )
        awarded[rule.name] = points.astype("int64")

    return pd.DataFrame(awarded, index=frame.index)


def score(
    frame: pd.DataFrame,
    rules: tuple[Rule, ...],
    constants: Constants,
) -> pd.Series:
    """The risk score: total rule points, with ties broken by amount.

    Five integer-weighted rules emit a single-digit number of distinct totals,
    while the review budget is ~1% of daily volume. The capacity cut therefore
    lands *inside* a block of equally-scored transactions, and which of them
    gets reviewed would be decided by row order. That is arithmetic, not an
    empirical finding — a coarse score cannot rank a fine selection.

    Breaking ties by amount is a policy choice, stated rather than hidden: among
    transactions the rules cannot distinguish, review the expensive ones first.
    It aligns the incumbent with the USD headline. The coefficient is held below
    1 in config so the tiebreaker can never outweigh a rule.

    Args:
        frame: Transactions to score.
        rules: From ``build_rules``.
        constants: From ``fit``.

    Returns:
        Scores aligned to ``frame``. Higher is riskier. Not a probability —
        which is why the Phase 06 per-transaction policy cannot be applied to
        this engine at all.
    """
    total = contributions(frame, rules, constants).sum(axis=1)
    tiebreak = amount_percentile(frame["TransactionAmt"], constants)
    return total + constants["amount_tiebreaker"] * tiebreak


def main(config_path: Path = DEFAULT_CONFIG_PATH) -> None:
    """Fit the rules on train and score validation through the Phase 02 harness.

    Wiring only. Invoked by ``make baselines`` as
    ``python -m fraud_engine.models.rules``.

    TEST is not scored. ``report.DEFAULT_SPLITS`` is validation-only, and this
    stage has no reason to override it.

    Args:
        config_path: Path to ``config.yaml``.
    """
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    config = load_config(config_path)
    paths = config["paths"]
    rules_cfg = config["baselines"]["rules"]

    frame = pd.read_parquet(paths["interim"], columns=list(INTERIM_COLUMNS))
    splits = pd.read_parquet(paths["splits"], columns=["TransactionID", "split"])
    frame = frame.merge(splits, on="TransactionID", how="inner", validate="one_to_one")

    rules = build_rules(rules_cfg)
    constants = fit(frame[frame["split"] == "train"], rules_cfg)
    frame["score"] = score(frame, rules, constants)

    capacities = load_capacities(load_config(Path(paths["cost_matrix"])))
    metrics_path, predictions_path = write_run(
        REPORT_NAME, frame, capacities, paths["metrics_dir"], paths["predictions_dir"]
    )

    for rule in rules:
        log.info("%-20s %-11s %s", rule.name, f"[{rule.provenance}]", rule.rationale)
    log.info(
        "fitted amount cut points: %s", {k: round(v, 2) for k, v in constants["amount_p99"].items()}
    )
    log.info("wrote %s and %s", metrics_path, predictions_path)


if __name__ == "__main__":
    main()
