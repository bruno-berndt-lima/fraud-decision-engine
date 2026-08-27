"""Entity aggregates — how unusual this amount is for this card, or this region.

Group `TransactionAmt` by an entity, fit the group's typical amount on the
training rows, and score every transaction by how far it sits from that. The
entity's *size* is deliberately not here: `encoders.freq_card1` already is the
entity count, as a rate.

**Both the signed z-score and its absolute value ship.** E4 registers that a
linear probe cannot see a symmetric relationship, and "unusual in either
direction" is symmetric by construction — without |z| the family would be
untestable on this probe rather than merely unsupported by it. The signed
version stays because a tree can find asymmetry the absolute value hides.

**A training row contributes to its own entity's mean.** That is self-reference,
not leakage: `TransactionAmt` is a feature, not the label, so the
leave-one-out discipline that target encoding needs does not apply. Shrinkage is
what stops the degenerate case — without it a card seen once would have a mean
equal to its only amount, a z-score of exactly zero, and would look perfectly
typical on no evidence at all.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from fraud_engine.features.encoders import levels

# card1 is the closest thing to a card identifier; addr1 the billing region.
# P_emaildomain is left out: the probe already one-hots it, and its 59 levels
# make "this domain's typical amount" a coarse statistic rather than an entity's.
ENTITY_COLUMNS = ("card1", "addr1")

# mean is the entity's typical spend; z how far this transaction sits from it;
# absz the same without direction. See the module docstring for why both.
STATISTICS = ("mean", "z", "absz")

# Reserved level holding the training window's own mean and std. Kept inside the
# table rather than beside it so the persisted artifact is self-contained: a
# serving process reads one file and has both the lookup and its fallback.
GLOBAL = "__global__"


def entity_column(entity: str, statistic: str) -> str:
    """The output column for one statistic of one entity."""
    return f"amt_{statistic}_{entity}"


COLUMNS = tuple(
    entity_column(entity, statistic) for entity in ENTITY_COLUMNS for statistic in STATISTICS
)


def _shrink(observed: pd.Series, count: pd.Series, prior: float, strength: float) -> pd.Series:
    """Pull a per-entity statistic toward the global one, by how little evidence backs it.

    ``(n * observed + k * prior) / (n + k)``. At n=1 the result is almost the
    prior; by n >> k it is almost the observation. A hard minimum-count cutoff
    would do a similar job with an arbitrary threshold to defend and a
    discontinuity at it.

    Applied to the standard deviation it also removes the division-by-zero this
    family would otherwise have: an entity whose amounts are all identical has an
    observed spread of zero, and the shrunk spread stays strictly positive —
    provided the prior is, which ``fit_amount_stats`` checks.
    """
    return (count * observed.fillna(0.0) + strength * prior) / (count + strength)


def fit_amount_stats(
    train: pd.DataFrame, entities: tuple[str, ...], prior_strength: float
) -> dict[str, pd.DataFrame]:
    """Each entity's typical amount and spread, measured on the training rows alone.

    Null is a level of its own, as in ``encoders`` — and for `addr1` a populous
    one: H3 found null-address rows are almost entirely a single `ProductCD`,
    which makes them a real population with a real typical amount, not an
    absence.

    Args:
        train: The training rows, and only those.
        entities: Columns to group by.
        prior_strength: ``k`` in the shrinkage above, in units of transactions.

    Returns:
        ``{entity: DataFrame}`` indexed by level, with ``count``, ``mean`` and
        ``std``, plus a ``GLOBAL`` row carrying the training window's own.

    Raises:
        ValueError: If the training amounts have no positive spread. Shrinkage
            keeps every entity's spread above zero by pulling it toward the
            prior, so a prior of zero is the one input that can still put a
            division by zero into the output — and this family promises finite,
            null-free columns. On real data it cannot happen; a slice small or
            degenerate enough to trigger it is a defect worth hearing about
            rather than scoring.
    """
    amount = train["TransactionAmt"]
    prior_mean, prior_std = amount.mean(), amount.std()

    if not prior_std > 0:
        raise ValueError(
            f"training amounts have no spread (std={prior_std!r}); "
            "every z-score would be a division by zero."
        )

    tables = {}
    for entity in entities:
        grouped = amount.groupby(levels(train[entity]))
        table = pd.DataFrame(
            {"count": grouped.size(), "mean": grouped.mean(), "std": grouped.std()}
        )
        table["mean"] = _shrink(table["mean"], table["count"], prior_mean, prior_strength)
        table["std"] = _shrink(table["std"], table["count"], prior_std, prior_strength)

        table.loc[GLOBAL] = {"count": len(train), "mean": prior_mean, "std": prior_std}
        tables[entity] = table

    return tables


def apply_amount_stats(frame: pd.DataFrame, tables: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Score every row against the fitted statistics.

    Applied to all splits — it is the fit that is confined to the training
    window, never the transform. An entity train never saw falls back to the
    ``GLOBAL`` row, so its z-score is measured against the training window as a
    whole. That is the honest answer for an entity nothing is known about, and it
    leaves the family null-free.

    Returns:
        A copy of ``frame`` with ``COLUMNS`` added, as ``float32``.
    """
    amount = frame["TransactionAmt"]
    built = {}

    for entity, table in tables.items():
        key = levels(frame[entity])
        mean = key.map(table["mean"]).fillna(table["mean"][GLOBAL])
        std = key.map(table["std"]).fillna(table["std"][GLOBAL])
        deviation = (amount - mean) / std

        built[entity_column(entity, "mean")] = mean.astype("float32")
        built[entity_column(entity, "z")] = deviation.astype("float32")
        built[entity_column(entity, "absz")] = deviation.abs().astype("float32")

    return frame.assign(**built)


def add_amount_stats(frame: pd.DataFrame, aggregations_cfg: dict) -> pd.DataFrame:
    """The entity family: fitted on ``frame``'s train rows, applied to all of it.

    Reaches the training window through ``frame["split"] == "train"``, the
    discipline ``build_features`` documents. Which entities to group by is code
    rather than config, because ``COLUMNS`` is imported by the family registry
    and a config edit could otherwise name entities no column exists for.
    """
    train = frame[frame["split"] == "train"]
    tables = fit_amount_stats(train, ENTITY_COLUMNS, aggregations_cfg["prior_strength"])
    return apply_amount_stats(frame, tables)


def write_tables(tables: dict[str, pd.DataFrame], path: Path | str) -> None:
    """Persist the fitted statistics — the second artifact a served model carries.

    Long form, one row per level, carrying its own ``GLOBAL`` fallback. Like the
    frequency tables this is what makes the family **tier 2**: a static lookup
    beside the model, no online store and no entity history at request time.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    rows = pd.concat(
        [
            table.rename_axis("level").reset_index().assign(entity=entity)
            for entity, table in tables.items()
        ],
        ignore_index=True,
    )
    rows[["entity", "level", "count", "mean", "std"]].to_parquet(path, index=False)
