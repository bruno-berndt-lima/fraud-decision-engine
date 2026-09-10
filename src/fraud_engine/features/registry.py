"""Which columns each feature family owns.

The single source of truth for that mapping. Both the linear probe and the
LightGBM ablation ask it — one to add a family, the other to remove one — and a
registry either of them owned would put a dict behind an import of a model.

Four families are lists, fixed by the module that builds them. The V-block is
not: its membership is whatever survived a correlation threshold at build time,
so it is declared by prefix and resolved against the matrix on disk. Hardcoding
it would let the registry drift from the threshold in config.
"""

from __future__ import annotations

from pathlib import Path

import pyarrow.parquet as pq

from fraud_engine.features import aggregations, amounts, encoders, vblock, velocity

# Name -> the engineered columns it contributes. "none" is the bare probe, the
# reference every delta and the noise floor itself are measured against.
FAMILIES: dict[str, tuple[str, ...]] = {
    "none": (),
    "amount": amounts.COLUMNS,
    "frequency": encoders.COLUMNS,
    "entity": aggregations.COLUMNS,
    "velocity": velocity.COLUMNS,
}

# Families whose membership is chosen by a fit, so it is not knowable until
# build.py has run. Declared by the prefix every one of their columns carries.
FAMILY_PREFIXES = {"vblock": vblock.PREFIX}


def resolve_families(features_dir: Path | str) -> dict[str, tuple[str, ...]]:
    """``FAMILIES`` with the prefix-declared families filled in from the matrices.

    Reads the built matrix's schema — the names only, never the data — so the
    resolution costs no I/O beyond a parquet footer.

    Args:
        features_dir: Directory holding ``{split}.parquet``.

    Returns:
        ``{family: columns}``, every family concrete.

    Raises:
        FileNotFoundError: If the train matrix is absent. The prefix families
            cannot be resolved without it, and a registry missing one of them
            would hand a caller an empty tuple that reads as an empty family.
    """
    names = pq.read_schema(Path(features_dir) / "train.parquet").names

    resolved = dict(FAMILIES)
    for family, prefix in FAMILY_PREFIXES.items():
        resolved[family] = tuple(name for name in names if name.startswith(prefix))
    return resolved


# The serving tiers, from `docs/features.md`. Four values rather than the
# roadmap's servable/needs-cache, because "cache" states something false about
# tier 0: those columns are Vesta's own pre-computed aggregates over windows
# nobody published, so no store this project chose to build would reproduce
# them. That is a different claim from expensive.
#
# Only three tiers are named. Tier 0 is the remainder, which is the documented
# rule rather than a shortcut — `features.md` assigns undocumented columns to it
# by default, deliberately over-assigning, because an inventory that guesses
# generously about what it could serve is worth nothing. A column added later
# lands there until someone proves its derivation.

# Nothing beyond the fields of the request itself. Also the fail-open set: the
# rules engine is built entirely from it.
TIER_1 = (
    "TransactionAmt",
    "ProductCD",
    "card1",
    "card2",
    "card3",
    "card4",
    "card5",
    "card6",
    "addr1",
    "addr2",
    "dist1",
    "dist2",
    "P_emaildomain",
    "R_emaildomain",
    *(f"M{index}" for index in range(1, 10)),
    "DeviceType",
    "DeviceInfo",
    "has_identity",
    "hour",
    "weekday",
    *amounts.COLUMNS,
    "id_14",
    "id_30",
    "id_31",
    "id_32",
    "id_33",
)

# A file fitted at train time and shipped beside the model. `id_23` is here
# without an artifact of ours: it is a third-party IP classification, which is
# the same serving shape — a lookup that has to arrive with the deployment.
TIER_2 = (*encoders.COLUMNS, *aggregations.COLUMNS, "id_23")

# A per-entity running window, written on every transaction.
TIER_3 = velocity.COLUMNS

# Not features. The label and the two time axes a model handed either would
# learn instead of fraud, plus the join key.
KEYS = ("TransactionID", "TransactionDT", "isFraud", "day")

TIER_0 = "tier_0"

# What `features.md` publishes. Asserted rather than described: the document is
# a deliverable, and a matrix that has drifted from it should stop a run rather
# than quietly re-partition itself.
TIER_SIZES = {"tier_1": 36, "tier_2": 14, "tier_3": 4, TIER_0: 295, "keys": 4}


def resolve_tiers(features_dir: Path | str) -> dict[str, tuple[str, ...]]:
    """Every column of the built matrix, assigned to exactly one serving tier.

    Args:
        features_dir: Directory holding ``{split}.parquet``.

    Returns:
        ``{tier: columns}`` for ``tier_1``, ``tier_2``, ``tier_3``, ``tier_0``
        and ``keys``, in matrix order.

    Raises:
        ValueError: If a named column is absent from the matrix, if the tiers
            overlap, or if any tier's size differs from what ``features.md``
            publishes. All three are the same failure — the inventory and the
            table have drifted — and it has to stop a run, because a tier
            missing a column produces an arm that removes less than it claims.
    """
    names = pq.read_schema(Path(features_dir) / "train.parquet").names
    named = {"tier_1": TIER_1, "tier_2": TIER_2, "tier_3": TIER_3, "keys": KEYS}

    missing = {tier: sorted(set(columns) - set(names)) for tier, columns in named.items()}
    missing = {tier: absent for tier, absent in missing.items() if absent}
    if missing:
        raise ValueError(f"columns absent from the matrix: {missing}")

    seen: dict[str, str] = {}
    for tier, columns in named.items():
        for column in columns:
            if column in seen:
                raise ValueError(f"{column} is in both `{seen[column]}` and `{tier}`")
            seen[column] = tier

    resolved = {
        tier: tuple(name for name in names if name in set(columns))
        for tier, columns in named.items()
    }
    resolved[TIER_0] = tuple(name for name in names if name not in seen)

    sizes = {tier: len(columns) for tier, columns in resolved.items()}
    if sizes != TIER_SIZES:
        raise ValueError(
            f"tier sizes {sizes} do not match what features.md publishes ({TIER_SIZES}); "
            "the inventory and the matrix have drifted apart"
        )

    return resolved
