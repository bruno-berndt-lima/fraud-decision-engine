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
