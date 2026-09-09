"""The family ablation, re-run under a tree.

Phase 04 measured each family by *adding* it to a bare linear probe. E4 recorded
in advance which shapes that instrument structurally could not see, and handed
the re-measurement here rather than pretending the probe had settled it.

Two things invert. **The direction:** an arm is the full matrix *minus* one
family, because the question a serving decision asks is what is lost by not
building it, and because E3 needs exactly that shape for the history-dependent
columns. **The instrument:** a tree can represent thresholds and interactions a
linear probe cannot, which is the whole reason the handoff exists.

The reference config is the untuned one. It does not subsample, so repeated fits
return identical digits and a delta carries no seed noise — `seeds.py` measured
that, and the tuned config's spread against it. The tuned config is stronger and
does subsample, putting a wide bar under every delta; families that move here go
back to it for confirmation, the shape E6 already established.

**The blind spot, registered before running.** Leave-one-out against a matrix
this correlated measures redundancy, not signal. A family whose columns are
recoverable from the survivors reads as zero, and that is a different claim from
having nothing to contribute. It is also why the Phase 04 deltas and these are
not two columns of one table: different base, opposite direction.
"""

from __future__ import annotations

import pandas as pd

from fraud_engine.features.registry import resolve_families

# The arm that removes nothing. Every delta is measured against it.
#
# Deliberately not `none`, which is what Phase 04 called its reference. There it
# meant "add no family" and here it would mean "drop no family" — the same label
# on opposite ends of the comparison, in files a reader meets side by side.
REFERENCE = "full"

# Phase 04's reference key, which carries no columns and does not become an arm.
BARE = "none"


def resolve_arms(features_dir) -> dict[str, tuple[str, ...]]:
    """Arm name -> the columns that arm removes.

    The registry is `features/evaluate`'s, read rather than restated: a second
    list of what a family contains would let the two drift with neither looking
    wrong, and the V-block's membership is decided by a fitted threshold that
    only the built matrix knows.

    Args:
        features_dir: Directory holding `{split}.parquet`.

    Returns:
        `{arm: columns to drop}`, the reference first, holding no columns.

    Raises:
        ValueError: If the bare reference has acquired columns, or if two
            families share one. Overlapping families make the deltas
            non-additive, and nothing downstream would show it.
    """
    families = resolve_families(features_dir)

    if families.get(BARE):
        raise ValueError(
            f"`{BARE}` carries columns: {sorted(families[BARE])}; it is Phase 04's "
            "bare probe and has no meaning as a leave-one-out arm"
        )

    arms = {name: columns for name, columns in families.items() if name != BARE}

    seen: dict[str, str] = {}
    for name, columns in arms.items():
        for column in columns:
            if column in seen:
                raise ValueError(
                    f"{column} belongs to both `{seen[column]}` and `{name}`; "
                    "removing either arm would remove part of the other"
                )
            seen[column] = name

    return {REFERENCE: (), **arms}


def drop_columns(frame: pd.DataFrame, columns: tuple[str, ...]) -> pd.DataFrame:
    """The matrix without one arm's columns.

    The guard is the point. A name that no longer matches the matrix would drop
    nothing, the arm would be the reference fitted twice, and its delta would
    come back at exactly zero — which reads as *this family adds nothing*, the
    most plausible-looking wrong answer this experiment can produce. Same reason
    `feature_columns` requires every excluded name to be present.

    Args:
        frame: Any split's matrix.
        columns: The arm's columns, from `resolve_arms`. Empty returns a copy,
            which is the reference arm.

    Returns:
        A new frame; the input is not modified.

    Raises:
        ValueError: If a named column is absent — the registry and the matrix
            have drifted apart.
    """
    missing = [column for column in columns if column not in frame.columns]
    if missing:
        raise ValueError(
            f"columns absent from the matrix: {missing}; "
            "this arm would remove nothing and score as the reference"
        )

    return frame.drop(columns=list(columns))
