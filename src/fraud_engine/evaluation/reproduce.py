"""Proof that a model scoring now is the one a record was made from.

Its own module rather than part of `report.py`, which most recorded stages depend on:
a guard added there would mark every one of them stale. `usd_halves` keeps the
VAL-FIT copy it was committed and run with, for the same reason.
"""

from __future__ import annotations

import pandas as pd


def check_reproduces(
    scored: pd.DataFrame, recorded: pd.DataFrame, name: str, split: str = "val_fit"
) -> None:
    """Refuse scores on `split` that differ at all from a recorded run's.

    Used before anything new is attached to a model: a USD figure for a refit, or a
    test score for a model reloaded from disk.

    Raises:
        ValueError: If the transactions differ or any score differs at all.
    """
    label = split.replace("_", "-").upper()
    new = scored[scored["split"] == split].set_index("TransactionID")["score"]
    old = recorded[recorded["split"] == split].set_index("TransactionID")["score"]

    if old.empty or set(new.index) != set(old.index):
        raise ValueError(f"{name}: scored different {label} rows than the record")

    differ = int((new.loc[old.index] != old).sum())
    if differ:
        raise ValueError(
            f"{name}: {differ} {label} scores differ from the recorded run; this is not the "
            "model the record was made from"
        )
