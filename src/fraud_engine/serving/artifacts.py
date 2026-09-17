"""The fitted tables, read back from what the pipeline wrote.

Five artifacts, and until this module existed every one of them was **write-only**:
the stages needing a vocabulary refit it from train through ``prepare_matrices``, so
nothing in the project had ever read ``categories.parquet`` back. Serving is their
first reader, which is why the guards here are heavier than a file format usually
deserves — ``docs/serving.md`` §8 registers this as the phase's named risk.

**The codes are positional.** A booster records which features are categorical and the
integer codes its splits test, never what those codes stand for. A read that recovered
the levels in a different order would hand the model correct-looking codes standing for
the wrong levels, and nothing would raise: the scores would simply be wrong. So the code
is read from the column the writer put it in — ``write_categories`` persists it for
exactly this reason — and checked to be the dense range it claims to be.

**The readers live here rather than beside their writers.** Every writer sits in a module
a frozen stage depends on: adding one to ``train.py`` would mark a 2,990-tree booster
stale and leave the guarded headline target unsatisfiable, and adding one to
``encoders.py`` would restage the feature build. The cost is that a writer and its reader
are two files apart, and the round-trip tests are what hold them together.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from fraud_engine.features.aggregations import GLOBAL
from fraud_engine.features.encoders import MISSING
from fraud_engine.features.vblock import PREFIX, V_COLUMNS
from fraud_engine.models.train import OTHER

# What `vblock.write_tables` calls a flag, against what it calls a kept column.
PRESENCE_PREFIX = f"{PREFIX}present_"
REPRESENTATIVE, PRESENCE = "representative", "presence"

# Both belong to every column's vocabulary whether or not train needed them — an
# absent field and an unrecognised one are different facts and each has a level.
# A vocabulary missing one does not fail: `apply_categories` would produce a null
# for the value it cannot place, and a null is what the level exists to avoid.
SENTINELS = (OTHER, MISSING)


@dataclass(frozen=True)
class Tables:
    """Everything the transform needs that was fitted rather than requested.

    One object because they are loaded once, at startup, and travel together. A
    process holding three of the five would not fail — it would score a request
    against a distribution the booster was never fitted on.

    Attributes:
        vocabulary: ``{column: levels}``, levels in code order.
        medians: Numeric fill values, or ``None`` when the model ships unimputed.
        frequencies: ``{column: Series}`` indexed by level, as a share of train.
        amount_stats: ``{entity: DataFrame}`` with ``count``, ``mean``, ``std``.
        vblock: The reduction, in the shape ``vblock.apply_fitted`` expects.
    """

    vocabulary: dict[str, pd.Index]
    medians: pd.Series | None
    frequencies: dict[str, pd.Series]
    amount_stats: dict[str, pd.DataFrame]
    vblock: dict


def read_categories(path: Path | str) -> dict[str, pd.Index]:
    """The category vocabulary, in the order the codes were assigned.

    Args:
        path: ``categories.parquet``, as ``train.write_categories`` wrote it.

    Returns:
        ``{column: levels}``. Position in ``levels`` is the code the booster
        tests, which is the whole reason this file exists.

    Raises:
        ValueError: If a column's codes are not ``0..n-1``, or a column has no
            sentinel to route an unrecognised or absent value to. The first means
            the file no longer describes the model; the second means a value the
            vocabulary cannot place becomes a null rather than a level.
    """
    rows = pd.read_parquet(path)
    vocabulary: dict[str, pd.Index] = {}

    for column, group in rows.groupby("column", sort=False):
        ordered = group.sort_values("code")
        codes = ordered["code"].to_numpy()

        if not np.array_equal(codes, np.arange(len(codes))):
            raise ValueError(
                f"{column!r} carries codes {codes.tolist()[:5]}… rather than a dense range "
                "from zero; the levels no longer stand for what the booster's splits test"
            )

        # `object`, not the string dtype a parquet read now returns: the vocabulary
        # this reconstructs was built by `fit_categories` from object values, and a
        # CategoricalDtype compares unequal to one over identically-spelled `str`
        # categories. The codes would match and the dtypes would not, which is a
        # served column that is not the column the model was fitted on.
        levels = pd.Index(ordered["level"].tolist(), dtype=object)
        absent = [sentinel for sentinel in SENTINELS if sentinel not in levels]
        if absent:
            raise ValueError(
                f"{column!r} has no {absent} level; a value the vocabulary cannot place "
                "would become a null instead of the bucket that was fitted for it"
            )

        vocabulary[str(column)] = levels

    return vocabulary


def read_medians(path: Path | str) -> pd.Series:
    """The numeric fill values, indexed by column.

    Raises:
        ValueError: If a column appears twice, or its median is null — a null
            fill leaves the gap it was shipped to close.
    """
    rows = pd.read_parquet(path)
    medians = rows.set_index("column")["median"]

    if medians.index.has_duplicates:
        duplicated = medians.index[medians.index.duplicated()].tolist()
        raise ValueError(f"two fill values for {duplicated}; which one applies is row order")
    if medians.isna().any():
        raise ValueError(f"null fill values for {medians.index[medians.isna()].tolist()}")

    return medians


def read_frequencies(path: Path | str) -> dict[str, pd.Series]:
    """The fitted rates, one lookup per encoded column.

    Raises:
        ValueError: If a level appears twice in one column's table, which would
            make the rate a row-order choice.
    """
    rows = pd.read_parquet(path)
    tables = {}

    for column, group in rows.groupby("column", sort=False):
        table = group.set_index("level")["frequency"]
        if table.index.has_duplicates:
            raise ValueError(f"{column!r} has repeated levels in its frequency table")
        tables[str(column)] = table

    return tables


def read_amount_stats(path: Path | str) -> dict[str, pd.DataFrame]:
    """The per-entity amount statistics, with the fallback each one carries.

    Raises:
        ValueError: If an entity's table has no ``GLOBAL`` row. That row is what
            an entity train never saw is scored against, and without it the first
            unseen card raises a ``KeyError`` mid-request.
    """
    rows = pd.read_parquet(path)
    tables = {}

    for entity, group in rows.groupby("entity", sort=False):
        table = group.set_index("level")[["count", "mean", "std"]]
        if GLOBAL not in table.index:
            raise ValueError(
                f"{entity!r} has no {GLOBAL} row; an entity the training window never saw "
                "has nothing to be scored against"
            )
        tables[str(entity)] = table

    return tables


def read_vblock(path: Path | str) -> dict:
    """The V-block reduction, in the shape ``vblock.apply_fitted`` expects.

    The file records the *outcome* — which columns survived and what fills them —
    and not the block they were chosen from, so ``source`` is restored as the
    whole of ``V_COLUMNS``. That is what the shipped fit consumed; a narrower one
    is only ever passed by a test exercising the reduction itself.

    Raises:
        ValueError: If a row carries an unknown role, or a kept column has no
            median. A kept column without one puts nulls back into a family whose
            contract is that it has none.
    """
    rows = pd.read_parquet(path)

    unknown = sorted(set(rows["role"]) - {REPRESENTATIVE, PRESENCE})
    if unknown:
        raise ValueError(f"unknown roles in the V-block reduction: {unknown}")

    kept = rows[rows["role"] == REPRESENTATIVE]
    flags = rows[rows["role"] == PRESENCE]

    if kept["median"].isna().any():
        without = kept.loc[kept["median"].isna(), "column"].tolist()
        raise ValueError(f"kept columns with no fill value: {without}")

    representatives = [name.removeprefix(PREFIX) for name in kept["column"]]

    return {
        "source": list(V_COLUMNS),
        "representatives": representatives,
        "presence": [name.removeprefix(PRESENCE_PREFIX) for name in flags["column"]],
        "medians": pd.Series(kept["median"].to_numpy(), index=representatives),
    }


def load_tables(paths: Mapping[str, str], impute: bool) -> Tables:
    """Every fitted table a request needs, read once at startup.

    Args:
        paths: The ``paths`` block of ``config.yaml``.
        impute: ``model.impute``. When off there are no fill values to read, and
            ``None`` says that rather than an empty table, which would silently
            fill nothing.

    Returns:
        The five, as ``Tables``.

    Raises:
        FileNotFoundError: If any is absent. Serving reads that as the model
            being unavailable — a table is not optional equipment.
        As each reader.
    """
    return Tables(
        vocabulary=read_categories(paths["categories"]),
        medians=read_medians(paths["medians"]) if impute else None,
        frequencies=read_frequencies(paths["encoders"]),
        amount_stats=read_amount_stats(paths["amount_stats"]),
        vblock=read_vblock(paths["vblock"]),
    )
