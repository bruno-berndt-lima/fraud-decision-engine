"""The model's row, assembled with arithmetic instead of per-column pandas.

`docs/serving.md` §5, under the amendment registered there before this file existed. What
that amendment permits is narrow, and the narrowness is the point:

- **The values are never recomputed here.** The vocabulary, the fill values, the surviving
  V columns and their medians are read from the shipped artifacts by `artifacts.py`, once.
  This module only *applies* them — filling with a vector rather than a per-column
  `fillna`, and finding a level's code by lookup rather than by re-levelling a Series.
- **The feature families are still called**, on a frame of the dozen columns they read.
  Recomputing `amt_round_band` or a frequency in numpy would be a second definition of a
  feature, which the amendment does not permit and the gate could not usefully police.
- **`transform.py` remains the reference**, and §8's gate proves the two agree on real
  transactions every time it runs. Where they disagree, what goes is this file.

**Why it exists, in one number.** The reference path costs 211 ms for one row against a
budget of 100, and 163 ms of that is three functions charging per column for a single row.
This assembles the same row in about three.

**The rounding is part of the contract.** A passthrough value is stored as `float32`
because that is what the pipeline stores, and a split threshold can fall between a
`float32` and the `float64` it came from. The same is true of the velocity default and the
V-block fills: every value here is rounded the way the reference rounds it, which is why
the gate compares cells and not only scores.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import numpy as np
import pandas as pd

from fraud_engine.data.load import add_time_columns
from fraud_engine.features import aggregations, amounts, encoders, vblock, velocity
from fraud_engine.features.encoders import MISSING
from fraud_engine.models.train import OTHER
from fraud_engine.serving.artifacts import Model
from fraud_engine.serving.transform import (
    FIRST_SIGHTING_COUNT,
    SECONDS_PER_DAY,
    raw_inputs,
    typed,
)

# What the families build, and what they read to build it. Both derived from the family
# modules rather than listed, so a family that gained a column cannot be missed here.
FAMILY_OUTPUTS = (
    *amounts.COLUMNS,
    *encoders.COLUMNS,
    *aggregations.COLUMNS,
    "hour",
    "weekday",
)
# Deduplicated: `card1` and `addr1` are both frequency sources and entity keys, and a
# frame carrying either twice hands `apply_frequencies` a DataFrame where it expects a
# Series — which fails inside pandas rather than here.
FAMILY_INPUTS = tuple(
    dict.fromkeys(
        (
            "TransactionDT",
            "TransactionAmt",
            "ProductCD",
            *encoders.FREQUENCY_COLUMNS,
            *aggregations.ENTITY_COLUMNS,
        )
    )
)

PRESENCE_PREFIX = f"{vblock.PREFIX}present_"


@dataclass(frozen=True)
class Layout:
    """Where every value lands in the row, worked out once at startup.

    A request then costs three array writes and a dozen dictionary lookups. Everything
    that could be computed before the request arrived has been.

    Attributes:
        columns: The booster's feature names, in order.
        families: `{built column: its position}`.
        passthrough: `{input: its position}` for the columns that arrive as numbers.
        codes: `{categorical: (position, {level: code})}` — the vocabulary, inverted.
        history: `{velocity column: (position, its no-history value)}`.
        kept: Source indices into the V block, their positions, and their fills.
        presence: Source indices into the V block and the positions of their flags.
        fill: The model's medians, aligned to `columns`, `NaN` where a column has none.
    """

    columns: tuple[str, ...]
    families: dict[str, int]
    passthrough: dict[str, int]
    codes: dict[str, tuple[int, dict[str, int]]]
    history: dict[str, tuple[int, float]]
    kept: tuple[np.ndarray, np.ndarray, np.ndarray]
    presence: tuple[np.ndarray, np.ndarray]
    fill: np.ndarray
    fillable: np.ndarray


def build_layout(model: Model, features_cfg: dict) -> Layout:
    """Precompute the row's shape from the artifacts the service loaded.

    Args:
        model: From `artifacts.load_model`.
        features_cfg: The `features` block — only the velocity default is read from it.

    Returns:
        A layout whose every array is aligned to `model.columns`.
    """
    columns = tuple(model.columns)
    position = {name: index for index, name in enumerate(columns)}
    tables = model.tables

    block = {name: index for index, name in enumerate(vblock.V_COLUMNS)}
    representatives = tables.vblock["representatives"]
    sources = tables.vblock["presence"]

    gap = np.float32(
        np.log1p(float(features_cfg["velocity"]["first_seen_gap_days"]) * SECONDS_PER_DAY)
    )

    fill = np.full(len(columns), np.nan)
    if tables.medians is not None:
        for name, value in tables.medians.items():
            # Rounded to the dtype the column is stored in: the reference fills a float32
            # column and keeps it float32, so the served value is the rounded median.
            fill[position[name]] = np.float32(value)

    return Layout(
        columns=columns,
        families={name: position[name] for name in FAMILY_OUTPUTS if name in position},
        passthrough={
            name: position[name]
            for name in raw_inputs(columns)
            if name in position and name not in tables.vocabulary and name not in block
        },
        codes={
            name: (position[name], {level: code for code, level in enumerate(levels)})
            for name, levels in tables.vocabulary.items()
            if name in position
        },
        history={
            name: (position[name], float(gap) if name == velocity.RECENCY else FIRST_SIGHTING_COUNT)
            for name in velocity.COLUMNS
            if name in position
        },
        kept=(
            np.array([block[name] for name in representatives]),
            np.array([position[f"{vblock.PREFIX}{name}"] for name in representatives]),
            tables.vblock["medians"].to_numpy(dtype="float32"),
        ),
        presence=(
            np.array([block[name] for name in sources]),
            np.array([position[f"{PRESENCE_PREFIX}{name}"] for name in sources]),
        ),
        fill=fill,
        fillable=~np.isnan(fill),
    )


def carried(values: Mapping, name: str) -> bool:
    """Whether the request actually gave a value for `name`.

    An absent key and an explicit null are the same fact — §1 — and a float `nan` is how
    one arrives once pandas has touched it.
    """
    value = values.get(name)
    return value is not None and value == value


def family_row(values: Mapping, model: Model, load_cfg: dict, features_cfg: dict) -> pd.DataFrame:
    """The families' own columns, built on a frame of what they read.

    Still `build_features`' functions, still in its order. A dozen columns rather than
    seven hundred is what makes them cost milliseconds; nothing about what they compute
    changes, and the typing rule is `transform`'s own.
    """
    narrow = typed(
        pd.DataFrame({name: [values.get(name)] for name in FAMILY_INPUTS}),
        FAMILY_INPUTS,
        model.tables.vocabulary,
        load_cfg,
    )

    built = add_time_columns(narrow)
    built = amounts.add_amount_features(built, features_cfg["amounts"])
    built = encoders.apply_frequencies(built, model.tables.frequencies)
    return aggregations.apply_amount_stats(built, model.tables.amount_stats)


def row(
    values: Mapping, model: Model, layout: Layout, load_cfg: dict, features_cfg: dict
) -> np.ndarray:
    """One request as the matrix the booster consumes.

    Args:
        values: The request's fields. Absent keys and nulls are the same thing.
        model: From `artifacts.load_model`.
        layout: From `build_layout`, for this model.
        load_cfg: The `load` block of `config.yaml`.
        features_cfg: The `features` block.

    Returns:
        A `(1, n_features)` array of float64, categoricals carrying their codes.
    """
    built = np.full(len(layout.columns), np.nan)

    for name, index in layout.passthrough.items():
        if carried(values, name):
            value = values[name]
            built[index] = np.float64(value) if name == "TransactionAmt" else np.float32(value)

    families = family_row(values, model, load_cfg, features_cfg)
    for name, index in layout.families.items():
        built[index] = families[name].iloc[0]

    # §2: tier-3 state is used when a caller has it and defaulted when it does not.
    for name, (index, default) in layout.history.items():
        built[index] = np.float32(values[name]) if carried(values, name) else default

    block = np.array([values.get(name, np.nan) for name in vblock.V_COLUMNS], dtype="float32")
    source, target, medians = layout.kept
    survivors = block[source]
    built[target] = np.where(np.isnan(survivors), medians, survivors)

    flag_source, flag_target = layout.presence
    built[flag_target] = (~np.isnan(block[flag_source])).astype("int8")

    for name, (index, table) in layout.codes.items():
        level = str(values[name]) if carried(values, name) else MISSING
        built[index] = table.get(level, table[OTHER])

    filled = np.where(np.isnan(built) & layout.fillable, layout.fill, built)
    return filled.reshape(1, -1)


def coverage(values: Mapping, columns: Sequence[str]) -> tuple[int, int]:
    """How many of the inputs the request carried, against how many the model reads.

    §1's guard against a silent integration failure: a caller that stopped sending the
    inherited block gets scores that look ordinary, and this is the number that moves.
    """
    expected = raw_inputs(columns)
    return sum(carried(values, name) for name in expected), len(expected)


def history_supplied(values: Mapping) -> tuple[str, ...]:
    """Which tier-3 columns the caller carried, for the response to declare."""
    return tuple(name for name in velocity.COLUMNS if carried(values, name))
