"""One raw transaction, as the booster consumes it.

The serving half of ``features/build.py`` and the apply half of ``models/train.py``,
composed for a request instead of for a table. Every family is *called*, never
reimplemented: a second definition of the vocabulary or the fill values would be
train/serve skew written by hand, which is the defect ``docs/serving.md`` §8's gate
exists to catch.

**Vectorised, with one row as the case that matters.** ``rules.py`` takes the same
position for the same reason — the gate compares hundreds of rows against the matrices
at once, and a transform that only worked one row at a time could not be tested that way.

**Order is the contract, twice over.** Dtypes are coerced *before* the fitted families
run, because ``encoders`` and ``aggregations`` look their tables up by ``str(value)``: a
``card1`` arriving as ``13926`` rather than ``13926.0`` misses every row of the frequency
table and is scored as a level the training window never saw — a wrong number, not an
error. And categories are applied before the medians, because a null is not an
unrecognised value, and ``apply_categories`` is what tells the two apart.

**Velocity is the one family that cannot be called**: it reads the card's history, which
a request does not carry. ``serving.md`` §2 registers what serving sends instead.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd

from fraud_engine.data.load import add_time_columns
from fraud_engine.features import aggregations, amounts, encoders, vblock, velocity
from fraud_engine.models.train import apply_categories, apply_medians
from fraud_engine.serving.artifacts import Tables

# Derived from the timestamp, never carried. A request that sent its own `hour`
# could disagree with its own `TransactionDT`, and the model would be scored on
# whichever of the two the pipeline happened to read.
DERIVED_FROM_TIME = ("day", "hour", "weekday")

# What nothing can be decided without. The amount prices the decision, the
# timestamp places it in the day and the week, `ProductCD` is read by two
# families, and `has_identity` records whether an identity record existed at all
# — which is not recoverable from the identity values, since a matched record may
# be null in every field.
REQUIRED_INPUTS = ("TransactionDT", "TransactionAmt", "ProductCD", "has_identity")

# A null in one of these is not a missing field, it is a field with no honest
# default: `astype("bool")` reads a null `has_identity` as True, and an amount
# that is not a number cannot be compared to a break-even. `ProductCD` is absent
# from this list deliberately — null is a level the vocabulary holds.
NON_NULL_INPUTS = ("TransactionDT", "TransactionAmt", "has_identity")

# `load.build_dtype_map` reads the timestamp as an integer, and `add_time_columns`
# divides it — so `hour` and `weekday` are integers in the matrices. A float
# timestamp here would make them floats, and the booster would meet a dtype it was
# never fitted on.
TIME_DTYPE = "int32"

SECONDS_PER_DAY = 86_400

# The trailing windows are half-open and count the transaction itself, so a card
# seen for the first time carries one in all three — not zero.
FIRST_SIGHTING_COUNT = 1.0

# Every column some family builds. Subtracted from the model's own feature names
# to leave the ones a request has to supply.
ENGINEERED = frozenset(
    (*amounts.COLUMNS, *encoders.COLUMNS, *aggregations.COLUMNS, *velocity.COLUMNS)
)


def raw_inputs(columns: Sequence[str]) -> tuple[str, ...]:
    """The fields a request has to be able to carry, derived from the model's own.

    Not a hand-written list: a column the booster holds that no family builds is one the
    request must supply, so deriving it means the two cannot drift apart. The V block is
    added whole — the reduction consumes 339 columns and emits 234 under a prefix, so
    none of its inputs appear among the model's feature names at all.

    Args:
        columns: The booster's feature names.

    Returns:
        Input names, deduplicated, in the order they were derived.
    """
    passthrough = [
        column
        for column in columns
        if column not in ENGINEERED
        and not column.startswith(vblock.PREFIX)
        and column not in DERIVED_FROM_TIME
    ]
    return tuple(dict.fromkeys([*passthrough, *vblock.V_COLUMNS, "TransactionDT"]))


def prepare_inputs(
    raw: pd.DataFrame, vocabulary: dict[str, pd.Index], load_cfg: dict, columns: Sequence[str]
) -> pd.DataFrame:
    """The request in the pipeline's own dtypes, with the fields it omitted as nulls.

    **An omitted inherited column is a null, not an error.** The model was fitted on a
    matrix where those columns were null on most rows, and the V block's presence flags
    encode nullness as signal — so a field that did not arrive is a value the booster has
    mass on. Whether a caller is *allowed* to omit it is the request schema's question,
    not this function's.

    The frame is narrowed to the inputs the model needs, so anything else a caller sent —
    a label, a split, an ``hour`` of its own — can neither be read nor collide with a
    column derived later.

    Args:
        raw: One row per transaction, carrying the request's fields.
        vocabulary: From ``artifacts.read_categories``. Its keys are left alone here;
            ``apply_categories`` is where a level becomes a code.
        load_cfg: The ``load`` block of ``config.yaml``.
        columns: The booster's feature names.

    Returns:
        A frame of exactly ``raw_inputs(columns)``, typed as the pipeline types them.

    Raises:
        ValueError: If a required input is absent, or one that has no honest default
            is null.
    """
    absent_required = [name for name in REQUIRED_INPUTS if name not in raw.columns]
    if absent_required:
        raise ValueError(
            f"the request is missing {absent_required}; nothing about this transaction "
            "can be decided without them"
        )

    null_required = [name for name in NON_NULL_INPUTS if raw[name].isna().any()]
    if null_required:
        raise ValueError(
            f"{null_required} arrived null; these have no default that is not a guess "
            "about the transaction"
        )

    # Tier 3 is optional rather than required (§2), so it is not among the inputs
    # `raw_inputs` derives — but narrowing the frame without it would drop state a
    # caller did supply and default it silently.
    needed = [*raw_inputs(columns), *history_supplied(raw)]
    frame = raw.reindex(columns=needed)

    casts = {}
    for name in needed:
        if name in vocabulary:
            continue
        if name == "TransactionAmt":
            casts[name] = load_cfg["amount_dtype"]
        elif name == "TransactionDT":
            casts[name] = TIME_DTYPE
        elif name == "has_identity":
            casts[name] = "bool"
        else:
            casts[name] = load_cfg["default_float_dtype"]

    return frame.astype(casts)


def fill_history(frame: pd.DataFrame, velocity_cfg: dict) -> pd.DataFrame:
    """The velocity family as the card nobody has seen before — ``serving.md`` §2.

    The values are the family's own, not invented for serving: ``trailing_counts`` gives a
    first sighting one in every window because the window includes the transaction, and
    ``velocity.recency`` gives it ``log1p`` of the configured gap in seconds. A request
    scored this way meets values the model was fitted against, rather than a hole.

    **Supplied state is kept.** §2's contract accepts tier-3 columns from a caller that
    has them and defaults only what is absent, so the store stays optional rather than
    forbidden.

    Args:
        frame: The prepared inputs.
        velocity_cfg: The ``features.velocity`` block.

    Returns:
        ``frame`` with all four velocity columns, as ``float32``.
    """
    gap_seconds = float(velocity_cfg["first_seen_gap_days"]) * SECONDS_PER_DAY
    defaults = {
        column: np.log1p(gap_seconds) if column == velocity.RECENCY else FIRST_SIGHTING_COUNT
        for column in velocity.COLUMNS
    }
    absent = [column for column in velocity.COLUMNS if column not in frame.columns]

    if absent:
        frame = pd.concat(
            [
                frame,
                pd.DataFrame(
                    {column: defaults[column] for column in absent},
                    index=frame.index,
                    dtype="float32",
                ),
            ],
            axis=1,
        )

    return frame.astype(dict.fromkeys(velocity.COLUMNS, "float32"))


def history_supplied(raw: pd.DataFrame) -> tuple[str, ...]:
    """Which tier-3 columns the caller carried, for the response to declare.

    A consumer has to be able to tell a decision scored with the card's history from one
    scored without it, and inferring that from the values would mean guessing.
    """
    return tuple(column for column in velocity.COLUMNS if column in raw.columns)


def transform(
    raw: pd.DataFrame,
    tables: Tables,
    load_cfg: dict,
    features_cfg: dict,
    columns: Sequence[str],
) -> pd.DataFrame:
    """A request, or a batch of them, as the matrix the booster was fitted on.

    The order below is ``build_features``' order followed by ``prepare_matrices``' apply
    half, and it is not interchangeable — see the module docstring.

    Args:
        raw: One row per transaction, carrying the request's fields.
        tables: From ``artifacts.load_tables``.
        load_cfg: The ``load`` block of ``config.yaml``.
        features_cfg: The ``features`` block.
        columns: The booster's feature names, which are also the output's order.

    Returns:
        One row per input row, holding exactly ``columns``.

    Raises:
        ValueError: If a column the model expects was not built, or per the
            functions this composes.
    """
    frame = prepare_inputs(raw, tables.vocabulary, load_cfg, columns)
    frame = add_time_columns(frame)

    frame = amounts.add_amount_features(frame, features_cfg["amounts"])
    frame = encoders.apply_frequencies(frame, tables.frequencies)
    frame = aggregations.apply_amount_stats(frame, tables.amount_stats)
    frame = fill_history(frame, features_cfg["velocity"])
    frame = vblock.apply_fitted(frame, tables.vblock)

    frame = apply_categories(frame, tables.vocabulary)
    if tables.medians is not None:
        frame = apply_medians(frame, tables.medians)

    unbuilt = [column for column in columns if column not in frame.columns]
    if unbuilt:
        raise ValueError(
            f"{len(unbuilt)} columns the model expects were not built: {unbuilt[:5]}…; "
            "the transform and the booster describe different feature sets"
        )

    return frame[list(columns)]
