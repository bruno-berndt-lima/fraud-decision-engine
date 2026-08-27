"""Velocity — how much this card has been transacting, and how recently.

The only family in this phase whose relationship with fraud is **monotone**: on
train, fraud risk falls steadily as the gap since the card's previous
transaction grows. E4 registers that a linear probe can only fit monotone
shapes, so this is the family with a real chance of registering on it. The
measured quintiles are there, not here.

Computed **causally over the whole frame**, gap rows included. A trailing window
looks backwards only, and days 121-140 legitimately reach back into days 114-120
— purged for immature *labels*, but the transactions themselves happened and a
production system would count them. Computing per split would reset every card's
history at the boundary and invent a train/serve skew that production does not
have.

Two things were built, measured, and left out: **burst ratios** — a short window
against a long one, the scale-free form of a count — which turn over at the top
rather than rising, and a **first-sighting indicator**, whose lift turned out to
sit on top of the slowest recency quintile's, so it belongs at the slow end of
that scale rather than as a column of its own. E4 carries both measurements.

Serving cost is real and is the point of E3: these need the card's history at
request time, which a single API call does not carry. **Tier 3** on the serving
table — an online store, not a shipped lookup.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from fraud_engine.features.encoders import levels

# card1 is the closest thing to a card identifier this dataset has, and it is not
# one: many transactions share a level. So these are "this card type's" velocity,
# and the raw counts are confounded by how popular a bin is — which is why the
# top count bucket turns over instead of continuing to rise.
ENTITY = "card1"

# Trailing windows, in seconds. Shortest first.
WINDOWS = {"1h": 3_600, "24h": 86_400, "7d": 604_800}

RECENCY = f"vel_recency_{ENTITY}"
COLUMNS = (*(f"vel_n{window}_{ENTITY}" for window in WINDOWS), RECENCY)


def _window_count(moments: np.ndarray, seconds: int) -> np.ndarray:
    """How many of ``moments`` fall in each one's own trailing window.

    ``moments`` must be one entity's transaction times, ascending. The window is
    half-open — ``(t - seconds, t]`` — so a transaction counts itself and
    anything at the very edge is included exactly once.

    Done by binary search rather than a rolling window because the alignment has
    to survive ties. Position ``i`` looks up where ``t - seconds`` would be
    inserted and counts forward from there, so the answer depends on a row's
    place in the order, never on its timestamp being unique. Many rows share a
    ``TransactionDT`` with another; a pandas time-indexed rolling window unwinds
    by that timestamp and would scramble them.
    """
    first_inside = np.searchsorted(moments, moments - seconds, side="right")
    return np.arange(len(moments)) - first_inside + 1


def trailing_counts(frame: pd.DataFrame, windows: dict[str, int]) -> dict[str, pd.Series]:
    """How many transactions this card had in each trailing window, this one included.

    The frame must already be in causal order — ``order_by_time``'s job. Each
    entity's rows are then ascending in time, which is what makes the search
    above valid, and no row can see its own future.

    Windows are measured against ``TransactionDT``, so "24 hours" means 86,400
    seconds rather than "the previous 24 rows".

    Returns:
        ``{column: Series}`` aligned to ``frame``'s index.
    """
    moments = frame["TransactionDT"].to_numpy()
    # levels() rather than a raw groupby, so a null card1 is one entity here and
    # in every other family. A plain groupby would drop those rows instead.
    groups = frame.groupby(levels(frame[ENTITY]), observed=True, sort=False).indices

    counts = {}
    for window, seconds in windows.items():
        column = np.empty(len(frame), dtype="float32")
        for positions in groups.values():
            column[positions] = _window_count(moments[positions], seconds)
        counts[f"vel_n{window}_{ENTITY}"] = pd.Series(column, index=frame.index)

    return counts


def recency(frame: pd.DataFrame, first_seen_gap_days: float) -> pd.Series:
    """Log seconds since this card's previous transaction.

    Log, because the raw gap spans seconds to months and a linear model reading
    it would treat one extra second at the top of that range as it does one at
    the bottom. On the log scale the quintile lifts are monotone.

    A card's first transaction has no predecessor. It takes
    ``first_seen_gap_days`` rather than a null or a flag: measured on train, a
    first sighting carries about the same risk as the slowest recency quintile,
    so the honest place for it is the slow end of this scale. That also keeps the
    family null-free.
    """
    gap = frame.groupby(levels(frame[ENTITY]), observed=True, sort=False)["TransactionDT"].diff()
    return np.log1p(gap.fillna(first_seen_gap_days * 86_400)).astype("float32")


def add_velocity_features(frame: pd.DataFrame, velocity_cfg: dict) -> pd.DataFrame:
    """The velocity family, over every row in causal order.

    Unlike the fitted families this needs no training window at all: a trailing
    count is a function of a row's own past, not of a statistic estimated from
    one. There is nothing here to leak, provided the order is right — which is
    why the tests pin the ordering rather than a fit/apply boundary.
    """
    built = trailing_counts(frame, WINDOWS)
    built[RECENCY] = recency(frame, velocity_cfg["first_seen_gap_days"])
    return frame.assign(**built)
