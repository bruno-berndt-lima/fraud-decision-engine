"""LightGBM — the challenger.

The rules engine is what this has to beat in dollars; ``logistic.py`` is the
reference point that says whether the extra complexity is earning its keep.

Categoricals are handed to LightGBM natively rather than one-hot encoded. A tree
splits a category into two *sets* of levels, so it can represent "these six
browsers behave alike" in one split where a linear model needs six coefficients.
The cost is that the split is fitted, and a level backed by three transactions
lets it memorise three transactions — which is why the vocabulary has a floor.
"""

from __future__ import annotations

import pandas as pd

from fraud_engine.features.encoders import MISSING

# Levels the training window never saw, and levels it saw too rarely to learn
# anything from, share one bucket. Separate from MISSING, which is a different
# fact about a row: the field did not arrive, rather than arrived unrecognised.
OTHER = "__other__"

SENTINELS = (OTHER, MISSING)


def fit_categories(train: pd.DataFrame, min_rows: int) -> dict[str, pd.Index]:
    """The category vocabulary each column is scored against, from train alone.

    Today every split file carries an identical vocabulary, because ``partition``
    sliced one frame and the dtype travelled with it. That vocabulary was built
    from the whole table — validation and test included — and **production has no
    whole table**. A request arriving with a browser version released after
    training has no code, and nothing currently decides what happens to it.

    Fitting here forces training to face what serving faces. The levels are the
    ones train saw at least ``min_rows`` times; everything else routes to a
    sentinel at apply time.

    **Rare training levels join the unseen ones in ``OTHER`` rather than getting
    their own.** If ``OTHER`` collected only the levels validation brings, it
    would carry no training rows at all and the model would score them against
    nothing. Folding the rare levels in is what gives the bucket real mass.

    Both sentinels are in every column's vocabulary whether or not train needed
    them. Train having no nulls in a column does not mean a request will not
    arrive without it, and the vocabulary is what ships.

    Levels are sorted, so the integer codes behind them depend on the training
    window and not on the order ``value_counts`` happened to break ties in.

    Args:
        train: Training rows. Columns of dtype ``category`` are the ones fitted;
            the matrices are the source of truth for which those are.
        min_rows: Occurrences in train below which a level is not learned
            separately.

    Returns:
        ``{column: levels}``, sentinels last. A column whose every level is rare
        comes back holding only the sentinels — nothing is dropped, but there is
        nothing left for a split to separate.
    """
    vocabulary = {}

    for column in train.select_dtypes("category").columns:
        counts = train[column].value_counts()
        frequent = sorted(counts[counts >= min_rows].index)
        vocabulary[column] = pd.Index([*frequent, *SENTINELS], dtype="object")

    return vocabulary
