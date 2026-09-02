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

from pathlib import Path

import pandas as pd

from fraud_engine.data.splits import SPLIT_NAMES
from fraud_engine.features.encoders import MISSING

# Levels the training window never saw, and levels it saw too rarely to learn
# anything from, share one bucket. Separate from MISSING, which is a different
# fact about a row: the field did not arrive, rather than arrived unrecognised.
OTHER = "__other__"

# Keys, the label, and the split axis. `day` and `TransactionDT` are the same
# exclusion twice at different resolutions: a model handed either learns the
# timeline instead of fraud, and on a temporal split that reads as skill.
EXCLUDED_COLUMNS = ("TransactionID", "TransactionDT", "isFraud", "day")

# What a training run reads. TEST is absent and has to be typed to be loaded,
# the same shape as report.DEFAULT_SPLITS: the invariant says test is touched
# once at the very end, and a default that reads it makes that a matter of
# remembering rather than of asking.
TRAINING_SPLITS = ("train", "val_fit", "val_cal")

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


def apply_categories(frame: pd.DataFrame, vocabulary: dict[str, pd.Index]) -> pd.DataFrame:
    """Re-level every categorical against the fitted vocabulary.

    This is where the split files stop carrying a vocabulary built from the
    whole table. Afterwards each categorical holds exactly the levels
    ``fit_categories`` learned, in that order, so a code means the same thing in
    every split — which is the property that lets the model be trained on one and
    scored on another at all.

    Order matters in the body: nulls become ``MISSING`` *before* the membership
    test, because a null is not an unrecognised value. Testing first would route
    every missing field into ``OTHER`` and merge two facts the model should be
    able to tell apart.

    Args:
        frame: Any split's matrix.
        vocabulary: ``{column: levels}`` from ``fit_categories``.

    Returns:
        A new frame — the input is not modified — whose fitted columns are
        null-free and carry the fitted dtype.

    Raises:
        ValueError: If the frame carries a categorical the fit never saw. Left
            alone it would keep its whole-frame vocabulary, which is the exact
            defect this function exists to remove, and nothing downstream would
            look wrong.
        KeyError: If a fitted column is absent from the frame.
    """
    unfitted = set(frame.select_dtypes("category").columns) - set(vocabulary)
    if unfitted:
        raise ValueError(
            f"categorical columns with no fitted vocabulary: {sorted(unfitted)}; "
            "they would keep the vocabulary the whole table gave them"
        )

    prepared = frame.copy()

    for column, levels in vocabulary.items():
        values = prepared[column].astype("object").fillna(MISSING)
        prepared[column] = values.where(values.isin(levels), OTHER).astype(
            pd.CategoricalDtype(levels)
        )

    return prepared


def feature_columns(frame: pd.DataFrame) -> list[str]:
    """Everything the model may see: the matrix, less the columns it may not.

    ``logistic.py`` names its features positively, and gives the reason: trusting
    a drop rule puts the label one edit away from becoming a feature. That reason
    is right and the mechanism does not survive three hundred columns — a stale
    allow-list drops new features silently, which is the worse failure, since a
    model quietly trained on less than it was given looks fine.

    The guard is what makes the inversion honest: every excluded name must be
    present. A deny-list describing a table that no longer exists is one rename
    away from letting the label through, and this is what turns that into an
    error instead of a very good score.

    Args:
        frame: A split matrix.

    Returns:
        Feature names, in matrix order.

    Raises:
        ValueError: If any excluded column is missing — the deny-list and the
            matrix have drifted apart.
    """
    missing = [column for column in EXCLUDED_COLUMNS if column not in frame.columns]
    if missing:
        raise ValueError(
            f"excluded columns absent from the matrix: {missing}; "
            "the deny-list no longer describes this table"
        )

    return [column for column in frame.columns if column not in EXCLUDED_COLUMNS]


def load_split_matrices(
    features_dir: Path | str, splits: tuple[str, ...] = TRAINING_SPLITS
) -> dict[str, pd.DataFrame]:
    """Each split's matrix, read whole and kept apart.

    Apart, because LightGBM early-stops against a validation set and needs it as
    its own object. ``evaluate.load_matrices`` returns one concatenated frame for
    the probe, which fits on a mask; concatenating here only to split again would
    also point ``models`` at a module that already imports ``models``.

    Read whole rather than by column list: the matrices are self-contained by
    design, and naming columns here would be a second place for the feature set
    to be decided.

    Args:
        features_dir: Directory holding ``{split}.parquet``.
        splits: Which to read. Defaults to the three a training run needs —
            naming ``test`` is possible, and has to be deliberate.

    Returns:
        ``{split: matrix}`` in the order given.

    Raises:
        ValueError: If a name is not a split.
        FileNotFoundError: If a split's matrix is absent.
    """
    unknown = set(splits) - set(SPLIT_NAMES)
    if unknown:
        raise ValueError(f"not splits: {sorted(unknown)}")

    features_dir = Path(features_dir)

    return {name: pd.read_parquet(features_dir / f"{name}.parquet") for name in splits}


def write_categories(vocabulary: dict[str, pd.Index], path: Path | str) -> None:
    """Persist the vocabulary — the model cannot be served without it.

    A trained booster stores which features are categorical and the sets of
    codes its splits test, but not what those codes *mean*. That mapping lives
    here, and a serving process without it would hand LightGBM integers derived
    from whatever levels happened to be in the request batch.

    This is the tier-2 artifact ``features.md`` promised for the category
    vocabulary, in the same long form as the frequency tables and the V-block
    reduction, so the file describes itself.

    The code is written out rather than left implicit in the row order. Row order
    survives a parquet round trip, but the codes *are* the contract with the
    model — a mapping that a re-sorted read could silently change is not a
    contract.

    Args:
        vocabulary: ``{column: levels}`` from ``fit_categories``.
        path: Destination. Parent directories are created.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    rows = pd.DataFrame(
        [
            {"column": column, "code": code, "level": level}
            for column, levels in vocabulary.items()
            for code, level in enumerate(levels)
        ]
    )
    rows.to_parquet(path, index=False)
