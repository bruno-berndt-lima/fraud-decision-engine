"""Frequency encoding — the first family with fitted state.

How common a level is, measured on the training rows and joined onto every
split. This is **rarity, not identity**: it deliberately discards which card or
which address a row carries and keeps only how often that value appears. H3's
fixed-volume lifts for these same columns are about identity, and do not transfer
here.

Fitted on train alone, per the project's encoder invariant. The invariant is
conservative in this case and knowingly so: frequency encoding reads no labels,
so the purge gap's label-maturity argument does not apply to it, and production
would count against all history. One training window is easier to defend than
two.

No target encoding here, and none planned for this family. That technique reads
labels and would need out-of-fold fitting; this one does not, so it needs no
folds either.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

# The identifiers logistic.py left out. P_emaildomain is absent because the probe
# already one-hots it, R_emaildomain because most of its train rows are null.
FREQUENCY_COLUMNS = ("card1", "card2", "card3", "card5", "addr1", "addr2", "DeviceInfo")

# Null is a level, not an absence. Missingness is signal in this dataset —
# has_identity and prepare()'s own __missing__ category already treat it so — and
# encoding it is what keeps the output null-free for build_features' contract.
MISSING = "__missing__"

# A level train never saw is the extreme of rarity, not a special case. Zero says
# exactly that, and on a scale where every real value is positive it also makes a
# separate "unseen" indicator redundant: a tree splits at zero, and a linear model
# could use neither.
UNSEEN_FREQUENCY = 0.0


def encoded_name(column: str) -> str:
    """The output column for ``column``'s frequency."""
    return f"freq_{column}"


COLUMNS = tuple(encoded_name(column) for column in FREQUENCY_COLUMNS)


def levels(values: pd.Series) -> pd.Series:
    """A column as lookup keys, with null promoted to a level of its own.

    Every fit and every apply in the project goes through here — this module's
    and ``aggregations``' — so all of them agree on how a level is spelled. That
    matters more than how it is spelled: ``card1`` arrives as a float and
    13926.0 has to be the same key on both sides of every join. Two families
    disagreeing on the spelling would silently group the same card differently.
    """
    return values.astype("object").fillna(MISSING).astype(str)


def fit_frequencies(train: pd.DataFrame, columns: tuple[str, ...]) -> dict[str, pd.Series]:
    """How common each level is, as a share of the training rows.

    Rates rather than counts. A count means "this many rows in the window we
    happened to fit on", so the same table refitted on a longer window says
    something different about an unchanged world; a share does not. That matters
    because this table is a serving artifact that outlives its fitting window —
    and because E1 refits on a longer one.

    Args:
        train: The training rows, and only those.
        columns: Source columns to encode.

    Returns:
        ``{column: Series}``, indexed by level and valued as a share of ``train``.
    """
    return {column: levels(train[column]).value_counts().div(len(train)) for column in columns}


def apply_frequencies(frame: pd.DataFrame, tables: dict[str, pd.Series]) -> pd.DataFrame:
    """Join the fitted rates onto every row.

    Applied to all splits, unrestricted — it is the *fit* that is confined to the
    training window, never the transform. A level absent from the table is one
    train never saw and takes ``UNSEEN_FREQUENCY``.

    Returns:
        A copy of ``frame`` with one ``freq_*`` column per table, as ``float32``.
    """
    return frame.assign(
        **{
            encoded_name(column): levels(frame[column])
            .map(table)
            .fillna(UNSEEN_FREQUENCY)
            .astype("float32")
            for column, table in tables.items()
        }
    )


def add_frequency_features(frame: pd.DataFrame) -> pd.DataFrame:
    """The frequency family: fitted on ``frame``'s train rows, applied to all of it.

    Takes no config, deliberately. The one candidate knob is which columns to
    encode, and that cannot live in YAML — ``COLUMNS`` is imported by the family
    registry, so a config edit could name columns no code produces. The remaining
    constants are design decisions with stated reasons, not tuning knobs.

    Reaches the training window through ``frame["split"] == "train"`` rather than
    by partitioning first. That is the discipline ``build_features`` documents,
    and the reason the split label rides this far down the stage.
    """
    train = frame[frame["split"] == "train"]
    return apply_frequencies(frame, fit_frequencies(train, FREQUENCY_COLUMNS))


def write_tables(tables: dict[str, pd.Series], path: Path | str) -> None:
    """Persist the fitted rates — the artifact a served model has to carry.

    Long form, one row per level, so the file describes itself and a serving
    process can load it without knowing which columns were encoded.

    This is what makes the family tier 2 rather than tier 3 on the serving table:
    a static lookup shipped beside the model, needing no online store and no
    entity history. Writing it is what turns that from a claim into a file.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    rows = pd.concat(
        [
            table.rename("frequency").rename_axis("level").reset_index().assign(column=column)
            for column, table in tables.items()
        ],
        ignore_index=True,
    )
    rows[["column", "level", "frequency"]].to_parquet(path, index=False)
