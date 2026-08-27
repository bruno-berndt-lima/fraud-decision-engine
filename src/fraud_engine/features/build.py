from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

from fraud_engine.data.load import DEFAULT_CONFIG_PATH, load_config
from fraud_engine.data.splits import SPLIT_NAMES
from fraud_engine.features import aggregations, amounts, encoders, velocity

log = logging.getLogger(__name__)


def attach_splits(frame: pd.DataFrame, splits: pd.DataFrame) -> pd.DataFrame:
    """Join each transaction's split label onto the interim table.

    An inner join whose result must match *both* inputs in length. That is what
    proves the two files describe the same transactions: a left join would
    record a stale ``splits.parquet`` as a null label, which is
    indistinguishable from a legitimately purged gap row.

    Args:
        frame: The interim table.
        splits: ``TransactionID`` and ``split`` as ``splits.py`` wrote them —
            gap rows present, carrying a null label.

    Returns:
        ``frame`` with a ``split`` column.

    Raises:
        ValueError: If either side carries duplicate IDs, or the two do not
            cover the same set of transactions.
    """
    merged = frame.merge(splits, on="TransactionID", how="inner", validate="one_to_one")

    if len(merged) != len(frame) or len(merged) != len(splits):
        raise ValueError(
            f"interim and splits describe different transactions: "
            f"{len(frame)} interim rows, {len(splits)} split rows, {len(merged)} matched"
        )

    return merged


def order_by_time(frame: pd.DataFrame) -> pd.DataFrame:
    """Sort into causal order: ``TransactionDT``, ties broken by ``TransactionID``.

    Every trailing-window feature reads rows in this order, so the tie-break is
    part of the feature definition and not a formatting choice. ``TransactionDT``
    is a whole number of seconds and a meaningful share of rows collide on one,
    several at a time, so without a second key those rows' window counts would
    depend on the order the interim table happened to arrive in.

    ``TransactionID`` is unique, so the ordering is total and the result does
    not depend on the sort algorithm's stability.

    Args:
        frame: Any frame carrying ``TransactionDT`` and ``TransactionID``.

    Returns:
        ``frame`` sorted, index reset — the surviving positional index is a row
        number in time order and nothing else.
    """
    return frame.sort_values(["TransactionDT", "TransactionID"], ignore_index=True)


def build_features(frame: pd.DataFrame, features_cfg: dict) -> pd.DataFrame:
    """Add the engineered columns.

    The seam every feature family lands in, and its position between
    ``order_by_time`` and ``partition`` is the contract:

    - **Causal families** — velocity, time since the card's previous
      transaction — read the whole frame, gap rows included. Those rows are
      purged for having immature *labels*; the transactions themselves happened,
      and a day-121 window legitimately reaches back into them.
    - **Fitted families** — frequency encoding, entity aggregates — restrict the
      *fit* to ``frame["split"] == "train"`` and apply the result to every row.
      Reaching train rows by partitioning first would serve them perfectly well
      and silently break the causal families, resetting every entity's history
      at the split boundary. So the label rides along as a column, and a mask is
      how train rows are selected.

    ``partition`` runs last. Always.

    **Families arrive null-free.** Where a column can be missing, the family
    decides its own fill here, in the module that knows what missing means for
    it — a velocity of "no history" is 0, not a median. The probe's imputer is a
    net that should never fire: inheriting its median would mean the evaluation
    harness quietly choosing a value production would never produce.

    Args:
        frame: Every transaction, in the order ``order_by_time`` established.
        features_cfg: The ``features:`` config block, passed on to each family.

    Returns:
        ``frame`` with the engineered columns added.
    """
    frame = amounts.add_amount_features(frame, features_cfg["amounts"])
    frame = encoders.add_frequency_features(frame)
    frame = aggregations.add_amount_stats(frame, features_cfg["aggregations"])
    return velocity.add_velocity_features(frame, features_cfg["velocity"])


def partition(frame: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """One matrix per split, chronological, gap rows dropped.

    Gap rows carry a null label. They were needed above — a trailing window
    reaches back into their history — and have no matured label to train or
    score on, so dropping them here is this function's purpose rather than a
    side effect.

    Selecting by name would discard an unrecognised label just as quietly, so
    the rows are accounted for: everything not kept must be a gap row.

    Args:
        frame: Every transaction, carrying ``split``.

    ``split`` itself does not survive. The column and the filename are not two
    facts that can check each other — both come from the filter directly above,
    so keeping the column states one fact twice. ``day`` is the independent
    evidence that a matrix holds the rows it claims to.

    Returns:
        ``{split: matrix}`` in ``SPLIT_NAMES`` order, each indexed from zero
        and carrying no ``split`` column — exactly what lands on disk.

    Raises:
        ValueError: If any row carries a label that is neither null nor a
            member of ``SPLIT_NAMES``.
    """
    matrices = {
        name: frame[frame["split"] == name].drop(columns="split").reset_index(drop=True)
        for name in SPLIT_NAMES
    }

    kept = sum(len(matrix) for matrix in matrices.values())
    purged = int(frame["split"].isna().sum())

    if kept + purged != len(frame):
        unknown = set(frame["split"].dropna().unique()) - set(SPLIT_NAMES)
        raise ValueError(f"unrecognised split labels: {sorted(unknown)}")

    return matrices


def write_matrices(matrices: dict[str, pd.DataFrame], features_dir: Path) -> None:
    """Write ``{features_dir}/{split}.parquet``, one file per matrix.

    Every matrix is already in memory before the first write, so a run that
    fails part-way cannot leave behind a set of files that disagree about which
    rows they hold — the property ``load.py`` gets by validating ahead of its
    write, here for free.

    ``index=False``, because the index is a row number ``partition`` handed out
    and not something the next stage should be able to read meaning into.

    Args:
        matrices: ``{split: matrix}`` as ``partition`` returned them.
        features_dir: Directory to write into. Created if absent.
    """
    features_dir.mkdir(parents=True, exist_ok=True)

    for name, matrix in matrices.items():
        path = features_dir / f"{name}.parquet"
        matrix.to_parquet(path, index=False)
        log.info("wrote %s — %d rows, %d columns", path, len(matrix), matrix.shape[1])


def main(config_path: Path = DEFAULT_CONFIG_PATH) -> None:
    """Build the feature matrices from the interim table and the splits.

    Wiring only — every decision lives in the functions this calls. Invoked by
    ``make features`` as ``python -m fraud_engine.features.build``.

    The interim table is read whole rather than by column list: the matrices are
    self-contained, so no later stage has to reach back past this one to
    assemble its inputs.

    Args:
        config_path: Path to ``config.yaml``. Defaults to a repo-root-relative
            location, which is where the Makefile runs from.
    """
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    config = load_config(config_path)
    paths = config["paths"]

    frame = pd.read_parquet(paths["interim"])
    splits = pd.read_parquet(paths["splits"], columns=["TransactionID", "split"])

    frame = build_features(order_by_time(attach_splits(frame, splits)), config["features"])
    matrices = partition(frame)

    log.info("purged %d gap rows", len(frame) - sum(len(matrix) for matrix in matrices.values()))
    write_matrices(matrices, Path(paths["features_dir"]))

    # Refitted rather than threaded out of build_features, so every family keeps
    # the same (frame, cfg) -> frame shape. value_counts over seven columns is
    # milliseconds and deterministic, so the two fits cannot disagree.
    encoders.write_tables(
        encoders.fit_frequencies(frame[frame["split"] == "train"], encoders.FREQUENCY_COLUMNS),
        paths["encoders"],
    )
    log.info("wrote %s", paths["encoders"])

    aggregations.write_tables(
        aggregations.fit_amount_stats(
            frame[frame["split"] == "train"],
            aggregations.ENTITY_COLUMNS,
            config["features"]["aggregations"]["prior_strength"],
        ),
        paths["amount_stats"],
    )
    log.info("wrote %s", paths["amount_stats"])


if __name__ == "__main__":
    main()
