"""Temporal split boundaries and the assignment of transactions to them.

Chronological, with a purged gap between train and validation: fraud labels
arrive weeks late via chargebacks, so training up to the validation boundary
assumes instant labels. See ``docs/problem-statement.md`` §3.3 and A1.

Nothing here is random — no seed, no shuffle, no stratification.
"""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

from fraud_engine.data.load import DEFAULT_CONFIG_PATH, load_config

log = logging.getLogger(__name__)

# Chronological. The single source of split identity and order — consumers
# iterate this rather than restating it.
SPLIT_NAMES = ("train", "val_fit", "val_cal", "test")


def resolve_boundaries(splits_cfg: dict) -> dict[str, tuple[int, int]]:
    """Turn the ``splits:`` config block into concrete day ranges.

    Pure — no file access — so the boundary policy is testable against a dict.
    Checks against the actual data belong downstream.

    Only starts and the gap are declared; every end is derived::

        train_end = val_fit_start - gap_days - 1
        val_fit_end = val_cal_start - 1
        val_cal_end = test_start - 1

    That is what makes ``docs/experiments.md`` E1 a one-value change:
    ``gap_days: 0`` extends train over the vacated days and leaves every
    evaluation range untouched. A literal ``train_end`` in config would state
    the same fact twice, and the two would drift.

    The gap is not returned — it is the days no split claims. Callers that
    assign by "which range contains this day" get gap rows unclaimed for free,
    and ``gap_days: 0`` needs no special case.

    Args:
        splits_cfg: The ``splits:`` block of ``config/config.yaml``.

    Returns:
        ``{name: (first_day, last_day)}`` per ``SPLIT_NAMES``, chronological.
        Ends are **inclusive**: ``(1, 90)`` puts day 90 in train and day 91 not.
        Every consumer must agree — this is where off-by-ones come from.

    Raises:
        KeyError: If a required config key is absent.
        ValueError: If the boundaries are internally incoherent.
    """
    gap_days = splits_cfg["gap_days"]
    train_start = splits_cfg["train_start"]
    val_fit_start = splits_cfg["val_fit_start"]
    val_cal_start = splits_cfg["val_cal_start"]
    test_start = splits_cfg["test_start"]
    test_end = splits_cfg["test_end"]

    # A negative gap reverses the purge rather than shortening it: train_end
    # lands past val_fit_start and the splits overlap. Row counts and start
    # ordering both stay valid, so nothing downstream notices.
    if gap_days < 0:
        overlap_end = val_fit_start - gap_days - 1
        raise ValueError(
            f"gap_days={gap_days} is negative: train would end on day {overlap_end}, "
            f"past val_fit_start={val_fit_start}, putting days "
            f"{val_fit_start}-{overlap_end} in both splits. The gap is subtracted "
            f"from the end of train, so a negative value extends train forward "
            f"instead of purging less. Use gap_days: 0 to run without a purge "
            f"(docs/experiments.md E1)."
        )

    # Ends derive from the next split's start, so one out-of-order value yields
    # an inverted range like (121, 99) — matches no rows, raises nothing.
    if not train_start < val_fit_start < val_cal_start < test_start:
        raise ValueError(
            f"split starts must be strictly increasing, got "
            f"train_start={train_start}, val_fit_start={val_fit_start}, "
            f"val_cal_start={val_cal_start}, test_start={test_start}. Ends derive "
            f"from the next split's start, so an out-of-order value produces an "
            f"inverted range that matches nothing."
        )

    # The only end not derived from a later split, so nothing else constrains it.
    if test_end < test_start:
        raise ValueError(
            f"test_end={test_end} is before test_start={test_start}: the test split "
            f"would be empty. test_end is declared rather than derived, so nothing "
            f"else catches this."
        )

    train_end = val_fit_start - gap_days - 1
    val_fit_end = val_cal_start - 1
    val_cal_end = test_start - 1

    # The opposite failure to a negative gap: a gap wider than the training
    # window drives train_end below train_start, possibly negative. An empty
    # range trains on an empty frame rather than crashing.
    if train_end < train_start:
        raise ValueError(
            f"gap_days={gap_days} leaves no training data: train would run "
            f"{train_start} to {train_end}. The gap cannot exceed "
            f"val_fit_start - train_start - 1 (={val_fit_start - train_start - 1})."
        )

    boundaries = {
        "train": (train_start, train_end),
        "val_fit": (val_fit_start, val_fit_end),
        "val_cal": (val_cal_start, val_cal_end),
        "test": (test_start, test_end),
    }

    # The literal above keeps the start/end pairing readable; this pins it to
    # SPLIT_NAMES so a rename in one place fails here, not in a consumer.
    assert tuple(boundaries) == SPLIT_NAMES, (
        f"boundaries {tuple(boundaries)} do not match SPLIT_NAMES {SPLIT_NAMES}"
    )

    return boundaries


def assign_splits(days: pd.Series, boundaries: dict[str, tuple[int, int]]) -> pd.Series:
    """Label each row with the split its day falls in.

    Takes the ``day`` column rather than the frame: this needs one fact per row,
    and a function that cannot see ``isFraud`` cannot leak it.

    Ends are inclusive, matching ``resolve_boundaries``. Days no split claims
    come back null — the purge gap, but also anything outside
    ``[train_start, test_end]``. Telling those apart is ``validate_splits``'
    job, not this one.

    Args:
        days: The ``day`` column. Index and length are preserved.
        boundaries: From ``resolve_boundaries``.

    Returns:
        An ordered ``category`` Series named ``split``, aligned to ``days``,
        with ``SPLIT_NAMES`` as categories and null for unclaimed days.
        Categorical so an invalid label is unrepresentable rather than merely
        wrong; ordered so ``groupby`` sorts chronologically without a key.

    Raises:
        ValueError: If ``boundaries`` does not match ``SPLIT_NAMES``, or if its
            ranges are inverted or overlapping.
    """
    if tuple(boundaries) != SPLIT_NAMES:
        raise ValueError(
            f"boundaries keys {tuple(boundaries)} do not match SPLIT_NAMES "
            f"{SPLIT_NAMES}. The order is load-bearing: the returned dtype is "
            f"ordered, so chronology comes from this sequence."
        )

    # Checked against the ranges, not against the rows: an overlap only shows up
    # in the data if some row happens to fall inside it, so a row-based check
    # passes on an unlucky sample and on an empty Series.
    previous_name, previous_end = None, None
    for name, (start, end) in boundaries.items():
        if start > end:
            raise ValueError(f"{name}=({start}, {end}) is inverted and matches no rows.")
        if previous_end is not None and start <= previous_end:
            raise ValueError(
                f"boundaries overlap: {name} starts on day {start}, on or before "
                f"{previous_name} ends on day {previous_end}."
            )
        previous_name, previous_end = name, end

    labels = pd.Series(index=days.index, dtype="object")
    for name, (start, end) in boundaries.items():
        # Two comparisons rather than .between() so the inclusivity is on the
        # page instead of in a pandas default.
        labels[(days >= start) & (days <= end)] = name

    return labels.astype(pd.CategoricalDtype(SPLIT_NAMES, ordered=True)).rename("split")


def validate_splits(
    days: pd.Series,
    split: pd.Series,
    boundaries: dict[str, tuple[int, int]],
) -> pd.Series:
    """Check an assignment against the data it was made from.

    The counterpart to the guards in ``resolve_boundaries``: those compare the
    config against itself, these compare it against the table. Runs before the
    split artifact is written, so a violation leaves nothing on disk.

    The load-bearing check is the last one. ``assign_splits`` returns null for
    any day no split claims — the purge gap, but equally anything outside the
    declared span — and this is the only place those are told apart.

    Overlap needs no check here: a row carries one label, so belonging to two
    splits is unrepresentable rather than merely untested.

    Args:
        days: The ``day`` column.
        split: Labels from ``assign_splits``, aligned to ``days``.
        boundaries: From ``resolve_boundaries``.

    Returns:
        ``split`` unchanged, so this composes into a pipeline expression.

    Raises:
        ValueError: On any violation, naming the split at fault.
    """
    if not days.index.equals(split.index):
        raise ValueError("days and split must share an index; they are not aligned.")
    if days.empty:
        raise ValueError("no rows to split.")

    train_start, train_end = boundaries["train"]
    val_fit_start = boundaries["val_fit"][0]
    test_end = boundaries["test"][1]

    # Data wider than the span silently drops rows; span wider than the data
    # means the config describes days that do not exist. Both are config errors.
    first, last = int(days.min()), int(days.max())
    if (first, last) != (train_start, test_end):
        raise ValueError(
            f"boundaries span days {train_start}-{test_end} but the data spans "
            f"{first}-{last}. The declared span must match the table exactly."
        )

    for name, (start, end) in boundaries.items():
        claimed = days[split == name]
        if claimed.empty:
            raise ValueError(f"{name} is empty: no rows fall in days {start}-{end}.")

        # Verifies the assignment against the boundaries instead of trusting
        # assign_splits to have applied them.
        if claimed.min() < start or claimed.max() > end:
            raise ValueError(
                f"{name} contains days {int(claimed.min())}-{int(claimed.max())}, "
                f"outside its declared range {start}-{end}."
            )

    # Every unclaimed row must be a gap row. Anything else is a day the
    # boundaries do not cover, and nothing upstream can see it.
    unclaimed = days[split.isna()]
    stray = unclaimed[(unclaimed <= train_end) | (unclaimed >= val_fit_start)]
    if not stray.empty:
        raise ValueError(
            f"{len(stray)} row(s) belong to no split and are outside the purge gap "
            f"(days {train_end + 1}-{val_fit_start - 1}): days "
            f"{sorted(int(day) for day in stray.unique())[:10]}."
        )

    return split


# The purge gap is not a split, but it is part of the partition and its size is
# a reported number: how much training data the purge costs (experiments.md E1).
GAP_LABEL = "gap"

# Chronological. The gap always sits between train and val_fit — resolve_boundaries
# derives exactly one, from gap_days — so the order follows from SPLIT_NAMES.
SECTION_ORDER = (SPLIT_NAMES[0], GAP_LABEL, *SPLIT_NAMES[1:])


def summarize(days: pd.Series, split: pd.Series, is_fraud: pd.Series) -> pd.DataFrame:
    """Count rows and positives per section of the partition.

    Phase 06's calibrator choice depends on how many positives ``val_cal``
    holds — isotonic regression is unstable on a few hundred — so these counts
    are an artifact of this phase, not a debugging print.

    The gap gets a row. It is not a split, but "what the purge discards" is a
    number the write-up needs, and it is the quantity E1 trades away.

    Args:
        days: The ``day`` column.
        split: Labels from ``assign_splits``, aligned to ``days``.
        is_fraud: The ``isFraud`` column, aligned to ``days``.

    Returns:
        A frame indexed by ``SECTION_ORDER`` — the four splits plus ``gap``, in
        chronological order — with ``first_day``, ``last_day``, ``n_days``,
        ``rows``, ``frauds`` and ``fraud_rate``. An empty gap (``gap_days: 0``)
        is a row of zeros rather than a missing one.

    Raises:
        ValueError: If the inputs are not aligned, or if any *split* holds no
            positives. The gap may hold none — it is discarded.
    """
    if not (days.index.equals(split.index) and days.index.equals(is_fraud.index)):
        raise ValueError("days, split and is_fraud must share an index.")

    section = split.astype("object").fillna(GAP_LABEL)
    frame = pd.DataFrame({"day": days, "section": section, "is_fraud": is_fraud})

    summary = (
        frame.groupby("section")
        .agg(
            first_day=("day", "min"),
            last_day=("day", "max"),
            rows=("is_fraud", "size"),
            frauds=("is_fraud", "sum"),
        )
        .reindex(SECTION_ORDER)
    )

    # reindex fills a section absent from the data with NaN. Zero rows is the
    # honest count; the day span stays null because there are no days.
    summary["rows"] = summary["rows"].fillna(0).astype(int)
    summary["frauds"] = summary["frauds"].fillna(0).astype(int)
    for column in ("first_day", "last_day"):
        summary[column] = summary[column].astype("Int64")
    summary["n_days"] = summary["last_day"] - summary["first_day"] + 1
    summary["fraud_rate"] = summary["frauds"] / summary["rows"]

    barren = [name for name in SPLIT_NAMES if summary.loc[name, "frauds"] == 0]
    if barren:
        raise ValueError(
            f"split(s) hold no positives: {barren}. PR-AUC is undefined on a slice "
            f"with no fraud and Phase 06 cannot fit a calibrator on one, so this "
            f"fails here rather than in Phase 03. Widen the split or move the boundary."
        )

    return summary[["first_day", "last_day", "n_days", "rows", "frauds", "fraud_rate"]]


def main(config_path: Path = DEFAULT_CONFIG_PATH) -> None:
    """Cut the interim table into temporal splits and record what is in each.

    Wiring only — every decision lives in the functions this calls. Invoked by
    ``make features``' prerequisite as ``python -m fraud_engine.data.splits``.

    Both checks run before either file is written, so a bad split leaves no
    artifact behind: ``validate_splits`` on the structure, ``summarize`` on
    whether every split has positives to score.

    Args:
        config_path: Path to ``config.yaml``. Defaults to a repo-root-relative
            location, which is where the Makefile runs from.
    """
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    config = load_config(config_path)
    paths = config["paths"]

    boundaries = resolve_boundaries(config["splits"])

    # Three columns of 438. The split is a function of time alone; isFraud is
    # here only to be counted, never to decide where a boundary falls.
    frame = pd.read_parquet(paths["interim"], columns=["TransactionID", "day", "isFraud"])

    split = assign_splits(frame["day"], boundaries)
    validate_splits(frame["day"], split, boundaries)
    summary = summarize(frame["day"], split, frame["isFraud"])

    # Gap rows are kept with a null label rather than dropped: the file then
    # describes the whole partition, including what the purge cost, and a
    # downstream join cannot quietly lose rows it was never told about.
    ids = pd.DataFrame({"TransactionID": frame["TransactionID"], "split": split})
    ids = ids.sort_values("TransactionID", ignore_index=True)

    splits_path = Path(paths["splits"])
    splits_path.parent.mkdir(parents=True, exist_ok=True)
    ids.to_parquet(splits_path, index=False)

    summary_path = Path(paths["split_summary"])
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(summary_path)

    log.info("boundaries (inclusive): %s", {k: f"{a}-{b}" for k, (a, b) in boundaries.items()})
    log.info("\n%s", summary.to_string(formatters={"fraud_rate": "{:.2%}".format}))
    log.info("wrote %s — %d rows", splits_path, len(ids))
    log.info("wrote %s", summary_path)


if __name__ == "__main__":
    main()
