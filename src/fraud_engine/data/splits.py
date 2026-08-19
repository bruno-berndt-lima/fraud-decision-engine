"""Temporal split boundaries and the assignment of transactions to them.

Chronological, with a purged gap between train and validation: fraud labels
arrive weeks late via chargebacks, so training up to the validation boundary
assumes instant labels. See ``docs/problem-statement.md`` §3.3 and A1.

Nothing here is random — no seed, no shuffle, no stratification.
"""

from __future__ import annotations

import pandas as pd

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
