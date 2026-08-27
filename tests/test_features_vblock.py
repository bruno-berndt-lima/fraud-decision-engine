"""Tests for V-block reduction.

Three claims carry the weight, and they are in the order the roadmap fixes.

*Missingness first, and structurally.* Grouping by null pattern is a fact about
which source produced a column, so it must not depend on a fitted parameter or
on which rows it is shown.

*Correlation second, on train only.* This is the statistic that decides which
columns get dropped. Measuring it on validation would let the evaluation set
choose its own features — a subtler leak than a fitted median, because nothing
about the output would look wrong.

*And the choice must survive the apply.* The names come out of the fit;
recomputing anything from the frame being scored is how a different null pattern
in validation silently renames a column.
"""

from pathlib import Path

import numpy as np
import pandas as pd

from fraud_engine.features.vblock import (
    PREFIX,
    add_vblock_features,
    apply_fitted,
    cluster,
    fit,
    nan_pattern_groups,
    presence_frame,
    write_tables,
)

CFG = {"correlation_threshold": 0.9, "presence_threshold": 0.95, "min_observed_rows": 2}

# The tests exercise the reduction, not the real block, so they name their own.
BLOCK = ("V1", "V2")


def block(**columns) -> pd.DataFrame:
    """A frame standing in for the V block, plus the split label the family reads."""
    built = pd.DataFrame(columns)
    if "split" not in built:
        built["split"] = "train"
    return built


# ------------------------------------------------------------------------------
# nan_pattern_groups — structural, not fitted
# ------------------------------------------------------------------------------
def test_columns_sharing_a_null_pattern_land_together():
    frame = block(
        V1=[1.0, None, 3.0],
        V2=[4.0, None, 6.0],  # same pattern as V1
        V3=[None, 8.0, 9.0],  # different
    )

    groups = nan_pattern_groups(frame, ("V1", "V2", "V3"))

    assert sorted(map(sorted, groups)) == [["V1", "V2"], ["V3"]]


def test_the_same_values_with_different_nulls_do_not_group():
    """Grouping is about missingness, never about correlation. These two columns
    are identical wherever both are present."""
    frame = block(V1=[1.0, 2.0, None], V2=[1.0, 2.0, 3.0])

    assert len(nan_pattern_groups(frame, ("V1", "V2"))) == 2


def test_grouping_needs_no_training_window():
    """No fitted parameter, so any slice gives the same answer for the rows it
    holds — which is why this may run before the split is considered."""
    frame = block(V1=[1.0, None, 3.0, None], V2=[4.0, None, 6.0, None])

    assert nan_pattern_groups(frame.head(2), ("V1", "V2")) == [["V1", "V2"]]


# ------------------------------------------------------------------------------
# cluster — greedy, head-first
# ------------------------------------------------------------------------------
def test_perfectly_correlated_columns_collapse_to_one():
    frame = block(V1=[1.0, 2.0, 3.0, 4.0], V2=[2.0, 4.0, 6.0, 8.0], V3=[9.0, 1.0, 8.0, 2.0])

    clusters = cluster(frame, ["V1", "V2", "V3"], 0.9)

    assert ["V1", "V2"] in clusters
    assert ["V3"] in clusters


def test_negative_correlation_counts_as_the_same_variable():
    """A column and its negation carry one piece of information between them."""
    frame = block(V1=[1.0, 2.0, 3.0, 4.0], V2=[-1.0, -2.0, -3.0, -4.0])

    assert cluster(frame, ["V1", "V2"], 0.9) == [["V1", "V2"]]


def test_the_head_of_a_cluster_is_the_one_kept():
    """The representative is chosen by position, not by a tie-break nobody could
    later justify — so the surviving name is predictable from column order."""
    frame = block(V7=[1.0, 2.0, 3.0], V8=[2.0, 4.0, 6.0])

    assert cluster(frame, ["V7", "V8"], 0.9)[0][0] == "V7"
    assert cluster(frame, ["V8", "V7"], 0.9)[0][0] == "V8"


# ------------------------------------------------------------------------------
# presence flags
# ------------------------------------------------------------------------------
def test_a_flag_records_whether_the_group_was_present():
    frame = block(V1=[1.0, None, 3.0])

    assert list(presence_frame(frame, ["V1"])["V1"]) == [1, 0, 1]


def test_a_group_that_is_never_null_gets_no_flag():
    """A constant column is a free coefficient fitted on nothing."""
    frame = block(V1=[1.0, 2.0, 3.0, 4.0], V2=[1.0, None, 3.0, None])

    assert fit(frame, CFG, BLOCK)["presence"] == ["V2"]


def test_a_flag_needs_enough_rows_on_its_rarer_side():
    """Two of the real NaN groups are null on about a dozen rows out of three
    hundred thousand. A flag for those is a coefficient fitted on noise."""
    frame = block(V1=[1.0] * 9 + [None], V2=[1.0, None] * 5)

    assert fit(frame, {**CFG, "min_observed_rows": 3}, BLOCK)["presence"] == ["V2"]


# ------------------------------------------------------------------------------
# the fit/apply boundary
# ------------------------------------------------------------------------------
def test_validation_rows_choose_neither_the_columns_nor_the_fill():
    """The subtle one. Correlation decides which columns are dropped, so a fit
    that saw validation would let the evaluation set pick its own features — and
    nothing about the output would look wrong."""
    train = block(
        V1=[1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
        V2=[2.0, 4.0, 6.0, 8.0, 10.0, 12.0],
        split=["train"] * 6,
    )
    # Validation where the two are anticorrelated and on a wildly different scale.
    contaminated = pd.concat(
        [
            train,
            block(V1=[1.0, 2.0, 3.0], V2=[900.0, 600.0, 300.0], split=["val_fit"] * 3),
        ],
        ignore_index=True,
    )

    assert fit(train[train["split"] == "train"], CFG, BLOCK)["representatives"] == ["V1"]
    assert (
        add_vblock_features(contaminated, CFG, BLOCK)[f"{PREFIX}V1"].iloc[0]
        == add_vblock_features(train, CFG, BLOCK)[f"{PREFIX}V1"].iloc[0]
    )
    assert f"{PREFIX}V2" not in add_vblock_features(contaminated, CFG, BLOCK).columns


def test_the_flag_names_come_from_the_fit_not_from_the_frame_being_scored():
    """Recomputing the grouping at apply time would let a different null pattern
    in validation rename a column, and the two frames would disagree about what
    they contain."""
    train = block(V1=[1.0, None, 3.0, None], V2=[1.0, 2.0, 3.0, 4.0], split=["train"] * 4)
    scored = pd.concat(
        [train, block(V1=[5.0, 6.0], V2=[None, None], split=["val_fit"] * 2)],
        ignore_index=True,
    )

    built = add_vblock_features(scored, CFG, BLOCK)

    assert f"{PREFIX}present_V1" in built.columns
    assert f"{PREFIX}present_V2" not in built.columns


def test_the_family_is_null_free_float32_behind_its_prefix():
    """`build_features` promises families make their own fill decisions — and the
    original name has to survive, because Phase 07 must say which Vesta column a
    contribution belongs to."""
    frame = block(V1=[1.0, None, 3.0, 5.0], V2=[9.0, 1.0, None, 2.0])

    built = add_vblock_features(frame, CFG, BLOCK)
    emitted = [column for column in built.columns if column.startswith(PREFIX)]

    assert built[emitted].notna().all().all()
    assert built[f"{PREFIX}V1"].dtype == "float32"
    assert any(column.removeprefix(PREFIX) == "V1" for column in emitted)


def test_a_gap_is_filled_with_the_training_median():
    train = block(V1=[1.0, 3.0, 5.0, None], V2=[1.0, 1.0, 2.0, 2.0], split=["train"] * 4)

    assert add_vblock_features(train, CFG, BLOCK)[f"{PREFIX}V1"].iloc[3] == np.float32(3.0)


def test_the_input_frame_is_not_mutated():
    original = block(V1=[1.0, None, 3.0], V2=[3.0, 1.0, 2.0])
    apply_fitted(original, fit(original, CFG, BLOCK))

    assert not any(column.startswith(PREFIX) for column in original.columns)


# ------------------------------------------------------------------------------
# persistence
# ------------------------------------------------------------------------------
def test_the_reduction_is_recorded_as_an_artifact(tmp_path: Path):
    """The threshold in config describes the outcome; this file pins it."""
    fitted = fit(block(V1=[1.0, None, 3.0, 5.0], V2=[9.0, 1.0, 4.0, 2.0]), CFG, BLOCK)
    write_tables(fitted, tmp_path / "vblock.parquet")

    written = pd.read_parquet(tmp_path / "vblock.parquet")

    assert list(written.columns) == ["column", "role", "median"]
    assert set(written["role"]) <= {"representative", "presence"}
    assert all(name.startswith(PREFIX) for name in written["column"])


def test_the_raw_block_does_not_survive_the_reduction():
    """A matrix carrying both V1 and vb_V1 has not been reduced, it has been
    duplicated — and the pair is perfectly correlated, which is the worst case
    for the linear reference model and waste for the tree."""
    frame = block(V1=[1.0, None, 3.0, 5.0], V2=[9.0, 1.0, 4.0, 2.0])

    built = add_vblock_features(frame, CFG, BLOCK)

    assert not set(BLOCK) & set(built.columns)
    assert f"{PREFIX}V1" in built.columns


def test_a_column_with_no_training_rows_is_not_kept():
    """It has no median, so filling it would leave nulls in a family that
    promises none — and there is nothing to fill it from in the first place."""
    frame = block(V1=[None, None, None, None], V2=[1.0, 2.0, 3.0, 4.0])

    built = add_vblock_features(frame, CFG, BLOCK)
    emitted = [column for column in built.columns if column.startswith(PREFIX)]

    assert f"{PREFIX}V1" not in emitted
    assert built[emitted].notna().all().all()
