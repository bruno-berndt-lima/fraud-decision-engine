"""Tests for reading back the tables the pipeline fitted.

Every reader here inverts a writer in another module, and the two can drift apart
without anything failing — the codes a vocabulary carries are positional, so a read
that recovered the same levels differently would produce correct-looking codes standing
for the wrong levels. `docs/serving.md` §8 registers that as this phase's named risk,
and what follows is the round trip that pins each pair.

No parquet from the pipeline is read: every table here is fitted by the real fitting
function on a frame small enough to check by eye, then written by the real writer.
"""

import numpy as np
import pandas as pd
import pytest

from fraud_engine.features import aggregations, encoders, vblock
from fraud_engine.models.train import (
    apply_categories,
    fit_categories,
    fit_medians,
    write_categories,
    write_medians,
)
from fraud_engine.serving.artifacts import (
    read_amount_stats,
    read_categories,
    read_frequencies,
    read_medians,
    read_vblock,
)

MIN_CATEGORY_ROWS = 2
PRIOR_STRENGTH = 10
VBLOCK_CFG = {
    "correlation_threshold": 0.90,
    "presence_threshold": 0.95,
    "min_observed_rows": 3,
}


@pytest.fixture
def frame() -> pd.DataFrame:
    """A small table carrying one column of each kind the readers have to restore."""
    rows = 40
    generator = np.random.default_rng(0)

    return pd.DataFrame(
        {
            "ProductCD": pd.Categorical(["W", "C", "H", "W"] * (rows // 4)),
            "card4": pd.Categorical(["visa", "mastercard", None, "visa"] * (rows // 4)),
            "card1": [float(1000 + index % 7) for index in range(rows)],
            "addr1": [float(300 + index % 5) for index in range(rows)],
            "TransactionAmt": generator.uniform(10, 500, rows),
            "C1": generator.normal(size=rows),
        }
    )


@pytest.fixture
def vframe() -> pd.DataFrame:
    """The whole V block, so the reduction can be applied and not merely compared.

    `read_vblock` restores `source` as all 339 columns — the file records which
    survived, never which were offered — so a frame carrying a handful of them could
    not exercise the apply at all.
    """
    rows = 60
    generator = np.random.default_rng(1)
    columns = {}

    for index, name in enumerate(vblock.V_COLUMNS):
        values = generator.normal(size=rows)
        # Three null patterns, so the reduction has groups to find and flags to keep.
        if index % 3 == 1:
            values[: rows // 2] = np.nan
        elif index % 3 == 2:
            values[: rows // 4] = np.nan
        columns[name] = values.astype("float32")

    return pd.DataFrame(columns)


# ---- the category vocabulary -------------------------------------------------


def test_the_vocabulary_survives_the_round_trip(frame, tmp_path):
    fitted = fit_categories(frame, MIN_CATEGORY_ROWS)
    path = tmp_path / "categories.parquet"
    write_categories(fitted, path)

    restored = read_categories(path)

    assert set(restored) == set(fitted)
    for column, levels in fitted.items():
        assert list(restored[column]) == list(levels)


def test_the_restored_vocabulary_types_a_column_identically(frame, tmp_path):
    """Same levels in the same order is not enough: the dtype is the contract.

    A `CategoricalDtype` over `str` categories compares unequal to one over identically
    spelled `object` categories. The codes would match and the dtypes would not, which
    is a served column that is not the column the model was fitted on.
    """
    fitted = fit_categories(frame, MIN_CATEGORY_ROWS)
    path = tmp_path / "categories.parquet"
    write_categories(fitted, path)

    restored = read_categories(path)

    pd.testing.assert_frame_equal(
        apply_categories(frame, restored), apply_categories(frame, fitted)
    )


def test_codes_that_are_not_a_dense_range_are_refused(frame, tmp_path):
    path = tmp_path / "categories.parquet"
    write_categories(fit_categories(frame, MIN_CATEGORY_ROWS), path)

    rows = pd.read_parquet(path)
    rows.loc[rows.index[0], "code"] = 99
    rows.to_parquet(path, index=False)

    with pytest.raises(ValueError, match="dense range"):
        read_categories(path)


def test_a_vocabulary_with_nowhere_to_put_an_unknown_value_is_refused(frame, tmp_path):
    path = tmp_path / "categories.parquet"
    write_categories(fit_categories(frame, MIN_CATEGORY_ROWS), path)

    rows = pd.read_parquet(path)
    kept = rows[rows["level"] != "__other__"].copy()
    # Re-densify the codes, so the failure under test is the absent sentinel and not
    # the gap removing it left behind.
    kept["code"] = kept.groupby("column", sort=False).cumcount()
    kept.to_parquet(path, index=False)

    with pytest.raises(ValueError, match="__other__"):
        read_categories(path)


# ---- the fill values ---------------------------------------------------------


def test_the_medians_survive_the_round_trip(frame, tmp_path):
    fitted = fit_medians(frame, ["card1", "addr1", "TransactionAmt", "C1"])
    path = tmp_path / "medians.parquet"
    write_medians(fitted, path)

    pd.testing.assert_series_equal(read_medians(path), fitted, check_names=False)


def test_a_null_fill_value_is_refused(frame, tmp_path):
    path = tmp_path / "medians.parquet"
    write_medians(fit_medians(frame, ["card1", "C1"]), path)

    rows = pd.read_parquet(path)
    rows.loc[rows.index[0], "median"] = None
    rows.to_parquet(path, index=False)

    with pytest.raises(ValueError, match="null fill values"):
        read_medians(path)


# ---- the fitted families -----------------------------------------------------


def test_the_frequency_tables_encode_identically_after_a_round_trip(frame, tmp_path):
    fitted = encoders.fit_frequencies(frame, ("card1", "addr1"))
    path = tmp_path / "encoders.parquet"
    encoders.write_tables(fitted, path)

    pd.testing.assert_frame_equal(
        encoders.apply_frequencies(frame, read_frequencies(path)),
        encoders.apply_frequencies(frame, fitted),
    )


def test_the_amount_statistics_score_identically_after_a_round_trip(frame, tmp_path):
    fitted = aggregations.fit_amount_stats(frame, ("card1", "addr1"), PRIOR_STRENGTH)
    path = tmp_path / "amount_stats.parquet"
    aggregations.write_tables(fitted, path)

    pd.testing.assert_frame_equal(
        aggregations.apply_amount_stats(frame, read_amount_stats(path)),
        aggregations.apply_amount_stats(frame, fitted),
    )


def test_an_entity_table_without_its_fallback_is_refused(frame, tmp_path):
    path = tmp_path / "amount_stats.parquet"
    aggregations.write_tables(
        aggregations.fit_amount_stats(frame, ("card1",), PRIOR_STRENGTH), path
    )

    rows = pd.read_parquet(path)
    rows[rows["level"] != aggregations.GLOBAL].to_parquet(path, index=False)

    with pytest.raises(ValueError, match=aggregations.GLOBAL):
        read_amount_stats(path)


def test_the_v_block_reduction_emits_the_same_columns_after_a_round_trip(vframe, tmp_path):
    fitted = vblock.fit(vframe, VBLOCK_CFG)
    path = tmp_path / "vblock.parquet"
    vblock.write_tables(fitted, path)

    pd.testing.assert_frame_equal(
        vblock.apply_fitted(vframe, read_vblock(path)),
        vblock.apply_fitted(vframe, fitted),
    )


def test_a_kept_column_with_no_fill_value_is_refused(vframe, tmp_path):
    path = tmp_path / "vblock.parquet"
    vblock.write_tables(vblock.fit(vframe, VBLOCK_CFG), path)

    rows = pd.read_parquet(path)
    rows.loc[rows[rows["role"] == "representative"].index[0], "median"] = None
    rows.to_parquet(path, index=False)

    with pytest.raises(ValueError, match="no fill value"):
        read_vblock(path)


def test_an_unknown_role_is_refused(vframe, tmp_path):
    path = tmp_path / "vblock.parquet"
    vblock.write_tables(vblock.fit(vframe, VBLOCK_CFG), path)

    rows = pd.read_parquet(path)
    rows.loc[rows.index[0], "role"] = "kept-ish"
    rows.to_parquet(path, index=False)

    with pytest.raises(ValueError, match="unknown roles"):
        read_vblock(path)
