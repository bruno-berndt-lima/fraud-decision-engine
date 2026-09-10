"""Tests for the family and serving-tier registries.

Synthetic schemas, and only schemas: `resolve_families` and `resolve_tiers` read
column names from a parquet footer and never touch a row, so the fixtures here
carry no data at all. A frame with rows would test pandas.

The centre of gravity is the four guards. Every one of them only runs when
something has already gone wrong, which is exactly the code a smoke test never
reaches — and the failure they exist to catch (an arm that removes less than it
claims) looks like a good number rather than an error.
"""

from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from fraud_engine.features import registry
from fraud_engine.features.registry import (
    KEYS,
    TIER_0,
    TIER_1,
    TIER_2,
    TIER_3,
    TIER_SIZES,
    resolve_families,
    resolve_tiers,
)

# 295 inherited names, shaped like the real ones: the V-block reduction's
# survivors, the day-deltas, the counts, and the undocumented identity columns.
# The `vb_` ones are load-bearing twice — they are also what `resolve_families`
# resolves the V-block by.
TIER_0_NAMES = (
    *(f"vb_V{index}" for index in range(1, 235)),
    *(f"D{index}" for index in range(1, 16)),
    *(f"C{index}" for index in range(1, 15)),
    *(f"id_{index}" for index in range(34, 66)),
)


def write_schema(directory: Path, names) -> Path:
    """A train matrix carrying these columns and no rows."""
    table = pa.table({name: pa.array([], type=pa.float64()) for name in names})
    pq.write_table(table, directory / "train.parquet")
    return directory


def full_names() -> list[str]:
    """Every column of a matrix that matches what `features.md` publishes.

    Interleaved between tiers and reversed within each one. Both matter: laid
    out tier by tier in declaration order, a function that ignored the matrix
    and handed back its own constants would pass the order test by coincidence.
    """
    names = [
        *reversed(KEYS),
        *reversed(TIER_1),
        *TIER_0_NAMES[:100],
        *reversed(TIER_2),
        *TIER_0_NAMES[100:],
        *reversed(TIER_3),
    ]
    assert len(names) == sum(TIER_SIZES.values())
    return names


@pytest.fixture
def matrix(tmp_path: Path) -> Path:
    return write_schema(tmp_path, full_names())


# --- the declared constants ---------------------------------------------------


def test_declared_sizes_match_the_published_inventory():
    """Editing a tier without editing what the document promises is the drift.

    The three named tiers are written out by hand; `TIER_SIZES` is what
    `features.md` publishes. Nothing keeps them in step except this.
    """
    declared = {"tier_1": TIER_1, "tier_2": TIER_2, "tier_3": TIER_3, "keys": KEYS}

    assert {tier: len(columns) for tier, columns in declared.items()} == {
        tier: size for tier, size in TIER_SIZES.items() if tier != TIER_0
    }


def test_named_tiers_carry_no_duplicates_within_themselves():
    for columns in (TIER_1, TIER_2, TIER_3, KEYS):
        assert len(set(columns)) == len(columns)


# --- resolve_tiers, the happy path --------------------------------------------


def test_tiers_partition_the_matrix(matrix: Path):
    """Every column in exactly one tier, and nothing invented."""
    tiers = resolve_tiers(matrix)
    names = full_names()

    assigned = [column for columns in tiers.values() for column in columns]

    assert sorted(assigned) == sorted(names)
    assert len(assigned) == len(set(assigned))


def test_tier_sizes_are_what_the_document_publishes(matrix: Path):
    assert {tier: len(columns) for tier, columns in resolve_tiers(matrix).items()} == TIER_SIZES


def test_columns_come_back_in_matrix_order(matrix: Path):
    """Order is the matrix's, not the constant's.

    A tier handed back in declaration order would still partition correctly and
    would quietly reorder any frame built from it.
    """
    names = full_names()
    tiers = resolve_tiers(matrix)

    for columns in tiers.values():
        positions = [names.index(column) for column in columns]
        assert positions == sorted(positions)


def test_tier_zero_is_the_remainder_and_names_nothing(tmp_path: Path):
    """An unrecognised column lands in tier 0 without anyone declaring it.

    This is the documented rule — undocumented columns default to inherited —
    and it is what stops a column added later from being assumed servable.
    """
    names = [name if name != "C1" else "some_undocumented_column" for name in full_names()]
    tiers = resolve_tiers(write_schema(tmp_path, names))

    assert "some_undocumented_column" in tiers[TIER_0]
    assert len(tiers[TIER_0]) == TIER_SIZES[TIER_0]


def test_the_buildable_set_is_the_complement_of_tier_zero(matrix: Path):
    """What this project could reconstruct is everything tier 0 is not.

    The distinction E7 turns on: tier 3 is expensive to *serve* and perfectly
    possible to *build*, so a reproducibility arm that dropped it would be
    answering the serving question instead. Pinned as a property because it is a
    definition, and definitions drift silently.
    """
    tiers = resolve_tiers(matrix)

    buildable = set(tiers["tier_1"]) | set(tiers["tier_2"]) | set(tiers["tier_3"])

    assert buildable == set(full_names()) - set(tiers[TIER_0]) - set(tiers["keys"])
    assert len(buildable) == TIER_SIZES["tier_1"] + TIER_SIZES["tier_2"] + TIER_SIZES["tier_3"]


# --- resolve_tiers, the guards ------------------------------------------------


def test_absent_column_raises(tmp_path: Path):
    names = [name for name in full_names() if name != "hour"]

    with pytest.raises(ValueError, match="absent from the matrix"):
        resolve_tiers(write_schema(tmp_path, names))


def test_unaccounted_column_raises_rather_than_growing_tier_zero(tmp_path: Path):
    """A column nobody classified must stop the run, not enlarge the remainder.

    The remainder rule and the size assertion are not in tension: an *unknown*
    name is inherited, and a *count* that no longer matches the inventory means
    the document has stopped describing the matrix.
    """
    with pytest.raises(ValueError, match=r"do not match what features\.md publishes"):
        resolve_tiers(write_schema(tmp_path, [*full_names(), "brand_new_feature"]))


def test_overlapping_tiers_raise(matrix: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(registry, "TIER_2", (*TIER_2, "hour"))

    with pytest.raises(ValueError, match="is in both"):
        resolve_tiers(matrix)


def test_missing_matrix_raises(tmp_path: Path):
    with pytest.raises(FileNotFoundError):
        resolve_tiers(tmp_path)


# --- resolve_families ---------------------------------------------------------


def test_families_resolve_the_v_block_from_the_matrix(matrix: Path):
    """The prefix family's members come from the artifact, never from a list.

    Its membership is decided by a fitted correlation threshold, so a hardcoded
    list would drift from config with neither looking wrong.
    """
    families = resolve_families(matrix)

    assert families["vblock"] == tuple(name for name in full_names() if name.startswith("vb_"))
    assert len(families["vblock"]) == 234


def test_the_bare_reference_carries_no_columns(matrix: Path):
    assert resolve_families(matrix)["none"] == ()


def test_families_are_disjoint_and_all_engineered(matrix: Path):
    families = resolve_families(matrix)
    columns = [column for columns in families.values() for column in columns]

    assert len(columns) == len(set(columns))
    assert set(columns) <= set(full_names())


def test_families_do_not_reach_into_tier_zero_except_the_v_block(matrix: Path):
    """Every hand-built family is something this project could rebuild.

    The V-block is the exception and the reason the tier table has four values:
    its columns are a reduction *of* inherited data, so they stay inherited.
    """
    families = resolve_families(matrix)
    inherited = set(resolve_tiers(matrix)[TIER_0])

    for name, columns in families.items():
        if name in ("none", "vblock"):
            continue
        assert not set(columns) & inherited, name
