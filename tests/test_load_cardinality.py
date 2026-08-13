"""Tests for load.measure_cardinality.

This one touches the filesystem, so each test writes its own CSV into pytest's
``tmp_path`` — a fresh directory per test, cleaned up automatically. No fixture
files in the repo, and no way for one test to affect another.
"""

import pytest

from fraud_engine.data.load import measure_cardinality

# Shaped so every assertion has something to bite on:
#   brand   low-cardinality text          → counted
#   note    text containing a blank       → the blank must not count as a value
#   amt     float                         → must be absent from the result
#   txn_id  integer                       → absent too; "id-like" is not "text"
SAMPLE_CSV = """\
brand,note,amt,txn_id
visa,alpha,1.50,1
amex,,2.00,2
visa,beta,3.25,3
visa,alpha,4.00,4
"""


@pytest.fixture
def csv_path(tmp_path):
    path = tmp_path / "sample.csv"
    path.write_text(SAMPLE_CSV)
    return path


def test_counts_distinct_values_in_text_columns(csv_path):
    assert measure_cardinality(csv_path, 10)["brand"] == 2  # visa, amex


@pytest.mark.parametrize("column", ["amt", "txn_id"])
def test_numeric_columns_are_absent(csv_path, column):
    """Presence in this map is how build_dtype_map tells text from numeric."""
    assert column not in measure_cardinality(csv_path, 10)


def test_nulls_are_not_counted_as_a_distinct_value(csv_path):
    """`note` holds alpha, beta and one blank — the blank is not a category.

    Pandas stores a missing value as code -1 rather than as an entry in the
    lookup table, so counting it would overstate the ratio.
    """
    assert measure_cardinality(csv_path, 10)["note"] == 2


def test_only_the_requested_rows_are_sampled(csv_path):
    """The sample size is honoured: one row sees `visa` but never `amex`."""
    assert measure_cardinality(csv_path, 1)["brand"] == 1


def test_returns_empty_when_no_text_columns(tmp_path):
    path = tmp_path / "numeric.csv"
    path.write_text("a,b\n1,2.5\n3,4.5\n")
    assert measure_cardinality(path, 10) == {}


@pytest.mark.parametrize("sample_rows", [0, -1])
def test_rejects_non_positive_sample(csv_path, sample_rows):
    """Zero rows gives every column cardinality 0 — a ratio below any threshold
    — which would silently turn every text column into a category."""
    with pytest.raises(ValueError, match="sample_rows must be positive"):
        measure_cardinality(csv_path, sample_rows)


def test_guard_runs_before_any_file_access(tmp_path):
    """Validation precedes I/O.

    Pinned deliberately: if the guard is ever moved below the read, this fails
    with FileNotFoundError instead of ValueError.
    """
    missing = tmp_path / "does-not-exist.csv"
    with pytest.raises(ValueError, match="sample_rows must be positive"):
        measure_cardinality(missing, 0)
