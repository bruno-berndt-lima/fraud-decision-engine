"""Tests for load.read_typed.

The point of this function is the header guard, not the read. Its two failure
directions are not equally dangerous, and the tests are weighted accordingly:
a dtype for a column that does not exist would make read_csv complain anyway,
but a column with *no* dtype fails silently — pandas infers it, and the table
comes back untyped with nothing to indicate the policy was bypassed.
"""

import pytest

from fraud_engine.data.load import read_typed

SAMPLE_CSV = """\
brand,amt,count
visa,1.50,3
amex,2.00,7
visa,3.25,1
"""

DTYPES = {"brand": "category", "amt": "float64", "count": "int32"}


@pytest.fixture
def csv_path(tmp_path):
    path = tmp_path / "sample.csv"
    path.write_text(SAMPLE_CSV)
    return path


def test_applies_the_declared_dtypes(csv_path):
    dtypes = read_typed(csv_path, DTYPES).dtypes.astype(str).to_dict()
    assert dtypes == DTYPES


def test_returns_every_row(csv_path):
    assert len(read_typed(csv_path, DTYPES)) == 3


def test_rejects_a_column_with_no_declared_dtype(csv_path):
    """The silent failure: pandas would infer it and the policy is bypassed."""
    incomplete = {k: v for k, v in DTYPES.items() if k != "amt"}
    with pytest.raises(ValueError, match="fall back to inference"):
        read_typed(csv_path, incomplete)


def test_error_names_the_offending_column(csv_path):
    """A guard that says only 'mismatch' costs you the debugging session."""
    incomplete = {k: v for k, v in DTYPES.items() if k != "amt"}
    with pytest.raises(ValueError, match=r"\['amt'\]"):
        read_typed(csv_path, incomplete)


def test_rejects_a_dtype_for_a_column_that_is_absent(csv_path):
    """Signals the map was built against a different header."""
    with pytest.raises(ValueError, match="absent from the file"):
        read_typed(csv_path, DTYPES | {"ghost": "float32"})


def test_guard_runs_before_the_full_read(tmp_path):
    """Only the header is read while validating.

    Pinned deliberately: a file whose header is fine but whose body cannot be
    parsed under the map must still fail on the *body*, proving the guard did
    not consume the whole file to reach its verdict.
    """
    path = tmp_path / "bad_body.csv"
    path.write_text("brand,amt\nvisa,not_a_number\n")
    with pytest.raises(ValueError) as excinfo:
        read_typed(path, {"brand": "category", "amt": "float64"})
    assert "fall back to inference" not in str(excinfo.value)
    assert "absent from the file" not in str(excinfo.value)
