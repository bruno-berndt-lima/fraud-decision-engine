"""Tests for the interim table schema.

Each test builds a minimal valid frame and breaks exactly one thing, so a
failure names the rule that stopped working. The two least obvious tests are at
the bottom: unlisted columns must be *allowed* but never *dropped*.
"""

import pandas as pd
import pytest
from pandera.errors import SchemaErrors

from fraud_engine.data.validate import validate_interim


def make_frame(**overrides) -> pd.DataFrame:
    """A minimal frame satisfying every rule, with one column swappable."""
    columns = {
        "TransactionID": pd.Series([1, 2], dtype="int32"),
        "isFraud": pd.Series([0, 1], dtype="int8"),
        "TransactionAmt": pd.Series([68.5, 29.0], dtype="float64"),
        "TransactionDT": pd.Series([86_400, 90_000], dtype="int32"),
        "day": pd.Series([1, 1], dtype="int32"),
        "hour": pd.Series([0, 1], dtype="int32"),
        "weekday": pd.Series([1, 1], dtype="int32"),
        "has_identity": pd.Series([True, False]),
        "ProductCD": pd.Series(["W", "C"], dtype="category"),
        "V1": pd.Series([1.0, None], dtype="float32"),
        "C1": pd.Series([None, 2.0], dtype="float32"),
        "D1": pd.Series([0.0, None], dtype="float32"),
    }
    columns.update(overrides)
    return pd.DataFrame(columns)


def test_a_valid_frame_passes():
    assert len(validate_interim(make_frame())) == 2


def test_returns_the_frame_unchanged():
    """It composes into a pipeline, so it must not alter what it validates."""
    original = make_frame()
    pd.testing.assert_frame_equal(validate_interim(original), original)


def test_missing_values_are_allowed_in_the_vesta_blocks():
    """Missingness is signal here — asserting non-null would fail on good data."""
    frame = make_frame(V1=pd.Series([None, None], dtype="float32"))
    assert len(validate_interim(frame)) == 2


@pytest.mark.parametrize(
    ("label", "override"),
    [
        ("negative amount", {"TransactionAmt": pd.Series([-1.0, 29.0], dtype="float64")}),
        ("zero amount", {"TransactionAmt": pd.Series([0.0, 29.0], dtype="float64")}),
        ("isFraud out of range", {"isFraud": pd.Series([0, 2], dtype="int8")}),
        ("hour out of range", {"hour": pd.Series([0, 24], dtype="int32")}),
        ("weekday out of range", {"weekday": pd.Series([1, 7], dtype="int32")}),
        ("negative day", {"day": pd.Series([1, -1], dtype="int32")}),
        ("duplicate TransactionID", {"TransactionID": pd.Series([1, 1], dtype="int32")}),
        ("null in has_identity", {"has_identity": pd.Series([True, None], dtype="object")}),
        ("unknown ProductCD", {"ProductCD": pd.Series(["W", "Z"], dtype="category")}),
        ("null TransactionAmt", {"TransactionAmt": pd.Series([68.5, None], dtype="float64")}),
    ],
)
def test_rejects_violations(label, override):
    with pytest.raises(SchemaErrors):
        validate_interim(make_frame(**override))


@pytest.mark.parametrize(
    ("label", "override"),
    [
        (
            "amount downcast to float32",
            {"TransactionAmt": pd.Series([68.5, 29.0], dtype="float32")},
        ),
        ("id widened to int64", {"TransactionID": pd.Series([1, 2], dtype="int64")}),
        ("V block as float64", {"V1": pd.Series([1.0, 2.0], dtype="float64")}),
    ],
)
def test_rejects_wrong_dtypes(label, override):
    """coerce=False, so the schema enforces the dtype policy rather than fixing it.

    With coercion on, a float32 TransactionAmt would be silently upcast and
    reported as valid — undoing the precision decision the cost model depends on.
    """
    with pytest.raises(SchemaErrors):
        validate_interim(make_frame(**override))


def test_reports_every_violation_not_just_the_first():
    """lazy=True. On a ~438-column table, one-error-per-run means one 683 MB
    reload per mistake."""
    frame = make_frame(
        TransactionAmt=pd.Series([-1.0, 29.0], dtype="float64"),
        hour=pd.Series([0, 24], dtype="int32"),
    )
    with pytest.raises(SchemaErrors) as excinfo:
        validate_interim(frame)
    assert len(excinfo.value.failure_cases) >= 2


def test_unlisted_columns_are_allowed():
    """The M* and id_* families are deliberately not in the schema."""
    frame = make_frame()
    frame["M1"] = pd.Series(["T", "F"], dtype="category")
    frame["id_01"] = pd.Series([0.0, None], dtype="float32")
    assert len(validate_interim(frame)) == 2


def test_unlisted_columns_are_not_dropped():
    """Guards against strict="filter", which would silently delete ~400 features.

    Allowing extra columns and removing them look identical from a pass/fail
    view — only checking the output columns tells them apart.
    """
    frame = make_frame()
    frame["M1"] = pd.Series(["T", "F"], dtype="category")
    assert "M1" in validate_interim(frame).columns
