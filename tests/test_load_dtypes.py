"""The dtype policy in load.build_dtype_map.

Pure logic, so these run against a handful of fake column names rather than
parsing 683 MB. The most important assertion here is that TransactionAmt stays
at double precision: it is summed across hundreds of thousands of rows to
produce the headline USD figure, and a future refactor that swept it into the
float32 fallback would degrade that number silently.
"""

import pytest

from fraud_engine.data.load import build_dtype_map

N_ROWS = 100_000

LOAD_CFG = {
    "amount_dtype": "float64",
    "default_float_dtype": "float32",
    "category_max_ratio": 0.05,
}

COLUMNS = [
    "TransactionID",
    "isFraud",
    "TransactionDT",
    "TransactionAmt",
    "ProductCD",
    "DeviceInfo",
    "unique_per_row",
    "V1",
    "C1",
    "D1",
]

CARDINALITY = {
    "ProductCD": 5,  # ratio 0.00005 — far below the threshold
    "DeviceInfo": 2_000,  # ratio 0.02    — below
    "unique_per_row": N_ROWS,  # ratio 1.0     — must be refused
}


@pytest.fixture
def dtypes() -> dict[str, str]:
    return build_dtype_map(COLUMNS, CARDINALITY, N_ROWS, LOAD_CFG)


def test_amount_keeps_double_precision(dtypes):
    """float32 carries ~7 significant digits; the USD total is a large sum."""
    assert dtypes["TransactionAmt"] == "float64"


@pytest.mark.parametrize(
    ("column", "expected"),
    [
        ("TransactionID", "int32"),
        ("isFraud", "int8"),
        ("TransactionDT", "int32"),
    ],
)
def test_role_columns_are_not_swept_into_the_fallback(dtypes, column, expected):
    assert dtypes[column] == expected


@pytest.mark.parametrize("column", ["ProductCD", "DeviceInfo"])
def test_low_cardinality_text_becomes_category(dtypes, column):
    assert dtypes[column] == "category"


def test_high_cardinality_text_is_refused(dtypes):
    """A threshold that never rejects anything is not a threshold."""
    assert dtypes["unique_per_row"] == "str"


@pytest.mark.parametrize("column", ["V1", "C1", "D1"])
def test_unnamed_numeric_columns_take_the_default_float(dtypes, column):
    assert dtypes[column] == "float32"


def test_map_is_total(dtypes):
    """Every column gets an explicit dtype, so none falls back to inference."""
    assert set(dtypes) == set(COLUMNS)


def test_threshold_boundary_is_exclusive():
    """A ratio exactly at the threshold is refused, not converted."""
    cfg = LOAD_CFG | {"category_max_ratio": 0.05}
    at_threshold = build_dtype_map(["x"], {"x": 5_000}, N_ROWS, cfg)
    just_under = build_dtype_map(["x"], {"x": 4_999}, N_ROWS, cfg)
    assert at_threshold["x"] == "str"
    assert just_under["x"] == "category"


def test_identity_file_has_no_isfraud_column():
    """The map is built per file; identity carries no target."""
    dtypes = build_dtype_map(["TransactionID", "DeviceType"], {"DeviceType": 2}, N_ROWS, LOAD_CFG)
    assert "isFraud" not in dtypes


class TestGuards:
    def test_rejects_non_positive_row_count(self):
        with pytest.raises(ValueError, match="n_rows must be positive"):
            build_dtype_map(["V1"], {}, 0, LOAD_CFG)

    @pytest.mark.parametrize("bad_ratio", [0, 1.5, 5])
    def test_rejects_out_of_range_threshold(self, bad_ratio):
        """A config typo of 5 for 0.05 would categorise a unique-per-row column."""
        cfg = LOAD_CFG | {"category_max_ratio": bad_ratio}
        with pytest.raises(ValueError, match="category_max_ratio"):
            build_dtype_map(["V1"], {}, N_ROWS, cfg)

    def test_rejects_cardinality_for_unknown_columns(self):
        """Sample and file disagreeing would otherwise pass silently."""
        with pytest.raises(ValueError, match="absent from the header"):
            build_dtype_map(["V1"], {"ghost": 3}, N_ROWS, LOAD_CFG)
