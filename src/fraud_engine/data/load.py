"""Raw CSVs to one typed, joined parquet table.

The only module that touches ``data/raw/``. Every later stage reads the parquet
this produces, which is what makes them re-runnable without a 683 MB re-parse.

Nothing here is *fitted* to the data: no imputation, no scaling, no encoding,
no aggregates. Anything learned from data before the Phase 02 split is leakage,
so it belongs in ``features/``, not here.
"""

from __future__ import annotations

# Columns whose dtype follows from their role rather than from the family rules.
# Checked before the fallback so a role column can never be swept into the
# default float type.
_ROLE_DTYPES = {
    "TransactionID": "int32",
    "isFraud": "int8",
    "TransactionDT": "int32",
}

# TransactionAmt is deliberately not in the table above: its precision is a
# config decision, not a constant. It is summed across hundreds of thousands of
# rows to produce the headline USD figure, and float32 carries only ~7
# significant digits, so rounding error would accumulate in that sum.
# See docs/problem-statement.md §4.
_AMOUNT_COLUMN = "TransactionAmt"


def build_dtype_map(
    columns: list[str],
    cardinality: dict[str, int],
    n_rows: int,
    load_cfg: dict,
) -> dict[str, str]:
    """Map every column name to the pandas dtype it should be read as.

    Pure — no file access — so the whole dtype policy is unit-testable against a
    handful of fake column names rather than by parsing the real file.

    Args:
        columns: Every column name in the file, from its header.
        cardinality: Distinct-value counts for the *text* columns only, measured
            on a sample. Columns absent from this mapping are treated as numeric.
        n_rows: Total row count of the full file. The ratio compares sampled
            cardinality against total size, so it is conservative by
            construction: a sample can only undercount distinct values.
        load_cfg: The ``load:`` block of ``config/config.yaml``.

    Returns:
        A *total* map: every name in ``columns`` gets an entry, including the
        text columns that fail the category threshold, which are pinned to
        ``str`` rather than omitted. Nothing is left to pandas' inference, so
        the caller can assert coverage against the header.

    Raises:
        ValueError: If ``n_rows`` is not positive, if the category threshold is
            outside (0, 1], or if ``cardinality`` names a column that is not in
            ``columns`` — which would mean the sample and the file disagree.
    """
    if n_rows <= 0:
        raise ValueError(f"n_rows must be positive, got {n_rows}")

    max_ratio = load_cfg["category_max_ratio"]
    if not 0 < max_ratio <= 1:
        raise ValueError(
            f"category_max_ratio must be a fraction in (0, 1], got {max_ratio}. "
            "A value above 1 would make every text column a category, including "
            "one distinct per row — the case the threshold exists to prevent."
        )

    unknown = set(cardinality) - set(columns)
    if unknown:
        raise ValueError(
            f"cardinality names columns absent from the header: {sorted(unknown)}. "
            "The sample and the file being read do not match."
        )

    amount_dtype = load_cfg["amount_dtype"]
    default_float = load_cfg["default_float_dtype"]

    dtypes: dict[str, str] = {}
    for column in columns:
        if column in _ROLE_DTYPES:
            dtypes[column] = _ROLE_DTYPES[column]
        elif column == _AMOUNT_COLUMN:
            dtypes[column] = amount_dtype
        elif column in cardinality:
            # Few distinct values relative to rows: `category` stores one small
            # integer per row plus a lookup table. Above the threshold that
            # trade reverses, so leave it as plain text.
            dtypes[column] = "category" if cardinality[column] / n_rows < max_ratio else "str"
        else:
            dtypes[column] = default_float

    return dtypes
