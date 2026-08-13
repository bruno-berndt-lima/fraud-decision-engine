"""Schema for the interim table, enforced as a pipeline step.

Runs inside ``load.main`` *before* the parquet is written, so a violation means
no artifact exists at all — combined with ``.DELETE_ON_ERROR:`` in the Makefile,
nothing downstream can proceed on data that failed its contract.

Two things this deliberately does **not** do:

- **Coerce.** ``coerce=False`` throughout. Coercion would silently cast a wrong
  dtype into the expected one, which is precisely the failure the dtype policy
  in ``load.build_dtype_map`` exists to prevent. A mismatch must raise.
- **Assert distributional properties.** No "amount within three standard
  deviations". That is a statistic fitted to the data, and Phase 01 fits
  nothing. Schemas assert structural facts; drift is Phase 09's job.
"""

from __future__ import annotations

import pandas as pd
import pandera.pandas as pa

# Verified against data/raw/train_transaction.csv, not taken from the spec.
_PRODUCT_CODES = ("C", "H", "R", "S", "W")

INTERIM_SCHEMA = pa.DataFrameSchema(
    {
        # ---- Identity and target -------------------------------------------
        # A duplicate TransactionID would mean the identity join multiplied
        # rows, fabricating transactions. join_identity guards this too; the
        # cost of checking twice is nil and the failure modes differ.
        "TransactionID": pa.Column("int32", nullable=False, unique=True),
        "isFraud": pa.Column("int8", pa.Check.isin([0, 1]), nullable=False),
        # ---- Cost model inputs ---------------------------------------------
        # The threshold is C_fp / (amount + fee), so a zero or negative amount
        # makes the policy nonsense rather than merely wrong. Observed minimum
        # in the raw data is 0.251, so > 0 is a real constraint, not a rubber
        # stamp. See docs/problem-statement.md §4.
        "TransactionAmt": pa.Column("float64", pa.Check.gt(0), nullable=False),
        # ---- Time ------------------------------------------------------------
        "TransactionDT": pa.Column("int32", pa.Check.gt(0), nullable=False),
        # These three are derived, so a violation means the arithmetic in
        # add_time_columns is wrong — a second line of defence on the numbers
        # the Phase 02 temporal split is built from.
        "day": pa.Column("int32", pa.Check.ge(0), nullable=False),
        "hour": pa.Column("int32", pa.Check.in_range(0, 23), nullable=False),
        "weekday": pa.Column("int32", pa.Check.in_range(0, 6), nullable=False),
        # ---- Derived flags ---------------------------------------------------
        # Computed from the merge indicator, so it can never legitimately be null.
        "has_identity": pa.Column(bool, nullable=False),
        # ---- Categoricals ----------------------------------------------------
        "ProductCD": pa.Column("category", pa.Check.isin(_PRODUCT_CODES), nullable=False),
        # ---- Vesta feature blocks -------------------------------------------
        # One regex entry covers a whole family: ^V\d+$ alone validates 339
        # columns. Enumerating ~438 columns would be unmaintainable and unread.
        #
        # All three are nullable, and that is the point rather than an omission:
        # this data is missing-heavy by nature and the missingness is signal
        # LightGBM splits on. Asserting non-null here would fail on correct data.
        r"^V\d+$": pa.Column("float32", nullable=True, regex=True),
        r"^C\d+$": pa.Column("float32", nullable=True, regex=True),
        r"^D\d+$": pa.Column("float32", nullable=True, regex=True),
    },
    # strict=False: the M* and id_* families are intentionally unlisted. Their
    # dtypes depend on the cardinality threshold at load time, so pinning them
    # here would couple the schema to a config value and break on a legitimate
    # change. Never use strict="filter" — it silently DROPS unlisted columns,
    # which would delete 400 features without a word.
    strict=False,
    coerce=False,
    name="interim transactions",
)


def validate_interim(df: pd.DataFrame) -> pd.DataFrame:
    """Validate the interim table, raising on any violation.

    Args:
        df: The joined, time-augmented table, before it is written to parquet.

    Returns:
        The same frame, unchanged, so this composes into a pipeline expression.

    Raises:
        pandera.errors.SchemaErrors: Listing *every* violation found, not just
            the first. On a table this wide that difference matters — you fix
            the whole set in one pass instead of rerunning a 683 MB load to
            discover the next problem.
    """
    return INTERIM_SCHEMA.validate(df, lazy=True)
