"""Raw CSVs to one typed, joined parquet table.

The only module that touches ``data/raw/``. Every later stage reads the parquet
this produces, which is what makes them re-runnable without a 683 MB re-parse.

Nothing here is *fitted* to the data: no imputation, no scaling, no encoding,
no aggregates. Anything learned from data before the Phase 02 split is leakage,
so it belongs in ``features/``, not here.
"""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd
import yaml

from fraud_engine.data.validate import validate_interim

log = logging.getLogger(__name__)

# Resolved relative to the working directory. The Makefile is the interface and
# always runs from the repository root, so this holds for `make data`; the
# override argument on main() exists for everything else.
DEFAULT_CONFIG_PATH = Path("config/config.yaml")

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


def measure_cardinality(csv_path: Path, sample_rows: int) -> dict[str, int]:
    """Count distinct values per text column, from the head of the file.

    Feeds ``build_dtype_map``, which needs only a count to decide whether a
    column is worth storing as ``category``. Sampling rather than reading the
    whole file is the point: the dtype decision must be made *before* the full
    typed read, or there is nothing to pass as ``dtype=``.

    Nulls are excluded from the count, which is correct — pandas stores a
    missing value as code ``-1`` rather than as an entry in the category lookup
    table, so a column with four brands and some blanks has cardinality four.

    Args:
        csv_path: The CSV to sample.
        sample_rows: Number of rows to read from the head of the file.

    Returns:
        ``{column: n_distinct}`` for the text columns only. Numeric columns are
        absent, which is how ``build_dtype_map`` tells the two apart.

    Raises:
        ValueError: If ``sample_rows`` is not positive. Zero would produce an
            empty frame, a cardinality of 0 for every column, and therefore a
            ratio below any threshold — silently converting every text column
            to ``category``, including one distinct per row.

    Note:
        The sample is the **head** of the file, which is the first ~31 days,
        because the data is time-ordered. A column that happens to be empty
        across that window infers as numeric and is missed here.

        That is acceptable because the mistake cannot be silent: a string
        landing in a ``float32`` column makes the full ``read_csv`` raise. The
        failure mode is a loud crash, never a wrong answer — which is worth
        more than a cleverer sampler.
    """
    if sample_rows <= 0:
        raise ValueError(f"sample_rows must be positive, got {sample_rows}")

    df = pd.read_csv(csv_path, nrows=sample_rows)
    cardinality = df.select_dtypes(include="str").nunique().to_dict()

    return cardinality


def read_typed(csv_path: Path, dtype_map: dict[str, str]) -> pd.DataFrame:
    """Read the whole CSV under an explicit dtype policy.

    The map is checked against the file's header before anything is read, which
    is where ``build_dtype_map`` returning a *total* map finally pays off: a
    column the map does not mention would fall back to pandas' inference, and
    the entire dtype policy would be bypassed for it without any error.

    Args:
        csv_path: The CSV to read.
        dtype_map: Column name to pandas dtype, from ``build_dtype_map``. Must
            match the file's header exactly, in both directions.

    Returns:
        The full table, typed as the map dictates.

    Raises:
        ValueError: If the map and the header disagree either way.
    """
    # nrows=0 reads the header line only, not the file.
    header = set(pd.read_csv(csv_path, nrows=0).columns)
    mapped = set(dtype_map)

    # The dangerous direction: these columns exist but have no declared dtype,
    # so pandas would infer them. Nothing fails, the data just comes back
    # float64/object and the memory work is silently undone.
    untyped = header - mapped
    if untyped:
        raise ValueError(
            f"{csv_path.name}: {len(untyped)} column(s) have no declared dtype "
            f"and would fall back to inference: {sorted(untyped)}"
        )

    # The loud direction: read_csv would raise anyway, but a clear message here
    # names the real cause — the map was built against a different header.
    unknown = mapped - header
    if unknown:
        raise ValueError(
            f"{csv_path.name}: dtype map names {len(unknown)} column(s) absent "
            f"from the file: {sorted(unknown)}"
        )

    # .copy() consolidates the block layout, and it is not optional here.
    # Passing an explicit per-column dtype= map makes read_csv return a frame
    # with ONE BLOCK PER COLUMN — 394 blocks for 394 columns. Every later
    # operation then pays for that layout, and the merge in join_identity warns
    # about it outright. Consolidating collapses the joined table from 435
    # blocks to 37, which is cheaper for every column access downstream.
    return pd.read_csv(csv_path, dtype=dtype_map).copy()


def join_identity(transactions: pd.DataFrame, identity: pd.DataFrame) -> pd.DataFrame:
    """LEFT join identity onto transactions, flagging which rows matched.

    Only about 24% of transactions have an identity record, and *whether one
    exists at all* is predictive. So ``has_identity`` comes from the join itself
    rather than being inferred afterwards from a null check — those are
    different things. A null in ``DeviceInfo`` can mean "no identity record" or
    "a record whose DeviceInfo happened to be empty", and only the join knows
    which.

    Nothing is filled. Unmatched rows keep their nulls across every identity
    column, because that missingness is signal the model will split on.

    Args:
        transactions: The full transaction table. Every row survives.
        identity: Device and network attributes, covering a subset.

    Returns:
        ``transactions`` plus the identity columns and a boolean
        ``has_identity``, with the row count unchanged.

    Raises:
        pandas.errors.MergeError: If either side has duplicate TransactionIDs.
            A LEFT join whose right side has duplicate keys *multiplies* rows,
            so without this check the function would fabricate transactions
            rather than fail.
    """
    merged = transactions.merge(
        identity,
        on="TransactionID",
        how="left",
        validate="one_to_one",
        indicator=True,
    )

    # The comparison is already a per-row boolean Series — no conditional needed.
    # Built standalone and concatenated rather than assigned in: inserting a
    # column into a wide frame full of categorical blocks makes pandas rebuild
    # its block layout and warn about fragmentation.
    has_identity = (merged["_merge"] == "both").rename("has_identity")

    # _merge was scaffolding for the flag above; it must not reach the parquet.
    return pd.concat([merged.drop(columns="_merge"), has_identity], axis=1)


def add_time_columns(transactions: pd.DataFrame) -> pd.DataFrame:
    """Derive relative day, hour and weekday from ``TransactionDT``.

    ``TransactionDT`` is **seconds elapsed from an unpublished reference point**,
    not a Unix timestamp. Handing it to a date parser produces dates in 1970 that
    look entirely plausible and mean nothing, so the arithmetic here is plain
    integer division and nothing is converted to a datetime.

    What the three columns actually mean:

    ``day``
        Days since the reference. Relative, but internally consistent — this is
        what Phase 02 slices the temporal split on.
    ``hour``
        Hour relative to the reference. Equals wall-clock hour *only if* that
        reference is midnight-aligned. ``min(TransactionDT)`` is exactly 86,400 —
        one whole day — which suggests it is, but that is an inference, not a
        documented fact. Confirm it in the EDA notebook: plot volume by hour and
        look for a plausible overnight trough. A flat or oddly-phased curve means
        the alignment assumption is wrong.
    ``weekday``
        A consistent 7-day cycle, but position 0 is **not** necessarily Monday.
        Usable as a categorical feature; not usable as a claim. "Fraud peaks on
        Tuesdays" would be unfounded — the grouping is real, the labels are not
        knowable.

    Args:
        transactions: Table containing ``TransactionDT``.

    Returns:
        A new frame with ``day``, ``hour`` and ``weekday`` appended. The input is
        left unmodified, consistent with the rest of this module.
    """
    seconds = transactions["TransactionDT"]
    # Assembled as one frame and concatenated in a single operation. `.assign()`
    # inserts column by column, which on a wide frame with many categorical
    # blocks forces a block-layout rebuild per column and warns about it.
    derived = pd.DataFrame(
        {
            "day": seconds // 86_400,
            "hour": seconds // 3_600 % 24,
            "weekday": seconds // 86_400 % 7,
        },
        index=transactions.index,
    )
    return pd.concat([transactions, derived], axis=1)


def count_rows(csv_path: Path) -> int:
    """Count data rows without parsing the file.

    ``build_dtype_map`` needs the *full* row count to compute its category
    ratio, and it needs it before the typed read exists — so the count cannot
    come from a DataFrame. Counting newlines takes about a second on 683 MB.

    The tempting shortcut is to pass the sample size instead. That would run,
    but it silently redefines the ratio: a threshold tuned against 590,540 rows
    would be measured against 100,000, and far fewer columns would qualify as
    categorical. Wrong answers, no error.

    Args:
        csv_path: The CSV to count.

    Returns:
        Rows excluding the header.
    """
    with csv_path.open("rb") as handle:
        return sum(1 for _ in handle) - 1


def load_typed_csv(csv_path: Path, load_cfg: dict) -> pd.DataFrame:
    """Run the sample → policy → typed-read cycle for one CSV.

    Both raw files go through this independently: they have different columns
    (only one carries ``isFraud``, only one the ``id_*`` block) and different
    row counts, so each gets its own cardinality measurement and dtype map.

    Args:
        csv_path: The CSV to load.
        load_cfg: The ``load:`` block of the config.

    Returns:
        The fully typed table.
    """
    n_rows = count_rows(csv_path)
    cardinality = measure_cardinality(csv_path, load_cfg["cardinality_sample_rows"])
    header = list(pd.read_csv(csv_path, nrows=0).columns)
    dtype_map = build_dtype_map(header, cardinality, n_rows, load_cfg)

    n_category = sum(1 for dtype in dtype_map.values() if dtype == "category")
    log.info(
        "%s: %d rows, %d columns, %d text column(s) encoded as category",
        csv_path.name,
        n_rows,
        len(header),
        n_category,
    )
    return read_typed(csv_path, dtype_map)


def load_config(config_path: Path) -> dict:
    """Read the pipeline config."""
    return yaml.safe_load(config_path.read_text())


def main(config_path: Path = DEFAULT_CONFIG_PATH) -> None:
    """Turn the two raw CSVs into one validated, typed parquet table.

    Wiring only — every decision lives in the functions this calls. Invoked by
    ``make data`` as ``python -m fraud_engine.data.load``.

    Validation runs *before* the write, so a schema violation leaves no parquet
    behind. Together with ``.DELETE_ON_ERROR:`` in the Makefile, that means no
    downstream stage can ever read a table that failed its contract.

    Args:
        config_path: Path to ``config.yaml``. Defaults to a repo-root-relative
            location, which is where the Makefile runs from.
    """
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    config = load_config(config_path)
    paths, load_cfg = config["paths"], config["load"]

    transactions = load_typed_csv(Path(paths["raw"]["transactions"]), load_cfg)
    identity = load_typed_csv(Path(paths["raw"]["identity"]), load_cfg)

    joined = join_identity(transactions, identity)
    final = add_time_columns(joined)

    # The only end-to-end check in the pipeline. join_identity guards against
    # duplicate keys and the schema asserts TransactionID is unique, but this is
    # the one place holding both the input and the output, so it is the only
    # place that can compare them.
    if len(final) != len(transactions):
        raise ValueError(
            f"row count changed during the join: {len(transactions)} in, {len(final)} out"
        )

    validate_interim(final)

    interim_path = Path(paths["interim"])
    interim_path.parent.mkdir(parents=True, exist_ok=True)
    final.to_parquet(interim_path, index=False)

    log.info(
        "wrote %s — %d rows x %d columns, %.1f MB in memory (%.1f%% with identity)",
        interim_path,
        len(final),
        final.shape[1],
        final.memory_usage(deep=True).sum() / 1024**2,
        100 * final["has_identity"].mean(),
    )


if __name__ == "__main__":
    main()
