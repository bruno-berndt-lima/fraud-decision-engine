"""V-block reduction — 339 Vesta columns down to a set that is not mostly copies.

The order is fixed by the roadmap and it matters. **Missingness first**: the
block splits into a small number of columns sharing an identical null pattern,
which is structural — it needs no training window and no fitted parameter,
because it is a fact about which source produced which column. **Correlation
second, and inside the training split only**, since correlation is exactly the
statistic that decides which columns get dropped and measuring it on validation
would let the evaluation set choose its own features.

Two things fall out of the grouping that are worth keeping.

*The missingness is itself a feature, and one indicator per group captures all
of it.* Within a group every column is null on exactly the same rows, so a
single presence flag says everything the 339 null masks say. Most of those flags
turn out to be `has_identity` under another name; the ones that are not stay.

*A group is where a median belongs.* Filling the whole block from one statistic
would be meaningless when null rates run from nothing to most of the data.

The raw block is **replaced**, not supplemented. Keeping both would leave the
matrices carrying each surviving column twice, one copy perfectly correlated
with the other, waiting for the first stage that reaches for "every numeric
column". The originals live on in ``interim/transactions.parquet``.

Clustering is greedy rather than hierarchical: walk the columns in order, and
each one not yet claimed starts a cluster and absorbs everything correlated with
it above the threshold. That makes the representative the cluster's head by
construction, keeps the result deterministic given column order, and needs no
dependency this project would otherwise not have.

The threshold is the decision here, and it lives in config. The column list is
derived from it, so a diff of the config shows what changed rather than a
diff of two hundred names.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import pandas as pd

V_COLUMNS = tuple(f"V{index}" for index in range(1, 340))

# Every column this family emits carries it, which is how the family registry
# finds them: membership is chosen by a fit and is not knowable until build.py
# has run.
PREFIX = "vb_"


def nan_pattern_groups(frame: pd.DataFrame, columns: tuple[str, ...]) -> list[list[str]]:
    """Columns grouped by identical missingness, in first-appearance order.

    Structural, not fitted. Two columns share a group because the same upstream
    source produced them, and that is true of any slice of the data — which is
    why this may run before the training window is considered at all.

    Returns:
        One list per distinct null pattern, each in ``columns`` order.
    """
    groups: dict[str, list[str]] = {}
    for column in columns:
        pattern = hashlib.md5(frame[column].isna().to_numpy().tobytes()).hexdigest()
        groups.setdefault(pattern, []).append(column)
    return list(groups.values())


def cluster(frame: pd.DataFrame, columns: list[str], threshold: float) -> list[list[str]]:
    """Greedy correlation clusters over ``columns``, heads first.

    Walks ``columns`` in order. Each column not already claimed opens a cluster
    and takes every remaining column whose absolute correlation with it reaches
    ``threshold``. The head is therefore the representative, chosen by position
    rather than by a tie-break nobody could later justify.

    Correlation is measured on the rows where these columns are present. Within a
    NaN group that is all of them or none of them, so no row is partially
    counted.
    """
    if not columns:
        return []
    if len(columns) == 1:
        return [list(columns)]

    correlations = frame[columns].dropna().corr().abs()

    clusters: list[list[str]] = []
    unclaimed = list(columns)
    while unclaimed:
        head, *rest = unclaimed
        members = [head] + [
            column for column in rest if correlations.loc[head, column] >= threshold
        ]
        clusters.append(members)
        unclaimed = [column for column in rest if column not in members]

    return clusters


def presence_frame(frame: pd.DataFrame, sources: list[str]) -> pd.DataFrame:
    """One flag per source column: was that column's group present for this row.

    Keyed by the source column rather than by an output name, so the same
    function serves the clustering inside ``fit`` and the emission inside
    ``apply_fitted`` without either having to parse a name apart.
    """
    return pd.DataFrame(
        {source: frame[source].notna().astype("int8") for source in sources},
        index=frame.index,
    )


def fit(train: pd.DataFrame, vblock_cfg: dict, columns: tuple[str, ...] = V_COLUMNS) -> dict:
    """Choose the columns to keep and the medians to fill them with, on train alone.

    Args:
        train: The training rows, and only those.
        vblock_cfg: The ``features.vblock`` config block.
        columns: The block to reduce. Defaults to the whole of ``V1..V339``; a
            caller passes a narrower set only to exercise the reduction itself.

    Returns:
        ``representatives`` (kept V columns), ``presence`` (kept source columns
        whose flags survive) and ``medians`` (per representative, from train).
    """
    groups = nan_pattern_groups(train, columns)

    minimum = vblock_cfg["min_observed_rows"]

    # Flags first, from the full grouping. One needs that many rows on its rarer
    # side to be worth a coefficient: three groups are never null at all, and two
    # more are null on a dozen rows out of three hundred thousand.
    #
    # The survivors are deduplicated against each other by the rule that
    # deduplicates the columns, because several groups share a source in practice
    # and a linear model pays for every near-copy.
    varying = [
        group[0]
        for group in groups
        if min(train[group[0]].isna().sum(), train[group[0]].notna().sum()) >= minimum
    ]

    # A column needs the same floor in non-null training rows to be kept at all:
    # its median fills every gap in it, and one drawn from a handful of
    # observations is imposed on the rest. A column with none has no median, which
    # would put nulls into a family that promises none.
    observed = {column for column in columns if train[column].notna().sum() >= minimum}

    representatives = [
        members[0]
        for group in groups
        for members in cluster(
            train,
            [column for column in group if column in observed],
            vblock_cfg["correlation_threshold"],
        )
    ]
    flags = presence_frame(train, varying)
    presence = [members[0] for members in cluster(flags, varying, vblock_cfg["presence_threshold"])]

    return {
        # What was reduced, so the apply can remove it. A matrix carrying both
        # V95 and vb_V95 has not been reduced — it has been duplicated, and the
        # next stage to reach for "every numeric column" gets a perfectly
        # correlated pair for each one.
        "source": list(columns),
        "representatives": representatives,
        "presence": presence,
        "medians": train[representatives].median(),
    }


def apply_fitted(frame: pd.DataFrame, fitted: dict) -> pd.DataFrame:
    """Emit the kept columns, filled, plus the kept presence flags.

    Filled with the training median rather than left null, because this family
    honours ``build_features``' contract: the fill is decided here, where the
    null rates are known, and not by the evaluation harness's imputer.

    The original ``V*`` names survive behind the prefix. Phase 07 has to be able
    to say which Vesta column a contribution belongs to.
    """
    kept = fitted["representatives"]
    built = frame[kept].fillna(fitted["medians"]).astype("float32")
    built.columns = [f"{PREFIX}{column}" for column in kept]

    # Built from the columns `fit` chose, never by regrouping this frame. The
    # grouping is a property of the training window; recomputing it here would
    # let a different null pattern in validation silently rename the flags.
    flags = presence_frame(frame, fitted["presence"])
    flags.columns = [f"{PREFIX}present_{source}" for source in fitted["presence"]]

    # The raw block does not survive. Every column of it stays available in
    # interim/transactions.parquet, which is never modified, and the original
    # name is recoverable from the prefix — so Phase 07 can still say which
    # Vesta column a contribution belongs to without carrying a second copy.
    return pd.concat([frame.drop(columns=fitted["source"]), built, flags], axis=1)


def add_vblock_features(
    frame: pd.DataFrame, vblock_cfg: dict, columns: tuple[str, ...] = V_COLUMNS
) -> pd.DataFrame:
    """The V-block family: chosen on ``frame``'s train rows, applied to all of it."""
    return apply_fitted(frame, fit(frame[frame["split"] == "train"], vblock_cfg, columns))


def write_tables(fitted: dict, path: Path | str) -> None:
    """Persist what was kept and what it is filled with.

    The reduction is a decision, and this is the record of it: which columns
    survived, and the median a served model needs to fill them. Without the file
    the threshold in config describes the outcome but does not pin it.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    rows = pd.DataFrame(
        {
            "column": [
                *(f"{PREFIX}{column}" for column in fitted["representatives"]),
                *(f"{PREFIX}present_{source}" for source in fitted["presence"]),
            ],
            "role": ["representative"] * len(fitted["representatives"])
            + ["presence"] * len(fitted["presence"]),
            "median": [*fitted["medians"].to_numpy(), *[None] * len(fitted["presence"])],
        }
    )
    rows.to_parquet(path, index=False)
