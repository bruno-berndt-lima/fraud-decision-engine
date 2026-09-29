"""Feature drift and prediction drift — what can be watched without a single label.

`docs/monitoring.md` §3 and §6. Nothing here reads `isFraud` on the horizon, because the
horizon has none: it measures how far the inputs and the scores have moved, not whether
the model got worse.

**Feature drift** is PSI per column against train, on both horizons — the labelled
windows, where it can be set beside §4's PR-AUC and the claim that drift leads decay can
be tested, and the unlabelled windows, where it is the monitor. **Prediction drift** is the
score distribution against test, the mean calibrated probability, and what the frozen
policy would do with each window — block rate and review eligibility, in the unit the
business reads.

**The policy is run, never re-derived.** Test's own block rate, recomputed here from the
persisted vector, must equal the headline record's before a horizon window is costed.
"""

from __future__ import annotations

import json
import logging
from datetime import UTC, datetime
from pathlib import Path

import mlflow
import numpy as np
import pandas as pd

from fraud_engine.data.load import DEFAULT_CONFIG_PATH, load_config
from fraud_engine.evaluation.cost import ALLOW, BLOCK, REVIEW, ev_policy, expected_costs, load_costs
from fraud_engine.evaluation.plots import save_figure
from fraud_engine.evaluation.report import git_revision, load_operating_capacity
from fraud_engine.evaluation.tracking import configure_tracking, tracked_run
from fraud_engine.features import velocity
from fraud_engine.features.encoders import MISSING
from fraud_engine.models.train import LABEL, OTHER, apply_categories, feature_columns, run_name
from fraud_engine.monitoring.drift_plots import plot_drift
from fraud_engine.monitoring.psi import categorical_counts, numeric_counts, psi, quantile_edges
from fraud_engine.monitoring.windows import describe, window_index
from fraud_engine.serving.artifacts import read_categories

log = logging.getLogger(__name__)

NAME = "drift"
FIGURE = "drift.png"
LABELLED = ("val_fit", "val_cal", "test")
HEADLINE_PREDICTIONS = "headline_test"
HORIZON_FEATURES = "features.parquet"
HORIZON_SCORES = "scores.parquet"
HORIZON_MANIFEST = "build.json"
# §3's first exclusion: filled as the service fills them, so a constant on the horizon.
EXCLUDED_ON_HORIZON = tuple(velocity.COLUMNS)
PERCENTILES = (50, 90, 99)


def band(value: float, psi_cfg: dict) -> str:
    """The convention's name for a PSI, stated as a convention (§3)."""
    if value < psi_cfg["moderate"]:
        return "stable"
    return "moderate" if value < psi_cfg["significant"] else "significant"


def weighted_columns(ranking: list[dict], top_k: int) -> pd.DataFrame:
    """Condition 1's columns: the top `k` by contribution the horizon can support.

    Returns:
        `feature`, `tier` and `weight` — contribution mass renormalised over the `k`.
    """
    ranked = pd.DataFrame(ranking)
    top = ranked[~ranked["feature"].isin(EXCLUDED_ON_HORIZON)].head(top_k)
    return top.assign(weight=top["mean_abs"] / top["mean_abs"].sum())[["feature", "tier", "weight"]]


def reference_bins(reference: pd.DataFrame, columns: list[str], bins: int) -> dict:
    """Each column's bins, cut once on the reference.

    Returns:
        `{column: edges}` for a numeric column, `{column: None}` for a categorical
        one, whose bins are its vocabulary.
    """
    return {
        column: None
        if isinstance(reference[column].dtype, pd.CategoricalDtype)
        else quantile_edges(reference[column], bins)
        for column in columns
    }


def counts(values: pd.Series, edges: np.ndarray | None) -> np.ndarray:
    """A column counted into its bins: numeric with missing last, categorical by level."""
    return categorical_counts(values) if edges is None else numeric_counts(values, edges)


def column_psi(
    reference: pd.DataFrame,
    frame: pd.DataFrame,
    windows: np.ndarray,
    bins: dict,
    epsilon: float,
) -> pd.DataFrame:
    """Every column's PSI in every window, with what it cannot be read without.

    Returns:
        One row per (window, column): `psi`, `value_bins` (bins the reference populates),
        and the missing and unseen-category shares on both sides — the unseen share
        only for a categorical, whose `OTHER` level it is.

    Raises:
        ValueError: If a categorical's levels are not the reference's, in order. Counts
            are taken by code, so a reordered vocabulary would compare one level's rows
            against another's and produce a plausible, wrong index.
    """
    positions = {int(window): np.flatnonzero(windows == window) for window in np.unique(windows)}
    rows = []
    for column, edges in bins.items():
        expected = counts(reference[column], edges)
        levels = None if edges is not None else list(reference[column].cat.categories)
        if levels is not None and list(frame[column].cat.categories) != levels:
            raise ValueError(
                f"{column!r} is not levelled as the reference is; its bins would not align"
            )
        missing_at = len(expected) - 1 if edges is not None else levels.index(MISSING)
        other_at = None if edges is not None else levels.index(OTHER)
        populated = int((expected[:-1] > 0).sum() if edges is not None else (expected > 0).sum())

        for window, taken in positions.items():
            actual = counts(frame[column].iloc[taken], edges)
            rows.append(
                {
                    "window": window,
                    "column": column,
                    "kind": "numeric" if edges is not None else "categorical",
                    "psi": psi(expected, actual, epsilon),
                    "value_bins": populated,
                    "missing_share_reference": expected[missing_at] / expected.sum(),
                    "missing_share": actual[missing_at] / actual.sum(),
                    "unseen_share_reference": None
                    if other_at is None
                    else expected[other_at] / expected.sum(),
                    "unseen_share": None if other_at is None else actual[other_at] / actual.sum(),
                }
            )
    return pd.DataFrame(rows)


def summarise(
    table: pd.DataFrame, layout: pd.DataFrame, weights: pd.DataFrame, ranking: list[dict], cfg: dict
) -> list[dict]:
    """One entry per window: the weighted PSI, the band counts, and the leading columns.

    The leading columns are the top `report_top` by contribution, not by PSI — §3's
    question is whether the columns that move decisions moved.
    """
    leading = [row["feature"] for row in ranking if row["feature"] in set(table["column"])]
    leading = leading[: cfg["report_top"]]
    tiers = {row["feature"]: row["tier"] for row in ranking}

    entries = []
    for spec in layout.itertuples():
        window = table[table["window"] == spec.window].set_index("column")
        aggregate = float(
            (window.loc[weights["feature"], "psi"] * weights["weight"].to_numpy()).sum()
        )
        entries.append(
            {
                "window": int(spec.window),
                "first_day": int(spec.first_day),
                "last_day": int(spec.last_day),
                "partial": bool(spec.partial),
                "rows": int(spec.rows),
                "weighted_psi": aggregate,
                "weighted_band": band(aggregate, cfg["psi"]),
                "columns_by_band": window["psi"]
                .map(lambda v: band(v, cfg["psi"]))
                .value_counts()
                .to_dict(),
                "leading": [
                    {
                        "column": column,
                        "tier": tiers[column],
                        "psi": float(window.loc[column, "psi"]),
                        "band": band(float(window.loc[column, "psi"]), cfg["psi"]),
                    }
                    for column in leading
                ],
            }
        )
    return entries


def policy_rates(
    p: np.ndarray, amount: np.ndarray, day: np.ndarray, costs, capacity: float
) -> dict:
    """What the frozen EV policy does with a window, without a label.

    `Decision.summary` reports USD, which needs labels, beside the block rate and the
    review volume, which do not. It is called with placeholder labels and only those two
    fields are kept, so there is one definition of each; the tests hold that neither
    moves with the labels. Eligibility is `ev_policy`'s own rule: reviewing is cheaper
    in expectation than the better of allowing and blocking.
    """
    decision = ev_policy(p, amount, day, costs, capacity)
    summary = decision.summary(np.zeros(len(p), dtype=int), amount, day, costs)
    expected = expected_costs(p, amount, costs)
    gain = np.minimum(expected[ALLOW], expected[BLOCK]) - expected[REVIEW]
    return {
        "block_rate": summary["block_rate"],
        "reviews_per_day": summary["reviews_per_day"],
        "review_eligible_rate": float((gain > 0).mean()),
    }


def score_views(
    scores: pd.DataFrame, edges: np.ndarray, reference_counts, costs, capacity, cfg
) -> dict:
    """§6's three views of one window's scores."""
    p = scores["calibrated"].to_numpy()
    return {
        "score_psi": psi(
            reference_counts, numeric_counts(scores["calibrated"], edges), cfg["psi"]["epsilon"]
        ),
        **{f"p{q}": float(np.percentile(p, q)) for q in PERCENTILES},
        "mean_calibrated": float(p.mean()),
        "identity_share": float(scores["has_identity"].mean()),
        **policy_rates(p, scores["amount"].to_numpy(), scores["day"].to_numpy(), costs, capacity),
    }


def check_policy_record(test: pd.DataFrame, headline_record: dict, costs, capacity: float) -> None:
    """Refuse a policy that does not reproduce the headline's EV row on test.

    Raises:
        ValueError: If the block rate or the review volume differs at all, with real
            labels or without them.
    """
    recorded = headline_record["policies"]["ev"]
    p, amount, day = (
        test["calibrated"].to_numpy(),
        test["amount"].to_numpy(),
        test["day"].to_numpy(),
    )
    labelled = ev_policy(p, amount, day, costs, capacity).summary(
        test[LABEL].to_numpy(), amount, day, costs
    )
    unlabelled = policy_rates(p, amount, day, costs, capacity)

    for field in ("block_rate", "reviews_per_day"):
        if not labelled[field] == unlabelled[field] == recorded[field]:
            raise ValueError(
                f"EV {field} on test: {labelled[field]!r} with labels, {unlabelled[field]!r} "
                f"without, {recorded[field]!r} on record"
            )


def leads_decay(labelled: list[dict], decay_record: dict) -> dict:
    """§3's reading of drift against decay, over the full labelled windows.

    Raises:
        ValueError: If the two stages did not cut the same windows.
    """
    windows = {w["window"]: w for w in decay_record["windows"]}
    for entry in labelled:
        spec = windows[entry["window"]]
        if (spec["first_day"], spec["last_day"]) != (entry["first_day"], entry["last_day"]):
            raise ValueError(
                f"window {entry['window']} differs between this stage and the decay chart"
            )

    baseline = decay_record["baseline"]["pr_auc"]
    full = pd.DataFrame(
        {
            "window": entry["window"],
            "weighted_psi": entry["weighted_psi"],
            "shortfall": baseline - windows[entry["window"]]["pr_auc"],
            "reading": windows[entry["window"]]["reading"],
        }
        for entry in labelled
        if not entry["partial"]
    )
    full["psi_rank"] = full["weighted_psi"].rank(ascending=False).astype(int)
    return {
        "n": len(full),
        "spearman": float(full["weighted_psi"].corr(full["shortfall"], method="spearman")),
        "flagged": full.loc[
            full["reading"] != "no decline", ["window", "reading", "psi_rank"]
        ].to_dict(orient="records"),
    }


def main(config_path: Path = DEFAULT_CONFIG_PATH) -> None:
    """Measure both horizons' feature drift and the unlabelled horizon's prediction drift.

    Wiring only. Invoked by `make drift` as `python -m fraud_engine.monitoring.drift`.

    Args:
        config_path: Path to `config.yaml`.
    """
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    config = load_config(config_path)
    paths, cfg, splits_cfg = config["paths"], config["monitoring"], config["splits"]
    configure_tracking(config["tracking"])
    cost_matrix = load_config(Path(paths["cost_matrix"]))
    costs, capacity = load_costs(cost_matrix), load_operating_capacity(cost_matrix)

    predictions = Path(paths["predictions_dir"]) / f"{HEADLINE_PREDICTIONS}.parquet"
    horizon_dir = Path(paths["horizon_dir"])
    for required, remedy in (
        (predictions, "make headline"),
        (horizon_dir / HORIZON_SCORES, "make horizon"),
    ):
        if not required.exists():
            raise FileNotFoundError(f"{required} is absent; `{remedy}` writes it")

    vocabulary = read_categories(paths["categories"])
    features_dir = Path(paths["features_dir"])
    train = pd.read_parquet(features_dir / "train.parquet")
    columns = feature_columns(train)
    reference = apply_categories(train[columns], vocabulary)
    ranking = json.loads(Path(paths["shap_global"]).read_text())["splits"]["test"]["ranking"]
    weights = weighted_columns(ranking, cfg["trigger"]["top_k"])

    # Labelled: every column, five-day windows anchored where validation starts.
    labelled = pd.concat([pd.read_parquet(features_dir / f"{split}.parquet") for split in LABELLED])
    labelled_days = labelled["day"].to_numpy()
    width = cfg["window_days"]["labelled"]
    labelled_windows = window_index(labelled_days, width, splits_cfg["val_fit_start"])
    labelled_table = column_psi(
        reference,
        apply_categories(labelled[columns], vocabulary).reset_index(drop=True),
        labelled_windows,
        reference_bins(reference, columns, cfg["psi"]["bins"]),
        cfg["psi"]["epsilon"],
    )
    labelled_layout = describe(labelled_days, width, splits_cfg["val_fit_start"])

    # Unlabelled: every column the horizon can support, twenty-eight-day windows.
    supported = [column for column in columns if column not in EXCLUDED_ON_HORIZON]
    horizon = pd.read_parquet(horizon_dir / HORIZON_FEATURES)
    horizon_width = cfg["window_days"]["horizon"]
    horizon_windows = window_index(horizon["day"], horizon_width)
    horizon_table = column_psi(
        reference,
        horizon[supported],
        horizon_windows,
        reference_bins(reference, supported, cfg["psi"]["bins"]),
        cfg["psi"]["epsilon"],
    )
    horizon_layout = describe(horizon["day"], horizon_width)

    # Prediction drift: test is the reference, and it must reproduce the headline first.
    test = pd.read_parquet(predictions)
    check_policy_record(test, json.loads(Path(paths["headline"]).read_text()), costs, capacity)
    edges = quantile_edges(test["calibrated"], cfg["psi"]["bins"])
    test_counts = numeric_counts(test["calibrated"], edges)
    test_identity = pd.read_parquet(paths["interim"], columns=["TransactionID", "has_identity"])
    test = test.merge(test_identity, on="TransactionID", how="left", validate="one_to_one")

    scores = pd.read_parquet(horizon_dir / HORIZON_SCORES).rename(
        columns={"TransactionAmt": "amount"}
    )
    score_windows = window_index(scores["day"], horizon_width)
    views = [
        score_views(scores[score_windows == window], edges, test_counts, costs, capacity, cfg)
        for window in np.unique(score_windows)
    ]

    labelled_summary = summarise(labelled_table, labelled_layout, weights, ranking, cfg)
    horizon_summary = [
        entry | {"prediction": view}
        for entry, view in zip(
            summarise(horizon_table, horizon_layout, weights, ranking, cfg), views, strict=True
        )
    ]
    reading = leads_decay(labelled_summary, json.loads(Path(paths["decay"]).read_text()))

    params = {"model": run_name(config["model"]), "top_k": cfg["trigger"]["top_k"]}
    with tracked_run(NAME, params, config_path):
        mlflow.log_metrics(
            {
                f"labelled.window_{e['window']}.weighted_psi": e["weighted_psi"]
                for e in labelled_summary
            }
            | {
                f"horizon.window_{e['window']}.weighted_psi": e["weighted_psi"]
                for e in horizon_summary
            }
            | {
                f"horizon.window_{e['window']}.score_psi": e["prediction"]["score_psi"]
                for e in horizon_summary
            }
            | {"leads_decay.spearman": reading["spearman"]}
        )

    table = pd.concat(
        [
            labelled_table.assign(horizon="labelled"),
            horizon_table.assign(horizon="unlabelled"),
        ],
        ignore_index=True,
    )
    table["tier"] = table["column"].map({row["feature"]: row["tier"] for row in ranking})
    table["band"] = table["psi"].map(lambda v: band(v, cfg["psi"]))
    table.to_csv(paths["drift_table"], index=False)

    record = {
        "name": NAME,
        "created": datetime.now(UTC).isoformat(timespec="seconds"),
        "git_revision": git_revision(),
        "model": run_name(config["model"]),
        "horizon_build": json.loads((horizon_dir / HORIZON_MANIFEST).read_text()),
        "references": cfg["reference"],
        "psi": cfg["psi"],
        "excluded_on_horizon": list(EXCLUDED_ON_HORIZON),
        "weighted_columns": weights.to_dict(orient="records"),
        "score_reference": score_views(test, edges, test_counts, costs, capacity, cfg),
        "labelled": labelled_summary,
        "horizon": horizon_summary,
        "drift_leads_decay": reading,
    }
    Path(paths["drift"]).write_text(json.dumps(record, indent=2) + "\n")
    figure = save_figure(plot_drift(record), Path(paths["figures_dir"]) / FIGURE)

    for entry in horizon_summary:
        view = entry["prediction"]
        log.info(
            "horizon days %d-%d  weighted PSI %.3f (%s)  score PSI %.3f  mean p %.4f  block %.4f%s",
            entry["first_day"],
            entry["last_day"],
            entry["weighted_psi"],
            entry["weighted_band"],
            view["score_psi"],
            view["mean_calibrated"],
            view["block_rate"],
            " (partial)" if entry["partial"] else "",
        )
    log.info("drift leads decay: Spearman %.3f over %d windows", reading["spearman"], reading["n"])
    for path in (paths["drift"], paths["drift_table"], figure):
        log.info("wrote %s", path)


if __name__ == "__main__":
    main()
