"""What the shipped booster used, per row, in the space where it is additive.

`docs/explainability.md` §1-§4. This module computes and persists; it decides nothing.
The model, the calibrator, the costs and the policy were frozen before it existed, and
a contribution that embarrasses one of them is written about rather than acted on.

**LightGBM's own TreeSHAP**, through `pred_contrib`, not `shap.TreeExplainer`. Exact,
and tree-path-dependent — the reference distribution comes from the training-time cover
already stored in the trees, so there is no background sample to choose and no second
answer to reconcile. `shap` is a plotting dependency here and nothing in the serving
path imports it.

**Proven before use.** The reloaded booster must reproduce its recorded VAL-CAL scores
exactly before a single contribution is computed. SHAP attached to a model is something
new attached to a model, and `decision-policy.md` §7 registered that rule for this case.

**Computed once, then drawn from.** Contributions are expensive enough that recomputing
them per figure is not an option, and a figure drawn from a model rather than from the
persisted record is one that can disagree with the record beside it.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

import lightgbm as lgb
import mlflow
import numpy as np
import pandas as pd

from fraud_engine.data.load import DEFAULT_CONFIG_PATH, load_config
from fraud_engine.evaluation.report import git_revision
from fraud_engine.evaluation.reproduce import check_reproduces
from fraud_engine.evaluation.tracking import configure_tracking, tracked_run
from fraud_engine.features.registry import KEYS, resolve_tiers
from fraud_engine.models.train import (
    LABEL,
    feature_columns,
    prepare_matrices,
    run_name,
    score,
)

log = logging.getLogger(__name__)

NAME = "shap_global"
PROOF_SPLIT = "val_cal"

# The last column `pred_contrib` returns is the model's expected raw margin, not a
# feature. Named rather than indexed everywhere, because an off-by-one here renames
# every contribution and nothing downstream would look wrong.
BASE_VALUE = "__base_value__"

# How far the contributions may sit from the margin they decompose, as a fraction of
# the row's own contribution mass.
#
# Not a config key: it is a property of TreeSHAP's arithmetic, not a decision anyone
# would make differently. The algorithm's unwind step divides by subset weights and
# loses digits doing it, so `sum(contributions) == margin` is false in floating point
# on a correct implementation.
#
# **Relative, and to the mass rather than to the margin.** The error comes from
# cancellation between three hundred and fifty terms, so it scales with what is being
# summed, not with the small number left over; dividing by a margin that can approach
# zero would make the check explode on exactly the rows it should be quietest about.
#
# **Not a precision claim.** What the check is for is catching contributions that are
# not of this model — the wrong tree count, the wrong rows, a stale artifact — and
# those are out by a share of order one, two decades above this bound. Set generously
# on purpose: a spurious failure costs a long rerun and, worse, invites loosening the
# bound *after* watching it fail, which is the one way a guard becomes a formality.
# The measured deviation is written into the record every run instead, so the headroom
# is visible without the threshold having to be tight.
#
# **The size is `shap`'s own, arrived at independently.** `TreeExplainer` refuses a
# decomposition on `np.allclose(atol=1e-2, rtol=1e-2)` against the model output; at the
# scale of this booster's margins that bound and this one agree to within a factor of
# two. The difference is the denominator: `shap` normalises by the output, which goes
# to zero on the rows where the classes meet while the cancellation error does not, so
# its bound tightens exactly where nothing about the arithmetic got easier. The mass
# being summed is the better-conditioned choice, and costs nothing to prefer.
ADDITIVITY_TOLERANCE = 1e-2


def contributions(booster: lgb.Booster, features: pd.DataFrame) -> np.ndarray:
    """Per-feature contributions to the raw margin, plus the base value they sit around.

    `num_iteration` is passed as `score` passes it, so the trees explained are the trees
    that produced the recorded scores. A booster reloaded from `model.txt` reports
    `best_iteration` as -1 because the file was already written truncated at the peak;
    naming the argument keeps the two call sites saying the same thing rather than
    relying on that.

    Args:
        booster: The shipped booster, reloaded.
        features: Rows to explain, columns in the booster's own order.

    Returns:
        `(n_rows, n_features + 1)` in raw margin space — log-odds here, never
        probability, and never calibrated probability. The last column is the base
        value.
    """
    return booster.predict(features, num_iteration=booster.best_iteration, pred_contrib=True)


def check_additive(contribs: np.ndarray, margin: np.ndarray) -> float:
    """Refuse contributions that do not sum to the margin they claim to decompose.

    The one property everything downstream rests on. It does not catch a column
    misalignment — a sum is indifferent to order — which is why `explain_split` also
    checks that the base value is constant, a thing only the last column is.

    Args:
        contribs: As `contributions` returned them.
        margin: `predict(..., raw_score=True)` on the same rows.

    Returns:
        The largest deviation as a fraction of the row's contribution mass, for the
        record. Reported rather than discarded: it is how the next model's tolerance
        gets set, and how this one's is checked against a wider sample than the one
        that chose it.

    Raises:
        ValueError: If any row is further out than `ADDITIVITY_TOLERANCE`.
    """
    deviation = np.abs(contribs.sum(axis=1) - margin) / np.abs(contribs).sum(axis=1)
    worst = float(deviation.max())

    if worst > ADDITIVITY_TOLERANCE:
        raise ValueError(
            f"contributions do not sum to the raw margin: worst row off by a relative "
            f"{worst:.3e}, tolerance {ADDITIVITY_TOLERANCE:.0e}. These do not decompose "
            "this model."
        )

    return worst


def check_base_value(contribs: np.ndarray) -> float:
    """Refuse an array whose last column is a feature rather than the base value.

    The check `check_additive` structurally cannot be. A sum is indifferent to order, so
    a decomposition named one place out still adds up perfectly; what it does not do is
    leave the same number in the last column of every row, because only the base value
    is a property of the model rather than of the transaction.

    Args:
        contribs: As `contributions` returned them.

    Returns:
        The base value — the model's expected raw margin, which every explanation is
        stated relative to.

    Raises:
        ValueError: If the last column varies by row.
    """
    base = contribs[:, -1]

    if not (base == base[0]).all():
        raise ValueError(
            "the base value column is not constant; the last column of the contribution "
            "array is a feature, and every contribution is named one place out"
        )

    return float(base[0])


@dataclass(frozen=True)
class Explained:
    """One split's decomposition, with the numbers its record has to carry.

    A frozen record rather than a tuple, for `cost.Decision`'s reason: the members are
    heterogeneous and two of them are scalars, so a caller unpacking positionally is one
    edit away from writing a deviation where a base value belongs and seeing nothing
    wrong.
    """

    contributions: np.ndarray
    rows: pd.DataFrame
    deviation: float
    base_value: float


def take_sample(frame: pd.DataFrame, rows: int, seed: int) -> pd.DataFrame:
    """A deterministic subset, in the order the split already has.

    Positions rather than labels, and sorted after drawing: the matrices are ordered by
    time, and a sample that reshuffled them would make every persisted artifact depend
    on numpy's draw order as well as on the seed.

    Args:
        frame: A prepared split matrix.
        rows: How many to take. Everything, if the split is no larger.
        seed: Fixed in config, so the figure is a rerunnable object and not a draw.

    Returns:
        The subset, chronological.
    """
    if rows >= len(frame):
        return frame

    drawn = np.random.default_rng(seed).choice(len(frame), size=rows, replace=False)
    return frame.iloc[np.sort(drawn)]


def ranking(contribs: np.ndarray, columns: list[str], tiers: dict[str, str]) -> pd.DataFrame:
    """Every feature by how much of the margin it moves, with the tier that bounds it.

    Mean absolute contribution is the ordering, because a feature that pushes both ways
    is used by the model however its signs cancel. The signed mean rides along: the two
    disagreeing is what a feature with a threshold effect looks like.

    Args:
        contribs: As `contributions` returned them, base value included.
        columns: Feature names, in the booster's order.
        tiers: `{feature: serving tier}`.

    Returns:
        One row per feature, ordered by `mean_abs` descending.

    Raises:
        ValueError: If the array's width is not the feature count plus a base value.
    """
    if contribs.shape[1] != len(columns) + 1:
        raise ValueError(
            f"{contribs.shape[1]} contribution columns for {len(columns)} features; "
            "the base value column is not where it is assumed to be"
        )

    values = contribs[:, :-1]
    frame = pd.DataFrame(
        {
            "feature": columns,
            "tier": [tiers[column] for column in columns],
            "mean_abs": np.abs(values).mean(axis=0),
            "mean_signed": values.mean(axis=0),
        }
    )
    return frame.sort_values("mean_abs", ascending=False, ignore_index=True)


def tier_shares(ranked: pd.DataFrame) -> dict[str, float]:
    """The share of total absolute contribution falling in each serving tier.

    `docs/explainability.md` §4's metric, and the explanation-shaped version of what E7
    measured on PR-AUC: tier 0 is Vesta's inherited aggregates, which no reason code can
    put into a sentence.

    Args:
        ranked: As `ranking` returned it.

    Returns:
        `{tier: share}`, summing to one.
    """
    total = ranked["mean_abs"].sum()
    by_tier = ranked.groupby("tier", observed=True)["mean_abs"].sum() / total
    return {tier: float(share) for tier, share in by_tier.sort_values(ascending=False).items()}


def feature_tiers(features_dir: Path | str, columns: list[str]) -> dict[str, str]:
    """Each feature's serving tier, from the registry that `features.md` is asserted against.

    Args:
        features_dir: Directory holding `{split}.parquet`.
        columns: Feature names, in the booster's order.

    Returns:
        `{feature: tier}`.

    Raises:
        ValueError: If a feature belongs to no tier — the registry and the model have
            drifted, and §4's shares would silently omit a column.
    """
    resolved = resolve_tiers(features_dir)
    tiers = {
        column: tier for tier, members in resolved.items() for column in members if tier != "keys"
    }

    unassigned = [column for column in columns if column not in tiers]
    if unassigned:
        raise ValueError(f"features in no serving tier: {unassigned}")

    return tiers


def write_contributions(
    contribs: np.ndarray, keys: pd.DataFrame, columns: list[str], path: Path | str
) -> None:
    """Persist one split's contributions beside the keys that identify their rows.

    Float64, deliberately: the additivity the whole phase rests on does not survive a
    downcast, and a file that could not be re-checked is not a record.

    Args:
        contribs: As `contributions` returned them.
        keys: `TransactionID`, `day` and the label for the same rows, in the same order.
        columns: Feature names, in the booster's order.
        path: Destination parquet.
    """
    frame = pd.DataFrame(contribs, columns=[*columns, BASE_VALUE], index=keys.index)
    pd.concat([keys, frame], axis=1).to_parquet(path, index=False)


def explain_split(
    booster: lgb.Booster,
    matrix: pd.DataFrame,
    columns: list[str],
    explain_cfg: dict,
) -> Explained:
    """One split's sample, decomposed and past both guards.

    Wiring: draw the sample, compute, and refuse the result if it does not decompose
    this model or is not named the way it is read.

    Args:
        booster: The shipped booster, reloaded.
        matrix: A prepared split matrix.
        columns: Feature names, in the booster's order.
        explain_cfg: The `explain:` config block.

    Returns:
        The decomposition and what the record needs to state about it.

    Raises:
        ValueError: Per `check_additive` and `check_base_value`.
    """
    sample = take_sample(matrix, explain_cfg["sample_rows"], explain_cfg["seed"])
    features = sample[columns]

    contribs = contributions(booster, features)
    margin = booster.predict(features, num_iteration=booster.best_iteration, raw_score=True)

    return Explained(
        contributions=contribs,
        rows=sample,
        deviation=check_additive(contribs, margin),
        base_value=check_base_value(contribs),
    )


def main(config_path: Path = DEFAULT_CONFIG_PATH) -> None:
    """Prove the model, explain each registered split, and record the global view.

    Wiring only. Invoked by `make explain` as
    `python -m fraud_engine.explain.contributions`.

    Args:
        config_path: Path to `config.yaml`.
    """
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    config = load_config(config_path)
    paths, model_cfg, explain_cfg = config["paths"], config["model"], config["explain"]

    configure_tracking(config["tracking"])

    splits = tuple(explain_cfg["splits"])
    matrices, _, _ = prepare_matrices(paths["features_dir"], model_cfg, ("train", *splits))

    booster = lgb.Booster(model_file=paths["model"])
    columns = feature_columns(matrices[splits[0]])
    if booster.feature_name() != columns:
        raise ValueError("the booster's features are not the matrices' features, in order")

    model_name = run_name(model_cfg)
    recorded = pd.read_parquet(Path(paths["predictions_dir"]) / f"{model_name}.parquet")
    check_reproduces(
        score(booster, {PROOF_SPLIT: matrices[PROOF_SPLIT]}, columns),
        recorded,
        model_name,
        PROOF_SPLIT,
    )
    log.info("proof passed: %s reproduces its %s record", model_name, PROOF_SPLIT)

    tiers = feature_tiers(paths["features_dir"], columns)
    explain_dir = Path(paths["explain_dir"])
    explain_dir.mkdir(parents=True, exist_ok=True)

    params = {
        "model": model_name,
        "splits": ",".join(splits),
        "sample_rows": explain_cfg["sample_rows"],
        "seed": explain_cfg["seed"],
    }

    blocks: dict[str, dict] = {}
    with tracked_run(NAME, params, config_path):
        for split in splits:
            explained = explain_split(booster, matrices[split], columns, explain_cfg)
            ranked = ranking(explained.contributions, columns, tiers)
            shares = tier_shares(ranked)

            write_contributions(
                explained.contributions,
                explained.rows[["TransactionID", "day", LABEL]].reset_index(drop=True),
                columns,
                explain_dir / f"{split}.parquet",
            )

            blocks[split] = {
                "rows": len(explained.rows),
                "sampled": len(explained.rows) < len(matrices[split]),
                "of": len(matrices[split]),
                "base_value": explained.base_value,
                "additivity_deviation_relative": explained.deviation,
                "tier_shares": shares,
                "ranking": ranked.to_dict(orient="records"),
            }

            mlflow.log_metrics(
                {f"{split}.share.{tier}": share for tier, share in shares.items()}
                | {f"{split}.additivity_deviation_relative": explained.deviation}
            )
            log.info(
                "%-8s %6d rows   base %+.3f   tier shares %s",
                split,
                len(explained.rows),
                explained.base_value,
                {tier: round(share, 4) for tier, share in shares.items()},
            )

    record = {
        "name": NAME,
        "created": datetime.now(UTC).isoformat(timespec="seconds"),
        "git_revision": git_revision(),
        "model": model_name,
        "features": len(columns),
        "keys": list(KEYS),
        "additivity_tolerance": ADDITIVITY_TOLERANCE,
        "splits": blocks,
    }
    path = Path(paths["shap_global"])
    path.write_text(json.dumps(record, indent=2) + "\n")
    log.info("wrote %s", path)


if __name__ == "__main__":
    main()
