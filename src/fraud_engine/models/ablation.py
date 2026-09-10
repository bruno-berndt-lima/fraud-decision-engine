"""The family ablation, re-run under a tree.

Phase 04 measured each family by *adding* it to a bare linear probe. E4 recorded
in advance which shapes that instrument structurally could not see, and handed
the re-measurement here rather than pretending the probe had settled it.

Two things invert. **The direction:** an arm is the full matrix *minus* one
family, because the question a serving decision asks is what is lost by not
building it, and because E3 needs exactly that shape for the history-dependent
columns. **The instrument:** a tree can represent thresholds and interactions a
linear probe cannot, which is the whole reason the handoff exists.

The reference config is the untuned one. It does not subsample, so repeated fits
return identical digits and a delta carries no seed noise — `seeds.py` measured
that, and the tuned config's spread against it. The tuned config is stronger and
does subsample, putting a wide bar under every delta; families that move here go
back to it for confirmation, the shape E6 already established.

**The blind spots are registered in E4 before running**, and two of them shape
how a flat result reads. Leave-one-out against a matrix this correlated measures
redundancy rather than signal, so a family recoverable from the survivors comes
back at zero. And the arms do not stop at the same round, so the early-stopping
unfairness E6 recorded applies here too — a family that lands inside it is
undecided, not absent.

It is also why the Phase 04 deltas and these are not two columns of one table:
different base, opposite sign.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import pandas as pd

from fraud_engine.evaluation.report import write_run
from fraud_engine.features.registry import resolve_families
from fraud_engine.models.train import feature_columns, fit, score, to_dataset

log = logging.getLogger(__name__)

# The arm that removes nothing. Every delta is measured against it.
#
# Deliberately not `none`, which is what Phase 04 called its reference. There it
# meant "add no family" and here it would mean "drop no family" — the same label
# on opposite ends of the comparison, in files a reader meets side by side.
REFERENCE = "full"

# Phase 04's reference key, which carries no columns and does not become an arm.
BARE = "none"


def resolve_arms(features_dir) -> dict[str, tuple[str, ...]]:
    """Arm name -> the columns that arm removes.

    The registry is `features/evaluate`'s, read rather than restated: a second
    list of what a family contains would let the two drift with neither looking
    wrong, and the V-block's membership is decided by a fitted threshold that
    only the built matrix knows.

    Args:
        features_dir: Directory holding `{split}.parquet`.

    Returns:
        `{arm: columns to drop}`, the reference first, holding no columns.

    Raises:
        ValueError: If the bare reference has acquired columns, or if two
            families share one. Overlapping families make the deltas
            non-additive, and nothing downstream would show it.
    """
    families = resolve_families(features_dir)

    if families.get(BARE):
        raise ValueError(
            f"`{BARE}` carries columns: {sorted(families[BARE])}; it is Phase 04's "
            "bare probe and has no meaning as a leave-one-out arm"
        )

    arms = {name: columns for name, columns in families.items() if name != BARE}

    seen: dict[str, str] = {}
    for name, columns in arms.items():
        for column in columns:
            if column in seen:
                raise ValueError(
                    f"{column} belongs to both `{seen[column]}` and `{name}`; "
                    "removing either arm would remove part of the other"
                )
            seen[column] = name

    return {REFERENCE: (), **arms}


def drop_columns(frame: pd.DataFrame, columns: tuple[str, ...]) -> pd.DataFrame:
    """The matrix without one arm's columns.

    The guard is the point. A name that no longer matches the matrix would drop
    nothing, the arm would be the reference fitted twice, and its delta would
    come back at exactly zero — which reads as *this family adds nothing*, the
    most plausible-looking wrong answer this experiment can produce. Same reason
    `feature_columns` requires every excluded name to be present.

    Args:
        frame: Any split's matrix.
        columns: The arm's columns, from `resolve_arms`. Empty returns a copy,
            which is the reference arm.

    Returns:
        A new frame; the input is not modified.

    Raises:
        ValueError: If a named column is absent — the registry and the matrix
            have drifted apart.
    """
    missing = [column for column in columns if column not in frame.columns]
    if missing:
        raise ValueError(
            f"columns absent from the matrix: {missing}; "
            "this arm would remove nothing and score as the reference"
        )

    return frame.drop(columns=list(columns))


def measure(
    matrices: dict[str, pd.DataFrame],
    arms: dict[str, tuple[str, ...]],
    model_cfg: dict,
    capacities: list[float],
    paths: dict,
) -> pd.DataFrame:
    """Fit and score every arm, writing each through the Phase 02 harness.

    **The matrices arrive prepared, and that is what makes the arms comparable.**
    The vocabulary and the medians are fitted once, on the full training window,
    and applied before anything is removed — so an arm inherits exactly the
    levels and fills its surviving columns would have had, and the arms differ in
    one thing rather than in one thing plus two refits that happen to agree.

    `model_cfg` is passed through untouched, so the caller chooses the
    instrument. The search runs on the untuned configuration, which does not
    subsample and returns identical digits on repeated fits; a delta there is
    signal with no seed in it. The tuned configuration is stronger and does
    subsample, which puts a bar under every delta wide enough to hide most of
    them — it is where a family that moved goes to be confirmed, not where the
    moving is detected.

    `VAL-CAL` is not scored, and must not be. Deciding which features ship is
    tuning, and `VAL-CAL` is held back so Phase 06's calibrator and threshold
    meet data no tuning decision has touched.

    Args:
        matrices: Prepared matrices, as `train.py` produces them. `train` and
            `val_fit` are required; anything else is ignored.
        arms: `{arm: columns to drop}`, from `resolve_arms`.
        model_cfg: The `model:` config block, with `tuned` set to the
            instrument this pass measures with.
        capacities: Review capacities.
        paths: The `paths:` config block.

    Returns:
        One row per arm: `arm`, `removed`, `n_features`, `pr_auc`, `delta`,
        `best_iteration`. `delta` is against the reference arm and is zero there.

    Raises:
        KeyError: If `arms` carries no reference arm — there would be nothing
            to measure the others against.
    """
    if REFERENCE not in arms:
        raise KeyError(f"no `{REFERENCE}` arm: every delta is measured against it")

    measured = []

    for arm, dropped in arms.items():
        frames = {split: drop_columns(matrices[split], dropped) for split in ("train", "val_fit")}
        columns = feature_columns(frames["train"])

        train = to_dataset(frames["train"], columns)
        val_fit = to_dataset(frames["val_fit"], columns, reference=train)

        booster = fit(train, val_fit, model_cfg)

        scored = score(booster, {"val_fit": frames["val_fit"]}, columns)
        metrics_path, _ = write_run(
            f"ablation_{arm}",
            scored,
            capacities,
            paths["metrics_dir"],
            paths["predictions_dir"],
            splits=("val_fit",),
        )

        pr_auc = json.loads(Path(metrics_path).read_text())["splits"]["val_fit"]["pr_auc"]
        measured.append(
            {
                "arm": arm,
                "removed": len(dropped),
                "n_features": len(columns),
                "pr_auc": pr_auc,
                "best_iteration": booster.best_iteration,
            }
        )
        log.info(
            "arm=%-12s removed=%3d val_fit pr_auc=%.5f -> %s",
            arm,
            len(dropped),
            pr_auc,
            metrics_path,
        )

    comparison = pd.DataFrame(measured)

    # Signed so it reads as the family's contribution, not as the arm's score:
    # removing something that helped comes back negative.
    reference = comparison.loc[comparison["arm"] == REFERENCE, "pr_auc"].item()
    comparison["delta"] = comparison["pr_auc"] - reference

    return comparison
