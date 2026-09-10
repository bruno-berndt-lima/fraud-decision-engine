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

from fraud_engine.data.load import DEFAULT_CONFIG_PATH, load_config
from fraud_engine.evaluation.report import load_capacities, write_run
from fraud_engine.features.registry import TIER_0, resolve_families, resolve_tiers
from fraud_engine.models.train import (
    apply_categories,
    apply_medians,
    feature_columns,
    fit,
    fit_categories,
    fit_medians,
    load_split_matrices,
    score,
    to_dataset,
)

log = logging.getLogger(__name__)

# The arm that removes nothing. Every delta is measured against it.
#
# Deliberately not `none`, which is what Phase 04 called its reference. There it
# meant "add no family" and here it would mean "drop no family" — the same label
# on opposite ends of the comparison, in files a reader meets side by side.
REFERENCE = "full"

# Phase 04's reference key, which carries no columns and does not become an arm.
BARE = "none"

# Everything this project could construct from a raw transaction feed — which is
# the complement of the inherited tier, the expensive-to-serve tier included.
# Expensive and impossible are different claims, and E7 turns on not merging them.
REPRODUCIBLE = "reproducible"


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


def resolve_tier_arms(features_dir) -> dict[str, tuple[str, ...]]:
    """The serving-tier arms: what is lost by keeping only what could be rebuilt.

    One arm, and it removes the inherited tier. The live-entity tier stays — it
    is expensive to serve and perfectly possible to build, and dropping it here
    would answer E3's question a second time under E7's name.

    E3's own arm is not here either, for the opposite reason: it removes the
    live-entity tier, which is exactly the `velocity` family, so naming it would
    write a second record identical to one this run already produces.

    Args:
        features_dir: Directory holding `{split}.parquet`.

    Returns:
        `{arm: columns to drop}`, carrying no reference — it is merged into one
        that has it.

    Raises:
        ValueError: Per `resolve_tiers`, if the matrix and the published
            inventory have drifted apart.
    """
    return {REPRODUCIBLE: resolve_tiers(features_dir)[TIER_0]}


def all_arms(features_dir) -> dict[str, tuple[str, ...]]:
    """Every arm one run measures: the families, and the tiers.

    Merged rather than run twice so they share a reference fit. The families
    answer E4 and the tier arm answers E7, and a delta is only comparable to
    another delta measured against the same reference — two runs would fit an
    identical reference twice and invite the two tables to be read as one.

    Args:
        features_dir: Directory holding `{split}.parquet`.

    Returns:
        `{arm: columns to drop}`, the reference first.

    Raises:
        ValueError: If a tier arm and a family arm share a name. They index one
            dict, so a collision would drop an arm silently and the missing one
            would look like an experiment nobody ran.
    """
    families = resolve_arms(features_dir)
    tiers = resolve_tier_arms(features_dir)

    collisions = sorted(set(families) & set(tiers))
    if collisions:
        raise ValueError(f"tier and family arms share a name: {collisions}; one would be lost")

    return {**families, **tiers}


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


def fit_without(
    matrices: dict[str, pd.DataFrame], dropped: tuple[str, ...], model_cfg: dict
) -> tuple[pd.DataFrame, int, int]:
    """Fit with these columns gone and score `VAL-FIT`.

    One arm, whether the columns came from the registry or from a draw. The
    two callers differ in what they do with the result — one records it as a
    run, the other only needs its number — and not in how it was produced,
    which is the property that lets a family be compared to the floor at all.

    Args:
        matrices: Prepared matrices. `train` and `val_fit` are required.
        dropped: Columns to remove. Empty is the reference arm.
        model_cfg: The `model:` config block, carrying the instrument.

    Returns:
        `(scored, best_iteration, n_features)`, the frame carrying `VAL-FIT`
        alone.
    """
    frames = {split: drop_columns(matrices[split], dropped) for split in ("train", "val_fit")}
    columns = feature_columns(frames["train"])

    train = to_dataset(frames["train"], columns)
    val_fit = to_dataset(frames["val_fit"], columns, reference=train)

    booster = fit(train, val_fit, model_cfg)
    scored = score(booster, {"val_fit": frames["val_fit"]}, columns)

    return scored, booster.best_iteration, len(columns)


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
        scored, best_iteration, n_features = fit_without(matrices, dropped, model_cfg)

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
                "n_features": n_features,
                "pr_auc": pr_auc,
                "best_iteration": best_iteration,
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


def main(config_path: Path = DEFAULT_CONFIG_PATH) -> None:
    """Measure every arm on the untuned configuration and write the comparison.

    Wiring only. Invoked by ``make ablation`` as
    ``python -m fraud_engine.models.ablation``.

    `VAL-CAL` is not loaded. Naming it in `write_run` would keep it out of the
    records, but a field of candidates should not be able to reach the
    calibration slice at all — the same structural guarantee `tune.py` and
    `seeds.py` already make.

    Args:
        config_path: Path to `config.yaml`.
    """
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    config = load_config(config_path)
    paths, model_cfg = config["paths"], config["model"]

    matrices = load_split_matrices(paths["features_dir"], ("train", "val_fit"))
    vocabulary = fit_categories(matrices["train"], model_cfg["min_category_rows"])
    matrices = {split: apply_categories(frame, vocabulary) for split, frame in matrices.items()}

    # Fitted on the full training window and applied before anything is removed,
    # so an arm inherits exactly what its surviving columns would have had.
    if model_cfg["impute"]:
        medians = fit_medians(matrices["train"], feature_columns(matrices["train"]))
        matrices = {split: apply_medians(frame, medians) for split, frame in matrices.items()}

    arms = all_arms(paths["features_dir"])
    capacities = load_capacities(load_config(Path(paths["cost_matrix"])))

    # `tuned` emptied, and `impute` left as config has it. The instrument is the
    # untuned reference exactly as E2 measured it — repeated fits return
    # identical digits, so a delta carries no seed noise. Anything else would be
    # a third configuration nobody has a spread for.
    comparison = measure(matrices, arms, {**model_cfg, "tuned": {}}, capacities, paths)

    path = Path(paths["ablation"])
    path.parent.mkdir(parents=True, exist_ok=True)
    comparison.to_csv(path, index=False)

    log.info("\n%s", comparison.to_string(index=False))
    log.info("wrote %s", path)


if __name__ == "__main__":
    main()
