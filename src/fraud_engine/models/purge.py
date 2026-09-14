"""What the label-maturity gap costs — E1, at three arms.

The purge is the most defensible decision in this project and the one with no
number behind it. Removing it hands a model two advantages production never has,
and E1 measures them apart rather than together: a run trained up to the
validation boundary has *recency*, and one trained over the vacated days has
recency plus *volume*. A retrain cadence can partly buy the first. Nothing buys
the second, because those labels could not exist yet.

**Arms are built beside the shipped pipeline, never over it.** Each runs the
split and feature stages against a config whose outputs are redirected into a
working directory. Editing the shipped config in place and restoring it after
would leave every downstream stage one interruption away from reading a split
nobody chose.

**An arm that starts later cuts its data, not its boundaries.** A declared span
narrower than the table is refused by `validate_splits` as silent row loss, so
such an arm writes its own copy of the pre-split frame and every stage reads
that. E1 registers what the cut costs.
"""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd
import yaml

from fraud_engine.data import splits
from fraud_engine.data.load import DEFAULT_CONFIG_PATH, load_config
from fraud_engine.data.splits import resolve_boundaries
from fraud_engine.data.validate import validate_interim
from fraud_engine.evaluation.report import load_capacities, write_run
from fraud_engine.evaluation.tracking import (
    configure_tracking,
    log_report,
    tracked_child,
    tracked_run,
)
from fraud_engine.features import build
from fraud_engine.models.ablation import fit_without
from fraud_engine.models.train import LABEL, prepare_matrices, resolve_params

log = logging.getLogger(__name__)

# The shipped split, rebuilt rather than read from disk. Reading the shipped
# matrices would compare artifacts produced by two code paths on two days.
REFERENCE_ARM = "purged"

# Every path the split and feature stages write. Named rather than derived from
# the config, because the failure this guards is an arm overwriting the shipped
# artifact it is being compared against — and a stage that starts writing
# somewhere new has to be added here deliberately for that to keep holding.
REDIRECTED = ("splits", "split_summary", "features_dir", "encoders", "amount_stats", "vblock")

# Read and never written. Left where it is for every arm that starts on the
# shipped first day, so those arms read the same rows and differ only in how
# they are labelled. An arm starting later gets its own cut instead — see
# `cut_interim` — and never a write to this one.
SHARED = "interim"


def redirect(config: dict, directory: Path | str, *, train_start: int, gap_days: int) -> dict:
    """The shipped config, pointed at one arm's window and one arm's output.

    Two blocks move and nothing else. `splits` gets the arm's window, which is
    the whole experimental variable — `resolve_boundaries` derives every other
    boundary from it, so an arm is two integers rather than a table of days.
    `paths` gets every written location rewritten under `directory`, keeping
    each file's own name so an arm's directory reads like a small copy of the
    project. An arm starting after the shipped first day also gets `interim`
    redirected, to the cut `build_arm` writes.

    The input is not modified. A caller holding the shipped config after
    building three arms still holds the shipped config.

    Args:
        config: The loaded `config.yaml`.
        directory: Where this arm's artifacts go. Not created here — the stages
            that write into it do that.
        train_start: First training day.
        gap_days: Purged days between train and `VAL-FIT`. Zero runs unpurged.

    Returns:
        A new config, shallow-copied except for the two blocks that change.

    Raises:
        ValueError: If `train_start` precedes the shipped first day, if a
            redirected path is missing from config, or if two of them would
            land on the same name. Either one leaves a stage writing
            where the shipped pipeline writes, and the arm would be measured
            against an artifact it had just overwritten.
    """
    directory = Path(directory)

    missing = [key for key in REDIRECTED if key not in config["paths"]]
    if missing:
        raise ValueError(
            f"paths to redirect are absent from config: {missing}; "
            "an arm would write where the shipped pipeline writes"
        )

    shipped_start = config["splits"]["train_start"]
    if train_start < shipped_start:
        raise ValueError(
            f"train_start={train_start} precedes the shipped first day {shipped_start}; "
            "an arm cannot train on days the table does not hold"
        )

    paths = {**config["paths"]}
    for key in REDIRECTED:
        paths[key] = str(directory / Path(paths[key]).name)

    # A later start reads its own cut of the pre-split frame. Redirected here,
    # alongside everything else, so the arm's config names the file it read.
    if train_start > shipped_start:
        paths[SHARED] = str(directory / Path(config["paths"][SHARED]).name)

    landed = [paths[key] for key in (*REDIRECTED, SHARED) if paths[key] != config["paths"][key]]
    if len(set(landed)) != len(landed):
        raise ValueError(
            f"redirected paths collide under {directory}: {sorted(landed)}; "
            "two stages would write the same file"
        )

    return {
        **config,
        "paths": paths,
        "splits": {**config["splits"], "train_start": train_start, "gap_days": gap_days},
    }


def cut_interim(source: Path | str, destination: Path | str, first_day: int) -> int:
    """Write the pre-split frame from `first_day` onward, and nothing else about it.

    A row filter and a re-validation. The schema is checked again because the
    cut is a new artifact every later stage trusts, and a filter that quietly
    changed a dtype on the way through parquet would surface as a feature
    difference nobody could trace back here.

    Args:
        source: The shipped `interim` parquet. Read, never written.
        destination: Where the cut goes.
        first_day: The first day kept.

    Returns:
        How many rows were dropped.

    Raises:
        ValueError: If `destination` is `source`, or if the cut drops nothing —
            the arm would then be the unpurged arm under another name, and its
            difference from it would read as volume having no effect.
    """
    source, destination = Path(source), Path(destination)
    if source.resolve() == destination.resolve():
        raise ValueError(f"refusing to cut {source} in place; the shipped frame is read-only here")

    frame = pd.read_parquet(source)
    kept = frame[frame["day"] >= first_day]

    dropped = len(frame) - len(kept)
    if dropped == 0:
        raise ValueError(
            f"cutting at day {first_day} drops no rows from {source}; "
            "this arm would be indistinguishable from one that was never cut"
        )

    validate_interim(kept.reset_index(drop=True))

    destination.parent.mkdir(parents=True, exist_ok=True)
    kept.to_parquet(destination, index=False)
    return dropped


def build_arm(config: dict, directory: Path | str, *, train_start: int, gap_days: int) -> dict:
    """Run the split and feature stages for one arm, into its own directory.

    The derived config is written beside the artifacts it produces, and the
    stages are handed its path rather than its contents — which is the interface
    they already have, so nothing about the shipped pipeline changes to
    accommodate this. It also leaves each arm's provenance on disk: what
    produced these matrices is a file next to them, diffable against the
    shipped config.

    Paths inside a config are relative to the working directory rather than to
    the config, so a config living under `data/` still addresses the project.

    An arm that starts after the shipped first day has its cut written before
    either stage runs, so the split stage never sees a row the arm excluded.

    Args:
        config: The loaded `config.yaml`.
        directory: Where this arm's artifacts and its config go. Created here.
        train_start: First training day.
        gap_days: Purged days between train and `VAL-FIT`.

    Returns:
        The arm's config, as `redirect` produced it.
    """
    arm = redirect(config, directory, train_start=train_start, gap_days=gap_days)

    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)

    if arm["paths"][SHARED] != config["paths"][SHARED]:
        dropped = cut_interim(config["paths"][SHARED], arm["paths"][SHARED], train_start)
        log.info("cut %d rows before day %d -> %s", dropped, train_start, arm["paths"][SHARED])

    config_path = directory / "config.yaml"
    config_path.write_text(yaml.safe_dump(arm, sort_keys=False))

    splits.main(config_path)
    build.main(config_path)

    return arm


def measure(
    arms: dict[str, dict],
    model_cfg: dict,
    capacities: list[float],
    paths: dict,
) -> pd.DataFrame:
    """Fit the untuned reference on every arm and score `VAL-FIT`.

    Each arm brings its own matrices, and its own fitted vocabulary and medians
    with them. That is the experiment rather than a detail: the encoders and
    aggregates are fitted on `TRAIN`, and `TRAIN` is what moves — so an arm
    scores validation encoded against its own training window. Reusing one
    arm's fits across the others would score an unpurged model through a purged
    model's encoders, which is none of the three runs.

    The instrument is whatever `model_cfg` carries; `main` passes the untuned
    reference. The tuned knobs were selected on `VAL-FIT` under the shipped
    split, and carrying them into an arm that trains on different data would
    import a selection that arm never made.

    Args:
        arms: `{name: arm config}`, the reference first. `REFERENCE_ARM` must be
            among them.
        model_cfg: The `model:` config block, carrying the instrument.
        capacities: Review capacities.
        paths: The shipped `paths:` block — records go where every other run's
            records go, not into the arm directories.

    Returns:
        One row per arm: `arm`, `train_rows`, `train_frauds`, `pr_auc`,
        `best_iteration`, `delta`. `delta` is against the purged arm.

    Raises:
        KeyError: If the purged arm is absent. It is what the others are read
            against, and a table of unpurged runs answers nothing.
    """
    if REFERENCE_ARM not in arms:
        raise KeyError(f"no `{REFERENCE_ARM}` arm: it is what the gap is measured against")

    measured = []

    for name, arm in arms.items():
        # The arm's own fits, from the arm's own training window. That is the
        # experiment: reusing one arm's tables would score an unpurged model
        # through a purged model's encoders, which is none of the three runs.
        matrices, _, _ = prepare_matrices(
            arm["paths"]["features_dir"], model_cfg, ("train", "val_fit")
        )

        # The arm's effective config rides on the child. No committed config
        # holds this training window, so the commit alone cannot say what
        # produced the number.
        with tracked_child(
            f"purge_{name}",
            {
                **resolve_params(model_cfg),
                "arm": name,
                "train_rows": len(matrices["train"]),
                "train_frauds": int(matrices["train"][LABEL].sum()),
            },
            config=arm,
        ):
            scored, best_iteration, _ = fit_without(matrices, (), model_cfg)
            metrics_path, _ = write_run(
                f"purge_{name}",
                scored,
                capacities,
                paths["metrics_dir"],
                paths["predictions_dir"],
                splits=("val_fit",),
            )
            report = log_report(metrics_path, {"best_iteration": best_iteration})

        pr_auc = report["splits"]["val_fit"]["pr_auc"]
        measured.append(
            {
                "arm": name,
                "train_days": f"{arm['splits']['train_start']}-{_train_end(arm)}",
                "train_rows": len(matrices["train"]),
                "train_frauds": int(matrices["train"][LABEL].sum()),
                "pr_auc": pr_auc,
                "best_iteration": best_iteration,
            }
        )
        log.info("arm=%-10s val_fit pr_auc=%.5f -> %s", name, pr_auc, metrics_path)

    comparison = pd.DataFrame(measured)

    reference = comparison.loc[comparison["arm"] == REFERENCE_ARM, "pr_auc"].item()
    comparison["delta"] = comparison["pr_auc"] - reference

    return comparison


def _train_end(arm: dict) -> int:
    """The arm's last training day, derived the way `splits.py` derives it."""
    return resolve_boundaries(arm["splits"])["train"][1]


# Each arm's training window. The evaluation boundaries are not here: they come
# from the shipped config untouched, and an arm that moved them would compare
# two models on two different validation sets.
ARMS = {
    REFERENCE_ARM: {"train_start": 1, "gap_days": 30},
    "recent": {"train_start": 31, "gap_days": 0},
    "unpurged": {"train_start": 1, "gap_days": 0},
}


def main(config_path: Path = DEFAULT_CONFIG_PATH) -> None:
    """Build every arm, measure it, and write the comparison.

    Wiring only. Invoked by `make purge` as
    `python -m fraud_engine.models.purge`.

    Each arm rebuilds splits and features from `interim`, so the shipped
    artifacts are read by nothing here and written by nothing here. `VAL-CAL` is
    never loaded: three arms compete for an explanation and none of them ships.

    Args:
        config_path: Path to `config.yaml`.
    """
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    config = load_config(config_path)
    paths, model_cfg = config["paths"], config["model"]

    working = Path(paths["purge_dir"])

    arms = {}
    for name, window in ARMS.items():
        log.info("building arm %s — %s", name, window)
        arms[name] = build_arm(config, working / name, **window)

    capacities = load_capacities(load_config(Path(paths["cost_matrix"])))

    # The untuned reference, as with every other Phase 05 comparison. The tuned
    # knobs were selected on VAL-FIT under the shipped split; carrying them into
    # an arm that trains on other data imports a selection it never made.
    configure_tracking(config["tracking"])
    with tracked_run("purge", {"arms": ",".join(arms)}, config_path):
        comparison = measure(arms, {**model_cfg, "tuned": {}}, capacities, paths)

    path = Path(paths["purge"])
    path.parent.mkdir(parents=True, exist_ok=True)
    comparison.to_csv(path, index=False)

    log.info("\n%s", comparison.to_string(index=False))
    log.info("wrote %s", path)


if __name__ == "__main__":
    main()
