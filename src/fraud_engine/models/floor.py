"""The bar the tree ablation is read against — how far removing *anything* moves it.

Its own command rather than part of `ablation`, on the same reasoning that split
the probe's floor from the family run: fifty fits answer a question the six arms
do not change, and folding them together would pay for the bar every time a
family is re-measured.

**Width-matched, and that is the whole design.** How much removing columns costs
depends on how many are removed, so a family's delta is read only against draws
of its own width — and the widths are the family sizes, read from the built
matrices rather than configured, so no choice about them survives having seen a
result. E4 registers the design and the order it was registered in.

What this measures that the probe's floor could not: each draw early-stops at
its own round, so the spread already contains the unfairness between arms that
stop decades apart, instead of holding it constant and reporting a cleaner
number than the comparison deserves.
"""

from __future__ import annotations

import logging
from pathlib import Path

import mlflow
import numpy as np
import pandas as pd

from fraud_engine.data.load import DEFAULT_CONFIG_PATH, load_config
from fraud_engine.evaluation.report import evaluate_splits, load_capacities
from fraud_engine.evaluation.tracking import (
    configure_tracking,
    flatten_metrics,
    tracked_child,
    tracked_run,
)
from fraud_engine.models.ablation import REFERENCE, all_arms, fit_without
from fraud_engine.models.train import feature_columns, prepare_matrices, resolve_params

log = logging.getLogger(__name__)


def draw(columns: list[str], width: int, rng: np.random.Generator) -> tuple[str, ...]:
    """`width` feature names, sampled uniformly and without replacement.

    Uniform over all features, with no regard to which family a name belongs to.
    A draw that excluded the families would be measuring something narrower than
    the question — what removing this many arbitrary columns costs — and the
    families are most of the matrix anyway.

    Args:
        columns: Every feature name.
        width: How many to remove.
        rng: Seeded per draw by the caller, so one draw can be reproduced
            without replaying the ones before it.

    Returns:
        The drawn names.

    Raises:
        ValueError: If `width` exceeds what there is to draw from, which would
            leave a model with no features rather than a wide arm.
    """
    if width > len(columns):
        raise ValueError(f"width {width} exceeds {len(columns)} features; nothing would be left")

    return tuple(str(name) for name in rng.choice(columns, size=width, replace=False))


def log_evaluation(evaluated: dict, best_iteration: int) -> None:
    """Log an in-memory evaluation onto the open child, under the record's names.

    The draws write no JSON of their own — fifty records in `reports/` would be
    noise — so they go through `flatten_metrics` directly, which keeps their
    metric names identical to every recorded run's.
    """
    mlflow.log_metrics({**flatten_metrics({"splits": evaluated}), "best_iteration": best_iteration})


def measure(
    matrices: dict[str, pd.DataFrame],
    widths: list[int],
    draws: int,
    model_cfg: dict,
    capacities: list[float],
) -> pd.DataFrame:
    """Remove `draws` random column sets at each width, and score every one.

    The reference is fitted once, here, rather than taken from the ablation's
    record: a bar computed against a different fit than the draws would be
    measuring that difference too.

    Args:
        matrices: Prepared matrices, as `train.py` produces them.
        widths: The family sizes. Not configured — see the module docstring.
        draws: Random column sets per width.
        model_cfg: The `model:` config block, carrying the instrument. Must be
            the one the arms were measured on; a bar drawn on another
            configuration is not a bar.
        capacities: Review capacities.

    Returns:
        One row per draw: `width`, `draw`, `n_features`, `pr_auc`,
        `best_iteration`, `delta`. The reference is not a row; it is what
        `delta` is measured from.
    """
    columns = feature_columns(matrices["train"])

    params = resolve_params(model_cfg)

    with tracked_child(f"floor_{REFERENCE}", {**params, "width": 0}):
        scored, best_iteration, _ = fit_without(matrices, (), model_cfg)
        evaluated = evaluate_splits(scored, capacities, ("val_fit",))
        log_evaluation(evaluated, best_iteration)
    reference = evaluated["val_fit"]["pr_auc"]
    log.info("%s val_fit pr_auc=%.5f  best_iteration=%d", REFERENCE, reference, best_iteration)

    measured = []

    for width in widths:
        for index in range(draws):
            # Seeded per draw, so draw 7 at width 6 is the same set whether or
            # not the ones before it ran.
            dropped = draw(columns, width, np.random.default_rng([model_cfg["seed"], width, index]))

            with tracked_child(
                f"floor_w{width}_d{index}",
                {**params, "width": width, "draw": index},
                artifacts={"removed_columns.json": list(dropped)},
            ):
                scored, best_iteration, n_features = fit_without(matrices, dropped, model_cfg)
                evaluated = evaluate_splits(scored, capacities, ("val_fit",))
                log_evaluation(evaluated, best_iteration)
            pr_auc = evaluated["val_fit"]["pr_auc"]

            measured.append(
                {
                    "width": width,
                    "draw": index,
                    "n_features": n_features,
                    "pr_auc": pr_auc,
                    "best_iteration": best_iteration,
                }
            )
            log.info(
                "width=%-4d draw=%-3d val_fit pr_auc=%.5f  delta=%+.5f  best_iteration=%d",
                width,
                index,
                pr_auc,
                pr_auc - reference,
                best_iteration,
            )

    floor = pd.DataFrame(measured)
    floor["delta"] = floor["pr_auc"] - reference

    return floor


def summarize(floor: pd.DataFrame) -> pd.DataFrame:
    """Per width: the spread, and the bar a family at that width has to clear.

    The bar is the largest *absolute* delta rather than the largest positive
    one. Removing columns can move the metric either way — the frequency family
    is the standing demonstration that a feature set can cost more than it pays
    — so a one-sided bar would rule on half the arms and wave the other half
    through.

    Args:
        floor: From `measure`.

    Returns:
        One row per width: `width`, `draws`, `sd`, `bar`.
    """
    grouped = floor.groupby("width")["delta"]

    return pd.DataFrame(
        {
            "width": grouped.size().index,
            "draws": grouped.size().to_numpy(),
            "sd": grouped.std().to_numpy(),
            "bar": grouped.apply(lambda deltas: deltas.abs().max()).to_numpy(),
        }
    )


def main(config_path: Path = DEFAULT_CONFIG_PATH) -> None:
    """Measure the bar at every family width and write it to reports/.

    Wiring only. Invoked by `make ablation-floor` as
    `python -m fraud_engine.models.floor`.

    `VAL-CAL` is not loaded, for the reason `ablation.main` gives.

    Args:
        config_path: Path to `config.yaml`.
    """
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    config = load_config(config_path)
    paths, model_cfg = config["paths"], config["model"]

    matrices, _, _ = prepare_matrices(paths["features_dir"], model_cfg, ("train", "val_fit"))

    # The arm sizes, and nothing else. Deduplicated because two arms of the same
    # width share a bar — the draws do not know which family they stood in for.
    # Sorted only so the log reads in order; the reference removes nothing and
    # has no bar to measure.
    arms = all_arms(paths["features_dir"])
    widths = sorted({len(dropped) for arm, dropped in arms.items() if arm != REFERENCE})

    capacities = load_capacities(load_config(Path(paths["cost_matrix"])))

    configure_tracking(config["tracking"])
    run_params = {"draws": model_cfg["floor_draws"], "widths": ",".join(map(str, widths))}
    with tracked_run("ablation_floor", run_params, config_path):
        floor = measure(
            matrices, widths, model_cfg["floor_draws"], {**model_cfg, "tuned": {}}, capacities
        )

    path = Path(paths["ablation_floor"])
    path.parent.mkdir(parents=True, exist_ok=True)
    floor.to_csv(path, index=False)

    log.info("\n%s", summarize(floor).to_string(index=False))
    log.info("wrote %s", path)


if __name__ == "__main__":
    main()
