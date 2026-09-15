"""How the rehearsal responds when one cost assumption moves (decision-policy.md §5).

The costs are assumptions, not measurements, so the headline is only as strong as
its weakest one. Each is swept alone, the others held at version 1, and every row of
§4 — the rules engine included — is recomputed at every point: its actions do not
move with the costs, but what they cost does.

On VAL-CAL with the out-of-fold probabilities, like the rehearsal. Never on test: a
sweep on test is a threshold search on test.
"""

from __future__ import annotations

import json
import logging
from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path

import mlflow
import numpy as np
import pandas as pd

from fraud_engine.data.load import DEFAULT_CONFIG_PATH, load_config
from fraud_engine.evaluation.cost import Costs, load_costs
from fraud_engine.evaluation.plots import plot_sensitivity, save_figure
from fraud_engine.evaluation.policy import REFERENCE, load_rehearsal_frame, rehearse
from fraud_engine.evaluation.report import (
    git_revision,
    load_capacities,
    load_operating_capacity,
)
from fraud_engine.evaluation.tracking import configure_tracking, tracked_run
from fraud_engine.models.calibrate import SPLIT

log = logging.getLogger(__name__)

NAME = "sensitivity_val_cal"
HEADLINE = "ev"

# The reading rule of problem-statement.md §5: a win at 15%, not claimed below 5%.
WIN, NOISE = 0.15, 0.05

COST_ASSUMPTIONS = ("false_positive", "chargeback_fee")
CAPACITY = "review_capacity"

FIGURE = "sensitivity_false_positive.png"


def sweep_values(sensitivity: dict, costs: Costs, capacity: float) -> dict[str, list[float]]:
    """The values each assumption is swept over, from `cost_matrix.yaml`.

    `min`, `max` and `steps` give `steps` evenly spaced points, ends included. The
    value the headline was computed at is added when the grid misses it, so every
    sweep passes through the headline and can be read against it.

    Returns:
        `{assumption: ascending values}`.
    """
    values = {}
    for name in COST_ASSUMPTIONS:
        spec = sensitivity[name]
        grid = np.linspace(spec["min"], spec["max"], spec["steps"])
        values[name] = sorted({round(float(v), 10) for v in grid} | {getattr(costs, name)})

    values[CAPACITY] = load_capacities(
        {
            "constraints": {"review_capacity": {"value": capacity}},
            "sensitivity": sensitivity,
        }
    )
    return values


def sweep(frame: pd.DataFrame, costs: Costs, capacity: float, values: dict) -> pd.DataFrame:
    """Every §4 row at every point of every sweep.

    Returns:
        One row per (assumption, value, policy): `usd_per_1000`, `reviews_per_day`,
        `block_rate`, `reduction_vs_rules`, and `is_base` for the headline's value.
    """
    records = []

    for assumption, grid in values.items():
        for value in grid:
            if assumption == CAPACITY:
                point_costs, point_capacity = costs, value
                is_base = value == capacity
            else:
                point_costs, point_capacity = replace(costs, **{assumption: value}), capacity
                is_base = value == getattr(costs, assumption)

            for policy, summary in rehearse(frame, point_costs, point_capacity).items():
                records.append(
                    {
                        "assumption": assumption,
                        "value": value,
                        "is_base": is_base,
                        "policy": policy,
                        **summary,
                    }
                )

    return pd.DataFrame(records)


def verdict(points: pd.DataFrame) -> dict[str, dict]:
    """Per assumption: the headline's smallest margin over the rules, where, and against which bars."""
    headline = points[points["policy"] == HEADLINE]
    verdicts = {}

    for assumption, rows in headline.groupby("assumption", sort=False):
        worst = rows.loc[rows["reduction_vs_rules"].idxmin()]
        verdicts[assumption] = {
            "min_reduction": float(worst["reduction_vs_rules"]),
            "at_value": float(worst["value"]),
            "max_reduction": float(rows["reduction_vs_rules"].max()),
            "above_win_everywhere": bool((rows["reduction_vs_rules"] >= WIN).all()),
            "above_noise_everywhere": bool((rows["reduction_vs_rules"] >= NOISE).all()),
            "positive_everywhere": bool((rows["reduction_vs_rules"] > 0).all()),
        }

    return verdicts


def false_positive_points(points: pd.DataFrame) -> pd.DataFrame:
    """The chart's input: rules and EV side by side over the false-positive sweep."""
    rows = points[points["assumption"] == "false_positive"]

    def by_value(policy: str, column: str) -> pd.Series:
        return rows[rows["policy"] == policy].set_index("value")[column].sort_index()

    ev = by_value(HEADLINE, "usd_per_1000")

    return pd.DataFrame(
        {
            "value": ev.index,
            "rules_usd": by_value(REFERENCE, "usd_per_1000").loc[ev.index].to_numpy(),
            "ev_usd": ev.to_numpy(),
            "reduction": by_value(HEADLINE, "reduction_vs_rules").loc[ev.index].to_numpy(),
        }
    )


def main(config_path: Path = DEFAULT_CONFIG_PATH) -> None:
    """Sweep, judge each assumption, and draw the false-positive chart.

    Wiring only. Invoked by `make sensitivity` as `python -m fraud_engine.evaluation.sensitivity`.

    Args:
        config_path: Path to `config.yaml`.
    """
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    config = load_config(config_path)
    paths = config["paths"]

    configure_tracking(config["tracking"])

    cost_matrix = load_config(Path(paths["cost_matrix"]))
    costs = load_costs(cost_matrix)
    capacity = load_operating_capacity(cost_matrix)
    method = json.loads(Path(paths["calibration"]).read_text())["selected"]

    frame = load_rehearsal_frame(paths["predictions_dir"], paths["interim"], method)
    values = sweep_values(cost_matrix["sensitivity"], costs, capacity)

    params = {"split": SPLIT, "calibration": method, "cost_matrix_version": costs.version}

    with tracked_run(NAME, params, config_path):
        points = sweep(frame, costs, capacity, values)
        verdicts = verdict(points)
        mlflow.log_metrics(
            {
                f"{name}.{key}": float(value)
                for name, v in verdicts.items()
                for key, value in v.items()
            }
        )

    for name, v in verdicts.items():
        log.info(
            "%-16s EV vs rules: min %+.1f%% at %g, max %+.1f%%   win everywhere: %s",
            name,
            100 * v["min_reduction"],
            v["at_value"],
            100 * v["max_reduction"],
            v["above_win_everywhere"],
        )

    record = {
        "name": NAME,
        "created": datetime.now(UTC).isoformat(timespec="seconds"),
        "git_revision": git_revision(),
        "split": SPLIT,
        "probabilities": f"{method}, out-of-fold",
        "cost_matrix_version": costs.version,
        "base": {**{n: getattr(costs, n) for n in COST_ASSUMPTIONS}, CAPACITY: capacity},
        "bars": {"win": WIN, "noise": NOISE},
        "verdicts": verdicts,
        "points": points.to_dict("records"),
    }
    path = Path(paths["sensitivity"])
    path.write_text(json.dumps(record, indent=2) + "\n")
    log.info("wrote %s", path)

    figure = save_figure(
        plot_sensitivity(
            false_positive_points(points),
            base=costs.false_positive,
            bars=(WIN, NOISE),
            xlabel="Cost of declining a legitimate customer (USD)",
            title="Sensitivity to the false-positive cost — VAL-CAL",
        ),
        Path(paths["figures_dir"]) / FIGURE,
    )
    log.info("wrote %s", figure)


if __name__ == "__main__":
    main()
