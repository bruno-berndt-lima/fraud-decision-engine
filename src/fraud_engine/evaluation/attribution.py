"""What the headline was missing: a fair cut beside the break-even, and how far the days move it.

`docs/decision-policy.md` §8. Reads what is already on disk — test's persisted vector and
`VAL-CAL`'s out-of-fold probabilities — so no model is loaded and test is not scored again.
Nothing here can change the frozen policy.

**The ladder** changes one thing per row: the naive 0.5 cut, the best single cut chosen on
`VAL-CAL`, the break-even with nothing reviewed, and the EV policy. Each step isolates
choosing the cut, the threshold moving with the amount, and review.

**Proven before measured.** §4's rows, recomputed from the two files, must equal the
rehearsal's and the headline's records exactly before a ladder row is costed.

Its own module rather than functions in `cost.py` or `arms.py`: both are prerequisites of
the headline, which refuses to rerun, so an edit to either would leave it permanently
stale.
"""

from __future__ import annotations

import json
import logging
from datetime import UTC, datetime
from itertools import pairwise
from pathlib import Path

import numpy as np
import pandas as pd

from fraud_engine.data.load import DEFAULT_CONFIG_PATH, load_config
from fraud_engine.evaluation.arms import paired_day_bootstrap
from fraud_engine.evaluation.cost import (
    Costs,
    Decision,
    allow_or_block,
    ev_policy,
    load_costs,
    naive_policy,
    rules_policy,
)
from fraud_engine.evaluation.policy import REFERENCE, load_rehearsal_frame, rehearse
from fraud_engine.evaluation.report import git_revision, load_operating_capacity
from fraud_engine.models.train import LABEL

log = logging.getLogger(__name__)

NAME = "attribution_test"
HEADLINE_PREDICTIONS = "headline_test"
HEADLINE = "ev"
LADDER = ("naive", "global_cut", "break_even", "ev")
COMPARED = ("usd_per_1000", "reviews_per_day", "block_rate", "reduction_vs_rules")


def best_global_cut(p: np.ndarray, y: np.ndarray, amount: np.ndarray, costs: Costs) -> float:
    """The single cut that minimises realised cost, blocking at `p >= t` (§8.1).

    Searched over every distinct probability. Cuts of equal cost resolve to the highest,
    which declines the fewest customers.

    Raises:
        ValueError: If no cut is cheaper than blocking nothing, so there is no cut to name.
    """
    order = np.argsort(p, kind="stable")
    p, y, amount = p[order], y[order], amount[order]

    allow_cost = y * (amount + costs.chargeback_fee)
    block_cost = (1 - y) * costs.false_positive
    allowed_below = np.concatenate(([0.0], np.cumsum(allow_cost)))
    blocked_from = np.concatenate((np.cumsum(block_cost[::-1])[::-1], [0.0]))

    values, first = np.unique(p, return_index=True)
    cost = allowed_below[first] + blocked_from[first]

    if cost.min() >= allowed_below[-1]:
        raise ValueError("no cut is cheaper than blocking nothing")
    return float(values[np.flatnonzero(cost == cost.min())[-1]])


def break_even_only(p: np.ndarray, amount: np.ndarray, costs: Costs) -> Decision:
    """§2's allow-or-block with nothing reviewed: the EV policy less its review queue."""
    return Decision(fallback=allow_or_block(p, amount, costs), review_share=np.zeros(len(p)))


def ladder(frame: pd.DataFrame, costs: Costs, capacity: float, cut: float) -> dict[str, Decision]:
    """The rules engine and the four ladder rows, each deciding every row of `frame`."""
    p = frame["calibrated"].to_numpy()
    amount = frame["amount"].to_numpy(dtype="float64")
    day = frame["day"].to_numpy()
    return {
        REFERENCE: rules_policy(frame["rules_score"].to_numpy(), day, capacity),
        "naive": naive_policy(p),
        "global_cut": naive_policy(p, cut),
        "break_even": break_even_only(p, amount, costs),
        "ev": ev_policy(p, amount, day, costs, capacity),
    }


def reduction_interval(
    day: np.ndarray,
    policy_cost: np.ndarray,
    reference_cost: np.ndarray,
    resamples: int,
    interval: float,
    seed: int,
) -> tuple[float, float]:
    """Interval for `1 - policy / reference`, on the draws `paired_day_bootstrap` makes.

    The same days, the same generator and the same call, so this interval and the one on
    the USD difference describe one set of resamples.
    """
    days, index = np.unique(np.asarray(day), return_inverse=True)
    policy = np.bincount(index, weights=policy_cost)
    reference = np.bincount(index, weights=reference_cost)

    draws = np.random.default_rng(seed).integers(0, days.size, size=(resamples, days.size))
    reduction = 1 - policy[draws].sum(axis=1) / reference[draws].sum(axis=1)

    tail = (1 - interval) / 2
    low, high = np.quantile(reduction, [tail, 1 - tail])
    return float(low), float(high)


def check_rows(computed: dict[str, dict], record: dict, name: str) -> None:
    """Refuse rows that differ at all from the ones `name` recorded.

    Raises:
        ValueError: Naming the first row and quantity that differs.
    """
    for row, summary in record["policies"].items():
        for key in COMPARED:
            if computed[row][key] != summary[key]:
                raise ValueError(
                    f"{row}.{key} is {computed[row][key]!r}, and {name} recorded {summary[key]!r}"
                )


def measure(frame: pd.DataFrame, decisions: dict[str, Decision], costs: Costs, cfg: dict) -> dict:
    """Every row's summary, each ladder step and the headline with their intervals."""
    y = frame[LABEL].to_numpy()
    amount = frame["amount"].to_numpy(dtype="float64")
    day = frame["day"].to_numpy()
    bootstrap = (cfg["bootstrap_resamples"], cfg["interval"], cfg["seed"])

    rows = {name: d.summary(y, amount, day, costs) for name, d in decisions.items()}
    reference = rows[REFERENCE]["usd_per_1000"]
    for summary in rows.values():
        summary["reduction_vs_rules"] = float(1 - summary["usd_per_1000"] / reference)
    cost = {name: d.cost(y, amount, costs) for name, d in decisions.items()}

    steps = []
    for before, after in pairwise(LADDER):
        low, high = paired_day_bootstrap(day, cost[after], cost[before], *bootstrap)
        steps.append(
            {
                "from": before,
                "to": after,
                "usd_per_1000": rows[after]["usd_per_1000"] - rows[before]["usd_per_1000"],
                "usd_interval": [low, high],
                "points": 100
                * (rows[after]["reduction_vs_rules"] - rows[before]["reduction_vs_rules"]),
                "interval_excludes_zero": high < 0 or low > 0,
            }
        )

    low, high = paired_day_bootstrap(day, cost[HEADLINE], cost[REFERENCE], *bootstrap)
    headline = {
        "usd_per_1000": rows[HEADLINE]["usd_per_1000"] - reference,
        "usd_interval": [low, high],
        "interval_excludes_zero": high < 0 or low > 0,
        "reduction": rows[HEADLINE]["reduction_vs_rules"],
        "reduction_interval": list(
            reduction_interval(day, cost[HEADLINE], cost[REFERENCE], *bootstrap)
        ),
    }
    return {"rows": rows, "steps": steps, "headline": headline}


def main(config_path: Path = DEFAULT_CONFIG_PATH) -> None:
    """Prove both records, choose the cut on VAL-CAL, and measure the ladder on test.

    Wiring only. Invoked by `make attribution` as
    `python -m fraud_engine.evaluation.attribution`.

    Args:
        config_path: Path to `config.yaml`.
    """
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    config = load_config(config_path)
    paths, cfg = config["paths"], config["attribution"]
    cost_matrix = load_config(Path(paths["cost_matrix"]))
    costs, capacity = load_costs(cost_matrix), load_operating_capacity(cost_matrix)

    predictions = Path(paths["predictions_dir"]) / f"{HEADLINE_PREDICTIONS}.parquet"
    for required in (predictions, Path(paths["headline"])):
        if not required.exists():
            raise FileNotFoundError(f"{required} is absent; `make headline` writes it")

    method = json.loads(Path(paths["calibration"]).read_text())["selected"]
    val_cal = load_rehearsal_frame(paths["predictions_dir"], paths["interim"], method)
    test = pd.read_parquet(predictions)

    check_rows(
        rehearse(val_cal, costs, capacity), json.loads(Path(paths["rehearsal"]).read_text()),
        "the rehearsal",
    )  # fmt: skip
    check_rows(
        rehearse(test, costs, capacity), json.loads(Path(paths["headline"]).read_text()),
        "the headline",
    )  # fmt: skip
    log.info("proofs passed: §4's rows reproduce the rehearsal and the headline exactly")

    cut = best_global_cut(
        val_cal["calibrated"].to_numpy(),
        val_cal[LABEL].to_numpy(),
        val_cal["amount"].to_numpy(dtype="float64"),
        costs,
    )
    in_sample = naive_policy(val_cal["calibrated"].to_numpy(), cut).summary(
        val_cal[LABEL].to_numpy(),
        val_cal["amount"].to_numpy(dtype="float64"),
        val_cal["day"].to_numpy(),
        costs,
    )
    measured = measure(test, ladder(test, costs, capacity, cut), costs, cfg)

    record = {
        "name": NAME,
        "created": datetime.now(UTC).isoformat(timespec="seconds"),
        "git_revision": git_revision(),
        "split": "test",
        "days": int(np.unique(test["day"]).size),
        "probabilities": f"{method}; out-of-fold on val_cal, the shipped calibrator on test",
        "cost_matrix_version": costs.version,
        "review_capacity": capacity,
        "bootstrap": cfg | {"unit": "day"},
        "global_cut": {
            "threshold": cut,
            "chosen_on": "val_cal",
            "val_cal_in_sample": in_sample,
        },
        **measured,
    }
    Path(paths["attribution"]).write_text(json.dumps(record, indent=2) + "\n")

    log.info("global cut chosen on val_cal: block at p >= %.4f", cut)
    for name in (REFERENCE, *LADDER):
        row = measured["rows"][name]
        log.info(
            "%-11s $%9.2f per 1,000   block %.4f   vs rules %+.1f%%",
            name,
            row["usd_per_1000"],
            row["block_rate"],
            100 * row["reduction_vs_rules"],
        )
    for step in measured["steps"]:
        log.info(
            "%s -> %s: %+.1f points, $%+.2f per 1,000 [%+.2f, %+.2f]",
            step["from"],
            step["to"],
            step["points"],
            step["usd_per_1000"],
            *step["usd_interval"],
        )
    head = measured["headline"]
    log.info(
        "headline: $%+.2f per 1,000 [%+.2f, %+.2f], reduction %.1f%% [%.1f%%, %.1f%%]",
        head["usd_per_1000"],
        *head["usd_interval"],
        100 * head["reduction"],
        *(100 * v for v in head["reduction_interval"]),
    )
    log.info("wrote %s", paths["attribution"])


if __name__ == "__main__":
    main()
