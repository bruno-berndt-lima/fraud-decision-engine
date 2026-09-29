"""The retraining trigger, replayed against what this phase measured.

`docs/monitoring.md` §7. Reads records and fits nothing: drift's windows for the two
label-free conditions, the headline's record for the baseline the labelled one is judged
against, and the split layout for the clock. It reports which condition fires first, on
which day, and how stale a model retrained on that day would be.

**Condition 3 is carried and never fires here.** Test's windows are inside the baseline
they would be judged against, and the horizon has no labels. The record says, per window,
when its labels would have matured: how late the labelled condition would arrive is the
finding, and those dates show it.
"""

from __future__ import annotations

import json
import logging
from datetime import UTC, datetime
from pathlib import Path

from fraud_engine.data.load import DEFAULT_CONFIG_PATH, load_config
from fraud_engine.evaluation.report import git_revision

log = logging.getLogger(__name__)

NAME = "retraining_trigger"
BANDS = ("moderate", "significant")
CONDITIONS = {1: "feature_psi", 2: "score_psi", 3: "pr_auc", 4: "cadence"}


def thresholds(trigger_cfg: dict, psi_cfg: dict) -> dict[str, float]:
    """The PSI conditions' values, resolved from the convention's names.

    Raises:
        ValueError: If a condition names anything but one of the convention's bands.
    """
    resolved = {}
    for key in ("feature_psi", "score_psi"):
        name = trigger_cfg[key]
        if name not in BANDS:
            raise ValueError(f"trigger.{key} is {name!r}; it must name a band, one of {BANDS}")
        resolved[key] = psi_cfg[name]
    return resolved


def check_drift_record(drift: dict, cfg: dict) -> None:
    """Refuse a drift record measured under other settings than the rule reads.

    Make restages drift when `monitoring` changes, so this only catches a record written
    by hand or copied in, and it costs nothing.

    Raises:
        ValueError: If the weighted aggregate covers a different `top_k`, or the bands
            it was read against differ.
    """
    weighed, top_k = len(drift["weighted_columns"]), cfg["trigger"]["top_k"]
    if weighed != top_k:
        raise ValueError(f"drift.json weighs {weighed} columns and trigger.top_k is {top_k}")
    for name in BANDS:
        if drift["psi"][name] != cfg["psi"][name]:
            raise ValueError(f"drift.json read PSI against {name}={drift['psi'][name]}")


def clock(splits_cfg: dict, cadence_days: int) -> dict:
    """The replay's days: deployment, the cadence, and when each set of labels matures."""
    maturity, deployment = splits_cfg["gap_days"], splits_cfg["test_start"]
    return {
        "training_ends": splits_cfg["val_fit_start"] - maturity - 1,
        "deployment": deployment,
        "maturity_days": maturity,
        "calibration_labels_mature_on": deployment - 1 + maturity,
        "baseline_labels_mature_on": splits_cfg["test_end"] + maturity,
        "cadence_fires_on": deployment + cadence_days,
    }


def read_windows(drift: dict, limits: dict[str, float], days: dict) -> list[dict]:
    """Each window after deployment, with what every condition makes of it.

    Raises:
        ValueError: If a window straddles deployment, so it is neither before nor after.
    """
    entries = []
    for horizon in ("labelled", "horizon"):
        for window in drift[horizon]:
            first, last = window["first_day"], window["last_day"]
            if first < days["deployment"] <= last:
                raise ValueError(f"window {first}-{last} straddles deployment")
            if first < days["deployment"]:
                continue

            partial = window["partial"]
            entry = {
                "horizon": horizon,
                "first_day": first,
                "last_day": last,
                "partial": partial,
                "read_on": last,
            }
            values = {"feature_psi": window["weighted_psi"]}
            if horizon == "horizon":
                values["score_psi"] = window["prediction"]["score_psi"]
            for key, value in values.items():
                reaches = value >= limits[key]
                entry[key] = {"value": value, "reaches": reaches, "fires": reaches and not partial}

            entry["pr_auc"] = {
                "fires": False,
                "labels_mature_on": last + days["maturity_days"],
                "why_not": "inside the baseline" if horizon == "labelled" else "no labels",
            }
            entries.append(entry)
    return entries


def firings(windows: list[dict], days: dict) -> dict[int, int | None]:
    """The first day each condition fires, or None if it never does."""
    result = {
        number: min(
            (w["read_on"] for w in windows if w.get(CONDITIONS[number], {}).get("fires")),
            default=None,
        )
        for number in (1, 2, 3)
    }
    return result | {4: days["cadence_fires_on"]}


def first(fired: dict[int, int | None], windows: list[dict]) -> dict:
    """The condition that fires first, ties to the lower number, and the window it lands in."""
    day, number = min((day, number) for number, day in fired.items() if day is not None)
    inside = next(
        (
            {key: w[key] for key in ("horizon", "first_day", "last_day", "partial")}
            for w in windows
            if w["first_day"] <= day <= w["last_day"]
        ),
        None,
    )
    return {"condition": number, "name": CONDITIONS[number], "day": day, "window": inside}


def retrain(day: int, splits_cfg: dict) -> dict:
    """How far back a model retrained on `day` can see.

    Labels through `day - maturity` at best. Under the project's own layout, `VAL-CAL`,
    `VAL-FIT` and the purge must also fit before that day, at the lengths `splits` gives.
    """
    labelled_through = day - splits_cfg["gap_days"]
    val_cal_days = splits_cfg["test_start"] - splits_cfg["val_cal_start"]
    val_fit_days = splits_cfg["val_cal_start"] - splits_cfg["val_fit_start"]

    val_cal_start = labelled_through - val_cal_days + 1
    val_fit_start = val_cal_start - val_fit_days
    training_ends = val_fit_start - splits_cfg["gap_days"] - 1
    return {
        "trigger_day": day,
        "labelled_through": labelled_through,
        "under_the_layout": {
            "training_ends": training_ends,
            "val_fit": [val_fit_start, val_cal_start - 1],
            "val_cal": [val_cal_start, labelled_through],
            "days_stale_on_the_trigger_day": day - training_ends,
        },
    }


def main(config_path: Path = DEFAULT_CONFIG_PATH) -> None:
    """Replay the registered rule against drift's windows and write the record.

    Wiring only. Invoked by `make trigger` as `python -m fraud_engine.monitoring.trigger`.

    Args:
        config_path: Path to `config.yaml`.
    """
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    config = load_config(config_path)
    paths, cfg, splits_cfg = config["paths"], config["monitoring"], config["splits"]

    headline_path = Path(paths["headline"])
    if not headline_path.exists():
        raise FileNotFoundError(f"{headline_path} is absent; `make headline` writes it")
    baseline = json.loads(headline_path.read_text())["measured"]["model"]

    drift = json.loads(Path(paths["drift"]).read_text())
    check_drift_record(drift, cfg)

    trigger_cfg = cfg["trigger"]
    limits = thresholds(trigger_cfg, cfg["psi"])
    days = clock(splits_cfg, trigger_cfg["cadence_days"])
    windows = read_windows(drift, limits, days)
    fired = firings(windows, days)
    earliest = first(fired, windows)

    record = {
        "name": NAME,
        "created": datetime.now(UTC).isoformat(timespec="seconds"),
        "git_revision": git_revision(),
        "model": drift["model"],
        "drift_revision": drift["git_revision"],
        "rule": {
            "top_k": trigger_cfg["top_k"],
            "feature_psi": {"band": trigger_cfg["feature_psi"], "value": limits["feature_psi"]},
            "score_psi": {"band": trigger_cfg["score_psi"], "value": limits["score_psi"]},
            "pr_auc": {
                "drop": trigger_cfg["pr_auc_drop"],
                "baseline": {
                    "split": "test",
                    "pr_auc": baseline["pr_auc"],
                    "roc_auc": baseline["roc_auc"],
                },
                "threshold": baseline["pr_auc"] * (1 - trigger_cfg["pr_auc_drop"]),
            },
            "cadence_days": trigger_cfg["cadence_days"],
        },
        "clock": days,
        "windows": windows,
        "firings": {CONDITIONS[number]: day for number, day in fired.items()},
        "first": earliest,
        "retrain": retrain(earliest["day"], splits_cfg),
    }
    Path(paths["retraining_trigger"]).write_text(json.dumps(record, indent=2) + "\n")

    for w in windows:
        score = w.get("score_psi")
        log.info(
            "%-8s days %d-%d  weighted PSI %.3f%s%s%s",
            w["horizon"],
            w["first_day"],
            w["last_day"],
            w["feature_psi"]["value"],
            " (reaches)" if w["feature_psi"]["reaches"] else "",
            f"  score PSI {score['value']:.3f}" if score else "",
            " (partial)" if w["partial"] else "",
        )
    log.info(
        "first to fire: condition %d (%s) on day %d; retrained then, the model trains through "
        "day %d at best",
        earliest["condition"],
        earliest["name"],
        earliest["day"],
        record["retrain"]["labelled_through"],
    )
    log.info("wrote %s", paths["retraining_trigger"])


if __name__ == "__main__":
    main()
