"""The decay chart — PR-AUC window by window across the labelled days.

`docs/monitoring.md` §4. Every score is read from disk: the shipped model's `VAL-FIT`
and `VAL-CAL` vectors, and the test vector the single touch persisted. No model is
loaded and nothing is re-scored, so this measures what the records already say, and
the pooled figures are checked against those records before a window is drawn.

**`$(HEADLINE)` is not a prerequisite.** That target refuses to rerun while its record
exists, so a stage naming it could never be satisfied. The test vector is looked for
here instead, and its absence says what to run.

**Scores, not probabilities.** Platt is monotone, so ranking scores give the same
PR-AUC as the calibrated ones on test — and `VAL-CAL`'s probabilities are out-of-fold,
four fits rather than one transform, so they are not comparable across slices at all.

**Each window is read by the registered rules, in code.** Against the `VAL-CAL`
baseline with a day-level bootstrap bar; as prevalence when ROC-AUC does not fall
with PR-AUC; and against E5's bound on what a change of identity mix could explain.
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
from fraud_engine.evaluation.metrics import pr_auc, roc_auc
from fraud_engine.evaluation.plots import save_figure
from fraud_engine.evaluation.report import git_revision
from fraud_engine.evaluation.tracking import configure_tracking, tracked_run
from fraud_engine.models.train import LABEL, run_name
from fraud_engine.monitoring.plots import plot_decay
from fraud_engine.monitoring.windows import describe, window_index

log = logging.getLogger(__name__)

NAME = "decay"
FIGURE = "decay_pr_auc.png"
HEADLINE_PREDICTIONS = "headline_test"
STRATUM = "has_identity"
VALIDATION = ("val_fit", "val_cal")
BASELINE = "val_cal"
OPTIMISTIC = "val_fit"


def load_scores(predictions_dir: Path | str, model_name: str) -> pd.DataFrame:
    """Every labelled score after the gap, one row per transaction.

    Returns:
        `TransactionID`, `split`, `day`, `isFraud` and `score` — the raw booster score
        on every slice.

    Raises:
        FileNotFoundError: If test's vector is absent, which means test was never
            touched.
    """
    predictions_dir = Path(predictions_dir)
    headline = predictions_dir / f"{HEADLINE_PREDICTIONS}.parquet"
    if not headline.exists():
        raise FileNotFoundError(
            f"{headline} is absent: test has not been touched. `make headline` is the one "
            "touch; this stage only re-reads what it persisted."
        )

    columns = ["TransactionID", "split", "day", LABEL, "score"]
    validation = pd.read_parquet(predictions_dir / f"{model_name}.parquet", columns=columns)
    validation = validation[validation["split"].isin(VALIDATION)].astype({"split": str})
    test = (
        pd.read_parquet(headline, columns=["TransactionID", "day", LABEL, "uncalibrated"])
        .rename(columns={"uncalibrated": "score"})
        .assign(split="test")
    )
    return pd.concat([validation, test[columns]], ignore_index=True)


def attach_identity(scores: pd.DataFrame, interim_path: Path | str) -> pd.DataFrame:
    """Each row's `has_identity`, from the table the join wrote it into.

    Raises:
        ValueError: If a scored transaction has no row in the interim table.
    """
    identity = pd.read_parquet(interim_path, columns=["TransactionID", STRATUM])
    frame = scores.merge(identity, on="TransactionID", how="left", validate="one_to_one")
    if frame[STRATUM].isna().any():
        raise ValueError(f"{int(frame[STRATUM].isna().sum())} scored rows have no {STRATUM}")
    return frame.astype({STRATUM: bool})


def check_records(scores: pd.DataFrame, model_record: dict, headline_record: dict) -> None:
    """Refuse vectors whose pooled figures are not the ones the records hold.

    The chart is a view of what those runs said. If a pooled PR-AUC or ROC-AUC
    recomputed here differs at all, the vector on disk is not the one they measured.

    Raises:
        ValueError: On the first figure that differs.
    """
    recorded = {split: model_record["splits"][split] for split in VALIDATION}
    recorded["test"] = headline_record["measured"]["model"]

    for split, block in recorded.items():
        part = scores[scores["split"] == split]
        for metric, compute in (("pr_auc", pr_auc), ("roc_auc", roc_auc)):
            value = compute(part[LABEL], part["score"])
            if value != block[metric]:
                raise ValueError(
                    f"{split} {metric} is {value!r} from the vector on disk but "
                    f"{block[metric]!r} in its record"
                )


def day_bootstrap(
    frame: pd.DataFrame, resamples: int, interval: float, seed: int
) -> dict[str, tuple[float, float]]:
    """Intervals for PR-AUC and ROC-AUC, resampling the window's days whole.

    The scheme is `evaluation/arms.paired_day_bootstrap`'s — days as the unit, the same
    draws, the same central interval — but not its function, which sums a per-day
    quantity and PR-AUC is not one. Each draw repeats a day's rows as many times as the
    day was drawn and scores them with the harness's own metrics, so no second
    definition of either exists.

    Returns:
        `{"pr_auc": (low, high), "roc_auc": (low, high)}`.
    """
    days, index = np.unique(frame["day"].to_numpy(), return_inverse=True)
    rows = [np.flatnonzero(index == i) for i in range(days.size)]
    y = frame[LABEL].to_numpy()
    s = frame["score"].to_numpy()

    draws = np.random.default_rng(seed).integers(0, days.size, size=(resamples, days.size))
    values = {"pr_auc": [], "roc_auc": []}
    for draw in draws:
        taken = np.concatenate([rows[i] for i in draw])
        labels, scored = pd.Series(y[taken]), pd.Series(s[taken])
        values["pr_auc"].append(pr_auc(labels, scored))
        values["roc_auc"].append(roc_auc(labels, scored))

    tail = (1 - interval) / 2
    return {
        metric: tuple(float(q) for q in np.quantile(draws_, [tail, 1 - tail]))
        for metric, draws_ in values.items()
    }


def window_table(
    frame: pd.DataFrame, width: int, anchor: int, boundary: int, bootstrap: dict
) -> pd.DataFrame:
    """One row per window: its slice, counts, mix, both metrics and their intervals.

    Raises:
        ValueError: If a window holds more than one slice — §2 chose the width so that
            none would, and a window mixing a slice the model early-stopped against
            with one it did not is two measurements under one name.
    """
    frame = frame.assign(window=window_index(frame["day"], width, anchor))
    layout = describe(frame["day"], width, anchor).set_index("window")

    records = []
    for window, part in frame.groupby("window"):
        slices = part["split"].unique()
        if slices.size != 1:
            raise ValueError(f"window {window} straddles {sorted(slices)}")
        intervals = day_bootstrap(part, **bootstrap)
        spec = layout.loc[window]
        records.append(
            {
                "window": int(window),
                "first_day": int(spec["first_day"]),
                "last_day": int(spec["last_day"]),
                "days": int(spec["days"]),
                "partial": bool(spec["partial"]),
                "slice": str(slices[0]),
                "optimistic": slices[0] == OPTIMISTIC,
                "days_since_training": (spec["first_day"] + spec["last_day"]) / 2 - boundary,
                "rows": len(part),
                "positives": int(part[LABEL].sum()),
                "base_rate": float(part[LABEL].mean()),
                "identity_share": float(part[STRATUM].mean()),
                "pr_auc": pr_auc(part[LABEL], part["score"]),
                "pr_auc_low": intervals["pr_auc"][0],
                "pr_auc_high": intervals["pr_auc"][1],
                "roc_auc": roc_auc(part[LABEL], part["score"]),
                "roc_auc_low": intervals["roc_auc"][0],
                "roc_auc_high": intervals["roc_auc"][1],
            }
        )
    return pd.DataFrame(records)


def composition_bound(record: dict) -> dict:
    """What E5 lets a change of identity mix explain, per `monitoring.md` §5.

    Returns:
        `per_point` — PR-AUC per percentage point of identity share, the compositional
        part scaled linearly — and the `low`/`high` of the share `VAL-FIT` and `VAL-CAL`
        already span.
    """
    shares = record["identity_share"]
    shift = 100 * (shares["train"] - shares["validation"])
    spanned = [shares[split] for split in VALIDATION]
    return {
        "per_point": record["decomposition"]["compositional"] / shift,
        "low": min(spanned),
        "high": max(spanned),
    }


def read_windows(table: pd.DataFrame, baseline: dict, bound: dict) -> pd.DataFrame:
    """Each window under §4's and §5's registered reading rules.

    - Below the baseline only if the whole interval is.
    - A PR-AUC fall ROC-AUC does not share is `prevalence`; one it shares is `decline`.
    - `mix_bound` is the most a change of identity mix could explain: nothing inside
      the share validation already spans, E5's per-point figure for each point outside.
      A decline larger than it is `beyond_mix` — the model's, not the data's.
    """
    pr_below = table["pr_auc_high"] < baseline["pr_auc"]
    roc_below = table["roc_auc_high"] < baseline["roc_auc"]
    reading = np.select([~pr_below, ~roc_below], ["no decline", "prevalence"], "decline")

    outside = np.maximum(bound["low"] - table["identity_share"], 0) + np.maximum(
        table["identity_share"] - bound["high"], 0
    )
    mix_bound = 100 * outside * bound["per_point"]
    shortfall = baseline["pr_auc"] - table["pr_auc"]

    return table.assign(
        pr_auc_below_baseline=pr_below,
        roc_auc_below_baseline=roc_below,
        reading=reading,
        mix_bound=mix_bound,
        beyond_mix=(reading == "decline") & (shortfall > mix_bound),
    )


def main(config_path: Path = DEFAULT_CONFIG_PATH) -> None:
    """Check, window, read and draw the decay chart.

    Wiring only. Invoked by `make decay` as `python -m fraud_engine.monitoring.decay`.

    Args:
        config_path: Path to `config.yaml`.
    """
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    config = load_config(config_path)
    paths, splits_cfg, monitoring = config["paths"], config["splits"], config["monitoring"]
    configure_tracking(config["tracking"])

    model_name = run_name(config["model"])
    scores = load_scores(paths["predictions_dir"], model_name)
    check_records(
        scores,
        json.loads((Path(paths["metrics_dir"]) / f"{model_name}.json").read_text()),
        json.loads(Path(paths["headline"]).read_text()),
    )
    log.info("pooled figures agree with %s and the headline record", model_name)
    frame = attach_identity(scores, paths["interim"])

    anchor = splits_cfg["val_fit_start"]
    boundary = splits_cfg["val_fit_start"] - splits_cfg["gap_days"] - 1
    bootstrap = {
        "resamples": monitoring["bootstrap_resamples"],
        "interval": monitoring["interval"],
        "seed": monitoring["seed"],
    }
    base = frame[frame["split"] == BASELINE]
    baseline = {
        "split": BASELINE,
        "pr_auc": pr_auc(base[LABEL], base["score"]),
        "roc_auc": roc_auc(base[LABEL], base["score"]),
    }
    bound = composition_bound(json.loads(Path(paths["composition"]).read_text()))

    params = {"model": model_name, "width": monitoring["window_days"]["labelled"]}
    with tracked_run(NAME, params, config_path):
        table = read_windows(
            window_table(frame, monitoring["window_days"]["labelled"], anchor, boundary, bootstrap),
            baseline,
            bound,
        )
        mlflow.log_metrics(
            {f"window_{row.window}.pr_auc": row.pr_auc for row in table.itertuples()}
        )

    figure_path = save_figure(
        plot_decay(table, baseline, bound), Path(paths["figures_dir"]) / FIGURE
    )

    record = {
        "name": NAME,
        "created": datetime.now(UTC).isoformat(timespec="seconds"),
        "git_revision": git_revision(),
        "model": model_name,
        "score": "raw booster score; PR-AUC is invariant to the monotone calibrator",
        "training_boundary_day": boundary,
        "window_days": monitoring["window_days"]["labelled"],
        "bootstrap": bootstrap | {"unit": "day"},
        "baseline": baseline,
        "composition_bound": bound,
        # Not `to_json`, which rounds to ten digits: the pooled figures were just held to
        # their records exactly, and the windows are written at the same precision.
        "windows": table.to_dict(orient="records"),
    }
    record_path = Path(paths["decay"])
    record_path.write_text(json.dumps(record, indent=2) + "\n")

    for row in table.itertuples():
        log.info(
            "%-7s days %d-%d  PR-AUC %.4f [%.4f, %.4f]  ROC %.4f  base %.4f  id %.3f  %s",
            row.slice,
            row.first_day,
            row.last_day,
            row.pr_auc,
            row.pr_auc_low,
            row.pr_auc_high,
            row.roc_auc,
            row.base_rate,
            row.identity_share,
            row.reading + (" (partial)" if row.partial else ""),
        )
    for path in (record_path, figure_path):
        log.info("wrote %s", path)


if __name__ == "__main__":
    main()
