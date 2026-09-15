"""Turning the shipped booster's scores into probabilities, fitted on VAL-CAL.

The booster ranks well, but its outputs are not frequencies, and the decision
policy multiplies them by money. The method, the folds and the rule that chooses
between Platt and isotonic are registered in `docs/decision-policy.md` §1.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Mapping
from datetime import UTC, datetime
from pathlib import Path

import mlflow
import numpy as np
import pandas as pd
from scipy.special import expit
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, log_loss

from fraud_engine.data.load import DEFAULT_CONFIG_PATH, load_config
from fraud_engine.evaluation.report import git_revision
from fraud_engine.evaluation.tracking import configure_tracking, tracked_run
from fraud_engine.models.train import LABEL, run_name

log = logging.getLogger(__name__)

METHODS = ("platt", "isotonic")

SPLIT = "val_cal"

# The out-of-fold probabilities, written to predictions_dir under this name. The
# reliability diagram is drawn from that file, never recomputed.
OUT_OF_FOLD = "calibration"


def assign_folds(day: pd.Series, n_folds: int) -> pd.Series:
    """A fold id per row, from contiguous blocks of days in calendar order.

    Blocks of days, not of rows: a boundary inside a day would split the unit
    review capacity is enforced on. The days are read from the data rather than
    the config, so a day missing from the predictions is refused instead of
    silently shortening a fold.

    Args:
        day: The `day` column of the rows to be folded.
        n_folds: How many folds. At least two, since each is scored by a
            calibrator fitted on the others.

    Returns:
        Fold ids `0 … n_folds - 1`, aligned to `day`.

    Raises:
        ValueError: If `n_folds` is below two, the days are not contiguous, or
            they do not divide evenly into `n_folds`.
    """
    if n_folds < 2:
        raise ValueError(f"n_folds={n_folds}: cross-fitting needs at least two folds")

    days = np.sort(day.unique())

    if days[-1] - days[0] + 1 != len(days):
        missing = sorted(set(range(days[0], days[-1] + 1)) - set(days))
        raise ValueError(f"days are not contiguous; missing {missing}")

    if len(days) % n_folds:
        raise ValueError(
            f"{len(days)} days do not divide into {n_folds} folds; one fold would be "
            "shorter than the rest"
        )

    width = len(days) // n_folds
    fold_of_day = {d: i // width for i, d in enumerate(days)}

    return day.map(fold_of_day).astype("int64").rename("fold")


def logit(score: np.ndarray, clip: float) -> np.ndarray:
    """Log-odds of `score`, clipped away from 0 and 1 so the result is finite."""
    p = np.clip(np.asarray(score, dtype="float64"), clip, 1 - clip)
    return np.log(p) - np.log1p(-p)


def fit_platt(score: np.ndarray, y: np.ndarray, clip: float) -> tuple[float, float]:
    """Platt scaling: `p = sigmoid(a * logit(score) + b)`.

    Fitted on the logit rather than the score, because the booster's own link is
    logistic. Unregularised — `C=inf` — since the default L2 penalty would pull
    both parameters toward zero and miscalibrate by construction.

    Args:
        score: The booster's predicted probabilities.
        y: Labels, 0 or 1.
        clip: See `logit`.

    Returns:
        `(a, b)`. Two floats are the whole calibrator, so that is what ships.

    Raises:
        ValueError: If `a` is not positive. A calibrator that reverses the ranking
            is a broken score upstream, not a calibration.
    """
    z = logit(score, clip).reshape(-1, 1)
    model = LogisticRegression(C=np.inf).fit(z, np.asarray(y))

    a, b = float(model.coef_[0, 0]), float(model.intercept_[0])

    if a <= 0:
        raise ValueError(f"Platt slope a={a:.4g} would reverse the ranking")

    return a, b


def apply_platt(score: np.ndarray, a: float, b: float, clip: float) -> np.ndarray:
    """Calibrated probabilities from a fitted `(a, b)`."""
    return expit(a * logit(score, clip) + b)


def fit_isotonic(score: np.ndarray, y: np.ndarray) -> pd.DataFrame:
    """Isotonic regression: the best non-decreasing step function through the labels.

    Returned as its breakpoints rather than the fitted object, so the artifact is a
    table Phase 08 can read without scikit-learn, and `apply_isotonic` reproduces
    `predict` exactly.

    Args:
        score: The booster's predicted probabilities.
        y: Labels, 0 or 1.

    Returns:
        Columns `score` and `probability`, sorted by `score`.
    """
    model = IsotonicRegression(y_min=0.0, y_max=1.0, out_of_bounds="clip").fit(
        np.asarray(score, dtype="float64"), np.asarray(y)
    )
    return pd.DataFrame({"score": model.X_thresholds_, "probability": model.y_thresholds_})


def apply_isotonic(score: np.ndarray, table: Mapping) -> np.ndarray:
    """Calibrated probabilities from fitted breakpoints.

    Linear between breakpoints and flat beyond them, which is what `predict` does
    with `out_of_bounds="clip"`. A fold scored by a calibrator fitted on the others
    routinely reaches past their range, and the default would return NaN there.
    """
    return np.interp(np.asarray(score, dtype="float64"), table["score"], table["probability"])


def fit_calibrator(method: str, score: np.ndarray, y: np.ndarray, clip: float) -> dict:
    """A fitted calibrator as plain data: what ships, and what `apply_calibrator` reads.

    Raises:
        ValueError: If `method` is not one of `METHODS`.
    """
    if method == "platt":
        a, b = fit_platt(score, y, clip)
        return {"method": "platt", "a": a, "b": b, "clip": clip}
    if method == "isotonic":
        table = fit_isotonic(score, y)
        return {
            "method": "isotonic",
            "score": table["score"].tolist(),
            "probability": table["probability"].tolist(),
        }
    raise ValueError(f"unknown calibration method {method!r}; expected one of {METHODS}")


def apply_calibrator(calibrator: Mapping, score: np.ndarray) -> np.ndarray:
    """Calibrated probabilities from the output of `fit_calibrator`.

    Raises:
        ValueError: If the calibrator names a method this module cannot apply.
    """
    method = calibrator.get("method")
    if method == "platt":
        return apply_platt(score, calibrator["a"], calibrator["b"], calibrator["clip"])
    if method == "isotonic":
        return apply_isotonic(score, calibrator)
    raise ValueError(f"unknown calibration method {method!r}; expected one of {METHODS}")


def calibrate(
    method: str, fit_score: np.ndarray, fit_y: np.ndarray, score: np.ndarray, clip: float
) -> np.ndarray:
    """Fit `method` on one set of rows and apply it to another."""
    return apply_calibrator(fit_calibrator(method, fit_score, fit_y, clip), score)


def reliability_bins(y: np.ndarray, p: np.ndarray, bins: int) -> pd.DataFrame:
    """Rows grouped by predicted probability into quantile bins.

    Edges are quantiles of `p`, so bins hold similar counts at a base rate where
    equal-width bins would not. Assignment is by value, so tied probabilities —
    every row on one isotonic step — always share a bin; when ties make two edges
    coincide, those bins merge rather than splitting a tie by row order.

    Args:
        y: Labels, 0 or 1.
        p: Predicted probabilities.
        bins: Bins requested. Fewer come back when edges coincide.

    Returns:
        One row per non-empty bin: `rows`, `mean_probability`, `fraud_rate`.
    """
    y = np.asarray(y, dtype="float64")
    p = np.asarray(p, dtype="float64")

    edges = np.unique(np.quantile(p, np.linspace(0, 1, bins + 1)))
    last = max(len(edges) - 2, 0)
    index = np.clip(np.searchsorted(edges, p, side="right") - 1, 0, last)

    rows = np.bincount(index, minlength=last + 1)
    kept = rows > 0

    return pd.DataFrame(
        {
            "rows": rows[kept],
            "mean_probability": np.bincount(index, weights=p)[kept] / rows[kept],
            "fraud_rate": np.bincount(index, weights=y)[kept] / rows[kept],
        }
    )


def expected_calibration_error(y: np.ndarray, p: np.ndarray, bins: int) -> float:
    """Row-weighted mean gap between predicted probability and fraud rate, per bin."""
    table = reliability_bins(y, p, bins)
    gap = (table["mean_probability"] - table["fraud_rate"]).abs()
    return float((gap * table["rows"]).sum() / table["rows"].sum())


def calibration_metrics(y: np.ndarray, p: np.ndarray, bins: int) -> dict[str, float]:
    """Brier score, log-loss and ECE for one set of probabilities."""
    return {
        "brier": float(brier_score_loss(y, p)),
        "log_loss": float(log_loss(y, p, labels=[0, 1])),
        "ece": expected_calibration_error(y, p, bins),
    }


def fold_metrics(y: np.ndarray, p: np.ndarray, fold: np.ndarray, bins: int) -> pd.DataFrame:
    """One row per fold: `fold`, `rows`, `positives` and its `calibration_metrics`."""
    y, p, fold = np.asarray(y), np.asarray(p), np.asarray(fold)

    return pd.DataFrame(
        [
            {
                "fold": int(k),
                "rows": int((fold == k).sum()),
                "positives": int(y[fold == k].sum()),
                **calibration_metrics(y[fold == k], p[fold == k], bins),
            }
            for k in np.unique(fold)
        ]
    )


def cross_fit(
    score: np.ndarray, y: np.ndarray, fold: np.ndarray, method: str, clip: float, bins: int
) -> tuple[np.ndarray, pd.DataFrame]:
    """Out-of-fold probabilities: each fold scored by a calibrator fitted on the rest.

    Args:
        score: The booster's predicted probabilities.
        y: Labels, 0 or 1.
        fold: Fold id per row, from `assign_folds`.
        method: One of `METHODS`.
        clip: See `logit`.
        bins: ECE bins.

    Returns:
        `(probabilities, per_fold)`: probabilities aligned to the input, and
        `fold_metrics` of them.
    """
    score = np.asarray(score, dtype="float64")
    y = np.asarray(y)
    fold = np.asarray(fold)

    probabilities = np.full(len(score), np.nan)

    for k in np.unique(fold):
        held = fold == k
        probabilities[held] = calibrate(method, score[~held], y[~held], score[held], clip)

    return probabilities, fold_metrics(y, probabilities, fold, bins)


def select_method(per_fold: Mapping[str, pd.DataFrame]) -> str:
    """The rule registered in `decision-policy.md` §1.

    Isotonic only if its out-of-fold Brier score is lower than Platt's in every
    fold. Compared within each fold, never across folds: a fold's Brier score
    moves with its own base rate.

    Args:
        per_fold: `{method: cross_fit's per_fold}` for both methods.

    Returns:
        `"isotonic"` or `"platt"`.

    Raises:
        ValueError: If the two tables do not cover the same folds.
    """
    platt = per_fold["platt"].set_index("fold")["brier"]
    isotonic = per_fold["isotonic"].set_index("fold")["brier"]

    if not platt.index.equals(isotonic.index):
        raise ValueError(
            f"folds differ: platt {platt.index.tolist()}, isotonic {isotonic.index.tolist()}"
        )

    return "isotonic" if (isotonic < platt).all() else "platt"


def check_clip(score: np.ndarray, clip: float) -> None:
    """Refuse a clip that reaches a real score.

    The clip exists to keep logit finite at exactly 0 or 1. One that reaches real
    scores ties them, and Platt is then fitted on a flattened ranking with no error.

    Raises:
        ValueError: If any score lies outside `[clip, 1 - clip]`.
    """
    reached = int(((score < clip) | (score > 1 - clip)).sum())
    if reached:
        raise ValueError(
            f"score_clip={clip:g} reaches {reached} scores; lower it below every score "
            "the booster produces"
        )


def load_split_scores(predictions_dir: Path | str, name: str) -> pd.DataFrame:
    """One run's predictions on VAL-CAL, as `make train` wrote them.

    Raises:
        ValueError: If the run holds no VAL-CAL rows.
    """
    frame = pd.read_parquet(Path(predictions_dir) / f"{name}.parquet")
    rows = frame[frame["split"] == SPLIT].reset_index(drop=True)

    if rows.empty:
        raise ValueError(f"predictions for {name!r} hold no {SPLIT!r} rows")

    return rows


def main(config_path: Path = DEFAULT_CONFIG_PATH) -> None:
    """Compare both methods out-of-fold, apply the rule, and ship the one it picks.

    Wiring only. Invoked by `make calibrate` as `python -m fraud_engine.models.calibrate`.

    Writes the calibrator beside the model, the tracked record, and the out-of-fold
    probabilities the reliability diagram is drawn from. The shipped calibrator is
    refitted on all of VAL-CAL; every reported metric is out-of-fold.

    Args:
        config_path: Path to `config.yaml`.
    """
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    config = load_config(config_path)
    paths, cal_cfg = config["paths"], config["calibration"]
    clip, bins = cal_cfg["score_clip"], cal_cfg["ece_bins"]

    configure_tracking(config["tracking"])

    model = run_name(config["model"])
    frame = load_split_scores(paths["predictions_dir"], model)
    frame["fold"] = assign_folds(frame["day"], cal_cfg["n_folds"])

    score = frame["score"].to_numpy(dtype="float64")
    y = frame[LABEL].to_numpy()
    fold = frame["fold"].to_numpy()
    check_clip(score, clip)

    params = {"model": model, "n_folds": cal_cfg["n_folds"], "ece_bins": bins, "score_clip": clip}

    with tracked_run("calibration", params, config_path):
        probabilities = {"uncalibrated": score}
        per_fold = {"uncalibrated": fold_metrics(y, score, fold, bins)}

        for method in METHODS:
            probabilities[method], per_fold[method] = cross_fit(score, y, fold, method, clip, bins)

        selected = select_method(per_fold)
        pooled = {name: calibration_metrics(y, p, bins) for name, p in probabilities.items()}

        for name, metrics in pooled.items():
            log.info("%-12s %s", name, "  ".join(f"{k}={v:.5f}" for k, v in metrics.items()))
        log.info("selected: %s", selected)

        calibrator = {
            **fit_calibrator(selected, score, y, clip),
            "model": model,
            "fitted_on": SPLIT,
            "git_revision": git_revision(),
        }
        calibrator_path = Path(paths["calibrator"])
        calibrator_path.parent.mkdir(parents=True, exist_ok=True)
        calibrator_path.write_text(json.dumps(calibrator, indent=2) + "\n")

        mlflow.log_param("selected", selected)
        mlflow.log_metrics(
            {
                f"{name}.{metric}": value
                for name, metrics in pooled.items()
                for metric, value in metrics.items()
            }
        )
        mlflow.log_artifact(calibrator_path)

    out_of_fold = frame[["TransactionID", "day", "fold", LABEL, "score"]].assign(
        platt=probabilities["platt"], isotonic=probabilities["isotonic"]
    )
    out_of_fold_path = Path(paths["predictions_dir"]) / f"{OUT_OF_FOLD}.parquet"
    out_of_fold.to_parquet(out_of_fold_path, index=False)

    record = {
        "name": "calibration",
        "created": datetime.now(UTC).isoformat(timespec="seconds"),
        "git_revision": git_revision(),
        "model": model,
        "split": SPLIT,
        "rows": len(frame),
        "positives": int(y.sum()),
        "n_folds": cal_cfg["n_folds"],
        "ece_bins": bins,
        "score_clip": clip,
        "rule": "isotonic only if its out-of-fold Brier score beats Platt's in every fold",
        "selected": selected,
        "pooled_out_of_fold": pooled,
        "per_fold": {name: table.to_dict("records") for name, table in per_fold.items()},
    }
    record_path = Path(paths["calibration"])
    record_path.write_text(json.dumps(record, indent=2) + "\n")

    for path in (calibrator_path, out_of_fold_path, record_path):
        log.info("wrote %s", path)


if __name__ == "__main__":
    main()
