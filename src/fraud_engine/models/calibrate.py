"""Turning the shipped booster's scores into probabilities, fitted on VAL-CAL.

The booster ranks well, but its outputs are not frequencies, and the decision
policy multiplies them by money. The method, the folds and the rule that chooses
between Platt and isotonic are registered in `docs/decision-policy.md` §1.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.special import expit
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, log_loss

METHODS = ("platt", "isotonic")


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


def apply_isotonic(score: np.ndarray, table: pd.DataFrame) -> np.ndarray:
    """Calibrated probabilities from fitted breakpoints.

    Linear between breakpoints and flat beyond them, which is what `predict` does
    with `out_of_bounds="clip"`. A fold scored by a calibrator fitted on the others
    routinely reaches past their range, and the default would return NaN there.
    """
    return np.interp(np.asarray(score, dtype="float64"), table["score"], table["probability"])


def calibrate(
    method: str, fit_score: np.ndarray, fit_y: np.ndarray, score: np.ndarray, clip: float
) -> np.ndarray:
    """Fit `method` on one set of rows and apply it to another.

    Raises:
        ValueError: If `method` is not one of `METHODS`.
    """
    if method == "platt":
        return apply_platt(score, *fit_platt(fit_score, fit_y, clip), clip)
    if method == "isotonic":
        return apply_isotonic(score, fit_isotonic(fit_score, fit_y))
    raise ValueError(f"unknown calibration method {method!r}; expected one of {METHODS}")


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
        `(probabilities, per_fold)`. `probabilities` is aligned to the input;
        `per_fold` has one row per fold with `fold`, `rows`, `positives` and
        the `calibration_metrics` of that fold's out-of-fold probabilities.
    """
    score = np.asarray(score, dtype="float64")
    y = np.asarray(y)
    fold = np.asarray(fold)

    probabilities = np.full(len(score), np.nan)
    per_fold = []

    for k in np.unique(fold):
        held = fold == k
        probabilities[held] = calibrate(method, score[~held], y[~held], score[held], clip)
        per_fold.append(
            {
                "fold": int(k),
                "rows": int(held.sum()),
                "positives": int(y[held].sum()),
                **calibration_metrics(y[held], probabilities[held], bins),
            }
        )

    return probabilities, pd.DataFrame(per_fold)
