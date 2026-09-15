"""Turning the shipped booster's scores into probabilities, fitted on VAL-CAL.

The booster ranks well, but its outputs are not frequencies, and the decision
policy multiplies them by money. The method, the folds and the rule that chooses
between Platt and isotonic are registered in `docs/decision-policy.md` §1.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.special import expit
from sklearn.linear_model import LogisticRegression


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
