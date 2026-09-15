"""Tests for the calibration of the shipped booster.

The folds are the guard that matters: a fold that shared a day with another, or
quietly came up short, would let the cross-fit score a calibrator on rows it was
fitted on, and every out-of-fold figure would flatter it without a symptom.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from scipy.special import expit

from fraud_engine.models.calibrate import apply_platt, assign_folds, fit_platt, logit

CLIP = 1e-15


def days(first: int, last: int, rows_per_day: int = 3) -> pd.Series:
    return pd.Series(np.repeat(np.arange(first, last + 1), rows_per_day), name="day")


def test_folds_are_contiguous_blocks_in_calendar_order():
    day = days(141, 160)
    fold = assign_folds(day, 4)

    spans = day.groupby(fold).agg(["min", "max"])

    assert spans.to_dict("index") == {
        0: {"min": 141, "max": 145},
        1: {"min": 146, "max": 150},
        2: {"min": 151, "max": 155},
        3: {"min": 156, "max": 160},
    }


def test_no_day_is_split_across_folds():
    day = days(141, 160).sample(frac=1, random_state=0)
    fold = assign_folds(day, 4)

    assert fold.groupby(day).nunique().max() == 1


def test_folds_align_to_the_input_index():
    day = days(141, 160).sample(frac=1, random_state=0)

    assert assign_folds(day, 4).index.equals(day.index)


def test_days_that_do_not_divide_are_refused():
    with pytest.raises(ValueError, match="do not divide"):
        assign_folds(days(141, 160), 3)


def test_a_missing_day_is_refused():
    day = days(141, 160)

    with pytest.raises(ValueError, match=r"missing \[150\]"):
        assign_folds(day[day != 150], 4)


def test_a_single_fold_is_refused():
    with pytest.raises(ValueError, match="at least two"):
        assign_folds(days(141, 160), 1)


def test_logit_is_finite_at_zero_and_one():
    z = logit(np.array([0.0, 1.0]), CLIP)

    assert np.isfinite(z).all()


def test_the_clip_leaves_ordinary_scores_untouched():
    score = np.array([1e-13, 0.3, 1 - 1e-13])

    assert np.allclose(expit(logit(score, CLIP)), score, rtol=1e-6)


def test_platt_recovers_the_identity_on_calibrated_scores():
    rng = np.random.default_rng(0)
    score = expit(rng.normal(-3, 2, size=200_000))
    y = rng.binomial(1, score)

    a, b = fit_platt(score, y, CLIP)

    assert a == pytest.approx(1, abs=0.05)
    assert b == pytest.approx(0, abs=0.05)


def test_platt_recovers_a_known_distortion():
    """The shape it exists to fix: a model whose log-odds are too extreme."""
    rng = np.random.default_rng(0)
    true_logit = rng.normal(-3, 1, size=200_000)
    y = rng.binomial(1, expit(true_logit))
    overconfident = expit(3 * true_logit + 1)

    a, b = fit_platt(overconfident, y, CLIP)

    assert a == pytest.approx(1 / 3, abs=0.02)
    assert b == pytest.approx(-1 / 3, abs=0.05)


def test_platt_is_the_unpenalised_maximum_likelihood_fit():
    """Both score equations sit at zero, to solver tolerance. The default L2
    penalty would leave the slope's off by roughly the slope itself."""
    rng = np.random.default_rng(1)
    score = expit(rng.normal(-1, 1.5, size=500))
    y = rng.binomial(1, score)

    a, b = fit_platt(score, y, CLIP)

    z = logit(score, CLIP)
    residual = y - expit(a * z + b)

    assert abs(residual.sum()) < 1e-2
    assert abs((residual * z).sum()) < 1e-2


def test_platt_preserves_the_ranking():
    rng = np.random.default_rng(0)
    score = expit(rng.normal(-3, 2, size=10_000))
    y = rng.binomial(1, score)

    calibrated = apply_platt(score, *fit_platt(score, y, CLIP), CLIP)

    order = np.argsort(score)
    assert (np.diff(calibrated[order]) >= 0).all()


def test_a_reversed_score_is_refused():
    rng = np.random.default_rng(0)
    score = expit(rng.normal(-3, 2, size=10_000))
    y = rng.binomial(1, score)

    with pytest.raises(ValueError, match="reverse the ranking"):
        fit_platt(1 - score, y, CLIP)
