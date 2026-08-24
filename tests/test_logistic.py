"""Tests for the logistic regression baseline.

No parquet is read — CI has no data. Frames are synthetic and small.

What is tested is the leakage boundary and the transformations, not the score:
how well a linear model does on this data is a measurement, and it belongs in
reports/, not in an assertion.
"""

import numpy as np
import pandas as pd
import pytest

from fraud_engine.models.logistic import (
    CATEGORICAL_COLUMNS,
    COUNT_COLUMNS,
    DELTA_COLUMNS,
    FEATURE_COLUMNS,
    build_pipeline,
    prepare,
    select_delta_columns,
)

LOGISTIC_CFG = {"d_max_null_frac": 0.50, "max_iter": 200, "C": 1.0}


def make_frame(n=400, seed=0) -> pd.DataFrame:
    """A frame with every source column, a learnable signal, and both classes."""
    rng = np.random.default_rng(seed)
    frame = pd.DataFrame(
        {
            "TransactionAmt": rng.gamma(2.0, 60.0, n) + 1.0,
            "hour": rng.integers(0, 24, n),
            "weekday": rng.integers(0, 7, n),
            "ProductCD": rng.choice(["W", "C", "H"], n),
            "card4": rng.choice(["visa", "mastercard"], n),
            "card6": rng.choice(["debit", "credit"], n),
            "P_emaildomain": rng.choice(["gmail.com", "yahoo.com"], n),
            "has_identity": rng.random(n) < 0.3,
        }
    )
    for column in COUNT_COLUMNS:
        frame[column] = rng.poisson(2.0, n).astype("float32")
    for i, column in enumerate(DELTA_COLUMNS):
        values = rng.gamma(2.0, 30.0, n).astype("float32")
        # D8 onward are mostly null in the real data; mimic the gradient so the
        # threshold rule has something to bite on.
        values[rng.random(n) < min(0.95, i / 15)] = np.nan
        frame[column] = values
    for column in (c for c in CATEGORICAL_COLUMNS if c.startswith("M")):
        frame[column] = rng.choice(["T", "F", None], n)

    # Signal the model can actually find, so a fit is meaningful.
    logit = -3.0 + 1.2 * (frame["ProductCD"] == "C") + 0.9 * (frame["C1"] > 3)
    frame["isFraud"] = (rng.random(n) < 1 / (1 + np.exp(-logit))).astype("int8")
    frame["day"] = rng.integers(1, 91, n)
    frame["TransactionID"] = np.arange(n)
    return prepare(frame)


@pytest.fixture
def frame():
    return make_frame()


# ---- the leakage boundary ----------------------------------------------------


def test_the_label_and_the_split_axis_are_not_features():
    """Stated positively, so remainder='drop' is not the only thing guarding it."""
    for column in ("isFraud", "day", "TransactionID", "split"):
        assert column not in FEATURE_COLUMNS


def test_delta_selection_sees_only_the_frame_it_is_given():
    """Choosing columns from the full table would leak validation missingness."""
    train = make_frame(seed=1)
    train["D3"] = np.nan  # unusable in train...
    later = make_frame(seed=2)  # ...but fine later; must not resurrect it

    assert "D3" not in select_delta_columns(train, 0.50)
    assert "D3" in select_delta_columns(later, 0.50)


def test_delta_selection_respects_the_threshold(frame):
    kept = select_delta_columns(frame, 0.50)
    assert all(frame[column].isna().mean() < 0.50 for column in kept)
    assert all(column in DELTA_COLUMNS for column in kept)


# ---- transformations ---------------------------------------------------------


def test_prepare_makes_missingness_an_explicit_level(frame):
    """Missingness is signal here; it must not become NaN for the encoder."""
    for column in CATEGORICAL_COLUMNS:
        assert not frame[column].isna().any()
    assert (
        (frame[[c for c in CATEGORICAL_COLUMNS if c.startswith("M")]] == "__missing__").any().any()
    )


def test_hour_wraps_around_midnight(frame):
    """Raw hour puts 23 and 0 maximally apart; the cyclic encoding must not."""
    pipeline = build_pipeline(["D1"], LOGISTIC_CFG, None)
    pipeline.fit(frame[list(FEATURE_COLUMNS)], frame["isFraud"])
    encoder = pipeline.named_steps["preprocess"].named_transformers_["hour"]

    hours = pd.DataFrame({"hour": [23, 0, 12]})
    sin, cos = encoder.transform(hours)[:, 0], encoder.transform(hours)[:, 1]
    near = np.hypot(sin[0] - sin[1], cos[0] - cos[1])
    far = np.hypot(sin[0] - sin[2], cos[0] - cos[2])
    assert near < far


def test_the_cyclic_features_are_named(frame):
    """A linear baseline whose coefficients cannot be read back is pointless."""
    pipeline = build_pipeline(["D1"], LOGISTIC_CFG, None)
    pipeline.fit(frame[list(FEATURE_COLUMNS)], frame["isFraud"])
    names = list(pipeline.named_steps["preprocess"].get_feature_names_out())
    assert "hour_sin" in names and "hour_cos" in names


# ---- the pipeline ------------------------------------------------------------


def test_a_category_unseen_in_train_does_not_raise(frame):
    """The temporal-split gotcha: days 121+ carry values days 1-90 never had."""
    pipeline = build_pipeline(["D1"], LOGISTIC_CFG, None)
    pipeline.fit(frame[list(FEATURE_COLUMNS)], frame["isFraud"])

    later = frame.head(5).copy()
    later["P_emaildomain"] = "a-domain-that-did-not-exist.com"
    later["card4"] = "an-unseen-network"
    assert pipeline.predict_proba(later[list(FEATURE_COLUMNS)])[:, 1].shape == (5,)


def test_scores_are_probabilities(frame):
    pipeline = build_pipeline(["D1"], LOGISTIC_CFG, None)
    pipeline.fit(frame[list(FEATURE_COLUMNS)], frame["isFraud"])
    scores = pipeline.predict_proba(frame[list(FEATURE_COLUMNS)])[:, 1]
    assert ((scores >= 0) & (scores <= 1)).all()


@pytest.mark.parametrize("class_weight", [None, "balanced"])
def test_both_e2_variants_fit_and_score(frame, class_weight):
    """E2 requires the result either way, so neither variant may be unrunnable."""
    pipeline = build_pipeline(["D1"], LOGISTIC_CFG, class_weight)
    pipeline.fit(frame[list(FEATURE_COLUMNS)], frame["isFraud"])
    assert pipeline.predict_proba(frame[list(FEATURE_COLUMNS)])[:, 1].std() > 0


def test_scoring_one_row_matches_scoring_it_in_a_batch(frame):
    """The serving case: no transformation may depend on the rest of the batch."""
    features = frame[list(FEATURE_COLUMNS)]
    pipeline = build_pipeline(["D1"], LOGISTIC_CFG, None)
    pipeline.fit(features, frame["isFraud"])

    batch = pipeline.predict_proba(features.head(3))[:, 1]
    alone = [pipeline.predict_proba(features.iloc[[i]])[:, 1][0] for i in range(3)]
    assert batch == pytest.approx(alone)
