"""Tests for the test touch.

The guards are what matter here, because this stage runs once. A second touch, a
touch from uncommitted code, or a model that is not the recorded one would each
produce a headline that looks exactly like the right one.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from fraud_engine.evaluation.cost import ALLOW
from fraud_engine.evaluation.headline import (
    ALLOW_EVERYTHING,
    build_test_frame,
    check_single_touch,
    headline_rows,
)
from fraud_engine.evaluation.reproduce import check_reproduces
from test_policy import COSTS

CALIBRATOR = {"method": "platt", "a": 0.3, "b": 0.75, "clip": 1e-15}


# ---- the single touch --------------------------------------------------------------


def test_a_clean_first_touch_passes(tmp_path):
    check_single_touch(tmp_path / "policy_test.json", "abc123")


def test_a_dirty_tree_is_refused(tmp_path):
    with pytest.raises(RuntimeError, match="dirty"):
        check_single_touch(tmp_path / "policy_test.json", "abc123-dirty")


def test_a_touch_outside_git_is_refused(tmp_path):
    with pytest.raises(RuntimeError, match="git checkout"):
        check_single_touch(tmp_path / "policy_test.json", None)


def test_a_second_touch_is_refused(tmp_path):
    record = tmp_path / "policy_test.json"
    record.write_text("{}")

    with pytest.raises(RuntimeError, match="test has been touched"):
        check_single_touch(record, "abc123")


# ---- the reproduction proof ----------------------------------------------------------


def scores(split: str, ids, values) -> pd.DataFrame:
    return pd.DataFrame({"TransactionID": list(ids), "split": split, "score": values})


def test_the_proof_reads_the_split_it_is_given():
    values = np.linspace(0, 1, 10)
    scored = pd.concat(
        [scores("val_cal", range(10), values), scores("test", range(10, 20), values)]
    )

    check_reproduces(scored, scores("val_cal", range(10), values), "model", "val_cal")


def test_a_model_that_differs_on_the_proof_split_is_refused():
    values = np.linspace(0, 1, 10)
    changed = values.copy()
    changed[0] += 1e-12

    with pytest.raises(ValueError, match="1 VAL-CAL scores differ"):
        check_reproduces(
            scores("val_cal", range(10), changed),
            scores("val_cal", range(10), values),
            "model",
            "val_cal",
        )


def test_a_record_without_the_proof_split_is_refused():
    """Nothing to compare against must not pass as nothing differing."""
    values = np.linspace(0, 1, 10)

    with pytest.raises(ValueError, match="different VAL-CAL rows"):
        check_reproduces(
            scores("val_cal", range(10), values),
            scores("val_fit", range(10), values),
            "model",
            "val_cal",
        )


# ---- the test frame --------------------------------------------------------------------


@pytest.fixture
def scored() -> tuple[pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(0)
    n = 2_200
    ids = np.arange(n)
    day = np.repeat(np.arange(161, 183), n // 22)
    raw = np.clip(rng.beta(0.2, 20, size=n), 1e-9, 1 - 1e-9)
    y = rng.binomial(1, np.clip(raw * 5, 0, 1))

    model = pd.concat(
        [
            pd.DataFrame(
                {"TransactionID": ids, "day": day, "isFraud": y, "score": raw, "split": "test"}
            ),
            pd.DataFrame(
                {
                    "TransactionID": ids + n,
                    "day": 150,
                    "isFraud": 0,
                    "score": 0.1,
                    "split": "val_cal",
                }
            ),
        ]
    )
    engine = pd.DataFrame(
        {
            "TransactionID": ids[::-1],
            "day": day[::-1],
            "isFraud": y[::-1],
            "score": rng.integers(0, 6, size=n) + rng.uniform(0, 0.5, size=n),
            "TransactionAmt": rng.lognormal(4, 1.2, size=n),
            "split": "test",
        }
    )
    return model, engine


def test_the_frame_holds_test_rows_only_with_the_calibrator_applied(scored):
    model, engine = scored

    frame = build_test_frame(model, engine, CALIBRATOR).set_index("TransactionID")
    test_model = model[model["split"] == "test"].set_index("TransactionID")
    by_id = engine.set_index("TransactionID")

    assert len(frame) == len(test_model)
    assert np.array_equal(frame["uncalibrated"], test_model.loc[frame.index, "score"])
    assert np.array_equal(frame["rules_score"], by_id.loc[frame.index, "score"])
    assert np.array_equal(frame["amount"], by_id.loc[frame.index, "TransactionAmt"])

    z = np.log(frame["uncalibrated"]) - np.log1p(-frame["uncalibrated"])
    assert np.allclose(frame["calibrated"], 1 / (1 + np.exp(-(0.3 * z + 0.75))))


def test_rules_over_different_test_rows_are_refused(scored):
    model, engine = scored

    with pytest.raises(ValueError, match="different test rows"):
        build_test_frame(model, engine.iloc[1:], CALIBRATOR)


def test_rules_that_disagree_on_a_test_label_are_refused(scored):
    model, engine = scored
    engine = engine.copy()
    engine.loc[0, "isFraud"] = 1 - engine.loc[0, "isFraud"]

    with pytest.raises(ValueError, match="disagree on test labels"):
        build_test_frame(model, engine, CALIBRATOR)


def test_allowing_everything_sits_beside_the_four_rows(scored):
    model, engine = scored
    frame = build_test_frame(model, engine, CALIBRATOR)

    policies, references = headline_rows(frame, COSTS, 0.01)

    assert list(policies) == ["rules", "naive", "ev_uncalibrated", "ev"]
    allow = references[ALLOW_EVERYTHING]
    assert allow["block_rate"] == 0 and allow["reviews_per_day"] == 0

    fraud = frame["isFraud"] == 1
    expected = (frame.loc[fraud, "amount"] + COSTS.chargeback_fee).sum() / len(frame) * 1_000
    assert allow["usd_per_1000"] == pytest.approx(expected)
    assert allow["reduction_vs_rules"] == pytest.approx(
        1 - expected / policies["rules"]["usd_per_1000"]
    )
    assert ALLOW not in policies
