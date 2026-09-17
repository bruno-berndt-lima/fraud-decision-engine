"""Tests for the timing harness, never for the timings.

A duration is a property of the machine that ran it, so nothing here asserts one.
What is asserted is that the warmup is discarded rather than averaged in, that both
operations are timed on the same row, and that the record carries the machine — the
three ways a latency figure stops meaning anything.
"""

import numpy as np
import pytest

from fraud_engine.explain import latency

# ------------------------------------------------------------------------------
# percentiles
# ------------------------------------------------------------------------------


def test_the_points_a_latency_table_is_written_from():
    measured = latency.percentiles(np.arange(1.0, 101.0))

    assert set(measured) == {"p50", "p95", "p99", "mean"}
    assert measured["mean"] == pytest.approx(50.5)
    assert measured["p50"] < measured["p95"] < measured["p99"]


# ------------------------------------------------------------------------------
# time_call
# ------------------------------------------------------------------------------


def test_the_warmup_is_made_and_thrown_away():
    """A cold first request is not a latency figure, so it is not one of the samples."""
    calls = []

    latency.time_call(lambda: calls.append(1), rows=5, warmup=3)

    assert len(calls) == 8


def test_only_the_timed_calls_reach_the_distribution(monkeypatch):
    ticks = iter(range(1000))
    monkeypatch.setattr(latency.time, "perf_counter", lambda: next(ticks) / 1000)

    measured = latency.time_call(lambda: None, rows=4, warmup=0)

    assert measured["mean"] == pytest.approx(1.0)


# ------------------------------------------------------------------------------
# measure
# ------------------------------------------------------------------------------


class FakeBooster:
    """Records what it was asked for, so the harness can be checked without a model."""

    best_iteration = -1

    def __init__(self) -> None:
        self.calls: list[dict] = []

    def predict(self, data, **kwargs):
        self.calls.append({"rows": len(data), **kwargs})
        return np.zeros(len(data))


def test_scoring_and_explaining_are_timed_on_the_same_row():
    booster = FakeBooster()
    row = np.zeros((1, 3))

    measured = latency.measure(booster, row, threads=2, latency_cfg={"rows": 2, "warmup": 1})

    assert set(measured) == {"score", "explain"}
    assert {call["rows"] for call in booster.calls} == {1}


def test_the_thread_count_reaches_lightgbm():
    """The figure means nothing without it, so it is passed rather than defaulted."""
    booster = FakeBooster()

    latency.measure(booster, np.zeros((1, 3)), threads=2, latency_cfg={"rows": 1, "warmup": 0})

    assert all(call["num_threads"] == 2 for call in booster.calls)


def test_only_the_explaining_calls_ask_for_contributions():
    booster = FakeBooster()

    latency.measure(booster, np.zeros((1, 3)), threads=1, latency_cfg={"rows": 1, "warmup": 0})

    asked = [call.get("pred_contrib", False) for call in booster.calls]
    assert asked == [False, True]


# ------------------------------------------------------------------------------
# machine
# ------------------------------------------------------------------------------


def test_the_record_says_what_the_numbers_are_true_of():
    described = latency.machine()

    assert {"platform", "processor", "cpu_count", "python", "lightgbm"} <= set(described)
    assert described["cpu_count"]
