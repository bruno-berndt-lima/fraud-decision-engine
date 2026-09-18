"""Tests for bounding what a caller waits for.

The guarantee is narrow and the tests are about its edges: work that finishes in time is
returned untouched, work that does not raises rather than blocking, and the work itself is
never cancelled — because it cannot be. `docs/serving.md` §4.
"""

import time

import pytest

from fraud_engine.serving.budget import pool, within


@pytest.fixture
def workers():
    with pool() as running:
        yield running


def test_work_that_finishes_in_time_is_returned(workers):
    assert within(workers, 500, lambda: 7) == 7


def test_the_caller_is_released_at_the_budget(workers):
    began = time.perf_counter()

    with pytest.raises(TimeoutError):
        within(workers, 50, lambda: time.sleep(5))

    assert (time.perf_counter() - began) < 1.0, "the caller waited for the work anyway"


def test_the_work_is_not_cancelled_when_the_budget_passes(workers):
    """A blocking call into C cannot be interrupted, so the honest contract is that it runs on.

    Stated as a test rather than only in a docstring: someone reading "timeout" will assume
    the work stopped, and it did not.
    """
    finished = []

    with pytest.raises(TimeoutError):
        within(workers, 20, lambda: (time.sleep(0.2), finished.append("ran"))[1])

    time.sleep(0.4)
    assert finished == ["ran"]


def test_what_the_work_raised_reaches_the_caller(workers):
    """A failure inside the budget is a failure, not a timeout — they are handled apart."""
    with pytest.raises(ValueError, match="inside"):
        within(workers, 500, lambda: (_ for _ in ()).throw(ValueError("inside")))
