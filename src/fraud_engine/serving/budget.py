"""Bounding what a caller waits for, when the work itself cannot be interrupted.

`docs/serving.md` §4. The service holds itself to the p95 `problem-statement.md` §3.1
sets, and past it the transaction takes the rules decision rather than waiting.

**A prediction cannot be cancelled.** LightGBM's `predict` is a blocking call into C; there
is no interrupt for it, and Python cannot stop a thread from the outside. So a breach
returns the fallback's answer on time while the scoring work finishes in its own thread
and is thrown away. The request is bounded; the worker is not. That is the honest shape of
the guarantee, and the alternative — a timeout that quietly waits for the model anyway —
would meet the budget on paper only.

**The wait being bounded includes queueing.** Under load, part of what a request waits for
is a worker to run on, and that wait counts against the same budget. It should: a caller
does not care which half of the delay was scoring. It is said here so nobody reads a
breach as proof that the model itself was slow.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from concurrent.futures import Future, ThreadPoolExecutor
from concurrent.futures import TimeoutError as FutureTimeout

log = logging.getLogger(__name__)

# Named so a thread dump during an incident says which pool a stuck worker belongs to.
THREAD_PREFIX = "score"


def pool() -> ThreadPoolExecutor:
    """The workers a bounded decision runs on.

    Left at the default size deliberately. A pool sized to the number of requests in
    flight would never queue, and one sized smaller turns saturation into breaches —
    which is a real thing to know about a deployment, and the §7 load test is where it
    becomes a number rather than a worry.
    """
    return ThreadPoolExecutor(thread_name_prefix=THREAD_PREFIX)


def within[T](workers: ThreadPoolExecutor, budget_ms: float, call: Callable[[], T]) -> T:
    """`call`'s result, or `TimeoutError` if it did not arrive inside the budget.

    Args:
        workers: From `pool`.
        budget_ms: How long the caller may be made to wait, in milliseconds.
        call: The work, already bound to its arguments.

    Returns:
        Whatever `call` returned.

    Raises:
        TimeoutError: If the budget passed first. The work is **not** cancelled — see the
            module docstring — and whatever it eventually produces is discarded.
        Exception: Whatever `call` raised, re-raised here.
    """
    running: Future[T] = workers.submit(call)

    try:
        return running.result(timeout=budget_ms / 1000)
    except FutureTimeout as exceeded:
        raise TimeoutError(
            f"the decision did not arrive within {budget_ms:g} ms; the work continues in "
            "its own thread and its result is discarded"
        ) from exceeded
