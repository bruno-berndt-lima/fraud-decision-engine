"""How a timing distribution becomes a latency table.

One definition because two records are meant to be read against each other:
`explainability.md` §8 timed a prediction against its explanation to decide where the
explanation lives, and `serving.md` §7 times the service that decision shaped. A ratio
between them means nothing if one took its percentiles by linear interpolation and the
other by nearest rank.
"""

from __future__ import annotations

import numpy as np

# Reported for every timed operation. A p99 rests on far fewer observations than a p50
# and is recorded as the tail it is, not as a number to plan against.
PERCENTILES = (50, 95, 99)


def percentiles(samples: np.ndarray) -> dict[str, float]:
    """A timing distribution as the milliseconds a latency table is written from.

    Args:
        samples: Per-call durations, in milliseconds.

    Returns:
        `p50`, `p95`, `p99` and `mean`, in milliseconds.
    """
    measured = {f"p{point}": float(np.percentile(samples, point)) for point in PERCENTILES}
    return measured | {"mean": float(samples.mean())}
