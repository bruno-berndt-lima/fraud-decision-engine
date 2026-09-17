"""What an explanation costs to serve, beside what the score costs.

`docs/explainability.md` §8, which is not in the roadmap's Definition of Done and was
added deliberately: Phase 08 has a latency budget, the shipped booster is large, and
exact TreeSHAP walks every tree twice over. Finding that out in Phase 08, with the API
already written, would make it a surprise instead of a design input.

**This is not the Phase 08 load test.** `problem-statement.md` §3.1 asks for p95 under
sustained concurrent load, which needs the service that does not exist yet. What is
measured here is narrower and answerable now: the cost of one `predict` call against
the cost of the same call asked for contributions, one row at a time, warm.

**Threads are stated, not inherited.** LightGBM spreads a single prediction across every
core it can see. A figure from a development machine's core count describes a deployment
nobody would provision, so the measurement is repeated at thread counts a container
plausibly gets and the record carries both.

**What the explanation adds is the answer.** Scoring has to happen regardless; what
Phase 08 is deciding is whether the explanation can ride along on the same request. The
two are timed separately and the record reports the difference of their p95s, which is
not the p95 of their difference — a distinction that costs nothing to state and would
cost a redesign to remove.
"""

from __future__ import annotations

import json
import logging
import os
import platform
import time
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path

import lightgbm as lgb
import mlflow
import numpy as np
import pandas as pd

from fraud_engine.data.load import DEFAULT_CONFIG_PATH, load_config
from fraud_engine.evaluation.report import git_revision
from fraud_engine.evaluation.tracking import configure_tracking, tracked_run
from fraud_engine.models.train import feature_columns, prepare_matrices, run_name

log = logging.getLogger(__name__)

NAME = "explain_latency"

# Reported for every timed call. p99 over a hundred samples rests on one observation and
# is recorded as the tail it is, not as a number to plan against.
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


def time_call(call: Callable[[], object], rows: int, warmup: int) -> dict[str, float]:
    """Time one operation repeatedly, discarding the calls that pay for the cache.

    A cold first request is not a latency figure — `problem-statement.md` §3.1 says so
    in as many words — so the warmup calls are made and thrown away rather than
    averaged in and apologised for.

    Args:
        call: The operation, already bound to its arguments.
        rows: Timed repetitions.
        warmup: Repetitions made first and discarded.

    Returns:
        As `percentiles`.
    """
    for _ in range(warmup):
        call()

    samples = np.empty(rows, dtype="float64")
    for index in range(rows):
        started = time.perf_counter()
        call()
        samples[index] = (time.perf_counter() - started) * 1000

    return percentiles(samples)


def measure(
    booster: lgb.Booster, row: pd.DataFrame, threads: int, latency_cfg: dict
) -> dict[str, dict[str, float]]:
    """One row scored, then explained, at a fixed thread count.

    The same row for both, so the difference is the contributions and not the traffic.

    Args:
        booster: The shipped booster, reloaded.
        row: A single prepared transaction, columns in the booster's order.
        threads: What LightGBM is allowed to use.
        latency_cfg: The `explain_latency:` config block.

    Returns:
        `{"score": ..., "explain": ...}`, each as `percentiles`.
    """
    rows, warmup = latency_cfg["rows"], latency_cfg["warmup"]
    shared = {"num_iteration": booster.best_iteration, "num_threads": threads}

    return {
        "score": time_call(lambda: booster.predict(row, **shared), rows, warmup),
        "explain": time_call(
            lambda: booster.predict(row, pred_contrib=True, **shared), rows, warmup
        ),
    }


def machine() -> dict[str, object]:
    """What the numbers are true of.

    A latency figure without a machine attached is meaningless — `problem-statement.md`
    §3.1 — and this is the one record in the project whose numbers do not travel.
    """
    return {
        "platform": platform.platform(),
        "processor": platform.processor() or platform.machine(),
        "cpu_count": os.cpu_count(),
        "python": platform.python_version(),
        "lightgbm": lgb.__version__,
    }


def main(config_path: Path = DEFAULT_CONFIG_PATH) -> None:
    """Time scoring against explaining, and record both with the machine they ran on.

    Wiring only. Invoked by `make latency` as `python -m fraud_engine.explain.latency`.

    Args:
        config_path: Path to `config.yaml`.
    """
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    config = load_config(config_path)
    paths, model_cfg, latency_cfg = config["paths"], config["model"], config["explain_latency"]

    configure_tracking(config["tracking"])

    split = latency_cfg["split"]
    matrices, _, _ = prepare_matrices(paths["features_dir"], model_cfg, ("train", split))
    columns = feature_columns(matrices[split])

    booster = lgb.Booster(model_file=paths["model"])
    if booster.feature_name() != columns:
        raise ValueError("the booster's features are not the matrix's features, in order")

    row = matrices[split][columns].head(1)
    model_name = run_name(model_cfg)

    params = {
        "model": model_name,
        "split": split,
        "rows": latency_cfg["rows"],
        "threads": ",".join(str(count) for count in latency_cfg["threads"]),
    }

    by_threads: dict[str, dict] = {}
    with tracked_run(NAME, params, config_path):
        for threads in latency_cfg["threads"]:
            timings = measure(booster, row, threads, latency_cfg)

            # A difference of percentiles, which is not the percentile of the
            # difference — the two calls are timed in separate loops, so there are no
            # paired samples to take a percentile of. Named for what it computes. It
            # answers the question anyway while one term is two orders larger and tight
            # around its median; a future model where they are comparable would need
            # the calls timed together and this renamed again.
            difference = timings["explain"]["p95"] - timings["score"]["p95"]
            by_threads[str(threads)] = timings | {"p95_difference_ms": difference}

            mlflow.log_metrics(
                {
                    f"threads_{threads}.{operation}.{point}": value
                    for operation, measured in timings.items()
                    for point, value in measured.items()
                }
                | {f"threads_{threads}.p95_difference_ms": difference}
            )
            log.info(
                "%2d thread(s)   score p95 %8.2f ms   explain p95 %8.2f ms   diff %8.2f ms",
                threads,
                timings["score"]["p95"],
                timings["explain"]["p95"],
                difference,
            )

    record = {
        "name": NAME,
        "created": datetime.now(UTC).isoformat(timespec="seconds"),
        "git_revision": git_revision(),
        "model": model_name,
        "trees": booster.num_trees(),
        "features": len(columns),
        "split": split,
        "rows": latency_cfg["rows"],
        "warmup": latency_cfg["warmup"],
        "machine": machine(),
        "by_threads": by_threads,
    }
    path = Path(paths["explain_latency"])
    path.write_text(json.dumps(record, indent=2) + "\n")
    log.info("wrote %s", path)


if __name__ == "__main__":
    main()
