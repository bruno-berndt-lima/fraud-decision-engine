"""The §3.1 measurement: p95 under sustained concurrent load, against the container.

`docs/serving.md` §7. `explainability.md` §8 timed a prediction in a process; this times
a transaction through HTTP, a worker pool, a budget and a policy — the thing the budget
was written about.

**Replay is measurement and changes nothing.** Transactions are read from a scored split
and sent as a caller would send them. No probability, no threshold and no USD figure is
recomputed here; what is recorded is how long the service took to answer.

**The generator shares the machine.** It runs beside the container it is measuring, on
the same cores, and that inflates the tail. Said here and in the record rather than
presenting the figure as if it came from a load cell on another host.

**The container is started by this stage, not found running.** The worker count is what
the table has to name, and a service somebody started earlier with a different one would
be reported under this run's number. Starting it here makes the record's `workers` a fact
about the process that was measured.

**Nothing here decides anything.** A budget missed is reported as missed (§7): the
reading rule refuses a miss repaired by lowering the concurrency until it passes, and
§9 forbids moving a threshold, a parameter or a feature because of what this shows.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import platform
import subprocess
import time
from collections.abc import Generator
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path

import httpx2
import lightgbm as lgb
import mlflow
import numpy as np
import pandas as pd

from fraud_engine.data.load import DEFAULT_CONFIG_PATH, load_config
from fraud_engine.evaluation.report import git_revision
from fraud_engine.evaluation.timing import percentiles
from fraud_engine.evaluation.tracking import configure_tracking, tracked_child, tracked_run
from fraud_engine.serving.decide import MODEL
from fraud_engine.serving.transform import raw_inputs

log = logging.getLogger(__name__)

NAME = "serving_latency"

# How long to wait for a worker to load a booster of a few thousand trees before
# deciding the container is not coming up at all.
STARTUP_TIMEOUT_S = 120


def request_bodies(paths: dict, split: str, rows: int, seed: int) -> list[dict]:
    """Real transactions as a caller would send them.

    The field list comes from the booster, because the contract does: `schemas` derives
    it from the same names, so a body built from anything else would be refused for
    fields the model does not hold. The booster is released immediately — the generator
    shares this machine with the service and has no business keeping it resident.

    Args:
        paths: The `paths` block of `config.yaml`.
        split: Which scored split to replay.
        rows: How many distinct transactions to draw.
        seed: Fixed, so a rerun replays the same traffic.

    Returns:
        One JSON-ready body per transaction, nulls where the record has gaps.
    """
    booster = lgb.Booster(model_file=paths["model"])
    carried = raw_inputs(booster.feature_name())
    del booster

    identifiers = (
        pd.read_parquet(f"{paths['features_dir']}/{split}.parquet", columns=["TransactionID"])
        .sample(rows, random_state=seed)["TransactionID"]
        .tolist()
    )
    frame = pd.read_parquet(
        paths["interim"], filters=[("TransactionID", "in", identifiers)]
    ).convert_dtypes(convert_integer=False, convert_floating=False)

    columns = [name for name in carried if name in frame.columns]
    return [
        {
            name: (None if pd.isna(value) else bool(value) if name == "has_identity" else value)
            for name, value in row.items()
        }
        for row in frame[columns].astype("object").to_dict(orient="records")
    ]


@contextmanager
def container(image: str, workers: int, port: int) -> Generator[str]:
    """The service, up and answering, for as long as the block runs.

    `--rm` so a crashed run leaves nothing behind to be measured by the next one. The
    logs are printed when it never becomes ready, because a container that died on
    startup says why in them and nowhere else.

    Args:
        image: The tag `make image` built.
        workers: Passed as `WEB_CONCURRENCY`; uvicorn reads it when `--workers` is not
            given, which is why the image's CMD does not give it.
        port: Published on the host.

    Yields:
        The base URL to send requests to.

    Raises:
        RuntimeError: If the container never answers `/health`.
    """
    started = subprocess.run(
        [
            "docker",
            "run",
            "--rm",
            "--detach",
            "--env",
            f"WEB_CONCURRENCY={workers}",
            "--publish",
            f"{port}:8000",
            image,
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    name = started.stdout.strip()
    base_url = f"http://127.0.0.1:{port}"

    try:
        deadline = time.monotonic() + STARTUP_TIMEOUT_S
        while time.monotonic() < deadline:
            try:
                if httpx2.get(f"{base_url}/health", timeout=2).status_code == 200:
                    break
            except httpx2.HTTPError:
                time.sleep(0.5)
        else:
            logs = subprocess.run(
                ["docker", "logs", name], capture_output=True, text=True, check=False
            )
            raise RuntimeError(f"the container never became ready:\n{logs.stdout}{logs.stderr}")

        yield base_url
    finally:
        subprocess.run(["docker", "stop", name], capture_output=True, check=False)


async def drive(
    base_url: str, bodies: list[dict], concurrency: int, seconds: float, warmup: float
) -> dict[str, object]:
    """Hold `concurrency` requests in flight for `seconds`, and time every one.

    Each coroutine loops on its own slice of the bodies, so the traffic is real
    transactions rather than one row replayed — a single body would be scored through
    warm branches no production mix would give the predictor.

    **Warmup is discarded rather than skipped.** The load runs continuously and the
    samples from the first `warmup` seconds are dropped afterwards, so the service is
    never measured while it is ramping and never idles between the two phases either.

    **What a sample includes.** Connection reuse, the queue in front of the worker pool,
    the prediction, and the trip back. `budget.py` already says queueing counts: a caller
    does not care which half of the delay was scoring.

    Args:
        base_url: From `container`.
        bodies: The replay traffic.
        concurrency: Requests in flight.
        seconds: Total wall time, warmup included.
        warmup: Leading seconds whose samples are dropped.

    Returns:
        `samples` in milliseconds, `errors`, `fell_back`, and the `measured` seconds
        the surviving samples span.
    """
    # (offset from start, milliseconds, served by the model) per completed request, and
    # the offset is kept so every count in a row is taken on the same basis — an error
    # rate over the whole run beside a p95 over the measured window is two numbers about
    # different populations.
    completed: list[tuple[float, float, bool]] = []
    failures: list[float] = []

    # Above the concurrency, so the client is never the queue. A generator that throttles
    # itself measures the generator.
    limits = httpx2.Limits(max_connections=concurrency * 2, max_keepalive_connections=concurrency)

    async with httpx2.AsyncClient(base_url=base_url, timeout=30.0, limits=limits) as client:
        start = time.perf_counter()
        deadline = start + seconds

        async def sender(offset: int) -> None:
            position = offset

            while time.perf_counter() < deadline:
                body = bodies[position % len(bodies)]
                position += 1

                sent = time.perf_counter()
                try:
                    answer = await client.post("/score", json=body)
                except httpx2.HTTPError:
                    failures.append(sent - start)
                    continue

                elapsed = (time.perf_counter() - sent) * 1000
                if answer.status_code != 200:
                    failures.append(sent - start)
                    continue

                completed.append((sent - start, elapsed, answer.json()["mode"] == MODEL))

        await asyncio.gather(*(sender(index) for index in range(concurrency)))

    measured = [(elapsed, by_model) for offset, elapsed, by_model in completed if offset >= warmup]

    return {
        "samples": np.asarray([elapsed for elapsed, _ in measured], dtype="float64"),
        "errors": sum(1 for offset in failures if offset >= warmup),
        "fell_back": sum(1 for _, by_model in measured if not by_model),
        "measured": max(seconds - warmup, 0.0),
    }


def summarise(run: dict[str, object], concurrency: int, workers: int, budget_ms: float) -> dict:
    """One row of the §7 table.

    `met` is the verdict on this row alone. §7 reports a budget met at one concurrency
    and missed at another as both, so nothing here aggregates the rows into a single
    pass or fail.

    Raises:
        ValueError: If no sample survived the warmup, which is a run that measured
            nothing rather than a run that was fast.
    """
    samples: np.ndarray = run["samples"]  # type: ignore[assignment]
    if not samples.size:
        raise ValueError(
            f"concurrency {concurrency}: no request completed after the warmup; "
            "the duration is shorter than the warmup, or nothing was served"
        )

    measured = float(run["measured"])  # type: ignore[arg-type]
    timings = percentiles(samples)

    return {
        "concurrency": concurrency,
        "workers": workers,
        "requests": int(samples.size),
        "throughput_rps": samples.size / measured if measured else float("nan"),
        "errors": int(run["errors"]),  # type: ignore[arg-type]
        "fell_back": int(run["fell_back"]),  # type: ignore[arg-type]
        **timings,
        "met": bool(timings["p95"] < budget_ms),
    }


def host() -> dict[str, object]:
    """The machine both the container and the generator ran on.

    Only the hardware, deliberately. The Python and LightGBM versions doing the work are
    the *image's*, not this process's, and `/health` reports what the service is holding —
    recording the generator's versions here would describe the wrong program.
    """
    runtime = subprocess.run(
        ["docker", "info", "--format", "{{.NCPU}} {{.MemTotal}} {{.OperatingSystem}}"],
        capture_output=True,
        text=True,
        check=True,
    ).stdout.split(maxsplit=2)

    return {
        "platform": platform.platform(),
        "processor": platform.processor() or platform.machine(),
        "cpu_count": os.cpu_count(),
        # What the container actually got, which on a Mac is a Linux VM and not the
        # cores above it. A budget judged against the host's core count would be judged
        # against hardware the service never ran on.
        "container_cpus": int(runtime[0]),
        "container_memory_bytes": int(runtime[1]),
        "container_runtime": runtime[2].strip(),
        "generator_co_located": True,
    }


def image_id(image: str) -> str:
    """The digest of what was measured, so the table names an image and not a tag."""
    found = subprocess.run(
        ["docker", "image", "inspect", "--format", "{{.Id}}", image],
        capture_output=True,
        text=True,
        check=True,
    )
    return found.stdout.strip()


def main(config_path: Path = DEFAULT_CONFIG_PATH) -> None:
    """Replay traffic at each worker and concurrency, and write the latency record.

    Wiring only. Invoked by `make loadtest` as `python -m fraud_engine.serving.loadtest`.

    Args:
        config_path: Path to `config.yaml`.
    """
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    # Silenced because it is a contaminant, not because it is noisy: the client logs a
    # line per request, and at these rates the generator would spend its own CPU
    # formatting and writing them on the machine it is measuring.
    for chatty in ("httpx2", "httpcore2"):
        logging.getLogger(chatty).setLevel(logging.WARNING)

    config = load_config(config_path)
    paths, load_cfg = config["paths"], config["serving_latency"]
    budget_ms = float(config["serving"]["budget_ms"])

    configure_tracking(config["tracking"])

    bodies = request_bodies(paths, load_cfg["split"], load_cfg["rows"], load_cfg["seed"])
    log.info("replaying %d transactions from %s", len(bodies), load_cfg["split"])

    params = {
        "image": load_cfg["image"],
        "split": load_cfg["split"],
        "budget_ms": budget_ms,
        "threads": config["serving"]["threads"],
        "workers": ",".join(str(count) for count in load_cfg["workers"]),
        "concurrency": ",".join(str(count) for count in load_cfg["concurrency"]),
    }

    rows, health = [], {}
    with tracked_run(NAME, params, config_path):
        for workers in load_cfg["workers"]:
            with container(load_cfg["image"], workers, load_cfg["port"]) as base_url:
                health = httpx2.get(f"{base_url}/health", timeout=10).json()

                for concurrency in load_cfg["concurrency"]:
                    run = asyncio.run(
                        drive(
                            base_url,
                            bodies,
                            concurrency,
                            load_cfg["duration_seconds"],
                            load_cfg["warmup_seconds"],
                        )
                    )
                    row = summarise(run, concurrency, workers, budget_ms)
                    rows.append(row)

                    with tracked_child(f"w{workers}_c{concurrency}", row):
                        mlflow.log_metrics(
                            {key: value for key, value in row.items() if key != "met"}
                        )
                    log.info(
                        "%2d worker(s) %3d in flight   p50 %7.2f   p95 %7.2f   p99 %8.2f ms   "
                        "%6.1f req/s   %s",
                        workers,
                        concurrency,
                        row["p50"],
                        row["p95"],
                        row["p99"],
                        row["throughput_rps"],
                        "met" if row["met"] else "MISSED",
                    )

    record = {
        "name": NAME,
        "created": datetime.now(UTC).isoformat(timespec="seconds"),
        "git_revision": git_revision(),
        "image": image_id(load_cfg["image"]),
        "budget_ms": budget_ms,
        "threads_per_worker": config["serving"]["threads"],
        "split": load_cfg["split"],
        "transactions": len(bodies),
        "duration_seconds": load_cfg["duration_seconds"],
        "warmup_seconds": load_cfg["warmup_seconds"],
        "host": host(),
        "service": {
            "artifacts": health.get("artifacts"),
            "features": health.get("features"),
            "trees": health.get("trees"),
            "calibration": health.get("calibration"),
        },
        "runs": rows,
    }
    path = Path(paths["serving_latency"])
    path.write_text(json.dumps(record, indent=2) + "\n")
    log.info("wrote %s", path)


if __name__ == "__main__":
    main()
