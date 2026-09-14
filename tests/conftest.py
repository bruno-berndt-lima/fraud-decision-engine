"""Fixtures shared across test modules.

Only what more than one module needs lives here. MLflow's tracking URI and its
active run are process-global, so the fixture that provides them has to be one
definition — two slightly different copies would decide, by test order, which
store the next test wrote into.
"""

from collections.abc import Iterator
from pathlib import Path

import mlflow
import pytest

from fraud_engine.evaluation.tracking import configure_tracking

EXPERIMENT = "test-experiment"


@pytest.fixture
def experiment_run(tmp_path: Path) -> Iterator[mlflow.ActiveRun]:
    """An isolated store with a parent run open, as an experiment's `main` provides.

    Every `measure` that fits a booster logs a child run and refuses to without
    a parent. Nothing here touches the project's real `mlruns/`.
    """
    configure_tracking({"store": str(tmp_path / "mlruns"), "experiment_name": EXPERIMENT})
    with mlflow.start_run(run_name="parent") as run:
        yield run
    while mlflow.active_run() is not None:
        mlflow.end_run()


def children(parent: mlflow.ActiveRun) -> list[mlflow.entities.Run]:
    """Every run nested directly under `parent`, oldest first."""
    return mlflow.search_runs(
        experiment_names=[EXPERIMENT],
        filter_string=f"tags.mlflow.parentRunId = '{parent.info.run_id}'",
        order_by=["attributes.start_time ASC"],
        output_format="list",
    )
