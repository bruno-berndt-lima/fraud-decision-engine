"""MLflow wiring — what produced a number, recorded alongside the number.

Not logging. Three weeks into Phase 05 the question is *which run scored this*,
and it has to be answerable from the store rather than from memory.

The one rule this module exists to enforce: **an MLflow run and the JSON record
it accompanies cannot disagree.** Metrics reach MLflow by flattening the record
``report.build_report`` already produced, never by a second calculation, and the
commit comes from ``report.git_revision`` for the same reason. Two independent
notions of "which code produced this" would be a defect, not a redundancy.

``mlflow-skinny`` rather than ``mlflow``: every release of the full package pins
``pandas<3`` and this project is on pandas 3, so ``uv`` silently resolves the
full package back to 1.27.0, which does not import at all against a current
protobuf. The skinny package is the same tracking client without the server, and
carries no pandas constraint. ``sqlalchemy`` and ``alembic`` are direct
dependencies for the same reason: they are what register the database backend,
which the skinny package omits and which MLflow 3 now requires.
"""

from __future__ import annotations

from pathlib import Path

import mlflow

# Everything MLflow writes lives under the configured store directory: the
# database beside the artifacts, so the whole record moves or is deleted as one
# thing. Names are fixed here rather than in config — they are internal layout,
# and a config key nobody would ever set to anything else is not a decision.
DATABASE = "mlflow.db"
ARTIFACTS = "artifacts"


def configure_tracking(tracking_cfg: dict) -> None:
    """Point MLflow at its store and select the experiment.

    Process-level global state, so this belongs at the top of a ``main`` and
    nowhere else. The store and the experiment are created if absent.

    **A database backend, not the file store.** MLflow 3 put ``file:./mlruns``
    into maintenance mode and raises rather than opening one, so the roadmap's
    suggestion no longer runs. SQLite is the supported local equivalent and is
    still a single file under a gitignored directory.

    **The store path is resolved here.** Every other relative path in this
    project is read or written immediately, so a wrong working directory raises.
    This one would not: MLflow would open a second, empty store and the run
    comparison — the only thing the tool is for — would split in two with nothing
    to show for it. The artifact location is resolved for the same reason and
    must be set at creation time, because ``set_experiment`` cannot change it
    afterwards.

    Args:
        tracking_cfg: The ``tracking:`` config block — ``store`` and
            ``experiment_name``.
    """
    store = Path(tracking_cfg["store"]).resolve()
    artifacts = store / ARTIFACTS
    artifacts.mkdir(parents=True, exist_ok=True)

    mlflow.set_tracking_uri(f"sqlite:///{store / DATABASE}")

    name = tracking_cfg["experiment_name"]
    if mlflow.get_experiment_by_name(name) is None:
        mlflow.create_experiment(name, artifact_location=artifacts.as_uri())
    mlflow.set_experiment(name)
