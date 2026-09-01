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

from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

import mlflow

from fraud_engine.data.load import load_config
from fraud_engine.evaluation.report import git_revision

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


# What varies between runs, and therefore what the comparison view is for.
# ``n``, ``positives`` and ``base_rate`` are constant for a given split, so they
# are not comparisons — they are the check that two runs scored the same
# population, which is worth three columns.
SPLIT_METRICS = ("pr_auc", "roc_auc", "base_rate", "n", "positives")

# Per capacity. ``reviewed`` and ``caught`` are omitted: with n fixed they are
# monotone in recall and say nothing extra, and every excluded column makes the
# comparison table harder to read. ``recall_ceiling`` is a property of the data
# rather than of the run; ``share_of_ceiling`` is the part that varies.
# ``ambiguous_days`` stays because it flags a capacity cut landing inside a tied
# block of scores, which is otherwise invisible here.
CAPACITY_METRICS = ("recall", "share_of_ceiling", "ambiguous_days")


def capacity_suffix(capacity: float) -> str:
    """A review capacity as a whole number of basis points, for use in a name.

    Basis points rather than a percentage so the suffix carries no decimal
    point, leaving the dot to mean one thing: the split prefix. A capacity that
    is not a whole number of basis points is refused rather than rounded,
    because rounding is how two capacities silently become one name.

    Args:
        capacity: A review capacity as a fraction, e.g. ``0.01``.

    Returns:
        e.g. ``"100bps"``.

    Raises:
        ValueError: If the capacity is not a whole number of basis points.
    """
    points = capacity * 10_000
    rounded = round(points)

    if abs(points - rounded) > 1e-9:
        raise ValueError(
            f"capacity {capacity!r} is {points} basis points, which is not whole; "
            "it has no unambiguous metric name"
        )

    return f"{rounded}bps"


def flatten_metrics(report: dict) -> dict[str, float]:
    """A metrics record as MLflow's flat ``{name: float}`` space.

    MLflow takes scalars; ``build_report`` produces a nested record whose
    per-capacity results are a list. The flattening is therefore a naming
    scheme, and the scheme is the part that is expensive to change: renaming a
    metric partitions the run history into "before" and "after" and the
    comparison view — the whole reason the tool is here — stops working across
    the boundary.

    ``{split}.{metric}``, and ``{split}.{metric}_at_{capacity}bps`` for the
    per-capacity results. The dot is the split prefix and nothing else, which is
    what makes the MLflow UI group the columns.

    Whatever splits the record holds are flattened. This is not the guard on
    scoring TEST — that lives in ``report.DEFAULT_SPLITS``. If a test column
    appears here, the run already touched test and the leak happened upstream.

    Args:
        report: A record as ``report.build_report`` returned it.

    Returns:
        ``{name: value}``, ready for ``mlflow.log_metrics``.

    Raises:
        ValueError: If two entries claim the same name — which would otherwise
            mean one silently overwriting the other.
    """
    flat: dict[str, float] = {}

    def put(name: str, value: float) -> None:
        if name in flat:
            raise ValueError(f"duplicate metric name {name!r}: one value would overwrite the other")
        flat[name] = float(value)

    for split, block in report["splits"].items():
        for metric in SPLIT_METRICS:
            put(f"{split}.{metric}", block[metric])

        for entry in block["recall_at_capacity"]:
            suffix = capacity_suffix(entry["capacity"])
            for metric in CAPACITY_METRICS:
                put(f"{split}.{metric}_at_{suffix}", entry[metric])

    return flat


def log_provenance(config_path: Path) -> None:
    """Record what produced this run, onto the run already open.

    Params rather than tags: params are immutable and become sortable columns in
    the comparison view, which is the point. Split boundaries in particular have
    to be columns — E1 varies ``gap_days`` and compares the two runs, and a value
    visible only inside an attached file cannot be sorted on.

    **The revision is ours, under the name the JSON already uses.** MLflow has a
    standard ``mlflow.source.git.commit`` tag and does not populate it here, so
    there is no second answer to overwrite; and it could not hold this one
    anyway, because ``git_revision`` marks a dirty tree and a commit hash field
    has nowhere to put that. The mark is the part that matters: a number from
    uncommitted code cannot be reproduced from history.

    **Both config files are attached.** ``cost_matrix.yaml`` is versioned
    separately and every recall@capacity depends on it, so a run recorded
    without it is comparable to nothing once those assumptions move.

    Args:
        config_path: Path to ``config.yaml``.

    Raises:
        RuntimeError: If no run is open. Logging outside a run makes MLflow
            start one, which would bury the provenance in a stray run rather
            than fail.
    """
    if mlflow.active_run() is None:
        raise RuntimeError("log_provenance needs an open run; call it inside start_run")

    config = load_config(config_path)

    mlflow.log_param("git_revision", git_revision())
    mlflow.log_params({f"split.{key}": value for key, value in config["splits"].items()})

    mlflow.log_artifact(config_path)
    mlflow.log_artifact(config["paths"]["cost_matrix"])


@contextmanager
def tracked_run(name: str, params: dict, config_path: Path) -> Iterator[None]:
    """Open a run, record what it is and what produced it, and close it.

    Provenance is logged before the caller's params, so a run identifies itself
    before it describes its settings — and so a run that dies mid-body still
    says which commit it came from.

    A failing body does not lose the run: ``start_run`` marks it FAILED and
    keeps everything logged so far. A crashed experiment that leaves no trace is
    how the same mistake gets made twice.

    ``log_params`` raises if a key is already set to a different value, so a
    caller cannot quietly shadow ``git_revision`` or a ``split.*`` boundary. That
    collision is a bug worth hearing about, not a precedence rule.

    Args:
        name: Run name, as it appears in the comparison view.
        params: This run's own settings — hyperparameters, feature set, variant.
        config_path: Path to ``config.yaml``, for ``log_provenance``.

    Yields:
        Nothing. Log metrics and models inside the block with ``mlflow`` calls;
        the run is already active.
    """
    with mlflow.start_run(run_name=name):
        log_provenance(config_path)
        mlflow.log_params(params)
        yield
