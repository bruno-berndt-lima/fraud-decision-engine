"""Tests for the MLflow wiring.

Every test gets its own store under ``tmp_path``: MLflow's tracking URI and
active run are process-global, so a test that leaked either would decide what
the next one measured. Nothing here reads the project's real ``mlruns/``.

Synthetic records throughout — ``flatten_metrics`` takes a mapping, so nothing
has to be scored to test what it names.
"""

from collections.abc import Iterator
from pathlib import Path

import mlflow
import pytest
import yaml

from fraud_engine.evaluation.report import git_revision
from fraud_engine.evaluation.tracking import (
    ARTIFACTS,
    DATABASE,
    capacity_suffix,
    configure_tracking,
    flatten_metrics,
    log_provenance,
    tracked_run,
)

EXPERIMENT = "test-experiment"

BOUNDARIES = {
    "gap_days": 30,
    "train_start": 1,
    "val_fit_start": 121,
    "val_cal_start": 141,
    "test_start": 161,
    "test_end": 182,
}


def make_report(splits: tuple[str, ...] = ("val_fit",), capacities=(0.005, 0.01)) -> dict:
    """A metrics record shaped like the one ``build_report`` returns."""
    return {
        "name": "probe",
        "capacities": list(capacities),
        "splits": {
            split: {
                "n": 1000,
                "positives": 40,
                "base_rate": 0.04,
                "pr_auc": 0.4,
                "roc_auc": 0.8,
                "recall_at_capacity": [
                    {
                        "capacity": capacity,
                        "recall": 0.1 * index + 0.1,
                        "reviewed": 5,
                        "caught": 2,
                        "positives": 40,
                        "recall_ceiling": 0.5,
                        "share_of_ceiling": 0.6,
                        "ambiguous_days": 0,
                    }
                    for index, capacity in enumerate(capacities)
                ],
            }
            for split in splits
        },
    }


def write_config(tmp_path: Path) -> Path:
    """A config with absolute paths, so provenance does not depend on the cwd."""
    cost_matrix = tmp_path / "cost_matrix.yaml"
    cost_matrix.write_text(yaml.safe_dump({"review_capacities": [0.01]}))

    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        yaml.safe_dump({"splits": BOUNDARIES, "paths": {"cost_matrix": str(cost_matrix)}})
    )
    return config_path


@pytest.fixture
def store(tmp_path: Path) -> Iterator[Path]:
    """An isolated tracking store, torn down with no run left open."""
    location = tmp_path / "store"
    configure_tracking({"store": str(location), "experiment_name": EXPERIMENT})
    yield location
    if mlflow.active_run() is not None:
        mlflow.end_run()


def only_run() -> mlflow.entities.Run:
    """The single run in the test experiment."""
    runs = mlflow.search_runs(experiment_names=[EXPERIMENT], output_format="list")
    assert len(runs) == 1
    return runs[0]


# ------------------------------------------------------------------------------
# configure_tracking
# ------------------------------------------------------------------------------


def test_a_relative_store_resolves_against_the_working_directory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    monkeypatch.chdir(tmp_path)
    configure_tracking({"store": "mlruns", "experiment_name": EXPERIMENT})

    uri = mlflow.get_tracking_uri()
    assert uri == f"sqlite:///{tmp_path.resolve() / 'mlruns' / DATABASE}"


def test_the_experiment_is_created_if_it_does_not_exist(store: Path):
    assert mlflow.get_experiment_by_name(EXPERIMENT) is not None


def test_configuring_twice_reuses_the_same_experiment(store: Path):
    first = mlflow.get_experiment_by_name(EXPERIMENT).experiment_id
    configure_tracking({"store": str(store), "experiment_name": EXPERIMENT})

    assert mlflow.get_experiment_by_name(EXPERIMENT).experiment_id == first


def test_artifacts_are_located_under_the_store(store: Path):
    location = mlflow.get_experiment_by_name(EXPERIMENT).artifact_location

    assert location == (store / ARTIFACTS).as_uri()


# ------------------------------------------------------------------------------
# capacity_suffix
# ------------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("capacity", "expected"),
    [(0.005, "50bps"), (0.01, "100bps"), (0.02, "200bps"), (0.0125, "125bps")],
)
def test_a_capacity_becomes_whole_basis_points(capacity: float, expected: str):
    assert capacity_suffix(capacity) == expected


def test_a_fractional_basis_point_is_refused_rather_than_rounded():
    with pytest.raises(ValueError, match="not whole"):
        capacity_suffix(0.00505)


# ------------------------------------------------------------------------------
# flatten_metrics
# ------------------------------------------------------------------------------


def test_the_name_carries_the_split_and_the_capacity():
    flat = flatten_metrics(make_report())

    assert flat["val_fit.pr_auc"] == 0.4
    assert flat["val_fit.recall_at_100bps"] == 0.2


def test_every_split_is_flattened_under_its_own_prefix():
    flat = flatten_metrics(make_report(splits=("val_fit", "val_cal")))

    assert {name.split(".")[0] for name in flat} == {"val_fit", "val_cal"}


def test_the_columns_that_do_not_vary_between_runs_are_left_out():
    flat = flatten_metrics(make_report())

    assert not [name for name in flat if "reviewed" in name or "caught" in name]
    assert not [name for name in flat if "recall_ceiling" in name]
    assert "val_fit.share_of_ceiling_at_100bps" in flat


def test_a_duplicate_name_is_refused_rather_than_overwritten():
    report = make_report()
    entries = report["splits"]["val_fit"]["recall_at_capacity"]
    entries.append(dict(entries[0]))

    with pytest.raises(ValueError, match="duplicate metric name"):
        flatten_metrics(report)


def test_every_value_is_a_float_because_mlflow_takes_numbers():
    flat = flatten_metrics(make_report())

    assert all(isinstance(value, float) for value in flat.values())


# ------------------------------------------------------------------------------
# log_provenance
# ------------------------------------------------------------------------------


def test_logging_provenance_outside_a_run_is_refused(store: Path, tmp_path: Path):
    with pytest.raises(RuntimeError, match="needs an open run"):
        log_provenance(write_config(tmp_path))


def test_the_revision_is_the_one_the_json_record_would_carry(store: Path, tmp_path: Path):
    with mlflow.start_run():
        log_provenance(write_config(tmp_path))

    assert only_run().data.params["git_revision"] == str(git_revision())


def test_the_split_boundaries_become_sortable_params(store: Path, tmp_path: Path):
    with mlflow.start_run():
        log_provenance(write_config(tmp_path))

    params = only_run().data.params
    assert {key: params[f"split.{key}"] for key in BOUNDARIES} == {
        key: str(value) for key, value in BOUNDARIES.items()
    }


def test_both_config_files_are_attached(store: Path, tmp_path: Path):
    with mlflow.start_run():
        log_provenance(write_config(tmp_path))

    attached = mlflow.MlflowClient().list_artifacts(only_run().info.run_id)
    assert {item.path for item in attached} == {"config.yaml", "cost_matrix.yaml"}


# ------------------------------------------------------------------------------
# tracked_run
# ------------------------------------------------------------------------------


def test_a_run_carries_its_name_and_the_callers_params(store: Path, tmp_path: Path):
    with tracked_run("probe", {"num_leaves": 31}, write_config(tmp_path)):
        pass

    run = only_run()
    assert run.data.tags["mlflow.runName"] == "probe"
    assert run.data.params["num_leaves"] == "31"
    assert run.info.status == "FINISHED"


def test_a_failing_body_leaves_the_run_marked_failed(store: Path, tmp_path: Path):
    with (
        pytest.raises(RuntimeError, match="boom"),
        tracked_run("doomed", {"num_leaves": 31}, write_config(tmp_path)),
    ):
        raise RuntimeError("boom")

    assert only_run().info.status == "FAILED"


def test_provenance_survives_a_failing_body(store: Path, tmp_path: Path):
    with (
        pytest.raises(RuntimeError),
        tracked_run("doomed", {"num_leaves": 31}, write_config(tmp_path)),
    ):
        raise RuntimeError("boom")

    assert only_run().data.params["split.gap_days"] == "30"


def test_a_caller_cannot_shadow_a_provenance_param(store: Path, tmp_path: Path):
    with (
        pytest.raises(mlflow.exceptions.MlflowException),
        tracked_run("shadow", {"git_revision": "not-the-commit"}, write_config(tmp_path)),
    ):
        pass

    assert only_run().data.params["git_revision"] == str(git_revision())
