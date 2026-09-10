"""Tests for E1's arm construction.

The guard that matters is the one on writing. An arm that failed to redirect a
path would overwrite the shipped artifact it is being compared against, and the
comparison would then be against something the arm itself produced — a wrong
answer with no symptom.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from fraud_engine.data.splits import resolve_boundaries
from fraud_engine.models import purge
from fraud_engine.models.purge import (
    ARMS,
    REDIRECTED,
    REFERENCE_ARM,
    SHARED,
    build_arm,
    redirect,
)

CONFIG = {
    "paths": {
        "interim": "data/interim/transactions.parquet",
        "splits": "data/splits/splits.parquet",
        "split_summary": "reports/metrics/split_summary.csv",
        "features_dir": "data/features",
        "encoders": "models/encoders.parquet",
        "amount_stats": "models/amount_stats.parquet",
        "vblock": "models/vblock.parquet",
        "metrics_dir": "reports/metrics",
    },
    "splits": {
        "gap_days": 30,
        "train_start": 1,
        "val_fit_start": 121,
        "val_cal_start": 141,
        "test_start": 161,
        "test_end": 182,
    },
    "model": {"seed": 0},
}


@pytest.fixture
def arm() -> dict:
    return redirect(CONFIG, "data/e1/unpurged", train_start=1, gap_days=0)


def test_the_window_is_the_whole_experimental_variable(arm):
    """An arm is two integers; `resolve_boundaries` derives the rest."""
    assert arm["splits"]["train_start"] == 1
    assert arm["splits"]["gap_days"] == 0


def test_the_evaluation_boundaries_do_not_move(arm):
    """Only the training window differs, or the arms measure nothing."""
    for key in ("val_fit_start", "val_cal_start", "test_start", "test_end"):
        assert arm["splits"][key] == CONFIG["splits"][key]


def test_every_written_path_lands_under_the_arm(arm):
    for key in REDIRECTED:
        assert Path(arm["paths"][key]).parent == Path("data/e1/unpurged")


def test_each_file_keeps_its_own_name(arm):
    """An arm's directory reads like a small copy of the project."""
    for key in REDIRECTED:
        assert Path(arm["paths"][key]).name == Path(CONFIG["paths"][key]).name


def test_no_shipped_write_location_survives(arm):
    """The point of the whole function, asserted directly."""
    shipped = {CONFIG["paths"][key] for key in REDIRECTED}

    assert not shipped & set(arm["paths"].values())


def test_the_interim_frame_is_shared(arm):
    """Pre-split, so every arm reads the same rows and differs only in labelling."""
    assert arm["paths"][SHARED] == CONFIG["paths"][SHARED]


def test_paths_outside_the_pipeline_are_left_alone(arm):
    assert arm["paths"]["metrics_dir"] == CONFIG["paths"]["metrics_dir"]


def test_unrelated_config_blocks_pass_through(arm):
    assert arm["model"] == CONFIG["model"]


def test_the_shipped_config_is_not_modified():
    before = {"paths": dict(CONFIG["paths"]), "splits": dict(CONFIG["splits"])}

    redirect(CONFIG, "data/e1/recent", train_start=31, gap_days=0)

    assert CONFIG["paths"] == before["paths"]
    assert CONFIG["splits"] == before["splits"]


def test_two_arms_do_not_share_a_directory():
    first = redirect(CONFIG, "data/e1/recent", train_start=31, gap_days=0)
    second = redirect(CONFIG, "data/e1/unpurged", train_start=1, gap_days=0)

    assert not {first["paths"][key] for key in REDIRECTED} & {
        second["paths"][key] for key in REDIRECTED
    }


def test_a_missing_path_raises_rather_than_writing_where_it_shipped():
    """A renamed config key would leave one stage writing into the project."""
    stripped = {**CONFIG, "paths": {k: v for k, v in CONFIG["paths"].items() if k != "vblock"}}

    with pytest.raises(ValueError, match="an arm would write where the shipped pipeline writes"):
        redirect(stripped, "data/e1/unpurged", train_start=1, gap_days=0)


def test_paths_that_would_collide_raise():
    """Two stages writing one file is a silent overwrite, not an error."""
    colliding = {
        **CONFIG,
        "paths": {**CONFIG["paths"], "encoders": "models/vblock.parquet"},
    }

    with pytest.raises(ValueError, match="collide under"):
        redirect(colliding, "data/e1/unpurged", train_start=1, gap_days=0)


def test_the_shipped_window_can_be_reproduced():
    """The `purged` arm is the shipped split, built the same way as the others.

    Reading the shipped matrices for it instead would compare artifacts built by
    two different code paths on two different days.
    """
    purged = redirect(CONFIG, "data/e1/purged", train_start=1, gap_days=30)

    assert purged["splits"] == CONFIG["splits"]


# ------------------------------------------------------------------------------
# ARMS
# ------------------------------------------------------------------------------


def test_the_reference_arm_is_the_shipped_window():
    assert ARMS[REFERENCE_ARM] == {"train_start": 1, "gap_days": 30}


def test_the_reference_comes_first():
    assert next(iter(ARMS)) == REFERENCE_ARM


def test_the_middle_arm_holds_the_shipped_width():
    """Recency without volume. If its width drifted, the decomposition stops working."""
    boundaries = {
        name: resolve_boundaries({**CONFIG["splits"], **arm}) for name, arm in ARMS.items()
    }
    widths = {name: b["train"][1] - b["train"][0] + 1 for name, b in boundaries.items()}

    assert widths["recent"] == widths[REFERENCE_ARM]
    assert widths["unpurged"] > widths[REFERENCE_ARM]


def test_only_the_unpurged_arms_reach_the_validation_boundary():
    boundaries = {
        name: resolve_boundaries({**CONFIG["splits"], **arm}) for name, arm in ARMS.items()
    }
    first_validation_day = CONFIG["splits"]["val_fit_start"]

    assert boundaries[REFERENCE_ARM]["train"][1] < first_validation_day - 1
    for name in ("recent", "unpurged"):
        assert boundaries[name]["train"][1] == first_validation_day - 1


def test_no_arm_moves_an_evaluation_boundary():
    for arm in ARMS.values():
        boundaries = resolve_boundaries({**CONFIG["splits"], **arm})
        for split in ("val_fit", "val_cal", "test"):
            assert boundaries[split] == resolve_boundaries(CONFIG["splits"])[split]


# ------------------------------------------------------------------------------
# build_arm
# ------------------------------------------------------------------------------


@pytest.fixture
def stages(monkeypatch: pytest.MonkeyPatch) -> list[Path]:
    """Record which config path each stage was handed, and run nothing."""
    seen: list[Path] = []
    monkeypatch.setattr(purge.splits, "main", lambda path: seen.append(path))
    monkeypatch.setattr(purge.build, "main", lambda path: seen.append(path))
    return seen


def test_both_stages_run_against_the_arms_config(tmp_path: Path, stages):
    build_arm(CONFIG, tmp_path / "unpurged", train_start=1, gap_days=0)

    assert stages == [tmp_path / "unpurged" / "config.yaml"] * 2


def test_the_arms_config_is_left_beside_its_artifacts(tmp_path: Path, stages):
    """Provenance: what produced these matrices is a file next to them."""
    arm = build_arm(CONFIG, tmp_path / "unpurged", train_start=1, gap_days=0)

    written = yaml.safe_load((tmp_path / "unpurged" / "config.yaml").read_text())

    assert written == arm


def test_the_written_config_carries_the_arms_window(tmp_path: Path, stages):
    build_arm(CONFIG, tmp_path / "recent", train_start=31, gap_days=0)

    written = yaml.safe_load((tmp_path / "recent" / "config.yaml").read_text())

    assert (written["splits"]["train_start"], written["splits"]["gap_days"]) == (31, 0)


def test_the_written_config_points_at_the_arms_directory(tmp_path: Path, stages):
    """A stage reading it must not find a shipped path."""
    build_arm(CONFIG, tmp_path / "unpurged", train_start=1, gap_days=0)

    written = yaml.safe_load((tmp_path / "unpurged" / "config.yaml").read_text())

    for key in REDIRECTED:
        assert Path(written["paths"][key]).is_relative_to(tmp_path / "unpurged")


def test_the_directory_is_created(tmp_path: Path, stages):
    build_arm(CONFIG, tmp_path / "nested" / "unpurged", train_start=1, gap_days=0)

    assert (tmp_path / "nested" / "unpurged").is_dir()


# ------------------------------------------------------------------------------
# measure
# ------------------------------------------------------------------------------


def test_measuring_without_the_purged_arm_raises():
    """A table of unpurged runs answers nothing."""
    with pytest.raises(KeyError, match="it is what the gap is measured against"):
        purge.measure({"unpurged": {}}, {}, [0.1], CONFIG["paths"])
