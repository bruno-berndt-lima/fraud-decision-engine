"""Tests for the per-section config stamps.

The property that matters is the one make relies on: a stamp's modification time
moves when its section's content does, and only then.
"""

from __future__ import annotations

import os
from pathlib import Path

from fraud_engine.config_stamps import UNSTAMPED, digest, write_stamps

CONFIG = {
    "paths": {"interim": "data/interim/transactions.parquet"},
    "tracking": {"store": "mlruns"},
    "splits": {"gap_days": 30, "train_start": 1},
    "model": {"seed": 0},
}


def backdate(path: Path) -> None:
    os.utime(path, (1_000_000_000, 1_000_000_000))


def test_one_stamp_per_stamped_section(tmp_path):
    write_stamps(CONFIG, tmp_path)

    assert {path.stem for path in tmp_path.glob("*.stamp")} == set(CONFIG) - UNSTAMPED


def test_an_unchanged_section_keeps_its_timestamp(tmp_path):
    write_stamps(CONFIG, tmp_path)
    backdate(tmp_path / "splits.stamp")

    assert write_stamps(CONFIG, tmp_path) == []
    assert (tmp_path / "splits.stamp").stat().st_mtime == 1_000_000_000


def test_a_changed_section_rewrites_only_its_own_stamp(tmp_path):
    write_stamps(CONFIG, tmp_path)
    for path in tmp_path.glob("*.stamp"):
        backdate(path)

    changed = write_stamps({**CONFIG, "model": {"seed": 1}}, tmp_path)

    assert changed == ["model"]
    assert (tmp_path / "model.stamp").stat().st_mtime > 1_000_000_000
    assert (tmp_path / "splits.stamp").stat().st_mtime == 1_000_000_000


def test_key_order_is_not_a_change(tmp_path):
    write_stamps(CONFIG, tmp_path)

    reordered = {**CONFIG, "splits": {"train_start": 1, "gap_days": 30}}

    assert write_stamps(reordered, tmp_path) == []


def test_unstamped_sections_never_restage_anything(tmp_path):
    write_stamps(CONFIG, tmp_path)

    moved = {**CONFIG, "paths": {"interim": "elsewhere.parquet"}, "tracking": {"store": "x"}}

    assert write_stamps(moved, tmp_path) == []


def test_a_removed_section_loses_its_stamp(tmp_path):
    write_stamps(CONFIG, tmp_path)

    config = {key: value for key, value in CONFIG.items() if key != "model"}

    assert write_stamps(config, tmp_path) == ["model"]
    assert not (tmp_path / "model.stamp").exists()


def test_a_nested_change_changes_the_digest():
    assert digest({"tune": {"trials": 60}}) != digest({"tune": {"trials": 61}})
