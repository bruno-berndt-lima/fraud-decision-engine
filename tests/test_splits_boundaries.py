"""Tests for the config-to-boundaries arithmetic.

No data is loaded. `resolve_boundaries` is pure, so every case here is a dict
with one key changed — which is the whole reason it was written that way.
"""

from pathlib import Path

import pytest
import yaml

from fraud_engine.data.splits import SPLIT_NAMES, resolve_boundaries

REPO_ROOT = Path(__file__).resolve().parents[1]

# Layout B, as recorded in docs/experiments.md E1.
EXPECTED = {
    "train": (1, 90),
    "val_fit": (121, 140),
    "val_cal": (141, 160),
    "test": (161, 182),
}


def make_cfg(**overrides) -> dict:
    """A coherent splits config, with one key swappable."""
    cfg = {
        "gap_days": 30,
        "train_start": 1,
        "val_fit_start": 121,
        "val_cal_start": 141,
        "test_start": 161,
        "test_end": 182,
    }
    cfg.update(overrides)
    return cfg


def test_resolves_to_the_documented_layout():
    assert resolve_boundaries(make_cfg()) == EXPECTED


def test_the_committed_config_resolves_to_the_documented_layout():
    """Guards config.yaml against drifting from the numbers in docs/.

    The split counts in experiments.md and hypotheses.md are quoted against
    these boundaries; editing config without editing docs is otherwise silent.
    """
    config = yaml.safe_load((REPO_ROOT / "config" / "config.yaml").read_text())
    assert resolve_boundaries(config["splits"]) == EXPECTED


def test_keys_are_the_split_names_in_chronological_order():
    assert tuple(resolve_boundaries(make_cfg())) == SPLIT_NAMES


def test_every_range_is_non_empty():
    for name, (start, end) in resolve_boundaries(make_cfg()).items():
        assert start <= end, f"{name} is inverted: ({start}, {end})"


def test_evaluation_splits_abut_with_no_gap_between_them():
    """Only train is purged. A gap inside validation would silently drop days."""
    boundaries = resolve_boundaries(make_cfg())
    assert boundaries["val_fit"][1] + 1 == boundaries["val_cal"][0]
    assert boundaries["val_cal"][1] + 1 == boundaries["test"][0]


# ---- The E1 contract --------------------------------------------------------
# The reason config declares starts and derives ends. If these fail, the
# ablation is no longer a one-value change and the two runs differ in more than
# their training window.


def test_gap_days_zero_extends_train_and_moves_nothing_else():
    """docs/experiments.md E1: the unpurged run must differ only in train."""
    purged = resolve_boundaries(make_cfg())
    unpurged = resolve_boundaries(make_cfg(gap_days=0))

    assert unpurged["train"] == (1, 120)
    for name in ("val_fit", "val_cal", "test"):
        assert unpurged[name] == purged[name], f"{name} moved with the gap"


@pytest.mark.parametrize("gap_days", [0, 1, 15, 30, 60, 119])
def test_evaluation_ranges_are_independent_of_the_gap(gap_days):
    boundaries = resolve_boundaries(make_cfg(gap_days=gap_days))
    for name in ("val_fit", "val_cal", "test"):
        assert boundaries[name] == EXPECTED[name]


@pytest.mark.parametrize("gap_days", [0, 1, 15, 30, 60, 119])
def test_the_unclaimed_days_between_train_and_val_fit_equal_gap_days(gap_days):
    """The gap is not stored, so this is the only place its width is checked."""
    boundaries = resolve_boundaries(make_cfg(gap_days=gap_days))
    train_end = boundaries["train"][1]
    val_fit_start = boundaries["val_fit"][0]
    assert val_fit_start - train_end - 1 == gap_days


# ---- Guards -----------------------------------------------------------------


def test_the_widest_valid_gap_leaves_a_single_training_day():
    assert resolve_boundaries(make_cfg(gap_days=119))["train"] == (1, 1)


def test_one_day_past_the_widest_valid_gap_raises():
    """Pins the exact edge of the guard, which off-by-ones sit on."""
    with pytest.raises(ValueError):
        resolve_boundaries(make_cfg(gap_days=120))


@pytest.mark.parametrize(
    ("label", "override"),
    [
        ("negative gap overlaps train into val_fit", {"gap_days": -1}),
        ("gap wider than the training window", {"gap_days": 500}),
        ("train_start after val_fit_start", {"train_start": 200}),
        ("val_cal_start before val_fit_start", {"val_cal_start": 100}),
        ("test_start before val_cal_start", {"test_start": 130}),
        ("val_fit_start equal to val_cal_start", {"val_cal_start": 121}),
        ("test_end before test_start", {"test_end": 10}),
    ],
)
def test_rejects_incoherent_boundaries(label, override):
    with pytest.raises(ValueError):
        resolve_boundaries(make_cfg(**override))


def test_the_negative_gap_error_names_the_overlapping_days():
    """A negative gap is leakage, not a shorter purge.

    Asserted because an earlier version of this guard carried the message for
    the opposite failure — it reported "leaves no training data" for a config
    that produced 125 days of it, sending the reader the wrong way.
    """
    with pytest.raises(ValueError, match=r"121-125"):
        resolve_boundaries(make_cfg(gap_days=-5))


@pytest.mark.parametrize("key", sorted(make_cfg()))
def test_missing_config_key_raises(key):
    cfg = make_cfg()
    del cfg[key]
    with pytest.raises(KeyError):
        resolve_boundaries(cfg)
