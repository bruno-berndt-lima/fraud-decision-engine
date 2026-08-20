"""Tests for the metrics record written per scored run.

Synthetic frames throughout — no parquet is read, since CI has no data.
"""

import json

import numpy as np
import pandas as pd
import pytest

from fraud_engine.evaluation.report import (
    DEFAULT_SPLITS,
    build_report,
    evaluate_splits,
    load_capacities,
    write_report,
)

CAPACITIES = [0.05, 0.1]

# Two days per split, so a per-day capacity has something to divide.
LAYOUT = {
    1: "train",
    2: "train",
    3: "val_fit",
    4: "val_fit",
    5: "val_cal",
    6: "val_cal",
    7: "test",
    8: "test",
}


def make_frame(per_day: int = 60, seed: int = 0) -> pd.DataFrame:
    """A scored frame with every split populated and both classes present."""
    rng = np.random.default_rng(seed)
    parts = []
    for day, split in LAYOUT.items():
        is_fraud = np.zeros(per_day, dtype=int)
        is_fraud[: per_day // 10] = 1
        rng.shuffle(is_fraud)
        parts.append(
            pd.DataFrame(
                {
                    "isFraud": is_fraud,
                    "score": rng.random(per_day),
                    "day": day,
                    "split": split,
                }
            )
        )
    return pd.concat(parts, ignore_index=True)


def make_cost_matrix(headline=0.01, sweep=(0.005, 0.01, 0.02)) -> dict:
    return {
        "constraints": {"review_capacity": {"value": headline}},
        "sensitivity": {"review_capacity": {"values": list(sweep)}},
    }


# ---- load_capacities ---------------------------------------------------------


def test_capacities_come_back_sorted():
    assert load_capacities(make_cost_matrix(sweep=(0.02, 0.005, 0.01))) == [0.005, 0.01, 0.02]


def test_rejects_a_headline_capacity_missing_from_the_sweep():
    """Every report would omit the operating point §5 states the criteria against."""
    with pytest.raises(ValueError, match="headline review capacity"):
        load_capacities(make_cost_matrix(headline=0.01, sweep=(0.005, 0.02)))


# ---- which splits get scored -------------------------------------------------


def test_test_is_excluded_by_default():
    """The point of the default. Measurement is unlimited, but scoring test
    while still iterating should take a deliberate keystroke."""
    assert "test" not in DEFAULT_SPLITS
    assert set(evaluate_splits(make_frame(), CAPACITIES)) == {"val_fit", "val_cal"}


def test_test_is_scored_when_named():
    scored = evaluate_splits(make_frame(), CAPACITIES, splits=(*DEFAULT_SPLITS, "test"))
    assert "test" in scored


def test_splits_are_reported_in_the_order_requested():
    scored = evaluate_splits(make_frame(), CAPACITIES, splits=("val_cal", "val_fit"))
    assert list(scored) == ["val_cal", "val_fit"]


def test_rejects_a_split_name_that_matches_nothing():
    """A typo would otherwise yield a valid report that is silently about nothing."""
    with pytest.raises(ValueError, match="val-fit"):
        evaluate_splits(make_frame(), CAPACITIES, splits=("val-fit",))


def test_rejects_an_empty_split_request():
    with pytest.raises(ValueError, match="no splits requested"):
        evaluate_splits(make_frame(), CAPACITIES, splits=())


@pytest.mark.parametrize("column", ["isFraud", "score", "day", "split"])
def test_rejects_a_frame_missing_a_required_column(column):
    with pytest.raises(ValueError, match=column):
        evaluate_splits(make_frame().drop(columns=column), CAPACITIES)


def test_gap_rows_are_ignored_rather_than_failing():
    """assign_splits leaves purged rows null; they belong to no split."""
    frame = make_frame()
    frame.loc[frame["day"] == 1, "split"] = None
    assert set(evaluate_splits(frame, CAPACITIES)) == {"val_fit", "val_cal"}


# ---- the record --------------------------------------------------------------


def test_report_carries_the_provenance_fields():
    report = build_report("run", make_frame(), CAPACITIES)
    assert report["name"] == "run"
    assert report["capacities"] == CAPACITIES
    assert report["created"].startswith("20")
    assert "git_revision" in report  # None outside a checkout, but always present


def test_report_records_which_splits_it_touched():
    """The audit trail: the keys are the answer to 'did this run see test?'."""
    report = build_report("run", make_frame(), CAPACITIES)
    assert list(report["splits"]) == list(DEFAULT_SPLITS)


def test_report_is_json_serialisable():
    """It is written to tracked reports/ and compared across phases."""
    report = build_report("run", make_frame(), CAPACITIES)
    assert json.loads(json.dumps(report))["name"] == "run"


@pytest.mark.parametrize("name", ["../escape", "with/slash", "with space", "", "dots.in.name"])
def test_rejects_a_name_that_is_not_a_safe_filename(name):
    with pytest.raises(ValueError, match="must be letters"):
        build_report(name, make_frame(), CAPACITIES)


@pytest.mark.parametrize("name", ["rules_baseline", "logreg", "lightgbm-v2", "run1"])
def test_accepts_ordinary_run_names(name):
    assert build_report(name, make_frame(), CAPACITIES)["name"] == name


# ---- writing -----------------------------------------------------------------


def test_writes_one_file_named_for_the_run(tmp_path):
    report = build_report("rules_baseline", make_frame(), CAPACITIES)
    path = write_report(report, tmp_path / "metrics")

    assert path.name == "rules_baseline.json"
    assert json.loads(path.read_text())["name"] == "rules_baseline"


def test_written_json_is_indented_and_newline_terminated(tmp_path):
    """reports/ is tracked; a single-line blob diffs uselessly."""
    text = write_report(build_report("run", make_frame(), CAPACITIES), tmp_path).read_text()
    assert text.endswith("\n")
    assert "\n  " in text
