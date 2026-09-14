"""Tests for the leave-one-out ablation: what each arm removes, and what it reports.

Boosters on a few hundred synthetic rows with a planted signal. The score is
never the point — what is pinned is that an arm removes exactly what it claims,
that the reference is what deltas are measured from, and that the calibration
slice is unreachable from here.

The drop guard gets the most attention. Its failure mode is an arm that removes
nothing, scores as the reference, and reports a delta of zero — which reads as
*this family contributes nothing*, the most plausible-looking wrong answer this
experiment can produce.
"""

import json
from pathlib import Path

import mlflow
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from conftest import children
from fraud_engine.models import ablation
from fraud_engine.models.ablation import (
    BARE,
    REFERENCE,
    REPRODUCIBLE,
    all_arms,
    drop_columns,
    fit_without,
    measure,
    resolve_arms,
    resolve_tier_arms,
)
from fraud_engine.models.train import apply_categories, fit_categories

CAPACITIES = [0.1]
MODEL_CFG = {"tuned": {}, "seed": 0, "early_stopping_rounds": 5, "num_boost_round": 60}

FAMILY_COLUMNS = {
    "amount": ("amt_a", "amt_b"),
    "frequency": ("freq_a",),
    "velocity": ("vel_a",),
}
VB_COLUMNS = ("vb_1", "vb_2", "vb_3")
KEPT = ("signal", "noise", "brand")


def make_matrix(rows: int = 400, seed: int = 0) -> pd.DataFrame:
    """A matrix carrying every family's columns plus a planted signal."""
    rng = np.random.default_rng(seed)
    signal = rng.random(rows)

    frame = pd.DataFrame(
        {
            "TransactionID": range(rows),
            "TransactionDT": np.arange(rows) * 60,
            "isFraud": (signal > 0.8).astype(int),
            "day": 1,
            "signal": signal,
            "noise": rng.random(rows),
            "brand": pd.Categorical(rng.choice(["visa", "amex"], rows)),
        }
    )
    for column in (*VB_COLUMNS, *(c for cols in FAMILY_COLUMNS.values() for c in cols)):
        frame[column] = rng.random(rows)

    return frame


@pytest.fixture
def matrices() -> dict[str, pd.DataFrame]:
    raw = {name: make_matrix(seed=index) for index, name in enumerate(("train", "val_fit"))}
    vocabulary = fit_categories(raw["train"], min_rows=1)
    return {name: apply_categories(frame, vocabulary) for name, frame in raw.items()}


@pytest.fixture
def features_dir(tmp_path: Path) -> Path:
    """A directory whose train matrix carries the columns the registry names."""
    pq.write_table(
        pa.Table.from_pandas(make_matrix(rows=1), preserve_index=False),
        tmp_path / "train.parquet",
    )
    return tmp_path


INHERITED = (*VB_COLUMNS, "noise")


@pytest.fixture
def registry(monkeypatch: pytest.MonkeyPatch):
    """The real family and tier names, pointed at this file's synthetic columns."""
    families = {BARE: (), **FAMILY_COLUMNS}
    monkeypatch.setattr(
        ablation,
        "resolve_families",
        lambda directory: {**families, "vblock": VB_COLUMNS},
    )
    monkeypatch.setattr(
        ablation,
        "resolve_tiers",
        lambda directory: {
            ablation.TIER_0: INHERITED,
            "tier_1": ("signal", "brand"),
            "tier_2": FAMILY_COLUMNS["frequency"],
            "tier_3": FAMILY_COLUMNS["velocity"],
            "keys": ("TransactionID",),
        },
    )


@pytest.fixture
def paths(tmp_path: Path) -> dict:
    return {"metrics_dir": str(tmp_path / "metrics"), "predictions_dir": str(tmp_path / "preds")}


# ------------------------------------------------------------------------------
# resolve_arms
# ------------------------------------------------------------------------------


def test_the_reference_comes_first_and_removes_nothing(features_dir: Path, registry):
    arms = resolve_arms(features_dir)

    assert next(iter(arms)) == REFERENCE
    assert arms[REFERENCE] == ()


def test_the_bare_probe_key_does_not_become_an_arm(features_dir: Path, registry):
    """`none` means *add nothing* and has no leave-one-out reading."""
    assert BARE not in resolve_arms(features_dir)


def test_every_family_becomes_an_arm(features_dir: Path, registry):
    arms = resolve_arms(features_dir)

    assert set(arms) == {REFERENCE, *FAMILY_COLUMNS, "vblock"}
    assert arms["amount"] == FAMILY_COLUMNS["amount"]


def test_a_bare_reference_carrying_columns_raises(
    features_dir: Path, monkeypatch: pytest.MonkeyPatch
):
    monkeypatch.setattr(ablation, "resolve_families", lambda _: {BARE: ("signal",)})

    with pytest.raises(ValueError, match="no meaning as a leave-one-out arm"):
        resolve_arms(features_dir)


def test_overlapping_families_raise(features_dir: Path, monkeypatch: pytest.MonkeyPatch):
    """Removing either arm would remove part of the other, and deltas stop adding up."""
    monkeypatch.setattr(
        ablation,
        "resolve_families",
        lambda _: {BARE: (), "amount": ("amt_a",), "frequency": ("amt_a",)},
    )

    with pytest.raises(ValueError, match="belongs to both"):
        resolve_arms(features_dir)


# ------------------------------------------------------------------------------
# drop_columns
# ------------------------------------------------------------------------------


def test_dropping_removes_exactly_the_named_columns(matrices):
    reduced = drop_columns(matrices["train"], FAMILY_COLUMNS["amount"])

    assert set(matrices["train"].columns) - set(reduced.columns) == set(FAMILY_COLUMNS["amount"])


def test_dropping_nothing_returns_every_column(matrices):
    assert list(drop_columns(matrices["train"], ()).columns) == list(matrices["train"].columns)


def test_the_input_frame_is_not_modified(matrices):
    before = list(matrices["train"].columns)

    drop_columns(matrices["train"], FAMILY_COLUMNS["amount"])

    assert list(matrices["train"].columns) == before


def test_a_stale_column_name_raises_rather_than_removing_nothing(matrices):
    """The guard this function exists for.

    Left alone, a renamed column would make the arm identical to the reference
    and its delta exactly zero — indistinguishable from a family that adds
    nothing.
    """
    with pytest.raises(ValueError, match="would remove nothing"):
        drop_columns(matrices["train"], ("amt_a", "amt_renamed"))


# ------------------------------------------------------------------------------
# fit_without
# ------------------------------------------------------------------------------


def test_fitting_without_columns_narrows_the_feature_count(matrices):
    _, _, wide = fit_without(matrices, (), MODEL_CFG)
    _, _, narrow = fit_without(matrices, FAMILY_COLUMNS["amount"], MODEL_CFG)

    assert wide - narrow == len(FAMILY_COLUMNS["amount"])


def test_only_val_fit_is_scored(matrices):
    """The calibration slice is not reachable from an arm, by construction."""
    scored, _, _ = fit_without(matrices, (), MODEL_CFG)

    assert set(scored["split"].unique()) == {"val_fit"}
    assert len(scored) == len(matrices["val_fit"])


def test_the_same_arm_returns_the_same_number(matrices):
    """The untuned configuration samples nothing, so a delta carries no seed noise.

    The whole reason detection runs on this instrument rather than the shipped
    one — if this stopped holding, every delta would need a spread behind it.
    """
    first, first_iteration, _ = fit_without(matrices, FAMILY_COLUMNS["amount"], MODEL_CFG)
    second, second_iteration, _ = fit_without(matrices, FAMILY_COLUMNS["amount"], MODEL_CFG)

    assert first_iteration == second_iteration
    pd.testing.assert_series_equal(first["score"], second["score"])


def test_removing_the_signal_costs_more_than_removing_noise(matrices):
    """A sanity check on direction: the planted column is the one that matters."""
    without_signal, _, _ = fit_without(matrices, ("signal",), MODEL_CFG)
    without_noise, _, _ = fit_without(matrices, ("noise",), MODEL_CFG)

    labels = matrices["val_fit"]["isFraud"].to_numpy()
    assert without_signal["score"].corr(pd.Series(labels)) < without_noise["score"].corr(
        pd.Series(labels)
    )


# ------------------------------------------------------------------------------
# measure
# ------------------------------------------------------------------------------


@pytest.fixture
def comparison(matrices, paths, experiment_run) -> pd.DataFrame:
    """Three arms, one of which must come back negative.

    `signal` is the planted column, so removing it has to cost — and a delta
    that cannot go below zero is how an unsigned subtraction hides. Without an
    arm guaranteed to lose, the sign is never actually exercised.
    """
    arms = {REFERENCE: (), "amount": FAMILY_COLUMNS["amount"], "planted": ("signal",)}
    return measure(matrices, arms, MODEL_CFG, CAPACITIES, paths)


def test_one_row_per_arm(comparison):
    assert list(comparison["arm"]) == [REFERENCE, "amount", "planted"]


def test_the_reference_delta_is_zero(comparison):
    assert comparison.loc[comparison["arm"] == REFERENCE, "delta"].item() == 0.0


def test_deltas_are_signed_against_the_reference(comparison):
    """Removing something that helped comes back negative, not merely large.

    The sign is the reading: this table is the family's contribution, not the
    arm's score, and it runs opposite to the probe's.
    """
    reference = comparison.loc[comparison["arm"] == REFERENCE, "pr_auc"].item()

    for _, row in comparison.iterrows():
        assert row["delta"] == pytest.approx(row["pr_auc"] - reference)

    assert comparison.loc[comparison["arm"] == "planted", "delta"].item() < 0


def test_removed_and_remaining_counts_agree(comparison):
    widest = comparison.loc[comparison["arm"] == REFERENCE, "n_features"].item()

    for _, row in comparison.iterrows():
        assert row["n_features"] == widest - row["removed"]


def test_a_record_is_written_per_arm(comparison, paths):
    written = sorted(path.name for path in Path(paths["metrics_dir"]).glob("*.json"))

    assert written == ["ablation_amount.json", "ablation_full.json", "ablation_planted.json"]


def test_no_record_carries_the_calibration_slice(comparison, paths):
    for path in Path(paths["metrics_dir"]).glob("*.json"):
        assert list(json.loads(path.read_text())["splits"]) == ["val_fit"]


def test_arms_without_a_reference_raise(matrices, paths):
    with pytest.raises(KeyError, match="every delta is measured against it"):
        measure(matrices, {"amount": FAMILY_COLUMNS["amount"]}, MODEL_CFG, CAPACITIES, paths)


# ------------------------------------------------------------------------------
# resolve_tier_arms and all_arms
# ------------------------------------------------------------------------------


def test_the_reproducible_arm_removes_the_inherited_tier(features_dir: Path, registry):
    assert resolve_tier_arms(features_dir) == {REPRODUCIBLE: INHERITED}


def test_the_live_entity_tier_is_kept(features_dir: Path, registry):
    """Expensive to serve is not impossible to build, and E7 asks the second.

    Dropping tier 3 here would answer the serving question a second time under
    the rebuild question's name — the arm would look right and mean something
    else.
    """
    removed = set(resolve_tier_arms(features_dir)[REPRODUCIBLE])

    assert not removed & set(FAMILY_COLUMNS["velocity"])


def test_the_serving_arm_is_not_duplicated(features_dir: Path, registry):
    """E3's arm is the velocity family; naming it again writes a second, identical record."""
    assert set(resolve_tier_arms(features_dir)) == {REPRODUCIBLE}


def test_every_arm_shares_one_reference(features_dir: Path, registry):
    """Families and tiers are merged rather than run twice.

    A delta is comparable only to another measured from the same reference fit.
    """
    arms = all_arms(features_dir)

    assert next(iter(arms)) == REFERENCE
    assert set(arms) == {REFERENCE, *FAMILY_COLUMNS, "vblock", REPRODUCIBLE}


def test_a_name_shared_by_a_tier_and_a_family_raises(
    features_dir: Path, registry, monkeypatch: pytest.MonkeyPatch
):
    """They index one dict, so a collision drops an arm without saying so."""
    monkeypatch.setattr(ablation, "REPRODUCIBLE", "amount")

    with pytest.raises(ValueError, match="share a name"):
        all_arms(features_dir)


# ------------------------------------------------------------------------------
# what reaches MLflow
# ------------------------------------------------------------------------------


def test_every_arm_is_a_child_run(comparison, experiment_run):
    assert [run.info.run_name for run in children(experiment_run)] == [
        "ablation_full",
        "ablation_amount",
        "ablation_planted",
    ]


def test_each_child_records_the_columns_it_trained_without(comparison, experiment_run):
    """The commit cannot say this: the V-block's members come from a fitted threshold."""
    by_name = {run.info.run_name: run for run in children(experiment_run)}
    run = by_name["ablation_amount"]

    removed = mlflow.artifacts.load_dict(f"{run.info.artifact_uri}/removed_columns.json")

    assert removed == list(FAMILY_COLUMNS["amount"])
    assert run.data.params["removed"] == str(len(FAMILY_COLUMNS["amount"]))


def test_each_child_carries_the_record_and_its_provenance(comparison, experiment_run, paths):
    run = {r.info.run_name: r for r in children(experiment_run)}["ablation_full"]
    record = json.loads((Path(paths["metrics_dir"]) / "ablation_full.json").read_text())

    assert run.data.metrics["val_fit.pr_auc"] == record["splits"]["val_fit"]["pr_auc"]
    assert "git_revision" in run.data.params
    assert run.data.params["objective"] == "binary"
