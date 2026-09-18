"""Tests for turning a request into the matrix the booster consumes.

Two tiers, as `docs/serving.md` §8 registers them. Everything above the gate runs
anywhere: the derivation of what a request must carry, the coercions that happen before
the fitted families read a value, and the tier-3 default. The gate itself needs the
shipped artifacts and the built matrices, and is skipped — loudly, with a reason — where
they are absent.
"""

from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
import pytest

from fraud_engine.data.load import DEFAULT_CONFIG_PATH, load_config
from fraud_engine.features import encoders, velocity
from fraud_engine.models.train import prepare_matrices
from fraud_engine.serving.artifacts import load_model, load_tables
from fraud_engine.serving.fast import build_layout, row
from fraud_engine.serving.transform import (
    fill_history,
    history_supplied,
    prepare_inputs,
    raw_inputs,
    transform,
)

LOAD_CFG = {"amount_dtype": "float64", "default_float_dtype": "float32"}
VELOCITY_CFG = {"first_seen_gap_days": 30}
SECONDS_PER_DAY = 86_400

# A model holding one column of each kind: a passthrough, a categorical, a column the
# amount family builds, one the frequency family builds, and one derived from the clock.
COLUMNS = ("TransactionAmt", "ProductCD", "card1", "has_identity", "C1", "hour", "freq_card1")

VOCABULARY = {"ProductCD": pd.Index(["W", "C", "__other__", "__missing__"], dtype=object)}


def request_frame(**overrides) -> pd.DataFrame:
    """One request, with everything the transform refuses to proceed without."""
    row = {
        "TransactionDT": 86_400 * 3 + 3_600 * 5,
        "TransactionAmt": 300.0,
        "ProductCD": "W",
        "has_identity": True,
        "card1": 13926.0,
        "C1": 2.0,
    }
    row.update(overrides)
    return pd.DataFrame([row])


# ---- what a request has to carry ---------------------------------------------


def test_the_inputs_are_derived_from_the_model_rather_than_listed():
    inputs = raw_inputs(COLUMNS)

    assert "card1" in inputs, "a column no family builds has to arrive with the request"
    assert "freq_card1" not in inputs, "the frequency family builds this one"
    assert "hour" not in inputs, "derived from the timestamp, never carried"
    assert "TransactionDT" in inputs, "and that is what it is derived from"


def test_the_whole_v_block_is_required_though_none_of_it_is_a_feature():
    """The reduction consumes 339 columns and emits 234 under a prefix of its own."""
    inputs = raw_inputs(COLUMNS)

    assert "V1" in inputs and "V339" in inputs
    assert not any(name.startswith("vb_") for name in inputs)


def test_a_request_missing_something_nothing_can_be_decided_without_is_refused():
    incomplete = request_frame().drop(columns=["TransactionAmt"])

    with pytest.raises(ValueError, match="TransactionAmt"):
        prepare_inputs(incomplete, VOCABULARY, LOAD_CFG, COLUMNS)


def test_a_null_that_would_be_read_as_a_value_is_refused():
    """`astype("bool")` turns a null `has_identity` into True, which is a claim."""
    with pytest.raises(ValueError, match="has_identity"):
        prepare_inputs(request_frame(has_identity=None), VOCABULARY, LOAD_CFG, COLUMNS)


def test_a_product_that_did_not_arrive_is_allowed_through():
    """Null is a level the vocabulary holds, unlike the three inputs above."""
    prepared = prepare_inputs(request_frame(ProductCD=None), VOCABULARY, LOAD_CFG, COLUMNS)

    assert prepared["ProductCD"].isna().all()


def test_an_omitted_inherited_column_becomes_a_null_rather_than_an_error():
    """The model was fitted on a matrix where these were null on most rows."""
    prepared = prepare_inputs(
        request_frame().drop(columns=["card1"]), VOCABULARY, LOAD_CFG, COLUMNS
    )

    assert prepared["V200"].isna().all(), "the V block is never carried by this request"
    assert prepared["card1"].isna().all(), "and this one was dropped from it"


def test_anything_the_model_does_not_read_is_dropped():
    """A caller's own `hour` cannot disagree with the timestamp it was derived from."""
    prepared = prepare_inputs(request_frame(hour=23, isFraud=1), VOCABULARY, LOAD_CFG, COLUMNS)

    assert "hour" not in prepared.columns
    assert "isFraud" not in prepared.columns


# ---- the coercions the families depend on ------------------------------------


def test_the_timestamp_is_typed_so_the_clock_columns_are_integers():
    prepared = prepare_inputs(request_frame(), VOCABULARY, LOAD_CFG, COLUMNS)

    assert prepared["TransactionDT"].dtype == "int32"


def test_a_number_sent_as_an_integer_encodes_as_the_pipeline_encoded_it():
    """The frequency tables are keyed by `str(value)`, so 13926 and 13926.0 are two keys.

    This is the failure the coercion exists to prevent: an integer `card1` would miss
    every row of the table and be scored as a level the training window never saw — the
    smallest value in the column, and no error anywhere.
    """
    table = {"card1": pd.Series({"13926.0": 0.004, "15066.0": 0.002}, name="frequency")}

    as_integer = prepare_inputs(request_frame(card1=13926), VOCABULARY, LOAD_CFG, COLUMNS)
    as_float = prepare_inputs(request_frame(card1=13926.0), VOCABULARY, LOAD_CFG, COLUMNS)

    encoded = [
        encoders.apply_frequencies(frame, table)["freq_card1"].iloc[0]
        for frame in (as_integer, as_float)
    ]
    assert encoded[0] == encoded[1] == pytest.approx(0.004)


# ---- the tier-3 default ------------------------------------------------------


def test_a_card_with_no_history_is_scored_as_one_nobody_has_seen():
    filled = fill_history(request_frame(), VELOCITY_CFG)

    for window in ("1h", "24h", "7d"):
        assert filled[f"vel_n{window}_card1"].iloc[0] == 1.0, (
            "the trailing window counts the transaction itself, so a first sighting is 1"
        )
    assert filled[velocity.RECENCY].iloc[0] == pytest.approx(
        np.float32(np.log1p(VELOCITY_CFG["first_seen_gap_days"] * SECONDS_PER_DAY))
    )


def test_history_a_caller_supplies_is_kept():
    """serving.md §2 accepts tier-3 state rather than forbidding it."""
    supplied = request_frame()
    supplied["vel_n24h_card1"] = 7.0

    filled = fill_history(supplied, VELOCITY_CFG)

    assert filled["vel_n24h_card1"].iloc[0] == 7.0
    assert filled["vel_n1h_card1"].iloc[0] == 1.0


def test_the_velocity_columns_are_typed_as_the_family_types_them():
    filled = fill_history(request_frame(), VELOCITY_CFG)

    assert all(filled[column].dtype == "float32" for column in velocity.COLUMNS)


def test_which_history_arrived_is_reported_rather_than_inferred():
    supplied = request_frame()
    supplied["vel_recency_card1"] = 12.0

    assert history_supplied(supplied) == (velocity.RECENCY,)
    assert history_supplied(request_frame()) == ()


# ---- the gate ----------------------------------------------------------------

CONFIG = load_config(DEFAULT_CONFIG_PATH)
PATHS = CONFIG["paths"]
GATE_SPLIT, GATE_ROWS, GATE_SEED = "test", 300, 0

REQUIRED_ARTIFACTS = (
    PATHS["model"],
    PATHS["categories"],
    PATHS["medians"],
    PATHS["encoders"],
    PATHS["amount_stats"],
    PATHS["vblock"],
    PATHS["interim"],
    f"{PATHS['features_dir']}/train.parquet",
    f"{PATHS['features_dir']}/{GATE_SPLIT}.parquet",
)

gated = pytest.mark.skipif(
    not all(Path(path).exists() for path in REQUIRED_ARTIFACTS),
    reason=(
        "the shipped artifacts and built matrices are not in this checkout; "
        "docs/serving.md §8 registers this tier as artifact-gated"
    ),
)


@pytest.fixture(scope="module")
def gate() -> dict:
    """The transform's output beside the training path's, for the same transactions.

    Compared against `prepare_matrices` rather than against the parquet on disk: the
    matrices are written before the vocabulary is refitted and the medians applied, so
    the file is an earlier object than the one the booster was ever shown.

    Tier-3 columns are supplied from the matrix, so what is measured here is the
    transform and not §2's default — that default is measured on VAL-CAL, in its own
    stage, and never against the booster's own training inputs.
    """
    booster = lgb.Booster(model_file=PATHS["model"])
    columns = booster.feature_name()

    prepared, vocabulary, _ = prepare_matrices(
        PATHS["features_dir"], CONFIG["model"], splits=("train", GATE_SPLIT)
    )
    sample = (
        prepared[GATE_SPLIT]
        .sample(GATE_ROWS, random_state=GATE_SEED)
        .sort_values("TransactionID")
        .reset_index(drop=True)
    )
    identifiers = sample["TransactionID"].tolist()

    raw = pd.read_parquet(PATHS["interim"], filters=[("TransactionID", "in", identifiers)])
    raw = raw.set_index("TransactionID").loc[identifiers].reset_index()
    raw = raw.merge(sample[["TransactionID", *velocity.COLUMNS]], on="TransactionID", how="left")

    tables = load_tables(PATHS, CONFIG["model"]["impute"])
    built = transform(raw, tables, CONFIG["load"], CONFIG["features"], columns)

    # The same transactions through the fast path, and the reference as the numbers a
    # booster actually receives: a categorical reaches it as its code, not its level.
    model = load_model(PATHS, CONFIG["model"]["impute"])
    layout = build_layout(model, CONFIG["features"])
    requests = [
        {name: (None if pd.isna(value) else value) for name, value in raw.iloc[position].items()}
        for position in range(len(raw))
    ]

    coded = built.copy()
    for name in tables.vocabulary:
        coded[name] = coded[name].cat.codes

    return {
        "booster": booster,
        "columns": columns,
        "built": built,
        "expected": sample[columns],
        "shipped_vocabulary": tables.vocabulary,
        "refit_vocabulary": vocabulary,
        "reference_values": coded.to_numpy(dtype="float64"),
        "fast": np.vstack(
            [row(values, model, layout, CONFIG["load"], CONFIG["features"]) for values in requests]
        ),
    }


@pytest.mark.artifacts
@gated
def test_the_transform_reproduces_the_training_matrix_exactly(gate):
    """The gate. Not "close" — every column, every value, every dtype."""
    pd.testing.assert_frame_equal(
        gate["built"].reset_index(drop=True),
        gate["expected"].reset_index(drop=True),
        check_exact=True,
        check_dtype=True,
        check_categorical=True,
    )


@pytest.mark.artifacts
@gated
def test_the_booster_scores_a_transformed_request_identically(gate):
    """What the matrices prove about columns, this proves about the number served."""
    booster = gate["booster"]

    assert np.array_equal(booster.predict(gate["built"]), booster.predict(gate["expected"]))


@pytest.mark.artifacts
@gated
def test_the_shipped_vocabulary_is_the_one_the_training_path_refits(gate):
    """§8's named risk, measured against the model actually shipped.

    Every other stage refits the vocabulary from train; serving reads the file. If the
    two ever disagree, the codes a served request produces stand for different levels
    than the codes the booster's splits were fitted on.
    """
    shipped, refit = gate["shipped_vocabulary"], gate["refit_vocabulary"]

    assert set(shipped) == set(refit)
    for column, levels in refit.items():
        assert list(shipped[column]) == list(levels)
        assert pd.CategoricalDtype(shipped[column]) == pd.CategoricalDtype(levels)


@pytest.mark.artifacts
@gated
def test_the_fast_path_assembles_the_reference_row(gate):
    """§5's condition, and the only thing that keeps the amendment honest.

    Cell for cell, not scores alone: a value rounded differently survives most splits and
    crosses one eventually, and the run where it crosses is not the run you want to find
    out on.
    """
    np.testing.assert_array_equal(gate["fast"], gate["reference_values"])


@pytest.mark.artifacts
@gated
def test_the_fast_path_scores_what_the_reference_scores(gate):
    booster = gate["booster"]

    assert np.array_equal(booster.predict(gate["fast"]), booster.predict(gate["built"]))
