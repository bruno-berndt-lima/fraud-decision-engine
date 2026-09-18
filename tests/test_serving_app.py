"""Tests for the API: the contract, the decision it returns, and the degraded mode.

Two tiers, as `docs/serving.md` §8 registers. The tier that runs anywhere builds a whole
deployment in miniature — the real feature families, the real writers, and a booster of a
few trees over synthetic transactions — so the wiring is exercised without the 77 MB
artifact a clean checkout does not have. The gated tier answers the only question the
miniature cannot: whether the service returns what the shipped model says.
"""

import json
import time
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
import pytest
from starlette.testclient import TestClient

from fraud_engine.data.load import DEFAULT_CONFIG_PATH, add_time_columns, load_config
from fraud_engine.features import aggregations, encoders, vblock
from fraud_engine.features.build import build_features, order_by_time
from fraud_engine.models.calibrate import apply_calibrator, fit_calibrator
from fraud_engine.models.rules import build_rules, fit, write_constants
from fraud_engine.models.train import (
    apply_categories,
    apply_medians,
    feature_columns,
    fit_categories,
    fit_medians,
    write_categories,
    write_medians,
)
from fraud_engine.serving import app as serving_app
from fraud_engine.serving.app import create_app
from fraud_engine.serving.artifacts import load_model, load_tables
from fraud_engine.serving.fast import build_layout, row
from fraud_engine.serving.transform import raw_inputs, transform

ROWS = 240
SEED = 0
CLIP = 1.0e-15

CONFIG = load_config(DEFAULT_CONFIG_PATH)
CATEGORICALS = ("ProductCD", "card4", "card6", "DeviceType", "DeviceInfo", "P_emaildomain", "M4")


def synthetic_transactions() -> pd.DataFrame:
    """An interim table in miniature, carrying every column the families read."""
    generator = np.random.default_rng(SEED)
    index = np.arange(ROWS)

    frame = pd.DataFrame(
        {
            "TransactionID": 1_000_000 + index,
            "TransactionDT": (86_400 + index * 900).astype("int32"),
            "TransactionAmt": generator.choice([59.95, 100.0, 300.0, 450.5], ROWS),
            "ProductCD": generator.choice(["W", "C", "H"], ROWS),
            "card1": generator.choice([13926.0, 15066.0, 17188.0], ROWS),
            "card2": generator.choice([111.0, 222.0], ROWS),
            "card3": np.full(ROWS, 150.0),
            "card4": generator.choice(["visa", "mastercard"], ROWS),
            "card5": generator.choice([226.0, 224.0], ROWS),
            "card6": generator.choice(["debit", "credit"], ROWS),
            "addr1": generator.choice([299.0, 325.0], ROWS),
            "addr2": np.full(ROWS, 87.0),
            "P_emaildomain": generator.choice(["gmail.com", "yahoo.com"], ROWS),
            "DeviceType": generator.choice(["desktop", "mobile"], ROWS),
            "DeviceInfo": generator.choice(["Windows", "iOS"], ROWS),
            "M4": generator.choice(["M0", "M2"], ROWS),
            "C1": generator.integers(1, 20, ROWS).astype("float32"),
            "D1": generator.integers(0, 400, ROWS).astype("float32"),
            "has_identity": generator.random(ROWS) > 0.4,
            "isFraud": (generator.random(ROWS) > 0.7).astype("int8"),
            # Two thirds train, the rest scored — the families fit on the first alone.
            "split": np.where(index < ROWS * 2 // 3, "train", "val_fit"),
        }
    )

    block = {}
    for position, name in enumerate(vblock.V_COLUMNS):
        values = generator.normal(size=ROWS).astype("float32")
        if position % 3:
            values[: ROWS // (2 + position % 3)] = np.nan
        block[name] = values

    frame = pd.concat([frame, pd.DataFrame(block)], axis=1)
    # `day`, `hour` and `weekday` arrive at load time in the real pipeline, so they are
    # here too: `day` is the split axis every matrix carries and the model may not see.
    return add_time_columns(frame.astype(dict.fromkeys(CATEGORICALS, "category")))


@pytest.fixture(scope="module")
def deployment(tmp_path_factory) -> dict:
    """Every artifact the service loads, produced by the code that produces the real ones.

    Nothing here is a stand-in for a writer: the parquets are written by `write_tables`,
    the vocabulary by `write_categories`, the constants by `write_constants`. Only the
    transactions are invented, and only the booster is small.
    """
    directory = tmp_path_factory.mktemp("deployment")
    raw = synthetic_transactions()

    built, fitted = build_features(order_by_time(raw), CONFIG["features"])

    encoders.write_tables(fitted["encoders"], directory / "encoders.parquet")
    aggregations.write_tables(fitted["amount_stats"], directory / "amount_stats.parquet")
    vblock.write_tables(fitted["vblock"], directory / "vblock.parquet")

    train = built[built["split"] == "train"]
    vocabulary = fit_categories(train, CONFIG["model"]["min_category_rows"])
    prepared = apply_categories(built.drop(columns="split"), vocabulary)

    columns = feature_columns(prepared)
    medians = fit_medians(apply_categories(train.drop(columns="split"), vocabulary), columns)
    prepared = apply_medians(prepared, medians)

    write_categories(vocabulary, directory / "categories.parquet")
    write_medians(medians, directory / "medians.parquet")

    fitting = prepared[prepared["TransactionID"].isin(train["TransactionID"])]
    booster = lgb.train(
        {"objective": "binary", "num_leaves": 4, "min_data_in_leaf": 5, "verbosity": -1},
        lgb.Dataset(fitting[columns], label=fitting["isFraud"]),
        num_boost_round=8,
    )
    booster.save_model(str(directory / "model.txt"))

    scored = booster.predict(fitting[columns])
    calibrator = fit_calibrator("platt", scored, fitting["isFraud"].to_numpy(), CLIP)
    (directory / "calibrator.json").write_text(json.dumps(calibrator))

    rules_cfg = CONFIG["baselines"]["rules"]
    write_constants(fit(raw[raw["split"] == "train"], rules_cfg), directory / "rules.json")

    paths = dict(CONFIG["paths"]) | {
        name: str(directory / f"{name.replace('rules_constants', 'rules')}.parquet")
        for name in ("categories", "medians", "encoders", "amount_stats", "vblock")
    }
    paths |= {
        "model": str(directory / "model.txt"),
        "calibrator": str(directory / "calibrator.json"),
        "rules_constants": str(directory / "rules.json"),
    }

    return {
        "config": dict(CONFIG) | {"paths": paths},
        "directory": directory,
        "raw": raw,
        "booster": booster,
        "columns": columns,
        "calibrator": calibrator,
        "rules": build_rules(fit(raw[raw["split"] == "train"], rules_cfg)),
    }


@pytest.fixture(scope="module")
def client(deployment) -> TestClient:
    return TestClient(create_app(deployment["config"]))


def request_body(deployment, row: int = 0) -> dict:
    """One synthetic transaction as a caller would send it."""
    raw = deployment["raw"].iloc[[row]]
    carried = [name for name in raw_inputs(deployment["columns"]) if name in raw.columns]

    body = {
        name: (None if pd.isna(value) else value)
        for name, value in raw[carried].astype("object").iloc[0].items()
    }
    return {
        name: (bool(value) if name == "has_identity" else value) for name, value in body.items()
    }


# ---- health ------------------------------------------------------------------


def test_health_reports_what_is_loaded(client, deployment):
    body = client.get("/health").json()

    assert body["status"] == "ok"
    assert body["mode"] == "model"
    assert body["model_loaded"] and body["fallback_loaded"]
    assert body["features"] == len(deployment["columns"])
    assert body["trees"] == deployment["booster"].num_trees()
    assert body["calibration"] == "platt"


def test_health_identifies_the_artifacts_it_is_holding(client):
    """A decision has to be traceable to the objects behind it (§5)."""
    artifacts = client.get("/health").json()["artifacts"]

    assert artifacts["model"] != "absent"
    assert artifacts["calibrator"] != "absent"
    assert len(set(artifacts.values())) > 1, "identical stamps would identify nothing"


# ---- scoring -----------------------------------------------------------------


def test_a_well_formed_request_is_decided(client, deployment):
    body = client.post("/score", json=request_body(deployment)).json()

    assert body["decision"] in {"allow", "block"}
    assert 0.0 <= body["probability"] <= 1.0
    assert 0.0 < body["break_even"] < 1.0
    assert body["mode"] == "model"


def test_the_probability_served_is_the_one_the_offline_path_computes(client, deployment):
    """Transform, predict, calibrate — the endpoint composes them and adds nothing.

    Exactly, not closely: a served probability differing in the last bits from the one the
    pipeline computes is a served probability nobody can reproduce.
    """
    body = request_body(deployment)
    served = client.post("/score", json=body).json()["probability"]

    features = transform(
        pd.DataFrame([body]),
        load_tables(deployment["config"]["paths"], CONFIG["model"]["impute"]),
        CONFIG["load"],
        CONFIG["features"],
        deployment["columns"],
    )
    offline = apply_calibrator(deployment["calibrator"], deployment["booster"].predict(features))

    assert served == float(offline[0])


def test_the_calibrator_is_not_skipped(client, deployment):
    """decision-policy.md §1: serving without it returns raw scores and raises nothing."""
    body = request_body(deployment)
    served = client.post("/score", json=body).json()["probability"]

    features = transform(
        pd.DataFrame([body]),
        load_tables(deployment["config"]["paths"], CONFIG["model"]["impute"]),
        CONFIG["load"],
        CONFIG["features"],
        deployment["columns"],
    )

    assert served != float(deployment["booster"].predict(features)[0])


def test_a_request_declares_what_it_was_decided_on(client, deployment):
    body = client.post("/score", json=request_body(deployment)).json()

    assert body["inherited_present"] > 0
    assert body["inherited_expected"] >= body["inherited_present"]
    assert body["history_supplied"] == [], "no store is attached; §2's default applied"


def test_an_unknown_field_is_refused(client, deployment):
    """§1: if omission is permitted, a misspelled field is an omitted one."""
    body = request_body(deployment) | {"C_13": 4.0}

    assert client.post("/score", json=body).status_code == 422


def test_a_missing_required_field_is_refused(client, deployment):
    body = request_body(deployment)
    del body["TransactionAmt"]

    assert client.post("/score", json=body).status_code == 422


def test_a_negative_amount_is_refused_by_the_contract(client, deployment):
    """Caught here rather than by the cost model: a 422 names the field, a 500 does not."""
    body = request_body(deployment) | {"TransactionAmt": -10.0}

    assert client.post("/score", json=body).status_code == 422


def test_an_omitted_inherited_column_is_scored_rather_than_refused(client, deployment):
    body = {name: value for name, value in request_body(deployment).items() if name != "V200"}

    answer = client.post("/score", json=body)
    assert answer.status_code == 200
    assert answer.json()["inherited_expected"] > answer.json()["inherited_present"]


# ---- fail open ---------------------------------------------------------------
# §4's three arms. The first — an artifact that never loaded — is the degraded mode
# below. These two are the ones that happen while the service is up and healthy.


def test_a_model_that_raises_does_not_take_the_transaction_down(client, deployment, monkeypatch):
    """A transaction declined by an exception is the outage fail-open exists to prevent."""

    def fell_over(*arguments, **keywords):
        raise RuntimeError("the booster fell over mid-request")

    monkeypatch.setattr(serving_app, "decide", fell_over)
    answer = client.post("/score", json=request_body(deployment))

    assert answer.status_code == 200
    assert answer.json()["mode"] == "rules"


def test_a_decision_that_misses_the_budget_is_not_waited_for(client, deployment, monkeypatch):
    """The caller is bounded; the work is not — it finishes in its thread and is dropped."""
    budget = client.get("/health").json()["budget_ms"]

    def slowly(*arguments, **keywords):
        time.sleep(budget / 1000 * 3)
        raise AssertionError("this result should have been abandoned")

    monkeypatch.setattr(serving_app, "decide", slowly)

    began = time.perf_counter()
    answer = client.post("/score", json=request_body(deployment))
    elapsed = (time.perf_counter() - began) * 1000

    assert answer.status_code == 200
    assert answer.json()["mode"] == "rules"
    assert elapsed < budget * 3, "the request waited for work it had already given up on"


def test_falling_back_is_counted_even_though_the_response_looks_ordinary(
    client, deployment, monkeypatch
):
    """A service answering every request from the incumbent looks healthy from outside."""
    before = client.get("/health").json()["fallback_decisions"]

    def fell_over(*arguments, **keywords):
        raise RuntimeError("again")

    monkeypatch.setattr(serving_app, "decide", fell_over)
    client.post("/score", json=request_body(deployment))

    assert client.get("/health").json()["fallback_decisions"] == before + 1


def test_a_disarmed_service_lets_the_failure_surface(deployment, monkeypatch):
    """`fail_open: false` is for a test that wants to see the break, never for a deployment."""
    config = dict(deployment["config"])
    config["serving"] = dict(config["serving"]) | {"fail_open": False}
    client = TestClient(create_app(config), raise_server_exceptions=False)

    def fell_over(*arguments, **keywords):
        raise RuntimeError("no net")

    monkeypatch.setattr(serving_app, "decide", fell_over)

    assert client.post("/score", json=request_body(deployment)).status_code == 500


# ---- the degraded mode -------------------------------------------------------


@pytest.fixture(scope="module")
def degraded(deployment) -> TestClient:
    """A deployment whose booster is not where it should be — §4's failure, not a crash."""
    config = dict(deployment["config"])
    config["paths"] = dict(config["paths"]) | {"model": str(deployment["directory"] / "gone.txt")}

    return TestClient(create_app(config))


def test_a_service_without_its_model_still_starts(degraded):
    body = degraded.get("/health").json()

    assert body["status"] == "degraded"
    assert body["mode"] == "rules"
    assert body["model_loaded"] is False
    assert body["fallback_loaded"] is True
    assert body["artifacts"]["model"] == "absent"


def test_the_incumbent_decides_while_the_model_is_gone(degraded, deployment):
    body = degraded.post("/score", json=request_body(deployment)).json()

    assert body["mode"] == "rules"
    assert body["decision"] == "allow", "the engine never blocks (decision-policy.md §3)"
    assert body["probability"] is None, "points are not a frequency"
    assert body["review_eligible"] is True
    assert body["break_even"] > 0, "a property of the amount, not of the model"


def test_the_degraded_contract_does_not_refuse_the_record_it_cannot_read(degraded, deployment):
    """Refusing unknown names here would decline traffic the fallback can still decide."""
    body = request_body(deployment) | {"C_13": 4.0}

    assert degraded.post("/score", json=body).status_code == 200


def test_a_service_holding_nothing_says_so_rather_than_inventing_a_number(deployment):
    config = dict(deployment["config"])
    config["paths"] = dict(config["paths"]) | {
        "model": str(deployment["directory"] / "gone.txt"),
        "rules_constants": str(deployment["directory"] / "gone.json"),
    }
    client = TestClient(create_app(config))

    assert client.get("/health").json()["fallback_loaded"] is False
    assert client.post("/score", json=request_body(deployment)).status_code == 503


def test_the_fast_path_assembles_the_reference_row(deployment):
    """The §5 proof, on a model small enough to carry in the repository.

    The gate makes this comparison on the shipped booster and real transactions; that tier
    is skipped where the artifacts are absent, so the same equality is checked here on the
    miniature deployment, which runs anywhere.
    """
    paths = deployment["config"]["paths"]
    model = load_model(paths, CONFIG["model"]["impute"])
    layout = build_layout(model, CONFIG["features"])

    bodies = [request_body(deployment, position) for position in range(6)]
    fast = np.vstack(
        [row(values, model, layout, CONFIG["load"], CONFIG["features"]) for values in bodies]
    )

    reference = transform(
        pd.DataFrame(bodies),
        load_tables(paths, CONFIG["model"]["impute"]),
        CONFIG["load"],
        CONFIG["features"],
        model.columns,
    )
    for name in model.tables.vocabulary:
        reference[name] = reference[name].cat.codes

    np.testing.assert_array_equal(fast, reference.to_numpy(dtype="float64"))


# ---- against the shipped artifacts -------------------------------------------

REQUIRED_ARTIFACTS = tuple(
    CONFIG["paths"][name]
    for name in (
        "model",
        "categories",
        "medians",
        "encoders",
        "amount_stats",
        "vblock",
        "rules_constants",
        "interim",
    )
)

gated = pytest.mark.skipif(
    not all(Path(path).exists() for path in REQUIRED_ARTIFACTS),
    reason=(
        "the shipped artifacts are not in this checkout; "
        "docs/serving.md section 8 registers this tier as artifact-gated"
    ),
)


def json_safe(row: pd.Series) -> dict:
    """One row as a caller would send it: nulls for gaps, plain Python for numbers."""
    body = {}
    for name, value in row.items():
        if pd.isna(value):
            body[name] = None
        elif name == "has_identity":
            body[name] = bool(value)
        else:
            body[name] = value.item() if hasattr(value, "item") else value
    return body


@pytest.fixture(scope="module")
def shipped() -> dict:
    """The real service, and one real transaction to ask it about."""
    booster = lgb.Booster(model_file=CONFIG["paths"]["model"])
    raw = pd.read_parquet(CONFIG["paths"]["interim"]).head(1)
    carried = [name for name in raw_inputs(booster.feature_name()) if name in raw.columns]

    return {
        "client": TestClient(create_app(CONFIG)),
        "booster": booster,
        "body": json_safe(raw[carried].iloc[0]),
    }


@pytest.mark.artifacts
@gated
def test_the_shipped_service_reports_the_model_it_is_holding(shipped):
    body = shipped["client"].get("/health").json()

    assert body["status"] == "ok"
    assert body["features"] == len(shipped["booster"].feature_name())
    assert body["trees"] == shipped["booster"].num_trees()
    assert body["calibration"] == "platt"


@pytest.mark.artifacts
@gated
def test_the_shipped_service_decides_a_real_transaction(shipped):
    """End to end on the objects that ship: a request in, a priced action out."""
    answer = shipped["client"].post("/score", json=shipped["body"])
    assert answer.status_code == 200, answer.json()

    body = answer.json()
    assert body["decision"] in {"allow", "block"}
    assert body["mode"] == "model"
    assert 0 < body["inherited_present"] <= body["inherited_expected"]
    assert body["cost_matrix_version"] == 1


@pytest.mark.artifacts
@gated
def test_a_request_that_stopped_carrying_the_block_says_so(shipped):
    """§1's guard: the integration failure that would otherwise be invisible.

    Nothing refuses the second request — an omitted field is a null, and the model has
    mass on it. What changes is the number a monitor watches.
    """
    full = shipped["client"].post("/score", json=shipped["body"]).json()

    without = {name: value for name, value in shipped["body"].items() if not name.startswith("V")}
    stripped = shipped["client"].post("/score", json=without).json()

    assert stripped["inherited_present"] < full["inherited_present"]
    assert stripped["inherited_expected"] == full["inherited_expected"]
