"""Fixtures shared across test modules.

Only what more than one module needs lives here. MLflow's tracking URI and its
active run are process-global, so the fixture that provides them has to be one
definition — two slightly different copies would decide, by test order, which
store the next test wrote into.

The deployment in miniature is here for a plainer reason: `docs/serving.md` §8
registers a tier that runs anywhere, and more than one module now needs the small
booster it is built around. Building it twice would double the cost of every run
and leave two fixtures free to drift into describing different services.
"""

import json
from collections.abc import Iterator
from pathlib import Path

import lightgbm as lgb
import mlflow
import numpy as np
import pandas as pd
import pytest
from starlette.testclient import TestClient

from fraud_engine.data.load import DEFAULT_CONFIG_PATH, add_time_columns, load_config
from fraud_engine.evaluation.tracking import configure_tracking
from fraud_engine.features import aggregations, encoders, vblock
from fraud_engine.features.build import build_features, order_by_time
from fraud_engine.models.calibrate import fit_calibrator
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
from fraud_engine.serving.app import create_app
from fraud_engine.serving.transform import raw_inputs

EXPERIMENT = "test-experiment"


@pytest.fixture
def experiment_run(tmp_path: Path) -> Iterator[mlflow.ActiveRun]:
    """An isolated store with a parent run open, as an experiment's `main` provides.

    Every `measure` that fits a booster logs a child run and refuses to without
    a parent. Nothing here touches the project's real `mlruns/`.
    """
    configure_tracking({"store": str(tmp_path / "mlruns"), "experiment_name": EXPERIMENT})
    with mlflow.start_run(run_name="parent") as run:
        yield run
    while mlflow.active_run() is not None:
        mlflow.end_run()


def children(parent: mlflow.ActiveRun) -> list[mlflow.entities.Run]:
    """Every run nested directly under `parent`, oldest first."""
    return mlflow.search_runs(
        experiment_names=[EXPERIMENT],
        filter_string=f"tags.mlflow.parentRunId = '{parent.info.run_id}'",
        order_by=["attributes.start_time ASC"],
        output_format="list",
    )


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


@pytest.fixture(scope="session")
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


# A budget the miniature deployment cannot breach. Every test expecting the model's
# answer would otherwise depend on wall-clock timing: a breach falls back, and the
# incumbent reports no probability at all, so an arithmetic assertion fails on a busy
# machine for a reason that has nothing to do with the arithmetic. The budget's own
# behaviour is tested against `strict`, where a breach is forced rather than raced for.
PATIENT_BUDGET_MS = 30_000


def serving(deployment, **overrides) -> TestClient:
    """A client over the miniature deployment, with `serving:` adjusted."""
    config = deployment["config"]
    return TestClient(create_app(config | {"serving": config["serving"] | overrides}))


@pytest.fixture(scope="session")
def client(deployment) -> TestClient:
    return serving(deployment, budget_ms=PATIENT_BUDGET_MS)


@pytest.fixture(scope="session")
def strict(deployment) -> TestClient:
    """The service under the budget `config.yaml` ships, for the tests about the budget."""
    return serving(deployment)


@pytest.fixture(scope="session")
def request_body(deployment):
    """A caller's body for one of the synthetic transactions, by position."""

    def build(row: int = 0) -> dict:
        return synthetic_body(deployment, row)

    return build


def synthetic_body(deployment, row: int = 0) -> dict:
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
