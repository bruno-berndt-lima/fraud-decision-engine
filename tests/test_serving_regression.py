"""The two regression tests the Definition of Done asks for — `docs/serving.md` §8.

A golden-prediction test pinning known requests to known probabilities and decisions,
and a PR-AUC floor on a fixed sample. Both in the two tiers §8 registers, for the reason
it gives: `models/` is gitignored and the booster is large, so a clean checkout can prove
the machinery and nothing else.

- **Everywhere, including CI** — that a golden record can be made, written, read back and
  compared, and that the comparison *fails* when a number moves. Against the miniature
  deployment, whose labels are random by construction: it can pin a pipeline, never a
  quality.
- **Where the artifacts exist** — the shipped model's own predictions and its PR-AUC
  floor, skipped with a stated reason otherwise.

**The golden file holds no transaction data.** It records identifiers and the outputs
they produced, and the bodies are rebuilt from `data/interim` at test time. A fixture
carrying real rows would put competition data in the repository, which is a licence
question rather than a size one.

Regenerating the goldens is deliberate and reviewable: `uv run python
tests/test_serving_regression.py` rewrites the file, and the diff is what says whether
the change was intended. Nothing regenerates it on a failure.
"""

import json
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
import pytest
from starlette.testclient import TestClient

from fraud_engine.data.load import DEFAULT_CONFIG_PATH, load_config
from fraud_engine.evaluation.metrics import pr_auc
from fraud_engine.models.calibrate import apply_calibrator
from fraud_engine.models.train import LABEL
from fraud_engine.serving.app import create_app
from fraud_engine.serving.artifacts import load_model, stamp
from fraud_engine.serving.decide import MODEL
from fraud_engine.serving.fast import build_layout, row
from fraud_engine.serving.transform import raw_inputs

CONFIG = load_config(DEFAULT_CONFIG_PATH)
PATHS = CONFIG["paths"]

GOLDEN = Path(__file__).parent / "golden" / "serving.json"

# What a pinned prediction records. `probability` is the number everything else is
# derived from, so a drift anywhere upstream lands here first.
PINNED = ("probability", "break_even", "decision", "review_eligible")

REQUIRED_ARTIFACTS = (
    PATHS["model"],
    PATHS["categories"],
    PATHS["medians"],
    PATHS["encoders"],
    PATHS["amount_stats"],
    PATHS["vblock"],
    PATHS["calibrator"],
    PATHS["interim"],
    f"{PATHS['features_dir']}/test.parquet",
)

# The same reasoning as `conftest.PATIENT_BUDGET_MS`, for the same reason: pinning a
# prediction is about arithmetic. Under the shipped budget a request that happens to be
# slow falls back, and the incumbent carries no probability at all — so a busy machine
# would record a golden that says nothing about the model. The budget's own behaviour is
# measured in `serving.md` §7, against the container, where it belongs.
PATIENT_BUDGET_MS = 30_000


def shipped_service() -> TestClient:
    """The real artifacts behind a client that will wait for them."""
    return TestClient(
        create_app(CONFIG | {"serving": CONFIG["serving"] | {"budget_ms": PATIENT_BUDGET_MS}})
    )


gated = pytest.mark.skipif(
    not all(Path(path).exists() for path in REQUIRED_ARTIFACTS),
    reason=(
        "the shipped artifacts and built matrices are not in this checkout; "
        "docs/serving.md §8 registers this tier as artifact-gated"
    ),
)


# ---- the machinery, which is what CI can run ---------------------------------


def pin(client: TestClient, bodies: dict[int, dict]) -> list[dict]:
    """Score each body and keep only the fields a regression would move.

    Keyed by `TransactionID` rather than by position, so a golden record stays readable
    against a sample that was drawn differently.
    """
    records = []

    for identifier, body in sorted(bodies.items()):
        answer = client.post("/score", json=body)
        assert answer.status_code == 200, answer.json()

        served = answer.json()
        assert served["mode"] == MODEL, (
            f"{identifier} was decided by the incumbent, which carries no probability; "
            "a golden record must not pin a fallback"
        )
        records.append({"TransactionID": identifier} | {key: served[key] for key in PINNED})

    return records


def differences(expected: list[dict], actual: list[dict]) -> list[str]:
    """Every field that moved, named — a count alone cannot be acted on.

    Exact comparison, including the probability: §8's standard for the transform is that
    a served number differing in the last bits is a number nobody can reproduce, and a
    golden record held to a looser standard than the gate would pin nothing the gate
    does not already.
    """
    if [record["TransactionID"] for record in expected] != [
        record["TransactionID"] for record in actual
    ]:
        return ["the golden record and the run cover different transactions"]

    return [
        f"{record['TransactionID']} {key}: recorded {record[key]!r}, served {now[key]!r}"
        for record, now in zip(expected, actual, strict=True)
        for key in PINNED
        if record[key] != now[key]
    ]


def serving_probabilities(model, layout, bodies: list[dict]) -> np.ndarray:
    """Calibrated probabilities for many transactions, one assembled row at a time.

    Through `fast.row` rather than the batch transform, because what this measures is the
    path a request takes. The predictions are batched afterwards: the assembly is what
    serving does per request, the arithmetic of predicting is not.
    """
    matrix = np.vstack(
        [row(body, model, layout, CONFIG["load"], CONFIG["features"]) for body in bodies]
    )
    scores = model.booster.predict(matrix, num_threads=model.threads)

    return apply_calibrator(model.calibrator, scores)


# ---- the tier that runs anywhere ---------------------------------------------


def test_a_golden_record_survives_being_written_and_read(
    client, deployment, request_body, tmp_path
):
    """The machinery: score, record, persist, read back, compare — and agree."""
    bodies = {
        int(deployment["raw"].iloc[position]["TransactionID"]): request_body(position)
        for position in range(6)
    }
    recorded = pin(client, bodies)

    path = tmp_path / "golden.json"
    path.write_text(json.dumps(recorded, indent=2))

    assert differences(json.loads(path.read_text()), pin(client, bodies)) == []


def test_a_probability_that_moved_is_caught_and_named(client, deployment, request_body):
    """Guards the guard. A comparison that cannot fail pins nothing.

    The failure it stands in for is a serving path that changed a number silently —
    which is exactly what a golden test is written to refuse.
    """
    bodies = {
        int(deployment["raw"].iloc[position]["TransactionID"]): request_body(position)
        for position in range(3)
    }
    recorded = pin(client, bodies)

    tampered = [record | {} for record in recorded]
    tampered[1] = tampered[1] | {"probability": tampered[1]["probability"] + 1e-12}

    found = differences(tampered, pin(client, bodies))
    assert len(found) == 1
    assert "probability" in found[0]


def test_a_decision_that_flipped_is_caught(client, deployment, request_body):
    bodies = {int(deployment["raw"].iloc[0]["TransactionID"]): request_body(0)}
    recorded = pin(client, bodies)

    flipped = [
        recorded[0] | {"decision": "block" if recorded[0]["decision"] == "allow" else "allow"}
    ]

    assert differences(flipped, pin(client, bodies)) != []


def test_the_measurement_runs_where_the_shipped_model_is_absent(deployment, request_body):
    """The PR-AUC machinery, on a deployment whose labels carry no signal.

    The miniature booster is fitted against random labels, so its PR-AUC is the base rate
    and would mean nothing as a floor. What is proven here is the path the gated test
    walks: a body per transaction, assembled by `fast.row`, calibrated, and reduced to a
    metric. The number belongs to the tier that has the model.
    """
    model = load_model(deployment["config"]["paths"], CONFIG["model"]["impute"], 1)
    layout = build_layout(model, CONFIG["features"])

    raw = deployment["raw"]
    scored = raw[raw["split"] == "val_fit"]
    bodies = [request_body(position) for position in range(len(raw))][-len(scored) :]

    probabilities = serving_probabilities(model, layout, bodies)

    assert len(probabilities) == len(scored)
    assert ((probabilities > 0) & (probabilities < 1)).all(), "a probability outside (0, 1)"

    measured = pr_auc(scored[LABEL].reset_index(drop=True), pd.Series(probabilities))
    assert 0.0 < measured < 1.0


# ---- the tier that needs the shipped artifacts -------------------------------


def bodies_for(identifiers: list[int]) -> dict[int, dict]:
    """Real transactions, by identifier, as a caller would send them."""
    booster = lgb.Booster(model_file=PATHS["model"])
    carried = raw_inputs(booster.feature_name())
    del booster

    frame = pd.read_parquet(PATHS["interim"], filters=[("TransactionID", "in", identifiers)])
    frame = frame.set_index("TransactionID").loc[identifiers].reset_index()
    columns = [name for name in carried if name in frame.columns]

    prepared = frame[columns].convert_dtypes(convert_integer=False, convert_floating=False)
    return {
        int(frame.loc[position, "TransactionID"]): {
            name: (None if pd.isna(value) else bool(value) if name == "has_identity" else value)
            for name, value in record.items()
        }
        for position, record in enumerate(prepared.astype("object").to_dict(orient="records"))
    }


@pytest.fixture(scope="module")
def golden() -> dict:
    return json.loads(GOLDEN.read_text())


@pytest.mark.artifacts
@gated
def test_the_shipped_service_returns_the_predictions_it_was_pinned_to(golden):
    """What the Definition of Done calls the golden-prediction test.

    The artifact stamps are checked first, so a record made against a different model
    fails saying that rather than reporting a wrong number.
    """
    assert golden["artifacts"]["model"] == stamp(PATHS["model"]), (
        "these goldens were recorded against a different booster"
    )
    assert golden["artifacts"]["calibrator"] == stamp(PATHS["calibrator"])

    client = shipped_service()
    identifiers = [record["TransactionID"] for record in golden["predictions"]]

    assert differences(golden["predictions"], pin(client, bodies_for(identifiers))) == []


@pytest.mark.artifacts
@gated
def test_pr_auc_on_the_fixed_sample_holds_its_floor(golden):
    """What the Definition of Done calls the PR-AUC regression test.

    The floor is below the recorded measurement by a stated margin, not at it: the sample
    is fixed and the model frozen, so the value is deterministic on one machine, and the
    margin covers a different LightGBM build rather than a different model. A real
    degradation in the serving path is orders of magnitude larger than it.
    """
    measurement = golden["pr_auc"]

    identifiers = (
        pd.read_parquet(
            f"{PATHS['features_dir']}/{measurement['split']}.parquet", columns=["TransactionID"]
        )
        .sample(measurement["rows"], random_state=measurement["seed"])["TransactionID"]
        .sort_values()
        .tolist()
    )
    bodies = bodies_for(identifiers)

    labels = (
        pd.read_parquet(
            f"{PATHS['features_dir']}/{measurement['split']}.parquet",
            columns=["TransactionID", LABEL],
        )
        .set_index("TransactionID")
        .loc[identifiers, LABEL]
    )

    model = load_model(PATHS, CONFIG["model"]["impute"], CONFIG["serving"]["threads"])
    layout = build_layout(model, CONFIG["features"])
    probabilities = serving_probabilities(model, layout, [bodies[key] for key in identifiers])

    measured = pr_auc(labels.reset_index(drop=True), pd.Series(probabilities))

    assert measured >= measurement["floor"], (
        f"PR-AUC through the serving path fell to {measured:.5f}, below the floor "
        f"{measurement['floor']} recorded at {measurement['observed']:.5f}"
    )


def record_goldens() -> dict:
    """Re-make the golden file from the shipped artifacts. Run this deliberately.

    Not a test and never called by one: a golden record that regenerates itself on
    failure pins nothing. `uv run python tests/test_serving_regression.py` rewrites the
    file, and the diff is what says whether the change was meant.
    """
    client = shipped_service()

    # Chosen to cover both branches rather than drawn at random. The EV policy blocks a
    # few percent of transactions, so a random dozen is a dozen allows — and the decline
    # is the decision with a consequence attached, the one `/explain` exists for. A pool
    # is scored and the pinned set is taken from each side of it.
    split, per_decision, seed, pool_size = "test", 6, 0, 400
    pool = (
        pd.read_parquet(f"{PATHS['features_dir']}/{split}.parquet", columns=["TransactionID"])
        .sample(pool_size, random_state=seed)["TransactionID"]
        .sort_values()
        .tolist()
    )
    scored = pin(client, bodies_for(pool))

    identifiers = [
        record["TransactionID"]
        for decision in ("allow", "block")
        for record in [r for r in scored if r["decision"] == decision][:per_decision]
    ]

    sample_rows, sample_seed = 1_000, 0
    sample = (
        pd.read_parquet(
            f"{PATHS['features_dir']}/{split}.parquet", columns=["TransactionID", LABEL]
        )
        .sample(sample_rows, random_state=sample_seed)
        .sort_values("TransactionID")
    )

    model = load_model(PATHS, CONFIG["model"]["impute"], CONFIG["serving"]["threads"])
    layout = build_layout(model, CONFIG["features"])
    bodies = bodies_for(sample["TransactionID"].tolist())
    probabilities = serving_probabilities(
        model, layout, [bodies[key] for key in sample["TransactionID"]]
    )
    observed = pr_auc(sample[LABEL].reset_index(drop=True), pd.Series(probabilities))

    return {
        "artifacts": {
            name: stamp(PATHS[name]) for name in ("model", "calibrator", "categories", "medians")
        },
        "predictions": pin(client, bodies_for(identifiers)),
        "pr_auc": {
            "split": split,
            "rows": sample_rows,
            "seed": sample_seed,
            "observed": observed,
            # Said in the file because the file is what a reader quotes from. A sample
            # this size holds a few dozen positives, so its PR-AUC is far coarser than
            # the model's — the headline reports the model's, on the whole slice.
            "note": "a property of this sample, not the model's test PR-AUC",
            # Below the measurement by a stated margin rather than at it. The sample is
            # fixed and the model frozen, so this can only move if the serving path
            # does — the margin covers a different LightGBM build, not a different
            # model, and a real degradation is far larger than it.
            "floor": round(observed - 0.02, 3),
        },
    }


if __name__ == "__main__":
    GOLDEN.parent.mkdir(exist_ok=True)
    GOLDEN.write_text(json.dumps(record_goldens(), indent=2) + "\n")
    print(f"wrote {GOLDEN}")
