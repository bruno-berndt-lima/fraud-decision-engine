"""The API: one transaction in, one action out.

`docs/serving.md` §1, §4 and §6. Wiring — every decision it serves was made in
`decide.py`, `cost.py` or a phase that closed before this file existed.

**Loaded once, at construction.** The roadmap's rule is that a model is never loaded per
request; the reason it happens here rather than in a lifespan handler is narrower: the
request contract is *derived* from the booster's feature names (`schemas.request_model`),
so the artifacts have to exist before the routes can be declared.

**`def`, not `async def`.** A prediction is CPU-bound and blocking. Inside `async def` it
would stall the event loop and take every concurrent request down with it; declared as a
plain function, FastAPI runs it in a threadpool, which is what the §7 load test measures.

**No `from __future__ import annotations` here, and that is load-bearing.** FastAPI reads
a route's annotations with `get_type_hints`, which resolves strings against the module's
globals — and the request model is built inside `create_app` from the loaded booster, so
it is a local name. Postponed evaluation would leave FastAPI unable to find it, and a
model it cannot resolve is treated as a query parameter: every request would 422 for a
missing `request` field. Nothing warns.

**The service starts without its model.** §4: an artifact that will not load leaves the
process up and serving the rules engine, because declining every transaction during a
model outage converts an availability incident into a total revenue outage. `GET /health`
is where the difference is visible.
"""

import logging
from collections.abc import Mapping
from pathlib import Path

import lightgbm as lgb
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from fraud_engine.data.load import DEFAULT_CONFIG_PATH, load_config
from fraud_engine.evaluation.cost import Costs, load_costs
from fraud_engine.models.rules import REQUIRED_COLUMNS
from fraud_engine.serving.artifacts import Fallback, Model, load_fallback, load_model, stamps
from fraud_engine.serving.decide import MODEL, RULES, Verdict, decide, decide_with_rules
from fraud_engine.serving.schemas import HealthResponse, ScoreResponse, request_model
from fraud_engine.serving.transform import REQUIRED_INPUTS, raw_inputs

log = logging.getLogger(__name__)

LIVE, DEGRADED = "ok", "degraded"

# What a request must be able to carry when the booster is not loaded: the four §1 names
# and the four the incumbent reads. Deliberately small — in this mode nothing else is
# looked at, and the contract says so rather than pretending to read three hundred
# columns it cannot.
FALLBACK_INPUTS = tuple(dict.fromkeys([*REQUIRED_INPUTS, *REQUIRED_COLUMNS]))

# Two of those are strings to the engine: `_product_tier` maps one and `_w_and_m4_m2`
# compares the other against "M2". Named here because the vocabulary that would otherwise
# say so is read by `load_model`, and this is the path where that did not load.
FALLBACK_CATEGORICALS = ("ProductCD", "M4")

UNAVAILABLE = "no model is loaded and the fail-open path is unavailable"


class Deployment:
    """What the process holds between requests.

    A class rather than module globals: the tests build several of these against
    different artifacts in the same interpreter, and globals would make which one answers
    a question of import order.
    """

    def __init__(self, config: Mapping) -> None:
        self.paths = config["paths"]
        self.load_cfg = config["load"]
        self.features_cfg = config["features"]
        self.costs: Costs = load_costs(load_config(Path(self.paths["cost_matrix"])))

        self.model = self._load(load_model, "model", self.paths, config["model"]["impute"])
        self.fallback = self._load(load_fallback, "fail-open path", self.paths)

        # Hashed once, with the artifacts that were just read. Computing them per request
        # would re-read every file to answer a question whose answer cannot change while
        # the process lives — and one of those files is ~77 MB, which measured at nearly
        # two hundred milliseconds against a budget of one hundred.
        self.stamps = stamps(self.paths)

    @staticmethod
    def _load(loader, what: str, *arguments):
        """Load one half of the service, or record that it is not there.

        A missing artifact is not a failure to start (§4). It is a mode, and the mode is
        reported — an exception here would turn a degraded service into no service.
        """
        try:
            return loader(*arguments)
        # LightGBM's own error is first because it is the one the case this exists for
        # actually raises: an absent `model.txt` comes back as a LightGBMError, which is
        # a plain Exception and would sail past a list of the obvious ones.
        except (
            lgb.basic.LightGBMError,
            FileNotFoundError,
            ValueError,
            KeyError,
            OSError,
        ) as failure:
            log.warning("the %s did not load: %s", what, failure)
            return None

    @property
    def mode(self) -> str:
        return MODEL if self.model is not None else RULES

    def request_model(self) -> type[BaseModel]:
        """The contract this deployment can honour.

        Strict when the booster is loaded, because the names are then known; permissive
        about unknown fields when it is not, for the reason `schemas.request_model`
        gives.
        """
        if self.model is None:
            return request_model(FALLBACK_INPUTS, FALLBACK_CATEGORICALS, forbid_unknown=False)

        return request_model(
            raw_inputs(self.model.columns), tuple(self.model.tables.vocabulary), forbid_unknown=True
        )

    def decide(self, raw: pd.DataFrame) -> Verdict:
        """The model's answer, or the incumbent's.

        Raises:
            RuntimeError: If neither is loaded. A service holding nothing cannot decide a
                transaction, and saying so is better than any number it could invent.
        """
        if self.model is not None:
            return decide(self.model, raw, self.costs, self.load_cfg, self.features_cfg)
        if self.fallback is not None:
            return decide_with_rules(self.fallback, raw, self.costs)

        raise RuntimeError(UNAVAILABLE)


def model_version(deployment: Deployment, stamped: Mapping[str, str]) -> str:
    """What decided this request, identified — the booster, or the engine standing in."""
    return (
        stamped["model"] if deployment.model is not None else f"rules:{stamped['rules_constants']}"
    )


def create_app(config: Mapping | None = None, config_path: Path = DEFAULT_CONFIG_PATH) -> FastAPI:
    """Build the service around whatever loaded.

    Args:
        config: A parsed `config.yaml`. Read from `config_path` when absent — the tests
            pass their own, pointing at artifacts they built.
        config_path: Where to read it from otherwise.

    Returns:
        An app with `POST /score` and `GET /health`.
    """
    deployment = Deployment(config if config is not None else load_config(config_path))
    ScoreRequest = deployment.request_model()

    app = FastAPI(
        title="fraud-decision-engine",
        summary="Cost-weighted allow / block decisions on card-not-present transactions.",
    )

    @app.post("/score", response_model=ScoreResponse)
    def score(request: ScoreRequest) -> ScoreResponse:  # type: ignore[valid-type]
        """Decide one transaction.

        Plain `def` on purpose — see the module docstring.
        """
        raw = pd.DataFrame([request.model_dump()])

        try:
            verdict = deployment.decide(raw)
        except RuntimeError as failure:
            # 503, not 500: nothing about the request is wrong, and a caller retrying
            # later is the right behaviour rather than a bug.
            raise HTTPException(status_code=503, detail=str(failure)) from failure

        return ScoreResponse(
            **vars(verdict),
            model_version=model_version(deployment, deployment.stamps),
            cost_matrix_version=deployment.costs.version,
        )

    @app.get("/health", response_model=HealthResponse)
    def health() -> HealthResponse:
        """Live, ready, and what is loaded (§4)."""
        model: Model | None = deployment.model
        fallback: Fallback | None = deployment.fallback

        return HealthResponse(
            status=LIVE if model is not None else DEGRADED,
            mode=deployment.mode,
            model_loaded=model is not None,
            fallback_loaded=fallback is not None,
            artifacts=deployment.stamps,
            features=len(model.columns) if model else None,
            trees=model.booster.num_trees() if model else None,
            calibration=model.calibrator.get("method") if model else None,
            cost_matrix_version=deployment.costs.version,
        )

    return app
