"""Logistic regression — the "is complexity earning its keep" reference point.

Not the incumbent. The rules engine is what the model must beat in dollars; this
answers a different question: how much of the signal is reachable by a linear
model on columns that arrive with the request? If Phase 05's LightGBM barely
clears this, the complexity is not paying for itself.

It is also the honest comparison for the rules engine, because the two have the
same shape. A rules engine is a weighted sum of indicators with the weights
chosen by hand; this is a weighted sum of features with the weights fitted by
maximum likelihood. Same functional form, different provenance.

Every transformation with a learned parameter — the medians, the scales, the
category vocabularies — is fitted inside a Pipeline on the train slice alone.
``.fit()`` is called exactly once, and there is no path to reach a step and
refit it on validation.

Run twice, per ``docs/experiments.md`` E2: with and without class weighting,
both reported.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler

from fraud_engine.data.load import DEFAULT_CONFIG_PATH, load_config
from fraud_engine.evaluation.report import build_report, load_capacities, write_report

log = logging.getLogger(__name__)

# Counts. Zero null on train, so no indicator is needed. Already Vesta
# aggregates over undocumented windows — the limitation named in Phase 01.
COUNT_COLUMNS = tuple(f"C{i}" for i in range(1, 15))

# Day-deltas. Which ones survive is decided by the null threshold in config,
# not by hand, so the choice is a stated rule rather than a preference.
DELTA_COLUMNS = tuple(f"D{i}" for i in range(1, 16))

# Match flags. 2-3 levels each and 34-75% null, and the missingness is signal
# rather than damage: it is one-hot encoded as its own level.
MATCH_COLUMNS = tuple(f"M{i}" for i in range(1, 10))

CATEGORICAL_COLUMNS = ("ProductCD", "card4", "card6", "P_emaildomain", "weekday", *MATCH_COLUMNS)

PASSTHROUGH_COLUMNS = ("has_identity",)

# What the model may see. Stated positively: passing the whole frame and
# trusting remainder="drop" to discard isFraud puts the label one config edit
# away from being a feature.
FEATURE_COLUMNS = (
    "TransactionAmt",
    "hour",
    *CATEGORICAL_COLUMNS,
    *PASSTHROUGH_COLUMNS,
    *COUNT_COLUMNS,
    *DELTA_COLUMNS,
)

SOURCE_COLUMNS = (
    "TransactionID",
    "isFraud",
    "day",
    "TransactionAmt",
    "hour",
    *CATEGORICAL_COLUMNS,
    *PASSTHROUGH_COLUMNS,
    *COUNT_COLUMNS,
    *DELTA_COLUMNS,
)

# One record per variant, so E2 is a read rather than a reconciliation.
VARIANTS = {"logistic_baseline": None, "logistic_balanced": "balanced"}


def _log1p_amount(frame: pd.DataFrame) -> np.ndarray:
    """TransactionAmt on a log scale.

    The amount distribution has a long right tail, and a linear model on the raw
    value assumes each extra dollar shifts the log-odds equally. On a log scale
    it assumes each *doubling* does, which is closer to how fraud behaves and is
    what H1 found once the pooled artifact was removed.
    """
    return np.log1p(frame.to_numpy(dtype="float64"))


def _hour_to_cycle(frame: pd.DataFrame) -> np.ndarray:
    """Hour as sine and cosine of its position in the 24-hour cycle.

    A linear model reading raw ``hour`` sees 23 and 0 as maximally distant when
    they are adjacent. Two orthogonal components make the wrap-around
    representable. Phase 01 established the bucket is a consistent cycle but not
    wall-clock time — which does not matter here, since only the periodicity is
    used.
    """
    radians = 2 * np.pi * frame.to_numpy(dtype="float64") / 24.0
    return np.hstack([np.sin(radians), np.cos(radians)])


def _hour_feature_names(transformer, input_features) -> np.ndarray:  # noqa: ARG001
    """Names for the two columns ``_hour_to_cycle`` emits."""
    return np.asarray(["hour_sin", "hour_cos"], dtype=object)


def select_delta_columns(train: pd.DataFrame, max_null_frac: float) -> list[str]:
    """The ``D*`` columns whose nullness on train is under the threshold.

    Fitted on train like anything else: the decision of *which* columns exist is
    made from the training window and applied unchanged downstream. Choosing
    them from the full table would leak the validation missingness pattern into
    the feature set.

    Args:
        train: Training rows.
        max_null_frac: Maximum tolerated null fraction, from config.

    Returns:
        Surviving column names, in ``D1..D15`` order.
    """
    present = [column for column in DELTA_COLUMNS if column in train.columns]
    return [column for column in present if train[column].isna().mean() < max_null_frac]


def build_pipeline(
    delta_columns: list[str],
    logistic_cfg: dict,
    class_weight: str | None,
) -> Pipeline:
    """The preprocessing and the model, as one object that can only be fitted whole.

    Wrapping preprocessing in the Pipeline is what makes "fit on train, transform
    everywhere else" structural. There is no handle to reach the imputer or the
    scaler separately, and ``predict_proba`` can only call ``transform`` — so the
    leakage this invariant guards against is unreachable rather than merely
    avoided.

    Args:
        delta_columns: From ``select_delta_columns``.
        logistic_cfg: The ``baselines.logistic`` config block.
        class_weight: ``None`` or ``"balanced"``. See E2.

    Returns:
        An unfitted pipeline.
    """
    numeric = Pipeline([("impute", SimpleImputer(strategy="median")), ("scale", StandardScaler())])
    # add_indicator on the deltas only: the counts have no nulls on train, and a
    # constant indicator column would be a free coefficient fitted on noise.
    delta = Pipeline(
        [
            ("impute", SimpleImputer(strategy="median", add_indicator=True)),
            ("scale", StandardScaler()),
        ]
    )
    preprocess = ColumnTransformer(
        [
            (
                "amount",
                Pipeline(
                    [
                        ("log", FunctionTransformer(_log1p_amount, feature_names_out="one-to-one")),
                        ("scale", StandardScaler()),
                    ]
                ),
                ["TransactionAmt"],
            ),
            (
                "hour",
                FunctionTransformer(_hour_to_cycle, feature_names_out=_hour_feature_names),
                ["hour"],
            ),
            ("counts", numeric, list(COUNT_COLUMNS)),
            ("deltas", delta, delta_columns),
            (
                "categorical",
                # handle_unknown="ignore" is load-bearing on a temporal split:
                # days 121+ carry card and email values absent from days 1-90,
                # and the default raises at transform time. min_frequency folds
                # rare levels into one column so a level seen twice in train
                # cannot get its own coefficient.
                OneHotEncoder(handle_unknown="infrequent_if_exist", min_frequency=50),
                list(CATEGORICAL_COLUMNS),
            ),
            ("passthrough", "passthrough", list(PASSTHROUGH_COLUMNS)),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )
    return Pipeline(
        [
            ("preprocess", preprocess),
            (
                "logistic",
                LogisticRegression(
                    C=logistic_cfg["C"],
                    max_iter=logistic_cfg["max_iter"],
                    class_weight=class_weight,
                ),
            ),
        ]
    )


def prepare(frame: pd.DataFrame) -> pd.DataFrame:
    """Cast the columns sklearn cannot take as they arrive from parquet.

    Categoricals become strings with an explicit ``"__missing__"`` level: the
    missingness is signal, and letting it become NaN would silently hand the
    encoder a decision this project has an opinion about.
    """
    prepared = frame.copy()
    for column in CATEGORICAL_COLUMNS:
        prepared[column] = prepared[column].astype("object").fillna("__missing__").astype(str)
    prepared["has_identity"] = prepared["has_identity"].astype("int8")
    return prepared


def main(config_path: Path = DEFAULT_CONFIG_PATH) -> None:
    """Fit both variants on train and score validation through the Phase 02 harness.

    Wiring only. Invoked by ``make baselines`` as
    ``python -m fraud_engine.models.logistic``.

    TEST is never scored — ``report.DEFAULT_SPLITS`` is validation-only, and it
    is not overridden here.
    """
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    config = load_config(config_path)
    paths = config["paths"]
    logistic_cfg = config["baselines"]["logistic"]

    frame = pd.read_parquet(paths["interim"], columns=list(SOURCE_COLUMNS))
    splits = pd.read_parquet(paths["splits"], columns=["TransactionID", "split"])
    frame = frame.merge(splits, on="TransactionID", how="inner", validate="one_to_one")
    frame = prepare(frame)

    train = frame[frame["split"] == "train"]
    delta_columns = select_delta_columns(train, logistic_cfg["d_max_null_frac"])
    log.info(
        "D* columns kept (<%.0f%% null on train): %s",
        logistic_cfg["d_max_null_frac"] * 100,
        delta_columns,
    )

    capacities = load_capacities(load_config(Path(paths["cost_matrix"])))
    for name, class_weight in VARIANTS.items():
        pipeline = build_pipeline(delta_columns, logistic_cfg, class_weight)
        # The only fit in this module, and it only ever sees the train slice.
        pipeline.fit(train[list(FEATURE_COLUMNS)], train["isFraud"])

        scored = frame.copy()
        scored["score"] = pipeline.predict_proba(frame[list(FEATURE_COLUMNS)])[:, 1]

        path = write_report(build_report(name, scored, capacities), paths["metrics_dir"])
        log.info("%-20s class_weight=%-9s -> %s", name, class_weight, path)


if __name__ == "__main__":
    main()
