"""LightGBM — the challenger.

The rules engine is what this has to beat in dollars; ``logistic.py`` is the
reference point that says whether the extra complexity is earning its keep.

Categoricals are handed to LightGBM natively rather than one-hot encoded. A tree
splits a category into two *sets* of levels, so it can represent "these six
browsers behave alike" in one split where a linear model needs six coefficients.
The cost is that the split is fitted, and a level backed by three transactions
lets it memorise three transactions — which is why the vocabulary has a floor.
"""

from __future__ import annotations

import logging
from pathlib import Path

import lightgbm as lgb
import mlflow
import pandas as pd

from fraud_engine.data.load import DEFAULT_CONFIG_PATH, load_config
from fraud_engine.data.splits import SPLIT_NAMES
from fraud_engine.evaluation.report import load_capacities, write_run
from fraud_engine.evaluation.tracking import configure_tracking, log_report, tracked_run
from fraud_engine.features.encoders import MISSING

# Levels the training window never saw, and levels it saw too rarely to learn
# anything from, share one bucket. Separate from MISSING, which is a different
# fact about a row: the field did not arrive, rather than arrived unrecognised.
OTHER = "__other__"

# Keys, the label, and the split axis. `day` and `TransactionDT` are the same
# exclusion twice at different resolutions: a model handed either learns the
# timeline instead of fraud, and on a temporal split that reads as skill.
# Fixed, and not open to tuning. `metric` is the load-bearing one: LightGBM's
# default for a binary objective is log-loss, and stopping early on log-loss at
# this base rate stops in the wrong place — the project names PR-AUC as primary,
# so early stopping has to agree with it.
#
# `deterministic` needs one of the force_*_wise flags set to have any effect,
# which is why both are here.
CONTRACT_PARAMS = {
    "objective": "binary",
    # Matches what to_dataset builds with. LightGBM compares the two and refuses
    # to train if they disagree, so this is not a duplicate of that setting —
    # it is the other half of it.
    "feature_pre_filter": False,
    "metric": "average_precision",
    "deterministic": True,
    "force_row_wise": True,
    "verbosity": -1,
}

log = logging.getLogger(__name__)


def run_name(model_cfg: dict) -> str:
    """What this run is, derived rather than typed.

    A constant would keep saying "untuned" the moment config stopped being, and
    it names the record in ``reports/metrics/``. Deriving it also means the
    untuned reference's record survives adoption instead of being overwritten —
    Phase 06 compares the shipped model against it, and both being on disk is
    the difference between a comparison and an excavation of the git history.
    """
    return "lightgbm_tuned" if model_cfg["tuned"] else "lightgbm_untuned"


LABEL = "isFraud"

EXCLUDED_COLUMNS = ("TransactionID", "TransactionDT", LABEL, "day")

# What a training run reads. TEST is absent and has to be typed to be loaded,
# the same shape as report.DEFAULT_SPLITS: the invariant says test is touched
# once at the very end, and a default that reads it makes that a matter of
# remembering rather than of asking.
TRAINING_SPLITS = ("train", "val_fit", "val_cal")

SENTINELS = (OTHER, MISSING)


def fit_categories(train: pd.DataFrame, min_rows: int) -> dict[str, pd.Index]:
    """The category vocabulary each column is scored against, from train alone.

    Today every split file carries an identical vocabulary, because ``partition``
    sliced one frame and the dtype travelled with it. That vocabulary was built
    from the whole table — validation and test included — and **production has no
    whole table**. A request arriving with a browser version released after
    training has no code, and nothing currently decides what happens to it.

    Fitting here forces training to face what serving faces. The levels are the
    ones train saw at least ``min_rows`` times; everything else routes to a
    sentinel at apply time.

    **Rare training levels join the unseen ones in ``OTHER`` rather than getting
    their own.** If ``OTHER`` collected only the levels validation brings, it
    would carry no training rows at all and the model would score them against
    nothing. Folding the rare levels in is what gives the bucket real mass.

    Both sentinels are in every column's vocabulary whether or not train needed
    them. Train having no nulls in a column does not mean a request will not
    arrive without it, and the vocabulary is what ships.

    Levels are sorted, so the integer codes behind them depend on the training
    window and not on the order ``value_counts`` happened to break ties in.

    Args:
        train: Training rows. Columns of dtype ``category`` are the ones fitted;
            the matrices are the source of truth for which those are.
        min_rows: Occurrences in train below which a level is not learned
            separately.

    Returns:
        ``{column: levels}``, sentinels last. A column whose every level is rare
        comes back holding only the sentinels — nothing is dropped, but there is
        nothing left for a split to separate.
    """
    vocabulary = {}

    for column in train.select_dtypes("category").columns:
        counts = train[column].value_counts()
        frequent = sorted(counts[counts >= min_rows].index)
        vocabulary[column] = pd.Index([*frequent, *SENTINELS], dtype="object")

    return vocabulary


def apply_categories(frame: pd.DataFrame, vocabulary: dict[str, pd.Index]) -> pd.DataFrame:
    """Re-level every categorical against the fitted vocabulary.

    This is where the split files stop carrying a vocabulary built from the
    whole table. Afterwards each categorical holds exactly the levels
    ``fit_categories`` learned, in that order, so a code means the same thing in
    every split — which is the property that lets the model be trained on one and
    scored on another at all.

    Order matters in the body: nulls become ``MISSING`` *before* the membership
    test, because a null is not an unrecognised value. Testing first would route
    every missing field into ``OTHER`` and merge two facts the model should be
    able to tell apart.

    Args:
        frame: Any split's matrix.
        vocabulary: ``{column: levels}`` from ``fit_categories``.

    Returns:
        A new frame — the input is not modified — whose fitted columns are
        null-free and carry the fitted dtype.

    Raises:
        ValueError: If the frame carries a categorical the fit never saw. Left
            alone it would keep its whole-frame vocabulary, which is the exact
            defect this function exists to remove, and nothing downstream would
            look wrong.
        KeyError: If a fitted column is absent from the frame.
    """
    unfitted = set(frame.select_dtypes("category").columns) - set(vocabulary)
    if unfitted:
        raise ValueError(
            f"categorical columns with no fitted vocabulary: {sorted(unfitted)}; "
            "they would keep the vocabulary the whole table gave them"
        )

    prepared = frame.copy()

    for column, levels in vocabulary.items():
        values = prepared[column].astype("object").fillna(MISSING)
        prepared[column] = values.where(values.isin(levels), OTHER).astype(
            pd.CategoricalDtype(levels)
        )

    return prepared


def fit_medians(train: pd.DataFrame, columns: list[str]) -> pd.Series:
    """Per-column medians from the training window.

    Only the numeric columns need them: the categoricals arrived null-free from
    ``apply_categories``, where missing became a level rather than a gap.
    """
    numeric = train[columns].select_dtypes("number")
    return numeric.median()


def apply_medians(frame: pd.DataFrame, medians: pd.Series) -> pd.DataFrame:
    """Fill numeric nulls with the fitted medians.

    Applied to validation as well as train. A model fitted on filled data and
    scored on data still carrying nulls would be measured on a distribution it
    never saw, which trades the confound this exists to remove for a different
    one.
    """
    filled = frame.copy()
    filled[medians.index] = filled[medians.index].fillna(medians)
    return filled


def feature_columns(frame: pd.DataFrame) -> list[str]:
    """Everything the model may see: the matrix, less the columns it may not.

    ``logistic.py`` names its features positively, and gives the reason: trusting
    a drop rule puts the label one edit away from becoming a feature. That reason
    is right and the mechanism does not survive three hundred columns — a stale
    allow-list drops new features silently, which is the worse failure, since a
    model quietly trained on less than it was given looks fine.

    The guard is what makes the inversion honest: every excluded name must be
    present. A deny-list describing a table that no longer exists is one rename
    away from letting the label through, and this is what turns that into an
    error instead of a very good score.

    Args:
        frame: A split matrix.

    Returns:
        Feature names, in matrix order.

    Raises:
        ValueError: If any excluded column is missing — the deny-list and the
            matrix have drifted apart.
    """
    missing = [column for column in EXCLUDED_COLUMNS if column not in frame.columns]
    if missing:
        raise ValueError(
            f"excluded columns absent from the matrix: {missing}; "
            "the deny-list no longer describes this table"
        )

    return [column for column in frame.columns if column not in EXCLUDED_COLUMNS]


def load_split_matrices(
    features_dir: Path | str, splits: tuple[str, ...] = TRAINING_SPLITS
) -> dict[str, pd.DataFrame]:
    """Each split's matrix, read whole and kept apart.

    Apart, because LightGBM early-stops against a validation set and needs it as
    its own object. ``evaluate.load_matrices`` returns one concatenated frame for
    the probe, which fits on a mask; concatenating here only to split again would
    also point ``models`` at a module that already imports ``models``.

    Read whole rather than by column list: the matrices are self-contained by
    design, and naming columns here would be a second place for the feature set
    to be decided.

    Args:
        features_dir: Directory holding ``{split}.parquet``.
        splits: Which to read. Defaults to the three a training run needs —
            naming ``test`` is possible, and has to be deliberate.

    Returns:
        ``{split: matrix}`` in the order given.

    Raises:
        ValueError: If a name is not a split.
        FileNotFoundError: If a split's matrix is absent.
    """
    unknown = set(splits) - set(SPLIT_NAMES)
    if unknown:
        raise ValueError(f"not splits: {sorted(unknown)}")

    features_dir = Path(features_dir)

    return {name: pd.read_parquet(features_dir / f"{name}.parquet") for name in splits}


def prepare_matrices(
    features_dir: Path | str, model_cfg: dict, splits: tuple[str, ...] = TRAINING_SPLITS
) -> tuple[dict[str, pd.DataFrame], dict[str, pd.Index], pd.Series | None]:
    """Matrices as a booster may see them, with the two tables that shaped them.

    Everything between reading parquet and building a ``Dataset``: the
    vocabulary and the medians are fitted on ``train`` and applied to every
    split, including the ones only scored. Both directions of that matter — a
    model fitted on filled data and scored on nulls meets a distribution it
    never saw, and a validation split re-levelled against its own categories
    means a code stands for two different things either side of the boundary.

    **One function because four callers need the same answer.** Training, the
    ablation, its floor and the purge experiment all start here, and this is
    leakage-sensitive code: a copy that drifted — fitting on the wrong split,
    or filling train alone — would produce an experiment that is wrong and
    looks fine.

    Args:
        features_dir: Directory holding ``{split}.parquet``.
        model_cfg: The ``model:`` config block.
        splits: Which to read. ``train`` is required; it is what the fits see.

    Returns:
        ``(matrices, vocabulary, medians)``. ``medians`` is ``None`` when
        ``impute`` is off, so a caller writing it as an artifact can tell the
        difference between "filled with nothing" and "not filled".

    Raises:
        ValueError: If ``train`` is not among ``splits``, or per the functions
            this composes.
    """
    if "train" not in splits:
        raise ValueError(f"train is not among {list(splits)}; there is nothing to fit on")

    matrices = load_split_matrices(features_dir, splits)

    vocabulary = fit_categories(matrices["train"], model_cfg["min_category_rows"])
    matrices = {split: apply_categories(frame, vocabulary) for split, frame in matrices.items()}

    medians = (
        fit_medians(matrices["train"], feature_columns(matrices["train"]))
        if model_cfg["impute"]
        else None
    )
    if medians is not None:
        matrices = {split: apply_medians(frame, medians) for split, frame in matrices.items()}

    return matrices, vocabulary, medians


def write_categories(vocabulary: dict[str, pd.Index], path: Path | str) -> None:
    """Persist the vocabulary — the model cannot be served without it.

    A trained booster stores which features are categorical and the sets of
    codes its splits test, but not what those codes *mean*. That mapping lives
    here, and a serving process without it would hand LightGBM integers derived
    from whatever levels happened to be in the request batch.

    This is the tier-2 artifact ``features.md`` promised for the category
    vocabulary, in the same long form as the frequency tables and the V-block
    reduction, so the file describes itself.

    The code is written out rather than left implicit in the row order. Row order
    survives a parquet round trip, but the codes *are* the contract with the
    model — a mapping that a re-sorted read could silently change is not a
    contract.

    Args:
        vocabulary: ``{column: levels}`` from ``fit_categories``.
        path: Destination. Parent directories are created.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    rows = pd.DataFrame(
        [
            {"column": column, "code": code, "level": level}
            for column, levels in vocabulary.items()
            for code, level in enumerate(levels)
        ]
    )
    rows.to_parquet(path, index=False)


def write_medians(medians: pd.Series, path: Path | str) -> None:
    """Persist the fill values — a served model trained on them needs them.

    Imputation moves the data before the model sees it, so a serving process
    without this file feeds the booster a distribution it was never fitted on:
    the columns it filled would arrive null instead. That makes this a tier-2
    artifact in ``features.md``, alongside the category vocabulary and the
    frequency tables, and it ships whenever ``model.impute`` is on.

    Args:
        medians: From ``fit_medians``.
        path: Destination. Parent directories are created.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    medians.rename("median").rename_axis("column").reset_index().to_parquet(path, index=False)


def to_dataset(
    frame: pd.DataFrame, columns: list[str], reference: lgb.Dataset | None = None
) -> lgb.Dataset:
    """The frame as LightGBM's own structure, binned and with categoricals named.

    **``reference`` is what makes a threshold portable.** LightGBM does not split
    on raw values; it buckets each continuous feature into histogram bins first,
    and a learned split is a bin boundary. Bins are computed from whatever data
    builds the dataset, so a validation set left to bin itself would place its
    own boundaries — and every threshold the model learned would land somewhere
    slightly different from where it was fitted. Passing the training dataset
    makes validation reuse its bin edges, which is the only way the two can be
    compared at all.

    Categoricals are named rather than left to detection. LightGBM will infer
    them from the pandas dtype, but the set of categorical features is a decision
    this project makes explicitly — and an inference is a decision that changes
    when someone else's code changes a dtype.

    The label is read from the frame directly. It is not in ``columns`` —
    ``feature_columns`` excluded it — so the two cannot be confused.

    **``feature_pre_filter`` is off, and it has to be here rather than in the
    tuning stage alone.** LightGBM otherwise drops, at construction time, the
    features that cannot be split under the current ``min_data_in_leaf`` — and
    then refuses to train the same dataset with a *smaller* one, because those
    features would stay wrongly dropped. A search that moves that knob downward
    hits it. Setting the flag only where the search runs would bin the tuned
    candidate differently from the reference it is judged against, which is the
    one thing E6's comparison cannot survive. It costs memory and nothing else:
    a pre-filtered feature is one that could not have been split anyway, and the
    untuned reference returns the same score to eight decimal places either way.

    Args:
        frame: A prepared matrix, already through ``apply_categories``. Passing
            one that has not been leaves each split's codes meaning whatever its
            own dtype said.
        columns: Feature names, from ``feature_columns``.
        reference: The training dataset, for every set that is not it.

    Returns:
        An unconstructed dataset — LightGBM builds it lazily, at ``train``.
    """
    categorical = [
        column for column in columns if isinstance(frame[column].dtype, pd.CategoricalDtype)
    ]

    return lgb.Dataset(
        frame[columns],
        label=frame[LABEL],
        categorical_feature=categorical,
        reference=reference,
        params={"feature_pre_filter": False},
    )


def resolve_params(model_cfg: dict) -> dict:
    """What actually reaches LightGBM: the contract, the tuned knobs, the seed.

    One function because two callers need the same answer — ``fit`` to train with
    them and ``main`` to record them — and a run whose logged parameters were
    computed separately from the ones it trained on is a run that cannot be
    reproduced from its own record.

    Args:
        model_cfg: The ``model:`` config block.

    Returns:
        The merged parameter mapping.

    Raises:
        ValueError: If config sets a contract parameter.
    """
    overridden = set(model_cfg["tuned"]) & set(CONTRACT_PARAMS)
    if overridden:
        raise ValueError(
            f"contract parameters cannot be set from config: {sorted(overridden)}; "
            "changing them changes what every earlier run's number meant"
        )

    return {**CONTRACT_PARAMS, **model_cfg["tuned"], "seed": model_cfg["seed"]}


def fit(train: lgb.Dataset, val_fit: lgb.Dataset, model_cfg: dict) -> lgb.Booster:
    """Train until ``VAL-FIT`` stops improving, and keep the best round.

    Boosting drives training loss down for as long as it is allowed to, so the
    number of trees cannot be read off the training curve. Early stopping reads
    it off a set the model is not fitting: while ``VAL-FIT`` improves it keeps
    going, and after ``early_stopping_rounds`` without improvement it stops and
    returns the best iteration rather than the last.

    **``VAL-FIT`` and not ``VAL-CAL``.** Early stopping looks at that set once per
    round and picks a model from it, which spends it — the metric there is
    optimistic afterwards, because it was the stopping criterion. ``VAL-CAL``
    stays untouched for Phase 06's calibrator and threshold.

    **Contract parameters are not config.** ``CONTRACT_PARAMS`` are fixed in code
    and refused from config, because a search space that reached ``metric`` or
    ``objective`` would silently change what "better" means between runs and the
    comparison view would be quietly comparing nothing.

    ``model_cfg["tuned"]`` is empty for the untuned reference, and it is empty
    deliberately: the reference has to be the model anyone would get without
    thinking, or Phase 05 cannot say what tuning bought.

    **Determinism holds for one machine.** ``deterministic`` guarantees a
    repeatable result given the same data, parameters *and thread count*, so this
    reproduces on the machine that ran it. Nothing here claims more than that.

    **The ceiling has to not bind.** ``num_boost_round`` is a stop, not a
    choice: if training runs out of rounds before the patience window closes,
    ``best_iteration`` is simply the last round, and the model is where the
    budget ended rather than where the metric peaked. LightGBM reports that the
    same way it reports a real peak, so it is checked rather than trusted.

    Args:
        train: The training dataset.
        val_fit: The early-stopping dataset, referencing ``train``'s bins.
        model_cfg: The ``model:`` config block.

    Returns:
        A booster truncated to its best iteration.

    Raises:
        ValueError: If config sets a contract parameter, or if the round ceiling
            bound before early stopping could confirm a peak.
    """
    patience, ceiling = model_cfg["early_stopping_rounds"], model_cfg["num_boost_round"]

    booster = lgb.train(
        resolve_params(model_cfg),
        train,
        num_boost_round=ceiling,
        valid_sets=[val_fit],
        valid_names=["val_fit"],
        callbacks=[lgb.early_stopping(patience), lgb.log_evaluation(period=100)],
    )

    if booster.best_iteration + patience > ceiling:
        raise ValueError(
            f"num_boost_round={ceiling} bound before early stopping could confirm a peak: "
            f"best_iteration={booster.best_iteration} leaves no room for {patience} rounds "
            "of patience. Raise the ceiling; the reported best iteration is the budget, "
            "not a measured optimum."
        )

    return booster


def score(
    booster: lgb.Booster, matrices: dict[str, pd.DataFrame], columns: list[str]
) -> pd.DataFrame:
    """Every row the booster was given a chance to score, as the harness expects.

    ``split`` is restored from the key rather than read from a column, for the
    same reason ``partition`` dropped it: the filename and the label were never
    two facts that could check each other.

    **The best iteration is named, not assumed.** ``predict`` falls back to it
    when early stopping set one, but the fallback is a library default and the
    truncation is a decision this run made — the trees after the peak were
    measured to be worse, and saying so costs one argument.

    Scoring more splits than get recorded is deliberate. ``write_run`` filters
    both of its artifacts to the splits it is asked for, so the gate that keeps
    test out lives there, structurally, instead of being repeated by every caller
    that assembles a frame.

    Args:
        booster: A fitted booster.
        matrices: ``{split: prepared matrix}``.
        columns: Feature names, in the order the booster was fitted on.

    Returns:
        One row per scored transaction, carrying what the harness requires.
    """
    parts = []

    for split, frame in matrices.items():
        part = frame[["TransactionID", LABEL, "day"]].copy()
        part["score"] = booster.predict(frame[columns], num_iteration=booster.best_iteration)
        part["split"] = pd.Categorical([split] * len(frame), categories=SPLIT_NAMES)
        parts.append(part)

    return pd.concat(parts, ignore_index=True)


def main(config_path: Path = DEFAULT_CONFIG_PATH) -> None:
    """Fit the untuned reference, score validation, record the run.

    Wiring only. Invoked by ``make train`` as
    ``python -m fraud_engine.models.train``.

    TEST is never loaded — ``load_split_matrices`` defaults to the three splits a
    training run needs, and ``write_run`` would filter it out even if it were.

    Args:
        config_path: Path to ``config.yaml``.
    """
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    config = load_config(config_path)
    paths, model_cfg = config["paths"], config["model"]

    configure_tracking(config["tracking"])

    matrices, vocabulary, medians = prepare_matrices(paths["features_dir"], model_cfg)
    columns = feature_columns(matrices["train"])

    train = to_dataset(matrices["train"], columns)
    val_fit = to_dataset(matrices["val_fit"], columns, reference=train)

    capacities = load_capacities(load_config(Path(paths["cost_matrix"])))
    params = resolve_params(model_cfg)
    name = run_name(model_cfg)

    with tracked_run(
        name, {**params, "impute": model_cfg["impute"], "n_features": len(columns)}, config_path
    ):
        booster = fit(train, val_fit, model_cfg)

        scored = score(booster, matrices, columns)
        metrics_path, _ = write_run(
            name, scored, capacities, paths["metrics_dir"], paths["predictions_dir"]
        )

        log_report(metrics_path, {"best_iteration": booster.best_iteration})

        booster.save_model(paths["model"], num_iteration=booster.best_iteration)
        write_categories(vocabulary, paths["categories"])
        if medians is not None:
            write_medians(medians, paths["medians"])

        # The two tables ship with the model, so they are recorded with it: a
        # booster without the vocabulary it was fitted against scores plausibly
        # and wrongly, and a run that kept only the booster could not be served.
        mlflow.log_artifact(paths["model"])
        mlflow.log_artifact(paths["categories"])
        if medians is not None:
            mlflow.log_artifact(paths["medians"])

    log.info("%s — %d trees -> %s", name, booster.num_trees(), metrics_path)


if __name__ == "__main__":
    main()
