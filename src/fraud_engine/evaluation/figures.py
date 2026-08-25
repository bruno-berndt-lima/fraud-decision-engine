"""The baseline comparison figures.

The stage that turns several runs' score vectors into the two figures the README
is obligated to carry. It reads ``predictions_dir``, never a model: a figure is a
view of what a run said, so re-running the model to draw one would make the
picture and the metrics record two independent computations that agree only by
luck.

This module owns the *selection* — which runs, which split, which operating
point. ``plots.py`` owns how a figure looks and is the only matplotlib importer;
importing this module pulls that in, which is why the Phase 08 container has no
reason to touch either.
"""

from __future__ import annotations

import logging
from collections.abc import Mapping
from pathlib import Path

import pandas as pd

from fraud_engine.data.load import DEFAULT_CONFIG_PATH, load_config
from fraud_engine.evaluation.plots import (
    plot_pr_curve,
    plot_recall_at_capacity,
    save_figure,
)
from fraud_engine.evaluation.report import load_operating_capacity

log = logging.getLogger(__name__)

# Run name -> the label drawn beside its curve. Order is the colour order.
#
# `logistic_balanced` is deliberately absent. It is the E2 loser, and this figure
# answers "can a linear model beat the incumbent", not "did class weighting
# help" — a third curve here would put an experimental variant in a comparison
# about the incumbent. The E2 result is a table in docs/experiments.md, which is
# the right shape for a two-row result.
BASELINE_RUNS: Mapping[str, str] = {
    "rules_baseline": "rules engine",
    "logistic_baseline": "logistic regression",
}

# VAL-FIT, matching how E2 reports its table. VAL-CAL would not be leakage —
# drawing a figure changes nothing, and measurement is unlimited — but VAL-CAL
# has one job in Phase 06 and there is no reason to spend a second look on it
# here.
FIGURE_SPLIT = "val_fit"

# Written into figures_dir. Named in docs/README-draft.md under `## Baselines`;
# changing them here means changing them there.
PR_CURVE_FIGURE = "pr_curve_baselines.png"
RECALL_FIGURE = "recall_at_capacity_baselines.png"


def load_predictions(
    names: tuple[str, ...],
    predictions_dir: Path | str,
    split: str = FIGURE_SPLIT,
) -> dict[str, pd.DataFrame]:
    """Read each run's score vector, restricted to one split.

    Args:
        names: Run names, matching the parquet filenames ``write_run`` produced.
        predictions_dir: Where those files live.
        split: Which split to draw. Must be one the runs actually scored — a run
            filters its predictions to what it was asked to score, so asking for
            a split it skipped raises here rather than plotting nothing.

    Returns:
        ``{name: frame}``, each indexed by ``TransactionID`` and sorted.

    Raises:
        FileNotFoundError: If a run has no predictions file.
        ValueError: If a run holds no rows in ``split``.
    """
    frames = {}
    for name in names:
        path = Path(predictions_dir) / f"{name}.parquet"
        if not path.exists():
            raise FileNotFoundError(
                f"no predictions for {name!r} at {path}. Run the stage that writes it "
                f"— `make baselines` — before drawing figures from it."
            )

        frame = pd.read_parquet(path)
        rows = frame[frame["split"] == split]
        if rows.empty:
            raise ValueError(
                f"run {name!r} holds no {split!r} rows. Present in its predictions: "
                f"{sorted(value for value in frame['split'].dropna().unique())}."
            )

        frames[name] = rows.set_index("TransactionID").sort_index()

    return frames


def align(frames: Mapping[str, pd.DataFrame]) -> tuple[pd.Series, pd.Series, dict[str, pd.Series]]:
    """Check the runs scored the same rows, and return the shared axes.

    Overlaying curves drawn on different populations is the failure this exists
    to prevent, and it does not announce itself: two runs that scored different
    row sets still produce two plausible curves on one pair of axes, and the
    comparison is meaningless. Cheap to check, invisible if unchecked.

    Args:
        frames: From ``load_predictions``, each indexed by ``TransactionID``.

    Returns:
        ``(y_true, days, {label: scores})`` — the labels and days taken from the
        first run, since by then every run is known to share them.

    Raises:
        ValueError: If ``frames`` is empty, or two runs scored different rows.
    """
    if not frames:
        raise ValueError("no runs to align; there would be nothing to draw.")

    reference_name, reference = next(iter(frames.items()))
    for name, frame in frames.items():
        if not frame.index.equals(reference.index):
            raise ValueError(
                f"runs {reference_name!r} and {name!r} scored different rows "
                f"({len(reference)} against {len(frame)}). Curves drawn on different "
                f"populations cannot be compared on one pair of axes."
            )

    scores = {BASELINE_RUNS.get(name, name): frame["score"] for name, frame in frames.items()}
    return reference["isFraud"], reference["day"], scores


def main(config_path: Path = DEFAULT_CONFIG_PATH) -> None:
    """Draw the baseline comparison figures from the runs already on disk.

    Wiring only. Invoked by ``make figures`` as
    ``python -m fraud_engine.evaluation.figures``.

    Args:
        config_path: Path to ``config.yaml``.
    """
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    config = load_config(config_path)
    paths = config["paths"]
    figures_dir = Path(paths["figures_dir"])

    capacity = load_operating_capacity(load_config(Path(paths["cost_matrix"])))
    frames = load_predictions(tuple(BASELINE_RUNS), paths["predictions_dir"])
    y_true, days, scores = align(frames)

    label = FIGURE_SPLIT.replace("_", "-").upper()
    log.info(
        "drawing %s on %s — %d rows, %d fraud", ", ".join(scores), label, len(y_true), y_true.sum()
    )

    written = [
        save_figure(
            plot_pr_curve(y_true, scores, title=f"Precision-recall — baselines, {label}"),
            figures_dir / PR_CURVE_FIGURE,
        ),
        save_figure(
            plot_recall_at_capacity(
                y_true,
                scores,
                days,
                operating_capacity=capacity,
                title=f"Recall at review capacity — baselines, {label}",
            ),
            figures_dir / RECALL_FIGURE,
        ),
    ]
    for path in written:
        log.info("wrote %s", path)


if __name__ == "__main__":
    main()
