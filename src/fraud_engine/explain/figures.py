"""The Phase 07 figures, drawn from what the explain stage persisted.

`docs/explainability.md` §5 and §6. Like `evaluation/figures.py`, this module reads
records and never a model: a figure is a view of what a run said, so redrawing one must
not be able to produce numbers the record beside it disagrees with.

**`shap` draws the beeswarm and the waterfalls**, because those are canonical forms a
reader recognises on sight and a hand-rolled waterfall is fiddly work with nothing to
show for it. The tier ranking and the amount dependence are the house style's, in
`explain/plots.py`, for reasons that module gives.

**The waterfalls are annotated with the bar they were decided against.** §7 registers
that a local explanation showing only the score half explains the model rather than the
decision, and `shap` has no way to know a threshold exists — so `p`, the break-even for
that amount, and which side it fell on are written onto the figure afterwards.

**Cases come from the explained rows**, not from all of test. Explaining three more
rows would cost seconds, but it would mean loading the booster here, and a figures
module that can score is one that can disagree with the record. The §6 rule still
selects rather than chooses; it selects within a sixth of the split.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap

from fraud_engine.data.load import DEFAULT_CONFIG_PATH, load_config
from fraud_engine.evaluation.cost import BLOCK, Costs, break_even, ev_policy, load_costs
from fraud_engine.evaluation.plots import INK, INK_MUTED, save_figure
from fraud_engine.evaluation.report import load_operating_capacity
from fraud_engine.explain.contributions import BASE_VALUE
from fraud_engine.explain.plots import plot_amount_dependence, plot_tier_ranking
from fraud_engine.models.train import LABEL, feature_columns, prepare_matrices

log = logging.getLogger(__name__)

SPLIT = "test"
HEADLINE_PREDICTIONS = "headline_test"
AMOUNT = "TransactionAmt"
PRODUCT = "ProductCD"

RANKING_FIGURE = "shap_ranking_by_tier.png"
BEESWARM_FIGURE = "shap_beeswarm_test.png"
DEPENDENCE_FIGURE = "shap_amount_dependence.png"
WATERFALL_FIGURE = "shap_waterfall_{case}.png"

# How many rows each figure shows before the tail is folded away. Not config: they are
# properties of what a page can hold, and nothing downstream reads them.
RANKING_ROWS = 20
BEESWARM_ROWS = 18
WATERFALL_ROWS = 12


def load_contributions(explain_dir: Path | str, split: str) -> pd.DataFrame:
    """One split's persisted contributions, keys included.

    Args:
        explain_dir: Directory holding `{split}.parquet`.
        split: Which split to read.

    Returns:
        `TransactionID`, `day`, the label, one column per feature, and the base value.
    """
    return pd.read_parquet(Path(explain_dir) / f"{split}.parquet")


def as_numeric(values: pd.DataFrame) -> np.ndarray:
    """Feature values as `shap`'s colour axis can take them.

    Categoricals become their codes. The colour then orders levels by a vocabulary
    position that means nothing, which is a limit of the beeswarm rather than of the
    data — and it is exactly the limit §4 is about, since the columns that dominate this
    ranking have no published meaning to colour by either.
    """
    numeric = values.copy()
    for column in numeric.columns:
        if isinstance(numeric[column].dtype, pd.CategoricalDtype):
            numeric[column] = numeric[column].cat.codes
    return numeric.to_numpy(dtype="float64")


def build_explanation(
    contributions: pd.DataFrame, values: pd.DataFrame, columns: list[str]
) -> shap.Explanation:
    """The persisted decomposition, in the object `shap`'s plots take.

    Args:
        contributions: As `load_contributions` returned it.
        values: The prepared feature values for the same rows, in the same order.
        columns: Feature names, in the booster's order.

    Returns:
        An `Explanation` in raw log-odds — never probability, and never calibrated
        probability, per §3.

    Raises:
        ValueError: If the two frames are not the same rows in the same order.
    """
    if len(contributions) != len(values) or not contributions.index.equals(values.index):
        raise ValueError("contributions and feature values are not the same rows, in order")

    return shap.Explanation(
        values=contributions[columns].to_numpy(dtype="float64"),
        base_values=contributions[BASE_VALUE].to_numpy(dtype="float64"),
        data=as_numeric(values[columns]),
        feature_names=columns,
    )


def decide(frame: pd.DataFrame, costs: Costs, capacity: float) -> pd.DataFrame:
    """The headline's own policy, re-applied to its own recorded probabilities.

    Not a second decision: `ev_policy` is deterministic and the inputs are the ones
    `policy_test.json` was written from, so this reproduces what the headline reported
    rather than deciding anything new.

    Args:
        frame: As `headline_test.parquet` holds it.
        costs: The loaded cost matrix.
        capacity: Daily review capacity as a fraction of volume.

    Returns:
        `frame` with `fallback`, `review_share` and `break_even` attached.
    """
    amount = frame["amount"].to_numpy(dtype="float64")
    decision = ev_policy(
        frame["calibrated"].to_numpy(dtype="float64"),
        amount,
        frame["day"].to_numpy(),
        costs,
        capacity,
    )
    return frame.assign(
        fallback=decision.fallback,
        review_share=decision.review_share,
        break_even=break_even(amount, costs),
    )


def select_cases(decided: pd.DataFrame, explained: pd.Index) -> dict[str, int]:
    """The three §6 waterfalls, chosen by rule rather than by eye.

    A typical case is the **median amount** of its kind, not the most convincing one:
    picking the highest-probability catch would be selecting the figure for how well it
    reads, which is the thing the rule exists to prevent. The high-value catch is the
    exception and is meant to be extreme.

    Args:
        decided: As `decide` returned it.
        explained: `TransactionID`s that have persisted contributions.

    Returns:
        `{case: TransactionID}` for `true_positive`, `false_positive` and
        `high_value_catch`.

    Raises:
        ValueError: If any case has no eligible row — a missing waterfall would
            otherwise be a silently shorter figure set.
    """
    blocked = decided[(decided["fallback"] == BLOCK) & decided["TransactionID"].isin(explained)]

    def median_amount(rows: pd.DataFrame, case: str) -> int:
        if rows.empty:
            raise ValueError(f"no explained row qualifies as the {case} case")
        middle = rows["amount"].sub(rows["amount"].median()).abs().idxmin()
        return int(rows.loc[middle, "TransactionID"])

    fraud = blocked[blocked[LABEL] == 1]
    if fraud.empty:
        raise ValueError("no explained row qualifies as the high_value_catch case")

    return {
        "true_positive": median_amount(fraud, "true_positive"),
        "false_positive": median_amount(blocked[blocked[LABEL] == 0], "false_positive"),
        "high_value_catch": int(fraud.loc[fraud["amount"].idxmax(), "TransactionID"]),
    }


def annotate_bar(figure: plt.Figure, case: str, row: pd.Series) -> None:
    """Write the half of the decision `shap` cannot draw onto the figure it drew.

    The threshold falls with the amount, so the same probability blocks a large
    transaction and allows a small one. A waterfall that shows only where the score came
    from is an explanation of the model; §7 requires the bar beside it.

    Args:
        figure: The waterfall `shap` just drew.
        case: Which of the three this is.
        row: The decided transaction — `amount`, `calibrated`, `break_even`, the label.
    """
    outcome = "fraud" if row[LABEL] == 1 else "legitimate"
    figure.suptitle(
        f"{case.replace('_', ' ')} — ${row['amount']:,.2f}, {outcome}",
        color=INK,
        fontsize=11,
        x=0.01,
        ha="left",
        y=1.02,
    )
    figure.text(
        0.01,
        -0.04,
        f"blocked because {row['calibrated']:.2%} clears the bar for this amount, "
        f"{row['break_even']:.2%}. The bar falls as the amount rises; "
        f"the bars above explain the {row['calibrated']:.2%}, not the threshold.",
        color=INK_MUTED,
        fontsize=9,
        ha="left",
    )


def draw_waterfall(explanation: shap.Explanation, position: int) -> plt.Figure:
    """One row's decomposition, as `shap` draws it, handed back unsaved."""
    plt.figure()
    shap.plots.waterfall(explanation[position], max_display=WATERFALL_ROWS, show=False)
    return plt.gcf()


def draw_beeswarm(explanation: shap.Explanation) -> plt.Figure:
    """The conventional global view, as `shap` draws it, handed back unsaved."""
    plt.figure()
    shap.plots.beeswarm(explanation, max_display=BEESWARM_ROWS, show=False)
    return plt.gcf()


def dependence_frame(
    contributions: pd.DataFrame, values: pd.DataFrame, column: str = AMOUNT
) -> pd.DataFrame:
    """What H1 is read from: one column's contribution against its value, per product.

    Args:
        contributions: As `load_contributions` returned it.
        values: The prepared feature values for the same rows, in order.
        column: The feature whose shape is being read.

    Returns:
        Columns `product`, `amount` and `contribution`.
    """
    return pd.DataFrame(
        {
            "product": values[PRODUCT].astype("object").to_numpy(),
            "amount": values[column].to_numpy(dtype="float64"),
            "contribution": contributions[column].to_numpy(dtype="float64"),
        }
    )


def main(config_path: Path = DEFAULT_CONFIG_PATH) -> None:
    """Draw every Phase 07 figure from the persisted record.

    Wiring only. Invoked by `make explain-figures` as
    `python -m fraud_engine.explain.figures`.

    Args:
        config_path: Path to `config.yaml`.
    """
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    config = load_config(config_path)
    paths, model_cfg = config["paths"], config["model"]
    figures_dir = Path(paths["figures_dir"])

    costs = load_costs(load_config(Path(paths["cost_matrix"])))
    capacity = load_operating_capacity(load_config(Path(paths["cost_matrix"])))

    contributions = load_contributions(paths["explain_dir"], SPLIT)
    matrices, _, _ = prepare_matrices(paths["features_dir"], model_cfg, ("train", SPLIT))
    columns = feature_columns(matrices[SPLIT])

    # Positional, not label-based: `.loc` on a three-hundred-column frame reindexes
    # column by column and pandas warns that the result is fragmented, where `.iloc`
    # with an array of positions takes the rows in one pass. The explained rows are a
    # subset of this split in the same order, so the positions are all that is needed.
    position = pd.Series(np.arange(len(matrices[SPLIT])), index=matrices[SPLIT]["TransactionID"])
    contributions = contributions.reset_index(drop=True)
    values = (
        matrices[SPLIT]
        .iloc[position.loc[contributions["TransactionID"]].to_numpy()]
        .reset_index(drop=True)
    )

    ranked = pd.DataFrame(
        json.loads(Path(paths["shap_global"]).read_text())["splits"][SPLIT]["ranking"]
    )
    log.info(
        "%s", save_figure(plot_tier_ranking(ranked, RANKING_ROWS), figures_dir / RANKING_FIGURE)
    )

    explanation = build_explanation(contributions, values, columns)
    log.info("%s", save_figure(draw_beeswarm(explanation), figures_dir / BEESWARM_FIGURE))

    dependence = dependence_frame(contributions, values)
    log.info("%s", save_figure(plot_amount_dependence(dependence), figures_dir / DEPENDENCE_FIGURE))

    # Checked rather than declared as a make prerequisite: the headline refuses to rerun
    # while its record exists, so naming it would leave this stage waiting on a target
    # that can never be satisfied. See the rule's comment in the Makefile.
    predictions = Path(paths["predictions_dir"]) / f"{HEADLINE_PREDICTIONS}.parquet"
    if not predictions.exists():
        raise FileNotFoundError(
            f"{predictions} is absent, so there are no decisions to draw the §6 waterfalls "
            "against. Run `make headline` — it is the stage that writes them."
        )

    decided = decide(pd.read_parquet(predictions), costs, capacity)
    positions = pd.Series(contributions.index.to_numpy(), index=contributions["TransactionID"])

    for case, transaction in select_cases(decided, positions.index).items():
        row = decided.loc[decided["TransactionID"] == transaction].iloc[0]
        figure = draw_waterfall(explanation, int(positions[transaction]))
        annotate_bar(figure, case, row)
        log.info("%s", save_figure(figure, figures_dir / WATERFALL_FIGURE.format(case=case)))


if __name__ == "__main__":
    main()
