"""Tests for feature and prediction drift.

The registered rules, each against a case whose answer is known: identical windows score
nothing, the aggregate is over the columns the horizon supports, a reordered vocabulary is
refused rather than miscounted, the policy's rates do not depend on labels the horizon
does not have, and the drift-against-decay reading on relationships of known sign.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

from fraud_engine.evaluation.cost import ev_policy, load_costs
from fraud_engine.evaluation.report import load_operating_capacity
from fraud_engine.features.encoders import MISSING
from fraud_engine.models.train import OTHER
from fraud_engine.monitoring.drift import (
    EXCLUDED_ON_HORIZON,
    band,
    check_policy_record,
    column_psi,
    leads_decay,
    policy_rates,
    reference_bins,
    summarise,
    weighted_columns,
)
from fraud_engine.monitoring.drift_plots import plot_drift

CONFIG = yaml.safe_load(Path("config/config.yaml").read_text())
COST_MATRIX = yaml.safe_load(Path(CONFIG["paths"]["cost_matrix"]).read_text())
COSTS = load_costs(COST_MATRIX)
CAPACITY = load_operating_capacity(COST_MATRIX)
PSI = CONFIG["monitoring"]["psi"]
LEVELS = pd.CategoricalDtype(["C", "W", OTHER, MISSING])


def frame(n=2_000, shift=0.0, seed=0, nulls=0.0):
    rng = np.random.default_rng(seed)
    amount = pd.Series(rng.lognormal(4 + shift, 1, n))
    amount[rng.random(n) < nulls] = np.nan
    return pd.DataFrame(
        {
            "TransactionAmt": amount,
            "has_identity": rng.random(n) < 0.25,
            "ProductCD": pd.Series(rng.choice(["C", "W"], n)).astype(LEVELS),
        }
    )


# ---- the convention and the aggregate ---------------------------------------------


@pytest.mark.parametrize(
    ("value", "name"),
    [(0.0, "stable"), (0.0999, "stable"), (0.1, "moderate"), (0.25, "significant")],
)
def test_the_bands_are_the_convention(value, name):
    assert band(value, PSI) == name


RANKING = [
    {"feature": "C13", "tier": "tier_0", "mean_abs": 0.8},
    {"feature": EXCLUDED_ON_HORIZON[0], "tier": "tier_3", "mean_abs": 0.6},
    {"feature": "C1", "tier": "tier_0", "mean_abs": 0.4},
    {"feature": "card1", "tier": "tier_1", "mean_abs": 0.2},
]


def test_the_aggregate_skips_what_the_horizon_cannot_support():
    weights = weighted_columns(RANKING, top_k=2)
    assert weights["feature"].tolist() == ["C13", "C1"]


def test_the_weights_are_contribution_renormalised_over_the_k():
    weights = weighted_columns(RANKING, top_k=3)
    assert weights["weight"].sum() == pytest.approx(1.0)
    assert weights["weight"].tolist() == pytest.approx([0.8 / 1.4, 0.4 / 1.4, 0.2 / 1.4])


# ---- the table ------------------------------------------------------------------


def table_for(reference, window_frame, windows):
    columns = list(reference.columns)
    return column_psi(
        reference, window_frame, windows, reference_bins(reference, columns, 10), PSI["epsilon"]
    )


def test_a_categorical_is_binned_by_its_vocabulary_and_a_number_by_edges():
    bins = reference_bins(frame(), ["TransactionAmt", "ProductCD"], 10)
    assert bins["ProductCD"] is None
    assert len(bins["TransactionAmt"]) == 9


def test_windows_identical_to_the_reference_score_nothing():
    reference = frame()
    table = table_for(reference, reference, np.zeros(len(reference), dtype=int))
    assert (table["psi"] == 0).all()


def test_a_shifted_column_scores_and_the_others_do_not():
    reference, shifted = frame(seed=0), frame(seed=0, shift=0.8)
    table = table_for(reference, shifted, np.zeros(len(shifted), dtype=int)).set_index("column")
    assert table.loc["TransactionAmt", "psi"] > PSI["significant"]
    assert table.loc["ProductCD", "psi"] == 0


def test_a_field_that_stops_arriving_shows_in_its_missing_share():
    reference, silent = frame(seed=0), frame(seed=0, nulls=0.4)
    row = table_for(reference, silent, np.zeros(len(silent), dtype=int)).set_index("column")
    assert row.loc["TransactionAmt", "missing_share_reference"] == 0
    assert row.loc["TransactionAmt", "missing_share"] == pytest.approx(0.4, abs=0.03)


def test_the_unseen_share_is_the_other_level_and_only_for_a_categorical():
    reference = frame()
    window = reference.copy()
    window.loc[window.index[:500], "ProductCD"] = OTHER
    table = table_for(reference, window, np.zeros(len(window), dtype=int)).set_index("column")
    assert table.loc["ProductCD", "unseen_share"] == pytest.approx(0.25)
    assert pd.isna(table.loc["TransactionAmt", "unseen_share"])


def test_a_flag_counts_the_bins_the_reference_populates_not_the_ones_it_cannot():
    table = table_for(frame(), frame(), np.zeros(2_000, dtype=int)).set_index("column")
    assert table.loc["has_identity", "value_bins"] == 2


def test_a_vocabulary_in_another_order_is_refused_rather_than_miscounted():
    reference = frame()
    reordered = reference.assign(
        ProductCD=reference["ProductCD"].cat.reorder_categories(["W", "C", OTHER, MISSING])
    )
    with pytest.raises(ValueError, match="not levelled"):
        table_for(reference, reordered, np.zeros(len(reordered), dtype=int))


def test_every_window_gets_every_column():
    reference = frame()
    table = table_for(reference, frame(seed=3), np.repeat([0, 1], 1_000))
    assert len(table) == 2 * reference.shape[1]


# ---- the summary ----------------------------------------------------------------


def summary_for(psi_by_column):
    table = pd.DataFrame(
        {"window": 0, "column": list(psi_by_column), "psi": list(psi_by_column.values())}
    )
    layout = pd.DataFrame(
        [{"window": 0, "first_day": 213, "last_day": 240, "partial": False, "rows": 9}]
    )
    weights = weighted_columns(RANKING, top_k=2)
    cfg = CONFIG["monitoring"] | {"report_top": 2}
    return summarise(table, layout, weights, RANKING, cfg)[0]


def test_the_weighted_psi_is_the_weighted_sum():
    entry = summary_for({"C13": 0.3, "C1": 0.06, "card1": 0.9})
    assert entry["weighted_psi"] == pytest.approx(0.3 * 0.8 / 1.2 + 0.06 * 0.4 / 1.2)
    assert entry["weighted_band"] == "moderate"


def test_the_leading_columns_are_led_by_contribution_not_by_psi():
    """card1 has the largest PSI and the smallest contribution; it does not lead."""
    entry = summary_for({"C13": 0.01, "C1": 0.02, "card1": 0.9})
    assert [row["column"] for row in entry["leading"]] == ["C13", "C1"]


# ---- the frozen policy, without labels --------------------------------------------


def horizon_scores(n=4_000, seed=0):
    rng = np.random.default_rng(seed)
    return (
        rng.beta(0.5, 12, n),
        rng.lognormal(4, 1.2, n),
        rng.integers(213, 223, n),
    )


def test_the_policy_rates_do_not_depend_on_the_labels():
    p, amount, day = horizon_scores()
    rates = policy_rates(p, amount, day, COSTS, CAPACITY)
    for seed in range(3):
        y = np.random.default_rng(seed).integers(0, 2, len(p))
        labelled = ev_policy(p, amount, day, COSTS, CAPACITY).summary(y, amount, day, COSTS)
        assert labelled["block_rate"] == rates["block_rate"]
        assert labelled["reviews_per_day"] == rates["reviews_per_day"]


def test_every_review_is_eligible_and_eligibility_is_not_capped():
    p, amount, day = horizon_scores()
    rates = policy_rates(p, amount, day, COSTS, CAPACITY)
    reviewed_share = rates["reviews_per_day"] * np.unique(day).size / len(p)
    assert rates["review_eligible_rate"] >= reviewed_share


def recorded_for(test):
    p, amount, day = (
        test["calibrated"].to_numpy(),
        test["amount"].to_numpy(),
        test["day"].to_numpy(),
    )
    summary = ev_policy(p, amount, day, COSTS, CAPACITY).summary(
        test["isFraud"].to_numpy(), amount, day, COSTS
    )
    return {"policies": {"ev": summary}}


def test_a_policy_that_reproduces_the_record_passes():
    p, amount, day = horizon_scores()
    test = pd.DataFrame(
        {"calibrated": p, "amount": amount, "day": day, "isFraud": (p > 0.2).astype(int)}
    )
    check_policy_record(test, recorded_for(test), COSTS, CAPACITY)


def test_a_policy_that_differs_from_the_record_at_all_is_refused():
    p, amount, day = horizon_scores()
    test = pd.DataFrame(
        {"calibrated": p, "amount": amount, "day": day, "isFraud": (p > 0.2).astype(int)}
    )
    record = recorded_for(test)
    record["policies"]["ev"]["block_rate"] += 1e-12
    with pytest.raises(ValueError, match="block_rate"):
        check_policy_record(test, record, COSTS, CAPACITY)


# ---- drift against decay ----------------------------------------------------------


def windows(psis, pr_aucs, readings=None, partial_last=True):
    labelled, decay = [], []
    for i, (w, pr) in enumerate(zip(psis, pr_aucs, strict=True)):
        partial = partial_last and i == len(psis) - 1
        labelled.append(
            {
                "window": i,
                "first_day": 121 + 5 * i,
                "last_day": 125 + 5 * i,
                "partial": partial,
                "weighted_psi": w,
            }
        )
        decay.append({"window": i, "first_day": 121 + 5 * i, "last_day": 125 + 5 * i, "pr_auc": pr,
                      "reading": (readings or {}).get(i, "no decline")})  # fmt: skip
    return labelled, {"baseline": {"pr_auc": 0.52}, "windows": decay}


def test_more_drift_with_more_shortfall_is_a_positive_correlation():
    labelled, decay = windows([0.05, 0.1, 0.2, 0.3, 0.9], [0.6, 0.55, 0.5, 0.45, 0.1])
    reading = leads_decay(labelled, decay)
    assert reading["spearman"] == pytest.approx(1.0)
    assert reading["n"] == 4


def test_the_partial_window_is_left_out():
    labelled, decay = windows([0.1, 0.2, 0.3], [0.5, 0.4, 0.3])
    assert leads_decay(labelled, decay)["n"] == 2


def test_a_flagged_window_is_ranked_by_its_drift():
    labelled, decay = windows(
        [0.05, 0.3, 0.1, 0.2], [0.55, 0.45, 0.5, 0.52], readings={2: "decline"}, partial_last=False
    )
    assert leads_decay(labelled, decay)["flagged"] == [
        {"window": 2, "reading": "decline", "psi_rank": 3}
    ]


def test_windows_cut_differently_by_the_two_stages_are_refused():
    labelled, decay = windows([0.1, 0.2, 0.3], [0.5, 0.4, 0.3])
    decay["windows"][1]["last_day"] += 1
    with pytest.raises(ValueError, match="differs"):
        leads_decay(labelled, decay)


# ---- the figure -------------------------------------------------------------------


def test_the_figure_has_one_axis_per_quantity():
    labelled, _ = windows([0.05, 0.08], [0.5, 0.5])
    horizon = [
        {"first_day": 213, "last_day": 240, "partial": False, "weighted_psi": 0.12,
         "prediction": {"score_psi": 0.02, "mean_calibrated": 0.037, "block_rate": 0.064}},
    ]  # fmt: skip
    record = {
        "psi": PSI,
        "score_reference": {"mean_calibrated": 0.0374, "block_rate": 0.0645},
        "labelled": labelled,
        "horizon": horizon,
    }
    assert len(plot_drift(record).axes) == 4
