<!--
README OUTLINE — headings only (Phase 00 deliverable).

Each HTML comment states the evidence that section is obligated to carry and the
phase that produces it. They don't render on GitHub, so the page stays clean
while the outline is still being filled in. Delete each one as its section lands.

The ordering is deliberate: the money result leads. Everything below it is
supporting evidence for that number.
-->

# fraud-decision-engine

<!-- One sentence: turns transaction fraud probabilities into cost-weighted
     allow/review/block decisions, and reports the result in dollars saved
     against the rules engine it replaces. -->

---

## The result

<!-- PHASE 06 + 09. Leads the document.
     - The money chart: USD lost per 1,000 transactions.
     - Table: rules baseline vs. model at a naive 0.5 threshold vs. model with
       the expected-value policy. Test set, scored once.
     - One paragraph interpreting it. No metrics here — this section is money. -->

## The problem

<!-- PHASE 00. Compressed from docs/problem-statement.md, which this links to.
     - The three actions, and what "review" operationally means.
     - Latency budget and review capacity.
     - The cost asymmetry: FN scales with amount, FP is ~fixed. This is the
       sentence the whole project hangs on. -->

## Data and splits

<!-- PHASE 01 + 02. Lead with the maturity gap — it's the differentiator.
     - Dataset, base rate, time span.
     - The temporal split diagram, including the 30-day purge and the
       VAL-FIT / VAL-CAL carve, with the reason for each.
     - Why not a random split.
     - The harness sanity check: random scorer lands at the base rate. -->

## Baselines

<!-- PHASE 03.
     - The rules engine, each rule with its rationale. This is the incumbent.
     - Logistic regression as the "is complexity earning its keep" reference.
     - Both scored through the same harness. -->

## Model

<!-- PHASE 05.
     - LightGBM, why trees over neural nets on this data.
     - Tuning approach and search space.
     - The imbalance experiment and its verdict — including if SMOTE lost.
     - PR-AUC and recall@capacity vs. both baselines, on validation. -->

## Calibration and decision policy

<!-- PHASE 06. The intellectual core.
     - Reliability diagram, before and after.
     - Why calibration matters here specifically: the policy multiplies
       probability by money.
     - The per-transaction threshold derivation.
     - The sensitivity sweep over friction cost, with its chart. -->

## Explainability

<!-- PHASE 07.
     - Global SHAP beeswarm.
     - Local waterfalls: a true positive, a false positive, a high-value catch.
     - Phase 01 hypotheses vs. what SHAP actually showed — including where the
       hypotheses were wrong.
     - Reason codes, and why an automated decision needs an explanation. -->

## Serving

<!-- PHASE 08.
     - API contract: POST /score returns probability, decision, reason codes.
     - How the train/serve feature gap was handled, and what it cost in quality.
     - Latency table: p50/p95/p99, at a stated concurrency, on stated hardware.
     - Whether the Phase 00 budget was met. -->

## Monitoring

<!-- PHASE 09.
     - PSI drift report across monthly windows.
     - Month-over-month PR-AUC decay chart.
     - The retraining trigger, justified against the label-maturity constraint. -->

## Limitations

<!-- PHASE 09. Written honestly, sourced from problem-statement.md §6.
     - Friction cost assumed, not measured.
     - Label maturity window estimated.
     - C*/D* are vendor aggregates with undisclosed lookback windows.
     - isFraud = 0 means "nobody disputed", not "legitimate".
     - US ticket-size distribution; rates wouldn't transfer, method would. -->

## What I'd do next

<!-- PHASE 09. Short, specific, and evidence-backed — not a wish list. -->

## Running it

<!-- PHASE 08/09.
     - Prerequisites, Kaggle credentials, `make download`.
     - The make targets in pipeline order.
     - `docker run` for the API. -->

## Repository layout

<!-- The annotated tree, plus one line on why the pipeline stages read and write
     to disk rather than passing dataframes. -->
