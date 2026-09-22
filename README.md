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

The model runs behind a FastAPI service in a container: `POST /score` for a decision,
`GET /health` for liveness and provenance, `POST /explain` for the adverse-action notice.
The full record, with every choice registered before its number existed, is
[`docs/serving.md`](docs/serving.md).

### The contract

A request carries **433 fields** — the 349 features the booster holds, less what the
pipeline builds, plus the 339 V columns the reduction consumes and never shows. The
Pydantic model is derived from the booster's feature names at startup, so the contract
cannot drift from the model it serves.

An omitted field is a null, not an error, because that is what this data looks like: the
first transaction of the interim table carries **199 of 433 non-null**. A contract
demanding them all would reject the dataset it was built from. Four fields are required,
three may not be null, and unknown field names are refused — a 422 naming the field,
never a 500.

`/score` returns the action, the calibrated probability, **the break-even probability for
this amount**, review eligibility with the expected saving behind it, which path decided,
and the identity of every artifact that produced it.

**Review is eligibility, never an outcome.** Whether a transaction reaches an analyst
depends on the other transactions that day, which is queue state a single request cannot
see. The response carries what a queue needs to rank by and stops there.

**Reason codes are not on `/score`.** Explaining a decline costs about **960 ms** against
**30 ms** to decide it, and the policy declines 6.45% of transactions — putting a
900 ms call inside the p95 it would have to fit under. The notice moved to its own
endpoint with its own budget, because §3.1 prices an authorisation and an adverse-action
notice is not one.

### The train/serve feature gap

`docs/features.md` sorts all 349 model features into four serving tiers:

| tier | what it needs | columns |
|---|---|---:|
| 0 | nothing — inherited, unreproducible | 295 |
| 1 | the request itself | 36 |
| 2 | a static table shipped with the model | 14 |
| 3 | live entity history at request time | 4 |

**Tier 3 is the problem, and it is the velocity family.** Those four columns need a
per-`card1` running window updated on every transaction — a feature store that was to be
costed rather than built. Two experiments had already failed to show it pays: the live
entity store could not be shown to earn its keep on PR-AUC, and could not be shown to
pay in USD either.

**The choice: serve the family's own no-history values.** Not zeros, and not invented for
serving — a card's first sighting takes a trailing count of one in every window and the
configured first-seen recency, which are values the model was fitted against rather than
a hole punched in the matrix. Callers that *do* have the history may supply it, and the
response declares which tier-3 inputs arrived, so a scored-with-history decision is
distinguishable from one scored without.

**What it costs was measured, not asserted.** Scoring VAL-CAL twice with the same frozen
booster — once as built, once with those four columns neutralised:

- PR-AUC falls **0.00729**, from 0.52145 to 0.51416
- cost rises **$56.19 per 1,000**, 95% interval **[−18.20, +123.99]**

The interval contains zero, so the serving default **cannot be shown to cost anything in
USD**. Only 0.67% of the slice already held those values, so the null is not the default
being quietly true already — and the booster does use the family, ranking one of the four
in its top ten contributors.

**The larger exposure is tier 0, and it is inherited.** 295 of 349 features are Vesta's
pre-computed aggregates over lookback windows that were never published; they carry
**57.2% of the contribution mass** on test. Keeping only the columns this project could
rebuild from scratch retains **53.8%** of PR-AUC. That is a property of the dataset, not
of the code, and it cannot be bounded from inside this repository.

### Latency

**p95 under sustained concurrent load**, 30 seconds per cell with 5 discarded as warmup,
one thread per worker, replaying test transactions. 16-core Intel i9-9980HK under Docker
Desktop — a Linux VM with all 16 cores — and the load generator sharing the machine,
which inflates the tail.

| workers | in flight | req/s | p50 | p95 | p99 | fell back | |
|---:|---:|---:|---:|---:|---:|---:|:--|
| 1 | 1 | 35.6 | 27.3 | **31.0** | 37.3 | 0.0% | met |
| 1 | 2 | 34.2 | 57.2 | **79.4** | 93.5 | 0.2% | met |
| 1 | 4 | 14.6 | 267.6 | 345.3 | 375.9 | 100% | missed |
| 1 | 8 | 19.2 | 408.7 | 544.7 | 612.9 | 100% | missed |
| 1 | 16 | 30.1 | 522.7 | 705.9 | 766.9 | 100% | missed |
| 1 | 32 | 37.6 | 839.3 | 1102.4 | 1192.7 | 100% | missed |
| 4 | 1 | 31.6 | 32.2 | **36.6** | 54.6 | 0.0% | met |
| 4 | 2 | 62.2 | 29.6 | **45.1** | 70.5 | 0.1% | met |
| 4 | 4 | 75.8 | 36.7 | 107.6 | 209.0 | 3.7% | missed |
| 4 | 8 | 37.2 | 80.9 | 687.9 | 842.6 | 39.3% | missed |
| 4 | 16 | 31.0 | 425.9 | 1199.5 | 1357.3 | 58.9% | missed |
| 4 | 32 | 31.4 | 1179.3 | 2023.3 | 2427.5 | 69.9% | missed |

Milliseconds. No request errored in any cell.

**The budget is `p95 < 100 ms`. It is met at one and two requests in flight, and missed
from four.** Both are reported, because repairing a miss by lowering the concurrency
until it passes is not an answer.

Per request the service is about three times faster than it has to be. The misses are
more requests in flight than workers to run them, which is arithmetic: four concurrent
CPU-bound calls of ~30 ms cannot leave one worker inside 100 ms. **Where the operating
point actually is:** test holds 61,585 transactions over 22 days — 2,799 a day, **0.032
requests per second**, which at ten times peak is 0.01 requests in flight. Two orders of
magnitude below the first missed row.

**Past the knee, throughput falls rather than flattening.** One worker serves 35.6 req/s
with a single request in flight and 14.6 with four — offered more work, it completes less
than half as much. That is the fail-open guarantee turning on itself: a breach releases
the caller, but a LightGBM prediction cannot be interrupted, so the worker finishes a
score nobody will read *and then* runs the rules decision. Every request in breach costs
the server both paths, which is what makes the next one breach. Shedding on queue depth
would fix it, and is listed under [what I'd do next](#what-id-do-next) rather than
changed here — the frozen set was fixed before this number existed.

### Failure, and what ships

**Fail open, to rules.** A missing artifact leaves the service up and degraded; a raise
anywhere in the scoring path falls back to the incumbent; a breach of the budget returns
the incumbent's decision on time. Declining every transaction during a model outage
converts an availability incident into a total revenue outage. `/health` counts the
fallbacks, because a service answering every request from the incumbent is otherwise
indistinguishable from a healthy one.

**Ten artifacts ship, not five** — the booster, its vocabulary and fill values, the
calibrator, three fitted tables the transform needs, the rules constants, the cost matrix
and the reason dictionary. Each fails *quietly* when absent, so the image refuses to build
without them, checking the manifest against the list the code defines rather than a copy.

The image is multi-stage on a slim base, runs as a non-root user, excludes `shap` and
`matplotlib`, and pins one thread per worker — which inverts the usual advice and is a
measurement: a single row gives LightGBM nothing to parallelise, and more threads made
the identical call slower.

`make image` builds it, `make serve` runs it, `make loadtest` reproduces the table above.


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
