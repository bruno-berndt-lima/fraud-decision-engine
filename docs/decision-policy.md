# Decision policy — how probabilities become actions

| | |
|---|---|
| **Status** | Registered before any Phase 06 number exists. |
| **Last updated** | 2026-09-15 |
| **Cost matrix** | `config/cost_matrix.yaml`, version 1 |
| **Fitted on** | `VAL-CAL` (days 141–160) — 56,949 rows, 1,796 positives |
| **Touches test** | Once, in the last section, with everything above frozen |

---

Phase 06 turns the shipped booster's scores into allow / review / block and reports
the result in USD against the rules engine. Every choice that can move that number is
written down here first, for the same reason `experiments.md` registers its bars before
its runs: a decision made after seeing the USD figure is a fit to it.

The economic inputs are assumptions (`problem-statement.md` §4). This document does not
defend them; it fixes how they are used, and the sensitivity section characterises how
much the conclusion depends on them.

## 1. Calibration

**Why it is needed.** The policy multiplies a probability by money. The booster ranks
well, but its outputs are not frequencies, and an overstated 0.30 blocks transactions
the cost matrix says to allow.

**Candidates.** Two, fitted on the booster's predicted probability for `VAL-CAL`:

- **Platt** — a one-feature logistic regression on `logit(score)`. Two parameters;
  strictly increasing, so ranking, PR-AUC and recall@capacity are unchanged.
- **Isotonic** — a non-decreasing step function. Fits any monotone distortion, but each
  step rests on few positives at the tails, and a step assigns one probability to every
  transaction inside it.

The input to Platt is the logit, not the probability: the booster's own link is
logistic, so a logistic fit on the probability itself has the wrong shape.

**How they are compared — chronological folds inside `VAL-CAL`.** Four folds of five
days each. Each fold is scored by a calibrator fitted on the other three. Scoring a
calibrator on the rows it was fitted on is the flattery the `VAL-FIT` / `VAL-CAL` split
exists to prevent, so every calibration figure in this phase is out-of-fold.

Fitting on folds from both sides of the scored one is not causal. It is acceptable
here because the folds measure the method; the calibrator that ships is refitted on all
of `VAL-CAL` and applied only forward.

**Metrics.** Brier score (primary, named in `problem-statement.md` §5), log-loss and
expected calibration error, per fold. ECE uses ten **quantile** bins: at a 3% base rate,
equal-width bins put nearly every transaction in the first one.

**Selection rule.** Isotonic is adopted only if its out-of-fold Brier score beats
Platt's **in every one of the four folds**. Anything less goes to Platt, which is
simpler, cannot create ties, and cannot reorder transactions.

The calibrator is not chosen on USD. Calibration is judged as calibration; choosing it
by the headline quantity would make the §4 rehearsal optimistic by construction.

**Deliverables.** Reliability diagram before and after, the "after" out-of-fold; the
per-fold table; the refitted calibrator saved beside the model. **It ships with the
booster** — `model.txt`, `categories.parquet` and `medians.parquet` become four
artifacts, and serving without the fourth returns uncalibrated probabilities with no
error.

### Result

**Platt.** Isotonic had the lower out-of-fold Brier score in folds 0 and 2 and the
higher in folds 1 and 3, so the rule goes to Platt. Record:
`reports/metrics/calibration.json`; diagram: `reports/figures/reliability_val_cal.png`.

| fold | days | positives | Platt Brier | isotonic Brier |
|---|---|---:|---:|---:|
| 0 | 141–145 | 462 | 0.024905 | **0.024833** |
| 1 | 146–150 | 398 | **0.021330** | 0.021430 |
| 2 | 151–155 | 558 | 0.018246 | **0.018214** |
| 3 | 156–160 | 378 | **0.018187** | 0.018352 |

**The margin is negligible either way.** Pooled out-of-fold, Platt is marginally ahead
on Brier and log-loss and isotonic marginally ahead on ECE. The rule was written for
exactly this case: without a consistent win, the simpler method that preserves the
ranking ships.

| pooled over `VAL-CAL` | Brier | log-loss | ECE |
|---|---:|---:|---:|
| constant base rate, 3.15% | 0.0305 | 0.140 | — |
| uncalibrated | 0.0232 | 0.263 | 0.0210 |
| Platt, out-of-fold | 0.0205 | 0.087 | 0.0018 |
| isotonic, out-of-fold | 0.0205 | 0.089 | 0.0014 |

The first row is a reference, not a run: the scores a model predicting the slice's
fraud rate for every transaction would get.

**The booster is heavily overconfident.** Its log-loss is worse than the constant
predictor's: it ranks well, but the fraud it scores near zero costs more than its
ranking earns. The shipped calibrator is `a = 0.298`, `b = 0.756` — the booster's
log-odds are roughly 3.4 times too extreme. On the diagram, uncalibrated scores near
1e-10 correspond to an observed fraud rate near 0.2%.

**Two consequences carried forward.**

- **Calibrated probabilities rarely fall below about 0.2%.** The §2 threshold falls
  with amount, so very large transactions will clear it almost regardless of score.
  That is the policy doing what it was registered to do, and it is noted here so the
  block rate in §4 is read with it in mind.
- **Base-rate drift inside `VAL-CAL` already moves calibration.** Fold 0 has the
  highest fraud rate of the four and roughly two to two-and-a-half times the ECE of the
  others under both methods: a calibrator fitted on other days underestimates it. The
  shipped calibrator carries `VAL-CAL`'s base rate forward. Nothing here can correct
  that without a later slice; the reliability diagram on test (§7) measures how much it
  costs, and it is a limitation of the headline rather than a defect to fix.

## 2. The expected-value policy

For each transaction, with calibrated probability `p` and `amount` in USD, the expected
cost of each action is:

| action | expected cost | realised cost, label `y` |
|---|---|---|
| allow | `p · (amount + chargeback_fee)` | `y · (amount + chargeback_fee)` |
| block | `(1 − p) · false_positive` | `(1 − y) · false_positive` |
| review | `review + (1 − p) · review_friction` | `review + (1 − y) · review_friction` |

A caught fraud costs nothing beyond the review (`true_positive: 0`).

**Between allow and block**, the policy takes the cheaper. The break-even is the exact
per-transaction threshold:

```
p* = false_positive / (amount + chargeback_fee + false_positive)
```

The `+ false_positive` term is there because a block only costs anything when the
transaction is legitimate. At the version-1 costs, a $20 transaction is blocked above
25.0% and an $8,000 transaction above 0.187%.

**Review.** The gain of reviewing a transaction is

```
gain = min(allow, block) − review
```

On each day, transactions with positive gain are ranked by it and the top ones are
reviewed, up to that day's capacity. Everything else takes the cheaper of allow and
block. This routes the band the roadmap describes — uncertain and material — without a
hand-set band.

**Capacity is per day.** `floor(0.01 × transactions that day)`. Pooling over the slice
would let the policy spend reviews on the days that turned out to hold the fraud,
which an operation deciding in the morning cannot do. Floor rather than round keeps the
constraint hard.

**Block has no capacity.** Nothing limits how many transactions are declined; the block
rate is reported beside every USD figure so that a policy which wins by declining
customers is visible.

**Ties at the capacity cut are prorated**, for every policy in this document. If `r`
slots remain and `m` transactions share the value at the cut, each of the `m` is costed
as `r/m` of its review cost plus `1 − r/m` of its fallback action's cost. The review
count stays exactly at capacity, and the result does not depend on row order.

**Assumption — analysts are perfect.** A reviewed fraud is always caught, a reviewed
legitimate customer always released. This closes `problem-statement.md` §7.3 for this
version rather than answering it. It favours review, and it favours **both** sides of
the comparison: the rules engine acts only through review (§3).

## 3. The rules engine in USD

The rules engine emits integer points, not probabilities, so the §2 policy cannot price
its transactions (`rules-baseline.md`, limitation 3). It acts the way an incumbent with
only rules acts:

- On each day, the highest-scoring transactions are **reviewed**, up to that day's
  capacity.
- Everything else is **allowed**. The engine never blocks.
- Ties at the cut are prorated as in §2. With a handful of distinct scores the cut
  almost always lands inside a block, which is what `CapacityResult.ambiguous_days`
  already exposes.

Giving the engine a block threshold would mean choosing one, and choosing it on
validation is tuning the opponent. No such threshold exists in `rules-baseline.md`, and
none is added here.

## 4. What gets reported

**Four rows**, each in USD lost per 1,000 transactions, with reviews per day and block
rate beside it:

| row | policy | what it isolates |
|---|---|---|
| rules | §3 | the incumbent |
| naive | block if calibrated `p ≥ 0.5`, otherwise allow; no review | the model without a policy |
| EV, uncalibrated | §2 on the booster's raw predicted probability | what calibration is worth |
| **EV** | §2 on the calibrated probability | **the headline** |

"Uncalibrated" is the output of `booster.predict` — a probability, not the log-odds.

**Rehearsal on `VAL-CAL`.** The four rows are computed on `VAL-CAL` first, with the
out-of-fold probabilities from §1. Nothing changes because of what they show; if the
model does not beat the rules here, that is learned before the test touch rather than
from it. A change made after reading the rehearsal is recorded as a change, with its
reason.

**Reading rule, from `problem-statement.md` §5.** A reduction of at least 15% against
the rules engine counts as a win; below 5% is inside the noise of the cost assumptions
and is not claimed.

## 5. Sensitivity

On `VAL-CAL`, out-of-fold probabilities. Never on test — a sweep on test is a threshold
search on test.

**One assumption at a time**, the others at their version-1 values:

| assumption | range | output |
|---|---|---|
| `false_positive` | $5–$100, 20 steps | chart: USD per 1,000 for rules and EV, and the difference |
| `chargeback_fee` | $15–$100, 10 steps | table |
| `review_capacity` | 0.5%, 1%, 2% of daily volume | table |

The ranges are the ones in `cost_matrix.yaml`. **The rules engine is recomputed at every
point**: its actions do not move with the costs, but what they cost does, and capacity
changes which transactions it reviews.

If the advantage over the rules engine disappears anywhere inside these ranges, the
headline is reported as depending on that assumption (`cost_matrix.yaml`,
`problem-statement.md` §5).

## 6. The USD halves owed by E1 and E3

`experiments.md` records both as owing a USD figure. Their boosters were not saved, so
each arm is refitted.

- Each arm is calibrated with **the method §1 selected** — not re-selected per arm —
  using the same four folds, and costed under the version-1 matrix on `VAL-CAL`.
- **This amends the rule that a field of candidates scores `VAL-FIT` only.** The rule
  exists to stop selection. Nothing in E1 or E3 ships, and neither result is allowed to
  change anything, so measuring them on `VAL-CAL` is reporting.
- **E3 reports; it does not decide.** The model served in Phase 08 is the one Phase 05
  elected on `VAL-FIT`. E3's USD figure says what the live-entity store is worth, for
  whoever decides whether to build one. Swapping the served model because of it would
  be a choice made on `VAL-CAL`, and would invalidate every figure in §1–§5.

Neither arm is ever scored on test.

## 7. The test touch

Before it, on a clean tree and committed: this document, the calibrator, the policy
code, and `cost_matrix.yaml` at the version used.

Then one run, recorded with a `git_revision` that carries no `-dirty`:

- the four §4 rows on test, in USD per 1,000 transactions, with reviews per day and
  block rate;
- the reliability diagram on test, as measurement.

After it, no calibrator, cost, threshold, capacity or feature changes. A disappointing
number is written about, not worked on.
