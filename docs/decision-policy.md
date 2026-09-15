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

### Rehearsal result

Record: `reports/metrics/policy_val_cal.json`. Platt probabilities out-of-fold, review
capacity 1% of daily volume, cost matrix version 1. Nothing below changed the policy.

| row | USD per 1,000 | reviews / day | block rate | vs rules |
|---|---:|---:|---:|---:|
| rules | 5,140 | 27.95 | 0% | — |
| naive, 0.5 | 4,170 | 0 | 1.28% | −18.9% |
| EV, uncalibrated | 3,946 | 14.25 | 0.90% | −23.2% |
| **EV** | **2,343** | 27.95 | 5.59% | **−54.4%** |

All three model rows clear the 15% bar. A reduction that large was taken apart before
being read, against two references that are not rows of §4 — allowing every
transaction, and an oracle that blocks exactly the fraud:

| | total | fraud allowed | false positives | review | blocked | fraud among blocked | fraud USD stopped |
|---|---:|---:|---:|---:|---:|---:|---:|
| allow everything | 5,362 | 5,362 | 0 | 0 | 0 | — | 0% |
| rules | 5,140 | 5,100 | 0 | 40 | 0 | — | 5.0% |
| naive, 0.5 | 4,170 | 4,133 | 37 | 0 | 731 | 80.7% | 21.2% |
| EV, uncalibrated | 3,946 | 3,918 | 14 | 14 | 512 | 89.6% | 25.7% |
| EV | 2,343 | 1,725 | 575 | 43 | 3,181 | 31.4% | 69.8% |
| oracle | 0 | 0 | 0 | 0 | 1,796 | 100% | 100% |

USD per 1,000 transactions throughout. Nothing points to leakage: the probabilities are
out-of-fold from a booster that never stopped on this slice, the amount is known at
authorisation, fraud and legitimate tickets have similar means ($145 and $138), the saving
appears in every amount band, and review volume matches capacity exactly.

**What the rehearsal supports.**

- **Calibration is worth about $1,600 per 1,000 transactions.** The uncalibrated
  booster's probabilities are so extreme that almost nothing is worth reviewing and
  blocking is timid.
- **The per-transaction threshold is worth 35 points over a fixed 0.5 cut**, on the
  same model and the same probabilities.

**What the headline has to be read with.**

- **Most of the margin is the ability to block.** The rules engine saves 4% against
  allowing everything, because §3 lets it review 28 transactions a day and nothing else.
  A fixed cut that only blocks already takes 19% off it. That is the incumbent this
  project set out to measure against — a rules engine cannot price a transaction — but
  "54% less than the rules" means "a policy that can decline, against one that cannot",
  and is reported that way.
- **The saving is bought with declined customers.** The EV policy blocks 5.6% of
  transactions, and 69% of what it blocks is legitimate: roughly 2,180 good customers in
  57,000 transactions, 4.0% of legitimate purchases. At $15 each that is affordable; the
  $15 is the lowest-confidence number in the cost matrix, and a decline rate that high
  is one a merchant would question. §5 measures how the margin responds when it moves,
  and the headline is not stated without it.

**A note on §3.** The rules score persisted in Phase 03 is total points plus an amount
tiebreaker held below one point, so it has far more than a handful of distinct values —
1,042 on this slice. It is still the engine's own score, and the policy is unchanged:
the highest scores are reviewed, with remaining ties prorated.

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

### Result

Record: `reports/metrics/sensitivity_val_cal.json`; chart:
`reports/figures/sensitivity_false_positive.png`. Same probabilities, capacity and cost
matrix as the rehearsal, which the headline point of every sweep reproduces.

**The advantage never falls below the 15% bar in any sweep.** By the rule above, the
headline does not depend on any one assumption within its registered range. How large
it is depends heavily on one of them.

| assumption | range | smallest EV reduction | at | largest |
|---|---|---:|---|---:|
| false-positive cost | $5–$100 | **34.0%** | $100 | 69.1% |
| chargeback fee | $15–$100 | 54.4% | $24.44 | 56.4% |
| review capacity | 0.5%–2% | 54.0% | 2% | 54.7% |

The fee grid of ten points skips the headline's $25, so that value was added to it;
the sweep has eleven points.

**The false-positive cost sets the size of the margin.**

| false-positive cost | EV, USD per 1,000 | reduction | EV block rate | naive reduction |
|---:|---:|---:|---:|---:|
| $5 | 1,588 | 69.1% | 15.4% | 19.4% |
| **$15** | **2,343** | **54.4%** | **5.6%** | 18.9% |
| $30 | 2,887 | 43.8% | 2.9% | 18.1% |
| $50 | 3,208 | 37.6% | 1.8% | 17.2% |
| $100 | 3,392 | 34.0% | 1.2% | 14.8% |

- **The margin halves across the range but does not vanish.** The EV cost rises
  steeply up to about $40 and then flattens: past that point the policy blocks only
  near-certain fraud, and about a third less than the rules engine is what remains.
  That floor is the part of the headline that does not depend on what declining a
  customer costs.
- **The low end is a policy that wins by declining.** At $5 it refuses 15.4% of all
  transactions. The 69% there is real under the matrix and not a figure to quote
  without its block rate; the same holds, less starkly, at the headline's 5.6%.
- **A fixed cut does not adapt; the per-transaction threshold does.** The naive 0.5
  policy drops below the 15% bar from about $95, while the EV policy stays above 34%.
  Deriving the threshold from the costs is what keeps the result standing when the
  cost assumption is badly wrong.
- **The rules engine is flat across this sweep.** It never blocks, so it never pays a
  false positive; every movement in the margin is the EV policy's.
- The EV curve is not perfectly smooth — $75 costs slightly less than $70. The policy
  minimises expected cost; the chart shows realised cost, under the labels.

**The fee and the capacity barely move the margin.**

- The fee is paid on allowed fraud by both sides, so their costs rise together — the
  rules engine from $4,839 to $7,403 per 1,000, the EV policy from $2,201 to $3,228 —
  and the reduction stays within two points. A higher fee makes the EV policy block a
  little more (5.3% to 7.8%).
- Doubling capacity helps the rules engine more than the EV policy ($259 against $98
  per 1,000), without changing the margin by more than a point. Review is a small
  lever beside blocking, as the rehearsal showed.

**Limitation.** One assumption at a time, as registered. The sweep does not measure
the corner least favourable to the EV policy — a high false-positive cost with a low
fee. The fee moves the margin so little that the corner is unlikely to sit far below
34%, but that is an inference, not a measurement, and is not stated as one.

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

**Method, registered before the run.**

- **The arms are Phase 05's, on Phase 05's instrument.** E1's three training windows
  (`purged`, `recent`, `unpurged`) from the matrices `make purge` built, and E3's `full`
  and `velocity`-removed arms on the shipped matrices, all on the untuned configuration.
  That instrument samples nothing, so a refit returns identical digits: **each refit's
  VAL-FIT scores must equal the predictions Phase 05 recorded, exactly, or the stage
  stops.** A USD figure is never attached to a fit other than the one behind the PR-AUC.
- **The untuned instrument is not the shipped model.** These figures compare arms with
  each other; none of them is comparable to the §4 headline, which is the tuned booster.
- **Each arm is costed under the EV policy** of §2, at the version-1 costs and 1% review
  capacity, with the arm's own out-of-fold Platt probabilities. The rules engine's
  rehearsal cost is reported beside them for scale.
- **The difference is the arm minus its reference,** in USD per 1,000 transactions:
  `recent` and `unpurged` against `purged`; `velocity`-removed against `full`. Positive
  means the arm costs more.
- **Each difference carries a paired day-bootstrap interval:** VAL-CAL's twenty days
  resampled with replacement, 2,000 times, the same days for the arm and its reference;
  the central 95% is reported. **An interval containing zero reads as "not shown to
  differ in USD".** The interval captures which days VAL-CAL happened to hold. It does
  not capture how a refit would move, which E4 found dominates the metric bar — so an
  interval excluding zero is necessary for a difference to be read, not sufficient for
  it to be believed.
- **Expected readings, carried from `experiments.md`.** E1: the unpurged arms cost less,
  since they are handed recency and labels production does not have; the size of that
  saving is what the purge costs in dollars. E3: if the interval for removing the
  velocity family contains zero, the reported result is *the live-entity store cannot be
  shown to pay for itself in USD either*, not that velocity is worthless.

### Result

Record: `reports/metrics/usd_halves.json`; the full readings are in `experiments.md`
under E1 and E3. **Every refit reproduced its Phase 05 VAL-FIT scores to the digit**,
and the clip reached no score on the untuned instrument.

| experiment | arm | USD per 1,000 | vs reference | 95% interval | read |
|---|---|---:|---:|---|---|
| E1 | `purged` (reference) | 2,569 | — | — | — |
| E1 | `recent` | 2,289 | −280 | [−438, −124] | yes |
| E1 | `unpurged` | 2,214 | −355 | [−460, −257] | yes |
| E3 | `full` (reference) | 2,569 | — | — | — |
| E3 | velocity removed | 2,467 | −102 | [−171, −29] | sign not read |

The rules engine's rehearsal cost, for scale: 5,140.

- **E1: the purge costs $280 to $355 per 1,000 in claimed saving.** The registered
  direction holds and both intervals are clear of zero.
- **E3: the store cannot be shown to pay for itself in USD.** The measured difference
  runs against it, but the arms stopped at 561 and 283 rounds, and the interval does
  not see refit variation; the registered rule does not let that sign be read.

Nothing here changes the served model, the policy, or any figure in §1–§5.

## 7. The test touch

Before it, on a clean tree and committed: this document, the calibrator, the policy
code, and `cost_matrix.yaml` at the version used.

Then one run, recorded with a `git_revision` that carries no `-dirty`:

- the four §4 rows on test, in USD per 1,000 transactions, with reviews per day and
  block rate;
- the reliability diagram on test, as measurement.

After it, no calibrator, cost, threshold, capacity or feature changes. A disappointing
number is written about, not worked on.
