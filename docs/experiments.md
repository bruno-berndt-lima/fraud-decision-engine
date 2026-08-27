# Planned experiments

Comparisons this project has committed to running and reporting **either way**,
including where the result is unflattering or boringly null.

Each entry is recorded before the run, with its method and what would count as a
result, so that neither the design nor the reporting can be adjusted after seeing
the number. An experiment invented after the fact to explain a result is not
evidence; one registered beforehand is.

Results land back in this file as each phase closes.

---

## E1 — What the purge gap costs

**Status:** registered, not yet run. Phase 05.

**Question.** How much of the model's apparent performance comes from the
label-maturity purge being absent? Equivalently: what is methodological honesty
worth, in PR-AUC and in USD?

**Why it is worth a run.** The purge is currently an *argument* — the 30-day gap
in A1, justified from how chargebacks arrive rather than from anything measured.
It is the most defensible decision in the project and the one with no number
behind it. A single extra training run turns "we did the correct thing" into "the
correct thing cost this much", which is a materially better answer to the
question an interviewer will actually ask.

**Method.** Two runs against the identical harness, differing in one config value:

| | TRAIN | gap | VAL-FIT | VAL-CAL |
|---|---|---|---|---|
| shipped (`gap_days: 30`) | 1–90 — 315,927 rows, 10,702 fraud | 91–120 | 121–140 | 141–160 |
| ablation (`gap_days: 0`) | 1–120 — 414,542 rows, 14,600 fraud | — | 121–140 | 141–160 |

Dropping the gap hands the model 31% more rows and 36% more fraud.

**Two constraints on the comparison, both load-bearing:**

1. **Evaluation slices are identical in both runs.** Only the training window
   moves. If VAL shifted too, the two numbers would not be comparable and the
   experiment would measure nothing.
2. **It runs on `VAL-FIT`, never on `TEST`.** This is a validation-level
   comparison. Scoring the ablation against test would spend the single test
   touch on a run that is not the shipped model.

**What gets reported.** PR-AUC and recall@capacity on `VAL-FIT` for both, and in
Phase 06, USD per 1,000 transactions for both under the same policy.

**Expected direction — and the sanity check hidden in it.** The unpurged run
should score *better*. It is handed two advantages production never has: recency
(training up to the validation boundary) and labels that could not exist yet
(this dataset ships matured labels, so days 91–120 are cleaner than reality would
supply). **If the unpurged run does not beat the purged one, suspect the harness
before believing the result** — that ordering is close to guaranteed, so its
absence is evidence of a bug, most likely a boundary or leakage error in
`splits.py`.

**What the result does not do.** It does not change which split ships. The
purged split is the headline regardless of how large the gap turns out to be,
because the project's headline claim is a deployment claim. A large delta makes
the purge more worth writing about, not less worth doing.

**Design consequence for `splits.py`.** The two runs must differ by a config
value and nothing else. That means boundaries are anchored on the **start of
`VAL-FIT`**, with the end of `TRAIN` derived backwards through the gap —
something like `train_end = val_fit_start - gap_days - 1` — rather than every
boundary being written out as a literal. Hardcoding `train_end: 90` alongside
`gap_days: 30` states the same fact twice, and setting the gap to 0 would then
leave days 91–120 belonging to no split at all: silently discarded, and the
ablation would measure a smaller training set rather than an unpurged one.

---

## E2 — Class weighting versus doing nothing

**Status:** first run complete — logistic regression, Phase 03. LightGBM run
still to come in Phase 05.

Split in two because the answer is not expected to be the same. A tree ensemble
optimising a ranking metric is fairly indifferent to class weights; a linear
model is not, because weighting changes the coefficients themselves. Reporting
one run and generalising from it would be the mistake.

At a 3.5% positive rate the reflex is to resample. This project does not, by
default: class weighting is run as an explicit A/B against unweighted training
and the result recorded either way, rather than SMOTE being applied because the
problem is imbalanced. `class_weight='balanced'` against `None` for the Phase 03
logistic regression, and LightGBM's `scale_pos_weight` against no weighting in
Phase 05. Same harness both times, PR-AUC and recall@capacity on `VAL-FIT`.

Reported even if the answer is "no meaningful difference", which is the likely
outcome for a tree ensemble on a well-specified ranking metric — and is itself
the point worth making.

### Result — logistic regression, Phase 03

**Class weighting made it worse.** Not "no meaningful difference": a clear loss
on the metrics this project selected in advance.

| | PR-AUC | ROC-AUC | recall @ 1% |
|---|---:|---:|---:|
| `class_weight=None` | **0.3217** | 0.8259 | **16.2%** |
| `class_weight='balanced'` | 0.2935 | **0.8271** | 14.3% |

VAL-FIT, base rate 0.0346. VAL-CAL agrees and the gap widens: 0.2289 against
0.1954 PR-AUC, 13.9% against 10.9% recall.

**The two metrics disagree, and that is the interesting part.** ROC-AUC is
marginally *better* weighted; PR-AUC and recall@capacity are clearly worse.
Weighting inflates the loss contribution of the 3.5% positive class, which pulls
the decision surface toward separating classes on average — what ROC-AUC
rewards. What this project needs is precise ranking in the top 1% of scores,
and that is where the reweighting costs accuracy. Had this project reported
ROC-AUC as primary, the same run would have looked like a small win.

**Consequence.** The Phase 03 baseline of record is the unweighted run. The
weighted variant stays in `reports/metrics/logistic_balanced.json` — recording
the loss is the point, so it is not deleted.

**Carry into Phase 05.** This does not predict the LightGBM answer and must not
be used to skip it. A linear model's coefficients move under reweighting; a tree
ensemble's split ordering is far less sensitive, so "no meaningful difference"
remains the expected outcome there. Two model families, two runs, two results.

---

## E3 — Servable features versus entity history

**Status:** registered, not yet run. Phases 04–08.

Committed in `problem-statement.md` §3.4. A scoring request carries the
transaction and its immediate attributes, not the card's history, so velocity and
entity-aggregate features require an online store that does not exist here.

Both feature sets get built and evaluated. The model behind the API uses only
request-computable features; the history-dependent model runs alongside it, and
the difference in PR-AUC and USD saved is reported as **the measured cost of not
building a feature store** — a more useful result than either silently training
on unservable features or quietly dropping them.

---

## E4 — What each feature family measurably adds

**Status:** detection threshold registered, floor measured, first family
reported. Phase 04.

**Question.** Of the feature families this phase builds — amount transforms,
frequency encodings, entity aggregates, velocity, V-block representatives —
which actually add signal, and which only appear to?

**Why it needs registering in advance.** The roadmap's trap for this phase is
"adding features until the metric moves, with no hypothesis". The defence is not
willpower; it is deciding *before* the first family exists how large a movement
has to be to count. Otherwise every family looks like it worked, because on a
metric with this much variance every family moves the number.

**This experiment does not select features.** Every family built in this phase
ships in the feature matrices regardless of what it measures here, and Phase 05
trains on all of them. A gradient-boosted tree is robust to columns it cannot
use, and pre-selecting with a linear probe would discard precisely the features
a tree exists to exploit. What follows is a **detection threshold** — how large a
movement has to be before it is worth believing — not a gate. Nothing is
accepted or rejected; families are characterised.

### Method

Families are evaluated as **groups**, never column by column. One run per family
through the Phase 02 harness, each adding only that family's columns to a fixed
probe.

**The probe is the Phase 03 logistic pipeline**, with `class_weight=None` —
E2's stronger variant — extended by a single `ColumnTransformer` branch that
median-imputes and scales the family's columns. Fixing the probe is what makes
the comparison about features: a probe whose baseline moved between runs would
be measuring itself.

**`VAL-FIT` only.** Choosing which features ship is tuning, and `VAL-CAL` is held
back so the Phase 06 calibrator and threshold meet data no tuning decision has
touched. `report.write_run` scores both validation slices by default, so the
family runs name their splits explicitly.

**Baseline.** `family_none` — the probe with no family added — scores **0.32169**
PR-AUC on `VAL-FIT`, reproducing `logistic_baseline.json` to full precision. That
equality is the check that the Phase 04 feature matrices did not perturb the
Phase 03 result.

### The detection threshold

> A family's delta is distinguishable from chance only if it exceeds the
> **largest** PR-AUC gain achieved by the same number of columns of pure
> Gaussian noise, over 20 seeds.

Measured rather than assumed, because the number turned out to be large.
Twenty seeds, one meaningless column each, against a 0.32169 baseline:

| | delta vs `family_none` |
|---|---:|
| mean | +0.00094 |
| std | 0.00142 |
| min (seed 5) | −0.00143 |
| p95 | +0.00283 |
| **max (seed 0)** | **+0.00433** |
| above baseline | 14 of 20 |

So a column carrying no information at all can add **+0.0043 PR-AUC** — about
1.3% relative. Any family claiming less than that has demonstrated nothing.

**The max rather than the 95th percentile.** A 5%-per-family error rate across
roughly six families gives a 26% chance of calling at least one pure-noise family
real somewhere in the sweep. The max costs almost nothing in sensitivity and
removes that.

**The floor is measured at the family's own width.** More columns are more
chances for the fit to read signal into noise, so the floor rises with width; a
width-1 floor would understate what a six-column family has to clear. Each
family's sweep is run at its own width and recorded beside it.

**An observation, recorded rather than explained away.** 14 of 20 noise runs
landed *above* baseline, and the mean delta is positive at +0.00094 — a sign test
gives p ≈ 0.06. Marginal, and not pursued: the threshold uses the maximum, which
is unaffected by a small shift in the centre.

### What gets reported

PR-AUC and recall@capacity on `VAL-FIT` for every family, **including — and
especially — the ones that move nothing.** A family below the threshold is
reported as below it, with the reason where one is known.

### Blind spots, registered in advance

The probe is linear, so a family whose relationship with fraud is non-monotonic,
step-shaped, or symmetric cannot register no matter how real it is. Naming these
before the runs is what keeps a null result from being read as evidence of
absence:

| family | shape of the relationship | linear probe can see it? |
|---|---|---|
| amount / round bands | spiky, and 0.33% of rows | no |
| frequency encoding | non-monotonic on train: 0.75x, 0.98x, 1.24x, 0.95x, 1.08x by rarity quintile | no |
| velocity | step-like — one transaction is normal, eight is not | barely |
| entity deviation | symmetric in the absolute z-score | not unless the absolute value is built |
| V-block reduction | a dense, largely linear block | yes |

A null result on a row marked "no" is uninformative about the feature and
informative about the probe. Phase 05 is where those families get a fair test.

### What the result does not do

**It does not settle anything for Phase 05.** The probe is linear. A family whose
value lies in interactions — which is most of what a gradient-boosted tree is for
— will read flat here. A family below the bar is *unproven against a linear
probe*, not useless, and stays available for Phase 05 to re-test rather than
being deleted.

**The floor does not transfer.** +0.0043 is the noise floor for this probe on
this metric on this split. LightGBM needs its own if the same question is asked
of it.

### Result — the noise floor, measured twice

Twenty seeds at width 1 and twenty at width 3, against `family_none` = 0.32169:

| width | mean | std | min | max | p95 | above baseline |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | +0.00094 | 0.00142 | −0.00143 | +0.00433 | +0.00283 | 14 / 20 |
| 3 | −0.00005 | 0.00155 | −0.00356 | +0.00257 | +0.00216 | 8 / 20 |

**Two corrections to the registration above, both from the data.**

*The floor does not rise with width* — at least not between 1 and 3. The standard
deviation moves 0.00142 → 0.00155, and the maximum falls. At 316,000 training
rows under L2 at `C=1.0`, three meaningless columns are not meaningfully more
dangerous than one. The instruction to re-measure per family width is therefore
withdrawn: `n_columns` stays, because a fifteen-column V-block family may yet
behave differently, but a fresh twenty-seed sweep per family costs nine minutes
and buys almost nothing.

*The upward shift was chance.* The registration recorded 14 of 20 above baseline
at width 1 (sign test p ≈ 0.06) and declined to pursue it. At width 3 it is 8 of
20 and the mean is −0.00005. It did not replicate.

**The bar, revised.** Pooling all 40 draws: mean ≈ +0.0004, σ ≈ 0.0015, largest
observed +0.0043. Max-observed and mean-plus-3σ agree closely, so:

> A family must add **≥ +0.005 PR-AUC** on `VAL-FIT` to be believed.

Stated as one number rather than a per-width maximum because the maximum of
twenty draws is itself unstable — it moved by more between widths than the
underlying spread did.

### Result — the amount family: no measurable gain

| | PR-AUC | ROC-AUC | recall @ 1% | recall @ 2% |
|---|---:|---:|---:|---:|
| `family_none` | 0.32169 | 0.82585 | 16.23% | 23.92% |
| `family_amount` | 0.32054 | 0.82092 | 16.53% | 24.47% |
| delta | **−0.00115** | −0.00493 | +0.00302 | +0.00553 |

**Below the threshold, and negative.** It sits about 0.7σ from the width-3 mean
— indistinguishable from three columns of noise, in the unhelpful direction. The
columns stay in the matrices; this is a measurement, not a removal.

**Why, and it is not because the feature is wrong.** H2's signal is real and it
generalises: `amt_round_band_product` carries 6.02× lift on train and **5.71× on
`VAL-FIT`**, data it was never fitted against. It is simply tiny — 187 of 57,464
rows, holding 37 frauds, **1.9% of all fraud in the split**. PR-AUC scores the
whole ranking. A feature touching a third of a percent of rows cannot move it,
however good it is on those rows.

That is a coverage result, not a quality one, and it restates H3's note that risk
ranking and money ranking disagree: high lift and high value are different
quantities.

**H2 is not falsified by this.** H2 predicts a *SHAP* result in Phase 07 — spiky
contributions at $150 / $300 / $450 — and its own falsifier held here exactly as
written: `amt_whole_dollar` is worth 1.02× on train and 1.05× on `VAL-FIT`,
nothing. A linear probe failing to profit from an indicator says little about
what a gradient-boosted tree will do with it. The columns stay in the matrices
for Phase 05 to re-test.

### Registration defect, recorded rather than repaired quietly

Recall@capacity moved **up** while PR-AUC moved down: +6 frauds caught at the 1%
capacity, +11 at 2%. That is the operationally meaningful direction, and it is
also exactly what this experiment exists to stop anyone claiming — **there is no
noise floor for recall@capacity**, so six frauds out of 1,990 may be nothing.

The defect is real and predates the result: this rule was registered naming
PR-AUC alone, while the project names *two* primary metrics and a USD headline.
It should have specified both from the start.

**Resolution, settled before the next family was built** — deliberately, so that
it is a rule and not a reaction to the number above.

> **The threshold applies to PR-AUC alone.** Recall@capacity is measured against
> its own floor and reported for every family, but no claim is made from it.

Two reasons, neither of which is "PR-AUC gave the answer we already had".

*One threshold, so there is nothing to shop for.* The moment two metrics can
each vindicate a family, every borderline family gets argued on whichever one
flatters it, and the pre-registration stops doing any work.

*PR-AUC is the more stable statistic.* It is threshold-free and scores the whole
ranking. Recall@capacity is a single cut over roughly 570 reviewed rows, so it
carries more variance per unit of signal — a poor instrument for detecting the
small effects feature families produce.

The narrowing costs little, because **recall@capacity is not being ignored — it
is being optimised properly later.** Phase 06 derives the decision threshold
against a real cost matrix on `VAL-CAL`. Reading feature claims off it in Phase
04 would be a worse version of a job that phase does with the right instrument.

**Consequence, stated plainly.** The amount family's +6 frauds at the 1% capacity
are recorded as an observation, not a result.

### Handoff to Phase 05

Phase 04 hands Phase 05 **every family it builds**, with a measured delta and a
stated blind spot beside each. The numbers here are evidence about what a linear
model can use; they are not a feature-selection decision, and Phase 05's DoD
contains no ablation of its own. Running the family comparison again with the
trained LightGBM — where the fits are already being paid for, and where E3's
servable-versus-history question needs a model that can actually use velocity —
belongs there rather than here.

### Result — the frequency family: worse than noise

Seven columns — `card1`, `card2`, `card3`, `card5`, `addr1`, `addr2`,
`DeviceInfo` — each encoded as its level's share of the training rows.

| | PR-AUC | ROC-AUC | recall @ 0.5% | recall @ 1% | recall @ 2% |
|---|---:|---:|---:|---:|---:|
| `family_none` | 0.32169 | 0.82585 | 10.05% | 16.23% | 23.92% |
| `family_frequency` | 0.31652 | 0.82269 | 9.75% | 16.08% | 24.12% |
| delta | **−0.00517** | −0.00316 | −0.00302 | −0.00151 | +0.00201 |

**Below the minimum of all 40 noise draws** (−0.00356). Seven columns of
frequency encoding cost this probe more than seven columns of pure noise would.

**The blind spot registered above fired exactly as written.** Rarity's
relationship with fraud is not monotonic — by quintile on train, `card1` runs
0.75x, 0.98x, **1.24x**, 0.95x, 1.08x. A linear model can only fit a monotonic
term to that, so it does not merely fail to gain: it spends coefficients on a
shape it cannot represent, adding variance with no signal to pay for it. At
`C=1.0` nothing shrinks those coefficients to zero.

**Caveat on the strength of the claim.** The floor was measured at widths 1 and
3; this family is width 7. Against the pooled σ of 0.00156 the delta is about
3.3σ, which indicates real harm rather than chance — but the extrapolation to
width 7 is not itself measured, so "worse than noise" is well-indicated and not
certified.

**What this does not say.** Frequency encoding is standard practice for
gradient-boosted trees, and for a good reason this result illustrates rather than
contradicts: a tree splits on the *value* of a frequency, so a non-monotonic
relationship is exactly what it handles and exactly what a linear term cannot.
The result here is about the probe, and it is the cleanest demonstration in the
phase of why a null — or negative — reading from a linear probe is uninformative
about a feature.

The seven columns stay in the matrices, and `models/encoders.parquet` carries the
fitted rates. That file is what makes the family **tier 2** on the serving table:
a static lookup shipped beside the model, no online store, no entity history.

**Method note.** Measuring the floor was split out of the family run
(`make floor` against `make families`) after this result. The floor is
deterministic given its seeds, so recomputing it per family run spent eighteen
minutes reproducing a constant; family runs now take under ninety seconds, which
is what makes the remaining families cheap to evaluate.

### Result — the entity family: negative, and the best of the three on recall

Six columns over two entities — `card1` and `addr1` — each contributing the
entity's typical amount, the signed z-score of this transaction against it, and
that z-score's absolute value.

| | PR-AUC | ROC-AUC | recall @ 0.5% | recall @ 1% | recall @ 2% |
|---|---:|---:|---:|---:|---:|
| `family_none` | 0.32169 | 0.82585 | 10.05% | 16.23% | 23.92% |
| `family_entity` | 0.31718 | 0.82197 | 10.10% | 16.73% | 24.47% |
| delta | **−0.00451** | −0.00388 | +0.00050 | +0.00503 | +0.00553 |

**Below the floor's observed minimum** (−0.00356), about 2.9σ below zero.

**Predicted, and for the registered reason.** |z| against `card1` is flat across
four quintiles and then spikes — on `VAL-FIT`, 0.84x, 0.87x, 0.72x, 0.83x,
**1.75x**. The signal is real, it is on rows the fit never saw, and it covers a
fifth of the data. A linear term cannot represent flat-then-spike any better than
it could represent the frequency curve, so it spends coefficients on the shape it
can reach and pays for them.

**It is the strongest family so far on recall@capacity** — +0.50pp at the 1%
capacity, +0.55pp at 2%, and the only one that does not lose ground at 0.5%.
Under the rule settled above this is an observation and no claim is made from it.
Recording it is the point.

**Design note: shrinkage, and what testing it caught.** Per-entity means and
spreads are pulled toward the training window's own by `(n·observed + k·prior) /
(n + k)` at k=10. That stops two degenerate readings: a card seen once would
otherwise have a mean equal to its only amount, a z-score of exactly zero, and
would read as perfectly typical on one observation; and a card whose amounts
never vary would have zero spread and divide by it.

Writing the tests found that the second guarantee was conditional and the
docstring had stated it flatly: shrinkage keeps an entity's spread positive only
if the *prior* is positive, and a training window with no amount variance gives
0/0. `fit_amount_stats` now refuses that input rather than emitting NaN into a
stage that promises null-free columns. Unreachable on real data; the point is
that the claim now matches the code.

`models/amount_stats.parquet` carries the fitted table with its own fallback row,
making this family **tier 2** on the serving table alongside frequency encoding.

### The three families together — the actual Phase 04 result

| family | coverage | shape of the relationship | PR-AUC delta |
|---|---:|---|---:|
| amount | 0.33% of rows | spiky | −0.00115 |
| frequency | 100% | non-monotonic by rarity | −0.00517 |
| entity | 100% | flat, then a spike | −0.00451 |

Three families, three negative deltas, **all three predicted in advance** by the
blind-spot table registered before any of them was built. The two with full
coverage are the two that hurt most, and that is the expected direction rather
than a puzzle: a linear model handed six or seven columns whose shape it cannot
represent does not ignore them — it fits coefficients to the part it can reach,
and at `C=1.0` nothing shrinks those to zero. Having the feature is worse than
not having it, for this model.

**This is the phase's result, and it is not "the features do not work".** Each
family carries measured signal that generalises to `VAL-FIT` — 5.71x lift on the
amount interaction, 1.75x on the top |z| quintile — and each is invisible or
costly to a linear probe for a reason stated before it was measured. The
instrument is the finding.

**It also makes the Phase 03 logistic baseline more useful than it was.** The gap
between it and the Phase 05 model is now partly *explained* rather than merely
observed: part of what a gradient-boosted tree buys on this dataset is the
ability to use features whose shape a linear model cannot express.

**Registered prediction for Phase 05.** Retraining these families against the
tuned LightGBM should show gains where the linear probe showed losses,
concentrated in the entity and frequency families. If it does not — if a tree
also gains nothing from them — then the diagnosis here is wrong and the features
are genuinely weak, and that outcome gets recorded too.

### Result — the velocity family: the first that costs nothing

Four columns over `card1`: transactions in the trailing 1h, 24h and 7d, and the
log seconds since that card was last seen.

| | PR-AUC | ROC-AUC | recall @ 0.5% | recall @ 1% | recall @ 2% |
|---|---:|---:|---:|---:|---:|
| `family_none` | 0.32169 | 0.82585 | 10.05% | 16.23% | 23.92% |
| `family_velocity` | 0.32208 | 0.82542 | 10.05% | 16.13% | 24.32% |
| delta | **+0.00039** | −0.00043 | 0.00000 | −0.00101 | +0.00402 |

**Positive, and meaningless as a positive:** +0.00039 is 0.25σ against a floor σ
of 0.00156, and the bar is +0.005. Velocity did not help.

**It is a different null from the other three, and the difference is the point.**
This is the only family in the phase whose relationship with fraud is
*monotone* — log seconds since the card was last seen runs 1.49x, 1.00x, 0.90x,
0.97x, 0.71x by quintile on train. The three non-monotone families each cost the
probe between 0.001 and 0.005; the one monotone family costs nothing. A linear
model is not harmed by a shape it can represent.

**Half the diagnosis held; half did not, and that is recorded rather than
argued away.** "The instrument cannot see these shapes" explains the amount,
frequency and entity results. It does not explain velocity, where the shape was
representable and the metric still did not move. Two candidates, neither tested:
the recency signal may be partly redundant with `D3`, already in the probe at
Spearman 0.34 — Vesta's own day-deltas are undocumented aggregates and this is
the one that behaves like a recency measure; or a 2.1x spread across quintiles
is simply too little for a global ranking metric on 1,990 positives. Phase 05's
retrain is where that separates.

**Deliberately not in the family, both measured first.** Burst ratios — a short
window against a long one, the scale-free form of a count — turn over at the top
(0.89x, 1.04x, 1.10x, 1.14x, 0.83x) and carry less than recency alone. And a
first-sighting indicator: a card's first transaction has no predecessor, and its
0.63x lift sits on top of the slowest recency quintile's 0.71x, so it belongs at
the slow end of that scale rather than as a column of its own.

**Implementation note, because it nearly went wrong quietly.** The first version
used `groupby(...).rolling(window, on=...)`, whose result is indexed by
timestamp rather than by row. With 33,932 rows sharing a `TransactionDT`,
unwinding that misaligns counts across cards. Pandas happened to raise on this
data; on a slightly different frame it would have returned wrong numbers with no
error. The shipped version counts by binary search over each card's transaction
times, so alignment depends on a row's position and never on its timestamp being
unique — and a test reproduces the tie case that broke the first attempt.

**This is the phase's only tier-3 family.** A trailing count needs the card's
history at request time, which a single API call does not carry. E3 is where
that cost gets priced.
