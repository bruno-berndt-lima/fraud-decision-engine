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

**Status:** complete. Phase 05; the USD half owed by Phase 06.

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

### Method — Phase 05, revised to three arms

**Two arms answer with one number what the question above splits into two
advantages.** The unpurged run is handed *recency* — training up to the
validation boundary — and *volume*: 31% more rows and 36% more fraud. Measured
together, a large delta cannot say which one produced it, and the two have
completely different implications. Recency is something a production retrain
cadence can partly buy; labels that could not exist yet is something nothing
buys.

| arm | `TRAIN` | days | gap | what it is handed |
|---|---|---:|---|---|
| `purged` | 1–90 | 90 | 91–120 | nothing — the shipped split |
| `recent` | 31–120 | 90 | none | recency, at the shipped volume |
| `unpurged` | 1–120 | 120 | none | recency **and** volume |

`purged → recent` isolates recency. `recent → unpurged` isolates volume. The
`purged` and `unpurged` arms are a config value and nothing else — `gap_days` —
and `resolve_boundaries` derives the rest.

**`recent` cannot be a config value, and the attempt to make it one was refused
by the right guard.** `train_start: 31` leaves days 1–30 claimed by no split,
and `validate_splits` rejects a declared span narrower than the table because
that is how rows get discarded silently. The honest construction removes the
rows from the data rather than from the boundaries: the arm writes its own copy
of `interim` holding days 31 onward, and every stage reads that. The cut is a
file on disk, not an implication of a config.

**Nothing shipped is overwritten.** Each arm runs the split and feature stages
against a config whose outputs are redirected into a working directory —
`splits`, `split_summary`, `features_dir`, and the three fitted artifacts.
`interim` is shared by `purged` and `unpurged`, which read the same rows and
differ only in how they are labelled; `recent` reads its own cut, and that
difference is the arm. Editing the shipped config in place and restoring it
afterwards would put every downstream stage one interruption away from silently
reading the wrong split.

**Why the cut and not the cheaper alternative.** `recent` could reuse the
`unpurged` matrices and drop training rows before the booster. It would then fit
its encoders, aggregates and V-block reduction on 120 days and its booster on
90 — scoring a 90-day model through 120-day tables, which leaks volume into the
one arm that exists to hold volume fixed. The same objection the next paragraph
makes about sharing fits across arms.

**What the cut costs, registered before it runs.** The causal features start
cold. `velocity`'s seven-day window is not full until day 38, and
`vel_recency_card1` reads a card last seen before day 31 as never seen — for
infrequent cards, for longer than a week. The inherited columns are untouched,
being pre-computed upstream, and validation carries ninety days of lookback
inside the arm. So the artifact is confined to one family, on early training
rows, and E3 measured that family's whole removal inside its bar.

A degraded feature is not an absent one, though, and can mislead a tree more
than a missing column would. The bias most plausibly runs against `recent`,
which fixes how the arm is read: **if `recent` scores at or above `purged`,
recency survives the artifact; if it scores below, the artifact is a live
explanation and the recency reading is undecided.**

**"Identical evaluation slices" needs stating precisely.** The `VAL-FIT` *rows*
are the same in all three arms, and so is the label vector. **The feature values
are not.** Frequency encodings, entity aggregates and the V-block reduction are
fitted on `TRAIN`, and `TRAIN` is what moves — so validation arrives encoded
against a different fit in each arm. That is correct and unavoidable: it is part
of what the purge costs, not a confound to be removed. Removing it would mean
scoring an unpurged model through a purged model's encoders, which is neither
run.

**Instrument: the untuned reference**, as with every other Phase 05 comparison.
It samples neither rows nor columns, so no arm carries seed noise. The tuned
configuration is not used — its knobs were selected on this project's `VAL-FIT`
under the shipped split, and carrying them into an arm that trains on different
data would import a selection the arm did not make.

**This experiment has no bar, and that is registered rather than papered over.**
There is no natural control: the ablation floor measures removing *columns*, and
the seed spread is zero on this instrument. The early-stopping unfairness E6
recorded applies — more training data moves where the curve peaks — and nothing
here bounds it. So the reading rule is directional, not a threshold: E1 is a
measurement with a registered expected ordering and the sanity check above, and
the `recent` arm is the closest thing to a control, since it removes the purge
without adding volume.

**Reported on `VAL-FIT` only.** Three arms compete for an explanation and none
of them ships, so `VAL-CAL` is not scored — the rule `DEFAULT_SPLITS` states.
The USD half belongs to Phase 06 and is not owed by this run.

### Result — Phase 05

| arm | `TRAIN` | rows | frauds | `VAL-FIT` PR-AUC | vs `purged` | best iteration |
|---|---|---:|---:|---:|---:|---:|
| `purged` | 1–90 | 315,927 | 10,702 | 0.52820 | — | 561 |
| `recent` | 31–120 | 280,203 | 11,199 | **0.67345** | **+0.14524** | 1000 |
| `unpurged` | 1–120 | 414,542 | 14,600 | 0.65043 | +0.12223 | 966 |

The `purged` arm reproduces the shipped reference to every digit, so the rebuilt
pipeline is the shipped one and the three rows are comparable. The shipped
artifacts were fingerprinted before and after and did not change. `VAL-FIT` holds
the same 57,464 rows in every arm.

**The sanity check does not fire.** The unpurged run beats the purged one, as
the ordering registered above said it must; the harness is not under suspicion.

**Recency is essentially all of it.** `recent` clears `purged` by more than the
full unpurged run does, which under the rule registered for the cut means
recency survives the cold-start artifact. `recent` stopped on a round number
and was re-fitted to check: early stopping closed at 1100 with the peak at 1000,
against a ceiling of 5000. A coincidence, not a bound.

**Volume adds nothing, and the direction is against it.** Extending `recent`
back over days 1–30 moved PR-AUC from 0.67345 to 0.65043. The artifact biases
against `recent`, so the ordering survives it. This experiment has no bar, so
the size is a direction and not a finding — but the two arms stopped within a
few dozen rounds of each other, which leaves little room for the early-stopping
unfairness to explain it.

### Why a number this large is the expected one

**The purge costs twice what tuning bought.** That reads as too good, and the
check is whether it makes sense. It does, and the reason is the argument for the
purge. Days 91–120 sit immediately before `VAL-FIT`: cards, addresses and
devices committing fraud on day 115 are still committing it on day 125, and a
model trained on those labels learns those identities. `recent`'s training
window also carries a higher fraud rate than `purged`'s — the end of the window
looks more like validation than the start does.

In production those labels arrive weeks later, through chargebacks. So what the
unpurged arms are handed is almost entirely the one advantage the question above
said nothing buys, and the part a retrain cadence could buy — more data — turns
out to be worth nothing here.

### What it does and does not decide

**It does not change which split ships.** The purged split is the headline, as
registered. What changes is the size of the claim the purge supports: **an
evaluation without it would have overstated this model's `VAL-FIT` PR-AUC by
roughly a quarter**, and nearly all of that overstatement is labels that could
not have existed yet.

**It does not measure the gap's length.** Thirty days is argued from how
chargebacks arrive, and this experiment tests zero against thirty rather than
the curve between. A shorter gap that kept most of the honesty would need arms
at intermediate widths, registered before they run.

**It gives Phase 09 a prior.** E5 asks whether the train/validation gap is the
model or the data. That recency dominates volume this sharply says the
distribution near the boundary is moving fast, which is the shape E5 should
expect to find.

---

## E2 — Class weighting versus doing nothing

**Status:** complete. Logistic regression in Phase 03, LightGBM in Phase 05.

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

### The LightGBM arms, named before running — Phase 05

Five, on the untuned reference configuration:

| arm | what it does |
|---|---|
| `none` | the untuned reference, unchanged |
| `scale_pos_weight` | multiplies the positive class's gradient by `neg/pos` |
| `is_unbalance` | LightGBM derives the same ratio internally |
| `imputed` | medians in place of nulls, no resampling — the SMOTE arm's control |
| `smote` | medians, then synthesised positive rows interpolated between neighbours |

**`is_unbalance` is close to redundant** — it sets the same weight the arm above
sets by hand. It is run anyway because the roadmap names both, and because
"these are the same thing" is worth demonstrating once rather than asserting.

**No seed averaging is needed here.** The untuned configuration samples neither
rows nor columns, so every arm is deterministic and each is one number rather
than a distribution. E6's bar does not apply and is not borrowed: it measures
seed spread, and there is none to measure. That will not be true of the tuned
comparison, and the difference is why these two are separate experiments.

**Why `imputed` exists.** SMOTE is nearest-neighbour interpolation and cannot
compute a distance across a null, and a large minority of the numeric columns
here carry them — one is null in ninety-nine percent of training rows. So the
SMOTE arm has to impute, and the other arms do not: LightGBM takes nulls
natively, and Phase 04 established that missingness in this dataset is signal
rather than damage.

Without a control, the SMOTE arm would differ from the reference in two ways at
once and a loss would be unattributable. `imputed` is the same imputation
without the resampling, so the chain reads `none → imputed → smote` and each
step costs one fit. Medians are fitted on train and applied to validation too:
scoring a model on data shaped differently from what it trained on would trade
one confound for another.

**Three further objections to the SMOTE arm, recorded before it runs** rather
than discovered in its defence afterwards:

1. **Thirty-one categorical columns.** SMOTE interpolates in feature space, and
   there is no midpoint between `visa` and `amex`. `SMOTENC` takes the majority
   category among neighbours instead, which for a seventy-level `DeviceInfo`
   still pairs a fabricated device string with interpolated numerics.
2. **The split is temporal.** A synthetic row has synthetic history: its velocity
   counts and entity aggregates are interpolations of events that did not
   happen in that order, or at all.
3. **The features are mostly Vesta's.** The V block dominates the signal, and
   what an interpolated `vb_V258` means is unanswerable, because what `V258`
   means is unanswerable.

None of these is a reason to skip the arm. They are the reason the result is
worth having: this project's stated position is that SMOTE is not applied
because a problem is imbalanced, and the way to hold that position honestly is
to run it and report what happened. If it loses, these three objections are the
explanation rather than an excuse invented afterwards.

**Reported through the same harness as everything else.** `scale_pos_weight`
changes the training objective's gradients, not the metric — and PR-AUC here is
computed by `evaluation/metrics.py` on raw scores regardless, so no arm can be
flattered by being measured on its own terms.

### Result — LightGBM, Phase 05

| arm | `VAL-FIT` PR-AUC | `VAL-CAL` PR-AUC | `VAL-CAL` recall @ 1% | best iteration |
|---|---:|---:|---:|---:|
| `none` | 0.51558 | 0.46010 | 24.83% | 133 |
| `scale_pos_weight` | 0.49569 | 0.45359 | 25.06% | 312 |
| `is_unbalance` | 0.49569 | 0.45359 | 25.06% | 312 |
| `imputed` | **0.52820** | **0.47850** | **25.39%** | 561 |
| `smote` | 0.52570 | 0.45244 | 24.83% | 350 |

**`is_unbalance` and `scale_pos_weight` are the same thing**, to every digit, as
expected. Demonstrated once so it never has to be argued again.

**Class weighting loses, and the registered expectation held in magnitude.** E2
predicted a tree ensemble would be far less sensitive to reweighting than a
linear model. It is: logistic regression lost 0.0335 of `VAL-CAL` PR-AUC,
LightGBM loses 0.0065 — five times less. The direction is still negative, so
"no meaningful difference" was the wrong word for it, but the reasoning behind
the prediction was sound. Recall at capacity is a wash, marginally better in
some cells and worse in others; PR-AUC decides it.

**SMOTE loses on both slices.** Below its own control, `imputed`, on `VAL-FIT`
and on `VAL-CAL` alike, and below the untouched reference on `VAL-CAL`.

**This paragraph said the opposite until the arm was re-measured**, and the
correction is recorded rather than overwritten. As first measured, SMOTE was
first on `VAL-FIT` by a wide margin — 0.54507, stopping at round 897 — and
second on `VAL-CAL`, and the text read that as the long-run, lucky-peak shape
E6 registered in advance. That run was made before `to_dataset` turned
`feature_pre_filter` off. Re-fitting the identical resampled data with the flag
in each state reproduces both numbers exactly: on, 0.54507 at round 897; off,
0.52570 at round 350. The resampling is deterministic and the difference is the
flag alone. Why the flag moves this arm and not the other four was not
established.

So the win on the spent slice was a property of how the dataset was built, not
of the method, and the prediction it was offered as confirming has no
confirmation here. **The verdict does not move** — SMOTE earns no place — and it
now rests on the arm losing everywhere rather than on a gap between slices.

**The control won.** `imputed` was added only to make the SMOTE arm
interpretable, and it is the best arm on both slices on both metric families. That was not an expected outcome and it sits awkwardly beside Phase
04's finding that missingness in this dataset is signal.

An untested mechanism, offered as a hypothesis and not a conclusion: a column
that is null in ninety-nine percent of training rows gives LightGBM a free split
over almost nothing, and it decides which way to send the missing on the basis
of very few present rows. Filling those columns makes them near-constant and
effectively inert, which is regularisation arriving by accident. Nothing here
tests that.

### What this result does and does not decide

**It decides E2.** Neither class weighting nor synthetic minority oversampling
earns a place in the shipped model. The project's stated position — that SMOTE
is not applied because a problem is imbalanced — survives contact with the
measurement, and it survives it having been genuinely tested rather than
assumed.

**It does not decide whether the shipped model imputes.** That is a modelling
choice, it belongs to the tuning step, and it has to be made on `VAL-FIT` under
E6's rule. The `VAL-CAL` column above is reported, as every run in this project
reports it, but it must not select — reading it to rank arms would spend the
slice Phase 06 calibrates on.

This distinction is easy to lose precisely because the `VAL-CAL` numbers are the
more informative ones here. Naming it is the safeguard: E2 is a reporting
experiment. It says what each arm did. It does not choose.

---

## E3 — Servable features versus entity history

**Status:** the metric half answered in Phase 05; the USD half owed by Phase 06.

Committed in `problem-statement.md` §3.4. A scoring request carries the
transaction and its immediate attributes, not the card's history, so velocity and
entity-aggregate features require an online store that does not exist here.

Both feature sets get built and evaluated. The model behind the API uses only
request-computable features; the history-dependent model runs alongside it, and
the difference in PR-AUC and USD saved is reported as **the measured cost of not
building a feature store** — a more useful result than either silently training
on unservable features or quietly dropping them.

### Method — Phase 05

The arm is `tier_3`: the four `vel_*` columns, removed from the full matrix.
Everything else stays. Tier 3 is the only tier that needs a store written on
every transaction — a keyed rolling window whose stale write produces a silently
wrong count rather than an error — so the difference that arm makes *is* the
price of the store, with nothing else moving alongside it.

Tier 2 is not part of this question and stays in. A fitted table shipped beside
the model is a file in the deployment, not a store with its own availability;
`features.md` separates the two axes for exactly this reason, and folding them
together would report the cost of a lookup file as if it were the cost of Redis.

**Read against the bar at width 4**, drawn the same way and on the same
instrument as every other arm — see E4. The bar and the arms are one run.

**Half of this is already measured.** The `velocity` family in E4's tree
ablation *is* the tier-3 arm: same four columns, same removal, same reference.
Its delta did not clear the bar at its width. What E3 adds is naming that as an
answer to the serving question rather than to the feature question, and carrying
it into Phase 06 as USD.

**The expected finding, registered before the USD half exists.** If the arm does
not clear its bar, the reported result is *the feature store cannot be shown to
pay for itself*, not *velocity is worthless*. The blind spots E4 registered
apply unchanged: a redundant matrix reads a recoverable family as zero, and the
detecting instrument is not the shipped one.

### Result — Phase 05, the metric half

| | features | `VAL-FIT` PR-AUC | delta | bar at width 4 | clears |
|---|---:|---:|---:|---:|---|
| `full` | 349 | 0.52820 | — | — | — |
| tier 3 removed | 345 | 0.53074 | +0.00253 | 0.01598 | no |

**The store cannot be shown to pay for itself.** The arm sits well inside the
bar its own width produces, so the reading registered above is the one that
applies — this is not a measurement of velocity being worthless, and E4's blind
spots are not softened by the arm having been renamed for a different question.

**The sign is positive and should not be read.** Removing the four columns moved
the metric *up*, which at a fifth of the bar means the draws at that width move
further in both directions than the arm did. A delta inside the bar has no
direction to report.

**The contrast with E7 is the useful part.** Measured on the same reference, at
the same time, by the same rule: the tier that is expensive to serve and
possible to build costs nothing detectable, and the tier that is free to serve
and impossible to build costs nearly half the metric. The two axes `features.md`
separated turn out to point in opposite directions, and a project that had
merged them into one *needs-cache* label would have been unable to say so.

**What Phase 06 still owes this.** The USD half. A delta inside the bar on
PR-AUC does not by itself say the store is not worth building — the decision
runs at a capacity, against a cost matrix, and recall at the reviewed band is
where a small ranking change can still move money. The registered position is
that the metric half is settled and the economic half is not.

**A cheaper fallback stays on the table, and stays unmeasured.**
`features.md` notes that `vel_recency_card1` alone needs only a last-seen
timestamp rather than three rolling windows — a far smaller store, close to
tier 2 in cost. Nothing here tests that arm. Its width is 3, its bar already
exists, and running it would be a new arm rather than a re-reading of this one.

---

## E4 — What each feature family measurably adds

**Status:** complete under the linear probe in Phase 04, and re-measured
under a tree in Phase 05. Both bars registered before their results.

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

#### The Phase 05 ablation, and its blind spots — registered before running

Two things invert. **The direction:** an arm is the full matrix *minus* one
family, because what a serving decision needs to know is what is lost by not
building it, and because E3 needs that shape for the history-dependent columns.
**The instrument:** a tree can represent thresholds and interactions the probe
could not, which is the whole reason this was handed over.

The consequence, stated so no one draws the table wrong: **Phase 04's deltas and
Phase 05's are not two columns of one comparison.** Different base, opposite
sign. A family that reads +0.004 there and −0.004 here is not contradicting
itself.

**Detection runs on the untuned configuration.** It subsamples neither rows nor
columns, so repeated fits return identical digits and a delta carries no seed
noise at all. The tuned configuration is stronger and does subsample; the spread
`seeds.py` measured there is wide enough to hide most of these families by
itself. Families that move go back to it for confirmation, the shape E6
established for the tuning candidate.

Three blind spots, and none of them is expected to be corrected here:

1. **Leave-one-out against a correlated matrix measures redundancy, not
   signal.** A family whose columns are recoverable from the survivors reads as
   zero. Given where Phase 04 found the signal to live, this is the likeliest
   way a real family comes back flat, and *the tree found it elsewhere* is a
   different claim from *there was nothing there*.

2. **The detecting instrument is not the shipped one.** The untuned
   configuration has a hundred-odd trees and no subsampling. A family that only
   earns its place in a large, heavily regularised model will not appear.

3. **The arms will not stop at the same round**, so the same early-stopping
   unfairness E6 recorded applies within this comparison — removing a family
   changes the curve, and an arm that trains longer has had more chances at a
   lucky `VAL-FIT` peak. `VAL-CAL` is not the way out of it: this is a field of
   candidates, so it scores `VAL-FIT` only, and a family that lands inside that
   unfairness is reported as undecided rather than broken open on the slice
   Phase 06 calibrates against.

**Every family still ships**, exactly as Phase 04 left it. This experiment
characterises; it does not accept or reject.

#### The bar the tree ablation is read against — registered before it is measured

**The order was wrong and saying so is part of the record.** Phase 04 measured
its noise floor before any family was scored. Here the six arms were run first,
and the need for a bar became obvious from looking at them: the deltas span a
range no existing instrument can rule on, and the `best_iteration` column ranges
from under a hundred and fifty to nearly six hundred, which is blind spot 3
arriving in the results rather than in theory.

What is registered here is the bar's **design**, fixed in writing before it is
run. The distinction that makes this worth anything: every choice below is
determined by the families, not by what they scored. Choosing the widths, the
draw count, or the statistic after seeing which families need to clear them
would be picking a ruler by looking at what it has to measure, and that is the
thing this note exists to rule out.

**Neither of the two bars this project already owns applies.** The Phase 04
noise floor was measured under the linear probe, on a different instrument and
in the opposite direction. The seed spread says the untuned configuration has
none — true, and it removes seed noise rather than the confound that is actually
present here.

| | |
|---|---|
| **Widths** | 3, 4, 6, 7, 234 — the five family sizes, and nothing else |
| **Draws** | 10 per width, columns sampled uniformly from the 349 features without regard to family |
| **Statistic** | per width: the standard deviation, and the largest absolute delta over the ten draws |
| **The bar** | a family's delta is read as movement only if it exceeds the largest absolute delta at its own width |

**Why this bar sees something the Phase 04 floor could not.** Each draw stops at
its own round, so the spread it measures already contains the early-stopping
unfairness rather than holding it constant. A family whose delta sits inside the
bar is undecided for either reason, and this experiment does not need to
separate them to report honestly.

**The largest width is a degenerate control, and it is kept anyway.** 234 of the
349 features are `vb_*`, so a random draw of 234 columns is about two thirds
V-block by construction — the control at that width is partly made of the thing
it controls for. It still answers the question it is being asked, *what does
removing this many arbitrary columns cost*, and the V-block's delta is still
read against it. It is not a clean control and no reading of it should claim it
is.

**Fifty fits, on the untuned reference**, the same instrument the arms were
measured on. A bar drawn on a different configuration would not be a bar.

#### Result — nothing clears, and that is the finding

| arm | width | delta | \|delta\| | bar | draw σ | clears |
|---|---:|---:|---:|---:|---:|---|
| `amount` | 3 | −0.00659 | 0.00659 | 0.00673 | 0.00282 | no |
| `velocity` | 4 | +0.00253 | 0.00253 | 0.01598 | 0.00742 | no |
| `entity` | 6 | −0.00365 | 0.00365 | 0.01718 | 0.00760 | no |
| `frequency` | 7 | +0.01141 | 0.01141 | 0.01323 | 0.00720 | no |
| `vblock` | 234 | −0.00371 | 0.00371 | 0.06738 | 0.01423 | no |

Reference arm `full`: 0.52820 `VAL-FIT` PR-AUC on 349 features, identical to
E2's `imputed` arm to every digit — same instrument reached by two paths.

**No family moves the metric further than removing that many arbitrary columns
does.** `amount` comes closest and misses by 0.00014, which is the kind of
margin the bar exists to refuse; a ninth draw instead of ten would likely have
let it through.

#### Why the bar is so wide, and it is not seed noise

The untuned configuration samples neither rows nor columns, so repeated fits
return identical digits. What moves instead is where training stops:

| width | `best_iteration` across ten draws |
|---:|---|
| 3 | 177 – 804 |
| 4 | 161 – 784 |
| 6 | 142 – 491 |
| 7 | 170 – 1213 |
| 234 | 215 – 622 |

**Removing seven arbitrary columns moves the stopping round by a factor of
seven.** Blind spot 3 was registered as a caveat and arrives as the dominant
term: a draw's delta is mostly a statement about which peak early stopping
happened to find, and only secondarily about the columns that left.

That is why the tree bar lands near σ 0.007 at the small widths against the
linear probe's σ 0.00156 — roughly five times wider, on an instrument that has
no seed noise at all. A bar that held the stopping round fixed would have been
narrower and would have been measuring the wrong thing.

#### The V-block reads differently from how it scores

| | delta |
|---|---:|
| removing the 234 `vb_*` columns | −0.00371 |
| removing 234 arbitrary columns, mean of ten | −0.03771 |
| the worst of those ten | −0.06738 |

Formally the V-block does not clear its bar. The reading is not *the V-block
does not matter* — it is **the V-block is the most redundant thing in the
matrix**. Every random draw at that width hurts; the actual family is the one
cut that costs almost nothing. What it carries, `C*` and `D*` reconstruct.

The caveat registered before the run still applies and is not softened here:
two thirds of every draw at that width is `vb_*` by construction, so the
contrast is between the whole family and a random two thirds of it plus
whatever else came along. It is suggestive, not clean.

**E7 corrects the second half of that reading.** Right about redundancy, wrong
about where it lives: the 61 inherited columns left standing — `C*`, `D*`,
`id_*` — are what covered for the V-block, and removing the inherited tier
entire costs an order of magnitude more than removing this family did. The
V-block is not unusually redundant among the matrix's columns; it is part of a
block that is redundant with itself, and this arm never removed the block.

#### What this settles

**A null result, and it is the result.** Under a tree, measured against an
instrument that absorbs the early-stopping instability rather than assuming it
away, no hand-built family has a detectable contribution — in either direction.

This does not contradict Phase 04; it explains it. The probe found signal in
`vb_*` because a linear model needs those columns to express what the tree
recovers from `C*` and `D*` on its own. Blind spot 1 predicted a redundant
matrix would read as zero under leave-one-out, and it now has a number.

**Every family still ships.** E4 is a reporting protocol. Nothing here accepts
or rejects a family, and the shipped model's feature set is unchanged.

**What it does change is the claim.** Phase 04 could say the V-block reduction
moved the probe further than anything else built by hand. Phase 05 has to say
that under the model that ships, no hand-built family — the reduction included —
is distinguishable from removing that many columns at random.

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

### Result — the V block: the only family that clears the bar, by twenty-two times

339 Vesta columns reduced to 228 representatives and 6 presence flags.

| | PR-AUC | ROC-AUC | recall @ 0.5% | recall @ 1% | recall @ 2% |
|---|---:|---:|---:|---:|---:|
| `family_none` | 0.32169 | 0.82585 | 10.05% | 16.23% | 23.92% |
| `family_vblock` | 0.43360 | 0.84478 | 11.66% | 21.66% | 33.92% |
| delta | **+0.11192** | +0.01893 | +0.01608 | +0.05427 | +0.10000 |

The bar is +0.005. This clears it by a factor of twenty-two, and sits roughly
seventy standard deviations above the noise floor. At the committed 1% review
capacity it is **16.2% of fraud caught rising to 21.7%**; at 2%, 23.9% to 33.9%.

**Predicted, and the only one that was.** The blind-spot table registered before
any family was built marked the V block as the single family a linear probe
could see — "a dense, largely linear block — yes". Four families were marked no
or barely, and four families returned nothing or worse. The one marked yes
returned this.

### The caveat this number cannot be reported without

Phase 01 recorded a trap about `C1`-`C14` and `D1`-`D15`: they are already
aggregates, computed by Vesta over lookback windows it never published, across
the whole dataset, before we saw a row. The purge protects *labels*; it does
nothing about features built from information beyond the training window.

**That trap applies identically to all 339 V columns, and after this result it
stops being a footnote.** Essentially all of the measured signal in this phase
lives in features whose construction cannot be audited. Any headline number this
project reports rests substantially on them.

What is and is not clean, precisely:

- **This project's pipeline is clean, and tested.** The NaN grouping is
  structural, the correlation clustering and the medians are fitted on training
  rows alone, and the probe's own fit sees only train.
  `test_validation_rows_choose_neither_the_columns_nor_the_fill` is the specific
  guard, and it exists because correlation deciding which columns to drop is a
  subtler leak than a fitted median — nothing about the output would look wrong.
- **The exposure is inherited and unfixable.** The windows are not documented.
  It cannot be measured, corrected, or bounded from inside this repository.

The honest treatment is to state it in the README as the project's principal
limitation rather than to discount the number, and to be precise that it is a
property of the dataset rather than of the pipeline.

### What the reduction found

**The block is far less redundant than its reputation.** At an absolute
correlation of 0.90 within NaN group, two thirds of the columns survive as
distinct. The folklore that the V block is mostly copies of a handful of
signals is not what the training window says.

**Missingness is structural and carries its own signal.** The 339 columns fall
into a small number of groups sharing an identical null pattern, almost perfectly
contiguous in the V numbering — Vesta built the block in batches. Most of those
patterns turn out to be `has_identity` under another name and were deduplicated
away; one is the opposite, present on a minority of rows, negatively correlated
with `has_identity`, and protective.

**The raw block is replaced, not supplemented.** A matrix carrying both `V95` and
`vb_V95` has been duplicated rather than reduced, and the next stage to reach for
"every numeric column" would get a perfectly correlated pair for each one. The
originals remain in `interim/transactions.parquet`, and the original name is
recoverable from the prefix, so Phase 07 can still trace a contribution to the
Vesta column it came from.

### What Phase 04 actually concluded

Four families of hand-built features — amount structure, frequency encoding,
entity deviation, velocity — moved the metric by less than chance, three of them
downward. One block of somebody else's pre-computed feature engineering moved it
by twenty-two times the detection threshold.

That is the phase's result, and it is worth more than the opposite outcome would
have been. It says where the signal in this dataset actually lives, it says what
a linear model can and cannot use, and it says that the most important thing to
write down about this project is a limitation rather than a win.

---

## E5 — Is the train/validation gap the model, or the data it was given?

**Status:** registered, not yet run. Phase 09.

**Question.** Identity coverage falls sharply across the split boundary. How much
of the gap between training and validation performance is the model failing to
generalise, and how much is validation simply carrying less information per row?

**The observation that prompted it**, measured in Phase 05 while fitting the
LightGBM category vocabulary:

| split | `has_identity` | `id_31` null |
|---|---|---|
| train (days 1–90) | 29.02% | 71.69% |
| val_fit (121–140) | 17.87% | 82.67% |
| val_cal (141–160) | 17.29% | 83.17% |

A 38% relative fall in the share of rows carrying an identity block. This reaches
further than the 38 `id_*` columns: Phase 04 found that most of the V-block
presence flags are `has_identity` under another name, and the V block is where
essentially all measured signal lives.

**Why it is worth a run.** Without it, the default reading of any train/validation
gap is overfitting, and the fix that reading suggests is regularisation. If a
material share of the gap is feature availability, that fix is aimed at the wrong
thing — and the same confusion propagates into Phase 09's decay chart, where a
decline could be fraud behaviour changing or data collection changing. Those have
opposite responses: one calls for retraining, the other for fixing an upstream
pipeline. Nothing else in the project distinguishes them.

**Method.** Stratify, do not re-fit. The shipped model scores the validation
splits as it already does; the metrics are then reported separately for rows with
and without an identity block, on both sides of the split boundary.

| | train rows | validation rows |
|---|---|---|
| `has_identity` true | PR-AUC, recall@capacity | PR-AUC, recall@capacity |
| `has_identity` false | PR-AUC, recall@capacity | PR-AUC, recall@capacity |

If performance within each stratum is close across the boundary while the pooled
figures differ, the gap is composition rather than generalisation, and the size
of the compositional part is what this reports.

**Two constraints, both load-bearing:**

1. **`VAL-FIT` and `VAL-CAL` only.** The coverage numbers above stop at day 160
   deliberately. Test's own coverage was read once, incidentally, during Phase 05
   and is excluded from every figure and argument here — the honest handling of an
   unnecessary look is to refuse to use it, not to pretend it did not happen.
2. **No re-fit, no re-tune.** This is a re-reading of scores the shipped model
   already produced. A model fitted per stratum would answer a different question
   and would spend a training decision on something measurement can settle.

**What gets reported.** The four cells above, and the share of the pooled gap
attributable to composition. Both directions are publishable: "the gap is almost
entirely generalisation" is as useful as the alternative, because it retires an
explanation that would otherwise stay plausible forever.

**What the result does not do.** It changes no threshold, no hyperparameter and
no feature. The shipped model is already chosen by then. It changes how the
Phase 09 decay chart is *read*, and it is a caveat on the Phase 05 headline
rather than a correction to it.

---

## E6 — How large does a difference have to be before it is a difference?

**Status:** complete. Spread measured, candidate accepted, and the bias the
bar does not cover given a size of its own. Phase 05.

**Question.** Two LightGBM configurations score differently on `VAL-FIT`. How much
of that gap can be produced by nothing at all?

**Why it needs registering in advance.** This is E4's problem one level up. There
the trap was adding features until the metric moved; here it is accepting
hyperparameters until the metric moves. The defence is the same, and it only
works if the bar exists before the search does — a tuning run picks the best of
many trials, and the maximum of noise is biased upward whether or not anyone
intends it.

**What prompted it.** Three seeds, identical in every other respect:

| configuration | `VAL-FIT` average precision | best iteration |
|---|---|---|
| defaults, seed 0 | 0.51557624 | 133 |
| defaults, seed 1 | 0.51557624 | 133 |
| defaults, seed 2 | 0.51557624 | 133 |
| `feature_fraction` 0.8, `bagging` 0.8, seed 0 | 0.53496591 | 400 |
| same, seed 1 | 0.51963314 | 346 |
| same, seed 2 | 0.52497724 | 463 |

**The untuned reference has no seed sensitivity at all**, because LightGBM's
defaults sample neither rows nor columns. Every digit is identical. That is worth
recording on its own: it means the reference is exactly reproducible, and it
means a seed-spread measured *there* would be zero and would describe nothing
about the search.

Turn subsampling on and the same configuration spans 0.0153 across three seeds,
with the best iteration moving by over a hundred trees. The seed changes the
subsample, which changes the validation curve, which changes where early stopping
lands, which changes the model. The variance compounds.

The gap between the best of those three seeds and the untuned reference is
+0.019. The gap between the worst and the reference is +0.004. Reporting the
first as what subsampling bought would be a claim resting entirely on a lucky
draw.

### Method

**Characterising the spread.** `k` seeds at each of a small number of points in
the search space, plus the untuned reference. Only the seed varies within a
point. Reported as mean and standard deviation of `VAL-FIT` PR-AUC, and as the
spread in best iteration, since that is where the mechanism lives.

**The acceptance rule, fixed here.** During the search every trial uses one fixed
seed, so trials are comparable to each other. The winner is not accepted on that
number. Instead the winning configuration **and the untuned reference** are each
re-run under `k` seeds, and the winner is accepted only if

```
mean(candidate) − mean(reference)  >  σ(candidate) + σ(reference)
```

**This is deliberately stricter than a significance test.** The standard error of
a difference of means shrinks with `k`; the sum of the standard deviations does
not. The stricter bar is the price of having selected the candidate as the
maximum over many trials, which no single-comparison test accounts for.

**Two constraints:**

1. **`VAL-FIT` only.** Tuning already spends that slice; `VAL-CAL` is Phase 06's
   and is not read here.
2. **Distributions, not points.** A comparison of two single runs is not
   evidence, and this entry exists to say so before any single run is available
   to be flattering.

### What this bar does not cover

**Seed noise and early-stopping optimism are different things, and only the
first is measured here.** Early stopping picks the best round from the
`VAL-FIT` curve, so a configuration that trains for many rounds has more
opportunities to land on a lucky peak than one that stops early. The measured
points differ by an order of magnitude in exactly that respect — the untuned
reference chooses among a hundred-odd rounds, the aggressive point among up to
two and a half thousand — so comparing them on `VAL-FIT` is not entirely fair to
the shorter one.

The effect is smaller than the round counts suggest, because a boosting curve is
smooth rather than a sequence of independent draws, and the number of effectively
independent chances is far below the number of rounds. But the direction is real
and it favours the longer run. It is recorded here rather than corrected: the
correction would need a slice neither this experiment nor Phase 05 is allowed to
spend, and `VAL-CAL` in Phase 06 is where the shipped model meets data that
neither tuning nor early stopping has touched.

#### How large it turned out to be

Both configurations were elected before either was scored on `VAL-CAL` — the
study, the confirmation and the adoption all ran on `VAL-FIT`, and `tune.py` and
`seeds.py` never load the calibration slice at all. Measuring two already-chosen
models there is reporting, not selecting, and it is what `make train` writes for
each of them anyway.

| | `VAL-FIT` PR-AUC | `VAL-CAL` PR-AUC | trees |
|---|---:|---:|---:|
| untuned reference | 0.51558 | 0.46010 | 133 |
| tuned candidate | 0.58878 | 0.52145 | 2990 |
| **what tuning bought** | **+0.07320** | **+0.06134** | 22.5× the rounds |

**Tuning keeps 83.8% of its `VAL-FIT` gain on the untouched slice.** The
remaining 16.2% is the best estimate this project has of what early-stopping
optimism is worth here, and it arrives as the difference between a fair slice
and a spent one rather than as a correction anyone applied.

Two readings, and the second is the load-bearing one. The gap is real: a 22.5×
difference in rounds costs about a sixth of the measured gain, which is far less
than the round counts would suggest and confirms that a boosting curve is not
twenty-two hundred independent draws. And **the acceptance survives it** — E6
accepted the candidate by a factor of eight over the bar, and the shrunk gain
still clears it, so the number changes the size of the claim without changing
which model ships.

**It must change nothing in Phase 05, and it has not.** The candidate was
adopted before this table existed. Recording the size of a known bias is the
one thing an already-spent decision can still be given.

**A run that exhausts its round budget is not evidence at all.** If
`num_boost_round` binds before the patience window closes, the reported best
iteration is where the budget ended, not where the metric peaked — and LightGBM
reports it identically either way. `fit` raises on that condition rather than
letting such a run into a comparison.

**What gets reported.** The measured spread, and for the final candidate both
means, both standard deviations, and whether the rule accepted it.

**Both outcomes ship.** If the tuned model does not clear the bar, the untuned
reference is what Phase 06 calibrates and Phase 08 serves, and Phase 05 reports
that tuning bought nothing measurable. That is a publishable result, and a more
interesting one than a tuned model that beat its own noise by half a standard
deviation.

**What the result does not do.** It does not choose hyperparameters — the search
does. It decides whether the search's answer is distinguishable from its
starting point.

### Result — Phase 05

Sixty trials, then ten seeds each for the winner and the untuned reference.

| | mean | spread |
|---|---:|---:|
| candidate | 0.58478 | 0.00832 |
| untuned reference | 0.51558 | 0.00000 |

Gap 0.06920 against a bar of 0.00832. **Accepted**, by a factor of eight.

**The reference's spread is exactly zero**, over ten seeds. Re-running it was
the part of the method that looked wasteful, and it is the part that turned an
expectation into a check.

**The spread at the subsampling points was measured twice, and the first
measurement described a pipeline that no longer exists.** It was taken before
`feature_pre_filter` was turned off, and that flag changes which columns a
subsampled tree draws from; the check that it changed nothing had been run only
on the untuned reference, the one configuration with nothing to draw. Re-measured
under the current pipeline, every subsampled and aggressive seed returns a
different number and the spread barely moves:

| point | spread, first measured | spread, current pipeline |
|---|---:|---:|
| subsampled | 0.00827 | 0.00831 |
| aggressive | 0.00915 | 0.00813 |

The verdict above is unaffected. It was computed from the confirmation runs, which
were made after the flag changed and reproduce to every digit.

**The selection inflation was far smaller than predicted.** The search reported
0.58878; re-measured across seeds the same configuration averages 0.58478. So
0.00400 of the search's number was the maximum of a noisy sample — against an
estimate, made before the run, of roughly 0.019.

That estimate was wrong in an instructive way. It modelled sixty trials as sixty
independent draws from one distribution — a lottery — when a search spends most
of its trials somewhere genuinely better than where it started. The inflation
applies only to the noise riding on top of a real difference, and here the real
difference dominated. **The estimate was an upper bound on a worst case that did
not arrive**, and it is recorded as such rather than quietly dropped.

### The range bound the answer

Two of seven knobs finished against their limits, both pointing the same way:

| knob | range | winner | |
|---|---|---:|---|
| `min_child_samples` | 20–500 | 20 | at the floor |
| `num_leaves` | 15–255 | 251 | four from the ceiling |

The top ten trials carry between 172 and 254 leaves. The search wanted a larger,
less constrained model than the space allowed, and **what it found is therefore a
corner of the space rather than an interior optimum**. Whether four hundred
leaves would do better is unknown and stays unknown.

**The range is not widened.** Choosing a search space after seeing where the
search pressed is a second layer of selection, and E6's bar does not cover it —
it judges a candidate against a reference, not one space against another. A
wider study run now would produce a number whose provenance nobody could state.

If the range is ever revisited, the honest form is to declare the new bounds
before running and report both studies. That is not done here: the accepted gain
is large and unambiguous, and a limitation stated plainly is worth more to this
project than a marginally better number with a worse story behind it.

---

## E7 — What could be rebuilt from scratch

**Status:** complete. Phase 05.

**The question an interviewer asks.** This model scores well on a dataset whose
signal lives in columns nobody outside Vesta can reproduce. So: *how much of it
survives if you only keep what you could build yourself?*

`features.md` already partitions the matrix to answer that. Tiers 1, 2 and 3 are
all constructible from a raw transaction feed — the request's own fields, tables
fitted at train time from them, and a keyed store this project could stand up.
Tier 0 is 295 columns of pre-computed aggregates over windows never published,
and `CLAUDE.md` names the exposure the project's principal limitation. **It has
never been given a number.**

**The buildable set is the complement of tier 0, and tier 3 is inside it.** Tier
3 is the expensive tier, not an impossible one; `features.md` grades it *us,
expensively*. Removing it here would answer the serving question a second time
under the reproducibility question's name — which is the exact conflation the
next paragraph exists to prevent, so it is worth stating twice.

**Distinct from E3, and the difference is the axis.** E3 asks what is expensive
to *serve*; this asks what is impossible to *rebuild*. A tier-3 column is
buildable and costly; a tier-0 column is cheap to serve — it arrives in the
CSV — and cannot be derived at all. Two different reasons a feature might not
be available, and conflating them would let a cheap answer stand in for a hard
one.

**The arms, and the bar, registered before running:**

| arm | removed | features left | what it is |
|---|---:|---:|---|
| `full` | 0 | 349 | the reference, the shipped feature set |
| `reproducible` | 295 | 54 | tiers 1, 2 and 3 — everything this project could build |

Read against the bar at width 295, drawn by the rule E4 registers: ten random
draws of that width on the untuned reference, and the arm is movement only if
its absolute delta exceeds the largest absolute delta of those draws. Nothing
about the design is chosen after the fact — the width is what the partition
produces, and `resolve_tiers` asserts the partition against what `features.md`
publishes rather than trusting it.

**The control at that width is degenerate, worse than the V-block's was.** All
295 removed columns are tier 0, which is 84% of the matrix, so a random draw of
295 columns is overwhelmingly made of the same thing the arm removes.
The bar there is close to a restatement of the arm rather than a control for it,
and no reading may claim otherwise. It is measured anyway, because the
alternative is reporting a delta with no scale at all.

**What the number is for, and what it is not.** It is a statement about this
dataset and this project's reach, reported beside the headline rather than
subtracted from it. It does not change the shipped feature set: the model ships
on everything the matrix carries, because a portfolio project that discards
signal to look reproducible has optimised for the wrong reader.

**Registered expectation.** The reproducible arm should lose, and lose clearly —
Phase 04 measured essentially all linear signal inside tier 0, and the tree
ablation showed the tier's members reconstruct each other. **If it does not
lose, suspect the partition before believing the result**: a tier-0 column
misfiled as tier 1 would leave the arm holding the very thing it claims to have
removed, and `resolve_tiers` asserts sizes precisely because that failure looks
like a good number rather than an error.

### Result — the limitation, with a number

| arm | features | `VAL-FIT` PR-AUC | delta | bar at width 295 | clears |
|---|---:|---:|---:|---:|---|
| `full` | 349 | 0.52820 | — | — | — |
| `reproducible` | 54 | 0.28433 | **−0.24387** | 0.13508 | **yes**, by 1.8× |

**Keeping only what this project could build costs 46% of the metric.** The
reproducible model retains 53.8% of the reference's PR-AUC. It is the only arm
in either experiment that clears its bar, after six that moved nothing.

The registered expectation held in direction and in size, so the sanity check it
carried does not fire: the arm lost, clearly, and the partition is not under
suspicion.

For scale, on the same slice: the rules engine scores 0.12807 and the Phase 03
logistic 0.32169. **A tree trained on only the reproducible columns beats the
incumbent comfortably and lands below the linear baseline** — which is not a
contradiction, because that baseline reads `D*`, and `D*` is inherited.

### Removing the inherited tier costs more than removing that many columns at random

| what is removed | inherited columns left | delta |
|---|---:|---:|
| the `vblock` family (234) | 61 — `C*`, `D*`, `id_*` | −0.00371 |
| 295 arbitrary columns, mean of ten draws | ~46 | −0.10620 |
| tier 0 entire (295) | 0 | −0.24387 |

**The arm costs 2.3× what its own control does**, and the three rows are one
finding rather than three. A random draw of 295 from 349 leaves about 46
inherited columns standing, and those 46 recover most of what the tier carries.
The signal lives in tier 0 as a **redundant mass**: any slice of it reconstructs
much of the whole, and removing all of it has nothing left to reconstruct from.

**This corrects how E4's V-block result reads.** Removing the 234 `vb_*` columns
looked nearly free, and the reading offered there — that the V-block is the most
redundant thing in the matrix — was right about redundancy and wrong about where
it lives. `C*` and `D*` stayed behind and covered for it. They are not a
different kind of column; they are the part of the same block that was not
removed.

**The degeneracy registered in advance does not explain this away, and points
the other direction.** All 295 removed columns are tier 0, so a random draw at
that width is mostly made of the same thing the arm removes — which should pull
the control *toward* the arm. The arm still lands 2.3× further out. A
contamination that biases the bar upward makes clearing it harder, so the
finding survives its own worst caveat.

### What it does and does not decide

**It does not change the shipped feature set.** The model ships on everything
the matrix carries. A portfolio project that discarded signal to look
reproducible would have optimised for the wrong reader, and this number is
reported beside the headline rather than subtracted from it.

**It gives the README a figure where it had a paragraph.** The limitation was
stated honestly and could not be sized; it now can, in the only terms that
matter here — what the model would score if the unreproducible columns had never
arrived.

**One thing this does not measure, and it is the obvious next question.** The
table above suggests `C*` and `D*` — 61 columns — carry most of what tier 0
holds, with the V-block the redundant remainder. Nothing here tests that: it is
a different arm, at a different width, and its bar would have to be drawn before
it ran. Offered as a hypothesis, not a conclusion.
