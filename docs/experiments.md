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
