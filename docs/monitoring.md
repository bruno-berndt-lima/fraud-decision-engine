# Monitoring — what can be watched without labels, and what may fire a retrain

| | |
|---|---|
| **Status** | Registered before any Phase 09 number exists. |
| **Last updated** | 2026-09-23 |
| **Watches** | The shipped system — `models/model.txt`, its tables, the calibrator, the frozen policy |
| **Changes** | Nothing. Everything Phase 06 froze, Phase 07 explained and Phase 08 shipped stays as it is (§9) |
| **Touches test** | As measurement, in its own stage. Nothing here names `$(HEADLINE)` as a prerequisite |

---

Phase 09 asks a different question from every phase before it. Those asked whether the
system is good; this one asks how anyone would *find out that it stopped being good* —
and, critically, how they would find out at the moment they must act rather than a month
afterwards.

That timing constraint is the same one the whole project is built on. `problem-statement.md`
A1 assumes a 30-day label-maturity window, and Phase 02 spent 30 days of a 62-day budget
purging a gap to honour it. The consequence for monitoring is direct and is the reason
this document is shaped the way it is:

> **A monitor that needs labels is always 30 days late.** At the moment a production
> system must decide whether it has a problem, the labels for the period in question do
> not exist yet. Everything that can be watched *now* is label-free; everything that
> needs a label is retrospective confirmation.

So the work splits in two, and the split is not a compromise forced by a short test
window — it is what production monitoring actually looks like.

| horizon | days | labels | carries |
|---|---|---|---|
| **labelled** | 121–182 | yes | §3 feature drift, §4 decay, §5 composition (against train) |
| **unlabelled** | 213–395 | no | §3 feature drift, §6 prediction drift |

The unlabelled horizon is `test_transaction.csv` and `test_identity.csv` — the Kaggle
files whose labels were withheld and never released (`data-provenance.md`). They have no
`isFraud` and they never will, which makes them an honest stand-in for the rows a live
monitor sees: real transactions, scored by the shipped model, whose outcomes are not
known yet.

The registration discipline is `decision-policy.md`'s, `explainability.md`'s and
`serving.md`'s, for the reason all three gave. It bites hardest here: a drift threshold
chosen after seeing the drift table is a fit to it, and a retraining rule tuned until it
fires the number of times that feels right is not a rule.

**To fix before the first run** — values, not decisions. A `monitoring` section of
`config.yaml` holds the window widths, the PSI bin count and its epsilon, the two
references, how many features the summary reports, and the §7 thresholds. A **separate**
`horizon` section holds the expensive build's knobs, for the reason `explain_latency` is
separate from `explain`: a stamp covers a whole section, and correcting a bin count must
not restage a build of half a million rows.

## 1. What is watched, and what is not

**The shipped system, end to end.** Not the booster alone — the object that decays is the
pipeline plus the model plus the policy, and a monitor that watches only the model's
inputs will miss a threshold that has drifted out of the money.

**Four signals, in the order they become available:**

| # | signal | needs labels | earliest |
|---|---|---|---|
| 1 | Feature drift — PSI per column against a training reference (§3) | no | immediately |
| 2 | Prediction drift — the score distribution and what the frozen policy does with it (§6) | no | immediately |
| 3 | Composition — the *coverage rate* of the inputs the model leans on (§5) | no | immediately |
| 3b | Composition — how much of a performance gap that coverage explains (§5) | **yes** | one maturity window later |
| 4 | Performance — PR-AUC against the frozen baseline (§4) | **yes** | one maturity window later |

**What is deliberately not watched.** Nothing here monitors the *service*: request rates,
error rates, fallback counts and latency are `serving.md` §4's, exposed on `/health`, and
belong to an operational dashboard rather than to a model-quality record. The one place
they meet is §7, which may not fire a retrain on an availability incident.

### Result

*Pending.*

## 2. The window

**One definition, in `monitoring/windows.py`, shared by §3, §4 and §6.** The precedent is
`evaluation/timing.py`: three sections meant to be read against each other cannot each
hold their own idea of a window, or a PSI number and a PR-AUC number that appear side by
side will silently describe different spans.

**Fixed width, anchored at the first day of its horizon.** Forward from the boundary,
because every question here is "how far has this moved from where it started", and the
reference is the start.

**A trailing remainder shorter than the width is reported, marked `partial`, carries its
own day and row counts, and is excluded from §7.** Dropping it would throw away the most
recent data, which in a decay analysis is the data that matters; folding it in unmarked
would put a short window on a trend line as though it were a full one.

**Two widths, because the two horizons answer different questions.**

**Labelled, width 5 days.** The available span is days 121–182, so a monthly window is
not on offer at any split allocation — `problem-statement.md` §7 item 5 settled that in
Phase 02 and it is not reopened here. Five is chosen because it divides `VAL-FIT` and
`VAL-CAL` exactly, which is what keeps a window from straddling a split boundary and
mixing a slice the model early-stopped against with one it did not:

```
val_fit  121-125  126-130  131-135  136-140
val_cal  141-145  146-150  151-155  156-160
test     161-165  166-170  171-175  176-180   181-182 (partial)
```

Twelve full windows of roughly 14,000 rows each, and one partial of two days.

**Unlabelled, width 28 days.** The roadmap's monthly window, available here because the
horizon is 183 days long:

```
213-240  241-268  269-296  297-324  325-352  353-380   381-395 (partial)
```

Six full windows, and one partial of fifteen days.

**The horizons are not contiguous, and the gap is stated rather than smoothed.** Labelled
data ends at day 182; the unlabelled file begins at day 213. Thirty days are missing
between them. The first horizon window is not "the month after test" and no chart may
draw a line implying that it is.

### Result

*Pending.*

## 3. Feature drift — PSI

**The reference is `train`, days 1–90.** Not the previous window. Drift here means
"distance from what the model was fitted on", which is the quantity that bears on whether
the model is still appropriate; window-over-window movement answers a different question
and would make a slow, monotone shift invisible by comparing each step to the last one.

**Computed on both horizons, and the labelled one is where it can be tested.** On the
unlabelled horizon PSI is the monitor. On the labelled windows of §2 it is something
rarer: the only place in the project where a PSI number and a PR-AUC number exist for the
same rows, so the only place the claim that drift *leads* decay can be checked rather than
assumed. §7 builds its label-free conditions on that claim; the labelled windows are
where it is read, and the result is reported whichever way it goes.

**How it is read, fixed before any PSI exists.** Over the twelve full labelled windows,
the rank correlation between each window's contribution-weighted PSI (§7's condition 1
quantity) and its PR-AUC shortfall against the `VAL-CAL` baseline, reported with its n —
and whether the window §4's rules flag is the one with the highest weighted PSI. Twelve
points cannot establish that drift leads decay. They can contradict it, and a correlation
of the wrong sign, or a flagged window with unremarkable PSI, is exactly that.

**Ten quantile bins, cut on the reference.** Quantile rather than equal-width, for the
reason `calibration.ece_bins` is quantile: on a skewed distribution — the score at a 3%
base rate there, `TransactionAmt` and most counts here — equal-width bins put nearly
everything in the first one, and PSI then measures the tail alone. The cut points come
from the reference and are held fixed across every window, or the comparison is between
two different rulers.

**Ties collapse bins, and that is allowed.** Many `C*` columns are counts with most of
their mass at zero or one, and flags have two values, so ten quantiles of such a column
share edges. Duplicate edges are merged, a column may carry fewer than ten bins, and the
bin count is recorded beside its PSI: a column scored on three bins is not comparable in
magnitude to one scored on ten, and the table says which is which. **Every edge is a value
the reference holds** — the empirical quantile, never an interpolated one, which would put
an edge between a run of zeros and a run of ones and open a bin nothing can land in.

**Missing is its own bin, never imputed.** This is the one that matters most. A null-rate
shift is the most common real drift in production — an upstream field stops arriving —
and filling it with a median converts the most detectable failure into an invisible one.
The consequence is registered here and taken deliberately: **PSI is computed on the
pre-imputation matrix**, which is not quite what the booster consumes. The booster sees
medians in those positions. That difference is the point.

**Both sides at the same stage: after the vocabulary, before the medians.** The horizon
matrix is the service's transform with the fill left out; the reference is
`data/features/train.parquet` routed through the same shipped vocabulary. The vocabulary is
on both sides because it *is* the categorical binning, above; the medians are on neither,
because a fill erases the null-rate drift this section exists to see. A reference and a
window taken at different stages differ by the pipeline, not by the world.

**Categorical columns bin on the shipped vocabulary, through the function the booster's
inputs pass through.** Each level `categories.parquet` holds is a bin, its two sentinels
included: `OTHER` takes every level the vocabulary does not hold — new on the horizon, or
too rare in train to have earned a code — and `MISSING` takes nulls. The values are
routed by `train.apply_categories` itself rather than by a copy of it, so the bins are
exactly the distinctions the booster can make, and there is no second minimum-share
threshold to tune.

**The `OTHER` share is also reported on its own, as the unseen-category rate**, on both
sides: in the reference it is not zero, because the rare training levels live there.
Inside PSI it is one bin among several; beside it, it is the number that says new device
strings are arriving. It is never merged with the missing bin, and the guarantee is the
model's own: `apply_categories` turns a null into `MISSING` before it tests membership,
so an influx of new values cannot read as an upstream field going silent. Two different
things happened; two numbers say so.

**The zero-count guard is a registered value, not a convenience.** PSI takes a logarithm
of a ratio, so an empty bin on either side is undefined. The epsilon lives in
`monitoring.psi.epsilon` and is added to both sides, so it cannot quietly inflate one.

**The convention, and what it is worth.** PSI < 0.1 stable, 0.1–0.25 moderate, > 0.25
significant. This is the credit-scoring convention, it has no distributional derivation,
and it is used here because it is what a risk team expects to read — not because it was
earned. It is stated as a convention in the README for the same reason.

**Every feature the horizon can support is computed; the top ones by *contribution* are
reported.** That is all 349 on the labelled side and 345 on the horizon, per the first
exclusion below. A table of 345 columns × 7 windows is a record, not a finding. Which
ones lead the summary is decided by `explainability.md` §4's contribution ranking rather
than by PSI magnitude or by LightGBM's gain, because the question a reader has is "did
the columns that move decisions move", and a large shift in a column the booster barely
consults is not that. The full table is persisted regardless.

**The weighted aggregate uses the same columns on both horizons.** Condition 1 takes the
top-`k` by contribution *among the columns the horizon can support*, weights renormalised
over those `k`, on the labelled windows as well as on the unlabelled ones. A `vel_*`
column ranks inside the booster's top ten, and it is excluded on the horizon (below); an
aggregate over different columns on each side would make a labelled number and a horizon
number two different quantities, and the drift-leads-decay reading above would say
nothing about the horizon it is meant to license.

**Two exclusions, both registered now.**

1. **The four `vel_*` columns are excluded from horizon PSI.** The horizon is scored
   with them filled as the service fills them (§8) — the family's no-history defaults,
   the same constant on every row. Their distance from the training reference is then
   fixed and large and never moves, so no window-over-window reading of it can signal
   anything; computing them from the file instead would measure a card history that
   restarts at the file boundary, not the world. The first reason outlives this
   dataset: **a monitored system cannot detect drift in a feature it defaults.** That
   is a real cost of `serving.md` §2's decision, discovered here, and it belongs in
   the README's limitations rather than buried in a config comment.
2. **`C*` and `D*` are computed, reported, and read with a caveat that cannot be
   removed.** They are Vesta's aggregates over windows nobody published, computed for a
   different file. If they shift across the boundary there is no way to tell real drift
   from an artifact of how they were computed for the unlabelled rows. **Registered
   reading rule: a large PSI on a tier-0 column is reported and is not attributed.** It
   may not be called drift and it may not be dismissed as an artifact. This is the
   project's principal limitation arriving in a new place, and it is handled the way
   E3's sign was handled — by refusing to read it.

### Result

*Pending.*

## 4. Performance decay

**The chart Phase 02 paid for.** Thirty days of purge and a temporal split exist so that
degradation can be *shown* rather than asserted, and this is where that is spent.

**PR-AUC per window across days 121–182**, plotted against days elapsed since the
training boundary at day 90, with each point carrying its slice, its row count and its
positive count.

**No model is loaded, and nothing is re-scored.** Every score this needs is already
persisted: `data/predictions/headline_test.parquet` carries the test rows the single
touch scored, and `data/predictions/lightgbm_tuned.parquet` carries `VAL-FIT` and
`VAL-CAL`. The stage reads score vectors from disk and computes a metric, which is the
`evaluation/figures.py` precedent — a figure is a view of what a run said, and must not
be able to produce numbers the records disagree with.

**This is the one touch, and it is measurement.** `decision-policy.md` §7 and the
methodological invariant both say test is touched once for a *decision*; re-reading a
persisted vector to compute a metric changes nothing and decides nothing. The stage
therefore does not name `$(HEADLINE)` as a prerequisite — it cannot, because that target
refuses to rerun while its record exists and would leave anything depending on it
permanently unsatisfiable (`explainability.md`'s rule). It checks for the predictions
file itself and says what to run when it is absent.

**Scores, not probabilities.** PR-AUC is invariant under a monotone transform and Platt
is monotone, so ranking scores are used throughout. That is what makes the three slices
comparable at all: `VAL-CAL`'s calibrated probabilities are out-of-fold — four Platt
fits, so not one monotone transform of the score across the slice — and test's come from
the full calibrator. A chart drawn on probabilities would be comparing calibrators.

**PR-AUC moves with the fraud rate, so every point carries its base rate.** A random
ranker's PR-AUC *is* the base rate, and the rate is not constant here: 3.46% on `VAL-FIT`,
3.15% on `VAL-CAL`, 3.70% on test, and five-day windows move further than slices do. A
window with less fraud in it shows a lower PR-AUC from a model that has not changed at
all. ROC-AUC does not depend on prevalence, so it is carried beside PR-AUC on every point
— the secondary metric doing the job the Phase 02 metrics table gave it. **Registered
reading rule: a PR-AUC fall that ROC-AUC does not share is read as prevalence, not
decay.**

**And its identity share, for §5's reading rule.** Composition can only move a window's
PR-AUC through a change in mix, so a window that does not carry its mix cannot be read
against E5 at all.

**`VAL-FIT` is optimistic and is marked optimistic.** The model early-stopped on it. Its
four points are drawn and annotated rather than dropped, because the step from `VAL-FIT`
to `VAL-CAL` is worth seeing: it mixes the early-stopping optimism E6 sized with twenty
more days of distance and a lower fraud rate, and the chart shows the step without
attributing it.

**A bar, registered before the curve exists.** Twelve windows of ~500 positives each will
move on sampling alone. Each point carries a bootstrap interval over the days inside its
window — the `usd_halves` machinery, resamples and seed from config — and the **registered
reading rule** is: a window whose interval overlaps the baseline is not a decline. E6's
lesson one level further on. A bar written after seeing the curve is a fit to it.

**What that interval is, and what it is not.** Resampling days inside a five-day window
gives the bootstrap five units to draw from, so the interval it produces is coarse by
construction — it is a detector of large moves, not a significance claim, and it is
reported in the register `explainability.md` §3 used for the additivity check. Resampling
transactions instead would give a narrower interval by ignoring the day-level clustering
`usd_halves` was built to respect, which would be a tighter number about a weaker
question. The width is accepted: a bar whose job is to refuse small moves is not improved
by making it small.

**The baseline is `VAL-CAL`'s pooled figure.** The chart replays a deployment at day 161,
and the last number measured before that day is what a team would have expected the
model to do. Test's pooled figure would judge each test window against an average it is
a quarter of. §7 looks forward from today instead, so its baseline is test.

**What the chart may not do.** It may not move a threshold, a hyperparameter or a
feature, whatever it shows. §9.

### Result

Record `reports/metrics/decay.json`, figure `reports/figures/decay_pr_auc.png`. Before a
window was drawn, the pooled PR-AUC and ROC-AUC of each slice, recomputed from the vectors
on disk, matched their records exactly.

**No test window is shown to decline.** Every one of the five reaches the `VAL-CAL`
baseline of 0.5215 with its interval, across days 73 to 92 after the training window
ended. Twelve of the thirteen windows read `no decline`; none reads `prevalence`.

| slice | windows | PR-AUC range | identity share | verdicts |
|---|---:|---:|---:|---|
| `VAL-FIT` (early-stopped on) | 4 | 0.469–0.683 | 17.87% | no decline |
| `VAL-CAL` | 4 | 0.445–0.616 | 17.29% | one decline |
| test | 4 + 1 partial | 0.478–0.561 | 22.76% | no decline |

**The one decline is inside the baseline's own slice.** Days 146–150 of `VAL-CAL`: PR-AUC
0.445 [0.430, 0.468], ROC-AUC below as well, identity share 17.8% — inside the range
validation spans, so mix can explain none of it. The baseline is the pooled figure of the
slice this window belongs to, so what the rule found here is variation *within*
`VAL-CAL`, before the replayed deployment, not decay after it.

**A five-day window moves a long way on its own.** Inside `VAL-FIT`, PR-AUC falls from 0.683
to 0.469 in twenty days; inside `VAL-CAL` it ranges between 0.445 and 0.616, with no
direction to it. The intervals are as wide
as §4 said they would be — five days to resample is a detector of large moves — and that
width is what keeps any single window from being read as a trend.

**Two observations, neither a verdict.** The rules read PR-AUC first and use ROC-AUC only
to tell a decline from prevalence, so these are recorded, not read:

- **ROC-AUC dips where PR-AUC does not.** Its interval sits wholly below the baseline in two
  test windows — days 166–170, and the partial 181–182 — while PR-AUC's does not.
- **Identity coverage comes back in test.** 28.3% in its first window and 22.8% across the
  slice, against 17.3–17.9% in validation. Those are the rows the model ranks best, and no
  test window rises above the baseline with them; by E5's arithmetic, mix alone would
  have pushed PR-AUC up.

**One correction, made before this record.** The first run's code took the identity
share's distance from the validated range in either direction, which gave windows with
*more* identity a bound pointing the wrong way. §5 registered a fall as composition only
if the share moved with it, so the code was corrected to count only a share below the
range. No verdict changed, and every window's figures were identical across the two
runs.

Nothing here moves a threshold, a parameter or a feature (§9).

## 5. E5 — composition, or generalisation?

**Registered in `experiments.md` as Phase 09's, and run before §4 rather than after**,
because its result decides how §4 is read.

The question: identity coverage falls 38% in relative terms across the split boundary.
When performance falls with it, is the model failing to generalise, or is validation
simply carrying less information per row? Those have opposite responses — one calls for
retraining, the other for fixing an upstream pipeline — and nothing else in this project
distinguishes them.

**The method is E5's, with what its registration left implicit made explicit.** E5's
table compares train rows with validation rows, and it has to: the coverage fall happens
between them. `VAL-FIT` and `VAL-CAL` carry nearly the same identity share, so comparing
them with each other would test nothing. Train's scores are the one vector not on disk,
so this stage reloads the shipped booster, proves it reproduces its `VAL-CAL` record
exactly (`evaluation/reproduce.py`), and scores train. Both of E5's constraints hold: no
refit and no retune, and test's coverage stays out of every figure and argument.

**Train's scores are in-sample, and that is stated rather than corrected.** The booster
memorised those rows, so the train column sits at a level no validation window will
reach, and no train number is read as a generalisation estimate. What survives is the
comparison E5 asks for: whether the gap *within* each stratum is smaller than the gap
pooled.

**PR-AUC does not add across strata**, so "the share attributable to composition" needs a
method, and it is registered here rather than chosen afterwards. Validation rows are
reweighted to train's identity mix and PR-AUC is recomputed with those weights. The
difference between the reweighted and the raw validation figure is the compositional
part; the remainder of the pooled gap is what composition does not explain.

**Registered reading rule for §4, and why it is built from validation alone.** The train
column is in-sample, so both strata sit far above validation whatever composition does. A
rule asking whether within-stratum PR-AUC *holds* across the boundary could only ever
answer no, and a rule with one possible outcome is not a rule. §4 therefore inherits two
out-of-sample quantities instead: the compositional part, which says how far a mix shift
of the size the boundary produced moves PR-AUC on rows the model never saw, and the two
strata's validation figures, which say how differently it ranks with and without an
identity block.

**A window-to-window PR-AUC move is read as composition only if the identity share moved
with it**, and by no more than the compositional part scaled linearly to the size of that
move — an approximation, registered as one. A window whose share stays inside the range
`VAL-FIT` and `VAL-CAL` already span cannot owe its move to mix. Whatever composition does
not cover is the model's, and §7's PR-AUC condition is the one that applies to it.

The share of the in-sample gap is still reported, because it is the quantity E5 was
registered to answer — what part of an overfitting-shaped gap is not overfitting — but §4
does not read it. Test windows' identity shares are §4's measurement, taken under §9's
standing; E5's own figures stay on `VAL-FIT` and `VAL-CAL`, as its first constraint
requires.

### Result

Full table in `experiments.md` E5; record `reports/metrics/composition.json`.

**The train column saturated, as the amendment above anticipated.** In-sample PR-AUC is
exactly 1.0 in both strata, so the registered share — 16.6% of the pooled gap — measures
distance from a perfect ranking and is not read here.

**What §4 inherits, both out of sample:**

| quantity | value |
|---|---:|
| compositional part, for an 11.44-point shift in identity share | +0.07338 PR-AUC |
| per point of identity share | ~0.0064 PR-AUC |
| identity share already spanned by `VAL-FIT` and `VAL-CAL` | 17.29%–17.87% |
| validation PR-AUC with an identity block / without | 0.777 / 0.291 |

A decay window whose identity share sits inside that range cannot owe a move to mix. One
outside it can owe at most about 0.0064 PR-AUC per point of distance, and anything beyond
that is the model's.

**The `VAL-FIT` → `VAL-CAL` step is already one such case.** The share moved by 0.58
points and PR-AUC fell by 0.067, concentrated in rows without an identity block. Mix
could account for about 0.004 of it.

## 6. Prediction drift

**The most direct label-free signal.** It needs nothing but the model and the rows, and it
folds every feature's movement into the one quantity the policy acts on.

**Three views, per window, over the unlabelled horizon:**

1. **The score distribution** — PSI on the calibrated probability itself, plus p50, p90
   and p99.

   **Its reference is test, not train, and the difference from §3 is deliberate.** The
   booster memorised the training rows, so its scores there are not the scores it
   produces on data it has not seen; a horizon window compared against them would measure
   overfitting and report it as drift. Those scores are not even on disk — the shipped
   run scored `VAL-FIT` and `VAL-CAL` only. Test is out-of-sample, is the slice the
   headline was measured on, is the closest labelled data to the horizon, and is already
   persisted. §3's reference stays `train` because a *feature* distribution on train is
   exactly what the model was fitted against; a *score* distribution on train is not.
2. **Mean calibrated probability**, which on a calibrated model is an estimate of the
   base rate. The headline record gives the anchor: 3.74% predicted against 3.70%
   observed on test.
3. **What the frozen policy would do** — the EV policy run over the horizon's scores and
   amounts, reporting block rate and review-eligible rate per window. Label-free, and in
   the unit the business actually reads. A threshold that has drifted out of the money
   shows up here and in no other signal.

**Registered reading rule, and it is the important one in this section.** A rise in mean
predicted probability has two readings — fraud rose, or the model drifted — and
**without labels they cannot be distinguished.** Both are reported; neither is chosen.
This is E3's refusal applied to a different object, and a monitor that picked one would
be inventing the evidence that would settle it.

**The policy is run, never re-derived.** `evaluation/cost.py`'s functions with
`cost_matrix.yaml` v1 and the committed capacity. A block rate computed from a threshold
this phase invented would measure this phase.

### Result

*Pending.*

## 7. The retraining trigger

**The rule, whichever condition fires first:**

| # | condition | needs labels |
|---|---|---|
| 1 | Contribution-weighted PSI over the top-`k` features exceeds its threshold | no |
| 2 | PSI on the score distribution exceeds its threshold | no |
| 3 | Window PR-AUC falls more than `x`% below the frozen test baseline, interval clear, ROC-AUC agreeing (§4) | **yes** |
| 4 | Fixed cadence | no |

**Every threshold is committed to `config.yaml` before the run that evaluates it.** This
is the section where the registration discipline earns its keep: thresholds chosen after
seeing the PSI table would describe this dataset rather than state a policy, and the
rule would be unfalsifiable by construction.

**Weighted, not top-10-by-PSI.** Condition 1 uses `explainability.md` §4's contribution
mass as the weight, for §3's reason: PSI detects a change in inputs, not a loss in
performance, and the link between them runs through how much the model actually uses the
column. An unweighted rule fires on columns the booster barely consults and stays quiet
when a leading one moves.

**Justified against label maturity, which is the DoD item and the honest part.** The lag
arithmetic, stated as a sum:

```
detection lag  +  label maturity  +  fit and ship  =  response time
```

With a 30-day maturity window, data from day *T* cannot be trained on until *T* + 30. A
trigger firing on day *T* therefore produces a model whose training data ends at *T* − 30
at the very best, before any time is spent fitting, validating and shipping it. **The
consequence is structural: condition 3 can only ever fire late.** By the time a labelled
PR-AUC decline is visible and its interval is clear of the baseline, the decline has been
running for at least a maturity window. That is exactly why conditions 1 and 2 carry the
rule and why condition 1 is weighted rather than raw — the label-free signals are not a
weaker substitute for the labelled one, they are the only ones that can act in time.

**What the trigger may not fire on.** A service incident (§1), a partial window (§2), or
one column's PSI read on its own (§3). Condition 1 is an aggregate weighted by
contribution, and the columns leading that ranking are tier 0, whose individual shifts
§3 refuses to attribute. That refusal governs what this project may *claim* about the
horizon; it does not exempt those columns from a production rule, because a retrain
answers a moved input whatever moved it.

**Evaluated against what was measured, not asserted.** The stage reads the records this
phase wrote and reports which condition would have fired first, on which window, and
what the earliest legitimate retrain date would have been given the lag above. A rule
nobody ran against data is a paragraph.

### Result

*Pending.*

## 8. What the horizon build is proven against

**Nothing measures until it has been shown to reproduce something known.** Phase 06
registered that rule for models — `evaluation/reproduce.py` — and it applies with more
force to a measurement pipeline, because a subtly different matrix produces a PSI table
that is wrong and looks entirely plausible.

**The guard, and it is the stage's first step rather than only a test.** Test's labelled
rows go through the horizon's own path, carrying the tier-3 history the training matrix
built for them, and must reproduce that matrix on every column, value and dtype — and the
single touch's persisted scores, uncalibrated and calibrated, to the bit. This is
stronger than the tier 0–2 comparison first registered here, amended before any horizon
row was scored: agreeing columns could still feed the booster differently, and agreeing
scores could still hide a column the booster never splits on but PSI reads. The pieces
it composes are tested in CI against a synthetic deployment; the whole of it needs the
shipped artifacts.

**Tier 3 is filled the way the service fills it.** The horizon file starts thirty days
after the labelled data with no card history behind it, so computing the four `vel_*`
columns there would score every card as first-seen for a reason the service never
produces. §1 watches the shipped system, and `serving.md` §2 decided what that system
writes into those columns: the family's no-history defaults, through
`serving/transform.fill_history`. The builder is the service's transform, which calls that
same function, so the horizon is scored as the service would have scored it.

**What the guard proves about tier 3, and what it cannot.** It supplies the labelled rows'
real history, which the transform keeps, so it proves the path, the medians, the booster
and the calibrator — and says nothing about the defaults. Nothing could: a default
compared against a matrix built from history measures its cost, not its correctness, and
`serving.md` §2 measured that cost on `VAL-CAL`. Tier 3 stays out of horizon PSI (§3).

**No refit, ever.** `features/build.py` fits frequencies, entity amount statistics and
the V-block reduction on `split == "train"`. The horizon has no train rows and must not
acquire any: it is the service's transform, which applies the tables already shipped in
`models/`, read through `serving/artifacts.py`. A builder that refitted on the horizon
would encode the drift it exists to measure.

**The identity header differs, and it is caught twice rather than by a comment.**
`train_identity.csv` names its columns `id_01`; `test_identity.csv` names them `id-01`.
Unhandled, the join raises nothing — it matches on `TransactionID`, so even
`has_identity` comes out right — but the `id_01` … `id_38` the model reads are not among
the columns that arrived. Filled as absent, they are null on every identity row, the
missing bin of all 38 fills, and PSI tells a story that is completely plausible and
false: an upstream identity feed gone silent. This is the exact failure a monitoring
pipeline is supposed to catch, happening to the monitoring pipeline. So the block is
renamed on load, and the stage then refuses a horizon missing any input the model reads:
the transform's own rule, that an omitted field is a null, is right for a request and
wrong for a file.

**The unlabelled files are not checksum-enforced, and that stays true.**
`data-provenance.md` deliberately keeps them out of `raw_checksums.txt` so `make data`
does not fail for anyone who deleted 639 MB they did not need. The horizon stage gets its
own stamp over its own pair, so the control exists exactly where it is used and nowhere
else.

### Result

*Pending.*

## 9. What this phase may not change

`decision-policy.md` §7 froze the tuned booster with its vocabulary and medians, the
Platt calibrator, cost matrix version 1, the review capacity and the `cost.py` policies.
Phase 07 explained that system, Phase 08 served it, and Phase 09 watches it. None of the
three adjusts it.

**The invariant, stated the way the README must state it.** Phase 02 says test is touched
once. Phase 09 scores test window by window. These are not in tension, and the resolution
is the distinction the roadmap draws: **measurement is unlimited; tuning is the thing you
only get to spend once.** No threshold, no hyperparameter and no feature moves because of
anything measured here. A reader who is not told this in as many words will read the two
phases as a contradiction, and they will be right to ask.

**It holds especially where the result is unflattering.** A feature whose PSI is large, a
decay curve that falls, a trigger that would have fired on the data already collected —
each is written down, carried into the README's limitations, and acted on by nobody in
this phase. Retraining in response to a number produced here would mean a model selected
on test, which is the one thing the whole structure exists to prevent.

**What this phase may change is what the project claims**, and what it says it would do
next. The queue-depth shed `serving.md` §7 argues for, a feature store that would make
the tier-3 family monitorable, a labelled evaluation window long enough to hold a monthly
decay chart — those belong in the write-up as named next steps, with the evidence that
motivates them, and not in `config.yaml`.
