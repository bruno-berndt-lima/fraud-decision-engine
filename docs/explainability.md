# Explainability — what the model used, and what a customer can be told

| | |
|---|---|
| **Status** | Registered before any Phase 07 number exists. |
| **Last updated** | 2026-09-16 |
| **Explains** | The shipped booster — `models/model.txt`, its vocabulary and medians |
| **Changes** | Nothing. Everything Phase 06 froze stays frozen (§9) |
| **Touches test** | As measurement, in its own stage, never through `headline.py` |

---

Phase 07 attaches an explanation to a model that is already chosen, already calibrated
and already measured. That ordering is what makes this document short on decisions that
could move a number and long on decisions about what may honestly be *said* — the only
thing still open.

Two audiences, and they want different objects. A regulator asks whether the decision
was made on grounds the firm can state and defend; a customer-service agent asks what to
say to the person on the phone. Both are served by the same contributions and neither is
served by a feature importance ranking, so §7 is where most of the work is.

The registration discipline is `decision-policy.md`'s, for the same reason: a reading
rule written after seeing the beeswarm is a fit to it.

**To fix before the first run** — values, not decisions, and they belong in the `explain`
section of `config.yaml`: the beeswarm sample size, its seed, and `top_k` for reason
codes.

## 1. What is explained, and on which rows

**The booster, not the pipeline.** Contributions are computed on the matrix the model
actually consumes — post-imputation, post-vocabulary — because that is where additivity
holds. A contribution attaches to a column, so a category's contribution is the encoded
column's, and turning a code back into a level is §7's job.

**Two splits, both of them.** `VAL-CAL` (days 141–160) and test (days 161–182).

Test because the roadmap asks the global view to sit on the split the headline was
measured on, and because an explanation of a model on data three weeks from its training
boundary is the one that describes the shipped system. `VAL-CAL` because computing it
costs one more scoring pass and buys a comparison nothing else in the project provides.

**Registered reading rule for the comparison.** A material difference between the two
rankings is a *decay signal, recorded for Phase 09 and acted on by nobody in Phase 07*.
It is not evidence that a feature is broken, and it may not start a feature change — §9
holds regardless of what this shows.

**Test is measured, not re-touched.** `headline.py` refuses to run twice by design;
this stage is separate, writes its own records, and computes no policy figure that the
headline already reported. `decision-policy.md` §7 anticipated exactly this shape when it
said Phase 09's decay chart needs its own stage.

### Result

Both splits explained, 10,000 rows each, from `reports/metrics/shap_global.json`.

**Nothing diverged.** The serving-tier shares move by less than two points between
`VAL-CAL` and test, and the top of the ranking is the same set of columns in nearly the
same order. The registered reading rule was written for a difference that did not
appear: there is no decay signal to carry into Phase 09 from this comparison, which is
a result rather than an absence — the alternative was a model whose grounds had shifted
in the three weeks after the calibration slice, and it did not.

## 2. How contributions are computed

**LightGBM's own TreeSHAP**, through `Booster.predict(..., pred_contrib=True)`. Exact,
no sampling, and no background dataset: the tree-path-dependent algorithm takes the
reference distribution from the training-time cover already stored in the trees.

**Not `shap.TreeExplainer`'s interventional mode**, which takes a background sample and
answers a different question. Either is defensible; mixing them is not, and a document
that did not name one would leave the numbers unreproducible.

**Consequence, taken deliberately:** the `shap` package is a *plotting* dependency here.
Nothing in the serving path needs it, which keeps it out of the Phase 08 image.

**Proven before use.** The reloaded booster must reproduce its recorded `VAL-CAL` scores
exactly — `evaluation/reproduce.check_reproduces`, as `headline.py` and `usd_halves.py`
already do — before a single contribution is computed. SHAP attached to a model is
something new attached to a model, and Phase 06 registered that rule for this case.

**Persisted, then drawn.** Contributions are written once; every figure and every reason
code reads that file. A figure that could be redrawn from a model is a figure that can
disagree with the record beside it, which is why `evaluation/figures.py` exists at all.

**One sample per split, for the ranking as well as the plot** — amended from the original
registration, which said the ranking would run over every row. Exact TreeSHAP on a
booster this size costs enough per row that whole splits are hours of compute, and the
cost was measured before any contribution was read. What the width buys is the top of the
ranking and the §4 tier shares, which the large contributors carry; **the tail of the
ranking is not read**, and no claim in this phase may rest on it.

### Result

The proof passed before any contribution was computed: the reloaded booster reproduced
its recorded `VAL-CAL` scores exactly.

**The stage is deterministic, and this was measured rather than assumed.** It was run
twice — the first record was stamped `-dirty`, because unrelated files were edited while
it ran, and the project does not ship numbers whose code is not in history. The rerun
differed from the first in two lines, its timestamp and its revision; every contribution,
every tier share and both deviations were byte-identical.

`shap` was never imported. Contributions came from `Booster.predict(pred_contrib=True)`,
which is what keeps the dependency out of the Phase 08 image.

## 3. The space contributions live in

**Raw log-odds, and nothing else.** Contributions sum to the booster's raw margin plus a
base value. They do not sum to the calibrated probability, and the calibrator is not part
of the sum.

The shipped calibrator is Platt with a positive slope, so it is strictly increasing:
signs and ordering survive it, magnitudes do not. A feature that pushed the margin up
pushed the probability up; *how much* probability it was worth is not defined, because the
map depends on where the rest of the row put the margin.

**Registered: no reason code states a contribution as a share of probability, or as a
percentage of anything.** The tempting sentence — "the amount accounted for 40% of this
decision" — is not true in any space this model computes in. Contributions are reported
as signed log-odds, and ranked; the text a customer sees is ordinal.

**Additivity is asserted, not assumed.** Every run checks that contributions plus the
base value reproduce the raw margin on every row explained, and records how far the
worst row sat from it. Not *exactly*: TreeSHAP's unwind step divides by subset weights
and loses digits doing it, so bit equality is false on a correct implementation. The
check is sized to catch a decomposition of the wrong model rather than to certify
precision, and the measured deviation is reported so the headroom is visible.

### Result

Both splits passed, with room to spare: the worst row in either sat around four parts in
a hundred thousand of its own contribution mass, against a bound of one part in a
hundred. The measured figures are in the record, per split.

The base value is identical on both splits to every digit — as it must be, since it is
the model's expected margin and not a property of the rows — which is the second guard
reporting rather than merely passing.

## 4. The ceiling on what is explainable

`features.md` classifies all 353 columns into four serving tiers. The same table bounds
explanation, because a column nobody can define cannot be put in a sentence:

| tier | columns | can be phrased? |
|---|---:|---|
| 1 — request-only | 36 | yes, directly |
| 2 — static fitted table | 14 | yes, with the artifact's semantics |
| 3 — live entity state | 4 | yes |
| 0 — inherited, unreproducible | 295 | **no** — Vesta's aggregates over windows never published |

E7 already put a number on the same asymmetry from the other side: the buildable columns
keep about half the model's PR-AUC. This section asks the explanation-shaped version of
that question, and the metric is **the share of total absolute contribution falling in
each tier**.

**Registered reading rule.** If the top contributors for a case are tier 0, the reason
code says so — in the honest generic §7 defines — and does **not** reach further down the
ranking for a tier-1 feature it happens to be able to name. A list filtered until it
produces a quotable sentence is a fabricated explanation, and under the BACEN and LGPD
framing that motivates this phase it is worse than admitting the limit.

**Both outcomes are publishable.** A model whose explanation is mostly unnameable is a
finding about this dataset, stated as such. It is also the strongest available argument
for the tier-1/2/3 columns Phase 04 could not justify on PR-AUC.

### Result

| tier | `VAL-CAL` | test |
|---|---:|---:|
| 0 — inherited, unreproducible | 55.4% | 57.2% |
| 1 — request-only | 25.6% | 24.5% |
| 2 — static fitted table | 15.0% | 14.3% |
| 3 — live entity state | 4.1% | 4.0% |

**Between two fifths and a half of what moves this model can be put into a sentence.**
The rest is Vesta's, and the top of the ranking says so plainly: `C13`, `C1` and `C14`
lead both splits, with `D1` just behind. Those are counters and day-deltas over lookback
windows that were never published — `problem-statement.md` §6 records the assumption, and
E7 measured the same asymmetry from the other side.

**The majority is carried by the tail, not the head** — and this is the part that would
have been got wrong by reading the ranking alone. Inside the top twenty features, tier 0
holds two fifths of the mass and six of the twenty rows; across all 349 it holds well
over half, because there are 295 inherited columns each moving the model a little. The
figure `reports/figures/shap_ranking_by_tier.png` shows the head, and the head is more
nameable than the model as a whole.

What that means for a single decision — how often a reason code has nothing but the
generic to offer — is a per-row question and is measured in §7, not inferred here.

**Use is not value, and the two disagree here more sharply than anywhere else in the
project.** `freq_card1`, `amt_mean_card1` and `amt_z_addr1` sit in the top six of both
splits; the velocity family carries about four percent of the mass. Phase 04 measured
every one of those families as inside the noise floor under a linear probe, and E4 under
a tree found that none of them cleared its width-matched bar when removed. Both readings
are correct. The tree leans on columns whose removal costs nothing, because what they
carry is also carried elsewhere — correlated redundancy, not a contradiction, and not a
reason to revisit a frozen feature set.

## 5. The Phase 01 hypotheses against what the model shows

`hypotheses.md` was written before any model existed, and each hypothesis states the SHAP
behaviour it predicts and what would falsify it. Restated here so the comparison is
against the registered prediction and not a remembered one:

| | prediction for Phase 07 | falsified by |
|---|---|---|
| **H1** | `TransactionAmt` contribution rises with amount *within* `ProductCD`, flattening at the top rather than turning over; amount×product interaction visible | a hump inside a single product; or amount contributing nothing once product is known |
| **H2** | amount's contribution is *spiky* — steps at $150, $300, $450 — rather than smooth; the banded round-amount feature earns importance where a plain round-number flag would not | the plain flag carrying it; the band not holding outside days 1–120 |
| **H3** | `ProductCD` ranks in the top few; address-presence features lose most of their contribution beside it | address presence retaining a large contribution alongside product |

**Where the verdicts are written.** Under each hypothesis in `hypotheses.md`, where its
prediction already lives. This section carries the method and links to them.

**The interesting direction is the falsifying one.** H1 was already revised once, in
Phase 01, when its own stated criterion fired against the `ProductCD` control; that entry
records the lesson rather than hiding the revision, and it is the template for whatever
happens here.

**Two constraints on the comparison.** The hypotheses were formed on days 1–120, and the
model is being read on days 141–182 — so a hypothesis can fail here because the mechanism
was never real *or* because it did not survive sixty days, and the two are not separable
with what this phase measures. Say which one is unresolved rather than picking. And a
hypothesis about the world is not refuted by a model that did not need the feature: SHAP
reports what this booster used, given everything else it was handed.

### Result

*Pending.*

## 6. Local cases

Three waterfalls, chosen by rule rather than by eye:

- a **true positive** — the model was right and the decision was adverse;
- a **false positive** — a blocked transaction that was legitimate, the case the
  $15-per-false-positive assumption is actually about;
- a **high-value catch** — the largest-amount fraud the policy blocked, where the bar was
  lowest and the money most material. Ranking by amount and ranking by USD saved pick the
  same transaction: a blocked fraud realises no cost where allowing it realises
  `amount + chargeback_fee`, so the saving is monotone in the amount.

**Selected from what already exists.** `data/predictions/headline_test.parquet` carries
`calibrated`, `amount`, `day` and `isFraud` per test row; running those through
`cost.ev_policy` reproduces the headline's decisions exactly, with no new scoring and no
second touch. The selection rule is written here so the cases are not picked for how well
they read.

**Each waterfall is shown with its bar**, per §7: a local explanation that shows only the
score half is an explanation of the model, not of the decision.

### Result

Figures: `shap_waterfall_true_positive.png`, `shap_waterfall_false_positive.png`,
`shap_waterfall_high_value_catch.png`. Cases were selected from the explained rows
rather than from all of test — a figures module that could score is one that could
disagree with the record it draws from, so the rule selects within a sixth of the
split instead of over all of it. Typical cases are the median amount of their kind;
the high-value catch is the extreme by construction.

**The high-value catch is the figure that justifies §7's whole design.** It is a
$2,259.95 fraud, blocked at a calibrated 2.00% against a break-even of 0.65% — and
`TransactionAmt` is the second-largest contributor *in the direction of safety*, pushing
the log-odds down by more than one. The model found the amount reassuring. The policy
blocked it anyway, because at that size the bar had fallen below the score.

A waterfall alone would have said this transaction was blocked despite its amount. The
truth is that it was blocked because of it. Half the decision lives in a threshold that
`shap` cannot see, and the annotation is not a caption — it reverses the reading.

## 7. Reason codes

The deliverable Phase 08 imports: `explain/codes.py`, `reason_codes()`, with its
human-readable dictionary in `config/reason_codes.yaml` — versioned separately from
`config.yaml` for `cost_matrix.yaml`'s reason, that it is a customer-facing asset whose
edits are not code changes.

### The decision does not decompose into one list

The EV policy has three actions and they are not explainable in the same way:

| part | per-transaction? | explainable from the request |
|---|---|---|
| the **score** — where `p` came from | yes | yes, as contributions |
| the **bar** — `p* = C_fp / (amount + fee + C_fp)` | yes | yes, arithmetically |
| **review eligibility** — expected saving from a review is positive | yes | yes |
| **making the day's review cut** | **no** | **no** — depends on the day's other transactions |

So `reason_codes()` returns the first three and **states review as eligibility, never as
an outcome.** Whether a transaction actually reaches an analyst is queue state at serving
time, and a code that promised a review the capacity could not honour would be a false
statement to a customer.

### The bar is half the explanation

Amount enters the decision twice — through the model's score and through the threshold —
and SHAP sees only the first. A $3,400 transaction blocked at a probability that would
have been allowed at $40 was blocked mostly by its amount, and a contribution ranking
will never say so.

`reason_codes()` therefore takes `amount` and the loaded `Costs` alongside the row's
contributions, and returns the bar in the same object: the probability, the break-even it
was compared against, and which side it fell on.

**Consequence, registered:** the frozen cost matrix is now an input to customer-facing
text, so a future cost-matrix version silently changes what customers are told. The
returned object carries `cost_matrix_version` for that reason.

### Which contributors are listed

**The ones that argue for the decision that was made**, not the largest by absolute value.

For a block — the adverse decision, and the one an explanation is owed for — that is the
top-`k` *positive* contributors. Ranking by `|contribution|` regardless would put "this
looked safe because…" into a decline notice, which is not an explanation of the decline.
For an allow, no adverse decision was taken and no reason code is owed; the function may
still produce one for internal use, and it is marked as such.

### The dictionary

One entry per column the booster can split on: a phrase, and how the phrase changes with
the direction of the contribution. Tier 0 columns get one honest generic — a statement
that the signal came from an inherited aggregate whose definition is not published —
rather than 295 invented sentences.

**Guarded by test.** Every column in the booster's feature list either has an entry or
resolves to the tier-0 generic. A dictionary that has drifted from the matrix fails a
test rather than shipping a blank line to a customer.

### Result

*Pending.*

## 8. What an explanation costs to serve

Not in the roadmap's Definition of Done, added here deliberately — the way Phase 05
adopted E3 and E4's re-runs — because Phase 08 has a p95 budget of 100 ms and the shipped
booster is large.

**Measured:** per-row contribution latency, on the serving path's shape (one row, warm
model), reported beside the scoring latency it adds to. The marginal cost is the figure
that decides anything — scoring happens regardless, and what Phase 08 is choosing is
whether the explanation rides along on the same request.

**At a stated thread count, and more than one.** LightGBM spreads a single prediction
across every core it can see, so a number taken at a development machine's core count
describes a deployment nobody would provision. Both ends of what a container plausibly
gets are measured, and the record carries the machine — a latency figure without one is
meaningless, per `problem-statement.md` §3.1. This is emphatically **not** that section's
load test, which needs a service that does not exist yet.

**If it does not fit**, the fallback is a Phase 08 design input rather than a Phase 08
surprise. Options exist — compute codes only for non-allow decisions, or compute them
off the hot path — and choosing between them is Phase 08's, not this document's.

### Result

Record: `reports/metrics/explain_latency.json`. One hundred single-row calls per
operation, warm, on an x86_64 macOS machine with sixteen cores — the machine is in the
record, and these numbers are true of it and of nothing else.

| threads | score p95 | explain p95 | difference |
|---:|---:|---:|---:|
| 1 | 15.7 ms | 914 ms | 899 ms |
| 4 | 15.1 ms | 1,108 ms | 1,093 ms |

**Scoring fits the budget with room to spare. Explaining does not fit it at all.** The
contribution call costs around sixty times the prediction it decomposes, and roughly nine
times the entire p95 budget `problem-statement.md` §3.1 sets for the whole request. This
is not an implementation that can be tuned out of the way: it is the size of the booster
Phase 05 selected, presenting its bill two phases later.

**More threads made it slower, and that replicated across two independent runs.**
LightGBM parallelises prediction across rows; a single row gives the pool nothing to
divide, so the threads are overhead and nothing else. The practical consequence inverts
the usual assumption — for this call, a one-core container is the *best* case, and
provisioning more cores per worker would make the tail worse. Anyone reading a latency
figure for this model has to be told what thread count produced it.

**What this settles, and what it leaves to Phase 08.** Computing contributions inside the
request is out. The two fallbacks §8 registered before the measurement are both still
open — explaining only the transactions that are not allowed, or moving the work off the
hot path entirely — and choosing between them is Phase 08's decision, with this number as
its input rather than its surprise.

**Read the order of magnitude, not the digits.** A p95 over a hundred calls moved by
about ninety milliseconds between the two runs, and this is a single process with no
concurrent load. It is not the §3.1 load test and does not stand in for one; what it
supports is a conclusion three orders of magnitude clear of the noise.

## 9. What this phase may not change

`decision-policy.md` §7 froze the tuned booster with its vocabulary and medians, the
Platt calibrator, cost matrix version 1, the review capacity, and the `cost.py` policies.
Phase 07 explains that system; it does not adjust it.

That holds **especially** where the explanation is unflattering. A feature that turns out
to carry weight nobody can justify, a contribution that contradicts a hypothesis, a
ranking dominated by columns that cannot be named — each is written down here and carried
into the README's limitations, and none of them is a reason to retrain. The headline was
measured once on frozen inputs, and a model edited after seeing its explanation would
invalidate it.

The one thing this phase may change is what the project *claims*: if the explanation
shows the headline rests on grounds that cannot be stated, that belongs in the write-up,
in the same voice `features.md` used for the principal limitation.
