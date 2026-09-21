# Serving — the model behind an HTTP API, and what it may promise

| | |
|---|---|
| **Status** | Registered before any Phase 08 number exists. |
| **Last updated** | 2026-09-17 |
| **Serves** | The frozen Phase 06 model, its tables, its calibrator and its policy |
| **Changes** | Nothing. Everything Phase 06 froze and Phase 07 explained stays as it is (§9) |
| **Touches test** | As measurement only — replayed rows for latency, never a new headline |

---

Phase 08 puts an already-chosen, already-calibrated, already-explained model behind a
request. Nothing left open here can move the USD figure, so this document is not about
what the system decides — `decision-policy.md` settled that — but about what it can
*deliver under a request*, and what it is allowed to say while doing it.

Three constraints arrive from earlier phases already decided, and this phase does not
get to reopen them: the p95 budget of 100 ms (`problem-statement.md` §3.1), fail-open to
the rules engine on unavailability (§3.1 again), and the serving-feasibility position on
history-dependent features (§3.4). What is genuinely open is narrower: the request
contract, where the explanation lives, what the response promises, and what a container
must carry.

The registration discipline is `decision-policy.md`'s and `explainability.md`'s, for the
same reason both gave: a reading rule written after seeing the latency table is a fit to
it.

**To fix before the first run** — values, not decisions. A `serving` section of
`config.yaml` holds the request-path knobs (the p95 budget the service holds itself to,
the LightGBM thread count, whether fail-open is armed). The load generator's own
parameters go in a **separate** top-level section, for the reason `explain_latency` is
separate from `explain`: a stamp covers a whole section, and a concurrency knob living
beside the serving contract would restage the service's own stamp every time the load
test is re-run.

## 1. What a request carries

**One transaction, shaped like the record the model was trained on.** The request
carries the transaction block, the identity block when it arrived at all, and
`TransactionDT`. The service builds everything else it can and rejects what it cannot.

**Tier 0 arrives; it is not computed.** Of the 349 columns the booster holds, 295 are
Vesta's pre-computed aggregates — `C*`, `D*`, the reduced V block, most of `id_*` —
over windows nobody published. `features.md` records that no store built from this
repository reproduces them, and that does not change because a service now needs them.
So the contract **declares them inbound**: a caller supplies them exactly as the
authorisation record delivers them. This is the principal limitation of the project
stated where a caller can see it, rather than hidden behind an endpoint that appears to
compute its own features.

**What the service does build**, per request:

| tier | columns | built from |
|---|---:|---|
| 1 | 36 | the request itself — amount transforms, `hour` and `weekday` from `TransactionDT`, `has_identity` from whether the identity block arrived |
| 2 | 14 | the shipped fitted tables — frequency encodings, entity amount statistics |
| 3 | 4 | §2 |
| 0 | 295 | the request, unchanged |

**`hour` is the dataset's convention, not wall-clock time.** It is derived from
`TransactionDT` relative to the dataset's own reference, as `load.add_time_columns`
derives it, and `load.py` already records that the reference is unknown. A service that
quietly substituted a real clock hour would be feeding the model a column that means
something else. The contract therefore takes `TransactionDT` and says what it is.

**Malformed is 422, never 500.** The contract is the pydantic model. Beyond that, the
two sentinels training fitted do the work they were fitted for: a category level the
training window never saw routes to `OTHER`, an absent field to `MISSING` or to the
shipped median. An unseen browser version is a level the model has mass on rather than a
null it was never fitted against — `train.fit_categories` was written for exactly the
request that has now arrived.

**What happens when a field does not arrive**, registered before the schema rather than
after it, because the schema is where this stops being behaviour and becomes policy.

**Four fields are required**, and a request without one is a 422: `TransactionDT`,
`TransactionAmt`, `ProductCD` and `has_identity`. Three of them are also refused *null* —
the amount prices the decision, the timestamp places it in the day and the week, and a
null `has_identity` would be read as `True` by the coercion that types it, which is a
claim about the transaction rather than an absence. `ProductCD` may be null, because null
is a level the vocabulary holds.

**Everything else may be omitted, and an omitted field is a null.** This is not a
concession to callers. The booster was fitted on a matrix where the inherited columns are
null on most rows, and the V block's presence flags encode nullness as signal — so a field
that did not arrive is a value the model has mass on, not a hole punched in the matrix. A
request carrying the transaction and nothing else is scored, and scored honestly.

**The risk that creates is named rather than hedged.** A caller whose integration quietly
stops sending half the inherited block would get plausible scores and no error, and the
model's own measurements say those columns are where nearly all its signal lives. Two
things guard it, and only together:

- **Unknown fields are refused.** If a field may be omitted, a *misspelled* field is an
  omitted one — `C13` sent as `C_13` would read as absent and be scored as null. Rejecting
  a name the contract does not have turns a caller's typo into a 422 instead of a
  confident wrong number. Permissive omission and permissive naming cannot both be
  safe; this contract picks the one that keeps the failure visible.
- **The response says what it was given.** Beside §2's declaration of which tier-3 inputs
  arrived, the response carries how much of the inherited block did. A consumer can then
  tell a fully-populated decision from a partial one without inferring it from the score,
  and a monitor can watch that share move.

**Registered.** No feature may be added, dropped or redefined to make this contract
tidier. If a column is awkward to serve, that is a finding for the write-up, not a
licence to change the model.

### Result

**433 inputs**, derived rather than typed: the 349 features the booster holds, less what
the families build, plus the 339 columns the V-block reduction consumes and never shows.
`schemas.request_model` builds the Pydantic contract from that list at startup, so the two
cannot drift — a feature added to the model becomes a field of the contract on the next
deploy, and one that leaves stops being accepted.

**Nearly half of a real record arrives null.** The first transaction of the interim table
carries **199 of the 433 non-null — 46%**. That is not a malformed request; it is what this
data looks like, and it is the measured justification for §1's rule that an omitted field
is a null rather than an error. A contract that demanded them all would reject the
dataset it was built from.

The four required fields and the three refused null are enforced in the schema, so a
request missing one is a 422 naming it. Unknown field names are refused for the reason
registered above, and the response reports `inherited_present` against
`inherited_expected` so a caller that quietly stops sending the block can be seen doing
it.

## 2. Velocity — the tier-3 family, and what serving does without a store

**The problem, stated exactly.** The frozen booster holds four `vel_*` columns. They
need a per-`card1` running window updated on every transaction — the store
`problem-statement.md` §3.4 said would be costed rather than built. E3 has since
reported on it twice: the live-entity store cannot be shown to pay on PR-AUC, and cannot
be shown to pay in USD either. Building one now, to serve a family that has never been
shown to earn its keep, would contradict the measurement this project made.

**Three options, and the one taken.**

1. *Require the caller to supply them.* Honest about the dependency, but no caller in
   this scenario can populate them, so it makes the API undeployable in the name of
   purity.
2. *Build the store.* Contradicts E3, and Phase 08 is not where a feature-store decision
   gets made on no evidence.
3. *Accept them when supplied; otherwise serve the family's own no-history values* —
   the trailing counts a card sees on its first sighting, which are 1 rather than 0
   because `trailing_counts` half-open window counts the transaction itself, and the
   first-sighting recency `features.velocity.first_seen_gap_days` defines.

**Option 3 ships**, and the response declares which tier-3 inputs were present, so a
consumer can tell a scored-with-history decision from a scored-without-history one
without inferring it. The default is not invented for serving: it is the value the
family itself assigns a card it has never seen, which is a value the model was fitted
against rather than a hole punched in the matrix.

**This is train/serve skew, and it is measured rather than asserted.** Serving every
transaction as an unseen card shifts the distribution the model meets. The measurement:
score `VAL-CAL` with the tier-3 columns neutralised, and report ΔPR-AUC and ΔUSD against
the same slice scored with them intact.

**Registered reading rule.** `VAL-CAL`, not test — choosing a serving default is a
*decision*, and decisions do not get to touch test. Out-of-fold calibrated
probabilities throughout, per Phase 06's rule for everything measured on that slice. The
result is read as *the cost of not building the store*, on the shipped model, and as
nothing else. It may not change a feature, a threshold or a parameter whichever way it
falls (§9). If it disagrees with E3 — if neutralising the family costs materially more
than removing it was worth — that is a contradiction to write about, and the write-up
says so rather than resolving it by retraining.

### Result

*Pending.*

## 3. Where the explanation lives

**`explainability.md` §8 left two fallbacks open and the arithmetic closes one of them.**
The options registered there were: explain only the transactions that are not allowed, or
move the explanation off the hot path entirely.

The first fails on numbers already recorded. Scoring one row costs a p95 of about 16 ms
against a 100 ms budget; explaining it costs about 900 ms. The EV policy blocks **6.45%**
of transactions (`decision-policy.md` §4). Explaining only declines therefore puts a
~900 ms call on more than one request in twenty — and a p95 is exactly the percentile
that lands inside a 6.45% tail. The budget would be missed by roughly nine times, on the
decisions that matter most.

**So: `/score` never computes contributions.** The explanation is a separate endpoint,
over `explain/codes.py`, with its own stated budget which is **not** §3.1's — §3.1 prices
an authorisation, and an adverse-action notice is not one.

**This is also the right shape for the obligation.** A customer sees a generic decline at
authorisation; the explanation is owed when it is asked for — by the customer, by an
analyst, or by a regulator — and `codes.py` was already built to answer that question
from a row's contributions rather than to ride along inside the decision.

**Registered.** If the separate path turns out to be slower than measured, the response
is to say so in the latency table, not to move contributions back onto the request.

### Result

**`/explain`, and the separation holds under measurement.** On the shipped model, one
transaction through `/score` costs about **30 ms** and the same transaction through
`/explain` about **960 ms** — single warm calls through the test client, not the §7 load
test, and reported here only as the ratio that decided the design. It is the ratio
`explainability.md` §8 predicted, arriving where it predicted it would.

The endpoint takes the transaction rather than a reference to a stored decision: this
process keeps no store, which is the same reason §6 reports review eligibility rather
than promising a review.

**It does not fail open, and §4's reasoning is what rules it out.** A decision the model
cannot make is better made by the incumbent than not made at all; an explanation it cannot
produce has no substitute, because an invented one is a false statement to a customer. A
breach of the explanation's own budget returns 503 rather than a partial notice. And there
is nothing to explain in the degraded mode: the rules engine never blocks, so a service in
fallback issues no adverse decisions.

The notice reads as §7 of `explainability.md` designed it. On a real transaction the
leading contributor is `C13` — tier 0, and unnameable — so the first line a person would
be read is the honest generic, followed by two sentences that can be said plainly.

## 4. Fail open, to rules

`problem-statement.md` §3.1: if the model is unavailable or exceeds its budget, the
transaction falls back to the Phase 03 rules engine rather than being declined. With a
booster this size that is no longer prudence — it is the design.

**What counts as unavailable**, all three arms fail open:

- an artifact is missing or will not load at startup — the service comes up in degraded
  mode and says so, rather than refusing to start;
- the transform, the booster, the calibrator or the policy raises during a request;
- the request exceeds the budget the `serving` section names.

**What a deadline can and cannot do, stated rather than implied.** A LightGBM prediction
is a blocking call and cannot be interrupted from inside the process. A budget breach
therefore returns the rules decision on time while the scoring work finishes and is
discarded — the request is bounded, the worker is not. Anything else would be a claim
about cancellation that the library does not support, and a service that quietly waited
for the model while reporting a timeout would meet its budget on paper only.

**The fallback path needs the request and its own constants, and nothing else.** No
booster, no calibrator, no fitted table from the feature pipeline — a constraint
`rules-baseline.md` imposed for exactly this reason, so the rules engine survives every
model artifact being absent. It reads `TransactionAmt`, `ProductCD`, `M4` and `D1`; the
last of those is tier 0, which is inbound under §1's contract, and carries the
servability caveat `rules-baseline.md` already recorded against it.

**What the degraded mode costs, and it is not symmetric.** The rules engine never blocks;
it only reviews (`decision-policy.md` §3). A service in fallback therefore allows
everything it does not flag, and fraud detection degrades to the incumbent's quality for
the duration. That is the accepted, bounded risk §3.1 named, and it is the reason the
rules baseline is maintained rather than discarded once the model ships.

**`GET /health` distinguishes live from ready.** The process being up, the artifacts being
loaded, and the mode currently being served are three different facts, and an
orchestrator that cannot tell them apart will either restart a healthy degraded service
or route traffic at one that has nothing loaded.

### Result

**All three arms, and one of them was the wrong shape until a test said so.** A missing
artifact leaves the service up and degraded; a raise anywhere in the scoring path falls
back to the incumbent; a breach of `budget_ms` returns the incumbent's decision on time.

The catch around the scoring path is deliberately broad. Fail-open is worth nothing if it
covers the failures someone anticipated and not the one that happens, and the exception is
logged, so a fallback is never silent in the record even when it is invisible in the
response.

**What the budget actually promises.** A LightGBM prediction is a blocking call into C and
cannot be interrupted, so a breach releases the caller while the work finishes in its own
thread and is discarded. The request is bounded; the worker is not. A test asserts the
abandoned work really did run to completion, because "timeout" reads as "stopped" and it
is not. Under load, part of what a request waits for is a worker to run on, and that wait
counts against the same budget — it should, and it is said so nobody reads a breach as
proof the model itself was slow.

**503 and 500 had to be told apart.** The endpoint was catching `RuntimeError` to answer
"nothing is loaded" with a 503, which meant a scoring path that raised with fail-open
disarmed looked identical to an empty service. They are different answers to a caller: one
is fixed by waiting and the other is not. `NothingLoaded` is now its own type, and only it
becomes a 503.

**`/health` counts the fallbacks**, because a service whose model loaded but answers every
request from the incumbent is indistinguishable from a healthy one at the status code.
`mode` says so per request; the counter says so across them.

## 5. What ships in the image

**The handoff from Phase 07 said five artifacts. Serving needs more than five**, because
the service must *build* the row and not merely score it. The manifest, with what breaks
if each is absent — and every one of these fails quietly rather than loudly, which is why
the image refuses to build without them:

| artifact | tier | what its absence does |
|---|---|---|
| `models/model.txt` | — | nothing to score with |
| `models/categories.parquet` | 2 | codes derived from the request's own levels; a plausible, wrong score |
| `models/medians.parquet` | 2 | nulls where the model was fitted on filled data |
| `models/calibrator.json` | — | uncalibrated probabilities priced as if they were frequencies |
| `models/encoders.parquet` | 2 | the seven `freq_*` columns cannot be built |
| `models/amount_stats.parquet` | 2 | the six `amt_*` columns cannot be built |
| `models/vblock.parquet` | 2 | the 234 `vb_*` survivors and their fill values are unknown |
| rules constants | 1 | no fail-open path |
| `config/cost_matrix.yaml` | — | no break-even, no policy |
| `config/reason_codes.yaml` | — | no sentences for §3's endpoint |

**The rules constants are a new artifact and this phase produces them.** `rules.fit()`
currently computes its per-product cut points and amount ECDF from the training window at
run time, and a container has no `data/`. Persisting them changes no number: the same fit,
written down. The baselines stage's tracked record must come back **identical** when it is
re-run, and that is checked rather than assumed.

**Every artifact is stamped.** `/health` reports the identity of what is loaded — the
booster, the tables, the calibrator, the cost matrix version and the dictionary version —
so a decision can be traced to the objects that produced it without reading the image.

**One thread, several workers.** `explainability.md` §8 measured that a single row gives
LightGBM nothing to parallelise and that more threads made it *slower*, replicated across
two runs. The container therefore pins one thread per worker and scales with processes.
This inverts the usual advice and is a measurement, not a preference.

**The transform has one definition, and serving is a caller of it.** The service imports
`apply_categories`, `apply_medians` and `feature_columns` from the training module rather
than holding its own copy. A second implementation of the vocabulary and the fill values
would be train/serve skew written by hand — precisely what §8's gate exists to catch — so
the duplication is refused even though it would make the import graph tidier.

**Amended, before the fast path existed, and here is what the rule above cost.** Building
one row through the pipeline's own functions was measured at **211 ms against a budget of
100**, of which **163 ms** is three functions doing per-column work on a single row —
`apply_medians` at 86 ms over ~310 columns, the V-block reduction at 48 ms, and
`apply_categories` at 29 ms over 31. The booster's own prediction is 21 ms of the total.
Narrowing the frames does not touch it: measured on a frame holding exactly the columns
each function reads, every one of the three costs the same. The cost is per column, and a
request has one row to amortise it over.

This is the shape `explainability.md` §1 already set a precedent for, when "every row"
became 10,000 rows once TreeSHAP's per-row cost was known: a rule written before its price
was measured, amended once it was, with the amendment registered ahead of the result it
affects.

**What stays refused.** A second definition of *what the values are*. The vocabulary, the
fill values, the surviving V columns and the frequency tables are read from the shipped
artifacts, once, by `artifacts.py`, and nothing else may compute, infer or default them.

**What is permitted.** A second *application* of those same values, in a different data
structure, on the single-row path only: filling with a vector rather than a per-column
`fillna`, and finding a level's code by lookup rather than by re-levelling a Series.

**The condition, which is not a promise to be careful.** The pandas transform remains the
reference and is not deleted. The fast path ships only while §8's gate proves the two
produce the same matrix — same values, same dtypes, same scores — on the same real
transactions, and the proof is re-made every time the gate runs rather than asserted once.
Where they disagree, what is removed is the fast path, never the reference.

**What this costs in guarantees, stated plainly.** §5 no longer says *there is one
definition*; it says *there are two, and one is proven equal to the other*. A structural
guarantee cannot fail quietly and a verified one can — and the proof lives in the tier §8
registers as artifact-gated, which a clean checkout skips. So the guarantee now depends on
someone running the gate where the artifacts are. That is a real reduction, accepted for a
measured 163 ms, and recorded here rather than discovered later.

**What that costs, measured and accepted.** Importing the training module brings MLflow
into the serving process, and reading `config.yaml` brings pandera through `data/load.py`,
which every module in the project reads config through. Both are *startup* cost, roughly a
second between them, and startup is not what §3.1 budgets — loading a ~77 MB booster
dominates it regardless.

**What the image does not install, and what it cannot avoid.** `shap` and `matplotlib` stay
out, and with `shap` goes the largest single block of weight in the environment — its
`numba` and `llvmlite` chain is over a hundred megabytes, more than everything else the
image would drop put together. That exclusion rests on Phase 07's rule that the serving
path imports neither, and the image selects its dependencies accordingly rather than
trusting the rule. **scikit-learn cannot be excluded**: LightGBM imports it itself, so a
serving environment without it has no booster either. Recorded here so it is not
rediscovered by someone trying to remove it.

**The alternative was considered and declined.** Extracting the apply-side functions into a
module free of MLflow would edit the file the frozen booster's stage depends on — marking a
2,990-tree model stale, leaving the guarded headline target permanently unsatisfiable, and
requiring new prerequisites across the build graph — to save MLflow's share of the image
and about a second of startup. The extraction remains available if cold start ever becomes
a real constraint, and it would be done the way any edit near a frozen artifact is: the
artifacts' timestamps restored deliberately, and the reloaded model required to reproduce
its recorded scores exactly before it is trusted again.

### Result

**Ten, not five.** The manifest in the table above is what `stamps()` identifies, and the
rules constants were the artifact nobody had written: `fit()` is the only function in
`rules.py` that reads the training window, and a container has none. It is now produced by
the baselines stage as **11.9 KB of JSON**, carrying the config choices beside the fitted
values so a served rule can never pair current weights with older cut points. Re-running
that stage moved only the record's timestamp and revision — every metric identical, which
is what licensed the rest of the pipeline to be left alone.

The tier-2 tables came to **31 vocabulary columns, 317 fill values, and 228 kept V columns
with 6 presence flags** — read back for the first time by `artifacts.py`, under the guards
§8 asked for.

**The amendment above was taken, and here is what it bought.** With the reference
transform on the request path, one row cost **211 ms**; with the arithmetic assembly it
costs **29 ms**, of which 21 is the booster's own prediction. The fast path is proven
against the reference on 300 real transactions: **0 of 104,700 cells differ**, and no
score differs. The proof runs in both tiers — on the shipped booster where the artifacts
exist, and on the miniature deployment everywhere else — which is narrower than the
structural guarantee it replaced and wider than the amendment promised.

**One defect belongs in this section rather than in §7.** `stamps()` was being computed
per request, which re-read and hashed the 77 MB model file to answer a question that
cannot change while the process lives: **190 ms of the original 590 ms**, and the largest
single item on the bill. Identity is now hashed once, with the artifacts it identifies.

## 6. What the response says, and what it may not promise

**`POST /score` returns the decision, the bar it was taken against, and the state it was
taken in**: `allow` or `block`; the calibrated probability; the break-even probability for
this amount; review eligibility and the expected saving that makes it eligible; whether
the model or the fallback decided; which tier-3 inputs were present; the artifact stamps.

**Review is eligibility, never an outcome.** `codes.py` already refuses to promise a
review and the API keeps that refusal: whether a transaction reaches an analyst depends on
the other transactions that day, which is queue state this service does not hold. The
response carries the eligibility flag and the expected gain so a queue can rank by it; the
queue itself, and the 1% daily capacity it honours, are out of scope for the request path.

**The roadmap's three-action vocabulary is preserved without being faked.** A caller
reading `block` gets a decline; a caller reading `allow` with `review_eligible` gets the
transaction and the information needed to hold it. Returning `review` from a service that
cannot see the day's queue would be a decision the system is not in a position to make.

**No reason codes on this endpoint** (§3).

### Result

**Implemented as registered.** `/score` returns the action, the calibrated probability,
the break-even for this amount, review eligibility with the expected saving that makes it
eligible, which mode answered, which tier-3 inputs arrived, how much of the contract did,
and the identity of the artifacts behind it.

Review stayed eligibility. The queue that would turn it into a review is not in this
process, and the response gives that queue what it needs to rank — the expected saving —
without claiming an analyst will look. In the fallback mode every transaction is a
candidate and the engine's points are the priority, which is `cost.rules_policy` behaving
as it does in the evaluation harness rather than a serving invention.

**Two fields answer questions nothing else could.** `mode` says which path decided, and
`inherited_present` against `inherited_expected` says how much of the record the decision
was made on — §1's guard against an integration that quietly stops sending the block.

One field was written and removed before it shipped: a flag saying an explanation was
available. It would have been true on every response regardless of anything, which is a
field that reads as information and carries none.

## 7. The latency measurement

**This is the §3.1 measurement** that `explainability.md` §8 explicitly was not: p50, p95
and p99 under sustained concurrent load against the running container, with the hardware
attached.

**Registered before the number exists:**

- **What is replayed.** Rows drawn from a scored split, converted into requests. Replay is
  measurement and changes nothing; no policy figure is recomputed here.
- **Concurrency is stated, not chosen afterwards.** The reported table names the
  concurrency, the duration, the worker count and the thread count. A budget met at one
  concurrency and missed at another is reported as both.
- **Warmup is discarded.** A cold first request is not a latency figure — §3.1 says so.
- **The generator shares the machine.** It is co-located with the container, which inflates
  the tail, and the record says so rather than presenting a figure as if it came from a
  load cell on another host.
- **The record travels with its machine.** Like `explain_latency.json`, this is a number
  true of the hardware named inside it and of nothing else.

**Reading rule.** The budget is met or it is missed, and both are reported. A miss is not
repaired by lowering the concurrency until it passes; it is explained — the roadmap's
Definition of Done asks for the budget "met, or missed with an explanation of why", and
the explanation is the deliverable in either case.

### Result

*Pending.*

## 8. What the tests prove, and where

**The gate, before anything else is built.** The service's own transform must reproduce
the batch pipeline **exactly**: for sampled rows, the matrix it builds from a request must
equal `data/features/{split}.parquet`, and the booster's scores on it must equal the
recorded prediction vectors, to the digit. Tier-3 columns are supplied from the matrix for
this test, so the transform path is isolated from §2's default. This is `reproduce.py`'s
rule applied to a new consumer: a model is proven before anything new is attached to it.

**The named risk on that gate: the tier-2 artifacts have never been read.**
`categories.parquet` and `medians.parquet` are written by the training stage and read back
by nothing — every stage needing the vocabulary refits it from `train` through
`prepare_matrices`. Serving is their first reader, and the vocabulary's codes are
**positional**: the level order *is* the contract with the booster, which records the codes
its splits test and never what they stand for. A read that re-sorted the levels, or took
their order from row position, would hand the model correct-looking codes standing for the
wrong levels, and nothing would raise — the scores would simply be wrong.

`write_categories` anticipated this and persists the code beside the level rather than
leaving it implicit in row order; the reader must use that column. The gate's fixture
therefore has to include categoricals whose fitted order is **not** the order a naive read
would produce, and the round trip is asserted on the reconstructed dtype as well as on the
scores. The same applies to the frequency, entity-statistic and V-block tables, which
serving also reads for the first time.

**Two regression tests, as the Definition of Done asks.** A golden-prediction test pinning
known requests to known probabilities and decisions; a PR-AUC regression test on a fixed
sample, failing below a stated threshold.

**And an honest account of where they can run.** `models/` is gitignored and the booster is
~77 MB, so neither test can execute against the shipped artifacts in a clean CI checkout.
The split registered here:

- **Everywhere, including CI** — the contract, the policy wiring, the fallback path, the
  sentinel routing, and the golden-prediction *machinery*, exercised against a small
  booster built inside the fixture. This proves the pipeline is pinned.
- **Where the artifacts exist** — the golden predictions and the PR-AUC threshold for
  *this* model, skipped with a stated reason otherwise. This proves the shipped model is
  pinned.

**Registered.** The Definition of Done item is reported as met with this exception named,
not as met. A test that passes in CI because it silently skipped what it was written to
check is worse than one that says what it did not run.

### Result

*Pending.*

## 9. What this phase may not change

The frozen set is `decision-policy.md` §7's, unchanged: the tuned booster, its vocabulary
and medians, the calibrator, `cost_matrix.yaml` v1 and the policies in `cost.py`. Phase 07
added the reason dictionary. Phase 08 adds a transport and nothing else.

- **No threshold, parameter or feature moves** because of anything measured here — the §2
  neutralisation cost and the §7 latency table included.
- **Test is measurement.** `headline.py` refuses to run twice by design, no stage here
  names `$(HEADLINE)` as a prerequisite, and nothing in this phase recomputes a policy
  figure the headline already reported.
- **A serving default is not a modelling decision.** §2 chooses what the service sends when
  an input is unavailable. It does not choose what the model is.
- **If serving turns out to be awkward, that is a finding.** The write-up says so. The
  model does not move to make the API tidier.
