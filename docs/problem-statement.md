# Problem statement — transaction fraud decisioning

| | |
|---|---|
| **Status** | Draft |
| **Last updated** | 2026-08-18 |
| **Supersedes** | — |

---

## 1. Context

This system scores card-not-present e-commerce transactions inline with payment
authorisation and returns an action for each one. It is the automated first line
of a fraud operation: most transactions are decided without a human, and a small
minority are routed to analysts.

The data is the public **IEEE-CIS Fraud Detection** dataset (Vesta Corporation),
standing in for a production transaction stream: 590,540 transactions over
roughly 182 days, of which 20,663 are labelled fraud — a base rate of **3.50%**.
Amounts are **USD**; no currency conversion is applied anywhere in this project.

**Out of scope.** This system does not perform chargeback recovery or
representment, identity verification or KYC, account-takeover detection, or
merchant-side fraud detection. It decides individual payment transactions and
nothing else.

**A standing limitation.** This is a US e-commerce portfolio. The *method* here
transfers to any card-not-present book, but the specific decision rates it
produces depend on the ticket-size distribution, so a portfolio with materially
different amounts would need the policy recalibrated. See §6.

## 2. The decision

For each transaction, the system emits exactly one action:

| Action | What happens |
|---|---|
| **Allow** | Authorised and captured normally. The customer sees nothing. |
| **Review** | Authorised but **not captured**; fulfilment is held pending an analyst decision, targeted within 4 business hours. The customer sees the order as "processing". The analyst's decision resolves it to capture or void. |
| **Block** | Declined at authorisation. The customer sees a generic decline and may retry with another instrument. |

**Review holds the transaction.** This is a deliberate choice and it drives the
cost model in §4. The alternative — let the transaction through and inspect it
afterwards — means a reviewed fraud still costs the full amount, and review
becomes a detection tool rather than a prevention one. Holding means an analyst
decision can still prevent the loss, at the price of delay for legitimate
customers caught in the queue.

Consequences of that choice, carried through the rest of this document:

- A correctly-reviewed fraud costs only the review, not the transaction amount.
- A wrongly-reviewed good customer costs the review plus a delay-friction
  penalty — smaller than a block, but not zero.
- Review capacity is a hard operational constraint, not a soft preference,
  because held orders age.

**What the system does not decide.** It does not set credit or spend limits,
close or restrict accounts, handle disputes once filed, or choose whether to
represent a chargeback. Those are downstream decisions by other teams.

## 3. Constraints

### 3.1 Latency

Scoring is inline with authorisation, so the budget is **p95 < 100 ms** measured
under sustained concurrent load, not a single warm call. Reported with p50/p99
and the hardware, since a latency figure without a machine attached is
meaningless.

**Failure mode: fail open, to rules.** If the model is unavailable or exceeds
its budget, the transaction falls back to the Phase 3 rules engine rather than
being declined. Declining all traffic during a model outage converts a
availability incident into a total revenue outage, which is almost always the
worse failure. The cost is that fraud detection degrades to baseline quality for
the duration — an accepted, bounded risk, and the reason the rules baseline is
maintained rather than discarded once the model ships.

### 3.2 Review capacity

**1.0% of daily transaction volume**, measured against daily volume rather than
against the evaluation set.

Sanity check on what that means in people: 590,540 transactions over 182 days is
roughly **3,245 transactions/day**, so 1% is about **32 reviews/day**. At a
sustained rate of 20–30 manual reviews per analyst-hour, that is one analyst for
roughly 1–2 hours a day — consistent with a small merchant operation, not a
large fraud floor.

Stating it this way matters: a capacity constraint with no headcount behind it
can't be checked, and 1% of a much larger portfolio would imply a team that
doesn't exist in this scenario.

### 3.3 Label availability

**Assumed label-maturity window: 30 days.**

Fraud is confirmed when a cardholder disputes a transaction and a chargeback is
filed. Card network rules give cardholders a long window to dispute — commonly
120 days from the transaction date, and longer for some reason codes — but the
majority of fraud chargebacks arrive well inside that.

This creates a genuine tension, and 30 days is where it is resolved:

- Waiting for **full** maturity (~120 days) means the training data is always
  four months stale, on a problem where attack patterns shift week to week.
- Waiting **30 days** captures the bulk of fraud chargebacks and keeps the model
  current, at the cost of some fraud in the training window still being labelled
  legitimate — label noise that biases the model slightly toward under-calling.

30 days is a judgement, not a fact. It sets the purged gap between training and
validation in Phase 2, and it is one of the assumptions listed in §6.

It is also the project's most defensible decision with no number behind it, so
Phase 05 measures what it costs: `experiments.md` E1 runs the same harness with
the gap removed and reports the difference in PR-AUC and USD. The purged split
ships either way — the point is to state the price rather than assert the
principle.

### 3.4 Serving

A single scoring request carries the transaction and its immediate attributes.
It does **not** carry the card's history, so velocity and entity-aggregate
features cannot be computed from the request alone — they require an online
store maintained in real time.

**Position taken:** build both feature sets and quantify the gap. The model
shipped behind the API uses only request-computable features; the
history-dependent model is trained and evaluated alongside it, and the
difference in PR-AUC and USD saved is reported as the measured cost of not
building a feature store.

This is deliberate. Quantifying what the infrastructure would buy is a more
useful result than either silently training on unservable features or quietly
dropping them.

Registered as `experiments.md` E3.

## 4. Cost model

**Unit of account: USD**, throughout, no conversion. The machine-readable
version of these figures lives in `config/cost_matrix.yaml`; this section
carries the reasoning that file can only gesture at.

### 4.1 The matrix

| Outcome | Cost | Basis |
|---|---|---|
| **False negative** — fraud allowed | `amount + $25` | Full loss of goods and funds, plus the chargeback administration fee. **Scales with amount.** |
| **False positive** — good customer blocked | `$15` | Lost contribution margin plus an allowance for churn. **Approximately fixed.** |
| **Review** — any transaction routed to an analyst | `$1.50` | Analyst time. |
| **Review friction** — good customer delayed by review | `$3` | Partial-severity version of the block penalty. |
| **True positive** — fraud blocked or caught in review | `$0` | Loss prevented. |
| **True negative** — good transaction allowed | `$0` | The intended outcome. |

### 4.2 Where each number comes from

**Chargeback fee — $25.** Acquirers and card networks charge merchants a
per-dispute administration fee, typically quoted in the $15–$100 range depending
on acquirer, merchant category and risk profile. $25 sits at the low-middle,
appropriate for a merchant not yet in a network monitoring programme. *Source:
industry-typical range, not a measured figure for this portfolio.*

Note this is the *administration* fee only. Merchants above roughly 0.9% dispute
rate enter network monitoring programmes carrying additional fines; that
non-linear penalty is **not** modelled here and is listed in §7.

**False-positive cost — $15.** The weakest number in this document, and
deliberately flagged as such. Two components:

- *Lost margin* on the declined transaction. On a mean ticket in the low
  hundreds at typical e-commerce contribution margins, single-digit dollars.
- *Churn risk.* A wrongly declined customer may not come back. Widely cited
  industry surveys put the share of falsely-declined customers who reduce or
  abandon their use of a merchant at roughly a third — but that is survey
  self-report, not observed behaviour, and it is not measured for this portfolio.

$15 is a defensible round number, **not** a measured one. Phase 6 sweeps it from
$5 to $100 and reports how the policy and the savings respond. That sweep is the
point: the goal is not to be right about this number, it is to characterise how
much the conclusion depends on it.

**Review cost — $1.50.** Derived rather than assumed: at 20–30 reviews per
analyst-hour and a fully-loaded analyst cost of $30–$45/hour, the marginal cost
of one review lands at roughly $1.00–$2.25. $1.50 is the midpoint.

**Review friction — $3.** A held order delays a legitimate customer without
declining them, so the penalty is real but well below a block. Set at 20% of the
false-positive cost. Sensitivity-tested with it.

### 4.3 The asymmetry that drives the design

The false-negative cost **scales with the transaction amount**; the
false-positive cost is **approximately fixed**. Therefore the break-even
probability is not a constant:

```
block when   p(fraud) × (amount + chargeback_fee)  >  cost_of_false_positive

           ⇒   p* = cost_of_false_positive / (amount + chargeback_fee)
```

A $20 transaction and an $8,000 transaction should not face the same bar. This
is why Phase 6 derives a per-transaction threshold rather than tuning a single
global cutoff on F1, and it follows directly from the two rows above.

*Simplification noted:* treating the false-positive cost as purely fixed is an
approximation — lost margin does scale with amount. Modelling it as
`fixed + margin × amount` puts a floor under `p*` at large amounts. Deferred to
§7 rather than complicating the first version.

*To verify in Phase 01:* the actual `TransactionAmt` distribution — mean,
median, and tail. Every figure above is calibrated against an assumed
low-hundreds mean ticket, and if the real distribution differs materially, these
numbers need revisiting before Phase 6.

## 5. Success criteria

Committed before knowing whether they are achievable.

**Headline.** Beat the Phase 3 rules baseline on **USD lost per 1,000
transactions**, at equal or lower review volume. A reduction of **≥ 15%** counts
as a real win; below 5% is within the noise of the cost assumptions and should
not be claimed as one.

**Model metrics**, on validation only:

- **PR-AUC** — primary. Must beat both the rules engine and logistic regression.
- **Recall @ 1% capacity** — primary. The operational question: of all fraud,
  what share is caught within the review budget?
- **ROC-AUC** — secondary, reported for comparability.
- **Accuracy** — never reported. 96.5% is available by predicting "not fraud".

**Calibration.** The model must be calibrated well enough that the expected-value
policy is sound: reliability diagram plus Brier score, before and after.

**Latency.** The §3.1 budget met, or missed with an explanation.

**What failure looks like.** Any of the following means this did not work, and
gets reported as such rather than reframed:

- The model does not beat the rules baseline in USD at equal review volume.
- The USD advantage disappears within the §4 sensitivity sweep — i.e. the result
  is an artefact of the friction-cost guess.
- Calibration is poor enough that the EV policy underperforms a naive threshold.
- p95 latency misses the budget by enough to make inline scoring impractical.

## 6. Assumptions

Each of these could be wrong. They are the source material for the limitations
section in Phase 9.

A6 and A8 are questions about how the vendor built the data rather than about the world,
so neither can be settled from the data alone. A8 was narrowed in Phase 01: most of what
looked like a risk signal turned out to be the sales channel, leaving a smaller and more
specific concern. Both are recorded so the limitation is deliberate rather than discovered
late.

| # | Assumption | If wrong | How it would surface |
|---|---|---|---|
| A1 | 30-day label maturity captures most fraud chargebacks | Training labels understate fraud; model under-calls | Compare fraud rate in the most recent 30 days of training against earlier windows |
| A2 | False-positive cost ≈ $15, fixed | The optimal threshold shifts; savings change | Phase 6 sensitivity sweep, $5–$100 |
| A3 | Chargeback fee ≈ $25 | Minor — it is additive to a much larger amount term | Sweep alongside A2 |
| A4 | Review capacity is 1% of daily volume | Policy is infeasible or leaves capacity unused | Recall@capacity reported at several capacities |
| A5 | A US e-commerce ticket distribution stands in for a production portfolio | Decision *rates* don't transfer, though the method does | Stated, not testable with this data |
| A6 | `C*` and `D*` columns are safe to use | Vendor-computed aggregates over undisclosed lookback windows may leak future information past the purge gap | Not testable — the windows are not published |
| A7 | Fraud patterns are stable enough for a model trained on days 1–120 to hold. `day` is 1-indexed in this data — `min(TransactionDT)` is 86,400, so there is no day 0 — and 120 is `eda.horizon_day` in `config/config.yaml`, the last day Phase 01 was allowed to look at | Performance decays faster than expected | Phase 9 month-over-month PR-AUC |
| A8 | The residual signal in `has_identity`, once `ProductCD` is held fixed, describes the transaction rather than an upstream fraud decision | That residual encodes another system's judgment; it would not reproduce anywhere identity is collected by a different rule, and Phase 7 reason codes built on it would be unexplainable. Scope is limited: the flag agrees with `ProductCD != W` on 98.7% of rows, so most of its pooled 3.42× lift is channel, not risk. The within-product residual is 1.47×–2.08× | Collection rule unpublished, so not directly testable. Check whether the within-product residual is stable across time, and whether it survives alongside `ProductCD` in Phase 7 SHAP |

## 7. Open questions

1. **Non-linear chargeback penalties.** Network monitoring programmes impose
   fines above a dispute-rate threshold, which makes the true cost of false
   negatives convex rather than linear. Not modelled; would change the policy at
   the margin if the portfolio ran near the threshold.
2. **Amount-scaled false-positive cost.** Whether to model FP cost as
   `fixed + margin × amount` rather than purely fixed. Puts a floor under `p*`
   at large amounts. Revisit if the Phase 6 sweep shows the policy is unstable
   at the top of the amount distribution.
3. **Review outcome quality.** The model assumes analysts decide correctly.
   Real review queues have their own error rate, which would blunt the value of
   routing.
4. **Repeat-offender dynamics.** A blocked fraudster may retry with a different
   card or device. Per-transaction costing ignores this.
5. **The day budget Phase 02 inherits.** Phase 01's EDA horizon
   (`eda.horizon_day` = 120) is now spent: those rows informed every hypothesis
   in `hypotheses.md`, so VAL-FIT, VAL-CAL and TEST must all start after day 120.
   That leaves **days 121–182 — 62 days, 175,998 rows, 6,063 frauds** for the
   purge gap plus all three slices. A 30-day gap consistent with A1 consumes
   half of it, leaving 32 days and 92,427 rows to split three ways.

   That is tight, and the ROADMAP warns about exactly this: a ~11-day TEST
   cannot produce a month-over-month decay chart. The trade-off is real —
   a shorter gap buys test span but weakens the label-maturity argument that is
   the most defensible thing in the design, so shortening it to buy a prettier
   Phase 9 chart is the wrong trade.

   **The constraint is narrower than it first appears.** `test_transaction.csv`
   spans **days 213–395** — 183 unlabelled days beginning 30 days after the
   labelled data ends. PSI over feature distributions and prediction-distribution
   drift need no labels at all, so the bulk of Phase 09 monitoring can run on
   that horizon regardless of how small TEST is. Only the *labelled* PR-AUC decay
   chart is bounded by the 32-day squeeze.

   **Decide in Phase 02, not Phase 09:** either accept a short labelled decay
   window and state that monitoring runs on the unlabelled horizon, or shrink
   train. Do not shrink the purge gap.
