# Hypotheses — what will predict fraud

| | |
|---|---|
| **Status** | 3 of 3 judged against SHAP in Phase 07 — verdicts under each |
| **Last updated** | 2026-09-17 |
| **Evidence window** | Days 1–120 — `eda.horizon_day` in `config/config.yaml` |

---

Written **before** any model exists, so that Phase 07 can compare them against what
SHAP actually shows — including where they turn out to be wrong. A hypothesis that
cannot be wrong is worthless for that comparison, so each one below states the
mechanism it assumes, the specific SHAP behaviour it predicts, and what result would
falsify it.

All evidence is measured on days 1–120 only. The tail of the timeline is held back for
the Phase 02 test set and has not been looked at.

Because these rows informed every hypothesis below, they are spent: Phase 02 must place
VAL-FIT, VAL-CAL and TEST entirely after day 120. That constraint and the day budget it
leaves are recorded in `problem-statement.md` §7 item 5.

---

## H1 — Fraud amounts run higher within product, and the pooled "compression" is a mix artifact

> **Status: revised in Phase 01, before any model existed.** The original form of this
> hypothesis claimed a *non-monotonic* relationship — fraud concentrated in a mid-range
> band, sitting below legitimate amounts at both extremes. Its own stated falsification
> criterion ("the band disappearing once `ProductCD` is controlled for") fired during the
> section 5 confound check. What follows replaces it.

**Claim.** Within any given `ProductCD`, fraud amounts sit **above** legitimate ones
across almost the whole distribution, with the gap closing only at the extreme top. The
pooled data's apparent low-end reversal is Simpson's paradox, not attacker behaviour.

**Evidence.** Fraud amount minus legitimate amount, in USD, at each percentile within
each product:

| | C | H | R | S | W |
|---|---:|---:|---:|---:|---:|
| p10 | +1.0 | +25.0 | +50.0 | 0.0 | +8.0 |
| p25 | +1.2 | +40.0 | +50.0 | 0.0 | +10.0 |
| p50 | +1.8 | +100.0 | +75.0 | +15.0 | +38.5 |
| p75 | +7.2 | +100.0 | +100.0 | +50.0 | +115.0 |
| p90 | +4.1 | +180.0 | +100.0 | +100.0 | +182.5 |
| p99 | +31.1 | +180.0 | 0.0 | −2.4 | 0.0 |

Pooled, fraud sits *below* legitimate amounts at p10 and p25. Within every product it
sits at or above. The reversal comes from fraud concentrating in product C — 12.1% of
rows but 38.5% of frauds — whose median amount is $32.02 against product W's $80.00.
Mixing a small-ticket, fraud-heavy product with a large-ticket, fraud-light one drags the
pooled low percentiles the wrong way.

**Mechanism.** Nothing exotic: within a product line, an attacker extracts more value per
transaction than a typical customer. The gap closing at p99 is the ceiling effect that
large transactions attract scrutiny.

**Prediction for Phase 07.** `TransactionAmt`'s SHAP contribution rises with amount
*conditional on* `ProductCD`, flattening at the top rather than turning over. Interaction
between the two should be visible; a model given amount without product should look
noticeably worse on this axis.

**What would falsify it.**

1. A genuinely hump-shaped contribution *within* a single product — that would revive the
   original band hypothesis at a level where mix cannot explain it.
2. Amount contributing nothing once product is known.

**Lesson recorded.** The original H1 was built on pooled percentiles and survived a
significance test (t = 6.5) while being structurally wrong. Statistical strength is not
protection against a mix artifact; the only cure was conditioning on the confound the
hypothesis itself had named.

### Verdict — Phase 07: falsified, on the product that carries the traffic

Measured on test, from the persisted contributions:
`reports/figures/shap_amount_dependence.png`.

Within `ProductCD` W — 72% of transactions and 64% of fraud dollars — the amount's
contribution rises with amount to about +0.44 near $300 and then **turns sharply
negative**, reaching −0.56 above $475. That is falsification criterion 1 as written: a
hump inside a single product, where mix cannot explain it. It is not a binning artifact —
it sharpens at twenty bins, and the reversing band holds 385 transactions across 232
distinct `card1` values and all 22 days of test. C is noisy with no clean rise; H rises
throughout; R and S rise and do flatten at the top, which is what this hypothesis
predicted and what W does not do.

**The uncomfortable part.** The band where the amount's contribution goes negative is the
band with the *highest* observed fraud rate in W — about 6%, against roughly 2% below it.
The data says large W transactions are riskier; the model's use of the amount says the
opposite, on the margin.

Both can be true at once, and the resolution is the one §5 records. A SHAP value is
marginal: it is what the amount moved *given* `C13`, `C1`, `D1` and the rest. If those
already carry the risk of a large W transaction, the amount is left correcting for what
they overstate. What is falsified is this hypothesis's prediction about the
contribution. Whether the claim about amounts and fraud is wrong is a different question,
and the fraud rates above suggest it is not.

**Recorded, not repaired.** The frozen model is not revisited because its use of a
feature is surprising — `explainability.md` §9. This is a caveat on the Phase 05 model
and a question for anyone who retrains it.

## H2 — Round amounts are over-represented, but only between $150 and $500

**Claim.** Fraud clusters on round `TransactionAmt` values that are multiples of $50 in
the **$150–$500** range, concentrated in **`ProductCD` H**. This is *not* a general
preference for round numbers: $100 runs the other way, and the effect vanishes above $500.

Unlike H1 and H3, this hypothesis **survived the `ProductCD` control and grew stronger** —
the pooled figures were diluted, not inflated.

**Evidence.** Every round value scanned, not only those already common among frauds, so
the comparison is not selected on the outcome. Ratio is fraud share ÷ legit share:

| amount | fraud n | ratio | | amount | fraud n | ratio |
|---|---:|---:|---|---|---:|---:|
| $100 | 400 | **0.68×** | | $150 | 471 | 2.08× |
| $200 | 342 | 1.94× | | $250 | 179 | 1.89× |
| $300 | 297 | **4.74×** | | $350 | 35 | 4.75× |
| $400 | 59 | 3.17× | | $450 | 77 | **12.94×** |
| $500 | 92 | 2.38× | | $550+ | ≤1 | no effect |
| $600–$1100 | ≤8 | 0.41–1.01× | | | | |

Two features of this table matter as much as the peaks. **$100 is under-represented**
(0.68×) despite being the single most common round amount in the data — 4.01% of all
transactions. And **everything above $500 sits at parity**. A general round-number
preference would produce neither.

**Mechanism.** Round values inside the band H1 already identifies. Two candidates: goods
sold in fixed denominations (gift cards and stored-value products are the obvious case,
and they are attractive because they liquidate cleanly), or an attacker typing a chosen
amount rather than buying a specific priced item. The $100 result argues against
"round numbers are suspicious" as such — whatever drives this is specific to the band.

**Prediction for Phase 07.** `TransactionAmt`'s SHAP contribution is *spiky* rather than
smooth — visible steps at $150, $300, $450 rather than a continuous curve. Equivalently,
an engineered feature for "multiple of $50 **and** within $150–$500" earns importance
while a plain "is a round number" feature does not.

**What would falsify it.**

1. A plain `is_round_amount` feature carrying the signal on its own — that would make the
   effect general and the band incidental.
2. ~~`ProductCD` accounting for it.~~ **Tested in Phase 01 and rejected as an
   explanation.** Controlling for product amplifies the effect rather than dissolving it:
   $300 goes from 4.74× pooled to **14.58× within product H** (n=180), $450 from 12.94× to
   **24.20×** (n=71), $150 from 2.08× to 4.98× (n=332). The round amounts live in products
   R (12,563) and H (3,055), with essentially none in C — so this is a different
   population from H3 entirely.
3. The ratios not holding outside days 1–120.

**Caveat.** The largest ratio ($450 at 12.94×) rests on 77 fraud transactions. The band
as a whole is well-supported; individual multipliers at the sparse end are not.

---

### Verdict — Phase 07: the mechanism survives, the feature built for it did not

**Falsification criterion 3 was the real test, and it was passed.** The ratios were
re-measured on test — days 161–182, sixty days past the evidence window, never looked at
when this was written:

| amount | ratio, test | in `ProductCD` H |
|---|---:|---:|
| $100 | **0.85** | 1.03 |
| $150 | 1.49 | 2.83 |
| $250 | 1.93 | — |
| $300 | 7.44 | 9.60 |
| $450 | **57.30** | 67.89 |
| $500 | 3.47 | 1.26 |

Across the whole band, round amounts between $150 and $500 carry an 8.0% fraud rate
against 3.1% for the rest of the band. `$100` is still under-represented, which is the
observation that made this hypothesis specific rather than a superstition about round
numbers, and it is the one that would have been easiest to lose out of sample.

**The Phase 07 prediction is another matter.** It said the engineered band feature would
earn importance. `amt_round_band` ranks #163 of 349, and it and
`amt_round_band_product` together carry about a sixth of what the raw `TransactionAmt`
carries. The second form of the prediction — a spiky rather than smooth contribution —
has support: the amount's median contribution sits above its immediate neighbourhood at
$300, $400, $450 and $500, on counts small enough to be worth naming.

**What this settles.** The signal is real and it generalises. The column built to hand it
to a linear model was redundant to a tree, which can carve the same bands out of the raw
amount by itself. That is a lesson about feature engineering for boosted trees, not
about fraud, and it is the same lesson E4 recorded from the other direction.

## H3 — `ProductCD` is among the strongest single predictors, and the one others proxy for

**Claim.** `ProductCD` separates fraud risk as sharply as any other single field available
at scoring time, and it is the variable other candidate signals turn out to be proxies for.

> **Softened after the section 6 test.** The original said *more sharply than any other*.
> At fixed volume, `addr` (3.20×), `ProductCD` (3.18×) and `card2` (3.13×) are a
> three-way tie, so uniqueness is not supported. What survives is that nothing beats it,
> and that it costs nothing to obtain.

**Evidence.** Across the horizon:

| ProductCD | rows | share of rows | fraud rate | share of all frauds | share of fraud USD | median amount |
|---|---:|---:|---:|---:|---:|---:|
| C | 50,136 | 12.1% | **11.20%** | 38.5% | 12.5% | $32.02 |
| S | 7,214 | 1.7% | 6.04% | 3.0% | 1.5% | $30.00 |
| H | 28,134 | 6.8% | 4.52% | 8.7% | 9.6% | $50.00 |
| R | 30,185 | 7.3% | 3.52% | 7.3% | 12.0% | $125.00 |
| W | 298,873 | 72.1% | **2.08%** | 42.6% | 64.3% | $80.00 |

A **5.4× spread** between C and W, on populations of 50,136 and 298,873 — not a small-cell
effect. C carries 12.1% of transactions and 38.5% of frauds.

**It also explains other candidates.** A missing billing address looked like a 4.48× signal
until conditioned: 94.2% of product C rows have no address, and within C the residual lift
is only 1.30×. Elsewhere there are 47–153 null-address rows per product, too few to judge.
The address signal is largely a restatement of "this is product C".

**Mechanism.** `ProductCD` distinguishes what is being bought. Products differ in how
liquid they are to an attacker, whether they ship physically, whether delivery can be
intercepted, and how fast they convert to value. C behaves like a fast-converting,
low-ticket, address-free product; W behaves like conventional retail.

**Prediction for Phase 07.** `ProductCD` ranks among the top few SHAP features, and
several features that look independently important — address presence in particular —
lose most of their contribution once it is in the model.

**What would falsify it.**

1. ~~Another single field separating risk more sharply.~~ **Tested in section 6.** Raw
   fraud-rate spread across well-supported levels appears to falsify it outright — `card2`
   at 57.2× against `ProductCD`'s 5.4×. But spread rewards cardinality: `card1` has 12,251
   levels and its 46× comes from many small cells, not from separating bulk traffic. At
   **fixed volume** — the riskiest levels covering ~10% of transactions, the same shape as
   the capacity constraint — the ordering reverses:

   | field | levels used | volume | fraud captured | lift |
   |---|---:|---:|---:|---:|
   | addr1 / addr2 | 1 | 11.5% | 36.8% | 3.20× |
   | `ProductCD` | 1 | 12.1% | 38.5% | 3.18× |
   | card2 | 12 | 10.2% | 32.0% | 3.13× |
   | card5 | 4 | 11.2% | 27.1% | 2.43× |
   | card1 | 19 | 10.3% | 21.9% | 2.14× |

   The ranking bias runs *toward* the contenders — levels were ordered by fraud rate on the
   same data the capture is measured on — and they still do not win. `card2` is the one
   genuinely separate contender, and it is not free: 500 levels means target encoding
   fitted inside CV folds, which Phase 04 owns.

2. Address presence retaining a large contribution *alongside* product. **Largely settled
   already**: `addr1` and `addr2` share an identical null mask, and **99.2% of
   address-null rows are `ProductCD` C**. Its 3.20× is this hypothesis's own result under
   a different name. SHAP can still confirm the two do not separate.
3. The ordering not holding outside days 1–120.

**Note on the cost model.** Risk ranking and money ranking disagree here. C has the
highest fraud rate but only 12.5% of fraud USD, because its median amount is $32. W has
the lowest rate and 64.3% of fraud USD. A policy tuned on rate alone would chase C and
miss where the money is — which is the same lesson section 4's recall-versus-USD table
recorded.

**Servability.** Free. `ProductCD` arrives with the request; no entity history required.

---


### Verdict — Phase 07: not supported, and not cleanly falsified either

`ProductCD` ranks **#70 of 349** by mean absolute contribution on test. The fields this
hypothesis argued were proxies for it rank *above* it: `addr1` at #18, `P_emaildomain` at
#15. Falsification criterion 2 — address presence retaining a large contribution
alongside product — fires as written.

`addr2` sits at #309, near zero, which is consistent with `addr1` carrying geography
rather than the shared null mask this hypothesis was about.

**Why this is not a clean falsification.** The evidence in section 6 above was lift at
fixed volume with nothing else in the model. A contribution is marginal use given 348
other columns, and `C13`, `C1`, `C14` and `D1` — the four above everything — are
undocumented counters and day-deltas that could plausibly encode what product separates.
A field can contribute nothing because it is irrelevant or because it is redundant, and
nothing measured here tells those apart. Settling it would mean dropping the column and
refitting; `explainability.md` §9 forbids that, and an explanation is not worth a
retrain.

**Criterion 3, measured: the ordering holds.** The fixed-volume comparison was re-run on
test, both the way section 6 ran it — levels picked on the same rows the capture is
measured on — and the operational way, levels picked on train and applied forward:

| field | levels | volume | fraud captured | lift | (train-selected) lift |
|---|---:|---:|---:|---:|---:|
| `card2` | 20 | 10.2% | 42.6% | **4.16×** | 4.08× |
| `addr1` | 1 | 11.0% | 42.7% | 3.87× | 3.79× |
| `ProductCD` | 1 | 11.1% | **42.8%** | 3.86× | 3.86× |
| `card1` | 40 | 10.3% | 36.1% | 3.52× | 2.91× |
| `card5` | 6 | 19.6% | 37.1% | 1.90× | 2.09× |

The same three fields lead, sixty days later, and `card5` and `card1` still trail. The
lifts are all higher than in section 6, which is what a higher base rate does to this
statistic.

**`card2`'s edge is the denominator, not more fraud.** It captures 42.6% where
`ProductCD` captures 42.8%; the lift favours it because it does so at nine tenths of a
point less volume. Selected on train and applied forward — the only version an operator
could run — `ProductCD` captures the most fraud of any single field here, and does it
with **one level against forty-three**.

**What survives, and what does not.** The claim about the world is in better shape than
when it was softened: at the capacity this project operates under, nothing beats
`ProductCD` on fraud captured, and it still costs nothing to obtain. The claim about
*this model* does not survive — the field ranks #70 and the model barely uses it.

That is the sharpest instance in the project of a distinction that runs through all of
Phase 07: a field can separate risk as well as anything available and still contribute
almost nothing to a model that has three hundred other ways to know the same thing.

> **On `has_identity`.** Fraud is 7.34% where an identity record exists against 2.14%
> where it does not — strongly predictive, and in the opposite direction to intuition.
> It is deliberately *not* a hypothesis: its predictiveness is already measured rather
> than predicted, and its mechanism is not knowable from this data. If identity is
> collected because something already looked wrong, the flag encodes an upstream fraud
> decision. That belongs in `problem-statement.md` §6 as an assumption, alongside A6 on
> `C*`/`D*` provenance.

> **On `D8` and `D9` — a structural finding, not a hypothesis.** The `D*` family is
> documented as day-deltas, and two members do not behave like the other thirteen: `D8`
> is the only non-integer-valued one (10,896 distinct values, median 41.62), and `D9`
> holds just 24 values inside [0, 1]. They are not two features. They are one
> measurement stored twice.
>
> Measured on days 1–120, like everything else here, though the relationship is
> arithmetic rather than statistical and does not depend on the window:
>
> - `D8` and `D9` share an **identical null mask** — both 86.4%, the same rows.
> - `D9` equals the **fractional part of `D8`** to within float32 storage noise
>   (maximum deviation 4.1e-5).
> - `round(D9 × 24)` equals the derived `hour` column on **100.00%** of non-null rows.
>
> So `D8` is a day-delta carried at sub-day resolution — `floor(D8)` is integer-valued
> and spans 0–1707 days — and `D9` is nothing but its time-of-day component, rescaled
> to a fraction of a day.
>
> **Consequence for Phase 04.** `D9` carries no information beyond the `hour` column
> already derived from `TransactionDT`; it should be dropped as a known duplicate rather
> than rediscovered by correlation clustering. `D8` should be split into `floor(D8)`,
> the actual timedelta, and its fraction — which is `hour` a third time. Used raw it
> mixes two unrelated quantities on one axis, and a linear model reading it assumes an
> extra hour of day and an extra day of age move the log-odds by the same amount.
> Neither column currently reaches a model: at 86.4% null both fall outside
> `d_max_null_frac: 0.50`, so nothing downstream is wrong today.
>
> Knowing the units does not recover the definition. A6 stands — the lookback windows
> are undocumented, and this says nothing about *which* prior event `D8` measures.
