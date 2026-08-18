# Hypotheses — what will predict fraud

| | |
|---|---|
| **Status** | Open — 2 of 3 recorded |
| **Last updated** | 2026-08-18 |
| **Evidence window** | Days 1–120 (the Phase 01 EDA horizon) |

---

Written **before** any model exists, so that Phase 07 can compare them against what
SHAP actually shows — including where they turn out to be wrong. A hypothesis that
cannot be wrong is worthless for that comparison, so each one below states the
mechanism it assumes, the specific SHAP behaviour it predicts, and what result would
falsify it.

All evidence is measured on days 1–120 only. The tail of the timeline is held back for
the Phase 02 test set and has not been looked at.

---

## H1 — Fraud amounts are compressed into a mid-range band

**Claim.** `TransactionAmt` predicts fraud **non-monotonically**. Risk peaks in the
middle of the amount distribution and falls away at both extremes, rather than rising
with transaction value.

**Evidence.** Fraud amounts sit above legitimate ones from roughly the 30th to the 98th
percentile, and *below* them outside that band in both directions:

| percentile | legit | fraud |
|---|---:|---:|
| 1% | 9.92 | 7.37 |
| 25% | 43.95 | 34.92 |
| 50% | 68.95 | **76.02** |
| 75% | 125.00 | **171.00** |
| 97.5% | 640.95 | **744.95** |
| 99% | 1104.00 | 994.00 |
| 99.9% | 2754.07 | 2282.05 |

Fraud is also *less* variable than legitimate activity (std 217.5 vs 239.1) despite a
higher mean ($146.15 vs $134.28, +8.8%, t = 6.5) and a higher median (+10.3%). The
shape is a compression toward the middle, not a shift upward.

**Mechanism.** Two opposing pressures. Large amounts attract manual review, step-up
authentication and issuer scrutiny, so they convert poorly for an attacker. Very small
amounts do not repay the cost of obtaining and burning a compromised instrument. What
survives is a band that is large enough to be worth taking and small enough to pass.

**Prediction for Phase 07.** The SHAP curve for `TransactionAmt` is hump-shaped:
positive contribution through the mid range, negative at both the low and high ends.

**What would falsify it.**

1. A monotone increasing SHAP relationship — risk simply rising with amount.
2. The band disappearing once `ProductCD` is controlled for. Different product
   categories carry different price points *and* different fraud rates, so the
   compression could be a product-mix artifact rather than attacker behaviour. This is
   the alternative explanation to beat, and it is checkable in Phase 04.

**Caveat.** The effect is statistically unambiguous but economically modest — a ~9%
difference in means. Amount alone is a weak separator, and this hypothesis is about the
*shape* of its contribution, not its strength.

---

## H2 — Round amounts are over-represented, but only between $150 and $500

**Claim.** Fraud clusters on round `TransactionAmt` values that are multiples of $50 in
the **$150–$500** range. This is *not* a general preference for round numbers: $100 runs
the other way, and the effect vanishes above $500.

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
2. `ProductCD` accounting for it. If $300 is simply a price point in a fraud-heavy
   category, this is product mix, not attacker choice. **Same confound as H1**, and the
   two hypotheses likely stand or fall together — they may be one mechanism observed
   twice, once as distribution shape and once as exact values.
3. The ratios not holding outside days 1–120.

**Caveat.** The largest ratio ($450 at 12.94×) rests on 77 fraud transactions. The band
as a whole is well-supported; individual multipliers at the sparse end are not.

---

## H3 — (to be written)

> Phase 01's Definition of Done requires three. Candidates still to be grounded in
> evidence from EDA sections 5–7: missingness structure (whether *which* fields are
> absent predicts fraud independently of their values), entity concentration in the
> `card`/`addr` columns, and the identity-record join itself — `has_identity` was built
> as an explicit flag precisely because its absence is expected to carry signal.
