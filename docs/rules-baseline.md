# The rules baseline — the incumbent

| | |
|---|---|
| **Status** | 5 rules settled, weights fixed. Phase 03. |
| **Last updated** | 2026-08-24 |
| **Evidence window** | Days 1–90 — TRAIN only. Validation and test untouched. |
| **Stability check** | Days 1–45 (half A) vs 46–90 (half B), inside train |

---

This is the thing the model has to beat. It is also the fail-open path: if the model
is unavailable or exceeds its latency budget, this engine decides
(`problem-statement.md` §2). Both roles demand the same discipline — every rule must
be servable from a single request, and every constant must be fitted on train alone.

## Provenance

Each rule carries a `provenance` field, because two different processes produced them
and they carry different risks:

| value | meaning | risk |
|---|---|---|
| `hypothesis` | Formed in Phase 01 EDA and written into `hypotheses.md` *before* any modelling | Low — stated in advance, falsifiable, already survived a `ProductCD` control |
| `search` | Found by a directed scan over a column family on the training window during Phase 03 | **Selection bias** — roughly twenty flag/value combinations were scanned and the best kept, so the full-train lift is optimistic |

The half-split stability check exists specifically to price the `search` rules. It
changed a decision: R5's lift fell from 2.25× to 1.16× between halves, so it carries a
weight of 1 rather than the 3 its full-train figure would have justified.

## How the weights were set

```
weight = round( min(lift_half_A, lift_half_B) )
```

The conservative half, not the average and not the full-train figure. Stated in advance,
computed on train, and it discounts unstable rules automatically rather than by judgement.

**No rule, threshold, band, cut point or weight was evaluated against validation.**
Doing so would make the incumbent a model tuned on the same data as its challenger, and
the Phase 06 comparison meaningless. The corollary is that the weighting scheme itself
could not be selected on validation either — so it was chosen on principle and checked
against train, which is a weaker basis and is recorded as such.

**Integer points, not real-valued weights.** This is a decision on principle: a rules
engine emits points, and a baseline that emits `lift`-valued scores is a small fitted
model wearing an incumbent's name. Weighting by raw measured lift would also spend the
stability check's benefit — the rounding is what keeps an unstable 1.16× and a stable
1.42× from being treated as meaningfully different.

An exploratory comparison on the training window found the real-valued variant scored no
better (see *Provenance of the numbers* below). That is corroboration, not the reason,
and it is a train-window figure — it does not license a claim about which scheme
generalises.

---

## R1 — Round amount in the $150–$500 band

**Predicate.** `TransactionAmt % 50 == 0 and 150 <= TransactionAmt <= 500`
**Provenance.** `hypothesis` — H2.
**Weight 2** — lift 1.72× (A) → 4.46× (B), full train 2.26×.

Fires 184.7/day at a 7.66% fraud rate, carrying $316,100 of train fraud.

H2 survived the `ProductCD` control and *strengthened* under it: $300 goes from 4.74×
pooled to 14.58× within product H, $450 from 12.94× to 24.20×. The band is specific
rather than a general preference for round numbers — $100 runs backwards at 0.68×
despite being the most common round amount in the data, and everything above $500 sits
at parity.

**Volume is unstable, lift is not.** R1 fires 293/day in half A and 76/day in half B.
That is not decay: it tracks the product-mix shift recorded under *Limitations* below.
H2 places the round amounts in products R and H, and those two products lose three
quarters of their share across the training window. What remains is more concentrated,
which is why the lift rises as the volume falls.

**Fitted on train:** nothing. The band edges and the $50 step are asserted from H2 and
live in `config/config.yaml`; only their continued validity was re-checked on days 1–90.

---

## R2 — Product tier

**Predicate.** `ProductCD` mapped to points: **C=3, S=2, H=1, R=0, W=0**
**Provenance.** `hypothesis` — H3.
**Weight: the tier itself** — C lift 3.28× (A) → 3.03× (B).

| ProductCD | rows/day | fraud rate | lift | fraud USD | tier |
|---|---:|---:|---:|---:|---:|
| C | 414.8 | 10.64% | 3.14× | $186,916 | 3 |
| S | 65.4 | 5.66% | 1.67× | $24,256 | 2 |
| H | 287.1 | 4.11% | 1.21× | $170,406 | 1 |
| R | 301.4 | 3.19% | 0.94× | $200,500 | 0 |
| W | 2,441.7 | 2.03% | 0.60× | $892,817 | 0 |

Tiered rather than binary so that one rule slot carries the entire channel axis. A
binary `ProductCD == "C"` rule plus a separate `has_identity` rule would spend two slots
on one signal — the two agree on 98.7% of rows (A8).

R and W score zero because both sit *below* the base rate. They are not evidence of
safety, only of nothing added.

**The money runs the other way, and that is the point.** C has the highest fraud rate
and only $186,916 of fraud USD; W has the lowest rate and $892,817. A policy tuned on
rate alone chases C and misses where the money is. The rules engine cannot resolve this
— it has no per-transaction cost model — and that limitation is one of the structural
reasons the Phase 05 model is expected to beat it.

**Fitted on train:** nothing. The tier map is a judgement recorded in config, derived
from the rates above.

---

## R3 — Amount above the within-product 99th percentile

**Predicate.** `TransactionAmt > p99(train, within ProductCD)`
**Provenance.** `hypothesis` — H1, in the form that survived.
**Weight 1** — lift 2.46× (A) → 1.42× (B), full train 1.90×.

Fires 31.3/day at 6.42%, carrying $126,517.

H1 was falsified pooled and holds only *within* product, so this is deliberately the
per-product percentile rather than a global threshold. A global one would be the pooled
artifact H1 was revised to reject.

**The low weight is the stability check working.** 1.42× in the later half is barely
above base rate, and the weight reflects that rather than the flattering full-train
1.90×.

Percentile level chosen for precision per review slot — p90 and p95 fire 327.6/day and
169.0/day at essentially the same lift (1.73× and 1.88×), so the extra volume buys
nothing at a ~35/day review budget.

**Fitted on train:** five per-product cut points — C $193.91, H $300.00, R $1,000.00,
S $400.00, W $1,331.00. These are the only fitted values in the engine.

---

## R4 — New card proxy

**Predicate.** `0 < D1 <= 3`
**Provenance.** `search` — directed scan of the `D*` family.
**Weight 3** — lift 3.60× (A) → 2.64× (B), full train 3.11×.

Fires 100.6/day at a 10.54% fraud rate, carrying $158,393. The strongest and most
stable rule in the set.

The cleanest monotone signal in the data:

| D1 | rows/day | fraud rate | lift |
|---|---:|---:|---:|
| 0 | 1,793.6 | 3.89% | 1.15× |
| 1 | 50.2 | 10.66% | 3.15× |
| 2–3 | 50.3 | 10.42% | 3.08× |
| 4–7 | 74.2 | 7.31% | 2.16× |
| 8–14 | 92.5 | 4.67% | 1.38× |
| 15–30 | 159.2 | 3.79% | 1.12× |
| 31–90 | 345.3 | 2.23% | 0.66× |
| 91–180 | 285.1 | 2.23% | 0.66× |
| 180+ | 659.8 | 1.33% | 0.39× |

`D1 == 0` is excluded deliberately: it covers 51% of all rows at 1.15×, so including it
would convert a precise rule into a broad one.

**Servability caveat.** `D1` is a Vesta-computed day-delta, and the lookback window that
produced it is undocumented — the limitation recorded in ROADMAP Phase 01. It arrives
with the request in this dataset, so the rule is servable *here*; reproducing it in a
real system would require the entity history the flag summarises. Documented, not
ignored.

**Fitted on train:** nothing. The cut of 3 is a choice from the table above, in config.

---

## R5 — Product W with `M4 == "M2"`

**Predicate.** `ProductCD == "W" and M4 == "M2"`
**Provenance.** `search` — directed scan of the `M*` family.
**Weight 1** — lift 2.25× (A) → 1.16× (B), full train 1.65×.

Fires 22.0/day at 5.60%, carrying $16,151.

The only rule in the set that is not on the amount or product axis, which is why it
earns a slot despite a modest weight. It is also sized to the constraint: 22 firings a
day against a ~35/day review budget.

**The `ProductCD == "W"` scope is load-bearing, not decoration.** Pooled, `M4 == "M2"`
looks like a 3.05× signal — but 94% of its rows are product C, and *within* C its lift
is 1.01×. Unscoped it is R2 restated for the third time. Scoped to W it carries genuine
independent signal at 2.75× within that product.

**This is the rule the selection caveat is about.** Its full-train 2.75× within-W figure
was inflated by having been chosen as the best of roughly twenty comparisons. The
half-split check caught it, and the weight of 1 is that correction.

**Fitted on train:** nothing.

---

## The tiebreaker

```
score += 0.5 * TransactionAmt.rank(pct=True)
```

**Why one is necessary — structurally, before any measurement.** Five integer-weighted
rules can produce at most a few dozen distinct point totals, and in practice far fewer,
because most transactions trigger no rule at all. The review budget is ~1% of ~3,500
transactions a day — roughly 35 slots — drawn from a population whose scores take a
single-digit number of values. Unless the high-scoring blocks happen to sum to almost
exactly 35, the capacity cut must land *inside* a block of equally-scored transactions,
and which of them gets reviewed is then decided by row order. That is arithmetic, not an
empirical finding: a coarse score cannot rank a fine selection.

`metrics.CapacityResult.ambiguous_days` exists to expose exactly this, and the Phase 02
docstring already anticipated the rules baseline as its ordinary case.

**Why break ties by amount.** Among transactions the rules cannot distinguish, review the
expensive ones first. This aligns the incumbent with the USD headline the project
reports, and it is what a real fraud team would do. It is a policy choice, stated rather
than hidden — the alternative is letting row order make the same decision silently and
worse.

**It shrinks the ambiguity, it does not eliminate it.** Exact-duplicate amounts share a
rank, so some ties survive by construction. No figure is quoted here for how much
improves — see *Provenance of the numbers*. Whatever `ambiguous_days` reports on VAL-FIT
and VAL-CAL is the number that gets published, and it is reported rather than engineered
away.

---

## Rejected rules

Three of these are suggested by the ROADMAP itself. All were dropped on measured
evidence, not preference.

### Email-domain mismatch — **falsified, and backwards**

`R_emaildomain` is null on 72.8% of train, and the nullness is structural rather than
informative: 100% null in W, 0% null in R. On the 25.4% of rows where both domains are
present, the direction is the opposite of the premise:

| | rows | fraud rate | lift |
|---|---:|---:|---:|
| domains **match** | 63,782 | 8.31% | 2.45× |
| domains **mismatch** | 16,531 | 2.29% | **0.68×** |

It survives the `ProductCD` control in that reversed direction (H: 2.62% vs 8.05%;
R: 2.08% vs 3.84%). As specified the rule fires on 5.23% of train and catches 3.5% of
fraud — worse than selecting rows at random.

### Missing identity — **direction reversed once controlled**

Pooled, `has_identity == False` looks like a 3.42× signal. Within product it is 0.54×
in C, 0.80× in H, 0.84× in R, 0.77× in S — and W is 100% no-identity, so it admits no
comparison at all. The signal points at identity being *present*: 11.23% vs 6.07% in C,
a 1.85× residual that sits inside the band A8 recorded.

The inverted rule is real but redundant — it agrees with `ProductCD != W` on 98.7% of
rows, making it a third pass at the channel axis R2 already covers.

### `M2 == "F"` — **too weak once ranked globally**

Within W it looks strong at 1.92×. But the score ranks transactions across all products
within a day, so pooled lift is the decision-relevant figure: **1.15×**, decaying to
0.97× in half B — below base rate. A rule that finds risky-for-W transactions still
selects rows less risky than an average C transaction, and it drags the ranking.

### Missing billing address — **restatement of R2**

99.2% of address-null rows are product C, and `addr1`/`addr2` share an identical null
mask. Its 4.48× pooled lift falls to a 1.30× residual within C. H3 settled this.

### Unusual hour — **denominator artifact**

Bucket 8 shows a 10.01% fraud rate against 3.45% on the plateau — the most tempting
number in the Phase 01 notebook. But it is 1.6 frauds/day against 8.5, because
legitimate volume collapses overnight harder than fraud does. At a 35/day review budget
the rule spends its weight on a window containing almost no fraud to catch.

`hour` is also a cyclic feature and not a wall-clock label — bucket 0 is not midnight —
so no time-of-day rationale could have been stated honestly in any case.

---

## Limitations

1. **Two rules came from a search, not a hypothesis.** R4 and R5 were found by scanning
   the `D*` and `M*` families on train — roughly twenty flag/value comparisons, best two
   kept. Their full-train lift is optimistic by an unmeasured amount. The half-split
   check is the correction, and it demonstrably bit: R5's weight is 1 because of it.

2. **The product mix shifts sharply inside the training window.** H falls from 12.4% to
   2.5% of volume between halves, R from 12.1% to 3.8%, while W rises from 61.4% to
   80.5%. This is the non-stationarity the temporal split exists to respect, visible
   *before* the purge gap. It means R1 and R2 will fire at different rates on validation
   than on train, and it gives Phase 09 a known-nontrivial `ProductCD` PSI to start from.
   Whether the shift is gradual or a step change has not been established.

3. **The engine cannot price a transaction.** It emits points, not probabilities, so the
   Phase 06 policy `p* = C_fp / (amount + fee + C_fp)` cannot be applied to it. It can
   only select a fixed set. This is a structural ceiling, not a tuning failure, and it is
   one of the two reasons a model is expected to beat it — the other being that 8
   distinct scores cannot rank 35 reviews out of ~3,500 daily transactions.

4. **R1's band edges were formed on days 1–120**, which includes the purge gap. This is
   not leakage: the gap is purged for label maturity, not held out for evaluation, and
   everything evaluated on begins at day 121. Recorded because it is a fair question.
