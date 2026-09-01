# Feature inventory and serving feasibility

| | |
|---|---|
| **Status** | Phase 04. 353 columns, every one classified. |
| **Last updated** | 2026-08-28 |
| **Source of truth** | `data/features/{train,val_fit,val_cal,test}.parquet` |
| **Measurements cited** | `docs/experiments.md` E4, `reports/metrics/family_*.json` |

---

This table answers one question per column: **what would have to exist at serving
time for this feature to be computable for a single incoming transaction?**

It is a record of *cost*, not a selection decision. Nothing is dropped on the
strength of it. Which features the model actually uses is Phase 05's question,
and whether a feature's contribution pays for its infrastructure is Phase 06's,
in USD. Deciding either here would be spending evidence this phase does not have.

## The four tiers

The roadmap's DoD asks for `servable` / `needs-cache`. Two labels turned out to
state something false about this dataset, so the axis has four values:

| tier | what serving needs | who can build it |
|---|---|---|
| **1 — request-only** | nothing beyond the fields of the request itself | us, for free |
| **2 — static table** | a file fitted at train time, shipped beside the model, versioned with it | us, cheaply |
| **3 — live entity state** | a per-entity running window updated on every transaction — Redis or a feature store | us, expensively |
| **0 — inherited** | *unreproducible* — a pre-computed aggregate over a window nobody published | nobody, from this repository |

Tier 0 is not a fourth grade of cache. It is a different kind of claim. `C*`, `D*`
and the V block are Vesta's own feature engineering, delivered pre-computed in
the CSV. We do not know what they count, or over what window, so no store we
chose to build would reproduce them. Calling them `needs-cache` would imply a
cache exists that solves it.

**The rule for undocumented columns is: default to tier 0.** A column is promoted
out of it only where its own values prove the derivation — `id_31` holds
`"chrome 62.0"`, which a request carries in a header, so it is tier 1 on
evidence rather than on optimism. This deliberately over-assigns to tier 0. An
inventory that guesses generously about what it could serve is worth nothing.

---

## The inventory

### Tier 1 — request-only (36 columns)

| columns | n | source |
|---|---|---|
| `TransactionAmt`, `ProductCD` | 2 | the request |
| `card1`–`card6` | 6 | the request (raw; the *encodings* are tier 2) |
| `addr1`, `addr2`, `dist1`, `dist2` | 4 | the request |
| `P_emaildomain`, `R_emaildomain` | 2 | the request |
| `M1`–`M9` | 9 | match flags carried on the transaction |
| `DeviceType`, `DeviceInfo`, `has_identity` | 3 | the request; `has_identity` is whether the identity block arrived at all |
| `hour`, `weekday` | 2 | derived from the request timestamp |
| `amt_whole_dollar`, `amt_round_band`, `amt_round_band_product` | 3 | row-local arithmetic on `TransactionAmt` — H2 |
| `id_14` | 1 | timezone offset in minutes (`-480`, `-300`, `-360`) |
| `id_30`, `id_31`, `id_32`, `id_33` | 4 | OS string, browser string, colour depth, screen resolution |

This tier costs nothing to serve and would still work if every store in the
system were down. It is also the fail-open feature set — the rules engine
(`rules-baseline.md`) is built entirely from it, which was a stated constraint
there for exactly this reason.

### Tier 2 — static fitted table (14 columns)

| columns | n | artifact | fitted on |
|---|---|---|---|
| `freq_card1/2/3/5`, `freq_addr1/2`, `freq_DeviceInfo` | 7 | `models/encoders.parquet` | train only |
| `amt_mean/z/absz_card1`, `amt_mean/z/absz_addr1` | 6 | `models/amount_stats.parquet` | train only |
| `id_23` | 1 | IP proxy classification — a third-party list, not fitted here | — |

Plus one artifact that produces no columns of its own: `models/vblock.parquet`,
the 234 surviving V names and the median each gap is filled with.

**The staleness contract.** These tables are frozen at train time, and that has a
cost the other tiers do not pay:

- A `card1` first seen after the training window scores `UNSEEN_FREQUENCY = 0.0`
  — not null, not a flag, but the smallest value in the column — and keeps
  scoring it until the next retrain.
- The same card gets the global prior for its amount statistics, which at
  `prior_strength: 10` is the honest answer for an entity nothing is known
  about, and stays the answer no matter how many transactions it accumulates
  in production.

So model quality decays between retrains in a way the tier-1 features do not.
That decay is measurable and Phase 09 measures it; what this table records is
that tier 2 is where it comes from.

### Tier 3 — live entity state (4 columns)

| columns | n | what it needs |
|---|---|---|
| `vel_n1h_card1`, `vel_n24h_card1`, `vel_n7d_card1` | 3 | a rolling count per `card1` over three windows |
| `vel_recency_card1` | 1 | the last-seen timestamp per `card1` |

The only family here, and the one the roadmap's train/serve-skew note is about.
Serving it means a keyed store written on every transaction, with its own
availability, latency and correctness problems — a stale write produces a
silently wrong count rather than an error.

A cheaper approximation exists and is not taken: `vel_recency_card1` alone needs
only a last-seen timestamp, which is a far smaller store than three rolling
windows. If Phase 06 finds the counts do not pay, that is the fallback, and it
would move this family's cost close to tier 2's.

### Tier 0 — inherited, unreproducible (295 columns)

| columns | n | what they are |
|---|---|---|
| `vb_V*` | 234 | 228 representatives + 6 presence flags, reduced from 339 raw `V*` |
| `D1`–`D15` | 15 | day-deltas — time since some prior event Vesta defines and does not name |
| `C1`–`C14` | 14 | counts of "how many addresses/cards/etc. are associated", window unpublished |
| `id_12`, `id_15`, `id_16`, `id_27`, `id_28`, `id_29` | 6 | `Found` / `NotFound` / `New` — outcomes of a lookup against Vesta's own history store |
| remaining `id_*` | 26 | undocumented; tier 0 by the default rule above |

**This is where the entire measured signal of Phase 04 lives.** It is also the
project's principal limitation, recorded in full in E4: a Phase 01 trap about
`C*` and `D*` that now applies to the headline. The pipeline that reduces the V
block is clean and tested — grouping is structural, clustering and medians are
train-only — but the exposure is inherited and cannot be measured or bounded
from inside this repository.

**A note on `vb_*` specifically.** The reduction *replaces* the raw block rather
than supplementing it, so the matrices went 694 → 353 columns: 339 `V*` out, 234
`vb_*` in. Serving carries the survivor list and the medians, never the block.
The original name is recoverable from the `vb_` prefix, so Phase 07 can still
trace a SHAP contribution back to the Vesta column it came from.

### Not features

`TransactionID` and `TransactionDT` are keys. `isFraud` is the label. **`day` is
the split axis and must never enter a model** — it is already excluded from the
logistic baseline by name in `config.yaml`, and a model given it learns the
timeline instead of fraud.

---

## What each tier bought

Measured by the Phase 03 logistic probe on VAL-FIT, one family at a time against
the same baseline. Full method and per-metric table in E4.

| tier | family | ΔPR-AUC | vs. the floor |
|---|---|---|---|
| 1 | amount | −0.00115 | inside |
| 2 | frequency | −0.00517 | inside, negative |
| 2 | entity | −0.00451 | inside, negative |
| 3 | velocity | +0.00039 | inside |
| 0 | vblock | **+0.11192** | 22× the bar |

Noise floor: σ 0.00156 over 20 seeds; the bar is +0.005.

**This is a measurement by one instrument, and the instrument's blind spots were
registered before the run.** The E4 blind-spot table predicted in advance that a
linear probe structurally cannot see velocity's shape — threshold effects
("three in an hour" is a different state, not three times worse than one) and
interaction with card identity (five per hour is routine on one card and fraud
on another). It predicted the V block would be visible. Both predictions held.

So the correct reading of the −0.005 and +0.0004 rows is *"the linear probe found
nothing"*, not *"the feature is worthless"*. **Every family ships in the
matrices**, which is the E4 protocol: families are characterised, never accepted
or rejected. Phase 05 re-runs this comparison with LightGBM, and if velocity
moves under a tree while sitting still under the probe, that is the blind-spot
table confirmed by measurement — a stronger result than the original numbers,
because the prediction was on record first.

## The uncomfortable read

The roadmap asks the project to restrict itself to servable features and
*quantify the quality it gives up*. On the probe, the answer inverts the usual
shape of that trade:

- The expensive tier (3) is the one measurement cannot yet justify.
- The cheap fitted tier (2) measured negative.
- Everything that works is tier 0 — and tier 0 is precisely the tier we could
  not build.

A real deployment would have to re-derive the `C*`, `D*` and `V*` equivalents
in-house from an entity store. That is tier 3 infrastructure at roughly 269
columns' worth of scale, and it is the honest architectural consequence of this
result: the cost this project's headline number quietly assumes is far larger
than the Redis instance the roadmap warns about.

## Handoff

- **Phase 05** — re-run the family ablation with LightGBM and compare against the
  table above. Phase 05's own DoD contains no ablation, so this is added there
  deliberately. E3's servable-vs-history comparison belongs in the same run.
- **Phase 06** — price tier 3 in USD. The question is whether the velocity
  family's contribution under the real model exceeds the cost of the store, and
  the fallback if it does not is `vel_recency_card1` alone.
- **Phase 08** — the serving feature set is chosen from this table, not from
  scratch. Tier 0 columns arrive from the dataset; a production system would not
  have them, and the README must say so.
