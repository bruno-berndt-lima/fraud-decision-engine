# Data provenance

Recorded so that "the same data" is a checkable claim rather than an assumption.
If a future run produces different numbers, the first question is whether the
inputs changed — this file is how that gets answered.

## Source

| | |
|---|---|
| **Dataset** | IEEE-CIS Fraud Detection |
| **Provider** | Vesta Corporation |
| **Competition** | https://www.kaggle.com/competitions/ieee-fraud-detection |
| **Closed** | 2019-10-03 |
| **Files dated** | 2019-12-11 |
| **Retrieved** | 2026-08-12 |

Access requires a Kaggle account, an API token at `~/.kaggle/access_token`, and
acceptance of the competition rules — the API returns 403 without the last one.

**Licensing.** The data is used here under the competition rules for
non-commercial portfolio work. It is **not redistributed**: `data/` is entirely
gitignored, and notebook outputs are stripped by the `nbstripout` pre-commit hook
so that no transaction rows can reach the repository through a committed
`.ipynb`. Analysis outputs — aggregate statistics, figures, metrics, the trained
model — are published; the data itself is not.

## Retrieval

```bash
make download
```

which runs:

```bash
uvx kaggle competitions download -c ieee-fraud-detection -p data/raw
cd data/raw && unzip -o ieee-fraud-detection.zip
```

The client is invoked through `uvx` rather than being a project dependency.
Nothing in `src/` imports it, and reproducibility of the *data* is established by
the checksums below, not by pinning the tool that fetched the bytes.

## Verified contents

Row and column counts were measured, not taken from the spec. Columns exclude
nothing; rows exclude the header.

| File | Rows | Cols | Bytes | SHA-256 |
|---|---:|---:|---:|---|
| `train_transaction.csv` | 590,540 | 394 | 683,351,067 | `3a5c83ab6b3cc13dcabe5ffa9f522307fd5f7f7b6e6f6a60c32284ca6283d642` |
| `train_identity.csv` | 144,233 | 41 | 26,529,680 | `b63c725d8377be90a995268d97f347c17d456b95db45807adcf9f59cd603c37c` |
| `test_transaction.csv` | 506,691 | 393 | 613,194,934 | `2a8e51f1d335a86025d2b7f45beb9b78d0ab1edd726ef531d8b71a8a0065c011` |
| `test_identity.csv` | 141,907 | 41 | 25,797,161 | `3e5978cb13ca5e72f52babc4349ae0125e14b87ca8bfabe952ab67bb4ff1e10b` |

`test_transaction.csv` has 393 columns to `train_transaction.csv`'s 394 — the
missing one is `isFraud`.

To re-verify:

```bash
cd data/raw && shasum -a 256 train_transaction.csv train_identity.csv
```

## Which files are usable

**`train_transaction.csv` and `train_identity.csv` only.**

The test pair has no `isFraud` column — it is absent from the header entirely,
not null-filled. Kaggle withheld the test labels to score submissions and never
released them after the competition closed. There is therefore no way to
evaluate a prediction on those rows, which is why the project carves its own
train / validation / test split out of `train_transaction.csv` (Phase 02) and why
the `Makefile` declares only the two training files as pipeline inputs.

`sample_submission.csv` is a Kaggle submission-format template and is unused.

**One later use for the unlabelled test files.** They cover the period *after*
the training window. Phase 09's drift monitoring — PSI over feature
distributions, and prediction-distribution drift — requires no labels at all, so
those rows provide an extra horizon for the decay analysis even though they can
never be scored for accuracy. Worth keeping rather than deleting.

## Join relationship

`train_identity.csv` covers 144,233 of 590,540 transactions — about 24%. Identity
is joined as a LEFT join on `TransactionID`, and its **absence is itself a
signal**, so an explicit `has_identity` flag is created before any filling or
imputation happens.

## Open item

These checksums are currently recorded but not enforced. A recorded checksum that
nothing verifies is documentation, not a control. Decide when writing `load.py`
whether verification runs at load time (stronger, costs ~10s per run) or as a
separate `make verify-data` target (cheaper, relies on being remembered).
