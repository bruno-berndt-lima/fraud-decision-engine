# ==============================================================================
# Preamble
# ==============================================================================
SHELL := bash -euo pipefail
.DELETE_ON_ERROR:
MAKEFLAGS += --warn-undefined-variables
MAKEFLAGS += --no-builtin-rules
.DEFAULT_GOAL := help

# ==============================================================================
# Variables
# ==============================================================================
RUN    := uv run
# One-off fetch tool, deliberately NOT a project dependency: nothing in src/
# imports it, and adding it pulls ~20 transitive packages into every CI install.
# Data reproducibility comes from the recorded checksums, not from pinning the
# client that downloaded the bytes.
KAGGLE := uvx kaggle

# ---- Directories -------------------------------------------------------------
DATA_DIR        := data
RAW_DIR         := $(DATA_DIR)/raw
INTERIM_DIR     := $(DATA_DIR)/interim
SPLITS_DIR      := $(DATA_DIR)/splits
FEATURES_DIR    := $(DATA_DIR)/features
PREDICTIONS_DIR := $(DATA_DIR)/predictions
MODEL_DIR       := models
REPORTS_DIR     := reports
CONFIG_DIR      := config

# ---- Config (stage inputs: editing these should trigger a rebuild) -----------
CONFIG      := $(CONFIG_DIR)/config.yaml
COST_MATRIX := $(CONFIG_DIR)/cost_matrix.yaml
RAW_SUMS    := docs/raw_checksums.txt

# ---- Stage outputs -----------------------------------------------------------
# Several stages write more than one file — splits.py produces train/val_fit/
# val_cal/test. Make cannot express "one recipe, many outputs" before v4.3
# (grouped targets, `&:`), and this machine has 3.81. So each stage is
# represented below by a single file: if that file is up to date, the stage
# is assumed to have run.
RAW_TXN   := $(RAW_DIR)/train_transaction.csv
RAW_ID    := $(RAW_DIR)/train_identity.csv
# Not a stage output: a stamp proving raw/ still hashes to what docs recorded.
# Make-only, so it has no config.yaml twin — see SENTINEL_ONLY in
# tests/test_config_paths.py.
VERIFIED  := $(RAW_DIR)/.verified
INTERIM   := $(INTERIM_DIR)/transactions.parquet
SPLITS    := $(SPLITS_DIR)/splits.parquet
FEATURES  := $(FEATURES_DIR)/train.parquet
MODEL     := $(MODEL_DIR)/model.pkl
# Unlike every other stage output, this one is TRACKED: reports/ is a
# deliverable. Represents the whole baselines stage per the note above.
BASELINES := $(REPORTS_DIR)/metrics/rules_baseline.json
LOGISTIC  := $(REPORTS_DIR)/metrics/logistic_baseline.json
# Each baselines run writes a JSON record AND a predictions parquet. Per the
# note above, the JSON stands for the pair — so the figures stage depends on the
# records, not on the parquets it actually reads. Also tracked.
FIGURES   := $(REPORTS_DIR)/figures/pr_curve_baselines.png
# Phase 04. Two separate stages, deliberately. The floor calibrates the probe and
# is deterministic given its seeds, so it is measured when the probe changes, not
# when the families do — folding it into the family run would recompute a
# constant for eighteen minutes. FAMILIES stands for the per-family records, per
# the single-sentinel note above. Both tracked, like the other reports/ outputs.
FAMILY_FLOOR := $(REPORTS_DIR)/metrics/noise_floor.csv
FAMILIES     := $(REPORTS_DIR)/metrics/family_none.json

# ==============================================================================
# Meta
# ==============================================================================
# Any target carrying a `## ` comment is listed by `make help`, so the menu
# cannot drift from the targets that actually exist.
.PHONY: help
help:  ## Show available targets
	@grep -E '^[a-zA-Z_-]+:.*?## ' $(MAKEFILE_LIST) \
		| awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-14s\033[0m %s\n", $$1, $$2}'

# ==============================================================================
# Development
# ==============================================================================
.PHONY: setup
setup:  ## Install the locked dependencies into .venv
	uv sync

.PHONY: test
test:  ## Run the test suite
	$(RUN) pytest

.PHONY: lint
lint:  ## Check formatting and lint rules (read-only — CI uses this)
	$(RUN) ruff format --check .
	$(RUN) ruff check .

.PHONY: format
format:  ## Apply formatting and autofixable lint rules
	$(RUN) ruff format .
	$(RUN) ruff check --fix .

.PHONY: check
check: lint test  ## Everything CI runs

# ==============================================================================
# Pipeline
# ==============================================================================
$(INTERIM_DIR) $(SPLITS_DIR) $(FEATURES_DIR) $(PREDICTIONS_DIR) $(MODEL_DIR) $(REPORTS_DIR):
	mkdir -p $@

.PHONY: download
download:  ## Fetch the IEEE-CIS CSVs from Kaggle into data/raw/
	$(KAGGLE) competitions download -c ieee-fraud-detection -p $(RAW_DIR)
	cd $(RAW_DIR) && unzip -o ieee-fraud-detection.zip

# A recorded checksum that nothing verifies is documentation, not a control. This
# makes it one: the stamp is a prerequisite of the load, so `make data` cannot run
# against raw files that no longer hash to what docs/raw_checksums.txt records.
#
# A stamp rather than a .PHONY target because verification is only meaningful when
# the inputs change. Hashing 710 MB costs ~3s; make skips it entirely on every run
# where the CSVs and the recorded sums are both older than the stamp, so the
# control is free in the common case and unforgettable in the case that matters.
#
# The `rm -f` is load-bearing and .DELETE_ON_ERROR: does not replace it. That
# only deletes a target the failed recipe actually wrote, and a failing shasum
# never reaches the touch — so without this line a mismatch leaves the previous
# run's stamp in place, a file on disk asserting the data was verified when the
# last attempt to verify it failed. Clearing it first makes the stamp's presence
# mean exactly one thing: the check passed.
$(VERIFIED): $(RAW_TXN) $(RAW_ID) $(RAW_SUMS)
	rm -f $@
	cd $(RAW_DIR) && shasum -a 256 -c $(abspath $(RAW_SUMS))
	touch $@

$(INTERIM): $(VERIFIED) $(CONFIG) \
            src/fraud_engine/data/load.py src/fraud_engine/data/validate.py \
            | $(INTERIM_DIR)
	$(RUN) python -m fraud_engine.data.load

$(SPLITS): $(INTERIM) $(CONFIG) src/fraud_engine/data/splits.py | $(SPLITS_DIR)
	$(RUN) python -m fraud_engine.data.splits

# Phase 03. Reads interim + splits directly and skips $(FEATURES) entirely:
# the incumbent must be servable from a single request, so it uses only columns
# that arrive with one. Engineered features are Phase 04 and are not available
# to it by design.
$(BASELINES): $(SPLITS) $(INTERIM) $(CONFIG) $(COST_MATRIX) \
              src/fraud_engine/models/rules.py \
              src/fraud_engine/evaluation/report.py \
              src/fraud_engine/evaluation/metrics.py | $(REPORTS_DIR) $(PREDICTIONS_DIR)
	$(RUN) python -m fraud_engine.models.rules

# Two records from one run - E2 requires both variants reported, so they are
# produced together and logistic_baseline.json stands for the pair.
$(LOGISTIC): $(SPLITS) $(INTERIM) $(CONFIG) $(COST_MATRIX) \
             src/fraud_engine/models/logistic.py \
             src/fraud_engine/evaluation/report.py \
             src/fraud_engine/evaluation/metrics.py | $(REPORTS_DIR) $(PREDICTIONS_DIR)
	$(RUN) python -m fraud_engine.models.logistic

# Reads the predictions both stages above wrote — never a model. A figure is a
# view of what a run said, so redrawing it must not be able to produce numbers
# the metrics record disagrees with.
$(FIGURES): $(BASELINES) $(LOGISTIC) $(CONFIG) $(COST_MATRIX) \
            src/fraud_engine/evaluation/figures.py \
            src/fraud_engine/evaluation/plots.py \
            src/fraud_engine/evaluation/metrics.py | $(REPORTS_DIR)
	$(RUN) python -m fraud_engine.evaluation.figures

# Phase 04. $(INTERIM) is named even though $(SPLITS) already depends on it:
# build.py reads the interim table itself, and a rule should declare the
# dependencies a stage has rather than the ones it happens to inherit.
$(FEATURES): $(SPLITS) $(INTERIM) $(CONFIG) \
             src/fraud_engine/features/build.py \
             src/fraud_engine/features/amounts.py \
             src/fraud_engine/features/encoders.py | $(FEATURES_DIR) $(MODEL_DIR)
	$(RUN) python -m fraud_engine.features.build

# Both depend on logistic.py because the probe IS the logistic pipeline: a change
# to build_pipeline changes every family's number and the floor they are read
# against, so both must go stale.
$(FAMILY_FLOOR): $(FEATURES) $(CONFIG) $(COST_MATRIX) \
                 src/fraud_engine/features/floor.py \
                 src/fraud_engine/features/evaluate.py \
                 src/fraud_engine/models/logistic.py \
                 src/fraud_engine/evaluation/report.py | $(REPORTS_DIR)
	$(RUN) python -m fraud_engine.features.floor

$(FAMILIES): $(FEATURES) $(CONFIG) $(COST_MATRIX) \
             src/fraud_engine/features/evaluate.py \
             src/fraud_engine/models/logistic.py \
             src/fraud_engine/evaluation/report.py | $(REPORTS_DIR) $(PREDICTIONS_DIR)
	$(RUN) python -m fraud_engine.features.evaluate

$(MODEL): $(FEATURES) $(CONFIG) src/fraud_engine/models/train.py | $(MODEL_DIR)
	$(RUN) python -m fraud_engine.models.train

# Forces the check the stamp normally lets make skip. `make data` already
# verifies whenever raw/ changed; this is for re-checking on demand — after a
# disk scare, or before trusting a number you are about to publish.
.PHONY: verify-data
verify-data:  ## Re-check raw/ against docs/raw_checksums.txt, ignoring the stamp
	rm -f $(VERIFIED)
	$(MAKE) --no-print-directory $(VERIFIED)

.PHONY: data splits baselines figures features families floor train
data:      $(INTERIM)   ## Build interim/transactions.parquet from raw CSVs
splits:    $(SPLITS)    ## Assign transactions to temporal splits
baselines: $(BASELINES) $(LOGISTIC) $(FIGURES) ## Score both baselines through the Phase 02 harness
figures:   $(FIGURES)   ## Redraw the baseline comparison figures from predictions
features:  $(FEATURES)  ## Build train/val/test feature matrices
families:  $(FAMILIES)  ## Score each feature family on VAL-FIT
floor:     $(FAMILY_FLOOR) ## Re-measure how far chance alone moves the metric
train:     $(MODEL)     ## Train the model

# ==============================================================================
# Housekeeping
# ==============================================================================
.PHONY: clean
clean:  ## Remove derived data, models and caches (never raw/ or reports/)
	find $(INTERIM_DIR) $(SPLITS_DIR) $(FEATURES_DIR) $(PREDICTIONS_DIR) \
	-type f ! -name '.gitkeep' -delete
	rm -rf $(MODEL_DIR)
	find src tests -type d -name '__pycache__' -prune -exec rm -rf {} +
	rm -rf .pytest_cache .ruff_cache
