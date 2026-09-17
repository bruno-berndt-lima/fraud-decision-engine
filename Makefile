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
# config.yaml is never a prerequisite itself: stages depend on the sections they
# read, through `$(call sections,...)` below. cost_matrix.yaml is small and read
# whole, so it stays a plain file prerequisite.
CONFIG_STAMPS := $(DATA_DIR)/config
COST_MATRIX   := $(CONFIG_DIR)/cost_matrix.yaml
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
MODEL     := $(MODEL_DIR)/model.txt
MEDIANS   := $(MODEL_DIR)/medians.parquet
CALIBRATOR  := $(MODEL_DIR)/calibrator.json
RELIABILITY := $(REPORTS_DIR)/figures/reliability_val_cal.png
REHEARSAL   := $(REPORTS_DIR)/metrics/policy_val_cal.json
SENSITIVITY := $(REPORTS_DIR)/metrics/sensitivity_val_cal.json
USD_HALVES  := $(REPORTS_DIR)/metrics/usd_halves.json
HEADLINE    := $(REPORTS_DIR)/metrics/policy_test.json
EXPLAIN_DIR := $(DATA_DIR)/explain
SHAP_GLOBAL := $(REPORTS_DIR)/metrics/shap_global.json
EXPLAIN_LATENCY := $(REPORTS_DIR)/metrics/explain_latency.json
EXPLAIN_FIGURE  := $(REPORTS_DIR)/figures/shap_ranking_by_tier.png
REASON_DICT     := $(CONFIG_DIR)/reason_codes.yaml
REASON_CODES    := $(REPORTS_DIR)/metrics/reason_codes.json
SEED_SPREAD := $(REPORTS_DIR)/metrics/seed_spread.csv
IMBALANCE   := $(REPORTS_DIR)/metrics/imbalance.csv
ABLATION    := $(REPORTS_DIR)/metrics/ablation.csv
ABL_FLOOR   := $(REPORTS_DIR)/metrics/ablation_floor.csv
PURGE       := $(REPORTS_DIR)/metrics/purge.csv
TUNING      := $(REPORTS_DIR)/metrics/tuning.json
# Unlike every other stage output, this one is TRACKED: reports/ is a
# deliverable. Represents the whole baselines stage per the note above.
BASELINES := $(REPORTS_DIR)/metrics/rules_baseline.json
# The same stage's other output, and the only one a served process reads: the
# incumbent's fitted constants. Gitignored with the rest of models/, unlike the
# record above — it is derived, and rebuilt whenever the stage runs.
RULES_CONSTANTS := $(MODEL_DIR)/rules.json
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

# ---- Config sections ----------------------------------------------------------
# One stamp per top-level section of config.yaml, rewritten only when that
# section's content changes. Refreshed here, while the Makefile is parsed, so
# make compares timestamps that are already current: as a phony prerequisite
# instead, make 3.81 reports every dependent as stale on a dry run.
#
# Skipped for goals that build nothing from config, so `make check` in CI never
# runs the pipeline's Python.
sections = $(foreach section,$(1),$(CONFIG_STAMPS)/$(section).stamp)

NON_PIPELINE_GOALS := help setup test lint format check download verify-data clean
ifneq ($(filter-out $(NON_PIPELINE_GOALS),$(or $(MAKECMDGOALS),$(.DEFAULT_GOAL))),)
_STAMPS_FAILED := $(shell $(RUN) python -m fraud_engine.config_stamps >&2 || echo failed)
ifneq ($(_STAMPS_FAILED),)
$(error could not refresh the config section stamps in $(CONFIG_STAMPS))
endif
endif

# A stamp the refresh did not write is a section config.yaml does not have.
$(CONFIG_STAMPS)/%.stamp:
	$(error config.yaml has no section '$*', but a stage depends on it)

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

$(INTERIM): $(VERIFIED) $(call sections,load) \
            src/fraud_engine/data/load.py src/fraud_engine/data/validate.py \
            | $(INTERIM_DIR)
	$(RUN) python -m fraud_engine.data.load

$(SPLITS): $(INTERIM) $(call sections,load splits) src/fraud_engine/data/splits.py | $(SPLITS_DIR)
	$(RUN) python -m fraud_engine.data.splits

# Phase 03. Reads interim + splits directly and skips $(FEATURES) entirely:
# the incumbent must be servable from a single request, so it uses only columns
# that arrive with one. Engineered features are Phase 04 and are not available
# to it by design.
$(BASELINES): $(SPLITS) $(INTERIM) $(call sections,load baselines) $(COST_MATRIX) \
              src/fraud_engine/models/rules.py \
              src/fraud_engine/evaluation/report.py \
              src/fraud_engine/evaluation/metrics.py \
              | $(REPORTS_DIR) $(PREDICTIONS_DIR) $(MODEL_DIR)
	$(RUN) python -m fraud_engine.models.rules

# Two records from one run - E2 requires both variants reported, so they are
# produced together and logistic_baseline.json stands for the pair.
$(LOGISTIC): $(SPLITS) $(INTERIM) $(call sections,load baselines) $(COST_MATRIX) \
             src/fraud_engine/models/logistic.py \
             src/fraud_engine/evaluation/report.py \
             src/fraud_engine/evaluation/metrics.py | $(REPORTS_DIR) $(PREDICTIONS_DIR)
	$(RUN) python -m fraud_engine.models.logistic

# Reads the predictions both stages above wrote — never a model. A figure is a
# view of what a run said, so redrawing it must not be able to produce numbers
# the metrics record disagrees with.
$(FIGURES): $(BASELINES) $(LOGISTIC) $(call sections,load) $(COST_MATRIX) \
            src/fraud_engine/evaluation/figures.py \
            src/fraud_engine/evaluation/plots.py \
            src/fraud_engine/evaluation/metrics.py | $(REPORTS_DIR)
	$(RUN) python -m fraud_engine.evaluation.figures

# Phase 04. $(INTERIM) is named even though $(SPLITS) already depends on it:
# build.py reads the interim table itself, and a rule should declare the
# dependencies a stage has rather than the ones it happens to inherit.
$(FEATURES): $(SPLITS) $(INTERIM) $(call sections,load splits features) \
             src/fraud_engine/features/build.py \
             src/fraud_engine/features/amounts.py \
             src/fraud_engine/features/encoders.py \
             src/fraud_engine/features/aggregations.py \
             src/fraud_engine/features/velocity.py \
             src/fraud_engine/features/vblock.py | $(FEATURES_DIR) $(MODEL_DIR)
	$(RUN) python -m fraud_engine.features.build

# Both depend on logistic.py because the probe IS the logistic pipeline: a change
# to build_pipeline changes every family's number and the floor they are read
# against, so both must go stale.
$(FAMILY_FLOOR): $(FEATURES) $(call sections,load splits baselines features) $(COST_MATRIX) \
                 src/fraud_engine/features/floor.py \
                 src/fraud_engine/features/evaluate.py \
                 src/fraud_engine/features/registry.py \
                 src/fraud_engine/models/logistic.py \
                 src/fraud_engine/evaluation/report.py | $(REPORTS_DIR)
	$(RUN) python -m fraud_engine.features.floor

$(FAMILIES): $(FEATURES) $(call sections,load splits baselines) $(COST_MATRIX) \
             src/fraud_engine/features/evaluate.py \
             src/fraud_engine/features/registry.py \
             src/fraud_engine/models/logistic.py \
             src/fraud_engine/evaluation/report.py | $(REPORTS_DIR) $(PREDICTIONS_DIR)
	$(RUN) python -m fraud_engine.features.evaluate

$(MODEL): $(FEATURES) $(call sections,load splits model) src/fraud_engine/models/train.py src/fraud_engine/evaluation/tracking.py | $(MODEL_DIR)
	$(RUN) python -m fraud_engine.models.train

# A bar rather than a result, so it is measured when the data or the pipeline
# changes and not once per tuning trial. Same reasoning as $(FAMILY_FLOOR).
$(SEED_SPREAD): $(FEATURES) $(call sections,load splits model) $(COST_MATRIX) \
                src/fraud_engine/evaluation/tracking.py \
                src/fraud_engine/models/seeds.py \
                src/fraud_engine/models/train.py \
                src/fraud_engine/evaluation/report.py | $(REPORTS_DIR)
	$(RUN) python -m fraud_engine.models.seeds

# E2's second run. Separate from $(MODEL) because the shipped model is one arm
# of it, and retraining should not re-answer a question that has not changed.
$(IMBALANCE): $(FEATURES) $(call sections,load splits model) $(COST_MATRIX) \
              src/fraud_engine/evaluation/tracking.py \
              src/fraud_engine/models/imbalance.py \
              src/fraud_engine/models/train.py \
              src/fraud_engine/evaluation/report.py | $(REPORTS_DIR) $(PREDICTIONS_DIR)
	$(RUN) python -m fraud_engine.models.imbalance

# E4's handoff, under a tree. Measures on the untuned reference rather than on
# the shipped configuration, so it does not depend on $(MODEL) and adopting a
# new one does not restage it.
$(ABLATION): $(FEATURES) $(call sections,load splits model) $(COST_MATRIX) \
             src/fraud_engine/evaluation/tracking.py \
             src/fraud_engine/models/ablation.py \
             src/fraud_engine/models/train.py \
             src/fraud_engine/features/registry.py \
             src/fraud_engine/evaluation/report.py | $(REPORTS_DIR) $(PREDICTIONS_DIR)
	$(RUN) python -m fraud_engine.models.ablation

# Separate from $(ABLATION) for the reason $(FAMILY_FLOOR) is separate from
# $(FAMILIES): fifty fits answer a question the six arms do not change, so
# re-measuring a family should not re-measure the bar it is read against.
$(ABL_FLOOR): $(FEATURES) $(call sections,load splits model) $(COST_MATRIX) \
              src/fraud_engine/evaluation/tracking.py \
              src/fraud_engine/models/floor.py \
              src/fraud_engine/models/ablation.py \
              src/fraud_engine/models/train.py \
              src/fraud_engine/features/registry.py \
              src/fraud_engine/evaluation/report.py | $(REPORTS_DIR)
	$(RUN) python -m fraud_engine.models.floor

# Depends on $(INTERIM) rather than $(FEATURES): every arm rebuilds its own
# splits and matrices from the pre-split frame, so the shipped ones are neither
# read nor written here.
$(PURGE): $(INTERIM) $(call sections,load splits features model) $(COST_MATRIX) \
          src/fraud_engine/evaluation/tracking.py \
          src/fraud_engine/models/purge.py \
          src/fraud_engine/models/ablation.py \
          src/fraud_engine/models/train.py \
          src/fraud_engine/data/splits.py \
          src/fraud_engine/features/build.py \
          src/fraud_engine/evaluation/report.py | $(REPORTS_DIR) $(PREDICTIONS_DIR)
	$(RUN) python -m fraud_engine.models.purge

# Writes a verdict, never the model. Adopting the winning parameters is a
# committed edit to config.yaml, so $(MODEL) stays the one thing `make train`
# produces and the history shows when tuning changed it.
$(TUNING): $(FEATURES) $(call sections,load splits model) $(COST_MATRIX) \
           src/fraud_engine/evaluation/tracking.py \
           src/fraud_engine/models/tune.py \
           src/fraud_engine/models/train.py \
           src/fraud_engine/models/imbalance.py \
           src/fraud_engine/evaluation/report.py | $(REPORTS_DIR)
	$(RUN) python -m fraud_engine.models.tune

# Reads the shipped model's VAL-CAL scores, which `make train` writes beside it,
# so it depends on $(MODEL) rather than on the matrices.
$(CALIBRATOR): $(MODEL) $(call sections,load splits model calibration) \
               src/fraud_engine/models/calibrate.py \
               src/fraud_engine/evaluation/tracking.py | $(MODEL_DIR) $(REPORTS_DIR)
	$(RUN) python -m fraud_engine.models.calibrate

$(RELIABILITY): $(CALIBRATOR) $(call sections,load splits model calibration) \
                src/fraud_engine/evaluation/reliability.py \
                src/fraud_engine/evaluation/plots.py | $(REPORTS_DIR)
	$(RUN) python -m fraud_engine.evaluation.reliability

# Phase 06's rehearsal. Reads the out-of-fold probabilities $(CALIBRATOR) wrote,
# the rules engine's VAL-CAL scores and the amounts — never the test split.
$(REHEARSAL): $(CALIBRATOR) $(BASELINES) $(INTERIM) $(COST_MATRIX) \
              $(call sections,load splits model calibration) \
              src/fraud_engine/evaluation/policy.py \
              src/fraud_engine/evaluation/cost.py \
              src/fraud_engine/evaluation/tracking.py | $(REPORTS_DIR)
	$(RUN) python -m fraud_engine.evaluation.policy

# Same inputs as the rehearsal, costed again at every point of each cost sweep.
$(SENSITIVITY): $(CALIBRATOR) $(BASELINES) $(INTERIM) $(COST_MATRIX) \
                $(call sections,load splits model calibration) \
                src/fraud_engine/evaluation/sensitivity.py \
                src/fraud_engine/evaluation/policy.py \
                src/fraud_engine/evaluation/cost.py \
                src/fraud_engine/evaluation/plots.py \
                src/fraud_engine/evaluation/tracking.py | $(REPORTS_DIR)
	$(RUN) python -m fraud_engine.evaluation.sensitivity

# Refits E1's and E3's arms, so it depends on the stages that built their matrices
# and recorded the VAL-FIT scores each refit must reproduce.
$(USD_HALVES): $(PURGE) $(ABLATION) $(FEATURES) $(CALIBRATOR) $(BASELINES) $(INTERIM) $(COST_MATRIX) \
               $(call sections,load splits features model calibration usd_halves) \
               src/fraud_engine/models/usd_halves.py \
               src/fraud_engine/models/ablation.py \
               src/fraud_engine/models/train.py \
               src/fraud_engine/models/calibrate.py \
               src/fraud_engine/evaluation/policy.py \
               src/fraud_engine/evaluation/cost.py \
               src/fraud_engine/evaluation/tracking.py | $(REPORTS_DIR)
	$(RUN) python -m fraud_engine.models.usd_halves

# The one touch of test. If any prerequisite changes after it has run, make will
# try again and the stage will refuse: the record's existence is the guard.
$(HEADLINE): $(MODEL) $(CALIBRATOR) $(BASELINES) $(FEATURES) $(SPLITS) $(INTERIM) $(COST_MATRIX) \
             $(call sections,load splits baselines model calibration) \
             src/fraud_engine/evaluation/headline.py \
             src/fraud_engine/evaluation/policy.py \
             src/fraud_engine/evaluation/cost.py \
             src/fraud_engine/evaluation/reproduce.py \
             src/fraud_engine/evaluation/plots.py \
             src/fraud_engine/models/rules.py \
             src/fraud_engine/models/calibrate.py \
             src/fraud_engine/models/train.py \
             src/fraud_engine/evaluation/tracking.py | $(REPORTS_DIR) $(PREDICTIONS_DIR)
	$(RUN) python -m fraud_engine.evaluation.headline

# Phase 07. Explains the shipped booster and changes nothing about it, so it hangs
# off $(MODEL) rather than off anything the headline produced. $(SHAP_GLOBAL) stands
# for the pair, per the single-sentinel note above: the same run writes one
# contributions parquet per explained split under $(EXPLAIN_DIR).
$(SHAP_GLOBAL): $(MODEL) $(FEATURES) $(call sections,load splits model explain) \
                src/fraud_engine/explain/contributions.py \
                src/fraud_engine/features/registry.py \
                src/fraud_engine/models/train.py \
                src/fraud_engine/evaluation/reproduce.py \
                src/fraud_engine/evaluation/tracking.py | $(REPORTS_DIR)
	$(RUN) python -m fraud_engine.explain.contributions

# Reads the contributions the explain stage persisted and the headline's own recorded
# probabilities — never a model, per the $(FIGURES) precedent. $(EXPLAIN_FIGURE) stands
# for the six figures one run draws.
#
# $(HEADLINE) is deliberately NOT a prerequisite, though this stage reads what that run
# wrote. It is the one guarded target here: it refuses to rerun while its record exists,
# so any change to a file it depends on leaves it permanently stale, and a stage naming
# it inherits a prerequisite make will retry forever and never satisfy. The stage checks
# for the predictions itself and says what to run.
$(EXPLAIN_FIGURE): $(SHAP_GLOBAL) $(FEATURES) $(COST_MATRIX) \
                   $(call sections,load splits model explain) \
                   src/fraud_engine/explain/figures.py \
                   src/fraud_engine/explain/plots.py \
                   src/fraud_engine/evaluation/plots.py \
                   src/fraud_engine/evaluation/cost.py | $(REPORTS_DIR)
	$(RUN) python -m fraud_engine.explain.figures

# Like the figures, this reads the persisted contributions and the headline's recorded
# probabilities, and $(HEADLINE) is deliberately not a prerequisite for the same reason.
# The dictionary is a plain file prerequisite: it is small, read whole, and editing a
# sentence has to re-measure how many declines it covers.
$(REASON_CODES): $(SHAP_GLOBAL) $(FEATURES) $(COST_MATRIX) $(REASON_DICT) \
                 $(call sections,load splits model explain) \
                 src/fraud_engine/explain/codes.py \
                 src/fraud_engine/features/registry.py \
                 src/fraud_engine/evaluation/cost.py | $(REPORTS_DIR)
	$(RUN) python -m fraud_engine.explain.codes

# Separate from $(SHAP_GLOBAL) for the reason $(FAMILY_FLOOR) is separate from
# $(FAMILIES): it answers a question the contributions do not change, and folding a
# measurement of seconds into a stage of hours means it can never be rerun alone.
$(EXPLAIN_LATENCY): $(MODEL) $(FEATURES) $(call sections,load splits model explain_latency) \
                    src/fraud_engine/explain/latency.py \
                    src/fraud_engine/models/train.py \
                    src/fraud_engine/evaluation/tracking.py | $(REPORTS_DIR)
	$(RUN) python -m fraud_engine.explain.latency

# Forces the check the stamp normally lets make skip. `make data` already
# verifies whenever raw/ changed; this is for re-checking on demand — after a
# disk scare, or before trusting a number you are about to publish.
.PHONY: verify-data
verify-data:  ## Re-check raw/ against docs/raw_checksums.txt, ignoring the stamp
	rm -f $(VERIFIED)
	$(MAKE) --no-print-directory $(VERIFIED)

.PHONY: data splits baselines figures features families floor train spread imbalance tune ablation ablation-floor purge calibrate rehearsal sensitivity usd-halves headline explain latency explain-figures reason-codes
data:      $(INTERIM)   ## Build interim/transactions.parquet from raw CSVs
splits:    $(SPLITS)    ## Assign transactions to temporal splits
baselines: $(BASELINES) $(LOGISTIC) $(FIGURES) ## Score both baselines through the Phase 02 harness
figures:   $(FIGURES)   ## Redraw the baseline comparison figures from predictions
features:  $(FEATURES)  ## Build train/val/test feature matrices
families:  $(FAMILIES)  ## Score each feature family on VAL-FIT
floor:     $(FAMILY_FLOOR) ## Re-measure how far chance alone moves the metric
train:     $(MODEL)     ## Train the model
spread:    $(SEED_SPREAD) ## Measure how far one configuration moves on seed alone
imbalance: $(IMBALANCE)   ## E2: none vs class weighting vs SMOTE, on LightGBM
ablation:  $(ABLATION)    ## E4: what each feature family costs when removed
ablation-floor: $(ABL_FLOOR)  ## E4: what removing that many arbitrary columns costs
purge:     $(PURGE)       ## E1: what the label-maturity gap costs
tune:      $(TUNING)      ## Search hyperparameters and judge the winner by E6
calibrate: $(CALIBRATOR) $(RELIABILITY) ## Fit the calibrator on VAL-CAL and draw its reliability diagram
rehearsal: $(REHEARSAL) ## Cost rules, naive and EV policies on VAL-CAL, before the test touch
sensitivity: $(SENSITIVITY) ## Sweep each cost assumption and chart the false-positive cost
usd-halves: $(USD_HALVES) ## E1 and E3 in USD: refit, verify, calibrate and cost each arm
headline:  $(HEADLINE)   ## The one test touch: rules, naive and EV in USD, frozen policy
explain:   $(SHAP_GLOBAL) ## Contributions for the shipped booster, and the global ranking
latency:   $(EXPLAIN_LATENCY) ## Time one row scored against the same row explained
explain-figures: $(EXPLAIN_FIGURE) ## Draw the beeswarm, the tier ranking and the waterfalls
reason-codes: $(REASON_CODES) ## Code every declined transaction and measure what covers them

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
