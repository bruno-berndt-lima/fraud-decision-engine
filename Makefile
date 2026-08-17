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
DATA_DIR     := data
RAW_DIR      := $(DATA_DIR)/raw
INTERIM_DIR  := $(DATA_DIR)/interim
SPLITS_DIR   := $(DATA_DIR)/splits
FEATURES_DIR := $(DATA_DIR)/features
MODEL_DIR    := models
REPORTS_DIR  := reports
CONFIG_DIR   := config

# ---- Config (stage inputs: editing these should trigger a rebuild) -----------
CONFIG      := $(CONFIG_DIR)/config.yaml
COST_MATRIX := $(CONFIG_DIR)/cost_matrix.yaml

# ---- Stage outputs -----------------------------------------------------------
# Several stages write more than one file — splits.py produces train/val_fit/
# val_cal/test. Make cannot express "one recipe, many outputs" before v4.3
# (grouped targets, `&:`), and this machine has 3.81. So each stage is
# represented below by a single file: if that file is up to date, the stage
# is assumed to have run.
RAW_TXN   := $(RAW_DIR)/train_transaction.csv
RAW_ID    := $(RAW_DIR)/train_identity.csv
INTERIM   := $(INTERIM_DIR)/transactions.parquet
SPLITS    := $(SPLITS_DIR)/train_ids.parquet
FEATURES  := $(FEATURES_DIR)/train.parquet
MODEL     := $(MODEL_DIR)/model.pkl

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
$(INTERIM_DIR) $(SPLITS_DIR) $(FEATURES_DIR) $(MODEL_DIR) $(REPORTS_DIR):
	mkdir -p $@

.PHONY: download
download:  ## Fetch the IEEE-CIS CSVs from Kaggle into data/raw/
	$(KAGGLE) competitions download -c ieee-fraud-detection -p $(RAW_DIR)
	cd $(RAW_DIR) && unzip -o ieee-fraud-detection.zip

$(INTERIM): $(RAW_TXN) $(RAW_ID) $(CONFIG) \
            src/fraud_engine/data/load.py src/fraud_engine/data/validate.py \
            | $(INTERIM_DIR)
	$(RUN) python -m fraud_engine.data.load

$(SPLITS): $(INTERIM) $(CONFIG) src/fraud_engine/data/splits.py | $(SPLITS_DIR)
	$(RUN) python -m fraud_engine.data.splits

$(FEATURES): $(SPLITS) $(CONFIG) src/fraud_engine/features/build.py | $(FEATURES_DIR)
	$(RUN) python -m fraud_engine.features.build

$(MODEL): $(FEATURES) $(CONFIG) src/fraud_engine/models/train.py | $(MODEL_DIR)
	$(RUN) python -m fraud_engine.models.train

.PHONY: data features train
data:     $(INTERIM)   ## Build interim/transactions.parquet from raw CSVs
features: $(FEATURES)  ## Build train/val/test feature matrices
train:    $(MODEL)     ## Train the model

# ==============================================================================
# Housekeeping
# ==============================================================================
.PHONY: clean
clean:  ## Remove derived data, models and caches (never raw/ or reports/)
	find $(INTERIM_DIR) $(SPLITS_DIR) $(FEATURES_DIR) \
	-type f ! -name '.gitkeep' -delete
	rm -rf $(MODEL_DIR)
	find src tests -type d -name '__pycache__' -prune -exec rm -rf {} +
	rm -rf .pytest_cache .ruff_cache
