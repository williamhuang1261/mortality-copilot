# mortality-copilot — every stage of the pipeline has a target, and the
# targets run in the order they are listed below.

SHELL   := /bin/bash
VENV    := .venv
PY      := $(VENV)/bin/python
PIP     := $(VENV)/bin/pip
RSCRIPT := Rscript

.PHONY: help setup setup-rag data features eda models sklearn-baseline index demo finetune agent-demo equipment-data equipment-features equipment-models test all clean

help:
	@echo "mortality-copilot"
	@echo ""
	@echo "  make setup      install core Python deps (venv) and R deps"
	@echo "  make setup-rag  additionally install the retrieval stack (torch, faiss)"
	@echo "  make data       download NHANES + NCHS mortality into DuckDB"
	@echo "  make features   build the analytic cohort in SQL"
	@echo "  make eda        exploratory analysis, hypothesis tests, figures"
	@echo "  make models     fit logistic / Cox / random forest, export artifacts"
	@echo "  make sklearn-baseline  scikit-learn cross-check of the R models"
	@echo "  make index      embed the NCHS corpus into a FAISS index"
	@echo "  make demo       run the copilot on one held-out case"
	@echo "  make finetune   fine-tune the retriever, report recall@5"
	@echo "  make agent-demo run a scripted multi-turn session against the tool-calling agent"
	@echo "  make equipment-data      download NASA C-MAPSS FD001 into DuckDB (second domain)"
	@echo "  make equipment-features  build the equipment health-score analytic tables"
	@echo "  make equipment-models   fit/validate equipment models, score the external holdout"
	@echo "  make test       run the test suite"
	@echo "  make all        data -> features -> eda -> models"
	@echo "  make clean      remove generated data, index and artifacts"

# ---------------------------------------------------------------- setup

setup: $(VENV)/.installed r-deps

$(VENV)/.installed: requirements.txt
	python3 -m venv $(VENV)
	$(PIP) install --quiet --upgrade pip
	$(PIP) install --quiet -r requirements.txt
	@touch $@

.PHONY: r-deps
r-deps:
	$(RSCRIPT) R/install_deps.R

setup-rag: $(VENV)/.installed
	$(PIP) install -r requirements-rag.txt

# ---------------------------------------------------------------- pipeline
# Implemented in later steps; each prints a clear message until then.

data:
	$(RSCRIPT) R/01_ingest.R
	$(PY) pipeline/01_load_duckdb.py

features:
	$(PY) pipeline/02_features.py

eda:
	$(RSCRIPT) R/03_eda.R

models:
	$(RSCRIPT) R/04_models.R
	$(RSCRIPT) R/05_export.R

sklearn-baseline:
	$(PY) pipeline/09_sklearn_baseline.py

index: setup-rag
	$(PY) pipeline/06_index.py

demo:
	$(PY) -m pipeline.copilot --case 44

finetune: setup-rag
	$(PY) pipeline/08_finetune.py

agent-demo:
	printf 'case 1\nwhat if age were 80\nnow raise hba1c by 20%%\nhow many people are in the cohort?\n' | $(PY) -m pipeline.agent

equipment-data:
	$(RSCRIPT) R/eq01_ingest.R
	$(PY) pipeline/eq_load_duckdb.py

equipment-features:
	$(PY) pipeline/eq_features.py

equipment-models:
	$(RSCRIPT) R/eq03_models.R
	$(RSCRIPT) R/eq04_export.R

all: data features eda models

# ---------------------------------------------------------------- quality

test:
	$(PY) -m pytest tests/ -q

clean:
	rm -rf data/raw data/*.duckdb data/*.csv data/equipment index/ corpus/ models/ artifacts/*.json docs/figs/*.png
