# Makefile for meeami-tts (docs/PLAN.md Section 10). Targets map to the laptop tasks T1-T14.
SHELL := /bin/bash
CONDA ?= conda
TTS_RUN := $(CONDA) run --no-capture-output -n tts
MFA_RUN := $(CONDA) run --no-capture-output -n mfa

STUBS := lexicon align-lj stats align-l2-dev align-dev-eval m0-demo smoke \
         aug-a pace-stats-b aug-b edge eval eval-all tune profile bundle check-data demo

.PHONY: help env env-tts env-mfa env-export test $(STUBS)

help:
	@echo "Targets: env test env-export $(STUBS)"

# --- T1: environments -------------------------------------------------------
# Creates each env if missing, otherwise updates it in place (safe to re-run).
env: env-tts env-mfa

env-tts:
	@if $(CONDA) env list | awk '{print $$1}' | grep -qx tts; then \
		$(CONDA) env update -n tts -f envs/tts.yml --prune; \
	else \
		$(CONDA) env create -n tts -f envs/tts.yml; \
	fi

env-mfa:
	@if $(CONDA) env list | awk '{print $$1}' | grep -qx mfa; then \
		$(CONDA) env update -n mfa -f envs/mfa.yml --prune; \
	else \
		$(CONDA) env create -n mfa -f envs/mfa.yml; \
	fi

# Full exports of the installed envs, for reference when pinning envs/*.yml.
env-export:
	@mkdir -p envs/lock
	$(CONDA) env export -n tts --no-builds > envs/lock/tts.full.yml
	$(CONDA) env export -n mfa --no-builds > envs/lock/mfa.full.yml

test:
	$(TTS_RUN) python -m pytest -q tests/

# --- T2: data and splits (PLAN Section 2) ---------------------------------------
.PHONY: data
data:
	$(TTS_RUN) python scripts/prepare_data.py --config configs/splits.yaml $(ARGS)

# --- not implemented yet (T3-T14) -------------------------------------------
$(STUBS):
	@echo "make $@: not implemented yet"
