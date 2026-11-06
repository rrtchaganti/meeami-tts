# Makefile for meeami-tts (docs/PLAN.md Section 10). Targets map to the laptop tasks T1-T14.
SHELL := /bin/bash
CONDA ?= conda
TTS_RUN := $(CONDA) run --no-capture-output -n tts
MFA_RUN := $(CONDA) run --no-capture-output -n mfa

STUBS := m0-demo smoke \
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

# --- T3: assets, normalizer, lexicon (PLAN Sections 2, 3) ------------------------
.PHONY: assets lexicon
assets:   # pinned MFA models (english_us_arpa) + NLTK corpora for g2p_en, sha256-verified
	$(TTS_RUN) python -m tts.assets --config configs/base.yaml

lexicon: assets
	$(TTS_RUN) python -m tts.text.lexicon --config configs/splits.yaml $(ARGS)

# --- T4: LJ alignment and phone stats (PLAN Sections 3, 5) -----------------------
.PHONY: align-lj stats
align-lj: lexicon   # CPU, ~tens of minutes; writes reports/align_lj.md and filled lj_* manifests
	$(TTS_RUN) python -m tts.align.lj --config configs/splits.yaml $(ARGS)

stats:
	$(TTS_RUN) python -m tts.stats --config configs/base.yaml $(ARGS)

# --- T5: L2 dev alignment and MFA-vs-manual evaluation (PLAN Sections 3, 7) ------
.PHONY: align-l2-dev align-dev-eval
align-l2-dev: lexicon   # all dev scripted utterances, speaker adaptation on; fills l2_dev.jsonl
	$(TTS_RUN) python -m tts.align.l2 --config configs/splits.yaml --stage all $(ARGS)

align-dev-eval:         # re-parse + compare with manual annotation -> reports/mfa_l2_dev.md
	$(TTS_RUN) python -m tts.align.l2 --config configs/splits.yaml --stage parse $(ARGS)

# --- not implemented yet (T6-T14) -------------------------------------------
$(STUBS):
	@echo "make $@: not implemented yet"
