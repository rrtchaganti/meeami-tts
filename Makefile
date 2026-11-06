# Makefile for meeami-tts (docs/PLAN.md Section 10). Targets map to the laptop tasks T1-T14.
SHELL := /bin/bash
CONDA ?= conda
TTS_RUN := $(CONDA) run --no-capture-output -n tts
MFA_RUN := $(CONDA) run --no-capture-output -n mfa

STUBS := edge eval eval-all tune profile bundle check-data demo

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

# --- T9: pace augmentation A/B + UTMOS filter (PLAN Sections 6, 7) ---------------
.PHONY: aug-a pace-stats-b aug-b assets-eval
assets-eval:   # pinned Section 8 models (faster-whisper large-v3, ECAPA, accent-id, UTMOS)
	$(TTS_RUN) python -m tts.eval.assets --config configs/base.yaml

aug-a:
	$(TTS_RUN) python -m tts.augment.generate --kind a $(ARGS)
	$(TTS_RUN) python -m tts.augment.verify --manifest data/manifests/aug_a_all.jsonl --n 100
	$(TTS_RUN) python -m tts.augment.filter --kind a

pace-stats-b:  # needs MFA dev alignments (make align-l2-dev); always on the laptop
	$(TTS_RUN) python -m tts.augment.pace_b --config configs/splits.yaml

aug-b:
	$(TTS_RUN) python -m tts.augment.generate --kind b $(ARGS)
	$(TTS_RUN) python -m tts.augment.verify --manifest data/manifests/aug_b_all.jsonl --n 100
	$(TTS_RUN) python -m tts.augment.filter --kind b

# --- T6: M0 demo on 5 dev references ---------------------------------------------------
.PHONY: m0-demo
m0-demo:
	$(TTS_RUN) python -m tts.infer.batch --system m0 --split dev --n 5 --out out/m0_demo

# --- T8: smoke runs (every config 200 steps at batch 4, resume, benchmark) -----------
.PHONY: smoke
smoke:
	$(TTS_RUN) python -m tts.smoke

# --- not implemented yet (T10-T14) -------------------------------------------
$(STUBS):
	@echo "make $@: not implemented yet"

# --- T7: one-utterance overfit check (~500 steps, batch 1) ---------------------------
.PHONY: overfit
overfit:
	$(TTS_RUN) python scripts/train.py --config configs/m1.yaml --tag overfit --max_steps 500 \
	  --batch_size 1 train.train_manifests=[data/manifests/overfit.jsonl] \
	  train.val_manifest=data/manifests/overfit.jsonl train.n_val_audio=1 train.val_interval=250 \
	  train.ckpt_interval=500 train.log_interval=50 train.num_workers=1 $(ARGS)
	$(TTS_RUN) python -m tts.models.vits_mfa.overfit_check --run runs/overfit
