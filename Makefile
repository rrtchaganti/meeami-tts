# Makefile for meeami-tts (docs/PLAN.md Section 10). Targets map to the laptop tasks T1-T14.
SHELL := /bin/bash
CONDA ?= conda
# env names: override if `tts`/`mfa` are taken on your machine (make env TTS_ENV=x MFA_ENV=y, or export them);
# MEEAMI_MFA_ENV carries the MFA env name to the Python code (configs/base.yaml mfa.conda_env)
TTS_ENV ?= tts
MFA_ENV ?= mfa
export MEEAMI_MFA_ENV := $(MFA_ENV)
TTS_RUN := $(CONDA) run --no-capture-output -n $(TTS_ENV)
MFA_RUN := $(CONDA) run --no-capture-output -n $(MFA_ENV)

.PHONY: help env env-tts env-mfa env-export test

help:
	@echo "Laptop:  env test data lexicon align-lj stats align-l2-dev align-dev-eval pace-stats-b"
	@echo "         aug-a aug-b aug-filter m0-demo overfit smoke edge eval eval-all tune phone-dyn paired profile"
	@echo "         bundle check-data demo  |  inference/demo: models infer live-demo rehearse text-stress web-demo (docs/DEMO.md)"
	@echo "GPU:     see docs/HANDOFF.md (setup_gpu.sh, fetch_bundle.sh, check-data, train.py)"

# --- T1: environments -------------------------------------------------------
# Creates each env if missing, otherwise updates it in place (safe to re-run).
env: env-tts env-mfa

env-tts:
	@if $(CONDA) env list | awk '{print $$1}' | grep -qx $(TTS_ENV); then \
		$(CONDA) env update -n $(TTS_ENV) -f envs/tts.yml --prune; \
	else \
		$(CONDA) env create -n $(TTS_ENV) -f envs/tts.yml; \
	fi

env-mfa:
	@if $(CONDA) env list | awk '{print $$1}' | grep -qx $(MFA_ENV); then \
		$(CONDA) env update -n $(MFA_ENV) -f envs/mfa.yml --prune; \
	else \
		$(CONDA) env create -n $(MFA_ENV) -f envs/mfa.yml; \
	fi

# Full exports of the installed envs, for reference when pinning envs/*.yml.
env-export:
	@mkdir -p envs/lock
	$(CONDA) env export -n $(TTS_ENV) --no-builds > envs/lock/tts.full.yml
	$(CONDA) env export -n $(MFA_ENV) --no-builds > envs/lock/mfa.full.yml

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

# --- D-90: V0 (original pretrained VITS, MAS + SDP) demo on 5 dev references ---------------
.PHONY: v0-demo
v0-demo:
	$(TTS_RUN) python -m tts.infer.batch --system v0 --split dev --n 5 --out out/v0_demo

# --- T8: smoke runs (every config 200 steps at batch 4, resume, benchmark) -----------
.PHONY: smoke
smoke:
	$(TTS_RUN) python -m tts.smoke

# --- T10: edge-case suite (PLAN Section 9) ------------------------------------------
# CKPT=m0 (default) or a VITS-MFA checkpoint; DEVICE=cuda|cpu. `make edge-build` regenerates
# tests/edge_cases/ from dev-speaker audio.
CKPT ?= m0
DEVICE ?= cuda
.PHONY: edge edge-build
edge:
	$(TTS_RUN) python -m tts.infer.edge_cases --run --ckpt $(CKPT) --device $(DEVICE)

edge-build:
	$(TTS_RUN) python -m tts.infer.edge_cases --build

# UTMOS filter only (needs the torch.hub SpeechMOS model; see docs/HANDOFF.md)
.PHONY: aug-filter
aug-filter:
	$(TTS_RUN) python -m tts.augment.filter --kind a
	$(TTS_RUN) python -m tts.augment.filter --kind b

# --- T11: evaluation (PLAN Section 8) ------------------------------------------------
# make eval SYSTEM=m0 SPLIT=dev ; SYSTEM=m1 uses export/m1/ckpt_last.pt unless CKPT_EVAL is set.
SYSTEM ?= m0
SPLIT ?= dev
CKPT_EVAL ?=
MAIN ?= m1_pos_diph
# a system's checkpoint: the full training file where it exists (the owner's laptop), otherwise the
# inference-only export that `make models` downloads (a fresh clone; D-95)
ckpt_of = $(firstword $(wildcard export/$(1)/ckpt_last.pt) export/$(1)/infer.pt)
SYSTEMS ?= m0 m1 m1_pos m1_pos_diph abl_a_pos abl_b_pos   # final set (D-85); V0 (D-90): make eval SYSTEM=v0
.PHONY: eval eval-all tune
eval:
	$(TTS_RUN) python scripts/evaluate.py --system $(SYSTEM) --split $(SPLIT) \
	  $(if $(CKPT_EVAL),--ckpt $(CKPT_EVAL),) $(ARGS)

eval-all:
	@for s in $(SYSTEMS); do \
	  if [ "$$s" = m0 ] || [ "$$s" = v0 ] || [ -f export/$$s/ckpt_last.pt ] || [ -f export/$$s/infer.pt ]; then \
	    $(TTS_RUN) python scripts/evaluate.py --system $$s --split $(SPLIT) $(ARGS) || exit 1; \
	  else echo "eval-all: skipping $$s (no checkpoint in export/$$s; make models)"; fi; \
	done

PD_CKPTS ?= $(call ckpt_of,$(MAIN))
.PHONY: phone-dyn
phone-dyn:   # within-phone spectral movement vs LJ (D-76); PD_CKPTS="export/m1_pos/ckpt_last.pt ..."
	$(TTS_RUN) python -m tts.eval.phone_dynamics $(foreach c,$(PD_CKPTS),--ckpt $(c)) --device $(DEVICE)

paired:   # paired Wilcoxon vs a control (D-82); CONTROL=m1_pos SYSTEMS_CMP="abl_a_pos abl_b_pos"
	$(TTS_RUN) python -m tts.eval.paired --control $(CONTROL) $(foreach s,$(SYSTEMS_CMP),--system $(s)) --split $(SPLIT)

FID_SYSTEMS ?= v0 _rejected/v0_blank_after m0 m1 m1_pos_diph
.PHONY: fidelity
fidelity:   # duration fidelity (output vs intended word times) + aligner calibration on dev (D-90)
	$(TTS_RUN) python -m tts.eval.fidelity $(foreach s,$(FID_SYSTEMS),--system $(s)) --split $(SPLIT) \
	  $(if $(filter dev,$(SPLIT)),--calibrate,) --device $(DEVICE)

tune:     # dev only: grid over cap x noise_scale (+ transfer variants); writes configs/inference.yaml
	$(TTS_RUN) python -m tts.eval.tune --split $(SPLIT) $(ARGS)

# --- T12: profiling ----------------------------------------------------------------------
CKPT_PROF ?= $(call ckpt_of,$(MAIN))
.PHONY: profile
profile:
	$(TTS_RUN) python scripts/profile.py --ckpt $(if $(filter-out m0,$(CKPT)),$(CKPT),$(CKPT_PROF)) --device $(DEVICE) $(ARGS)

# --- T13: GPU bundle -------------------------------------------------------------------
.PHONY: bundle bundle-m1 lj-download check-data
BUNDLE_AUG ?= 1
bundle:   # BUNDLE_AUG=0: light bundle without augmented audio (regenerate on the instance)
	CONDA=$(CONDA) BUNDLE_AUG=$(BUNDLE_AUG) bash scripts/bundle.sh export/bundle.tar

bundle-m1:   # M1 only, small: no augmented audio, no LJ wavs (instance: make lj-download) (D-69)
	CONDA=$(CONDA) BUNDLE_AUG=0 BUNDLE_LJWAV=0 bash scripts/bundle.sh export/bundle_m1.tar

lj-download:   # on the GPU instance: official LJSpeech 1.1 archive -> data/raw/LJSpeech-1.1
	$(TTS_RUN) python -m tts.fetch_lj --config configs/base.yaml $(ARGS)

check-data:
	$(TTS_RUN) python -m tts.check_data --root . $(ARGS)

# --- D-94: text-input stress cases (read-only; TEXT="..." for ad-hoc input) ---------------
.PHONY: text-stress
text-stress:
	$(TTS_RUN) python -m tts.text.stress $(if $(TEXT),--text "$(TEXT)" --phones,) $(ARGS)

# --- D-95: hosted checkpoints (Hugging Face) -------------------------------------------------
.PHONY: models export-infer publish-models outputs publish-outputs samples
models:          # download export/<system>/infer.pt + baseline weights, sha256-checked
	$(TTS_RUN) python scripts/fetch_models.py
export-infer:    # owner: export/<system>/ckpt_last.pt -> infer.pt (generator only), verified
	$(TTS_RUN) python scripts/export_infer.py --verify $(foreach n,m1_pos_diph m1 m1_pos abl_a_pos abl_b_pos,--name $(n))
publish-models:  # owner: upload to hosting.hf_repo (configs/base.yaml)
	$(TTS_RUN) python scripts/fetch_models.py --publish
outputs:         # download every evaluated test output (hosting.outputs) into results/<system>/test/
	$(TTS_RUN) python scripts/publish_outputs.py
publish-outputs: # owner: tar results/<system>/test/ and upload them (E-36)
	$(TTS_RUN) python scripts/publish_outputs.py --publish
samples:         # owner: rebuild samples/ (exact evaluated outputs + references + word timings, E-36)
	$(TTS_RUN) python scripts/export_samples.py

# --- inference on any reference: make infer REF=their.wav TEXT="what it says" [SYS=m1_pos_diph|v0|m0]
SYS ?= $(MAIN)
OUT ?= out/infer
.PHONY: infer
infer:
	@test -n "$(REF)" -a -n "$(TEXT)" || { echo 'usage: make infer REF=their.wav TEXT="what the recording says" [SYS=v0|m0|<system>] [OUT=dir]'; exit 2; }
	$(TTS_RUN) python scripts/infer.py --text "$(TEXT)" --ref "$(REF)" --out $(OUT) \
	  --ckpt $(if $(filter v0 m0,$(SYS)),$(SYS),$(call ckpt_of,$(SYS))) $(ARGS)

# --- D-96: live interview demo ---------------------------------------------------------------
.PHONY: live-demo rehearse
live-demo:       # UTT=HKK_arctic_a0003 [TEXT="..."] or REF=their.wav TEXT="..."
	$(TTS_RUN) python scripts/demo.py $(if $(UTT),--utt $(UTT),) $(if $(REF),--ref "$(REF)",) $(if $(TEXT),--text "$(TEXT)",) $(ARGS)
rehearse:        # every case in configs/live_demo.yaml -> out/live_demo/REHEARSAL.md
	$(TTS_RUN) python scripts/demo.py --suite $(ARGS)

# --- D-101: browser demo (record/upload + text -> every system); share via an HTTPS tunnel ------
.PHONY: web-demo web-tunnel
web-demo:        # http://localhost:7860/?key=<printed>; MEEAMI_DEMO_KEY=... fixes the key
	$(TTS_RUN) python scripts/web_demo.py $(ARGS)
WEB_PORT ?= 7860
web-tunnel:      # second terminal: public https URL for the running web-demo (cloudflared quick tunnel)
	@command -v cloudflared >/dev/null || { echo "cloudflared not found; install: docs/DEMO.md §6"; exit 2; }
	cloudflared tunnel --url http://localhost:$(WEB_PORT)

# --- T14: demo page ----------------------------------------------------------------------
.PHONY: demo
demo:
	$(TTS_RUN) python scripts/make_demo.py $(ARGS)

# --- T7: one-utterance overfit check (~500 steps, batch 1) ---------------------------
.PHONY: overfit
overfit:
	$(TTS_RUN) python scripts/train.py --config configs/m1.yaml --tag overfit --max_steps 500 \
	  --batch_size 1 train.train_manifests=[data/manifests/overfit.jsonl] \
	  train.val_manifest=data/manifests/overfit.jsonl train.n_val_audio=1 train.val_interval=250 \
	  train.ckpt_interval=500 train.log_interval=50 train.num_workers=1 $(ARGS)
	$(TTS_RUN) python -m tts.models.vits_mfa.overfit_check --run runs/overfit
