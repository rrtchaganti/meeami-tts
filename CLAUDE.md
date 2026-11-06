# CLAUDE.md
Read the docs/PLAN.md sections named in the task before writing code.
- Test speakers (ABA SKA YBAA ZHAA HJK HKK YDCK YKWK HQTV PNV THV TLV): never read their audio
  unless git tag freeze-v1 exists.
- data/lexicon/lexicon.txt is the only source of pronunciations.
- Two conda envs: `mfa` (MFA only, called via `conda run -n mfa`) and `tts` (everything else).
- All parameters live in configs/*.yaml (inheriting base.yaml); no hard-coded paths or hyperparameters.
- Every module gets a unit test in tests/ and a make target. Run `make test` before calling a task done.
- Laptop GPU has 8 GB: smoke runs use batch 4.
- After tag freeze-v1, never change model code or experiment configs.
- If a pinned package or model is unavailable, stop and report; never substitute silently.
- End each task with: files changed, commands run, acceptance-check result.
