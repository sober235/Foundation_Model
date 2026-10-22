# LR5 Supervised Capability Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox syntax for tracking.

**Goal:** Execute one bounded, TRAIN-only full-model II→III capability check from the verified LR5 Stage I backbone, reporting each mode's A/U/R learning and fixed-readout retention.

**Architecture:** Two external scripts reuse the unchanged frozen LR5 runtime, strict UnifiedRun initialization and qualified data loader. One implements private fixed-panel support/metrics/retention arithmetic; the other freezes registration and sources, executes bounded stages and saves anonymous aggregate outcomes without formal exports.

**Tech Stack:** Existing nvgen/PyTorch/NumPy/scipy/pytest; no new dependencies.

**Spec:** `docs/superpowers/specs/2026-10-22-lr5-supervised-capability-design.md`, refining the previously approved core-goal route; the user explicitly requested continued execution.

## Global Constraints

- Keep original data/runtime/G1 hashes immutable; use the exact execution_v5 runtime and separately hash external controllers.
- TRAIN only, unchanged model/loss/geometry; same full crops, points and alternating modes; no PLANAR or SSL LR multiplier in supervision.
- Label-only panel selection:64 candidates/mode,16 selected maximum; A32 coverage,U≥4,R≥2 hosts,3 sources,valid background per mode, otherwise INSUFFICIENT.
- II/III each512 steps and warmup32; II rates5e-5/5e-4, no backbone rate release; III rates1e-5/2.5e-4/5e-4.
- Measure steps0/128/256/512, save every64, one complete-state recovery at64/stage; no best selection. Scan1800s,stage3600s,total10800s hard limits.
- All arrays and per-observation material private outsideGit; stdout only allowlisted anonymous summaries. NOT_EVIDENCE, no formal certificates, heldout use or automatic next-stage release.
- CPU CUDA hidden,nice19,threads2; GPU visibility/scheduler constraints resolved to physicalUUID and checked before selection; preserve other workloads.

## Review Focus

- Role or shape errors must stop before model use; missing32-class or real2D R support must be INSUFFICIENT, never an empty-success average.
- Unknown R and host0 must retain different meanings; controlled query match counts cannot be mislabeled end-to-end detection performance.
- Fixed readout must fit once at baseline, use frozen normalization/class support and score only its observation parity; later measurement cannot refit or alter training RNG.
- Runtime/parent/controller changes must invalidate admission and completion; a resource checkpoint cannot masquerade as a formal supervised export.
- Timeout, parent death, exceptions, existing-output conflicts and malformed arguments must leave existing evidence intact and never print private values.

### Task 1: Fixed-panel support, prediction metrics and frozen retention

**Files:** Create `scripts/aur_capability_metrics.py`; test `tests/test_aur_capability_metrics.py`.

**Interfaces:** `batch_support(batch)->dict`; `select_panel(supports,max_observations=16)->list[int]`; `panel_support(supports)->dict`; `measure_batch(model,batch)->dict` with aggregate counters plus private cell features/labels; `aggregate_measurements(records)->dict`; `fit_retention(records)->dict`; `score_retention(records,readout)->dict`; `capability_verdict(initial,stage_two,stage_three,retention_baseline,retention_final)->dict`.

- [x] Write/run failing tests for nonignored A32 support, mixed R eligibility/host0/no-host, U/background support, greedy tie ordering and coverage refusal.
- [x] Implement deterministic support/selection using existing target tensors; test hand-derived mask/presence/R metrics on a controlled model and both true input modes.
- [x] Test fit-once ridge retention with held parity, frozen normalization, absent-class refusal and a deliberately permuted/degraded feature representation; implement helpers.
- [x] Test per-mode verdict thresholds and INSUFFICIENT/nonfinite paths with literal counters; run focused suite and commit owned files.

### Task 2: Registered bounded controller and strict inherited runtime

**Files:** Create `scripts/aur_supervised_capability.py`; test `tests/test_aur_supervised_capability.py`.

**Interfaces:** `run_capability(registration_path,out,device='cpu')->dict`; CLI `--registration --out --device`; registration binds runtime root/hash,qualified contract,parent Stage I/G1/completion hashes,controller source hashes and both complete recipes. Task1 helpers remain external to immutable runtime.

- [x] Write/run failing synthetic tests for TRAIN-only fixed factories, immutable output/privacy, prior-stage admission, source mismatch and bounded-stop behavior.
- [x] Implement source/parent/contract verification, private source and registration snapshots, deterministic pool freeze and support verdict before any model forward.
- [x] Use UnifiedRun purpose=resource_probe after explicitly verifying the formal parent certificate and G1; preserve the inherited exact runtime/data identity. III inherits II final state.
- [x] Evaluate fixed predictions/retention at the registered boundaries with RNG restoration; log real group rates/exposures/losses, save immutable64-step states and prove one64-step destroy/resume equality perstage.
- [x] Run focused integration tests including tiny actual II→III, private failures, recovery and zero formal certificate/export creation; commit ownedfiles.

### Task 3: Independent review, real bounded run and progress publication

**Files:** private `~/outputs/aur_lr5_supervised_capability_20261022/`; public `docs/verification/2026-10-22/aur_lr5_supervised_capability/{RESULTS.md,RUN.json}`; `STATUS.md`,`CLAUDE.md`,`README.md`.

**Interfaces:** Reviewed Task1/2 scripts plus exact previous execution_v5/runtime; all public evidence is aggregated.

- [x] Self-review spec coverage and shared interfaces; independently review full exactdiff,registration/source/parent commitments and risk before compute. Resolve substantive findings within three rounds.
- [x] Run project full CPU acceptance once after code changes; baseline already running from starting clean4bd2159. Record skips as environmentgaps.
- [x] Freeze one new private version and exactregistration; check liveGPUutilization/processes/visibility,launch with harddeadline/parentdeath protection. Do not change inputs or rescue failed thresholds after observing results.
- [x] Verify actual termination,checkpoint/RNG restoration and final input/source hashes; derive finalverdict from cachedmetrics rather than rerun inference.
- [x] Write anonymous exactoutcomes/limits/resource and testrecords; update currentmain progress,review publicationdiff,commit and report completed/failed/insufficient tasks accurately.

## Execution decisions

Existing isolated main worktree is clean at4bd2159; use it in accordance with the project's current main progress convention. The user's continue instruction authorizes implementing and executing this next finite block without another generic workflow-choice question. Numerical calls and any justified interface corrections are recorded in the execution ledger. No final quality or clinical claim follows from TRAIN capability success.
