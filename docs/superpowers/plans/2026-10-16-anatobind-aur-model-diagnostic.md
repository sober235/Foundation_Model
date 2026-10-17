# AUR Model Diagnostic Implementation Plan

> **For agentic workers:** Use superpowers:executing-plans inline, with an independent whole-change review. The user approved continuing the reviewed design on 2026-10-16.

**Goal:** Reproduce six frozen validation cases and locate failures affecting A anatomy, U abnormality instances, and R per-instance host binding.

**Architecture:** Optional inference observers leave the original arithmetic unchanged. A separate diagnostic module selects validation cases, measures natural-window query/presence/aggregation boundaries and relation inputs, and refuses attribution unless saved detections reproduce. No training or model repair is included.

**Tech Stack:** Existing Python/PyTorch/numpy/scipy/nibabel/pytest, one idle CUDA GPU.

**Spec:** `docs/superpowers/specs/2026-10-16-anatobind-aur-modeling-next-step-design.md` (approved by the user's continuation).

## Global Constraints

- Preserve A32 masks, unified abnormality instance masks U, and R13 + no-host. Existing gates unchanged; all results NOT_EVIDENCE.
- Validation only, two distinct patients for BMSR/T1c, ISLES/DWI, PDGM/FLAIR using frozen SHA256 ordering. No test inference or threshold search.
- Fixed evaluated checkpoint/data hashes, crop [128,160,160], original bf16 and batch size 2, presence >0.5, mask >=0.3; >0.5 separately labeled.
- Baseline detection count/boxes/voxel counts exact; volume atol 1e-3 mm³ and score atol 1e-5. Mismatch stops attribution.
- One idle GPU, first-case timing and peak memory, six-case replay at most 7200 seconds. Stream local arrays. No identities or private case-level files in commits.

## Review Focus

1. Observer enabled must return identical predictions to observer disabled, including padding and no-present-query windows.
2. Natural-window local IoU excludes voxels outside the window and uses the same mask threshold across compared stages.
3. Selection rejects test members/duplicate identities and enforces distinct patients per source.
4. Reproduction failures remain failures; no tolerance relaxation, partial success or oracle results replacing official scores.
5. R diagnostics describe actual coarse inputs, while A scores and GT-only oracle quantities remain distinguished.

### Task 1: Non-mutating observations and diagnostic primitives

**Files:** modify `anatobind/aur/infer.py`; create `anatobind/aur/diagnostics.py`, `tests/test_aur_diagnostics.py`.

**Interfaces:** `predict_volume(..., observer=None)` calls the observer once per natural window with window, event_prob, event_presence, entity_prob, valid, weight. `bind_instances(..., observer=None)` calls it once per GT-centred bind with model/out/target/query/window/result. `identity(row)`, `select_cases(val_rows, saved_cases, per_source=2)`, `verify_detections(actual, expected)` and `WindowAudit` are consumed by Task 2.

- [x] Write failing tests: exact observer parity on a tiny real model, bind-observer parity, deterministic distinct-patient selection, no test membership, missing/mismatched detections rejected, a useful low-presence query separated from retained max, valid/padding excluded.
- [x] Run targeted CPU tests and observe the intended failures.
- [x] Implement minimal hooks and diagnostics, retaining original computation and instance identity.
- [x] Run diagnostics/inference tests; commit.

### Task 2: Frozen replay runner

**Files:** create `scripts/aur_diagnose_model.py`; extend diagnostics tests.

**Interfaces:** CLI `--samples --val-patients --saved-eval --checkpoint --out --gpu --max-seconds` creates a new local run; freezes selection and provenance before inference; writes anonymous progress/summary and local-only arrays/selection. It loads existing exported model and dataset paths without modifying them.

- [x] Write failing tests for provenance/selection and baseline mismatch behavior (reuse Task 1 primitives; no GPU in unit tests).
- [x] Implement runner: verify hashes, select/freeze six rows, run original predict with observers, verify detections before interpretation, collect A inside/outside lesion and actual coarse R inputs, stream local arrays and per-window stats.
- [x] Run targeted then full CPU suite, inspect diff, commit. Independent reviewer checks Tasks 1–2 before GPU replay.

### Task 3: Execute and report the bounded validation diagnostic

**Files:** anonymous report under `docs/verification/2026-10-16/aur_model_diagnostic/`; update STATUS. Local raw run under `~/outputs/aur_model_diagnostic/`.

- [x] Confirm idle GPU, pin it, record first-case wall time and peak memory; run sequential six-case replay under 7200-second timeout.
- [x] Verify all frozen cases reproduce saved baseline or explicitly stop attribution at mismatch; do not alter tolerances.
- [x] Combine U window measurements, A breakdown and saved-val R stratification; identify supported/unsupported/unresolved mechanisms.
- [x] State the next single-variable model experiment supported by evidence (or the smallest missing observation), preserving all three goals. Do not start new training or activate N7 fallback in this plan.
- [x] Review anonymous output and final code, run checks appropriate to any fixes, commit completed report/status.

## Execution record

The per-task ledger is `.superpowers/sdd/2026-10-16-anatobind-aur-model-diagnostic/progress.md`. Task 3 is not complete merely because tooling passes tests; actual replay and an honest report are required.

Completed 2026-10-17: tooling reviewed,1065 passed/2 skipped; six-case replay and A/U/R cache analysis complete. NEXT_EXPERIMENT is a proposed separate decoder study, not executed.
