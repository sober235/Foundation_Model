# Dual-mode evaluation and frozen G1 diagnosis Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make completed unified supervised weights evaluable on both modes and explain what preserved evidence establishes about the failed 2D G1, without rerunning the failed training.

**Architecture:** Extend the existing strict unified loader, add a mode-explicit development evaluator using existing inference/matching and a separate held-out geometry-reference policy, and audit preserved checkpoints/features with no backbone forwards. No supervised unified checkpoint exists yet; synthetic end-to-end tests establish interface behavior, not medical quality.

**Tech Stack:** Existing Python/nvgen, PyTorch, NumPy, SciPy, nibabel and pytest; no new dependency.

**Spec:** `docs/superpowers/specs/2026-10-21-anatobind-core-goal-and-evidence-route-design.md`, approved by the user with “consistency, go on”. This plan implements its first step and prepares a justified candidate for step two; it does not authorize inventing a diagnosis.

## Global Constraints

- The final input scope includes standalone 2D and 3D MRI with complete A/U/R. Report coverage separately for both modes and all three tasks.
- Keep original Stage I run, selected step1792, completed2048, 2D FAIL / 3D PASS, G1 threshold0.05, patient roles and evidence immutable. No new training or G1 fitting in these tasks.
- All real arrays, checkpoint/feature derivatives and patient records stay under private home output directories outside Git; never print private manifest keys/values or case identifiers. Validate and write the same resolved path; errors must not echo private values.
- Pseudo-reference output is NOT_EVIDENCE; unavailable is not zero or no-host. Host index0 is valid; NO_HOST13 is distinct from unknown.
- CPU commands: `env CUDA_VISIBLE_DEVICES= PYTHONNOUSERSITE=1 PYTHONPATH=. OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 nice -n 19 /home/congcongliu/anaconda3/envs/nvgen/bin/python`; no GPU needed here.
- Preserve existing opt-in/default inference and TRAIN-only weak-R behavior. Formal artifact completion does not certify final quality or bypass the training gate.
- Existing untracked `2026-10-21-aur-g1-diagnosis-evaluation.md` is a predecessor draft; leave it untouched and unstaged.

## Review Focus

- Tampered or moved formal completion artifacts must reject before inference; test actual certificate verification and supplied data identity.
- Empty or unqualified U/A/R observations must remain unavailable, with explicit support; test no-U observations do not become negative controls.
- 2D area and 3D volume/FP denominators cannot mix; test unequal affine scales and normal-axis invariance.
- Score filtering precedes one-to-one matching; ignored small abnormalities cannot create hits or FP, and host0 remains evaluable.
- A diagnostic cache can be incomplete or nonfinite: report the limit, never silently substitute a refit or a model forward; pin the preserved input hashes.

## File responsibilities

- `anatobind/aur/unified_infer.py`: accept strict format2 II/III exports while retaining format1 pilot compatibility.
- `anatobind/aur/unified_evaluation.py`: frozen observations, mode-specific targets/metrics and private evaluation orchestration.
- `anatobind/aur/planar_relations.py`: extract pure geometric qualification only if needed; existing TRAIN gate remains unchanged.
- `anatobind/aur/data_contract.py`: evaluation-specific verification of qualified source snapshots while still validating current data bytes/role links; default live-source training verification remains unchanged.
- `scripts/aur_unified_eval.py`: development-only CLI with private output validation and anonymous stdout/errors.
- `tests/test_aur_unified_infer.py`, `tests/test_aur_unified_evaluation.py`: behavior and synthetic end-to-end tests.
- Private `~/outputs/aur_g1_diagnosis_20261021/v1/`: frozen diagnostic questions, source and read-only audit script/full artifacts.
- `docs/verification/2026-10-21/aur_g1_diagnosis/`: anonymous diagnostic report, hashes and limits only.

### Task 1: Strict supervised export loading

**Files:** Modify `anatobind/aur/unified_infer.py`; test `tests/test_aur_unified_infer.py`.

**Interfaces:** Consumes `verify_completion_certificate(initializer, *, data_contract)` from `unified_artifacts.py`. Produces `model_from_unified_export(path, device="cpu", *, data_contract=None) -> (AnatoBindBrain, metadata)`; metadata includes format/stage, pilot_only, phase_completed, quality_certified=False and NOT_EVIDENCE.

- [x] Write tests constructing real tiny completed format2 II/III boundaries/certificate using existing runner fixtures; load once and compare actual both-mode outputs. Reject wrong supplied contract, missing/tampered certificate, partial/nonfinite state, wrong architecture and Stage I; preserve format1 compatibility.
- [x] Run the new tests and observe failure from the missing formal-loader behavior; record command/output.
- [x] Implement strict format dispatch, certificate/identity and finite-state validation before moving the model to its device. Preserve generic error messages without artifact paths.
- [x] Run `... -m pytest tests/test_aur_unified_infer.py tests/test_aur_unified_launcher.py -q -p no:cacheprovider`; inspect actual output.
- [x] Commit only task files, report RED/GREEN and self-review to the task report; obtain fresh task-scoped review.

### Task 2: Complete mode-explicit development evaluation

**Files:** Create `anatobind/aur/unified_evaluation.py`, `scripts/aur_unified_eval.py`, `tests/test_aur_unified_evaluation.py`; modify `planar_relations.py` only for a shared pure geometry helper with regression coverage; modify `data_contract.py` for explicit evaluation source verification with focused tamper tests in `tests/test_aur_data_contract.py` (discover actual matching existing filename).

**Interfaces:** Consume Task1 loader, `read_contract`, `infer_input`, `load_unified_volume`, existing `bind_instances`, `host_targets`, `host_dice`, detection `scan_matches`. Produce `freeze_observations(contract, modes=("2d","3d")) -> plan`, `evaluate_observation(model, row, observation, *, crop, device="cpu", score_threshold=.5) -> private_record`, `aggregate_observations(records) -> anonymous_summary`, and `run_evaluation(contract_path, export_path, out_dir, *, device="cpu", modes=("2d","3d"), crops=None, score_threshold=.5) -> anonymous_summary`. Exact minor internal schemas are implementer decisions; keep each responsibility focused.

- [x] RED tests: accept only `role=development_validation`/`split=val` members from verified contract; choose one complete unaugmented central canonical plane per row from image header before target/model reads and freeze membership/config before inference. No label-selected planes, role relabeling, test reads or threshold search.
- [x] RED tests: evaluation may consume a historically qualified contract after inference code changes only by validating its immutable `execution_source` snapshot together with current role links, all source bytes and environment. Do not reuse `frozen_execution=True` as a top-level shortcut because it skips current role-link checks. Add an explicit evaluation source mode/helper; default training verification must continue checking live qualified code, and modified role/source bytes must reject in both modes. Record the new evaluator source hash separately.
- [x] Actual final_v3 has4 exact supplemental snapshot members and1 missing exact copy; all5 match training git revision65eb7df. Support explicit `qualified_source_root=None` in evaluation orchestration and CLI `--qualified-source` for a hash-verified private reconstruction from that revision. Label it audit-time reconstruction, not a historical saved snapshot; never update the old contract's outputs or hashes.
- [x] RED tests: A32/host13 Dice with explicit truth/predicted supports and A-ignore; non-supervised A/U/R unavailable. U uses 8CC/min1mm2 planes and existing 26CC/min10mm3 volumes, includes ignored undersized components and filters scores before bbox-IoU>=.1 matching. FP/image versus FP/scan, physical measures and source/patient/observation counts are distinct.
- [x] RED tests: held-out 2D geometry-only reference uses actual source/plane overlap, unambiguous visible host and source identity, via an evaluation-specific entry; TRAIN-only qualifier still rejects val. Unknown stays excluded; host0 stays valid. R reports same-observation controlled and end-to-end hits/eligible denominators and same-predicted-anatomy B0 separately, all NOT_EVIDENCE. Geometry pseudo-reference may not establish a new passing gate.
- [x] Implement using raw canonical image exactly once for inference normalization, existing discrete label decoding and real inference/binding. No new decode profile, loss or training path. Avoid using API mocks as proof of actual both-mode inference.
- [x] Add real tiny synthetic both-mode E2E including certified export and private JSON outputs. Verify argument/path errors do not echo private values; output writers use the validated normalized path outside Git. CLI exposes contract/export/out/device/modes and fixed score threshold, no test or tuning mode.
- [x] Run `... -m pytest tests/test_aur_unified_evaluation.py tests/test_aur_unified_infer.py tests/test_aur_planar_relations.py -q -p no:cacheprovider` (discover actual existing planar-test filename if different; record correction). Commit and obtain fresh task review.

### Task 3: Preserved-artifact 2D G1 diagnosis

**Files:** Private script/artifacts under the stated private root; public `docs/verification/2026-10-21/aur_g1_diagnosis/{RESULTS.md,aggregate_anonymous.json,RUN.json}`. Add a small repository diagnostic helper plus focused tests only if reusable numeric logic warrants it; otherwise a private, source-frozen audit script suffices.

**Interfaces:** Read original `~/outputs/aur_qualified_training_20261019/` registration/status/checkpoints and G1 feature caches read-only. Produce anonymous checked counts/norms/supports and bounded hypotheses; no new model forwards, readout training, source-image/label reads or GPU work.

- [x] Freeze questions and exact input/hash commitments before computing: did 2D adapters/shared weights update; are cached features finite/nondegenerate; are all13 host supports present; do train/probe input transformations agree; which classes account for weak 2D reading?
- [x] Reconstruct warm-start step0 only from preserved source/config and fixed seed when no step0 tensor exists; label reconstruction explicitly, verify source/hash/seed and selected/final optimizer parameter mapping. Never describe reconstruction as a saved historical boundary.
- [x] Stream/hash the two 2D feature caches (about0.5GB each) and selected/final weights on CPU; emit only allowlisted anonymous aggregates. The two 3D caches are about47GB each: use their existing report/support as context rather than silently adding a95GB array audit to the 2D failure investigation; explicitly mark 3D cache numerics untested in this audit. Compare initialized→selected→final adapter/shared update norms and optimizer counters. A cache hash computed now is an audit-time commitment, not a historically frozen one. Source comparisons use frozen historical checkout/snapshot so Task1/2 changes do not rewrite original source commitments.
- [x] Separate confirmed facts, ruled-out mechanisms and unproven causes. No inference that more epochs must fix failure. The new2048-step benefit versus inherited SSL cannot be established without a step0 readout; record that limit rather than fitting one silently.
- [x] Re-run the saved private audit from its frozen source, verify input hashes unchanged, and validate public JSON/links/privacy. Report if a missing artifact prevents a promised check; do not manufacture support.
- [x] Commit anonymous evidence and its exact replay command. Obtain fresh evidence/code review; propose one controlled candidate for next registration or a finite extra measurement if cause remains unresolved.

### Task 4: Integration, review and handoff

**Files:** Update `STATUS.md`, `CLAUDE.md` current entry, this plan checkboxes and public first-block report as needed. Keep medical details private.

- [x] Run full project CPU suite once after implementation: `... -m pytest tests/ -q -rs -p no:cacheprovider`. Keep log, return code, actual count and skip reasons. Sandbox multiprocessing failures require the specified escalated retry, not changes to tests.
- [x] Fresh whole-branch review against baseline `f3eb35f`, approved spec, plan and exact validation evidence; resolve substantive findings with regression checks. Cap automatic review/fix at three rounds per user instructions.
- [x] Report separately loader/evaluator readiness, actual preserved-artifact findings, unchanged original G1/quality FAIL, and unexecuted formal II/III. Record user-approved design and candidate experiment decisions without claiming candidate validation.
- [x] Commit only owned files. Publish to this feature branch under standing project authorization if remote access works; never push remote main. No stale historical replay updated to accept modified source hashes.
- [x] Continue into a separately frozen bounded experiment only when the diagnostic evidence determines an intervention and its initialization/data/metrics/budget; otherwise deliver the exact missing measurement and execute that finite measurement before proposing training.

## Self-review / execution decisions

Tasks1→2 share only the documented loader interface; Task3 reads frozen pre-change evidence; Task4 integrates all three. TRAIN weak-R behavior remains protected. No new final medical thresholds or independently labeled cohort are invented. User approved the design and explicitly requested execution, so proceed with subagent-driven implementation without another approval interview. The SDD ledger carries per-task commits, reviews and any rulings.
