# TRAIN U Supervision Diagnostic Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Run the approved twelve-crop TRAIN diagnostic and produce reproducible, anonymous evidence separating query assignment, presence supervision, sampled mask coverage and component false positives.

**Architecture:** Add pure CPU loss/gradient and component-analysis helpers alongside existing AUR modules. A thin frozen-checkpoint runner collects crop arrays once, archives them privately and rebuilds every reported measurement through complete-only CPU replay. Existing model, trainer, decoder, targets and official evaluation semantics stay unchanged.

**Tech Stack:** Existing Python/nvgen, PyTorch, NumPy, SciPy, nibabel and pytest; no new dependency.

**Spec:** `docs/superpowers/specs/2026-10-18-aur-u-train-supervision-design.md` (approved by the user's2026-10-18 "yes").

## Global Constraints

- Base production revision `bf2a9246dd2471b13374d6d322f83556c043baab`; reuse the clean diagnostic worktree. The two design/preflight files landed at230dc57.
- Read data only under `/data2/congcong/data/FM_data`; selection is exactly six fixed TRAIN rows, two per BMSR/T1c, ISLES/DWI, PDGM/FLAIR, zero source-qualified val/test patient overlap.
- Use the spec's exact checkpoint/data/selection hashes. Model checkpoint `38f6376333041a75be1cce83e927bc221468761f80ae952e2a3020805f7dcbde`, StageIII step3000/72000crops, actual profile `unmatched_empty_v1`,64event queries.
- Exactly12crops,128x160x160, original augmentation/lesion_share0.5, two CPU point draws of16384points per crop with the spec's fixed seeds.
- Forward/mask heads under existing bf16 autocast; convert logits to float32 before sigmoid, as `infer.predict_volume` does. Batch1; freeze parameters and eval mode; no optimizer or model-parameter backward.
- Call existing `event_loss`/target construction on detached float32 CPU logits. Training query-to-GT assignment and official component TP/FP/ignored matching remain different fields.
- Fixed existing query decoding and source score cutoffs0.70/0.60/0.75. Preserve original component IDs; no threshold sweep, rescoring, retraining or sample replacement.
- New private0700outputs only; no overwrite/partial resume. Public files contain anonymous whitelist summaries and hashes only. No raw identities, per-query/crop/component rows or medical arrays enter Git or terminal output.
- Maximum1800seconds for image-based measurement/decoding, external timeout plus internal checks. Preserve workloads and select a compatible idle/lightest eligible GPU after fresh utilization/memory/process inspection. CPU nice19, numerical threads2, decoder workers<=2.
- All conclusions NOT_EVIDENCE, TRAIN-crop only, no FP/scan/S7 or new A/R performance verdict. A32 and R14 goals and their separate pending diagnostics remain visible.
- Meaningful TDD and independent code/scientific review; automatic review/fix loop capped at three rounds per scope.

## Review Focus

1. Duplicate sampled indices: occurrence gradients must be summed at each voxel before unique-voxel sign/coverage statistics (Task2 test).
2. A detection TP can come from an unmatched query, and a query matched to GT1 can produce an FP or a TP on GT2; IDs/assignments must not be conflated (Task2 test).
3. A checksum-consistent archive whose semantic membership, query count, outcomes or controls were changed must fail complete-only replay before images/model loading (Task3 test).
4. U-unsupervised/empty/zero-weight/crowded crops are distinct cases; no-lesion does not imply unsupervised and GT>64 cannot be silently dropped (Task1 tests, Task3 count checks).
5. FP existence is unknown until execution: zero FP or no sampled coverage must still yield a valid bounded diagnostic and explicit undefined strata without replacing patients (Tasks2/4).

---

### Task 1: Exact detached-logit loss probes

**Files:**
- Create: `anatobind/aur/u_supervision_diagnostics.py`
- Test: `tests/test_aur_u_supervision.py`

**Interfaces:**
- Consumes existing `losses.event_loss`, `dataset.event_targets_at_points`; one crop's presence `[Q]`, mask logits `[Q,P]`, instance IDs/weights/indices `[P]` and dense crop target cardinality.
- Produces `probe_event_loss(presence, mask_logits, instance_points, point_weight, point_indices, n_instances, *, u_supervised=True, profile="unmatched_empty_v1") -> dict`.
- Probe has keys `u_supervised`, `n_queries`, `n_instances`, `matches` (`query_indices`/`target_indices`), `query_target_instance` (lengthQ, IDs1..N orNone), `losses` (`u_presence`/`u_mask`), `reconstructed_losses`, `normalization`, `queries` (private per-query scalar dictionaries), and `mask_logit_gradient`/`presence_logit_gradient` NumPy float32 arrays. Scalar dictionaries include presence probability/target/weight/gradient, BCE/Dice/contribution, sampled/unique/effective positive counts and gradient summaries; unsupervised targets remain explicitly undefined.
- Produces `compare_draw_assignments(first, second, n_instances) -> dict` with target counts, both-matched/same-query/different-query/one-only/neither counts and capacity/zero-positive context. Inputs are probe dictionaries from identical frozen logits with different points.

- [ ] **Step 1: Write failing tests** for exact actual loss/matches/gradients, matched/unmatched normalization, legacy vs unmatched-empty behavior, U-unsupervised zero gradients, no GT, zero effective weights, duplicate points, targets present densely but missed at points, and GT>queries. Assert actual API values against direct calls to the original implementation, including positive gradient at a painted unmatched negative voxel in the current profile.
- [ ] **Step 2: Run RED** with `env CUDA_VISIBLE_DEVICES='' PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=. OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 nice -n19 /home/congcongliu/anaconda3/envs/nvgen/bin/python -m pytest tests/test_aur_u_supervision.py -q -p no:cacheprovider`; expected missing new helper/API.
- [ ] **Step 3: Implement** the two functions using detached float32 CPU leaves and the actual official loss/target builder. Return zeros explicitly when the official unsupervised scalar has no gradient graph. Reconstruct normalizations without changing the official assignment. Validate finite inputs, binary weights, valid shape/cardinality and original duplicate index order. Use documented1e-6absolute/1e-5relative tolerance for independent scalar reconstruction; original actual losses/gradients are authoritative.
- [ ] **Step 4: Run GREEN** for new tests plus `tests/test_aur_losses.py`; expected all pass, no unexpected skip.
- [ ] **Step 5: Commit** only the two owned files with an ordinary English message and record RED/GREEN commands/results.

### Task 2: Component identity, dense masks and anonymous aggregation

**Files:**
- Create: `anatobind/aur/u_supervision_components.py`
- Test: `tests/test_aur_u_components.py`

**Interfaces:**
- Independent of Task1 implementation: each supplied draw has `draw`, `points` int64[P], `query_target_instance` lengthQ, `u_supervised`, `mask_logit_gradient` float32[Q,P], with optional query scalar summaries.
- Produces `detection_outcomes(gt_rows, detection_rows, score_threshold) -> list[dict]`, preserving row order and original IDs/origins. Fields: `instance`, `status` in below_threshold/TP/FP/ignored_excused, `official_gt_index`, `official_gt_instance`, `detector_origin`. Reconstruct ignored second-pass identities with existing `match_scan`, cross-check official `scan_matches` and conservation counts.
- Produces `component_measurements(detection_map, detection_rows, gt_instance_map, valid, point_weight, *, outcomes, draws) -> list[dict]`, private per-component dense and per-draw fields. Maximum-IoU GT is descriptive only (ties lowest positive ID, no positive intersection=>None). Assigned-training-target overlap is per draw. Sum duplicate-index gradients before unique-voxel summaries.
- Produces `aggregate_records(records) -> dict`, strict anonymous schema. Each record provides source/sequence, crop GT/query counts, probe summaries in `draws`, `components` and `outcomes`. Output includes evidence/scope, rows/crops/queries/draw counts, by-source/sequence and GT-present strata; query/detection cross-tabs, presence/coverage/gradient/mask distributions, matching stability and explicit denominators/undefined values. Include component-weighted and per-query-weighted statistics, not independence claims.

- [ ] **Step 1: Write failing fixtures** for nonconsecutive IDs, inclusive score filtering before rematching, real-first ignored matching, one ignored GT excusing one detection, matched-query FP, unmatched-query TP, TP with a different training target, duplicate-point gradient cancellation/summation, zero coverage, ignored/padded voxels, no detections/GT and anonymous output containing no PRIVATE marker.
- [ ] **Step 2: Run RED** via the Task1 environment command with `tests/test_aur_u_components.py`; expected missing helper/API.
- [ ] **Step 3: Implement** original-ID official outcome reconstruction, efficient joint component/GT counts, valid/foreground/background/ignored voxel accounting and sampled coverage. Do not use ID-compacting `filter_instances`. Iterate actual component IDs and do not allocate giant dense target-one-hot arrays. Aggregate only fixed allowed scalar/category keys and reject malformed joins/nonfinite measurements.
- [ ] **Step 4: Run GREEN** for the new file plus relevant `tests/test_detection_metrics_ignore.py` and query decoder tests; expected pass.
- [ ] **Step 5: Commit** only the two owned files and report exact interfaces/RED/GREEN evidence.

### Task 3: Frozen TRAIN runner and complete-only numeric archive

**Files:**
- Create: `scripts/aur_diagnose_u_supervision.py`
- Create: `anatobind/aur/u_supervision_archive.py`
- Test: `tests/test_aur_u_supervision_runner.py`

**Interfaces:**
- Consumes Tasks1/2 functions exactly as declared above and the unchanged dataset/model/export/decoder APIs.
- CLI: `--samples`, `--val-patients`, `--checkpoint`, `--selection-private`, `--out`, `--gpu` (logical visible index), `--max-seconds` default1800 and `--replay-only` (complete archive only, no GPU/model/image requirement). Real defaults point only to approved /data2 inputs and fixed preflight selection. No user-set score/model/crop/query overrides.
- Runner public `run(args) -> int`; public `collect_crop(model, crop, voxel_mm3, *, source, sequence, selection_ordinal, crop_ordinal, device, deadline) -> (metadata, arrays)`; archive helper `replay_archive(out, expected_contract) -> dict` recomputes all results from trusted numeric inputs before yielding anonymous aggregate.
- NPZ arrays (allow_pickle=False): `gt_instance_map`, `valid`, `point_weight`, `detection_map`, `presence_logits`[Q]float32, `points`[2,P]int64, `sampled_mask_logits`[2,Q,P]float32. JSON metadata has private selection/crop identity, source/sequence, target cardinality, voxel volume, unfiltered decoder rows, point/crop/input hashes, model/resource controls and observed recomputable summaries. Large gradients remain numeric/private, not public JSON.
- Archive files: contract.json, selection_private.json, input_manifest_private.json, twelve numbered crop JSON/NPZ pairs, aggregate.json and COMPLETE.json written last. Private0700directory; write atomically; refuse existing non-replay output.

- [ ] **Step 1: Write failing tests** using the real tiny AUR model (existing TINY settings) and synthetic crop arrays: observer-on/off outputs and selected R logits identical; state/parameters/buffers unchanged; all original model parameter gradients remainNone; same-point reporting matches actual loss. Also test fixed membership/selection before volume load, exact float32-before-sigmoid order, no overwrite, out-of-range points, wrong64query/count/shape, changed hashes and checksum-consistent semantic cache corruption. CPU complete replay must never call a model or image loader.
- [ ] **Step 2: Run RED** for `tests/test_aur_u_supervision_runner.py` under the fixed CPU environment; expected missing runner/archive APIs.
- [ ] **Step 3: Implement** the real six-row selection commitment and raw input hash checks, actual checkpoint config/profile check, all-tracked-source hash contract, strict/eval/frozen model load and full model-state digest over names/shapes/dtypes/params/buffers. Generate crops with original TRAIN row positions. One forward per crop; use float32 sigmoid under existing autocast. Collect two original shared sampler draws and call exact helpers. Archive enough arrays to rebuild targets, matching, gradients, components and public aggregate. Revalidate model/checkpoint/raw/source invariants; external and internal timeout; anonymize terminal exceptions; incomplete runs lack COMPLETE.
- [ ] **Step 4: Implement complete-only CPU replay** with fixed dtype/shape/finite/index/ID/count/source/member checks, exact contract/input/checkpoint/code commitments and semantic control reconstruction. Reject incomplete/changed archives before image/model loading; compare reconstructed scalars with1e-6absolute/1e-5relative tolerances, exact integer/ID/matching counts and deterministic anonymous JSON. No archive mutation during replay.
- [ ] **Step 5: Run GREEN** all three new test modules plus original losses; expected pass. Run the entire CPU suite with the CLAUDE command and record skips honestly; rerun relevant targeted tests after any fix.
- [ ] **Step 6: Commit** owned files; freeze runnable revision and make a review package. Fresh task-scoped code review must approve helper+runner integration before image execution. Resolve substantive findings with regression tests.

### Task 4: Execute fixed diagnostic, independently audit and record result

**Files:**
- Create: anonymous `aggregate_anonymous.json`, `RUN.json`, `RESULTS.md` under `docs/verification/2026-10-18/aur_u_train_supervision/`.
- Modify: `STATUS.md`, `CLAUDE.md` only after observed results exist; update this plan's checkbox evidence.
- Private: fresh run directory under `/home/congcongliu/outputs/aur_u_train_supervision_20261018/`.

**Interfaces:**
- Consumes Task3 frozen runner, complete archive, actual observed tests and reviewed source revision.
- Produces observed tables/decisions, hashes and replay/review evidence. Numerical conclusion can be positive, null or adverse; no model-fixed/performance-adoption claim.

- [ ] **Step 1: Inspect resources** with fresh GPU utilization/memory/compute workloads and scheduler/device visibility. Select a compatible idle/lightest sufficiently free GPU; snapshot evidence privately and preserve workloads.
- [ ] **Step 2: Run** the fixed CLI in a detached process with external `timeout1800`, internal max1800 and a private log. Use first fixed crop as the resource pilot and reuse it. Monitor anonymous progress and resource evidence; on specified stop condition report incomplete, do not alter contract to force completion.
- [ ] **Step 3: Validate** all6rows/12crops/24draws/64queries, raw/source/checkpoint/model-state controls and complete marker. Hide CUDA and run complete-only replay; require exact reconstructed aggregate and unchanged archive bytes. Validate public whitelist/privacy and relative links.
- [ ] **Step 4: Dispatch independent scientific audit** to recompute primary cross-tabs/coverage/gradients from private cached numeric inputs and challenge scope/denominators. Whole-branch fresh code review covers complete change and validation. Resolve findings within the three-round cap, test meaningful regressions and record declined hypotheses with reasons.
- [ ] **Step 5: Write results** distinguishing actual observations from mechanism hypotheses and untouched A/R/val decisions. Record exact commands, revisions, hashes, runtime/RSS/CUDA allocated/reserved, tests and review limits in RUN.json. Commit results and handoff state; merge/push only if separately authorized, otherwise present the verified local result.
