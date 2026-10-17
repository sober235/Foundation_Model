# AUR Query-preserving Decoder Implementation Plan

> **For agentic workers:** Use Superpowers task execution with an isolated core-module worker, inline integration, independent task/whole-change review and regression fixes. Continue without routine confirmation under the user's instruction to proceed with the reviewed model architecture.

**Goal:** Run a frozen-weight, GT-free U decoder counterfactual while preserving A32 and per-instance R13+no-host.
**Architecture:** Stream independent per-query CC candidates, deterministic mask NMS and ownership, then residual instance splitting. Optional inference profile preserves original default. Evaluate paired fixed validation only.
**Tech Stack:** Existing Python/PyTorch/numpy/scipy/pytest; one idle GPU initially.
**Spec:** `docs/superpowers/specs/2026-10-17-aur-query-cc-decoder-design.md`.

## Global Constraints

- Fixed checkpoint/data/validation membership; `legacy_max` is default and unchanged.
- `query_cc_v1`: presence>0.5, mask>=0.3,26CC,shared10mm³floor,maskNMS IoU>=0.5. Score float32 input component mean, float64 accumulation, inherited by residual children.
- Canonical `(-score,window_start,query,component)` order; full ownership before residual split/floor; no backfill or candidate cap.
- Score filter after completed ownership; continuous IDs with synchronized origin. Detector query and recentered R binding query are separate provenance.
- A and controlledR core outputs unchanged; all interfaces/shard compatibility include profile+rules; no training/test/threshold sweep/target change.
- No raw identifiers/case tables in git. CPU hiddenGPU,nice19,<=48threads; data only/data2; no overwrite outputs.

## Review Focus

1. Nested/partially overlapping masks and exact NMS .5; scores must not leak from legacy max or GT.
2. Padding/out-of-volume pieces cannot bridge components or inflate score/volume.
3. Ownership can split/discard children; no backfill and no union of touching different owners.
4. Profile changes cannot silently mix across eval shards, inference records or R token origins.
5. Memory/runtime on diffuse masks must be measured; no arbitrary cap hides candidates or FP.

### Task 1: Core candidate decoder (worker ownership)

Files: create `anatobind/aur/query_decoder.py`, `tests/test_aur_query_decoder.py` only.
Interfaces: `DECODER_PROFILES`, `decoder_rules(profile)->dict`; `QueryCCDecoder(shape,voxel_mm3,workers=4)` callable with same keyword payload as predict_volume observer; `.finish()->(instance_map,rows,stats)`; `filter_instances(instance_map,rows,score_threshold)->(map,rows)`.

- [x] Write and run failing behavioral tests for every core constraint above.
- [x] Implement pure GT-free candidates/NMS/ownership and bounded threaded collection. Rows: instance,box,n_voxels,volume_mm3,score,detector_origin; source fields window/query/component/rank and parent score/voxel counts. Stats aggregate candidate/suppression/ownership/fragment counts plus local trace when needed.
- [x] Run core tests, self-review and commit owned files. Independent scoped review; fix substantive findings (max3 rounds per AGENTS).

### Task 2: Integrate optional profile and provenance (root ownership)

Files: modify `anatobind/aur/{infer,eval}.py`, `scripts/{aur_eval,aur_eval_merge,infer_anatobind_brain}.py`; create integration tests.

- [x] Write failing tests for default compatibility, A invariance, profile dispatch, score/id/origin behavior and mixed-shard rejection.
- [x] Integrate collector without changing original accumulator/model call order; decoded-instance dispatcher; forward profile through library/CLI and metadata; reject unknown/incompatible rules.
- [x] Keep R recentered binding path; carry both origins into records and end-to-end rows. Run targeted and fullCPU tests, commit; independent review.

### Task 3: Frozen experiment runner and scientific check

Files: create `scripts/aur_compare_decoders.py` and relevant tests; anonymous records under `docs/verification/2026-10-17/aur_query_cc/`.

- [x] Freeze six selection/input hashes and code; legacy outputs reproduce before comparison. Collect both U decoders from same underlying window predictions when feasible, preserving exact float32 query scores.
- [x] Verify A and controlledR cores unchanged; compare new detector hit/FP with original matching at fixed source scores, actual R origins, candidate counts, time and memory.
- [x] Run six-case probe on idleGPU with7200s hard timeout; estimate fullval resources from observed wall time.
- [x] Run complete frozen405-row validation with deterministic/resumable per-row outputs, no new test inference. Compare all three goals and preserve pseudo-label/grid caveats. Do not finish with only tooling or six-case oracle results.

### Task 4: Close and hand off

- [x] Resolve independent code/scientific review findings, at most3 automatic rounds; verify affected behavior and final suite as needed.
- [x] Update STATUS/CLAUDE with measured outcome, A/R independent audit and next justified model action; mark unexecuted work honestly.
- [x] Commit, authoritative localmain handoff/tag under existing project convention; do not publish private data or force-push.

Completion evidence: `docs/verification/2026-10-17/aur_query_cc/RESULTS.md` and `RUN.json`. Full405val complete; fixed profile fails all3Ucriteria pairs; defaultlegacy retained. Product code1874af2,1111CPUtests passed/2skipped. Final scientific review0unresolved after narrowing oneA-summary wording claim. Localmain/tag handoff recorded by session closing git commands.
