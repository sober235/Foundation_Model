# Paired U background allocation implementation plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Execute the approved twelve-crop, two-draw ring-to-global shared-head tangent experiment with reproducible CPU replay and anonymous results.

**Architecture:** Pure CPU tangent mathematics and pure sampler/ROI/reporting helpers live under scripts, outside the prior diagnostic's tracked anatobind source closure. The controller implements private archive and frozen-forward orchestration. Two independent workers own the mathematical and measurement helpers; fresh reviewers inspect their changes and the integrated branch before execution.

**Tech Stack:** Existing nvgen Python, torch2.5.1, numpy/scipy, pytest; no new dependencies.

**Spec:** `docs/superpowers/specs/2026-10-18-aur-u-background-allocation-design.md` (approved2026-10-18).

## Global Constraints

- Start bd5b0f3; clean existing isolated diagnostic worktree, no CodeGraph index; preserve existing main/branch and user edits. Keep all original anatobind Python files and tracked file set unchanged.
- Reuse original six TRAIN rows/twelve exact crops, two seeds,64event/32A queries;16384points, focus_share0.5, boundary_share0.2 vs0.0; native bf16 outputs with float32 anchored tangent.
- No optimizer/training, threshold change, val/test images, calibration, new model verdict or remote push. Fixed matching for primary; rematching secondary; combined A+U primary and U-only secondary directions.
- Production weights/code and prior archive remain unchanged. New private output dirs0700/files0600; no private dict values/keys/arrays/IDs in tool output. Workers/reviewers access source and synthetic fixtures only.
- CPU nice19/threads2/CUDA hidden; model measurement <=1800seconds internal and external. Inspect GPU allocation/load/processes before launch, preserve other jobs.

## Review Focus

- New modules accidentally expand old source closure: integration asserts old tracked anatobind paths/hashes and replay contract still match.
- Sparse FP panels lose all voxels after paired-point exclusion: ROI tests preserve empty/undefined counts instead of dropping support.
- Native bf16 values are silently replaced by float32 heads: anchored-value and independent finite-difference fixtures enforce the declared surrogate.
- Sampler changes total positives or R assignment: sampler tests preserve focus slots only, report realized counts, official rematching remains secondary.
- Partial/altered private evidence passes as complete: archive tests reject mismatched members/contracts/nonfinite values and deadline expiry; replay writes nothing and loads no model.

## Numeric interfaces shared by tasks

Each crop has a metadata dict with source/sequence, n_instances, a_supervised/u_supervised, roi_rows, original control commitments/resources; JSON is private. Numeric inputs use allow_pickle=False NPZ with:

- `head_0_weight/head_0_bias/head_2_weight/head_2_bias`: float32 MLP parameters; `event_tokens[M,D]`, `entity_tokens[K,D]`, `event_presence[M]`, `entity_presence[K]` float32.
- `u_logits[2,2,M,P]`, `u_features[2,2,C,P]` float32; `u_points[2,2,P]`, `sampled_instances[2,2,P]` int64; `sampled_weights[2,2,P]` float32. Axis order draw,arm(control/global),query/channel,point.
- `a_logits[2,K,P]` float32, `a_entity_points[2,P]` int64, `a_ignore[2,P]` bool, `a_present[K]` bool. A uses control u_features.
- `roi_features[N,C]` float32; metadata roi_rows have draw,query,panel,n_voxels and optional private component ID. Panels: `fp_unsampled_query`, `fp_all_query`, `fp_unsampled_component`, `fp_all_component`, `target_all`, `target_fp_query`, `target_tp_query`. Empty rows remain with n_voxels0 and zero feature vector.
- `roi_reference_responses[N,2,2,2]` float64: independent streamed dense feature/JVP reduction for input validation; axes objective(combined/U-only),arm,normalization(raw/unit). Undefined rows/directions represented by explicit metadata flags, not NaNs.

`analyze_crop(meta, arrays) -> dict` returns fixed-loss/official-rematch diagnostics, A per-query loss responses, and numpy numerical arrays `g_u[2,2,Nphi]`, `g_a[2,Nphi]`, `directions[2,2,2,2,Nphi]`, `event_jvp[2,2,2,2,M,C]`, `entity_jvp[2,2,2,2,K,C]` (leading draw, then objective/arm/normalization). Zero unit directions have a parallel defined mask and zero numeric entries. The return structure is `dict(summary=dict(draws=[...]), numerics=dict(...))`; numerics also contains `direction_defined[2,2,2,2]` bool. Each summary draw exposes control_matches/global_matches dicts with query_indices/target_indices lists. Helpers may add explicitly named anonymous scalar diagnostics; never convert private records to terminal output.

### Task 1: Native-anchored shared-head mathematics

**Ownership:** worker creates `scripts/aur_u_background_tangent.py` and `tests/test_aur_u_background_tangent.py`; no other files.

**Interfaces:** `fixed_event_loss(presence, logits, target, weight, matches, *, u_supervised=True, profile='unmatched_empty_v1') -> dict` and `analyze_crop(meta,arrays) -> dict` as above. Match tuples are zero-based official query/target indices, target shape[N,P]. `analyze_crop` calls original matcher on control/native inputs, holds pairs for both primary arms, separately rematches global points. It returns JSON-safe summaries plus named numerical arrays; callers strip arrays before JSON serialization.

- [ ] Write tests: official scalar/logit-gradient equivalence for N0/N1/N>M, U-disabled, ignored/zero weights, duplicates, unmatched profile; exact native anchor; independent central finite difference for head gradients/JVP and shared A response; zero directions and disabled A; real model parameters not mutated. Tiny MLP shapes allowed for tests.
- [ ] Run focused tests RED, save observed failure evidence privately without inputs.
- [ ] Implement functional float32 parameter copies/anchored logits, fixed original losses, combined/U-only directions, event/entity JVP and A-query directional contributions (which sum to the total A response). Reuse original entity_loss for total; verify decomposition. No network/backbone gradients.
- [ ] Run focused tests GREEN, report command/count and commit only owned files.
- [ ] Fresh review with source/synthetic inputs, resolve substantive findings.

### Task 2: Paired sampler, ROI sufficient statistics and aggregation

**Ownership:** worker creates `scripts/aur_u_background_measurements.py`, `tests/test_aur_u_background_measurements.py`; no other files.

**Interfaces:** `paired_points(valid,instance,weight,*,n,seed)->(points[2,P] int64,summary dict)`; `build_rois(detection_map,detection_rows,outcomes,instance,valid,weight,pixels,points,matches)->(rows,features,indices)` where pixels[C,V] is detached torch tensor, points[2,2,P], matches=list of two(qi,ti), outcomes=original component outcome list. `indices` remains private/transient. `response_report(meta,arrays,analysis)->dict` consumes tangent output; `aggregate_reports(reports)->dict` produces only fixed source-level whitelist. Tests may use small query counts.

- [ ] Write tests for same focus/non-ring slots, initial uniform restoration, empty/no-ring, crowded quota0, ignored/duplicate counts and uniform extra positives.
- [ ] Write tests for matched FP true-BG vs other GT, TP/FP distinction, per-query union vs component weights, paired-point exclusion, no-support rows, target panels, source/draw/crop aggregation and undefined denominators. Exact enum validation, no arbitrary metadata echoed.
- [ ] Run RED; implement using original sample_points, explicit original slot rule, original component/query identities. Calculate ROI feature means with float64 accumulation, outputfloat32; return indices for independent streamed response check.
- [ ] Compute mean-logit responses by feature/JVP dot product; validate against supplied streamed references at declared rtol1e-5/atol1e-5 for nonempty/defined rows. Public aggregate includes per-source distributions/sign/support counts and fixed metadata only.
- [ ] Run GREEN and commit only owned files; fresh review/fix.

### Task 3: Private archive and bounded frozen-forward runner

**Ownership:** controller creates `scripts/aur_u_background_archive.py`, `scripts/aur_diagnose_u_background.py`, `tests/test_aur_u_background_runner.py` and integrates helper interfaces.

**Interfaces:** archive `create_archive(out,contract)`, `write_crop(out,index,meta,arrays)`, `reconstruct_crop(meta,arrays)->dict`, `complete_archive(out,contract,deadline)`, `replay_archive(out,expected_contract)->dict`. Runner CLI `--out`, `--reference`, original input overrides, `--gpu`, `--max-seconds<=1800`, `--replay-only`. Reuse prior runner prepare_contract/selection and numeric archive validation without changing original files.

- [ ] Write failing synthetic tests for no-model replay, bad/missing/extra/changed members, nonfinite input, incomplete markers, post-hash deadline, overwrite/path escape, output permissions and fixed safe errors/progress.
- [ ] Implement strict private numeric input schema/hash contract, source closure, baseline reference inventory, complete-only archive, CPU reconstruction and immutable replay. Exclude partial caches, no checkpoint/model load on replay. Freeze numerical comparison tolerances explicitly.
- [ ] Implement one frozen forward/crop, validate crop hashes, native control rtol/atol1e-5, exact matches/component rows/map; gather both arm features/logits and original A targets. Check full model state and fixed-query R logits before/after observation. Sampled features are float32 copies of native features; no dense autograd graph.
- [ ] Call tangent + ROI helpers, independently stream mean dense JVP projection over ROI indices, then reconstruct report from sufficient statistics and verify agreement. Save complete numeric evidence; clear per-crop GPU state serially. Enforce internal deadline through completion; progress only `Completed crops i/12` plus fixed completion line, errors class only.
- [ ] Run focused integration, existing prior diagnostic tests and entire project CPU suite; report actual failures/skips, rerun affected checks after fixes. Source-closure assertions and old CPU replay remain compatible.
- [ ] Fresh whole-branch code review, resolve findings <=3 rounds, commit runtime code before image-based execution.

### Task 4: Execute, audit, report and local handoff

**Ownership:** controller executes private inputs; independent reviewer only specified numeric-cache schema via fixed-output script, no selections/manifests.

- [ ] Preserve private verification logs (dirs0700/files0600), snapshot allocation/load/processes twice, select eligible GPU, launch exact frozen revision with external/internal1800s bounds. First crop is reused, not discarded.
- [ ] Verify all12crops completed, model/source/input/reference hashes unchanged; run complete CPU replay and prove archive bytes unchanged.
- [ ] Independent numeric audit and public-claim review, separate controller input-membership check; no unrestricted private introspection. Scan new public artifacts/actual stdout silently for selected identifiers.
- [ ] Write observed RESULTS.md/RUN.json/aggregate under approved public directory; source-level counts/distributions only, include mixed/null/harm patterns and limits. Keep original A/U/R verdicts and independent open tasks visible in STATUS/CLAUDE.
- [ ] Mark observed plan steps, commit, fast-forward local main and create unique handoff tag according to CLAUDE; preserve worktree/evidence and do not push.

Self-review: all spec sections map to the four tasks; interfaces and axis orders are defined above. Full design approval already explicitly includes implementation/execution; use the authorized subagent workflow without another permission round for this plan.
