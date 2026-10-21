# 2D optimizer and planar merging candidates Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement and validate independent LR5 and PLANAR candidates, then execute bounded registered Stage I/G1 comparisons.

**Architecture:** Optional optimizer and architecture extensions preserve the old default and all historical artifacts. PLANAR adds dimension-specific merge operators while keeping shared attention and the original3D path. Training uses a pinned current experimental runtime with exact historical qualification code and a new complete runtime commitment.

**Tech Stack:** Existing nvgen/PyTorch/NumPy/MONAI/pytest; no new dependency.

**Spec:** `docs/superpowers/specs/2026-10-21-aur-lr-and-planar-merge-design.md`; user explicitly authorized proceeding after the architectural discussion.

## Global Constraints

- Independent arms: LR5 legacy structure + ssl_2d_lr_scale5; PLANAR planar_mergeTrue + scale1. Never combine their factors in this experiment.
- Original G1FAIL/threshold0.05, original checkpoints/data/source hashes remain immutable.2048alternating steps/1024mode exposures; same seed20261019, baseLR1e-4, original8validation boundaries/best equal-mode loss selection.1792descriptiveonly.
- Default absent/False flags preserve old parameter/RNG/numeric behavior and format2 canonical cfg/certificates. Do not fabricate exact resume after config/architecture changes.
- Raw data only approved/data2 root; artifacts privatehome outsideGit, no private manifest keys/values/medical IDs in output. CPU CUDA hidden/nice19/threads2; full suite outside sandbox for multiprocessing.
- Tests/medical claims separate. Resource snapshots respect explicit visibility∩scheduler, no other workloads stopped. StageImax5400s andG1max7200s perarm, no automatic sweeps/budget extension or II/III.

## Review Focus

- Old serialized configs must still normalize identically and actual legacy formal certificates load; new optional fields must not silently invalidate old artifacts.
- New 2D merge construction/initialization must not alter shared3D or old4adapter starting values; test RNG and name-based state identities.
- False mode gradients/momentum after resume must not change inactive adapters or merges; test actual optimizer steps across both modes.
- Architecture flag must survive every factory/export/G1random/inference boundary; test real both-mode outputs and strict mismatch rejection.
- Qualified data code must match old immutable5hashes while new runtime is separately committed; no partial source copy, alias role changes or resumed old run.

### Task 1: Optional Stage I adapter LR profile

**Files:** `anatobind/aur/unified_run.py`, new `tests/test_aur_ssl_adapter_lr.py` (existing runner tests are reference).

**Interfaces:** `validate_run_config(cfg)` accepts optional `ssl_2d_lr_scale`; `UnifiedRun` builds unchanged single `ssl` group for omitted/1, or `ssl` and `ssl_2d` groups for non1. Only the four exact2DIO parameters get multiplier; same scheduler factor.

- [ ] RED: baseline omitted config stays without optional key after normalization; explicit1 produces identicalCPU model/optimizer-step outcomes; scale5 groups cover all parameters once and only exact4have5×rate; positivefinite/nonbool validation; non1 rejected forII/III.
- [ ] RED: actual alternating-mode updates, checkpoint/resume and config-change rejection; historical cfg/certificate case lacking optional field remains valid.
- [ ] Implement minimal optional-key validation/grouping, no change to pilot/default/architecture, preserve exact original list order for scale1.
- [ ] Run focused `tests/test_aur_ssl_adapter_lr.py tests/test_aur_unified_run.py tests/test_aur_unified_launcher.py tests/test_aur_unified_infer.py` with CPU nvgen nice19 threads2; record actualRED/GREEN, commit owned files and obtain fresh task review.

### Task 2: Opt-in planar merging throughout the unified model

**Files:** `anatobind/aur/{swin.py,model.py,unified.py,unified_train.py,unified_run.py}`, `anatobind/aur/ssl/{model.py,unified_init.py,unified_eval.py}`; tests new `test_aur_planar_merge.py` and affected existing migration/warm-start/G1 tests. Add no generic framework.

**Interfaces:** optional `planar_merge: bool=False` in model/backbone; `PatchMerging2D` D1-only four-neighbor norm4C/reduction4C→2C; `Stage.forward(...,input_mode='3d')` routes merge; strict old defaults. Carry optionalbackbone flag without requiring it on oldcfg. `warm_start_stage_one` explicitly derives additional9missing merge tensors per spec; ordinary strict loads stay strict.

- [ ] RED:2Dmerge receives4Cnonpadded-depth features, expectedgrid/coords/valid and oddH/W borders; refusesD>1. Defaultstatekeys/seed/RNG unchanged; PLANAR3Doutputs bitwise equal with same sharedweights; true2D outputs/gradients shapecorrect.
- [ ] RED: inactive-adapter pairs include nested3D/2Dmerges; realoptimizer/resume never changes inactive branch with oldmomenta. Dynamic migration expectedkeys use each module's actual state_dict, not assumedweight/bias on nested modules.
- [ ] RED: old176 copiedvalues+four seededIOvalues preserved; nine newmerge tensors equal documented first4Csource slices; provenance explicitnotexact2Dequivalence; any malformedsource/derivedshape rejects beforetargetmutation. Backbone-only compatibility explicit.
- [ ] RED: flag survives StageOne/AnatoBind/runner/G1same-architecture-random/export/infer, oldcfg absentFalse compatibility and strict wrongprofile export rejection; real bothmode integration.
- [ ] Implement opt-in operator/routing/config plumbing with no defaultRNGsideeffects, preserving defaultcanonical absence for oldformat2. Reuse D1-equivalent pooling only with numeric tests.
- [ ] Run targeted new+existing backbone/migration/SSLinit/runner/G1/loader suites; recordRED/GREEN and exactcoverage, commit owned files; fresh review, max3fixrounds per userrules.

### Task 3: Integration, qualification and bounded experimental execution

**Files:** public `docs/verification/2026-10-21/aur_2d_candidates/`; private `~/outputs/aur_2d_candidates_20261021/` runtime/source/registration/controller/outputs. `STATUS.md`,`CLAUDE.md`,`README.md` currentstatus.

**Interfaces:** existing `run_stage` and complete G1 `run/validate_gate`; strict contracts/certificates unchanged. Candidate runtime exported from git0c0f2f8 then overlay exactly reviewed changedruntime modules from this feature; record base+overlay+complete closure, retain original5qualification bytes and check all actualsources/roles. Never patch oldhashes or use pilot to replaceformal.

- [ ] Full CPU suite once after production changes, then fresh wholebranch review of exactdiff and evidence; resolve all substantive findings. AdditionalCUDA checks only where newrouting/recovery requires them on fresh compatibleidleGPU.
- [ ] Build/version runtime atomically, check actual5dataqualification hashes and dependencies, full original data contract verification using copiedqualified reader. Run syntheticbotharm load/step/resume/certificate/G1configuration checks in that exact runtime.
- [ ] Freeze/init actualweights for botharms beforeheldoutprobe: baseline176+IO4starting values, PLANARderived9mapping and3Dforwardcompatibility onsynthetic input; qualifiedTRAIN-only16-step perarm resource/resume smoke, max1800s each, no heldoutforward. Preserve allfailedpreflight versions.
- [ ] Register StageI/G1 budgets/settings/outputs/selection/stop rules and newsource/init/data hashes beforetrain. Freshreview registration/controller (no newnested implementer). RespectfreshGPUvisibility/scheduler/resources; queue/parallel independentarms onlycompatibleavailablecards, no interference.
- [ ] Execute formalStageI thenfullbothmodeG1 perarm underhardbounds; noII/III. AtG1FAIL/INSUFFICIENT stoparm; onbothPASS stop for subsequent approvedsupervisedcapabilityplan. Save atomiccompletion/failclosedstatus and anonymousmetrics; no automatedmetricselection.
- [ ] Postprocess savedcounts/artifacts rather thanrepeatforwards. Report perarm2D/3D gates, perhost support, lesionAUC, timing/resource/recovery, exactdifferencesandlimits. Historicalbaseline remains historicalsingletrajectory; no falseindependence/clinicalclaim.
- [ ] Update/publicationreview reports, commit/push existingfeaturebranch only, remote main untouched. Finalhandoff distinguishes implementationverified, runningexperiment and actualqualityresult; do not call runningjobscomplete.

## Pre-flight decisions

The user approved both candidate approaches; numerical/init/runtime choices above are implementation calls, not claimed user-specified values. Missing oldcfg optional flags stay absent in canonical exports. Structuralinit copies only unambiguous depth0source slices and explicitly changes2Dmath; 3Doperators/startingvalues stay unchanged. No checkpoint or reader hash is modified in place. The completed step0 diagnostic is not rerun. If a runtime-overlay dependency cannot be made exact, qualify a newimmutable data version instead of skipping checks.
