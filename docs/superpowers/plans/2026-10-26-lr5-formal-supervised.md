# LR5 formal supervised continuation

**Launch prerequisite:** the merged Q31–Q34 gate-layer/fastMRI plan (2026-10-22 block1) must complete its acceptance before any actual formal run. This controller is prepared block2, not a release.

## Goal and accepted scope

Continue from verified LR5 Stage I/G1 into the first bounded formal II/III run using all qualified TRAIN rows as the sampling pool. The latest user engineering-only acceptance in STATUS supersedes the earlier small-panel quality stop; it does not rewrite those failures. Keep G1, parent completion certificates, exact data/runtime identities and final quality reporting intact. Full data pool does not mean every row is visited or training has converged.

Baseline: `095d8784685064fe94b4fee316959550540a1246`. Reuse the immutable `execution_v5/runtime` and LR5 selected Stage I export; never initialize from resource-probe II/III checkpoints. No architecture, loss, MRI acquisition mask or data change. Image-domain SSL masking is unrelated to acquisition undersampling; no reconstruction masks are generated here.

## Registered first-pass budget

Preserve the originally proposed first formal supervised budgets: II4096 and III2048 optimizer steps, alternating real2D/3D, batch1/accumulation1, seed20261019, crops1×160×160 and128×160×160, points16384. Use the engineering-tested conservative supervision rates: II backbone5e-5/heads5e-4; III backbone1e-5/heads2.5e-4/relation5e-4; warmup128 forII and64 forIII, no late backbone-rate jump and no SSL-only LR multiplier. These are fixed before new validation results, not claimed optimal.

Validation every512 and checkpoint every256 steps, eight validation batches per mode; select by existing equal-mode mean loss. Training/validation segment caps II14400s andIII7200s, external pipeline wall cap28800s including input/output checks. Existing full-model measurements predict ~3.6h optimizer time; validation/data I/O/scheduling are extra. Stop at timeout/nonfinite/integrity failure, preserve last durable checkpoint; no unlimited automatic resume. III requires a valid completed formalII selected export.

The frozen validation factory disables planarR for development rows, by its unchanged TRAIN-only weak-label policy. Thus StageIII's periodic2D loss is A/U/S-only while3D includes eligibleR. This existing selection criterion is retained explicitly; it is not validation of2D R quality. Separate held-out geometry-reference evaluation remains necessary after formal training.

Use all3499 supervised TRAIN rows/894 source-qualified patients through the existing patient-balanced sampler and405 development-validation rows/101 patients. Report realized mode/source exposure, not an implied epoch count. Periodic validation is loss monitoring, not finalA/U/R accuracy. TRAIN-only 2D weakR remains TRAIN-only; final geometry-reference evaluation uses its existing separate evaluator.

## Task 1: Admission and bounded external controller

Files: `scripts/aur_formal_supervised.py`, `tests/test_aur_formal_supervised.py`; private `~/outputs/aur_lr5_formal_supervised_20261026/`.

Re-use the existing watchdog/private-path/parent-binding functions from `aur_supervised_capability.py` and existing physical-UUID GPU selection helpers when preparing the launch. The controller stays outside the frozen runtime. Registration binds the exact runtime tree, allfive parent inputs, fullrecipes and controller files; reject imported runtime mismatch, altered inputs or incomplete recipes. Verify full parent certificate and bothG1 outcomes without replaying model inference. Delegate training to frozen `run_stage` so default fullpool views, checkpoint recovery and certificate generation remain authoritative.

Acceptance: tests demonstrate wrongpurpose/multiplier/parent/runtime/hash/recipe refusal, anonymous argument errors, immutable existing-output refusal, II failure preventsIII, and completed formalII is supplied toIII. Parent death/timeout reuse already tested process protections; no training in the watchdog parent. Save private registration/source copies and start/end input hashes; stdout only anonymous status. Actual fullpool start verifies complete data bytes; no shortcuts based solely on cachedJSON. Real weights never run under synthetic purpose.

## Task 2: Review, registration and launch

Obtain fresh-context implementation and exact-registration review. Resolve substantive findings before launch. Run targeted tests and projectCPU suite, preserve skipped environment checks. Verify LR5 parent and smallpilot terminal hash records, exact runtime145-source closure, actual recipe canonicalization. Check GPU allocation/visibility/current load, select compatible idle UUID, preserve other jobs; confirm process liveness from host execution boundary. Detach finite watchdog so chat interruption does not lose checkpoints. Recheck selected GPU immediately before training.

## Task 3: Report and next evaluation

Update main with actual registered/started/completed states, timestamps, bounds, parentlineage and limits. Final StageIII per-mode A/U/R evaluation uses existing unified evaluation with frozen source/header-defined observations; keep reported production raw-maskA distinct from oldpresence×mask diagnostic. Do not call training loss a final quality score. Record independent-cohort, nativeDICOM and physicianR gaps separately. Final real evaluation registration follows valid formalIII completion; no unqualified checkpoint substitution.

Changes are reversible: remove no historic files, preserve allfailed attempts, and return to baselinecontroller/repo if new checksfail. Stopping a new owned job leaves its last committedcheckpoint; other processes remain running.
