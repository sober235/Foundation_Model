# Remaining-task review

Review date: 2026-10-29. Scope: main and the registered
`work/aur-formal-supervised` delta through `6878f13`. This document archives the
review; its branch does not replace the registered execution branch.

**One actionable plan gap (P2), plus an observed development-quality risk.**
No new correctness regression was verified in this remaining-task review.
All model measurements below are NOT_EVIDENCE. No training, model forward or
new evaluation was started for this review.

## P2: Visual verification has no retained prediction artifact

The development plan Task4 requires inspecting private prediction overlays
before claiming visual validation. The current controller saves observation
records, while the evaluator reduces predictions to metrics, supports,
detections and relation records. Neither predicted arrays nor overlays are
retained; the new callback receives only the reduced record.

Trigger: attempting Task4's visual inspection from this run's completed files.
Impact: scalar records cannot substantiate visual validation. Add a bounded,
separately registered overlay-generation/inspection task, or keep visual
validation explicitly unverified. Preserve the running frozen source; do not
repeat the full evaluation merely to obtain illustrations. Any later bounded
inference must be disclosed as additional execution.

Source evidence at reviewed revision:

- [Plan Task4](../../../superpowers/plans/2026-10-29-aur-development-selection.md), lines74–77.
- [Controller](../../../../scripts/aur_development_selection.py), lines114–140: private observation-record persistence.
- [Evaluator](../../../../anatobind/aur/unified_evaluation.py), lines223–266 and548–551: prediction reduction and committed final outputs.

## Observed running state and quality risk

At2026-10-29 11:03:57 UTC (19:03:57 Beijing), run_v1's anonymous progress
reported `phase=full_development`, completed5809/9809 observations (about59%).
The102 3D source/sequence observations for threshold selection had been saved.
No terminal or outer completion marker was present, and the final deployment
registration did not yet exist. These are timestamped observations, not a
claim about the run's later state.

Host-boundary inspection confirmed the registered controller PID3975686 and
start ticks449332314 matched, with one worker. Earlier handoff text describing
admission is a historical launch snapshot, not the latest progress above.

The saved threshold-selection results were:

|Source|Sequences|Selected cutoff|Hits / GT|Sensitivity|FP/scan|Reference FP/scan budget|Within budget|
|---|---:|---:|---:|---:|---:|---:|---|
|BMSR|41|0.55|1 /470|0.002128|0.682927|0.878049|true|
|ISLES|20|0.50|3 /173|0.017341|0.850000|1.650000|true|
|PDGM|41|0.55|8 /53|0.150943|0.195122|0.365854|true|

The corresponding completed development-reference sensitivities were0.676596,
0.601156 and0.905660. Thus satisfying the false-positive budget does not
establish adequate sensitivity. This is a substantial development-quality
signal; it is not a final historical-test verdict. First audit the completed
development results and diagnose the low sensitivity. Do not treat reading
final test as a tuning step or automatically add training budget.

Evidence locator: private `~/outputs/aur_development_selection_20261029/run_v1/`,
`progress.json` and `threshold_selection.json`; completed reference evidence is
documented in [the gate-layer report](../../2026-10-26/aur_gate_layer/RESULTS.md).
Medical arrays, identifiers and per-observation records remain outside Git and
tool output.

## Ordered remaining work

|Order|Task|Acceptance / dependency|
|---|---|---|
|1|Establish the development run's actual outcome.|Require terminal, complete marker, output hashes and closing integrity. Review independent2D/3D A/U/R coverage and low-sensitivity findings. Failure/timeout needs an audited decision; no duplicate launch or automatic retry.|
|2|Review threshold and deployment selection.|Use the actual final deployment_registration.json, not the diagnostic head registration. COMPLETE and deployment_frozen=true are separate facts. Both R modes need matched support; preserve unavailable results and original diagnostic records.|
|3|Complete bounded visual verification.|Resolve the P2 above with explicit private artifacts and an inspection record. Until then, visual validation remains unverified.|
|4|Build one canonical qualified historical-test successor.|Only after development completion/review and configuration freeze. Preserve the training contract; verify real source bytes, native labels, repaired derivatives and historical exposure disclosure. Real-data qualification remains unfinished.|
|5|Produce the test-role-specific same-grid S7 reference.|Use existing predictions, actual row intersection and exclusion counts. The completed41/20/41 validation reference has the wrong role hash for final test; do not substitute it or start new S7 inference.|
|6|Register and execute final evaluation once.|Bind export, evaluator, qualified role, reference, crops and final deployment-registration digest before scoring. Run2D/3D together: the model's exclusive claim is created before label-based selection, so separate mode invocations conflict. Preserve failed claims; do not change role/output directories to evade them. Judge all six gates using committed artifacts; retain NOT_EVIDENCE and historically-exposed wording.|
|7|Execute the4469-sequence fastMRI batch and annotation report.|Freeze the exact inventory; require n_ok+sum(skipped)=4469 and both modes, with nonpilot weights and no pilot limit. Use median frozen disease cutoffs and selected head/lookup. Preserve geometry=assumed, native-grid mapping, screen-side reporting and skip reasons. Report rejected boxes, unmatched annotations and annotation-bearing-slice denominators.|
|8|Close current128-query CUDA restoration evidence separately.|A finite isolated check must preserve formal checkpoints and distinguish exact restored state from future CUDA-update reproducibility. Historical64-query restoration and CPU skips do not prove this. This gap does not itself block inference-quality evaluation and does not justify retraining.|
|9|Update handoff and archive actual outcomes.|Keep main and the registered work-branch pointer coherent. Separate running, completed, quality-failed and unverified states; never promote a launch snapshot or CPU test result into final model acceptance.|

Relevant implementation constraints were checked in test_role.py:87–164,
reference_1mm.py:159–167, scripts/aur_gate.py:61–66,
unified_evaluation.py:512–522, fastmri_batch.py:160–163/216–226/305–310 and
fastmri_plus_report.py:144–181. Existing batch inventory acceptance is generic:
the final execution registration must explicitly enforce the required4469
coverage and accounting.

## Review evidence and limits

An independent read-only reviewer inspected the current plan and relevant
source. It reported the P2 overlay-evidence gap and no verified new correctness
regression. Parent inspection confirmed the source evidence and read only
whitelisted anonymous live progress/threshold aggregates. No private medical
array inspection or new execution was performed.

Previously observed implementation validation remains2142 passed/5 skipped;
this document-only review did not rerun that suite. Final quality, real test
qualification, overlays, full fastMRI execution and current CUDA restoration
remain unverified.
