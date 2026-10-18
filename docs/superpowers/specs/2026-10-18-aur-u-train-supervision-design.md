# Frozen TRAIN diagnostic of U matching and supervision

Status: approved by the user on2026-10-18 ("yes" to implementation and execution); metadata preflight complete. This is the next bounded diagnostic after `bf2a924`, not a completed model repair.

Design review: a fresh-context ML research engineer independently read this description, preflight and relevant source context. Builder-check outcome: "nothing important left". This review is not code or scientific-result validation.

## Intent and scope

Continue the user's requested next task by investigating why high-confidence U false positives persist. Preserve the three goals: A32 anatomy identity masks, unified annotated-abnormality U instance masks, and R13 sided host plus no-host per U instance. S remains internal sequence context. Prioritizing this U diagnostic is a practical local choice, not evidence that U has the globally optimal priority over the independent A32-query and R-geometry diagnostics.

The outcome is a reproducible anonymous report linking TRAIN-crop detection errors to current query matching, presence targets, sampled mask coverage, and detached-logit loss gradients. It can identify a mechanism to test next; it cannot establish historical training causality or generalization. No training, optimizer updates, calibration fitting, threshold selection, production decoder change, val/test image loading, or new clinical/Level R claim is included. The complete405val FAIL and default `legacy_max` remain the existing decisions.

Implementation defaults below are the agent's choices for this experiment, not user-selected hyperparameters. Approval concerns this complete diagnostic description, including its bounds and stopping rules.

## Evidence checked before proposing the experiment

Starting revision: `bf2a9246dd2471b13374d6d322f83556c043baab`, clean main and clean existing diagnostic worktree. Work is isolated in `/home/congcongliu/src/Foundation_Model-aur-diagnostic` on `codex/aur-model-diagnostic-2026-10-16`; no repository-root CodeGraph index exists.

Read-only preflight verified the full checkpoint hash, read its actual metadata on CPU, and selected eligible TRAIN rows from metadata before opening any image or target volume:

- C2 Stage III best export: `step=3000`, `seen_crops=72000`, seed0, checkpoint SHA256 `38f6376333041a75be1cce83e927bc221468761f80ae952e2a3020805f7dcbde`.
- Its actual `u_mask_profile` is **`unmatched_empty_v1`**, with16384points, crop128x160x160, two crops/volume and lesion_share0.5. Merely adding empty-mask supervision again is not a new repair.
- Metadata-eligible U-supervised TRAIN pools: BMSR/T1c318rows/225patients; ISLES/DWI180/180; PDGM/FLAIR359/356. Two distinct patients per source/sequence were fixed, six rows/patients total, zero source-qualified val/test patient intersections.
- Focused current-code check: `tests/test_aur_losses.py`,12passed in1.88s, CUDA hidden. These are current loss checks, not new model-performance results.

[Anonymous preflight](../../verification/2026-10-18/aur_u_train_supervision/preflight.json). Private selection and original preflight remain under `/home/congcongliu/outputs/aur_u_train_supervision_20261018/design_preflight/`.

## Source trace and alternative explanations

- `anatobind/aur/train.py::_forward_losses`: one model forward; entity and event logits at shared points; U targets/weights passed to `event_loss`; R uses the resulting query-to-target assignment.
- `anatobind/aur/losses.py::sample_points`: approximately50%lesion-instance points,20%adjacent-background ring, remainder uniform; sampling has replacement. Without retained instances it is uniform. The implemented quota can be zero for crowded crops, so coverage must be measured, not assumed.
- `losses.py::match_events`: Hungarian assignment on negative presence probability plus weighted mask BCE and Dice. It is not detection matching. A crop with more targets than queries cannot match every target.
- `losses.py::event_loss`: matched queries have positive presence targets and matched mask supervision; unmatched queries have no-object presence weight0.1 and, for this checkpoint's profile, empty-mask BCE scaled0.1. Entire U loss is off on U-unsupervised rows.
- `dataset.py::{load_volume,make_crop,crop_instances}`: grid verification, augmentation, within-crop instance renumbering and volume-floor handling define the actual training targets; sub-floor regions and padding have zero point weight.
- `query_decoder.py::QueryCCDecoder` and `eval.py::scan_matches`: fixed query components, NMS, ownership and residual splitting produce detections; score filtering precedes official real-first/ignored-second bbox matching.

Three explanations must remain distinguishable:

1. A training-unmatched, present query produces a high-score FP despite its negative supervision. Check whether that component is sampled and whether the actual current loss locally pushes its logits down.
2. A training-matched query covers a real lesion and also produces unrelated components. Correct positive presence for the query does not imply every component is a true lesion.
3. Point draws change Hungarian assignments or miss important positive/negative voxels; capacity overflow may also leave GT unmatched. Report these observations without calling two-draw instability a proven training failure.

Training-unmatched query, detection FP, and an individual background voxel are different labels. Their cross-tabulation is a primary output. No explanation is presumed true in advance.

Alternatives considered: fitting a final-scalar calibrator repeats a direction already bounded by the archived score-order result; immediate loss changes or additional training confound diagnosis; the separate R geometry and A32 query diagnostics remain valid next projects. This TRAIN experiment adds missing supervision-side evidence at a small fixed budget.

## Frozen inputs and selection

Read data only below `/data2/congcong/data/FM_data`; resolve paths and reject escapes. Never use the `/data0` cold backup.

- Samples: `derived/aur/samples_1mm_v1.json`, SHA256 `fb96acd741aa6c0ee689eb772a4332304501c8d63ebd7ccddcd657f41eb86d5d`.
- Validation-patient metadata: `derived/aur/ssl_manifest_v1/val_patients.json`, SHA256 `475f91d97d8b03f312df560f98f39ab81c068c39d0a24c632d8efa5fd52d9d34`.
- Checkpoint: `derived/aur/ssl_runs/c2_stage3_80k_20261014_1344/aur_stage3_best.pt`, exact hash above; strict load through the existing export loader, with all parameters frozen and model in eval mode.
- Reuse `train.split_rows`. Require `u_supervised=true`; do not turn A-only sequences or U-invisible labels into healthy negative examples. Do not require A supervision to enter this U sample.
- Fixed seed20261018. For each registered pair `(bmsr,T1c)`, `(isles,DWI)`, `(pdgm,FLAIR)`, rank TRAIN candidates by SHA256 of compact JSON `[seed,source,str(patient),sequence]`, tie-break by original TRAIN row position, and take the first two distinct source-qualified patients. The already frozen selection's row-position digest is `719779c337e9240e828abb1db94cb48c42e8c1c1f305624620974252de084858`; private selection file SHA256 is `7f542122cc1dd1623f616681d47b96c7003e1488718ab33f964533bfb5a42bfb`.
- Recompute eligibility and commitments before any volume load. Read test metadata only to exclude source-qualified test patients; no test images, labels or outcomes are used. No case replacement based on measured lesion counts, FP counts or results.
- Before execution, freeze code revision and hashes of all tracked `anatobind/**/*.py` dependencies plus the diagnostic entry point; record environment versions. Verify raw selected inputs' hashes before/after, and validate any cache against its full input/code/checkpoint/selection contract.

Source-qualified exclusion is not biological deduplication across datasets. TRAIN is already available to the base model and this is not an independent evaluation cohort.

## Crops, forwards and sampled-loss probe

Generate exactly two crops for each selected row,12crops total, using the existing `AURDataset` over the full TRAIN list at its original row positions, epoch0, seed20261018, crop128x160x160, do_augment=true and lesion_share0.5. Preserve its current normalization, rotations, intensity changes, validity, coordinate convention, instance renumbering and volume floor. This is a newly fixed draw from the training recipe, not a replay of the actual historical distributed training stream. Log the realized crop/instance strata; do not force a lesion-centred or lesion-free realization afterward.

One frozen model forward per crop, batch1, eval mode, the existing CUDA bf16 autocast path, no model-parameter gradients. All64event queries are included; do not preselect queries using GT or truncate detections. Dense query probabilities may be transient but are never public.

For each crop, draw two original shared-sampler point sets on CPU, each16384points. Use generator seeds `20261018 + 1000*selection_ordinal + 10*crop_ordinal + draw_ordinal`, with source-order selection ordinal0..5, crop ordinal0..1 and draw ordinal0..1. Hash point indices and generated crop tensors privately. These CPU draws deliberately are reproducible diagnostic draws, not the CUDA RNG stream of the historical trainer.

Evaluate masks at those points through the model's existing mask head. Export the exact float32 sampled logits and presence logits to CPU; detach and clone them as local gradient leaves. Call the actual `event_targets_at_points` and `event_loss` with the actual checkpoint profile. Record that these are CPU loss/logit probes on eval-mode frozen outputs, not historical optimizer or network-parameter gradients. No optimizer exists and nothing is backpropagated through the model.

For every query/draw, retain privately: matched target or unmatched status, presence probability/target/weight/gradient, sampled target-positive counts and unique voxel counts, effective point-weight coverage, BCE/Dice contributions and gradient summaries. Explain the actual normalization across matched queries, unmatched queries and samples. Check reconstructed contributions against the unmodified `event_loss`; document any decomposition tolerance rather than silently replacing its formula.

Compare assignments between the two draws on the same frozen forward. Report target-to-query disagreements and unassigned GT separately, including GT>64 capacity overflow, zero-weight targets, and crops with no retained GT. Empty/saturated strata must remain in the denominator or be explicitly undefined with their counts.

## Crop-level detection provenance and mask measurements

Feed each crop's full-resolution probabilities to the unchanged `QueryCCDecoder`, using local window `[0,D)x[0,H)x[0,W)`, actual crop validity and original voxel volume. The frozen rules are presence>0.5, mask>=0.3,26CC,10mm3floor, original-mask NMS IoU>=0.5, fixed priority/ownership/residual splitting and inherited parent score. Use existing score filtering only after ownership completes: BMSR0.70, ISLES0.60, PDGM0.75. Keep all pre-score rows privately; no threshold sweep or rescoring.

GT consists of the crop's retained instance map plus ignored components from `(valid>0.5) & (point_weight<0.5)`; padded voxels must not become ignored lesions. Use the existing official score-first, real-first/ignored-second bbox IoU>=0.1 matcher. Preserve original crop-local detection IDs through filtering/provenance joins, or store a verified explicit renumbering bijection. Derive FP identities from the official unmatched/non-excused set, not from a different greedy matcher. Verify `kept = matched + FP + ignored-excused`.

Join every kept component's detector query to the training match independently for both point draws. A matched query's detection may be FP; an unmatched query's detection may match GT. A component matched to a different GT than the query's training target is an explicit separate status.

For each component retain private parent/child identity, parent score and retained fraction, voxel volume, dense mask intersection/Dice/IoU with its assigned training target and best-overlap GT, foreground share on any retained GT, and coverage by effective sampled points. Best-overlap GT is a diagnostic measurement only and must not change prediction, Hungarian training assignment or official detection assignment. Include unmatched queries and zero-overlap components. Report component foreground/background/ignored fractions with explicit valid-voxel denominators so bbox matches cannot masquerade as mask-quality success.

Map sampled loss-logit gradients back to component voxels using the exact sampled indices. Count both draws and unique sampled voxels because sampling has replacement. For a negative-target voxel/query, a positive logit gradient means local gradient descent lowers that logit; zero sampled coverage establishes only absent direct loss on those voxels for that draw. It does not imply zero shared-parameter learning signal or explain the historical checkpoint causally.

This is a **TRAIN-crop** measurement with single-crop context and augmentation, not a full-volume natural-window replay. Report FP/crop, counts and masked error strata, not FP/scan or an S7-budget verdict. The numeric source cutoffs are fixed descriptive strata; unlike the earlier feasibility analysis, no new claim of equal observed FP is made. Conclusions about query_cc components do not automatically transfer to legacy fused decoding.

## Controls, outputs and interpretation

- The checkpoint file and in-memory model state must match before/after. Training, inference, target, decoder and evaluation implementations remain unchanged; instrumentation lives in new diagnostic helpers/entry point only. Existing A/R code and weights remain frozen. This is an invariance control, not a fresh A/R quality measurement or proof of improvement.
- A synthetic tiny-model observer-on/off check must preserve all returned outputs including selected R logits, and the same-point official loss and assignment must be preserved by the reporting helper. Fixed-input replay must reproduce private numerical summaries without another image-based forward.
- Private output root: a new0700directory under `/home/congcongliu/outputs/aur_u_train_supervision_20261018/`; refuse accidental overwrite. Selection, row/crop/query/component identities, derived masks, sampled logits/indices/targets and raw provenance stay private. Do not emit them in exceptions or terminal progress.
- Public outputs under `docs/verification/2026-10-18/aur_u_train_supervision/`: an anonymous aggregate, RESULTS.md and RUN.json with commands, hashes, environment, timings, memory and checks. Keep the metadata-only `preflight.json` separate from executed-model results. Aggregate by source/sequence, realized crop GT status, query matched/unmatched status, and official component TP/FP/ignored status; always show counts/denominators and undefined strata.
- Primary report tables: presence/assignment counts; sampled coverage and gradient-direction distributions for FP components; matched-query leakage versus unmatched-query FP cross-tab; two-draw matching stability; dense mask-quality summaries. Include query-weighted and component-weighted summaries when a query generates multiple components; do not treat components as independent patients.
- Cache enough private numeric inputs to recompute matching, sampled-loss gradients and public aggregates on CPU. Save sampled logits as float32, not lossy float16. Validate archive hashes/schema/provenance before reuse, and define numerical tolerances explicitly if a reconstruction cannot be bitwise exact. No resume from a partial or mismatched cache.

All results are NOT_EVIDENCE. A/R remain pseudo-reference tasks without Level R human evidence. Even zero TRAIN FP in these crops would not overturn complete405val FAIL; if no fixed-threshold FP occurs, report the absence and context mismatch without expanding the sample after inspection. A local suppressive gradient is not proof that more weight or training is the correct fix. A positive result permits a later separately frozen single-variable experiment; a null result completes this diagnostic too.

## Resources and acceptance

Use the existing nvgen Python with PYTHONNOUSERSITE=1 and explicit project PYTHONPATH; no dependencies installed. CPU nice19, numerical threads2, decoder workers<=2, volumes serially loaded. Before selecting a GPU, respect scheduler/device visibility and inspect current utilization, memory and workloads; choose a compatible idle device with enough memory, otherwise the lightest eligible load, preserving existing jobs. Existing standing GPU authorization applies.

First run one of the twelve fixed crops as a resource/control pilot and reuse its complete valid result. Maximum1800seconds for model measurement/decoding, enforced by an external timeout and internal deadline. Record actual peak allocated/reserved CUDA memory and CPU RSS. On OOM, invalid geometry/provenance, nonfinite values, contract mismatch or exhausted budget, stop with an explicit incomplete result; do not change crop/model/thresholds, silently drop queries or choose replacement patients. Tiny synthetic checks can run separately; the bound concerns this fixed image-based diagnostic. No broader training campaign is authorized by this design.

Before handoff require:

1. All frozen commitments and split exclusions verified; six fixed patients/rows and12crops accounted for, including empty/crowded crops. Raw inputs are read-only; privacy/schema checks pass.
2. Actual checkpoint profile and64queries verified. Training-unmatched status and detection FP status are independently computed and correctly joined through IDs and both point draws.
3. Actual same-point loss/matches agree with the reporting helper; gradient signs/weights and component point coverage are independently recomputed on synthetic fixtures. Meaningful tests cover U-unsupervised samples, empty GT, ignored/sub-floor regions, duplicate points, GT>queries, matched-query FP, unmatched-query TP, ID filtering and changed input/cache hashes.
4. Tiny model outputs and state are unchanged with diagnostic collection; private CPU replay reconstructs the measured summaries and no deployment/performance verdict is altered.
5. Focused diagnostic tests and the project's complete CPU suite pass, with environment-specific skips reported accurately. Run nnDetection/CUDA supplemental checks only if their relevant code paths change, not merely to repeat an unchanged prior result.
6. Fresh code and scientific review of the complete change and evidence; resolve substantive findings, maximum three automatic review/fix rounds. Report any remaining block rather than proceeding to an adoption claim.
7. Write the observed result and next diagnostic decision with limitations. Update STATUS/CLAUDE only after results exist; keep A/R tasks visible and distinguish diagnosis-complete from model-fixed.

After design approval, write an implementation plan. Suggested new boundaries are `anatobind/aur/u_supervision_diagnostics.py`, `scripts/aur_diagnose_u_supervision.py` and focused tests; the builder may choose equivalent narrow names. Existing production model/trainer/decoder semantics must remain unchanged. No implementation has been created by this design/preflight stage.
