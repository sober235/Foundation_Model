# Unified 2D/3D AUR: implementation and bounded TRAIN evidence

Date: 2026-10-19. Starting revision: `86e938e`. User authorization and task ledger: [execution plan](../../../superpowers/plans/2026-10-19-aur-unified-input-execution.md).

**Outcome:** one opt-in model/checkpoint now accepts genuine depth-one 2D and 3D observations. The I→II→III technical training path, original-grid inference, strict migration and recovery are implemented and exercised. **Formal unified training is NOT READY; no new validation/test performance verdict or clinical claim is made.** All scientific measurements remain NOT_EVIDENCE.

## What changed

- `AnatoBindBrain(..., dual_input=True)` and `StageOne(..., dual_input=True)` use explicit `input_mode="2d"/"3d"`. Swin, A/U decoders and R are shared. Only the input and mask upsampling adapters differ in the final AUR model; SSL additionally has a dimension-specific reconstruction output.
- Legacy defaults and 3D parameter names are retained. Explicit migration refuses unexpected missing/extra keys, wrong shapes or dtypes. Tests against the actual original source show identical legacy keys, initialized parameters and 3D outputs, including masks and R.
- Derived 2D observations normalize the raw complete plane, retain real physical coordinates, and use plane-local coordinates without the unobserved source-volume extent. Visible components use an explicit area threshold; ignored components/padding do not become healthy targets. Patient splits precede derivation.
- R eligibility follows each instance through planar component splitting, renumbering and Hungarian matching. Unknown or out-of-plane relations are excluded, not relabelled no-host. Host0 remains valid.
- Classic scalar MR intake validates IOP/IPP/spacing, sorts regular spatial stacks physically, retains original-file hashes and frame mapping, and preserves the original bytes. Unsupported enhanced objects, ambiguous temporal/diffusion dimensions and missing geometry fail explicitly. The public entry does not infer absent geometry from slice thickness.
- A single loaded model runs both modes. 3D uses the legacy closest-canonical layout and maps outputs back; 2D reports area, never an invented volume. Outputs remain on the original input grid.
- The pilot runner is explicitly single-process and refuses formal/distributed use. It records modes, cursors, exposures, all RNG states, optimizer/scheduler, frozen data/config and numerical backend settings. Inactive adapters are not updated by AdamW's old moments or decay.

## Real TRAIN tiny pilot

The fixed engineering run used one eligible TRAIN patient/sequence from each of BMSR, ISLES and PDGM, true 2D `(1,32,32)` and 3D `(8,32,32)` crops, seed0,256 points, LR5e-4, contrast weight0 and the existing `unmatched_empty_v1` supervised profile. It initialized a **tiny random model**, not a migrated full Stage I checkpoint.

| Stage | Optimizer steps | 2D / 3D crop exposures | Early→late 2D training loss | Early→late 3D training loss |
|---|---:|---:|---:|---:|
| I |24|12 /12|0.1262→0.1082|0.1227→0.1163|
| II |24|12 /12|6.5535→4.4646|6.0799→4.6351|
| III |24|12 /12|5.4922→3.6904|8.2500→6.4899|

Loss entries are means of the first/last three observations of each mode in the fixed-crop training stream. They describe optimization, not generalization or final A/U/R quality. Stage III supervised16 three-dimensional R instances over the draws; real planar R eligibility was disabled because reliable plane labels have not been established. Synthetic tests exercise qualified planar R, including mixed eligibility.

The training-pipeline timer was16.397s; peak CUDA allocated144,058,880 bytes (about137.4MiB), reserved165,675,008 bytes. Source image/label and manifest hashes were unchanged. No val/test image inference occurred. The launch-time25-source snapshot was copied privately and independently matched its recorded digest before subsequent hardening changes. Checkpoints, selections and full logs stay private.

Evidence: [anonymous pilot](pilot_anonymous.json). Actual resource/setup/check commands and the launch-versus-final distinction are in [RUN.json](RUN.json).

## Recovery: exact state versus future CUDA updates

For all three stages, CPU8-step uninterrupted runs equal3-step-save/5-step-resume runs in complete checkpoint content, including parameters, optimizer and all RNG states. CUDA checkpoint restoration itself is also exact; the resumed mode/draw/RNG/exposure and optimizer counters match. The original future-loss tolerance remains rtol1e-5/atol1e-6.

The first stronger CUDA test failed when it also compared future parameters. Controlled identical uninterrupted CUDA runs diverged too (maximum parameter differences2.027e-6/3.451e-5/1.115e-4 for I/II/III), despite equal initial state and RNG. Strict deterministic mode raised `upsample_trilinear3d_backward_out_cuda does not have a deterministic implementation`; enabling cuDNN deterministic alone did not remove all drift. We did not widen tolerances or alter the production model to conceal this limit.

The final CUDA check therefore verifies the guarantee checkpointing can provide: exact restored state and backend contract, then the original loss tolerance and exact control-state continuation. It does **not** claim future CUDA parameter bitwise reproducibility. Missing/changed numerical environments are refused. Evidence: [numerical controls](cuda_recovery_anonymous.json).

## Frozen full-model A/R mechanism measurement

Three new TRAIN crops (one/source), standard `(128,160,160)` size, seed20261019, fixed all64 queries and float32 were measured using the existing full C2 checkpoint. Each crop used one shared backbone forward for A and R; no U decoder, new training, threshold selection, or validation/test forward was run. The contract may overlap earlier TRAIN patients and is not a new independent cohort.

The original native CPU replay **failed**: three elements of nearest-host-distance/100 exceeded its predeclared tolerance, while all R logits passed. Same-centroid controls identified cancellation in float32 squared-norm distance expansion. Its nine native members, missing original COMPLETE marker and failure are retained unchanged.

Under a separately written [stable-distance contract](../../../superpowers/specs/2026-10-19-aur-ar-stable-distance-analysis.md), a new CPU-only derived archive recomputes that distance using double-precision direct differences, with the original centroid definitions and head dtype. It refuses drift in all other14 features and native logits at the original tolerance. No image, checkpoint, backbone or GPU reload was needed. Stable v2 passed repeat replay; native9 and derived17 members remained byte-identical. Intermediate v1 is retained as superseded with its original source snapshot.

The A32 global mean hides five failed queries. All five had GT support in these crops:

| Aseg label / zero-based query | GT voxels | Mean probability at GT | Winner margin | Winner fraction |
|---|---:|---:|---:|---:|
|5 /6|2193|3.90e-8|−0.4473|0%|
|44 /7|1851|5.65e-8|−0.6748|0%|
|18 /25|5128|6.63e-5|−0.6705|0%|
|26 /28|2002|2.71e-8|−0.6433|0%|
|58 /29|2042|2.36e-9|−0.6512|0%|

Each also had nonzero sampled-positive coverage in this measurement. That does not establish its historical training coverage or identify a unique cause. Their very low probabilities show that a favorable overall A average is insufficient; sampled supervision, query identity/fitting and competition must be distinguished before a loss intervention.

For R, replacing only predicted coarse A/U geometry by aligned dense A/U geometry changed2/192 preselected query answers; maximum logit difference0.938039. Stable versus native logits differed by at most4.53e-6. This is **not GT accuracy**, rescue or a new R gate result. Joint A/U replacement cannot separate their contributions; a frozen head trained on coarse geometry also limits interpretation of a negative result.

Stable construction took13.826s; extra read-only CPU replay4.796s. Global-only evidence: [A/R aggregate](ar_anonymous.json). No per-patient, per-crop or per-query medical records are published.

## Evaluation preparation and remaining work

The metadata audit reconfirmed TRAIN3499 rows/894 source-qualified patients, validation405/101 and historical test1056/249, with no pairwise source-qualified identity intersection. This does not prove biological cross-source deduplication. No new independent evaluation cohort was established. The default Level R database path was absent; other databases/readers were not audited or contacted.

A bounded MR DICOM metadata sample exists but lacks IOP/IPP; it cannot validate native brain geometry or original-frame links. The existing4960-row derived manifest does not supply the audited DICOM-frame linkage fields. Derived-plane compatibility is not native-DICOM generalization.

Next formal work requires a full-model unified initialization/compute contract and joint per-mode G1 evaluation; native DICOM provenance/geometry; reliable planar R eligibility; independent patient evaluation and actual Level R readers. Keep2D/3D×A/U/R scores separate and do not turn unlabelled U planes into negatives. [Evaluation readiness](evaluation_readiness.json) records these requirements.

The existing full-model test failures remain: A0.7988, U three-source sensitivity failure, controlled R0.5606 below B0 0.7545. The legacy default decoder is unchanged. No new test success is asserted by this implementation.

## Verification and recovery

Final observed commands/counts, skipped environments, failed experiments and review resolutions are in [RUN.json](RUN.json). Independent review covered model/data/DICOM/training/privacy, with separate A/R review. The final anonymous-path fix has its own RED/GREEN regression; the automatic review/fix loop was capped at three rounds.

The full CPU suite passed **1512 tests,5 skipped,196.97s**. Separate CUDA validation passed4 in14.04s; the dedicated nnDetection environment passed2 in1.91s with6 third-party warnings. These are separate invocations, not a single combined passing count. The final error-message-only copy was additionally checked by its synthetic symlink regression (1 passed,3.37s).

Recovery is opt-in: omit `dual_input` and use the existing 3D entry points/checkpoints for the legacy path. Historical medical replays use their immutable frozen checkout; synthetic replay fixtures now freeze a temporary source tree instead of repinning historical hashes. Never overwrite original checkpoints or archives. No deployment, external reader messages or formal long training was performed.
