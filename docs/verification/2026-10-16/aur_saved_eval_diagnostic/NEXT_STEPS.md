# U residual failure: audit conclusion and next diagnostic

**Status: recommendation after the saved-output audit; the model replay below has not run. NOT_EVIDENCE.**

## What this audit establishes

- The original A / U / R goals and final acceptance gates remain in force. The `unmatched_empty_v1` run failed all three gates and is closed; its preregistered descriptive comparison is [recorded separately](../../2026-10-09/anatobind_brain_ssl_first/eval/compare_c2empty_c2legacy_20261016/COMPARE.md).
- [Saved validation outputs](REPORT.md) rule out score filtering alone as a sufficient repair for BMSR and ISLES: before that filtering, the existing component boxes cover only 16/470 and 6/138 counted GT lesions at box IoU >= 0.1. These are per-GT coverage upper bounds, not one-to-one sensitivities.
- Every BMSR and ISLES validation scan contains a saved raw component larger than 100 mL. This is a descriptive observation, not a new failure gate. Component bounding boxes do not establish voxelwise lesion overlap or the cause of the large components.
- The distinct GT-centred oracle metric also reports low query-mask IoU. It uses all queries, no presence filter, and mask threshold 0.5; it cannot be equated to normal whole-volume detection with mask threshold 0.3 and box matching.
- Code inspection found no definite axis, physical-volume, padding, valid-mask or supervision-routing defect. This does not prove that the implementation or training recipe is correct.

## Next diagnostic: freeze six validation cases and observe the original path

This is a proposed bounded diagnostic, not a new training recipe or a change to the acceptance gates. The following selection and resource choices are the reviewer's proposal, not previously approved user decisions.

1. Before opening images or running inference, select two U-supervised validation volumes with at least one non-ignored GT lesion from each of BMSR/T1c, ISLES/DWI and PDGM/FLAIR. For each row, serialize `[0, source, patient, case, sequence]` with Python `json.dumps(..., ensure_ascii=False, separators=(',', ':'))`, encode as UTF-8 without a newline, and sort by its SHA256 hexadecimal digest (identity tuple as the tie-breaker). Take the first two distinct patients in each group; do not pick cases for attractive predictions. Keep the case list locally and publish only its hash and source counts. Reject any patient outside the frozen validation manifest.
2. Pin `/data2/congcong/data/FM_data/derived/aur/ssl_runs/c2_stage3_80k_20261014_1344/aur_stage3_best.pt`, whose evaluated SHA256 is `38f6376333041a75be1cce83e927bc221468761f80ae952e2a3020805f7dcbde`; verify it before replay. Preserve baseline code `77fd6ebc81ec945f77f7c9316f05119294b1aca7`, sample-manifest SHA256 `fb96acd741aa6c0ee689eb772a4332304501c8d63ebd7ccddcd657f41eb86d5d`, crop `[128,160,160]` and original inference settings. Replay the original full-volume sliding windows without GT-based window selection. GT is used only for after-the-fact measurements. Start with one volume to measure runtime and peak memory; run sequentially on one available GPU, capped at two hours for the six cases. Stop and report if the cap cannot be met.
3. Retain per-window anonymous summaries and local-only query outputs: presence probability, query mask volume, GT-inside/outside probabilities, best single-query mask IoU, and counts before/after presence >0.5. Compare the selected-query maximum with the final weighted window average and its 0.3 connected components. Account explicitly for valid voxels, padding and window weights. Keep original predictions unchanged.
4. Before interpreting new diagnostics, compare replayed detections against the selected cases' saved baseline, sorted by box coordinates and voxel count: require exact component count, boxes and voxel counts; absolute volume tolerance 1e-3 mm³ and absolute score tolerance 1e-5. These proposed tolerances must be frozen before replay. A mismatch pauses attribution until checkpoint, input, numeric precision and inference settings agree; do not relax tolerance after seeing results. Six cases are for mechanism inspection and cannot establish population performance.

## How the observations determine the next experiment

| Observation under the frozen replay | Interpretation supported | Next candidate, requiring a separate written experiment contract |
|---|---|---|
| Useful individual query masks exist but fail the presence filter | Presence scoring is a candidate source of lost detections | A validation-only presence intervention with a fixed comparison |
| Individual query masks separate lesions, but max/averaging/components merge or suppress them | A specific aggregation stage loses usable information | A single decoding intervention; specify cross-window deduplication and R-token ownership |
| Individual query masks already localize poorly | The failure precedes whole-volume component scoring | Inspect fixed training crops, supervision and optimization before altering training |
| Saved baseline cannot be reproduced | Diagnostic setup differs from the evaluated run | Resolve provenance/numerics first; make no model-quality claim |

Do not perform a threshold sweep on the test set, silently change the U target, or launch another full training run from this audit. Query-instance decoding needs a separate design for cross-window identity and R binding. The existing N7 semantic fallback changes U training and R token construction and requires the user notification specified in the original design; it is not activated here.

## Relation to the other goals

U is the immediate diagnostic priority because detection is far below its reference. A still fails 0.80, and controlled R still trails its own B0; both remain open after U. Human Level R validation, offline DICOM ingestion and an independent 2D model remain uncompleted delivery work. API, web display and public URL remain deferred under Q28/Q30.

## Code inspected

- `anatobind/aur/dataset.py`: valid-area and supervision masks.
- `anatobind/aur/losses.py`: matched/empty mask loss and presence weighting.
- `anatobind/aur/infer.py`: presence filter, per-query maximum, window averaging, connected components and GT-given oracle queries.
- `anatobind/aur/eval.py`, `anatobind/eval/detection_metrics.py`, `anatobind/eval/matching.py`: saved outputs, ignored GT and one-to-one box matching.

The saved outputs do not contain probability maps or window-query caches. Existing references to `scratchpad/udiag/{lesion_contrast.py,query_probe.py}` resolve to a scratch directory that is no longer present, so those historical probes were not treated as reproducible evidence for this checkpoint.
