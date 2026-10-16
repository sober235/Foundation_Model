# Saved validation U diagnostic

**NOT_EVIDENCE — exploratory saved-validation diagnostic.**

The new and old C2 evaluations have identical row identities and saved GT definitions. Only cases.json and aggregate.json were read from the supplied evaluation directories. No model, checkpoint, GPU, raw image, sealed test inference, or new threshold search was used.

Each arm uses its own already selected validation score threshold. Differences are descriptive; they do not isolate the training-loss effect from the changed operating point. The saved evaluator labels validation threshold selection as 'test-selected'; run.split=val is checked explicitly here.

| arm | source/sequence | scans | GT | score thr | raw / kept dets | hits / FP | raw ≥.1 box coverage | kept ≥.1 box coverage |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| old | bmsr/T1c | 41 | 470 | 0.7 | 3539 / 11 | 6 / 5 | 18 | 6 |
| old | isles/DWI | 20 | 138 | 0.7 | 973 / 6 | 2 / 4 | 8 | 2 |
| old | pdgm/FLAIR | 41 | 53 | 0.7 | 1514 / 37 | 33 / 4 | 40 | 34 |
| new | bmsr/T1c | 41 | 470 | 0.7 | 3661 / 5 | 2 / 2 | 16 | 2 |
| new | isles/DWI | 20 | 138 | 0.6 | 1254 / 28 | 4 / 24 | 6 | 4 |
| new | pdgm/FLAIR | 41 | 53 | 0.75 | 4273 / 33 | 27 / 6 | 34 | 27 |

Every recomputed scan count, GT count, ignored count, hit count, FP count, sensitivity and FP/scan equals its original aggregate value. Matching reuses the repository's one-to-one Hungarian box IoU ≥0.1 rule, with ignored GT matched after counted GT.

| arm | source/sequence | raw largest component mL p10 / median / p90 | raw >100 mL volumes | raw / kept bbox containment with IoU <.1 | missed GT oracle query IoU median / p90 |
|---|---|---:|---:|---:|---:|
| old | bmsr/T1c | 402.2 / 534.1 / 645.1 | 41/41 | 470 / 0 | 0.002248 / 0.03227 |
| old | isles/DWI | 359.4 / 420.1 / 520.8 | 20/20 | 136 / 0 | 0.003642 / 0.07574 |
| old | pdgm/FLAIR | 33.97 / 105.1 / 180.2 | 24/41 | 11 / 8 | 0.2016 / 0.8049 |
| new | bmsr/T1c | 379.7 / 473.7 / 572.1 | 41/41 | 470 / 0 | 0.002161 / 0.02515 |
| new | isles/DWI | 241 / 350.4 / 441.5 | 20/20 | 136 / 111 | 0.003189 / 0.06913 |
| new | pdgm/FLAIR | 53.59 / 141.6 / 209.7 | 30/41 | 20 / 9 | 0.4458 / 0.7466 |

## Findings supported by the saved validation data

- bmsr/T1c: the new saved components provide any box IoU ≥0.1 for 16/470 GT lesions (3.404%), before score filtering; only 2 are one-to-one hits at the existing threshold. Old raw coverage was 18/470. Score filtering alone cannot supply boxes absent from the saved component set.
- bmsr/T1c: the raw per-volume largest-component median is 473.7 mL (old 534.1 mL); 41/41 volumes exceed 100 mL. These component-size and box-geometry observations support a coarse component/localization failure pattern, while leaving voxelwise lesion overlap unresolved.
- isles/DWI: the new saved components provide any box IoU ≥0.1 for 6/138 GT lesions (4.348%), before score filtering; only 4 are one-to-one hits at the existing threshold. Old raw coverage was 8/138. Score filtering alone cannot supply boxes absent from the saved component set.
- isles/DWI: the raw per-volume largest-component median is 350.4 mL (old 420.1 mL); 20/20 volumes exceed 100 mL. These component-size and box-geometry observations support a coarse component/localization failure pattern, while leaving voxelwise lesion overlap unresolved.
- pdgm/FLAIR: the new saved components provide any box IoU ≥0.1 for 34/53 GT lesions (64.15%), before score filtering; only 27 are one-to-one hits at the existing threshold. Old raw coverage was 40/53. Score filtering alone cannot supply boxes absent from the saved component set.
- pdgm/FLAIR: the raw per-volume largest-component median is 141.6 mL (old 105.1 mL); 30/41 volumes exceed 100 mL. These component-size and box-geometry observations support a coarse component/localization failure pattern, while leaving voxelwise lesion overlap unresolved.

For BMSR and ISLES, local query masks also have poor IoU under this distinct saved oracle metric. These values cannot locate the same fault as the whole-volume box metric, identify a causal root cause, isolate window merging or score filtering, or measure end-to-end R.

## Interpretation and evidence limits

Raw here means all saved detections, which are already connected components after the saved instance probability threshold and component processing. It is not the unthresholded probability map. The >100 mL statistic is descriptive and is not a new acceptance gate. Empty prediction volumes contribute zero to the per-volume maximum distribution.

Raw ≥0.1 box coverage bounds what score filtering could localize from these saved components. Low raw coverage cannot be repaired merely by retaining more of those same components. The raw-to-kept difference describes additional score filtering losses. Coverage allows many GT boxes to share one predicted box; the actual hit count additionally applies one-to-one matching.

The bbox-containment count uses each volume's largest component only when its saved mask volume exceeds 100 mL: its box covers ≥90% of a GT box while their box IoU is <0.1. This is bbox geometry, not voxelwise evidence that a predicted mask swallowed a lesion. Saved boxes cannot establish mask contamination, probability calibration, voxelwise lesion overlap, or the causal role of the U loss.

The saved query IoU is a distinct oracle metric (anatobind/aur/infer.py, bind_instances): (1) each crop is centered on the GT instance's mean voxel position; (2) all event queries are considered without the presence gate; (3) each query mask is binarized with probability >0.5; (4) query_iou is the maximum mask IoU against the GT instance inside that crop (when every hard IoU is zero, selection uses maximum soft overlap and query_iou stays zero). This differs from whole-volume saved components produced at the saved instance probability threshold (0.3 here), then scored with one-to-one bbox IoU ≥0.1 and the saved score threshold. These different thresholds, geometry metrics and conditioning do not identify the same fault or a causal root cause. This is not end-to-end detection or binding ability. Missing R values are counted explicitly in summary.json. All distributions (min, p10, p25, median, p75, p90, max), kept-component volume statistics and both raw/kept containment counts are in the anonymous JSON.

Further model/image diagnostics would be necessary to distinguish voxelwise oversegmentation from box extent artifacts and to attribute failures to training, sliding-window probability merging, or component decoding. This audit neither performs nor authorizes such work.

## Reproduce

```sh
CUDA_VISIBLE_DEVICES='' PYTHONNOUSERSITE=1 PYTHONPATH=. OMP_NUM_THREADS=1 nice -n19 /home/congcongliu/anaconda3/envs/nvgen/bin/python /home/congcongliu/src/Foundation_Model-conformance-work/scripts/aur_audit_saved_eval.py --new-eval /data0/congcong/code/Project_Doing/foundation_model-conformance/docs/verification/2026-10-09/anatobind_brain_ssl_first/eval/c2_val_20261014_1344 --old-eval /data0/congcong/code/Project_Doing/foundation_model/docs/verification/2026-10-09/anatobind_brain_ssl_first/eval/c2_val_20261011_0500 --out /tmp/aur_saved_eval_diagnostic_oracle_corrected_20261016_reproduction
```

The output directory must not already exist. Source file SHA256 values are recorded below and in summary.json.

- new cases.json: `5950f91d1070dc47f9f3770c3f03bc6a5a3e5b3d0113a2d0109b3e3522f379bf` (`/data0/congcong/code/Project_Doing/foundation_model-conformance/docs/verification/2026-10-09/anatobind_brain_ssl_first/eval/c2_val_20261014_1344/cases.json`)
- new aggregate.json: `9b83bc1d8d6b3116962e400d5b6aaf103c17408e004adee842fc8824baa05670` (`/data0/congcong/code/Project_Doing/foundation_model-conformance/docs/verification/2026-10-09/anatobind_brain_ssl_first/eval/c2_val_20261014_1344/aggregate.json`)
- old cases.json: `61a87ded7491c25f9746f188bf798d6e073445eefb7ad970d5ef61d33f5e8e75` (`/data0/congcong/code/Project_Doing/foundation_model/docs/verification/2026-10-09/anatobind_brain_ssl_first/eval/c2_val_20261011_0500/cases.json`)
- old aggregate.json: `124b4b623c0d0e54ad29acbe308f2a2958d0530b748d1edd9a323ea34a743fd2` (`/data0/congcong/code/Project_Doing/foundation_model/docs/verification/2026-10-09/anatobind_brain_ssl_first/eval/c2_val_20261011_0500/aggregate.json`)
- audit_code aur_audit_saved_eval.py: `a89e3d6ad96f09a8e0828204735a48334fbf7e4b027bca91bce08f777ccb4183` (`/home/congcongliu/src/Foundation_Model-conformance-work/scripts/aur_audit_saved_eval.py`)
- audit_code detection_metrics.py: `54c5634af3e79b4939c27593d99e1e8f57542f6cbd24b532e16e9f41a02b5b9b` (`/home/congcongliu/src/Foundation_Model-conformance-work/anatobind/eval/detection_metrics.py`)
- audit_code matching.py: `729682d7aadcfd7c63597286b567583297f3f9eb5eee22cbd33e376c7e777fbb` (`/home/congcongliu/src/Foundation_Model-conformance-work/anatobind/eval/matching.py`)
- audit_code infer.py: `7e9e1898b99a421218b5a575ff4660bf87cd0e35e08b09aad627324ca3cfe414` (`/home/congcongliu/src/Foundation_Model-conformance-work/anatobind/aur/infer.py`)
