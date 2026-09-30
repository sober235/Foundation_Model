# Cross false-alarm check: glioma model on metastasis data (report only, spec M11)

The model never saw this dataset. Resolution, preprocessing and scanners differ from its training data, so these numbers describe this pair of datasets, not the diseases in general.

Counting: a detection is on ground truth when it shares a voxel with any labelled voxel, fragments under 10 mm3 included; a ground-truth lesion is claimed when a detection shares a voxel with it, and only lesions of at least 10 mm3 are counted. These are counts of overlap: not a sensitivity, not a precision and not a false-positive rate.

Threshold: it is the model's operating threshold, measured on single-fold models (every case predicted by the fold that held it out). Here the model's folds are averaged; the behaviour of the averaged model at this threshold was not measured on its own data.

```json
{
 "model": "glioma",
 "data": "metastasis",
 "cases": "fold 0 validation cases of Dataset905_BMSRMetastasis",
 "channels": [
  "T1pre",
  "T1post",
  "T2Synth",
  "FLAIR"
 ],
 "note": "channel 2 (T2) is BMSR's synthetic T2",
 "threshold": 0.6,
 "model_folds": [
  0,
  1,
  2,
  3,
  4
 ],
 "n_scans": 105,
 "n_det": 378,
 "n_det_on_gt": 164,
 "n_gt": 729,
 "n_gt_claimed": 222,
 "det_per_scan": 3.6,
 "scans_with_any_detection": 105,
 "share_of_detections_on_gt": 0.43386243386243384,
 "share_of_gt_claimed": 0.3045267489711934
}
```

## Command

```
scripts/brain_disease_crossrun.py --model glioma --data metastasis --threshold 0.6 --gpu 6 --work /data2/congcong/data/FM_data/derived/brain_disease/crossrun/glioma_on_metastasis --out docs/verification/2026-09-29/brain_multidisease/crossrun/glioma_on_metastasis
```

Code: commit 9a89ff1
