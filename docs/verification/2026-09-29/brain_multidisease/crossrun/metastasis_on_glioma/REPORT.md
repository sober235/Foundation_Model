# Cross false-alarm check: metastasis model on glioma data (report only, spec M11)

The model never saw this dataset. Resolution, preprocessing and scanners differ from its training data, so these numbers describe this pair of datasets, not the diseases in general.

Counting: a detection is on ground truth when it shares a voxel with any labelled voxel, fragments under 10 mm3 included; a ground-truth lesion is claimed when a detection shares a voxel with it, and only lesions of at least 10 mm3 are counted. These are counts of overlap: not a sensitivity, not a precision and not a false-positive rate.

Threshold: it is the model's operating threshold, measured on single-fold models (every case predicted by the fold that held it out). Here the model's folds are averaged; the behaviour of the averaged model at this threshold was not measured on its own data.

```json
{
 "model": "metastasis",
 "data": "glioma",
 "cases": "fold 0 validation cases of Dataset904_PDGMGlioma",
 "channels": [
  "T1",
  "T1c",
  "FLAIR"
 ],
 "note": null,
 "threshold": 0.65,
 "model_folds": [
  0,
  1,
  2,
  3,
  4
 ],
 "n_scans": 100,
 "n_det": 167,
 "n_det_on_gt": 153,
 "n_gt": 144,
 "n_gt_claimed": 95,
 "det_per_scan": 1.67,
 "scans_with_any_detection": 88,
 "share_of_detections_on_gt": 0.9161676646706587,
 "share_of_gt_claimed": 0.6597222222222222
}
```

## Command

```
scripts/brain_disease_crossrun.py --model metastasis --data glioma --threshold 0.65 --gpu 5 --work /data2/congcong/data/FM_data/derived/brain_disease/crossrun/metastasis_on_glioma --out docs/verification/2026-09-29/brain_multidisease/crossrun/metastasis_on_glioma
```

Code: commit 9a89ff1
