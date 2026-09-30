# Cross false-alarm check: infarct model on glioma data (report only, spec M11)

The model never saw this dataset. Resolution, preprocessing and scanners differ from its training data, so these numbers describe this pair of datasets, not the diseases in general.

Counting: a detection is on ground truth when it shares a voxel with any labelled voxel, fragments under 10 mm3 included; a ground-truth lesion is claimed when a detection shares a voxel with it, and only lesions of at least 10 mm3 are counted. These are counts of overlap: not a sensitivity, not a precision and not a false-positive rate.

Threshold: it is the model's operating threshold, measured on single-fold models (every case predicted by the fold that held it out). Here the model's folds are averaged; the behaviour of the averaged model at this threshold was not measured on its own data.

```json
{
 "model": "infarct",
 "data": "glioma",
 "cases": "fold 0 validation cases of Dataset904_PDGMGlioma",
 "channels": [
  "DWI",
  "ADC"
 ],
 "note": null,
 "threshold": 0.5,
 "model_folds": [
  0,
  1,
  2,
  3,
  4
 ],
 "n_scans": 100,
 "n_det": 46,
 "n_det_on_gt": 25,
 "n_gt": 144,
 "n_gt_claimed": 16,
 "det_per_scan": 0.46,
 "scans_with_any_detection": 36,
 "share_of_detections_on_gt": 0.5434782608695652,
 "share_of_gt_claimed": 0.1111111111111111
}
```

## Command

```
scripts/brain_disease_crossrun.py --model infarct --data glioma --threshold 0.5 --gpu 4 --work /data2/congcong/data/FM_data/derived/brain_disease/crossrun/infarct_on_glioma --out docs/verification/2026-09-29/brain_multidisease/crossrun/infarct_on_glioma
```

Code: commit 9a89ff1
