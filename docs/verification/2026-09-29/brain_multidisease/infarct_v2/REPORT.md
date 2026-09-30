# Brain multi-disease detector: infarct (Dataset906_ISLESInfarct), folds [0, 1, 2, 3, 4]

All five folds: the line below is the gate (spec M2).

## Verdict

```json
{
 "kind": "gate",
 "folds": [
  0,
  1,
  2,
  3,
  4
 ],
 "pass": true,
 "operating_point": true,
 "sensitivity": 0.5722406442444339,
 "thr": 0.5,
 "fp_per_scan": 1.556,
 "beyond_budget": null,
 "stop_remaining_folds": false,
 "early_stop_undecided": false
}
```

Scans 250; ground-truth lesions counted 2111; ignored (< 10 mm3) 238.

Scores are mean foreground probabilities over components of the argmax map, so they exceed 0.5 by construction: the rows up to threshold 0.50 are identical and the budget of 2 false positives per scan may not be reached.

## FROC

| thr | n_hit | sensitivity | FP per scan |
|---|---|---|---|
| 0.05 | 1208 | 0.5722 | 1.5560 |
| 0.10 | 1208 | 0.5722 | 1.5560 |
| 0.15 | 1208 | 0.5722 | 1.5560 |
| 0.20 | 1208 | 0.5722 | 1.5560 |
| 0.25 | 1208 | 0.5722 | 1.5560 |
| 0.30 | 1208 | 0.5722 | 1.5560 |
| 0.35 | 1208 | 0.5722 | 1.5560 |
| 0.40 | 1208 | 0.5722 | 1.5560 |
| 0.45 | 1208 | 0.5722 | 1.5560 |
| 0.50 | 1208 | 0.5722 | 1.5560 |
| 0.55 | 1206 | 0.5713 | 1.5480 |
| 0.60 | 1205 | 0.5708 | 1.5200 |
| 0.65 | 1203 | 0.5699 | 1.4560 |
| 0.70 | 1199 | 0.5680 | 1.4160 |
| 0.75 | 1191 | 0.5642 | 1.3360 |
| 0.80 | 1172 | 0.5552 | 1.2480 |
| 0.85 | 1139 | 0.5396 | 1.0760 |
| 0.90 | 1071 | 0.5073 | 0.8640 |
| 0.95 | 862 | 0.4083 | 0.5560 |

## Dice (report only, nnU-Net summary.json, cases with ground truth)

```json
{
 "n_cases": 247,
 "mean": 0.7847471491920139,
 "median": 0.8507137192704203
}
```

## False positives per scan at the operating threshold (report only)

```json
{
 "n_scans": 250,
 "median": 1.0,
 "max": 15,
 "n_scans_over_budget": 54
}
```

## Strata at the operating threshold (report only)

### Equivalent diameter (mm)

| stratum | N (GT) | N (hit) | sensitivity |
|---|---|---|---|
| <5 | 1026 | 345 | 0.3363 |
| 5-10 | 682 | 508 | 0.7449 |
| >=10 | 403 | 355 | 0.8809 |

## Binding agreement (NOT_EVIDENCE: the anatomy is a SynthSeg pseudo-label)

```json
{
 "n_pairs": 1208,
 "host_agreement": 0.9627483443708609,
 "side_agreement": 0.9933774834437086,
 "host_side_agreement": 0.9983443708609272,
 "n_detections": 1601,
 "no_host_rate": 0.0,
 "nearest_rate": 0.012492192379762648,
 "unlocated_rate": 0.0
}
```

## Command

```
scripts/eval_brain_disease.py --disease infarct --folds 0 1 2 3 4 --workers 8 --out docs/verification/2026-09-29/brain_multidisease/infarct_v2 --records /data2/congcong/data/FM_data/derived/brain_disease/infarct/records_v2
```

Code: commit c8b1a11
