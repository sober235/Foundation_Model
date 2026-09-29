# Brain multi-disease detector: metastasis (Dataset905_BMSRMetastasis), folds [1]

Fold subset: an early reading, NOT the gate (spec M4).

## Verdict

```json
{
 "kind": "early_reading",
 "folds": [
  1
 ],
 "pass": null,
 "operating_point": true,
 "sensitivity": 0.7189655172413794,
 "thr": 0.8,
 "fp_per_scan": 0.45348837209302323,
 "beyond_budget": null,
 "stop_remaining_folds": false,
 "early_stop_undecided": false
}
```

Scans 86; ground-truth lesions counted 580; ignored (< 10 mm3) 91.

Scores are mean foreground probabilities over components of the argmax map, so they exceed 0.5 by construction: the rows up to threshold 0.50 are identical and the budget of 2 false positives per scan may not be reached.

## FROC

| thr | n_hit | sensitivity | FP per scan |
|---|---|---|---|
| 0.05 | 417 | 0.7190 | 0.5814 |
| 0.10 | 417 | 0.7190 | 0.5814 |
| 0.15 | 417 | 0.7190 | 0.5814 |
| 0.20 | 417 | 0.7190 | 0.5814 |
| 0.25 | 417 | 0.7190 | 0.5814 |
| 0.30 | 417 | 0.7190 | 0.5814 |
| 0.35 | 417 | 0.7190 | 0.5814 |
| 0.40 | 417 | 0.7190 | 0.5814 |
| 0.45 | 417 | 0.7190 | 0.5814 |
| 0.50 | 417 | 0.7190 | 0.5814 |
| 0.55 | 417 | 0.7190 | 0.5814 |
| 0.60 | 417 | 0.7190 | 0.5698 |
| 0.65 | 417 | 0.7190 | 0.5581 |
| 0.70 | 417 | 0.7190 | 0.5349 |
| 0.75 | 417 | 0.7190 | 0.5233 |
| 0.80 | 417 | 0.7190 | 0.4535 |
| 0.85 | 414 | 0.7138 | 0.4419 |
| 0.90 | 401 | 0.6914 | 0.3837 |
| 0.95 | 323 | 0.5569 | 0.2093 |

## Dice (report only, nnU-Net summary.json, cases with ground truth)

```json
{
 "n_cases": 86,
 "mean": 0.8274601104276936,
 "median": 0.8661311914323963
}
```

## False positives per scan at the operating threshold (report only)

```json
{
 "n_scans": 86,
 "median": 0.0,
 "max": 6,
 "n_scans_over_budget": 4
}
```

## Strata at the operating threshold (report only)

### Equivalent diameter (mm)

| stratum | N (GT) | N (hit) | sensitivity |
|---|---|---|---|
| <5 | 336 | 188 | 0.5595 |
| 5-10 | 152 | 139 | 0.9145 |
| >=10 | 92 | 90 | 0.9783 |

### Prior craniotomy, biopsy or resection

| stratum | N (GT) | N (hit) | sensitivity |
|---|---|---|---|
| no | 474 | 338 | 0.7131 |
| yes | 106 | 79 | 0.7453 |

## Binding agreement (NOT_EVIDENCE: the anatomy is a SynthSeg pseudo-label)

```json
{
 "n_pairs": 417,
 "host_agreement": 0.9808153477218226,
 "side_agreement": 0.9928057553956835,
 "host_side_agreement": 0.9952038369304557,
 "n_detections": 464,
 "no_host_rate": 0.0,
 "nearest_rate": 0.0,
 "unlocated_rate": 0.0
}
```

## Command

```
scripts/eval_brain_disease.py --disease metastasis --folds 1 --workers 2 --out docs/verification/2026-09-29/brain_multidisease/metastasis_fold1_preliminary
```

Code: commit 7c4ae27
