# Brain multi-disease detector: glioma (Dataset904_PDGMGlioma), folds [0, 1, 2, 3, 4]

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
 "sensitivity": 0.810930576070901,
 "thr": 0.6,
 "fp_per_scan": 0.34930139720558884,
 "beyond_budget": null,
 "stop_remaining_folds": false,
 "early_stop_undecided": false
}
```

Scans 501; ground-truth lesions counted 677; ignored (< 10 mm3) 259.

Scores are mean foreground probabilities over components of the argmax map, so they exceed 0.5 by construction: the rows up to threshold 0.50 are identical and the budget of 2 false positives per scan may not be reached.

## FROC

| thr | n_hit | sensitivity | FP per scan |
|---|---|---|---|
| 0.05 | 549 | 0.8109 | 0.3932 |
| 0.10 | 549 | 0.8109 | 0.3932 |
| 0.15 | 549 | 0.8109 | 0.3932 |
| 0.20 | 549 | 0.8109 | 0.3932 |
| 0.25 | 549 | 0.8109 | 0.3932 |
| 0.30 | 549 | 0.8109 | 0.3932 |
| 0.35 | 549 | 0.8109 | 0.3932 |
| 0.40 | 549 | 0.8109 | 0.3932 |
| 0.45 | 549 | 0.8109 | 0.3932 |
| 0.50 | 549 | 0.8109 | 0.3932 |
| 0.55 | 549 | 0.8109 | 0.3892 |
| 0.60 | 549 | 0.8109 | 0.3493 |
| 0.65 | 545 | 0.8050 | 0.2934 |
| 0.70 | 544 | 0.8035 | 0.2455 |
| 0.75 | 543 | 0.8021 | 0.1816 |
| 0.80 | 543 | 0.8021 | 0.1477 |
| 0.85 | 543 | 0.8021 | 0.1158 |
| 0.90 | 538 | 0.7947 | 0.0878 |
| 0.95 | 526 | 0.7770 | 0.0319 |

## Dice (report only, nnU-Net summary.json, cases with ground truth)

```json
{
 "n_cases": 501,
 "mean": 0.9279037878225983,
 "median": 0.9580363859744399
}
```

## False positives per scan at the operating threshold (report only)

```json
{
 "n_scans": 501,
 "median": 0.0,
 "max": 7,
 "n_scans_over_budget": 10
}
```

## Strata at the operating threshold (report only)

### Equivalent diameter (mm)

| stratum | N (GT) | N (hit) | sensitivity |
|---|---|---|---|
| <5 | 64 | 4 | 0.0625 |
| 5-10 | 36 | 5 | 0.1389 |
| >=10 | 577 | 540 | 0.9359 |

## Binding agreement (NOT_EVIDENCE: the anatomy is a SynthSeg pseudo-label)

```json
{
 "n_pairs": 549,
 "host_agreement": 0.9653916211293261,
 "side_agreement": 0.9981785063752276,
 "host_side_agreement": 0.9981785063752276,
 "n_detections": 725,
 "no_host_rate": 0.0,
 "nearest_rate": 0.008275862068965517,
 "unlocated_rate": 0.0
}
```

## Command

```
scripts/eval_brain_disease.py --disease glioma --folds 0 1 2 3 4 --workers 8 --out docs/verification/2026-09-29/brain_multidisease/glioma --records /data2/congcong/data/FM_data/derived/brain_disease/glioma/records
```

Code: commit 9a89ff1
