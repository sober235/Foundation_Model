# Brain Small-Lesion Detector: Out-of-Fold Evaluation
## Gate
```json
{
  "pass": false,
  "thr": 0.55,
  "sensitivity_family": 0.3662297609868928,
  "fp_per_scan": 1.6363636363636365
}
```

## FROC Curve

| Threshold | Sensitivity | Sensitivity (Family) | FP per Scan |
|-----------|-------------|----------------------|-------------|
| 0.05 | 0.3662 | 0.3662 | 1.6759 |
| 0.10 | 0.3662 | 0.3662 | 1.6759 |
| 0.15 | 0.3662 | 0.3662 | 1.6759 |
| 0.20 | 0.3662 | 0.3662 | 1.6759 |
| 0.25 | 0.3662 | 0.3662 | 1.6759 |
| 0.30 | 0.3662 | 0.3662 | 1.6759 |
| 0.35 | 0.3662 | 0.3662 | 1.6759 |
| 0.40 | 0.3662 | 0.3662 | 1.6759 |
| 0.45 | 0.3662 | 0.3662 | 1.6759 |
| 0.50 | 0.3662 | 0.3662 | 1.6759 |
| 0.55 | 0.3662 | 0.3662 | 1.6364 |
| 0.60 | 0.3639 | 0.3639 | 1.5692 |
| 0.65 | 0.3585 | 0.3585 | 1.5257 |
| 0.70 | 0.3554 | 0.3554 | 1.4743 |
| 0.75 | 0.3516 | 0.3516 | 1.4308 |
| 0.80 | 0.3446 | 0.3446 | 1.3518 |
| 0.85 | 0.3362 | 0.3362 | 1.2292 |
| 0.90 | 0.3092 | 0.3092 | 1.0119 |
| 0.95 | 0.2128 | 0.2128 | 0.5613 |

## Normal-Volume False Positives

At operating threshold 0.55: 0.3523 FP per volume

## Strata

### Distance Band

| Band | N (GT) | N (Hit) | Sensitivity |
|------|--------|--------|-------------|
| 0 | 545 | 201 | 0.3688 |
| 0-2 | 210 | 72 | 0.3429 |
| 2-4 | 217 | 80 | 0.3687 |
| >4 | 325 | 122 | 0.3754 |

### Number of Slices

| Type | N (GT) | N (Hit) | Sensitivity |
|------|--------|--------|-------------|
| 1 slice(s) | 1029 | 346 | 0.3362 |
| >1 slice(s) | 268 | 129 | 0.4813 |

### In-Plane Size Tertile

| Tertile | N (GT) | N (Hit) | Sensitivity |
|---------|--------|--------|-------------|
| tertile_1 | 516 | 205 | 0.3973 |
| tertile_2 | 379 | 138 | 0.3641 |
| tertile_3 | 402 | 132 | 0.3284 |

### Stratum Geometry

| Geometry | N (GT) | N (Hit) | Sensitivity |
|----------|--------|--------|-------------|
| inplane_0.62_slice_3 | 32 | 14 | 0.4375 |
| inplane_0.69_slice_5 | 1175 | 421 | 0.3583 |
| inplane_0.86_slice_3 | 11 | 4 | 0.3636 |
| inplane_0.86_slice_5 | 79 | 36 | 0.4557 |

## Command

```
scripts/eval_brain_detector.py --config 2d --out docs/verification/2026-09-28/brain_detector/2d
```
