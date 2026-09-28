# Brain Small-Lesion Detector: Out-of-Fold Evaluation
## Gate
```json
{
  "pass": false,
  "thr": 0.6,
  "sensitivity_family": 0.36777178103315344,
  "fp_per_scan": 1.4308300395256917
}
```

## FROC Curve

| Threshold | Sensitivity | Sensitivity (Family) | FP per Scan |
|-----------|-------------|----------------------|-------------|
| 0.05 | 0.3678 | 0.3678 | 1.4743 |
| 0.10 | 0.3678 | 0.3678 | 1.4743 |
| 0.15 | 0.3678 | 0.3678 | 1.4743 |
| 0.20 | 0.3678 | 0.3678 | 1.4743 |
| 0.25 | 0.3678 | 0.3678 | 1.4743 |
| 0.30 | 0.3678 | 0.3678 | 1.4743 |
| 0.35 | 0.3678 | 0.3678 | 1.4743 |
| 0.40 | 0.3678 | 0.3678 | 1.4743 |
| 0.45 | 0.3678 | 0.3678 | 1.4743 |
| 0.50 | 0.3678 | 0.3678 | 1.4743 |
| 0.55 | 0.3678 | 0.3678 | 1.4743 |
| 0.60 | 0.3678 | 0.3678 | 1.4308 |
| 0.65 | 0.3639 | 0.3639 | 1.3715 |
| 0.70 | 0.3631 | 0.3631 | 1.3281 |
| 0.75 | 0.3601 | 0.3601 | 1.2767 |
| 0.80 | 0.3516 | 0.3516 | 1.2411 |
| 0.85 | 0.3377 | 0.3377 | 1.1186 |
| 0.90 | 0.3138 | 0.3138 | 0.9842 |
| 0.95 | 0.2290 | 0.2290 | 0.5059 |

## Normal-Volume False Positives

At operating threshold 0.60: 0.4886 FP per volume

## Strata

### Distance Band

| Band | N (GT) | N (Hit) | Sensitivity |
|------|--------|--------|-------------|
| 0 | 545 | 197 | 0.3615 |
| 0-2 | 210 | 77 | 0.3667 |
| 2-4 | 217 | 74 | 0.3410 |
| >4 | 325 | 129 | 0.3969 |

### Number of Slices

| Type | N (GT) | N (Hit) | Sensitivity |
|------|--------|--------|-------------|
| 1 slice(s) | 1029 | 343 | 0.3333 |
| >1 slice(s) | 268 | 134 | 0.5000 |

### In-Plane Size Tertile

| Tertile | N (GT) | N (Hit) | Sensitivity |
|---------|--------|--------|-------------|
| tertile_1 | 516 | 216 | 0.4186 |
| tertile_2 | 379 | 131 | 0.3456 |
| tertile_3 | 402 | 130 | 0.3234 |

### Stratum Geometry

| Geometry | N (GT) | N (Hit) | Sensitivity |
|----------|--------|--------|-------------|
| inplane_0.62_slice_3 | 32 | 11 | 0.3438 |
| inplane_0.69_slice_5 | 1175 | 430 | 0.3660 |
| inplane_0.86_slice_3 | 11 | 2 | 0.1818 |
| inplane_0.86_slice_5 | 79 | 34 | 0.4304 |

## Command

```
scripts/eval_brain_detector.py --config 3d_fullres --out docs/verification/2026-09-28/brain_detector/3d_fullres
```
