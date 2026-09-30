# nnDetection second arm, folds [0]

Fold subset: these numbers decide rule A only; they are NOT the D1 gate.

Inputs: `/data2/congcong/data/FM_data/derived/nndet_runs/fold0_default.json` sha256 185b1eda0682054d6440b53ab7d716573bd92e8c7c57e259dacac899da0849ff; swept `/data2/congcong/data/FM_data/derived/nndet_runs/fold0_swept.json` sha256 0c4086e49650da543da95e488fbac99f274d166e283c406107c917a5108d9e9d

## Rule A (spec N3)

```json
{
 "pass": false,
 "nndet_hits": 17,
 "baseline_hits": 92,
 "required": 106,
 "n_gt": 280,
 "margin": 0.05,
 "nndet_thr": 0.5,
 "baseline_thr": 0.6
}
```

## nnDetection, default postprocessing

```json
{
 "pass": false,
 "thr": 0.5,
 "sensitivity_family": 0.060714285714285714,
 "fp_per_scan": 0.7254901960784313
}
```

Normal-volume FP per volume at the operating threshold: 0.0000

### FROC

| thr | n_hit | sensitivity | FP per volume |
|---|---|---|---|
| 0.05 | 178 | 0.6357 | 29.8824 |
| 0.10 | 163 | 0.5821 | 21.4510 |
| 0.15 | 145 | 0.5179 | 12.6471 |
| 0.20 | 138 | 0.4929 | 10.3922 |
| 0.25 | 108 | 0.3857 | 7.1373 |
| 0.30 | 102 | 0.3643 | 5.8235 |
| 0.35 | 92 | 0.3286 | 5.0000 |
| 0.40 | 65 | 0.2321 | 3.0980 |
| 0.45 | 60 | 0.2143 | 2.5490 |
| 0.50 | 17 | 0.0607 | 0.7255 |
| 0.55 | 10 | 0.0357 | 0.1765 |
| 0.60 | 8 | 0.0286 | 0.1765 |
| 0.65 | 8 | 0.0286 | 0.0784 |
| 0.70 | 7 | 0.0250 | 0.0784 |
| 0.75 | 4 | 0.0143 | 0.0784 |
| 0.80 | 4 | 0.0143 | 0.0784 |
| 0.85 | 4 | 0.0143 | 0.0784 |
| 0.90 | 1 | 0.0036 | 0.0196 |
| 0.95 | 1 | 0.0036 | 0.0196 |

## nnU-Net 2d, same folds (rule A baseline)

```json
{
 "pass": false,
 "thr": 0.6,
 "sensitivity_family": 0.32857142857142857,
 "fp_per_scan": 1.5490196078431373
}
```

Normal-volume FP per volume at the operating threshold: 0.1667

## nnU-Net 3d_fullres, same folds (report only)

```json
{
 "pass": false,
 "thr": 0.75,
 "sensitivity_family": 0.31785714285714284,
 "fp_per_scan": 1.3529411764705883
}
```

Normal-volume FP per volume at the operating threshold: 0.1667

## Paired table (each model at its own operating point, not gated)

| group | lesions |
|---|---|
| both | 11 |
| only nnDetection | 6 |
| only nnU-Net 2d | 81 |
| neither | 182 |

## Strata (nnDetection at its operating point, not gated)

### band

| stratum | N (GT) | N (hit) | sensitivity |
|---|---|---|---|
| 0 | 125 | 12 | 0.0960 |
| 0-2 | 43 | 1 | 0.0233 |
| 2-4 | 51 | 2 | 0.0392 |
| >4 | 61 | 2 | 0.0328 |

### n_slices

| stratum | N (GT) | N (hit) | sensitivity |
|---|---|---|---|
| 1 | 222 | 7 | 0.0315 |
| >1 | 58 | 10 | 0.1724 |

### inplane_tertile

| stratum | N (GT) | N (hit) | sensitivity |
|---|---|---|---|
| tertile_1 | 120 | 3 | 0.0250 |
| tertile_2 | 69 | 3 | 0.0435 |
| tertile_3 | 91 | 11 | 0.1209 |

### stratum_geometry

| stratum | N (GT) | N (hit) | sensitivity |
|---|---|---|---|
| inplane_0.62_slice_3 | 7 | 0 | 0.0000 |
| inplane_0.69_slice_5 | 248 | 14 | 0.0565 |
| inplane_0.86_slice_3 | 5 | 0 | 0.0000 |
| inplane_0.86_slice_5 | 20 | 3 | 0.1500 |

## Swept postprocessing (NOT_GATE: parameters tuned on these validation cases)

```json
{
 "pass": false,
 "thr": 0.85,
 "sensitivity_family": 0.24285714285714285,
 "fp_per_scan": 1.8431372549019607
}
```

Normal-volume FP per volume at the operating threshold: 0.1667

## Command

```
scripts/eval_brain_nndet.py --dets /data2/congcong/data/FM_data/derived/nndet_runs/fold0_default.json --swept /data2/congcong/data/FM_data/derived/nndet_runs/fold0_swept.json --folds 0 --out docs/verification/2026-09-29/brain_nndet/fold0
```
