# fastMRI+ knee detector after Gate 0 (held-out patients, five folds)

IoU >= 0.1, FP budget 2.0/scan, transfer gate on clean x shared4 family sensitivity >= 0.5

| view | families | n_scans | n_gt | thr | sensitivity | sensitivity_family | fp_per_scan | fp_per_normal_scan | ceiling (sens_fam @ fp) |
|---|---|---|---|---|---|---|---|---|---|
| clean | shared4 | 1172 | 3401 | 0.18 | 0.155 | 0.091 | 1.53 | 1.41 (n=198) | 0.215 @ 65.70 |
| clean | all5 | 1172 | 4016 | 0.18 | 0.135 | 0.077 | 1.51 | 1.41 (n=198) | 0.189 @ 82.00 |
| noise_q1 | shared4 | 1172 | 3401 | 0.18 | 0.154 | 0.098 | 1.52 | 1.40 (n=198) | 0.222 @ 65.87 |
| noise_q1 | all5 | 1172 | 4016 | 0.18 | 0.134 | 0.082 | 1.51 | 1.40 (n=198) | 0.194 @ 82.02 |
| noise_q2 | shared4 | 1172 | 3401 | 0.17 | 0.165 | 0.106 | 1.89 | 1.91 (n=198) | 0.228 @ 67.19 |
| noise_q2 | all5 | 1172 | 4016 | 0.17 | 0.144 | 0.089 | 1.88 | 1.91 (n=198) | 0.199 @ 83.37 |
| noise_q3 | shared4 | 1172 | 3401 | 0.16 | 0.151 | 0.102 | 1.92 | 1.91 (n=198) | 0.230 @ 74.47 |
| noise_q3 | all5 | 1172 | 4016 | 0.16 | 0.132 | 0.086 | 1.90 | 1.91 (n=198) | 0.200 @ 90.32 |
| us4 | shared4 | 1172 | 3401 | 0.18 | 0.169 | 0.096 | 1.97 | 1.95 (n=198) | 0.195 @ 64.99 |
| us4 | all5 | 1172 | 4016 | 0.18 | 0.147 | 0.081 | 1.96 | 1.95 (n=198) | 0.172 @ 82.86 |
| us8 | shared4 | 1172 | 3401 | 0.18 | 0.151 | 0.088 | 1.76 | 1.84 (n=198) | 0.197 @ 62.08 |
| us8 | all5 | 1172 | 4016 | 0.18 | 0.130 | 0.074 | 1.75 | 1.84 (n=198) | 0.176 @ 73.97 |
| us16 | shared4 | 1172 | 3401 | 0.18 | 0.149 | 0.084 | 1.76 | 1.77 (n=198) | 0.179 @ 64.01 |
| us16 | all5 | 1172 | 4016 | 0.18 | 0.128 | 0.071 | 1.75 | 1.77 (n=198) | 0.160 @ 75.93 |

## decision view (clean x shared4)

```
{
 "pass": false,
 "thr": 0.18,
 "sensitivity_family": 0.09144369303146134,
 "fp_per_scan": 1.5281569965870307,
 "view": "clean",
 "families": "shared4",
 "n_scans": 1172,
 "n_gt": 3401,
 "missing_detection_files": 0,
 "fp_per_normal_scan": {
  "n_normal": 198,
  "fp_per_normal_scan": 1.4090909090909092
 },
 "per_family": {
  "cartilage": {
   "n_gt": 1324,
   "n_hit": 160,
   "n_hit_family": 110,
   "sensitivity": 0.12084592145015106,
   "sensitivity_family": 0.08308157099697885
  },
  "effusion": {
   "n_gt": 327,
   "n_hit": 0,
   "n_hit_family": 0,
   "sensitivity": 0.0,
   "sensitivity_family": 0.0
  },
  "ligament": {
   "n_gt": 523,
   "n_hit": 7,
   "n_hit_family": 0,
   "sensitivity": 0.01338432122370937,
   "sensitivity_family": 0.0
  },
  "meniscus": {
   "n_gt": 1227,
   "n_hit": 360,
   "n_hit_family": 201,
   "sensitivity": 0.293398533007335,
   "sensitivity_family": 0.16381418092909536
  }
 },
 "size_terciles": {
  "cartilage": [
   {
    "lo": 0.014361643762600806,
    "hi": 0.1796236119048913,
    "n": 446,
    "hit": 0
   },
   {
    "lo": 0.18317578125,
    "hi": 0.7559969276633065,
    "n": 437,
    "hit": 13
   },
   {
    "lo": 0.7582947906653226,
    "hi": 21.276488001417842,
    "n": 441,
    "hit": 147
   }
  ],
  "effusion": [
   {
    "lo": 0.7933376192466032,
    "hi": 12.692250474536003,
    "n": 109,
    "hit": 0
   },
   {
    "lo": 12.699734791698708,
    "hi": 35.98453461157258,
    "n": 109,
    "hit": 0
   },
   {
    "lo": 36.19134228175403,
    "hi": 351.50959129695644,
    "n": 109,
    "hit": 0
   }
  ],
  "ligament": [
   {
    "lo": 0.10478044027785324,
    "hi": 1.5152863670951084,
    "n": 175,
    "hit": 0
   },
   {
    "lo": 1.5165895813306451,
    "hi": 3.5984534611572583,
    "n": 174,
    "hit": 3
   },
   {
    "lo": 3.6062894390135867,
    "hi": 44.76773096706521,
    "n": 174,
    "hit": 4
   }
  ],
  "meniscus": [
   {
    "lo": 0.020106301267641128,
    "hi": 0.39178564184375,
    "n": 409,
    "hit": 4
   },
   {
    "lo": 0.3958069020972782,
    "hi": 2.4816920421774196,
    "n": 409,
    "hit": 44
   },
   {
    "lo": 2.5000749461935485,
    "hi": 68.58513104560801,
    "n": 409,
    "hit": 312
   }
  ]
 },
 "patient_coverage": {
  "n_patients": 496,
  "covered": 184,
  "coverage": 0.3709677419354839
 },
 "n_hits": 527,
 "centre_error_mm_median": 6.217000351822773,
 "iou_median": 0.24219408586440588,
 "family_correct_on_hits": 0.5901328273244781
}
```

TRANSFER_GATE: FAIL ({"thr": 0.18, "sensitivity_family": 0.09144369303146134, "fp_per_scan": 1.5281569965870307})
