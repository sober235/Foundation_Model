# Knee capability evaluation

IoU >= 0.1, FP budget 2.0/scan, gate on `clean` family sensitivity >= 0.5

| view | n_scans | n_gt | thr | sensitivity | sensitivity_family | fp_per_scan |
|---|---|---|---|---|---|---|
| clean | 155 | 465 | 0.75 | 0.282 | 0.254 | 1.57 |

## clean

per family at the operating point: Cartilage Lesion: 25/208 (0.12), Effusion: 64/116 (0.55), Ligament Tear: 9/38 (0.24), Meniscal Tear: 20/103 (0.19)

binding on hits (system): n=54, family correct 0.796, side correct 0.808 (n_side=26)
binding on annotated boxes (given-box control): n=311, family correct 0.961, side correct 0.979 (n_side=143)
anatomy Dice (Dataset901 vs export seg): 1: 0.886, 2: 0.873, 3: 0.852, 4: 0.858, 5: 0.845, 6: 0.839

GATE: FAIL ({"view": "clean", "pass": false, "thr": 0.75, "sensitivity_family": 0.2537634408602151, "fp_per_scan": 1.5677419354838709, "n_scans": 155, "n_gt": 465})
