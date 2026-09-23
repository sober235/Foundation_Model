# Knee capability evaluation

IoU >= 0.1, FP budget 2.0/scan, gate on `clean` family sensitivity >= 0.5

| view | n_scans | n_gt | thr | sensitivity | sensitivity_family | fp_per_scan |
|---|---|---|---|---|---|---|
| clean | 124 | 364 | 0.75 | 0.297 | 0.266 | 1.63 |

## clean

per family at the operating point: Cartilage Lesion: 21/160 (0.13), Effusion: 52/91 (0.57), Ligament Tear: 7/28 (0.25), Meniscal Tear: 17/85 (0.20)

binding on hits (system): n=45, family correct 0.800, side correct 0.773 (n_side=22)
binding on annotated boxes (given-box control): n=245, family correct 0.967, side correct 0.991 (n_side=113)
anatomy Dice (Dataset901 vs export seg): 1: 0.890, 2: 0.875, 3: 0.855, 4: 0.860, 5: 0.848, 6: 0.843

GATE: FAIL ({"view": "clean", "pass": false, "thr": 0.75, "sensitivity_family": 0.2664835164835165, "fp_per_scan": 1.6290322580645162, "n_scans": 124, "n_gt": 364})
