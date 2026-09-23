# Knee capability evaluation

IoU >= 0.1, FP budget 2.0/scan, gate on `clean` family sensitivity >= 0.5

| view | n_scans | n_gt | thr | sensitivity | sensitivity_family | fp_per_scan |
|---|---|---|---|---|---|---|
| clean | 31 | 82 | 0.75 | 0.280 | 0.256 | 1.39 |
| noise_q1 | 31 | 82 | 0.75 | 0.293 | 0.268 | 1.42 |
| noise_q2 | 31 | 82 | 0.75 | 0.293 | 0.268 | 1.39 |
| noise_q3 | 31 | 82 | 0.75 | 0.317 | 0.293 | 1.39 |
| us4 | 31 | 82 | 0.75 | 0.280 | 0.256 | 1.71 |
| us8 | 31 | 82 | 0.75 | 0.280 | 0.256 | 1.77 |
| us16 | 31 | 82 | 0.75 | 0.280 | 0.256 | 1.77 |

## clean

per family at the operating point: Cartilage Lesion: 6/35 (0.17), Effusion: 11/21 (0.52), Ligament Tear: 1/5 (0.20), Meniscal Tear: 3/21 (0.14)

binding on hits (system): n=10, family correct 0.900, side correct 1.000 (n_side=3)
binding on annotated boxes (given-box control): n=56, family correct 0.982, side correct 1.000 (n_side=30)
anatomy Dice (Dataset901 vs export seg): 1: 0.902, 2: 0.882, 3: 0.858, 4: 0.854, 5: 0.860, 6: 0.846

## noise_q1

per family at the operating point: Cartilage Lesion: 6/35 (0.17), Effusion: 11/21 (0.52), Ligament Tear: 1/5 (0.20), Meniscal Tear: 4/21 (0.19)

binding on hits (system): n=11, family correct 0.909, side correct 1.000 (n_side=4)
binding on annotated boxes (given-box control): n=56, family correct 0.982, side correct 1.000 (n_side=30)
anatomy Dice (Dataset901 vs export seg): 1: 0.902, 2: 0.882, 3: 0.858, 4: 0.854, 5: 0.859, 6: 0.846

## noise_q2

per family at the operating point: Cartilage Lesion: 6/35 (0.17), Effusion: 11/21 (0.52), Ligament Tear: 1/5 (0.20), Meniscal Tear: 4/21 (0.19)

binding on hits (system): n=11, family correct 0.909, side correct 1.000 (n_side=4)
binding on annotated boxes (given-box control): n=56, family correct 0.982, side correct 1.000 (n_side=30)
anatomy Dice (Dataset901 vs export seg): 1: 0.902, 2: 0.882, 3: 0.857, 4: 0.854, 5: 0.859, 6: 0.845

## noise_q3

per family at the operating point: Cartilage Lesion: 7/35 (0.20), Effusion: 11/21 (0.52), Ligament Tear: 1/5 (0.20), Meniscal Tear: 5/21 (0.24)

binding on hits (system): n=13, family correct 0.923, side correct 1.000 (n_side=5)
binding on annotated boxes (given-box control): n=56, family correct 0.982, side correct 1.000 (n_side=30)
anatomy Dice (Dataset901 vs export seg): 1: 0.901, 2: 0.881, 3: 0.857, 4: 0.853, 5: 0.858, 6: 0.844

## us4

per family at the operating point: Cartilage Lesion: 5/35 (0.14), Effusion: 11/21 (0.52), Ligament Tear: 1/5 (0.20), Meniscal Tear: 4/21 (0.19)

binding on hits (system): n=10, family correct 0.900, side correct 1.000 (n_side=4)
binding on annotated boxes (given-box control): n=56, family correct 0.982, side correct 1.000 (n_side=30)
anatomy Dice (Dataset901 vs export seg): 1: 0.895, 2: 0.878, 3: 0.855, 4: 0.852, 5: 0.852, 6: 0.841

## us8

per family at the operating point: Cartilage Lesion: 5/35 (0.14), Effusion: 11/21 (0.52), Ligament Tear: 1/5 (0.20), Meniscal Tear: 4/21 (0.19)

binding on hits (system): n=10, family correct 0.900, side correct 1.000 (n_side=4)
binding on annotated boxes (given-box control): n=56, family correct 0.982, side correct 1.000 (n_side=30)
anatomy Dice (Dataset901 vs export seg): 1: 0.889, 2: 0.874, 3: 0.852, 4: 0.848, 5: 0.845, 6: 0.831

## us16

per family at the operating point: Cartilage Lesion: 5/35 (0.14), Effusion: 11/21 (0.52), Ligament Tear: 1/5 (0.20), Meniscal Tear: 4/21 (0.19)

binding on hits (system): n=10, family correct 0.900, side correct 1.000 (n_side=4)
binding on annotated boxes (given-box control): n=56, family correct 0.982, side correct 1.000 (n_side=30)
anatomy Dice (Dataset901 vs export seg): 1: 0.877, 2: 0.866, 3: 0.848, 4: 0.844, 5: 0.836, 6: 0.818

GATE: FAIL ({"view": "clean", "pass": false, "thr": 0.75, "sensitivity_family": 0.25609756097560976, "fp_per_scan": 1.3870967741935485, "n_scans": 31, "n_gt": 82})
