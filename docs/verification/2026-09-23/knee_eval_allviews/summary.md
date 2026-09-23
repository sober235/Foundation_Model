# Knee capability evaluation

IoU >= 0.1, FP budget 2.0/scan, gate on `clean` family sensitivity >= 0.5

| view | n_scans | n_gt | thr | sensitivity | sensitivity_family | fp_per_scan |
|---|---|---|---|---|---|---|
| clean | 155 | 465 | 0.75 | 0.282 | 0.254 | 1.57 |
| noise_q1 | 155 | 465 | 0.75 | 0.284 | 0.256 | 1.59 |
| noise_q2 | 155 | 465 | 0.75 | 0.284 | 0.252 | 1.61 |
| noise_q3 | 155 | 465 | 0.75 | 0.299 | 0.262 | 1.61 |
| us4 | 155 | 465 | 0.75 | 0.277 | 0.241 | 1.74 |
| us8 | 155 | 465 | 0.75 | 0.284 | 0.243 | 1.75 |
| us16 | 155 | 465 | 0.75 | 0.280 | 0.245 | 1.77 |

## clean

per family at the operating point: Cartilage Lesion: 25/208 (0.12), Effusion: 64/116 (0.55), Ligament Tear: 9/38 (0.24), Meniscal Tear: 20/103 (0.19)

binding on hits (system): n=54, family correct 0.796, side correct 0.808 (n_side=26)
binding on annotated boxes (given-box control): n=311, family correct 0.961, side correct 0.979 (n_side=143)
anatomy Dice (Dataset901 vs export seg): 1: 0.886, 2: 0.873, 3: 0.852, 4: 0.858, 5: 0.845, 6: 0.839

## noise_q1

per family at the operating point: Cartilage Lesion: 26/208 (0.12), Effusion: 64/116 (0.55), Ligament Tear: 8/38 (0.21), Meniscal Tear: 21/103 (0.20)

binding on hits (system): n=56, family correct 0.804, side correct 0.815 (n_side=27)
binding on annotated boxes (given-box control): n=311, family correct 0.961, side correct 0.979 (n_side=143)
anatomy Dice (Dataset901 vs export seg): 1: 0.886, 2: 0.873, 3: 0.852, 4: 0.858, 5: 0.845, 6: 0.839

## noise_q2

per family at the operating point: Cartilage Lesion: 25/208 (0.12), Effusion: 63/116 (0.54), Ligament Tear: 8/38 (0.21), Meniscal Tear: 21/103 (0.20)

binding on hits (system): n=56, family correct 0.804, side correct 0.821 (n_side=28)
binding on annotated boxes (given-box control): n=311, family correct 0.965, side correct 0.979 (n_side=143)
anatomy Dice (Dataset901 vs export seg): 1: 0.886, 2: 0.872, 3: 0.852, 4: 0.858, 5: 0.844, 6: 0.839

## noise_q3

per family at the operating point: Cartilage Lesion: 29/208 (0.14), Effusion: 63/116 (0.54), Ligament Tear: 8/38 (0.21), Meniscal Tear: 22/103 (0.21)

binding on hits (system): n=62, family correct 0.790, side correct 0.774 (n_side=31)
binding on annotated boxes (given-box control): n=311, family correct 0.965, side correct 0.979 (n_side=143)
anatomy Dice (Dataset901 vs export seg): 1: 0.885, 2: 0.871, 3: 0.851, 4: 0.856, 5: 0.843, 6: 0.837

## us4

per family at the operating point: Cartilage Lesion: 20/208 (0.10), Effusion: 62/116 (0.53), Ligament Tear: 8/38 (0.21), Meniscal Tear: 22/103 (0.21)

binding on hits (system): n=52, family correct 0.769, side correct 0.833 (n_side=30)
binding on annotated boxes (given-box control): n=311, family correct 0.965, side correct 0.986 (n_side=143)
anatomy Dice (Dataset901 vs export seg): 1: 0.881, 2: 0.867, 3: 0.849, 4: 0.855, 5: 0.835, 6: 0.831

## us8

per family at the operating point: Cartilage Lesion: 19/208 (0.09), Effusion: 62/116 (0.53), Ligament Tear: 10/38 (0.26), Meniscal Tear: 22/103 (0.21)

binding on hits (system): n=53, family correct 0.755, side correct 0.828 (n_side=29)
binding on annotated boxes (given-box control): n=311, family correct 0.968, side correct 0.986 (n_side=143)
anatomy Dice (Dataset901 vs export seg): 1: 0.875, 2: 0.862, 3: 0.847, 4: 0.851, 5: 0.828, 6: 0.824

## us16

per family at the operating point: Cartilage Lesion: 20/208 (0.10), Effusion: 63/116 (0.54), Ligament Tear: 9/38 (0.24), Meniscal Tear: 22/103 (0.21)

binding on hits (system): n=51, family correct 0.784, side correct 0.857 (n_side=28)
binding on annotated boxes (given-box control): n=311, family correct 0.965, side correct 0.986 (n_side=143)
anatomy Dice (Dataset901 vs export seg): 1: 0.862, 2: 0.854, 3: 0.843, 4: 0.847, 5: 0.817, 6: 0.814

GATE: FAIL ({"view": "clean", "pass": false, "thr": 0.75, "sensitivity_family": 0.2537634408602151, "fp_per_scan": 1.5677419354838709, "n_scans": 155, "n_gt": 465})
