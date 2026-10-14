# C2v2 vs C0v2 on the same 1056 test rows (NOT_EVIDENCE: SynthSeg / lesion pseudo-labels)

Differences are C2v2 − C0v2; intervals are 95 % patient-level bootstrap intervals.

## A: 13-host macro Dice (A-supervised rows)

| rows | C2v2 | C0v2 | difference | 95 % CI |
|---|---|---|---|---|
| 1032 | 0.7961 | 0.7506 | 0.0454 | [0.0435, 0.0477] |

## U: lesion sensitivity at each arm's own operating point

| source/sequence | scans | lesions | thr C2v2 / C0v2 | sens C2v2 | sens C0v2 | FP/scan C2v2 | FP/scan C0v2 | difference | 95 % CI |
|---|---|---|---|---|---|---|---|---|---|
| bmsr/T1c | 102 | 824 | 0.70 / 0.65 | 0.0049 | 0.0061 | 0.118 | 0.647 | -0.0012 | [-0.0092, 0.0060] |
| isles/DWI | 50 | 289 | 0.70 / 0.60 | 0.0069 | 0.0242 | 0.180 | 1.140 | -0.0173 | [-0.0403, 0.0000] |
| pdgm/FLAIR | 101 | 130 | 0.70 / 0.60 | 0.7538 | 0.6077 | 0.158 | 0.604 | 0.1462 | [0.0720, 0.2250] |
| pdgm/T1c | 101 | 123 | 0.70 / 0.60 | 0.2927 | 0.0976 | 0.386 | 0.703 | 0.1951 | [0.1083, 0.2913] |

## R: controlled-track host accuracy

| instances | ABA C2v2 | ABA C0v2 | difference | 95 % CI | B0 C2v2 | B0 C0v2 |
|---|---|---|---|---|---|---|
| 1336 | 0.5142 | 0.5996 | -0.0853 | [-0.1186, -0.0496] | 0.6692 | 0.7013 |

Paired: both right 533, C2v2 only 154, C0v2 only 268, both wrong 381.
