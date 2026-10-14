# C0v2 vs C0 on the same 1056 test rows (NOT_EVIDENCE: SynthSeg / lesion pseudo-labels)

Differences are C0v2 − C0; intervals are 95 % patient-level bootstrap intervals.

## A: 13-host macro Dice (A-supervised rows)

| rows | C0v2 | C0 | difference | 95 % CI |
|---|---|---|---|---|
| 1032 | 0.7506 | 0.6289 | 0.1217 | [0.1186, 0.1243] |

## U: lesion sensitivity at each arm's own operating point

| source/sequence | scans | lesions | thr C0v2 / C0 | sens C0v2 | sens C0 | FP/scan C0v2 | FP/scan C0 | difference | 95 % CI |
|---|---|---|---|---|---|---|---|---|---|
| bmsr/T1c | 102 | 824 | 0.65 / 0.95 | 0.0061 | 0.0000 | 0.647 | 0.000 | 0.0061 | [0.0000, 0.0154] |
| isles/DWI | 50 | 289 | 0.60 / 0.55 | 0.0242 | 0.0346 | 1.140 | 1.300 | -0.0104 | [-0.0321, 0.0086] |
| pdgm/FLAIR | 101 | 130 | 0.60 / 0.70 | 0.6077 | 0.1692 | 0.604 | 0.139 | 0.4385 | [0.3534, 0.5360] |
| pdgm/T1c | 101 | 123 | 0.60 / 0.70 | 0.0976 | 0.0000 | 0.703 | 0.010 | 0.0976 | [0.0455, 0.1597] |

## R: controlled-track host accuracy

| instances | ABA C0v2 | ABA C0 | difference | 95 % CI | B0 C0v2 | B0 C0 |
|---|---|---|---|---|---|---|
| 1336 | 0.5996 | 0.5007 | 0.0988 | [0.0562, 0.1382] | 0.7013 | 0.6789 |

Paired: both right 491, C0v2 only 310, C0 only 178, both wrong 357.
