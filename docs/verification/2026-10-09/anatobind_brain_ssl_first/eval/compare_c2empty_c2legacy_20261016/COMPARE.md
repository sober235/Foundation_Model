# C2-empty vs C2-legacy on the same 1056 test rows (NOT_EVIDENCE: SynthSeg / lesion pseudo-labels)

Differences are C2-empty − C2-legacy; intervals are 95 % patient-level bootstrap intervals.

## A: 13-host macro Dice (A-supervised rows)

| rows | C2-empty | C2-legacy | difference | 95 % CI |
|---|---|---|---|---|
| 1032 | 0.7988 | 0.7961 | 0.0027 | [0.0016, 0.0040] |

## U: lesion sensitivity at each arm's own operating point

| source/sequence | scans | lesions | thr C2-empty / C2-legacy | sens C2-empty | sens C2-legacy | FP/scan C2-empty | FP/scan C2-legacy | difference | 95 % CI |
|---|---|---|---|---|---|---|---|---|---|
| bmsr/T1c | 102 | 824 | 0.70 / 0.70 | 0.0146 | 0.0049 | 0.098 | 0.118 | 0.0097 | [-0.0016, 0.0242] |
| isles/DWI | 50 | 289 | 0.60 / 0.70 | 0.0242 | 0.0069 | 1.120 | 0.180 | 0.0173 | [0.0038, 0.0347] |
| pdgm/FLAIR | 101 | 130 | 0.75 / 0.70 | 0.6538 | 0.7538 | 0.218 | 0.158 | -0.1000 | [-0.1575, -0.0462] |
| pdgm/T1c | 101 | 123 | 0.75 / 0.70 | 0.1463 | 0.2927 | 0.050 | 0.386 | -0.1463 | [-0.2414, -0.0550] |

## R: controlled-track host accuracy

| instances | ABA C2-empty | ABA C2-legacy | difference | 95 % CI | B0 C2-empty | B0 C2-legacy |
|---|---|---|---|---|---|---|
| 1336 | 0.5606 | 0.5142 | 0.0464 | [0.0050, 0.0861] | 0.7545 | 0.6692 |

Paired: both right 551, C2-empty only 198, C2-legacy only 136, both wrong 451.
