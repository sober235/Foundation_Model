# AnatoBind-Brain test-set evaluation (NOT_EVIDENCE: every number rests on SynthSeg / lesion pseudo-labels)

## A (gate G2, part 1)

13-host macro Dice 0.7988 over 1032 A-supervised volumes (entity macro 0.6460); gate ≥ 0.8: **fail**. Unsupervised (QC-failed) rows: 24, host macro 0.7436 (not gated).

| source | volumes | host macro Dice |
|---|---|---|
| bmsr | 306 | 0.8062 |
| isles | 76 | 0.7527 |
| pdgm | 404 | 0.8039 |
| sibbms | 246 | 0.7957 |

| host | Dice |
|---|---|
| white_matter_left | 0.7751 |
| white_matter_right | 0.7691 |
| cortex_left | 0.7130 |
| cortex_right | 0.7137 |
| thalamus_left | 0.8509 |
| thalamus_right | 0.8566 |
| basal_ganglia_left | 0.7589 |
| basal_ganglia_right | 0.7813 |
| brainstem | 0.8745 |
| cerebellum_left | 0.9049 |
| cerebellum_right | 0.9018 |
| other_deep_grey_left | 0.7059 |
| other_deep_grey_right | 0.7793 |

## U (gate G2, part 2)

Sensitivity at the reference detector's false positives per scan on the same patients; pass = ≥ reference − 0.05; the gate reads one sequence per source (the one with the reference's lesion labels); gate: **fail** (3 sources judged)

| source/sequence | gated | scans | GT (ignored) | thr | selection | sensitivity | FP/scan | budget | ref sens | ref thr | ref scans | ref GT | grid | <5 mm | 5–10 mm | ≥10 mm | pass |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| bmsr/T1c | True | 102 | 824 (94) | 0.70 | fixed (chosen on the validation split) | 0.0146 | 0.098 | 0.510 | 0.7316 | 0.65 | 102 | 827 | AUR 1 mm resampled vs native (approximate) | 0.0067 | 0.0271 | 0.0167 | False |
| isles/DWI | True | 50 | 289 (7) | 0.60 | fixed (chosen on the validation split) | 0.0242 | 1.120 | 1.460 | 0.6303 | 0.50 | 50 | 357 | AUR 1 mm resampled vs native 2 mm (approximate) | 0.0095 | 0.0167 | 0.0625 | False |
| pdgm/FLAIR | True | 101 | 130 (53) | 0.75 | fixed (chosen on the validation split) | 0.6538 | 0.218 | 0.228 | 0.8615 | 0.60 | 101 | 130 | same 1 mm grid | 0.0000 | 0.0000 | 0.7328 | False |
| pdgm/T1c | False | 101 | 123 (33) | nan | — | nan | nan | nan | nan | nan | — | — | — | nan | nan | nan | None |

## R, controlled track (gate G3)

1336 instances (zero-overlap bindings 3); ABA(R) 0.5606 vs ABA(B0) 0.7545; side R 0.9139 / B0 0.9813; tissue R 0.5913 / B0 0.7680; gate ABA(R) ≥ ABA(B0): **fail**. the controlled-track truth is the SynthSeg lookup itself (host_targets on the SynthSeg map); no B0* column, it would be 1.0 by construction.

All controlled instances: B0 wrong on 0.2455; R rescues 202, harms 461, both right 547, both wrong 126.

Recognition–binding gap (local 13-host Dice >= 0.8 in the bind window and a detection matched at the source's U threshold): 48 instances, B0 wrong on 0.2917; R rescues 9, harms 6.

End to end (predicted lesions matched to a truth at the U threshold): 120 lesions, ABA(R) 0.7417 vs ABA(B0) 0.7250.

| split | n | ABA(R) | ABA(B0) | rescue | harm |
|---|---|---|---|---|---|
| bmsr | 824 | 0.5182 | 0.7718 | 111 | 320 |
| isles | 259 | 0.5946 | 0.7452 | 42 | 81 |
| pdgm | 253 | 0.6640 | 0.7075 | 49 | 60 |
| <5 | 558 | 0.5376 | 0.7742 | 80 | 212 |
| 5-10 | 390 | 0.5436 | 0.7692 | 52 | 140 |
| >=10 | 388 | 0.6108 | 0.7113 | 70 | 109 |

Sequence-type accuracy 0.9981 over 1056 volumes.

Not in this report: SibBMS 10-subject MS plaque check (U, report only; a separate script: scripts/aur_eval_sibbms.py); fastMRI 433-volume reliable-slice A Dice and 1297-box host agreement (report only; a separate script: scripts/aur_eval_fastmri.py); end-to-end R on a human-labelled set (Level R: the sheet is exported, the statistics wait for the readers).
