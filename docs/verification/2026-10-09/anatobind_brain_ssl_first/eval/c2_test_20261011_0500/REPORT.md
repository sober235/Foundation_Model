# AnatoBind-Brain test-set evaluation (NOT_EVIDENCE: every number rests on SynthSeg / lesion pseudo-labels)

## A (gate G2, part 1)

13-host macro Dice 0.7961 over 1032 A-supervised volumes (entity macro 0.6625); gate ≥ 0.8: **fail**. Unsupervised (QC-failed) rows: 24, host macro 0.7346 (not gated).

| source | volumes | host macro Dice |
|---|---|---|
| bmsr | 306 | 0.8040 |
| isles | 76 | 0.7424 |
| pdgm | 404 | 0.7986 |
| sibbms | 246 | 0.7987 |

| host | Dice |
|---|---|
| white_matter_left | 0.7615 |
| white_matter_right | 0.7617 |
| cortex_left | 0.7034 |
| cortex_right | 0.6951 |
| thalamus_left | 0.8455 |
| thalamus_right | 0.8515 |
| basal_ganglia_left | 0.7430 |
| basal_ganglia_right | 0.7734 |
| brainstem | 0.8637 |
| cerebellum_left | 0.8951 |
| cerebellum_right | 0.8924 |
| other_deep_grey_left | 0.7836 |
| other_deep_grey_right | 0.7793 |

## U (gate G2, part 2)

Sensitivity at the reference detector's false positives per scan on the same patients; pass = ≥ reference − 0.05; the gate reads one sequence per source (the one with the reference's lesion labels); gate: **fail** (3 sources judged)

| source/sequence | gated | scans | GT (ignored) | thr | selection | sensitivity | FP/scan | budget | ref sens | ref thr | ref scans | ref GT | grid | <5 mm | 5–10 mm | ≥10 mm | pass |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| bmsr/T1c | True | 102 | 824 (94) | 0.70 | fixed (chosen on the validation split) | 0.0049 | 0.118 | 0.510 | 0.7316 | 0.65 | 102 | 827 | AUR 1 mm resampled vs native (approximate) | 0.0045 | 0.0078 | 0.0000 | False |
| isles/DWI | True | 50 | 289 (7) | 0.70 | fixed (chosen on the validation split) | 0.0069 | 0.180 | 1.460 | 0.6303 | 0.50 | 50 | 357 | AUR 1 mm resampled vs native 2 mm (approximate) | 0.0000 | 0.0000 | 0.0312 | False |
| pdgm/FLAIR | True | 101 | 130 (53) | 0.70 | fixed (chosen on the validation split) | 0.7538 | 0.158 | 0.228 | 0.8615 | 0.60 | 101 | 130 | same 1 mm grid | 0.0000 | 0.0000 | 0.8448 | False |
| pdgm/T1c | False | 101 | 123 (33) | nan | — | nan | nan | nan | nan | nan | — | — | — | nan | nan | nan | None |

## R, controlled track (gate G3)

1336 instances (zero-overlap bindings 4); ABA(R) 0.5142 vs ABA(B0) 0.6692; side R 0.8608 / B0 0.9394; tissue R 0.5509 / B0 0.6984; gate ABA(R) ≥ ABA(B0): **fail**. the controlled-track truth is the SynthSeg lookup itself (host_targets on the SynthSeg map); no B0* column, it would be 1.0 by construction.

All controlled instances: B0 wrong on 0.3308; R rescues 236, harms 443, both right 451, both wrong 206.

Recognition–binding gap (local 13-host Dice >= 0.8 in the bind window and a detection matched at the source's U threshold): 53 instances, B0 wrong on 0.5283; R rescues 23, harms 7.

End to end (predicted lesions matched to a truth at the U threshold): 139 lesions, ABA(R) 0.7050 vs ABA(B0) 0.4892.

| split | n | ABA(R) | ABA(B0) | rescue | harm |
|---|---|---|---|---|---|
| bmsr | 824 | 0.4624 | 0.7136 | 104 | 311 |
| isles | 259 | 0.4942 | 0.6409 | 42 | 80 |
| pdgm | 253 | 0.7036 | 0.5534 | 90 | 52 |
| <5 | 558 | 0.4462 | 0.7419 | 60 | 225 |
| 5-10 | 390 | 0.5128 | 0.6410 | 67 | 117 |
| >=10 | 388 | 0.6134 | 0.5928 | 109 | 101 |

Sequence-type accuracy 0.9981 over 1056 volumes.

Not in this report: SibBMS 10-subject MS plaque check (U, report only; a separate script: scripts/aur_eval_sibbms.py); fastMRI 433-volume reliable-slice A Dice and 1297-box host agreement (report only; a separate script: scripts/aur_eval_fastmri.py); end-to-end R on a human-labelled set (Level R: the sheet is exported, the statistics wait for the readers).
