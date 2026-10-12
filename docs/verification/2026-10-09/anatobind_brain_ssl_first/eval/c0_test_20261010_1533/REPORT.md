# AnatoBind-Brain test-set evaluation (NOT_EVIDENCE: every number rests on SynthSeg / lesion pseudo-labels)

## A (gate G2, part 1)

13-host macro Dice 0.6289 over 1032 A-supervised volumes (entity macro 0.3948); gate ≥ 0.8: **fail**. Unsupervised (QC-failed) rows: 24, host macro 0.5883 (not gated).

| source | volumes | host macro Dice |
|---|---|---|
| bmsr | 306 | 0.6372 |
| isles | 76 | 0.5885 |
| pdgm | 404 | 0.6298 |
| sibbms | 246 | 0.6298 |

| host | Dice |
|---|---|
| white_matter_left | 0.7117 |
| white_matter_right | 0.7058 |
| cortex_left | 0.6026 |
| cortex_right | 0.6174 |
| thalamus_left | 0.7260 |
| thalamus_right | 0.7079 |
| basal_ganglia_left | 0.5463 |
| basal_ganglia_right | 0.0000 |
| brainstem | 0.8134 |
| cerebellum_left | 0.8613 |
| cerebellum_right | 0.8659 |
| other_deep_grey_left | 0.6189 |
| other_deep_grey_right | 0.3992 |

## U (gate G2, part 2)

Sensitivity at the reference detector's false positives per scan on the same patients; pass = ≥ reference − 0.05; the gate reads one sequence per source (the one with the reference's lesion labels); gate: **fail** (3 sources judged)

| source/sequence | gated | scans | GT (ignored) | thr | selection | sensitivity | FP/scan | budget | ref sens | ref thr | ref scans | ref GT | grid | <5 mm | 5–10 mm | ≥10 mm | pass |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| bmsr/T1c | True | 102 | 824 (94) | 0.95 | fixed (chosen on the validation split) | 0.0000 | 0.000 | 0.510 | 0.7316 | 0.65 | 102 | 827 | AUR 1 mm resampled vs native (approximate) | 0.0000 | 0.0000 | 0.0000 | False |
| isles/DWI | True | 50 | 289 (7) | 0.55 | fixed (chosen on the validation split) | 0.0346 | 1.300 | 1.460 | 0.6303 | 0.50 | 50 | 357 | AUR 1 mm resampled vs native 2 mm (approximate) | 0.0381 | 0.0083 | 0.0781 | False |
| pdgm/FLAIR | True | 101 | 130 (53) | 0.70 | fixed (chosen on the validation split) | 0.1692 | 0.139 | 0.228 | 0.8615 | 0.60 | 101 | 130 | same 1 mm grid | 0.0000 | 0.0000 | 0.1897 | False |
| pdgm/T1c | False | 101 | 123 (33) | nan | — | nan | nan | nan | nan | nan | — | — | — | nan | nan | nan | None |

## R, controlled track (gate G3)

1336 instances (zero-overlap bindings 8); ABA(R) 0.5007 vs ABA(B0) 0.6789; side R 0.8563 / B0 0.9768; tissue R 0.5419 / B0 0.6946; gate ABA(R) ≥ ABA(B0): **fail**. the controlled-track truth is the SynthSeg lookup itself (host_targets on the SynthSeg map); no B0* column, it would be 1.0 by construction.

All controlled instances: B0 wrong on 0.3211; R rescues 185, harms 423, both right 484, both wrong 244.

Recognition–binding gap: no instance meets the condition (local 13-host Dice >= 0.8 in the bind window and a detection matched at the source's U threshold).

End to end (predicted lesions matched to a truth at the U threshold): 32 lesions, ABA(R) 0.5625 vs ABA(B0) 0.5938.

| split | n | ABA(R) | ABA(B0) | rescue | harm |
|---|---|---|---|---|---|
| bmsr | 824 | 0.4782 | 0.7221 | 102 | 303 |
| isles | 259 | 0.5212 | 0.6178 | 34 | 59 |
| pdgm | 253 | 0.5534 | 0.6008 | 49 | 61 |
| <5 | 558 | 0.4857 | 0.7133 | 71 | 198 |
| 5-10 | 390 | 0.5154 | 0.6872 | 47 | 114 |
| >=10 | 388 | 0.5077 | 0.6211 | 67 | 111 |

Sequence-type accuracy 0.9915 over 1056 volumes.

Not in this report: SibBMS 10-subject MS plaque check (U, report only; a separate script: scripts/aur_eval_sibbms.py); fastMRI 433-volume reliable-slice A Dice and 1297-box host agreement (report only; a separate script: scripts/aur_eval_fastmri.py); end-to-end R on a human-labelled set (Level R: the sheet is exported, the statistics wait for the readers).
