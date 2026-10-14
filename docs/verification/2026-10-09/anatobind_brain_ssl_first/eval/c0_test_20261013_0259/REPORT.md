# AnatoBind-Brain test-set evaluation (NOT_EVIDENCE: every number rests on SynthSeg / lesion pseudo-labels)

## A (gate G2, part 1)

13-host macro Dice 0.7506 over 1032 A-supervised volumes (entity macro 0.6130); gate ≥ 0.8: **fail**. Unsupervised (QC-failed) rows: 24, host macro 0.6917 (not gated).

| source | volumes | host macro Dice |
|---|---|---|
| bmsr | 306 | 0.7617 |
| isles | 76 | 0.7070 |
| pdgm | 404 | 0.7459 |
| sibbms | 246 | 0.7581 |

| host | Dice |
|---|---|
| white_matter_left | 0.7281 |
| white_matter_right | 0.7272 |
| cortex_left | 0.6512 |
| cortex_right | 0.6644 |
| thalamus_left | 0.7773 |
| thalamus_right | 0.7823 |
| basal_ganglia_left | 0.6957 |
| basal_ganglia_right | 0.6890 |
| brainstem | 0.8372 |
| cerebellum_left | 0.8760 |
| cerebellum_right | 0.8807 |
| other_deep_grey_left | 0.7260 |
| other_deep_grey_right | 0.7234 |

## U (gate G2, part 2)

Sensitivity at the reference detector's false positives per scan on the same patients; pass = ≥ reference − 0.05; the gate reads one sequence per source (the one with the reference's lesion labels); gate: **fail** (3 sources judged)

| source/sequence | gated | scans | GT (ignored) | thr | selection | sensitivity | FP/scan | budget | ref sens | ref thr | ref scans | ref GT | grid | <5 mm | 5–10 mm | ≥10 mm | pass |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| bmsr/T1c | True | 102 | 824 (94) | 0.65 | fixed (chosen on the validation split) | 0.0061 | 0.647 | 0.510 | 0.7316 | 0.65 | 102 | 827 | AUR 1 mm resampled vs native (approximate) | 0.0067 | 0.0039 | 0.0083 | False |
| isles/DWI | True | 50 | 289 (7) | 0.60 | fixed (chosen on the validation split) | 0.0242 | 1.140 | 1.460 | 0.6303 | 0.50 | 50 | 357 | AUR 1 mm resampled vs native 2 mm (approximate) | 0.0095 | 0.0250 | 0.0469 | False |
| pdgm/FLAIR | True | 101 | 130 (53) | 0.60 | fixed (chosen on the validation split) | 0.6077 | 0.604 | 0.228 | 0.8615 | 0.60 | 101 | 130 | same 1 mm grid | 0.0000 | 0.0000 | 0.6810 | False |
| pdgm/T1c | False | 101 | 123 (33) | nan | — | nan | nan | nan | nan | nan | — | — | — | nan | nan | nan | None |

## R, controlled track (gate G3)

1336 instances (zero-overlap bindings 1); ABA(R) 0.5996 vs ABA(B0) 0.7013; side R 0.9019 / B0 0.9775; tissue R 0.6475 / B0 0.7141; gate ABA(R) ≥ ABA(B0): **fail**. the controlled-track truth is the SynthSeg lookup itself (host_targets on the SynthSeg map); no B0* column, it would be 1.0 by construction.

All controlled instances: B0 wrong on 0.2987; R rescues 187, harms 323, both right 614, both wrong 212.

Recognition–binding gap: no instance meets the condition (local 13-host Dice >= 0.8 in the bind window and a detection matched at the source's U threshold).

End to end (predicted lesions matched to a truth at the U threshold): 103 lesions, ABA(R) 0.6796 vs ABA(B0) 0.5340.

| split | n | ABA(R) | ABA(B0) | rescue | harm |
|---|---|---|---|---|---|
| bmsr | 824 | 0.5825 | 0.7306 | 90 | 212 |
| isles | 259 | 0.6100 | 0.7027 | 32 | 56 |
| pdgm | 253 | 0.6443 | 0.6047 | 65 | 55 |
| <5 | 558 | 0.6022 | 0.7634 | 57 | 147 |
| 5-10 | 390 | 0.5974 | 0.6769 | 48 | 79 |
| >=10 | 388 | 0.5979 | 0.6366 | 82 | 97 |

Sequence-type accuracy 0.9905 over 1056 volumes.

Not in this report: SibBMS 10-subject MS plaque check (U, report only; a separate script: scripts/aur_eval_sibbms.py); fastMRI 433-volume reliable-slice A Dice and 1297-box host agreement (report only; a separate script: scripts/aur_eval_fastmri.py); end-to-end R on a human-labelled set (Level R: the sheet is exported, the statistics wait for the readers).
