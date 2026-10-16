# AnatoBind-Brain test-set evaluation (NOT_EVIDENCE: every number rests on SynthSeg / lesion pseudo-labels)

## A (gate G2, part 1)

13-host macro Dice 0.8035 over 394 A-supervised volumes (entity macro 0.6513); gate ≥ 0.8: **pass**. Unsupervised (QC-failed) rows: 11, host macro 0.7645 (not gated).

| source | volumes | host macro Dice |
|---|---|---|
| bmsr | 120 | 0.8111 |
| isles | 32 | 0.7441 |
| pdgm | 164 | 0.8137 |
| sibbms | 78 | 0.7945 |

| host | Dice |
|---|---|
| white_matter_left | 0.7785 |
| white_matter_right | 0.7711 |
| cortex_left | 0.7179 |
| cortex_right | 0.7196 |
| thalamus_left | 0.8628 |
| thalamus_right | 0.8510 |
| basal_ganglia_left | 0.7766 |
| basal_ganglia_right | 0.7884 |
| brainstem | 0.8791 |
| cerebellum_left | 0.9071 |
| cerebellum_right | 0.9049 |
| other_deep_grey_left | 0.7088 |
| other_deep_grey_right | 0.7792 |

## U (gate G2, part 2)

Sensitivity at the reference detector's false positives per scan on the same patients; pass = ≥ reference − 0.05; the gate reads one sequence per source (the one with the reference's lesion labels); gate: **fail** (3 sources judged)

| source/sequence | gated | scans | GT (ignored) | thr | selection | sensitivity | FP/scan | budget | ref sens | ref thr | ref scans | ref GT | grid | <5 mm | 5–10 mm | ≥10 mm | pass |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| bmsr/T1c | True | 41 | 470 (61) | 0.70 | test-selected (optimistic: the best of 19 thresholds on this very set) | 0.0043 | 0.049 | 0.878 | 0.6961 | 0.65 | 41 | 464 | AUR 1 mm resampled vs native (approximate) | 0.0036 | 0.0068 | 0.0000 | False |
| isles/DWI | True | 20 | 138 (11) | 0.60 | test-selected (optimistic: the best of 19 thresholds on this very set) | 0.0290 | 1.200 | 1.600 | 0.5943 | 0.50 | 20 | 175 | AUR 1 mm resampled vs native 2 mm (approximate) | 0.0147 | 0.0488 | 0.0345 | False |
| pdgm/FLAIR | True | 41 | 53 (19) | 0.75 | test-selected (optimistic: the best of 19 thresholds on this very set) | 0.5094 | 0.146 | 0.366 | 0.9057 | 0.60 | 41 | 53 | same 1 mm grid | 0.0000 | 0.0000 | 0.5625 | False |
| pdgm/T1c | False | 41 | 57 (15) | nan | — | nan | nan | nan | nan | nan | — | — | — | nan | nan | nan | None |

## R, controlled track (gate G3)

716 instances (zero-overlap bindings 1); ABA(R) 0.5684 vs ABA(B0) 0.6983; side R 0.9106 / B0 0.9818; tissue R 0.6006 / B0 0.7123; gate ABA(R) ≥ ABA(B0): **fail**. the controlled-track truth is the SynthSeg lookup itself (host_targets on the SynthSeg map); no B0* column, it would be 1.0 by construction.

All controlled instances: B0 wrong on 0.3017; R rescues 122, harms 215, both right 285, both wrong 94.

Recognition–binding gap (local 13-host Dice >= 0.8 in the bind window and a detection matched at the source's U threshold): 11 instances, B0 wrong on 0.5455; R rescues 1, harms 1.

End to end (predicted lesions matched to a truth at the U threshold): 38 lesions, ABA(R) 0.6579 vs ABA(B0) 0.6316.

| split | n | ABA(R) | ABA(B0) | rescue | harm |
|---|---|---|---|---|---|
| bmsr | 469 | 0.5458 | 0.7249 | 75 | 159 |
| isles | 137 | 0.5620 | 0.6496 | 27 | 39 |
| pdgm | 110 | 0.6727 | 0.6455 | 20 | 17 |
| <5 | 353 | 0.5326 | 0.7252 | 52 | 120 |
| 5-10 | 200 | 0.5800 | 0.6850 | 42 | 63 |
| >=10 | 163 | 0.6319 | 0.6564 | 28 | 32 |

Sequence-type accuracy 1.0000 over 405 volumes.

Not in this report: SibBMS 10-subject MS plaque check (U, report only; a separate script: scripts/aur_eval_sibbms.py); fastMRI 433-volume reliable-slice A Dice and 1297-box host agreement (report only; a separate script: scripts/aur_eval_fastmri.py); end-to-end R on a human-labelled set (Level R: the sheet is exported, the statistics wait for the readers).
