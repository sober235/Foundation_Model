# AnatoBind-Brain test-set evaluation (NOT_EVIDENCE: every number rests on SynthSeg / lesion pseudo-labels)

## A (gate G2, part 1)

13-host macro Dice 0.7568 over 394 A-supervised volumes (entity macro 0.6212); gate ≥ 0.8: **fail**. Unsupervised (QC-failed) rows: 11, host macro 0.7209 (not gated).

| source | volumes | host macro Dice |
|---|---|---|
| bmsr | 120 | 0.7722 |
| isles | 32 | 0.7100 |
| pdgm | 164 | 0.7545 |
| sibbms | 78 | 0.7570 |

| host | Dice |
|---|---|
| white_matter_left | 0.7310 |
| white_matter_right | 0.7296 |
| cortex_left | 0.6530 |
| cortex_right | 0.6657 |
| thalamus_left | 0.7908 |
| thalamus_right | 0.7843 |
| basal_ganglia_left | 0.7168 |
| basal_ganglia_right | 0.7033 |
| brainstem | 0.8440 |
| cerebellum_left | 0.8807 |
| cerebellum_right | 0.8845 |
| other_deep_grey_left | 0.7301 |
| other_deep_grey_right | 0.7241 |

## U (gate G2, part 2)

Sensitivity at the reference detector's false positives per scan on the same patients; pass = ≥ reference − 0.05; the gate reads one sequence per source (the one with the reference's lesion labels); gate: **fail** (3 sources judged)

| source/sequence | gated | scans | GT (ignored) | thr | selection | sensitivity | FP/scan | budget | ref sens | ref thr | ref scans | ref GT | grid | <5 mm | 5–10 mm | ≥10 mm | pass |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| bmsr/T1c | True | 41 | 470 (61) | 0.65 | test-selected (optimistic: the best of 19 thresholds on this very set) | 0.0106 | 0.780 | 0.878 | 0.6961 | 0.65 | 41 | 464 | AUR 1 mm resampled vs native (approximate) | 0.0108 | 0.0135 | 0.0000 | False |
| isles/DWI | True | 20 | 138 (11) | 0.60 | test-selected (optimistic: the best of 19 thresholds on this very set) | 0.0290 | 0.950 | 1.600 | 0.5943 | 0.50 | 20 | 175 | AUR 1 mm resampled vs native 2 mm (approximate) | 0.0147 | 0.0000 | 0.1034 | False |
| pdgm/FLAIR | True | 41 | 53 (19) | 0.60 | test-selected (optimistic: the best of 19 thresholds on this very set) | 0.5849 | 0.341 | 0.366 | 0.9057 | 0.60 | 41 | 53 | same 1 mm grid | 0.0000 | 0.0000 | 0.6458 | False |
| pdgm/T1c | False | 41 | 57 (15) | nan | — | nan | nan | nan | nan | nan | — | — | — | nan | nan | nan | None |

## R, controlled track (gate G3)

716 instances (zero-overlap bindings 2); ABA(R) 0.5796 vs ABA(B0) 0.6983; side R 0.9218 / B0 0.9749; tissue R 0.6173 / B0 0.7165; gate ABA(R) ≥ ABA(B0): **fail**. the controlled-track truth is the SynthSeg lookup itself (host_targets on the SynthSeg map); no B0* column, it would be 1.0 by construction.

All controlled instances: B0 wrong on 0.3017; R rescues 91, harms 176, both right 324, both wrong 125.

Recognition–binding gap (local 13-host Dice >= 0.8 in the bind window and a detection matched at the source's U threshold): 2 instances, B0 wrong on 0.0000; R rescues 0, harms 2.

End to end (predicted lesions matched to a truth at the U threshold): 43 lesions, ABA(R) 0.6047 vs ABA(B0) 0.5116.

| split | n | ABA(R) | ABA(B0) | rescue | harm |
|---|---|---|---|---|---|
| bmsr | 469 | 0.5416 | 0.7015 | 57 | 132 |
| isles | 137 | 0.6934 | 0.7591 | 13 | 22 |
| pdgm | 110 | 0.6000 | 0.6091 | 21 | 22 |
| <5 | 353 | 0.5722 | 0.7252 | 36 | 90 |
| 5-10 | 200 | 0.5550 | 0.7150 | 20 | 52 |
| >=10 | 163 | 0.6258 | 0.6196 | 35 | 34 |

Sequence-type accuracy 0.9975 over 405 volumes.

Not in this report: SibBMS 10-subject MS plaque check (U, report only; a separate script: scripts/aur_eval_sibbms.py); fastMRI 433-volume reliable-slice A Dice and 1297-box host agreement (report only; a separate script: scripts/aur_eval_fastmri.py); end-to-end R on a human-labelled set (Level R: the sheet is exported, the statistics wait for the readers).
