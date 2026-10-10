# AnatoBind-Brain test-set evaluation (NOT_EVIDENCE: every number rests on SynthSeg / lesion pseudo-labels)

## A (gate G2, part 1)

13-host macro Dice 0.0727 over 4 A-supervised volumes (entity macro 0.0564); gate ≥ 0.8: **fail**. Unsupervised (QC-failed) rows: 0, host macro nan (not gated).

| source | volumes | host macro Dice |
|---|---|---|
| bmsr | 1 | 0.0659 |
| isles | 1 | 0.0668 |
| pdgm | 1 | 0.0878 |
| sibbms | 1 | 0.0702 |

| host | Dice |
|---|---|
| white_matter_left | 0.0516 |
| white_matter_right | 0.0484 |
| cortex_left | 0.1126 |
| cortex_right | 0.0799 |
| thalamus_left | 0.0000 |
| thalamus_right | 0.0000 |
| basal_ganglia_left | 0.0000 |
| basal_ganglia_right | 0.0000 |
| brainstem | 0.0716 |
| cerebellum_left | 0.2054 |
| cerebellum_right | 0.3751 |
| other_deep_grey_left | 0.0000 |
| other_deep_grey_right | 0.0000 |

## U (gate G2, part 2)

Sensitivity at the reference detector's false positives per scan on the same patients; pass = ≥ reference − 0.05; the gate reads one sequence per source (the one with the reference's lesion labels); gate: **fail** (1 sources judged)

| source/sequence | gated | scans | GT (ignored) | thr | selection | sensitivity | FP/scan | budget | ref sens | ref thr | ref scans | ref GT | grid | <5 mm | 5–10 mm | ≥10 mm | pass |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| pdgm/FLAIR | True | 1 | 1 (0) | 0.65 | fixed (chosen on the validation split) | 0.0000 | 0.000 | 0.000 | 1.0000 | 0.60 | 1 | 1 | same 1 mm grid | nan | nan | 0.0000 | False |

## R, controlled track (gate G3)

1 instances (zero-overlap bindings 0); ABA(R) 0.0000 vs ABA(B0) 1.0000; side R 1.0000 / B0 1.0000; tissue R 0.0000 / B0 1.0000; gate ABA(R) ≥ ABA(B0): **fail**. the controlled-track truth is the SynthSeg lookup itself (host_targets on the SynthSeg map); no B0* column, it would be 1.0 by construction.

All controlled instances: B0 wrong on 0.0000; R rescues 0, harms 1, both right 0, both wrong 0.

Recognition–binding gap: no instance meets the condition (local 13-host Dice >= 0.8 in the bind window and a detection matched at the source's U threshold).

End to end (predicted lesions matched to a truth at the U threshold): 0 lesions, ABA(R) nan vs ABA(B0) nan.

| split | n | ABA(R) | ABA(B0) | rescue | harm |
|---|---|---|---|---|---|
| pdgm | 1 | 0.0000 | 1.0000 | 0 | 1 |
| >=10 | 1 | 0.0000 | 1.0000 | 0 | 1 |

Sequence-type accuracy 0.5000 over 4 volumes.

Not in this report: SibBMS 10-subject MS plaque check (U, report only; a separate script: scripts/aur_eval_sibbms.py); fastMRI 433-volume reliable-slice A Dice and 1297-box host agreement (report only; a separate script: scripts/aur_eval_fastmri.py); end-to-end R on a human-labelled set (Level R: the sheet is exported, the statistics wait for the readers).
