# Six-case replay A/R findings — 2026-10-17

**NOT_EVIDENCE.** This is an anonymous, aggregate-only analysis of the completed frozen validation replay. No inference, training or repository changes were performed by this analysis. Six volumes (two each BMSR/T1c, ISLES/DWI and PDGM/FLAIR) provide 55 controlled bindings; 33 come from one volume. Population and causal conclusions are not supported.

## A: five zero-Dice entities are absent predictions

All 30 volume×entity combinations below have nonzero pseudo-reference voxels and zero predicted voxels. This establishes absent output, rather than nonempty masks with zero overlap, in these six volumes. Their mean presence scores are nonzero and often near 1.

| label | repository name | volumes GT>0 / pred=0 | GT voxels total | pred voxels total | presence mean | presence min |
|---|---|---|---|---|---|---|
| 5 | L inf lat vent | 6/6 | 3382 | 0 | 0.9344 | 0.6290 |
| 44 | R inf lat vent | 6/6 | 4244 | 0 | 0.9284 | 0.5745 |
| 18 | L amygdala | 6/6 | 9732 | 0 | 0.9509 | 0.7105 |
| 26 | L accumbens | 6/6 | 3656 | 0 | 0.9300 | 0.5829 |
| 58 | R accumbens | 6/6 | 3777 | 0 | 0.9417 | 0.6523 |

Whole-volume A inference averages dense sigmoid probabilities, then applies argmax and the fixed entity threshold; it does **not** apply entity presence. Therefore inference presence gating cannot be the direct mechanism removing these A outputs. The saved data do not reveal whether the underlying issue is training, dense mask amplitude, competition with other entities, reference quality or another pathway. Label names are read from the repository ontology table, not recalled from memory.

## R: a large coarse-versus-dense A context gap survives grid controls

| A context / control | mean host Dice across55 bind crops | range |
|---|---|---|
| saved_whole_volume_local_a | 0.7377 | 0.7000–0.8293 |
| single_crop_dense_a | 0.7452 | 0.6996–0.8369 |
| same_grid_nearest_dense_a | 0.7440 | 0.6960–0.8354 |
| same_grid_centre_floor_dense_a | 0.7402 | 0.6913–0.8333 |
| same_grid_centre_ceil_dense_a | 0.7487 | 0.7014–0.8354 |
| coarse_a_ungated | 0.0485 | 0.0328–0.0643 |
| coarse_a_gated | 0.0487 | 0.0356–0.0642 |
| coarse_a_gated_centre_floor_gt | 0.0502 | 0.0357–0.0672 |
| coarse_a_gated_centre_ceil_gt | 0.0510 | 0.0372–0.0703 |

The single-crop dense map remains 0.7440 after the same nearest downsampling as its pseudo-reference, while original coarse A is 0.0485 ungated and 0.0487 gated. Every crop has the large gap. Presence gating changes a mean 0.424% of hard entity labels and does not account for the gap; removing it alone would leave coarse A poor.

Fixed sampling sensitivity controls use the lower and upper voxel around each (2,4,4) patch center. Dense A remains 0.7402/0.7487 and gated coarse A 0.0502/0.0510. Thus ordinary nearest-versus-center sample phase does not explain this large gap in these cases. This is a descriptive comparison of hard maps on the same grid, not calibrated equivalence of dense and coarse probabilities; exact feature-grid alignment/receptive-field behavior is not fully excluded.

By source, using original nearest-grid control:

| source | volumes | bindings | dense same-grid A | coarse ungated A | coarse gated A |
|---|---|---|---|---|---|
| bmsr | 2 | 13 | 0.7835 | 0.0530 | 0.0533 |
| isles | 2 | 40 | 0.7289 | 0.0469 | 0.0471 |
| pdgm | 2 | 2 | 0.7899 | 0.0498 | 0.0498 |

Actual R consumes independent soft host masks, not these hard argmax maps. Across bindings, mean 22.6% of valid voxels exceed 0.5 in at least two host masks; a mean 1.33 host masks exceed 0.5 per voxel. The true-host mask has mean Dice 0.0567 and median 0.00219 against nearest-grid pseudoGT. Its mean soft probability inside the true-host region has median 0.00107. Some channels are broad/co-overlapping, while others nearly miss their anatomical host; calling all masks uniformly diffuse would overstate the observation.

Actual geometry marks a median 12/13 hosts present and gives a median 8 candidates overlap > 0.05. The geometry rule sets distance to zero for those overlaps; all bindings have 3–12 such zero-distance candidates. This shows poor specificity of the observed coarse context, not that the relation head necessarily follows the overlap maximum.

## R: observed answers and target consistency

| source | n | R correct | B0 correct | rescue | harm | overlap-argmax correct |
|---|---|---|---|---|---|---|
| bmsr | 13 | 10 | 10 | 2 | 2 | 1 |
| isles | 40 | 21 | 24 | 10 | 13 | 4 |
| pdgm | 2 | 1 | 1 | 0 | 0 | 0 |

Overall R is 32/55=0.5818 versus B0 35/55=0.6364, with 12 rescues and 15 harms. Coarse overlap argmax is correct 5/55 and agrees with R 9/55; therefore neither poor overlap argmax nor its disagreement with truth proves the head itself is the causal failure. All 55 local-crop pseudo-host targets equal whole-volume targets, and all crop shares equal 1.0. Target truncation/host-rule discrepancy was not observed in these six cases; it remains untested elsewhere.

The three selected dense queries with IoU≥0.5 have coarse U probabilities>0.5 on 99.93–100% of the valid crop. In both PDGM crops, every coarse voxel exceeds0.5, despite dense selected-query IoU 0.8563/0.9066. This directly shows a dense/coarse event geometry mismatch for these examples. It does not establish general U learning failure: useful connected components may exist inside a query with low whole-mask IoU, and max/aggregation can damage instance separation.

## Sampling and lesion-region limits

Four of 55 complete GT lesions vanish under the original nearest coarse sampling. Fixed lower/upper patch-center sampling loses 4/16 respectively. Therefore coarse U target IoU for small lesions is sample-phase-sensitive and cannot substitute for dense overlap or whole-volume detection. The large-event coarse U saturation observations above do not depend on those vanished lesions.

Mean volume host Dice on lesion voxels is0.3637, versus0.7792 outside lesions. Every volume has lower lesion-region Dice. These region macros average different present-host sets and are not an additive decomposition; the gap does not establish that ignoring lesions during A loss caused the error.

## Next single-variable validation diagnostic

Proposed, **not executed**: retain the original full 716-instance controlled validation set and replace only the A probability source used for R geometry. For each unchanged GT-centred crop and unchanged selected query, compute dense predicted entity sigmoid probabilities, block-average over the fixed (2,4,4) patch footprints to the coarse grid, cast to the original coarse-mask dtype and multiply the original entity-presence sigmoid. Reuse `host_masks_from_entities`; recompute geometry with the unchanged original coarse U, validity and coordinates; pass the unchanged embeddings/tokens into the unchanged relation head.

Freeze checkpoint, crop, precision, all query selection/IDs, U masks/presence, 14-class vocabulary, cohort, original B0 and all thresholds. Require baseline reproduction and identical emitted A/U outputs and query IDs/IoU. Report paired R ABA and rescue/harm against baseline R and unchanged B0, source/size/host strata and geometry changes. No training, test selection, pseudoGT substitution or threshold search. Average pooling matches the coarse patch footprint and preserves small probability mass.

This diagnostic isolates sensitivity to the A geometry source. A null/adverse result would not exonerate A, because coarse U remains degraded and the relation head may be sensitive to input-distribution change. A positive result would justify a separate model experiment, not establish true host accuracy or count as a formal R pass. Ground-truth-U geometry replacement could be a later separate oracle, never combined in the same first intervention.

## Verification and reproduction

All saved coarse A Dice scores reproduce exactly from the NPZ arrays; saved geometry overlap equals the JSON entries exactly. A CPU unquantized dot-product cross-check differs from bf16-autocast overlap by at most 0.003576 and is not substituted for actual geometry. The parent replay already verified all six original detection/GT paths and all 55 controlled bindings; this analysis performs no new model forward.

Public artifacts: `REPORT.md`, `public_summary.json`. Local-only detail: `analysis.json`, `a_cases.csv`, `a_entities.csv`, `r_bindings.csv`, `r_host_soft_masks.csv`, `r_strata.csv`, input hashes and scripts. Public files contain no per-case or per-instance tables.

```sh
env CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 nice -n 19 python /tmp/aur-six-ar-analysis-20261017/analyze_six.py
env CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 nice -n 19 python /tmp/aur-six-ar-analysis-20261017/make_public_report.py
```
