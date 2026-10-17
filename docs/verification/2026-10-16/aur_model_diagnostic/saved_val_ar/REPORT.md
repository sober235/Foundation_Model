# Anonymous saved-val A/R analysis — 2026-10-16

All A/R references are SynthSeg pseudo-labels: **NOT_EVIDENCE**. These are computed descriptive aggregates from the frozen saved validation metrics; no inference or model change was run.

Repository snapshot: this report, the readable schema, aggregate CSVs, verification and reproducer are retained here. Large `analysis.json`/`schema.json` outputs remain local under the original `/tmp/aur-ar-saved-val-analysis-20261016/` and the replay directory’s `analysis_archives/`; they are not part of this public subset. CSV line endings were normalized to LF without changing values.

## Integrity and scope

405 saved volumes; 394 A-supervised volumes from 96 source-qualified patients; 11 A-unsupervised volumes reported separately. Controlled R has 716 instances from 134 volumes / 81 patients. The 718 nonignored U GT instances include two without R because `r_supervised=False` (one BMSR, one ISLES).

Recomputed A host/entity macro and R/B0 ABA match `aggregate.json` within 2.3e-16; both-right/both-wrong/rescue/harm counts match exactly (285/94/122/215). All 716 R GT volumes, boxes and size strata match the corresponding U GT rows. No numeric R context field is missing or nonfinite.

Input hashes:

| file | SHA256 |
|---|---|
| cases.json | 5950f91d1070dc47f9f3770c3f03bc6a5a3e5b3d0113a2d0109b3e3522f379bf |
| aggregate.json | 9b83bc1d8d6b3116962e400d5b6aaf103c17408e004adee842fc8824baa05670 |

## A source × sequence

These are mean row macro Dice scores over A-supervised volumes, using the original nan-mean convention. Repeated patient volumes are not independent replications. Full 13-host and 32-entity breakdowns, valid counts and zero counts are in the two corresponding CSVs.

| source | sequence | volumes | host macro | entity macro |
|---|---|---|---|---|
| bmsr | FLAIR | 40 | 0.7841 | 0.6344 |
| bmsr | T1 | 40 | 0.8493 | 0.6962 |
| bmsr | T1c | 40 | 0.8000 | 0.6491 |
| isles | ADC | 16 | 0.7646 | 0.6182 |
| isles | DWI | 16 | 0.7237 | 0.5876 |
| pdgm | FLAIR | 41 | 0.7925 | 0.6395 |
| pdgm | T1 | 41 | 0.8437 | 0.6844 |
| pdgm | T1c | 41 | 0.8125 | 0.6546 |
| pdgm | T2 | 41 | 0.8061 | 0.6481 |
| sibbms | FLAIR | 26 | 0.7928 | 0.6502 |
| sibbms | T1 | 26 | 0.8506 | 0.6975 |
| sibbms | T2 | 26 | 0.7402 | 0.5924 |

The aggregate host macro 0.8035 masks ISLES/DWI 0.7237, SibBMS/T2 0.7402, BMSR/FLAIR 0.7841, and PDGM/FLAIR 0.7925. The weakest overall hosts are left other-deep-grey 0.7088 and left/right cortex 0.7179/0.7196. On ISLES/DWI, left other-deep-grey is 0.5965, left basal ganglia 0.6333, and left cortex 0.6367; SibBMS/T2 left/right cortex are 0.6180/0.6137.

## A entity collapse candidates

Five entity labels have exactly zero Dice on every finite A-supervised volume. Label names below are parsed from the repository `docs/verification/2026-09-16-brain-probe/probe_common.py::LABEL_NAMES`, not recalled from an external ontology. Current `anatobind/anatomy/labels.py::STUDENT` groups 5/44 in ventricles; `anatobind/eval/geometry.py::LANDMARKS` describes inferior-lateral ventricles as landmarks.

| label | repository name | finite | zero | nonfinite |
|---|---|---|---|---|
| 5 | L inf lat vent | 394 | 394 | 0 |
| 44 | R inf lat vent | 391 | 391 | 3 |
| 18 | L amygdala | 394 | 394 | 0 |
| 26 | L accumbens | 394 | 394 | 0 |
| 58 | R accumbens | 394 | 394 | 0 |

Source × sequence counts below are zero / finite; denominator volumes appears separately. Every finite value is zero. Nonfinite values use the evaluator definition: both reference and prediction are empty. A finite zero proves no intersection, but cannot distinguish absent prediction, wrong identity, spatial mismatch or presence filtering without voxel/presence data.

| source | sequence | volumes | 5 zero/finite | 44 zero/finite | 18 zero/finite | 26 zero/finite | 58 zero/finite |
|---|---|---|---|---|---|---|---|
| bmsr | FLAIR | 40 | 40/40 | 40/40 | 40/40 | 40/40 | 40/40 |
| bmsr | T1 | 40 | 40/40 | 40/40 | 40/40 | 40/40 | 40/40 |
| bmsr | T1c | 40 | 40/40 | 40/40 | 40/40 | 40/40 | 40/40 |
| isles | ADC | 16 | 16/16 | 16/16 | 16/16 | 16/16 | 16/16 |
| isles | DWI | 16 | 16/16 | 16/16 | 16/16 | 16/16 | 16/16 |
| pdgm | FLAIR | 41 | 41/41 | 41/41 | 41/41 | 41/41 | 41/41 |
| pdgm | T1 | 41 | 41/41 | 41/41 | 41/41 | 41/41 | 41/41 |
| pdgm | T1c | 41 | 41/41 | 41/41 | 41/41 | 41/41 | 41/41 |
| pdgm | T2 | 41 | 41/41 | 41/41 | 41/41 | 41/41 | 41/41 |
| sibbms | FLAIR | 26 | 26/26 | 25/25 | 26/26 | 26/26 | 26/26 |
| sibbms | T1 | 26 | 26/26 | 25/25 | 26/26 | 26/26 | 26/26 |
| sibbms | T2 | 26 | 26/26 | 25/25 | 26/26 | 26/26 | 26/26 |

All 32 nonbackground entity means:

| label | repository name | mean Dice | finite | zero |
|---|---|---|---|---|
| 2 | L cerebral WM | 0.7785 | 394 | 0 |
| 41 | R cerebral WM | 0.7711 | 394 | 0 |
| 3 | L cortex | 0.7179 | 394 | 0 |
| 42 | R cortex | 0.7196 | 394 | 0 |
| 4 | L lat ventricle | 0.8402 | 394 | 0 |
| 43 | R lat ventricle | 0.8326 | 394 | 0 |
| 5 | L inf lat vent | 0.0000 | 394 | 394 |
| 44 | R inf lat vent | 0.0000 | 391 | 391 |
| 7 | L cerebellum WM | 0.7603 | 394 | 3 |
| 46 | R cerebellum WM | 0.7636 | 394 | 0 |
| 8 | L cerebellum cortex | 0.8438 | 394 | 0 |
| 47 | R cerebellum cortex | 0.8461 | 394 | 0 |
| 10 | L thalamus | 0.8628 | 394 | 0 |
| 49 | R thalamus | 0.8510 | 394 | 0 |
| 11 | L caudate | 0.7844 | 394 | 0 |
| 50 | R caudate | 0.7845 | 394 | 0 |
| 12 | L putamen | 0.7784 | 394 | 0 |
| 51 | R putamen | 0.7880 | 394 | 0 |
| 13 | L pallidum | 0.6941 | 394 | 2 |
| 52 | R pallidum | 0.7534 | 394 | 6 |
| 14 | 3rd ventricle | 0.7376 | 394 | 0 |
| 15 | 4th ventricle | 0.7470 | 394 | 0 |
| 16 | brainstem | 0.8791 | 394 | 0 |
| 17 | L hippocampus | 0.7173 | 394 | 1 |
| 53 | R hippocampus | 0.7326 | 394 | 3 |
| 18 | L amygdala | 0.0000 | 394 | 394 |
| 54 | R amygdala | 0.7578 | 394 | 4 |
| 24 | CSF | 0.5481 | 394 | 0 |
| 26 | L accumbens | 0.0000 | 394 | 394 |
| 58 | R accumbens | 0.0000 | 394 | 394 |
| 28 | L ventral DC | 0.7834 | 394 | 0 |
| 60 | R ventral DC | 0.7666 | 394 | 0 |

## R controlled context strata

Overall ABA(R)=0.5684 versus B0=0.6983; R rescues 122 B0 errors and harms 215 B0-correct instances. R harms 43.0% of B0-correct instances. Counts are instance-weighted and source, lesion size and query quality covary. Query IoU is for the selected dense >0.5 query within the GT-centred crop; it is not whole-volume detection IoU and not a direct measurement of the coarse R mask.

By source × sequence:

| group | n | ABA R | ABA B0 | rescue | harm |
|---|---|---|---|---|---|
| bmsr/T1c | 469 | 0.5458 | 0.7249 | 75 | 159 |
| isles/DWI | 137 | 0.5620 | 0.6496 | 27 | 39 |
| pdgm/FLAIR | 53 | 0.6792 | 0.6226 | 9 | 6 |
| pdgm/T1c | 57 | 0.6667 | 0.6667 | 11 | 11 |

By selected query IoU:

| group | n | ABA R | ABA B0 | rescue | harm |
|---|---|---|---|---|---|
| (0,0.01) | 486 | 0.5535 | 0.7140 | 80 | 158 |
| =0 | 1 | 1.0000 | 1.0000 | 0 | 0 |
| [0.01,0.1) | 117 | 0.5299 | 0.7009 | 19 | 39 |
| [0.1,0.3) | 30 | 0.6667 | 0.6333 | 7 | 6 |
| [0.3,0.5) | 16 | 0.6875 | 0.5000 | 5 | 2 |
| [0.5,1] | 66 | 0.6667 | 0.6515 | 11 | 10 |

604/716 instances have query IoU <0.1 and contain 197/215 harms (91.6%) versus 99/122 rescues. The >=0.1 group has n=112, 23 rescues and 18 harms. Query-IoU >=0.5 has n=66 and R/B0=0.6667/0.6515, but 59/66 are PDGM, four BMSR and three ISLES; this is not evidence that improving query IoU alone will cause the gain. Median query IoU is 0.00318 overall, 0.00240 for harms, 0.00329 for rescues.

By saved local A Dice:

| group | n | ABA R | ABA B0 | rescue | harm |
|---|---|---|---|---|---|
| <0.6 | 4 | 0.2500 | 1.0000 | 0 | 3 |
| [0.6,0.7) | 38 | 0.6316 | 0.7105 | 5 | 8 |
| [0.7,0.8) | 269 | 0.6059 | 0.6877 | 49 | 71 |
| [0.8,0.9) | 405 | 0.5407 | 0.7012 | 68 | 133 |

405 instances have local A Dice >=0.8, yet R/B0=0.5407/0.7012 with 68 rescues and 133 harms. Harms have mean local Dice 0.7911, compared with 0.7872 for rescues. This saved whole-volume hard-A statistic does not support a simple concentration of harm in low local Dice. It does not evaluate R's coarse or presence-gated A context.

Joint quality:

| group | n | ABA R | ABA B0 | rescue | harm |
|---|---|---|---|---|---|
| query_iou <0.5;local_dice <0.8 | 279 | 0.5914 | 0.6953 | 48 | 77 |
| query_iou <0.5;local_dice >=0.8 | 371 | 0.5337 | 0.7089 | 63 | 128 |
| query_iou >=0.5;local_dice <0.8 | 32 | 0.7188 | 0.6875 | 6 | 5 |
| query_iou >=0.5;local_dice >=0.8 | 34 | 0.6176 | 0.6176 | 5 | 5 |

By existing equivalent spherical diameter in mm:

| group | n | ABA R | ABA B0 | rescue | harm |
|---|---|---|---|---|---|
| 5-10 | 200 | 0.5800 | 0.6850 | 42 | 63 |
| <5 | 353 | 0.5326 | 0.7252 | 52 | 120 |
| >=10 | 163 | 0.6319 | 0.6564 | 28 | 32 |

Instances <5 mm account for 120 harms versus 52 rescues, with R/B0=0.5326/0.7252. Instances >=10 mm are closer (32 harms/28 rescues; R/B0=0.6319/0.6564). All 716 in-crop shares are **exactly 1.0**: no saved GT-instance truncation is observed, and this field cannot stratify the cohort. Host-target/context boundary discrepancies remain unmeasured.

By pseudo-truth side:

| group | n | ABA R | ABA B0 | rescue | harm |
|---|---|---|---|---|---|
| left | 385 | 0.5740 | 0.6883 | 72 | 116 |
| midline | 8 | 0.5000 | 1.0000 | 0 | 4 |
| right | 323 | 0.5635 | 0.7028 | 50 | 95 |

No no-host truths occur. Side agreement is 0.9106 for R versus 0.9818 for B0, so tissue choice is not the only observed error axis. Small midline strata require caution.

By pseudo-truth host:

| group | n | ABA R | ABA B0 | rescue | harm |
|---|---|---|---|---|---|
| basal_ganglia_left | 7 | 0.0000 | 0.5714 | 0 | 4 |
| basal_ganglia_right | 3 | 0.0000 | 0.3333 | 0 | 1 |
| brainstem | 8 | 0.5000 | 1.0000 | 0 | 4 |
| cerebellum_left | 46 | 0.7174 | 0.9348 | 0 | 10 |
| cerebellum_right | 45 | 0.7556 | 0.8889 | 2 | 8 |
| cortex_left | 215 | 0.8000 | 0.5907 | 68 | 23 |
| cortex_right | 189 | 0.6984 | 0.6190 | 47 | 32 |
| other_deep_grey_left | 4 | 0.0000 | 1.0000 | 0 | 4 |
| other_deep_grey_right | 1 | 0.0000 | 1.0000 | 0 | 1 |
| thalamus_left | 12 | 0.0000 | 0.6667 | 0 | 8 |
| thalamus_right | 6 | 0.0000 | 0.8333 | 0 | 5 |
| white_matter_left | 101 | 0.1584 | 0.7822 | 4 | 67 |
| white_matter_right | 79 | 0.2025 | 0.7975 | 1 | 48 |

R predicts cortex for 511/716 instances (truth cortex: 404/716); 158/215 harms predict cortex, including 103 harms on white-matter truths. White-matter truths total 180: R/B0 correct counts are 32/142, rescues/harms 5/115. R has zero correct answers on the 33 thalamus/basal-ganglia/other-deep-grey truths. These descriptive biases motivate inspection of coarse geometry, candidate inputs and 14-class probabilities, without identifying a causal mechanism from the saved scores.

## Supported, not observed, unresolved

| status | observation |
|---|---|
| supported | Five A entities have zero finite Dice across sources/sequences; raw count/presence diagnosis is needed. |
| supported | R harms are common with poor selected-query IoU and on small/white-matter truths; source and size confounding remain. |
| supported | High saved local full-volume A Dice does not remove R harm. |
| not observed | GT instance truncation by in_crop_share; all 716 are one. |
| unresolved | R coarse/presence-gated A/U mask quality, geometry/candidate inputs, crop/full-volume host-target discrepancies and relation-head contribution. |
| unresolved | A lesion-inside/outside error, SynthSeg reference-quality issues and true human relation accuracy. |

The replay should inspect selected query ID, coarse inputs, presence, candidate geometry, probabilities and local/full host targets. This analysis does not choose or claim a model repair. All A/R pseudo-reference findings remain NOT_EVIDENCE.

## Reproduction and artifacts

```sh
env CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 nice -n 19 python /tmp/aur-ar-saved-val-analysis-20261016/analyze_saved_val.py
```

`schema.json` / `SCHEMA.md`: exact machine/readable schemas; `analysis.json`: all anonymous results and validation/provenance; `a_source_sequence_summary.csv`, `a_source_sequence_hosts.csv`, `a_source_sequence_entities.csv`; `r_controlled_strata.csv`, `r_context_by_outcome.csv`, `r_host_confusions.csv`; `run.log`: compact aggregate stdout. No case/patient identities or individual instance rows are exported.
