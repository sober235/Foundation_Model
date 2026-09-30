# S4 data step 1: SynthSeg teacher labels on the SibBMS T1w volumes (2026-09-30, user-approved)

Read-only record of the run; the design of S4 is still being agreed section by section, so nothing here is a model
result. Every number is from the files named beside it.

## Input

`/data2/congcong/data/FM_data/SibBMS_ms/sibbms.zip` (11 GB) extracted with `unzip -q -n` into
`/data2/congcong/data/FM_data/SibBMS_ms/sibbms/` (13:03:52–13:05:52): `Output/{MS,Norm,Annotation}/sub-XXX/ses-YYY/…`,
1425 NIfTI files. Non-contrast T1w (`sub-*_ses-???_T1w.nii.gz`, which excludes `…_ce-GAD_T1w`): 371 = MS 271 + Norm 100.
MS and Norm both number their subjects from sub-001, so the plain stems collide (310 distinct of 371); cohort-prefixed
symlinks were staged in `/data2/congcong/data/FM_data/derived/synthseg/sibbms/inputs/{MS,Norm}_sub-XXX_ses-YYY_T1w.nii.gz`
(371 unique).

## Command (repository root; the runner already used for PDGM, BMSR, HCP and fastMRI)

```
PYTHONNOUSERSITE=1 PYTHONPATH=. nice -n 19 ~/anaconda3/envs/nvgen/bin/python scripts/run_synthseg_fastmri_brain.py \
  --glob '/data2/congcong/data/FM_data/derived/synthseg/sibbms/inputs/*_T1w.nii.gz' \
  --work-root /data2/congcong/data/FM_data/derived/synthseg/sibbms --workers 3 --threads 12
```

CPU only, 3 workers × 12 threads, 15 chunks of 25; started 13:08:51, `finished: 370/371 ok` after 93.0 min (about
18 min per chunk of 25, i.e. 43 s per volume per worker); log `derived/synthseg/sibbms/run_sibbms_20260930_130851.log`,
manifest `derived/synthseg/sibbms/manifest.csv`, outputs `derived/synthseg/sibbms/seg_native/<Cohort>_sub-XXX_ses-YYY_T1w_seg.nii.gz`.

## Outputs (`sibbms_seg_check.py`, output in `sibbms_seg_check.txt`)

- 370 maps (MS 270, Norm 100), all 197 × 233 × 189 at 1 mm, RAS, the native grid of SibBMS (template space,
  skull-stripped). 356 maps hold all 33 SynthSeg labels, 6 hold 31–32, 8 hold 2–7 labels (failures, below).
- Volume of all labels > 0: median 2013 mL (5th–95th percentile 1840–2080); the seven host classes of
  `anatobind.eval.geometry.HOST_CLASSES`: median 1719 mL; ventricles and CSF landmarks: median 283 mL. These are
  template-space volumes (the template brain is larger than a native one), fine for labels, not for volumetry.

## Failures and exclusions

| case | what | cause seen in the input |
|---|---|---|
| MS sub-057 ses-001 | SynthSeg refused (`input should have 3 dimensions, had 2`); manifest `missing` | the T1w file is a 256 × 256 2-D image of 35 KB (1.17 mm), not a volume |
| MS sub-011 ses-001 | 7 labels, 10.5 mL | T1w has 2.8 % non-zero voxels (nearly empty image) |
| MS sub-027 ses-005 | 2 labels, 0.1 mL | 3.7 % non-zero |
| MS sub-070 ses-003 / 004 / 005 | 3 labels, 0 mL | 7–27 % non-zero, SynthSeg found no brain (not investigated further) |
| Norm sub-010, sub-020, sub-037 (ses-001) | 3 labels, ≤ 0.1 mL | 6–10 % non-zero, low intensities (max ≈ 2100) |

Usable teacher maps: **362** = MS 265 sessions of 91 subjects + Norm 97. The eight poor maps and the 2-D file are
excluded from S4's data by name (a list, not a deletion); whether the eight sessions' FLAIR images are themselves
usable was not checked. SibBMS's own MS lesion annotations (`Output/Annotation`, 49 subjects) were not used here.

## Next (S4 design, section "data", awaiting the user's decisions of 2026-09-30)

The maps are the teacher labels for the same sessions' FLAIR (same space). Sources under discussion: SibBMS (this
record), UCSF-PDGM (501, maps in `derived/synthseg/pdgm`), UCSF-BMSR (461, `derived/synthseg/bmsr`); fastMRI FLAIR for
validation only. The spec will name the exclusion list above.
