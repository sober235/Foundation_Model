# Dataset903 build, preprocess, and splits (2026-09-28)

`scripts/brain_detector_prepare.py` builds nnU-Net raw `Dataset903_FastMRIBrainSmallLesion` from the Gate 0.5
registry (165 lesion FLAIR volumes, 1297 small lesions) plus the 88 FLAIR volumes whose every fastMRI+ row is
"Normal for age", preprocesses it with nnU-Net's planner, and writes the patient-level five-fold splits.

## Step 2: raw stage

```bash
source scripts/nnunet_env.sh
D=docs/verification/2026-09-28/brain_detector
PYTHONPATH=. nice -n 19 python scripts/brain_detector_prepare.py --stage raw | tee "$D/prepare_raw.txt"
```

Last line (full output in `prepare_raw.txt`):

```
wrote /data2/congcong/data/FM_data/derived/nnunet/raw/Dataset903_FastMRIBrainSmallLesion: 253 cases (165 lesion, 88 normal), 0 normal cases with label voxels
```

Matches the controller's expectation exactly: 253 cases, 165 lesion + 88 normal, 0 normal cases with any label
voxel. `imagesTr` and `labelsTr` each hold 253 files; `cases.json` records `{patient_id, kind, n_label_voxels}`
per case.

## Step 3: label check

One-off script (not committed), re-derives each lesion case's label map along the same path as
`brain_detector_prepare.py` (`match_registry(reg rows, merged_lesions(csv rows, n_rows))` ->
`check_members_inside` -> `paint_members`) and compares it voxel-for-voxel to the written
`labelsTr/<case>.nii.gz`; also checks the per-case registry lesion count and the total.

```bash
source scripts/nnunet_env.sh
PYTHONPATH=. python /tmp/.../label_check.py | tee "$D/label_check.txt"
```

Output (`label_check.txt`, head and tail):

```
registry total lesions: 1297
file_brain_AXFLAIR_200_6002425: OK, 6 registry lesions, label map matches re-derived paint_members
...
checked 165 lesion cases, total lesions 1297
ALL OK: every lesion case's label map matches re-derived paint_members; total registry lesions = 1297
```

All 165 lesion cases' written label maps match the re-derived `paint_members` output exactly; the registry's
1297 lesions are all accounted for with none dropped or duplicated.

## Step 4: plan and preprocess

```bash
source scripts/nnunet_env.sh
nice -n 19 nnUNetv2_plan_and_preprocess -d 903 -c 2d 3d_fullres --verify_dataset_integrity -np 4 2>&1 | tee "$D/plan_preprocess.txt"
```

`verify_dataset_integrity` passed with no error messages. Wall time: started 11:40:00, `nnUNetPlans.json`
written 11:40:16, preprocessing (both configurations, 253 cases each) finished ~11:40:59 — under one minute
total (CPU, `-np 4`), well inside the D8 24-hour bound (no GPU used).

Plans (from `nnUNetPlans.json`, also in `plan_preprocess.txt`):

| config | patch_size | spacing (mm) | batch_size |
|---|---|---|---|
| 2d | [320, 320] | [0.6875, 0.6875] | 32 |
| 3d_fullres | [16, 320, 320] | [5.0, 0.6875, 0.6875] | 2 |

`3d_lowres` was dropped (image size too close to `3d_fullres`: both `[16, 320, 320]`).

## Step 5: splits

```bash
source scripts/nnunet_env.sh
PYTHONPATH=. python scripts/brain_detector_prepare.py --stage splits | tee "$D/splits.txt"
```

```
wrote /data2/congcong/data/FM_data/derived/nnunet/preprocessed/Dataset903_FastMRIBrainSmallLesion/splits_final.json: cases per fold {0: 51, 1: 51, 2: 51, 3: 50, 4: 50}
```

Verified directly against `splits_final.json` and `cases.json`:
- 5 folds, val sizes 51/51/51/50/50 = 253, train sizes 202/202/202/203/203.
- Every one of the 253 cases appears in exactly one fold's `val` list; the union of all `val` lists equals the
  full case set.
- 165 distinct lesion patients, 88 distinct normal patients, zero overlap between the two sets.
- No patient (lesion or normal) has its cases split across two different folds.

## Files produced

- `raw/Dataset903_FastMRIBrainSmallLesion/{imagesTr,labelsTr,dataset.json,cases.json}` (253 + 253 files)
- `preprocessed/Dataset903_FastMRIBrainSmallLesion/{nnUNetPlans.json, dataset_fingerprint.json,
  nnUNetPlans_2d/, nnUNetPlans_3d_fullres/, gt_segmentations/, splits_final.json}`
- `docs/verification/2026-09-28/brain_detector/{prepare_raw.txt, label_check.txt, plan_preprocess.txt, splits.txt, dataset.md}`

No GPU was used; no training was run (Tasks 6-7).
