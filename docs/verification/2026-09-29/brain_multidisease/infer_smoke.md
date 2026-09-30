# Inference smoke: one fold 0 validation case per disease, fold 0 model only (plan Task 12 Step 3)

Each disease's first case of `splits_final.json[0]["val"]`, its native channels and its SynthSeg map, through
`scripts/infer_brain_disease.py` with `--folds 0` at the disease's operating threshold (`<disease>/verdict.json`).
GPU 0 was idle when the three runs were launched (2026-09-30, about 09:58); they ran one after another under
`nice -n 19` and each exited with code 0 (infarct 09:59:34, metastasis 09:59:54, glioma 10:00:16). Outputs under
`/data2/congcong/data/FM_data/derived/brain_disease/infer_smoke/<disease>/` (`input/`, `pred/`, `record.json`).

## Commands

```
PYTHONNOUSERSITE=1 PYTHONPATH=. nice -n 19 ~/anaconda3/envs/nvgen/bin/python scripts/infer_brain_disease.py --disease glioma \
  --images "/data2/congcong/data/FM_data/UCSF-PDGM_lh/PKG - UCSF-PDGM Version 5/UCSF-PDGM-v5/UCSF-PDGM-0008_nifti/UCSF-PDGM-0008_T1.nii.gz" \
           "/data2/congcong/data/FM_data/UCSF-PDGM_lh/PKG - UCSF-PDGM Version 5/UCSF-PDGM-v5/UCSF-PDGM-0008_nifti/UCSF-PDGM-0008_T1c.nii.gz" \
           "/data2/congcong/data/FM_data/UCSF-PDGM_lh/PKG - UCSF-PDGM Version 5/UCSF-PDGM-v5/UCSF-PDGM-0008_nifti/UCSF-PDGM-0008_T2.nii.gz" \
           "/data2/congcong/data/FM_data/UCSF-PDGM_lh/PKG - UCSF-PDGM Version 5/UCSF-PDGM-v5/UCSF-PDGM-0008_nifti/UCSF-PDGM-0008_FLAIR.nii.gz" \
  --anatomy /data2/congcong/data/FM_data/derived/synthseg/pdgm/seg_native/UCSF-PDGM-0008_T1_seg.nii.gz \
  --threshold 0.6 --out /data2/congcong/data/FM_data/derived/brain_disease/infer_smoke/glioma --folds 0 --gpu 0

PYTHONNOUSERSITE=1 PYTHONPATH=. nice -n 19 ~/anaconda3/envs/nvgen/bin/python scripts/infer_brain_disease.py --disease metastasis \
  --images /data2/congcong/data/FM_data/UCSF-BMSR_cbb/UCSF-BMSR/UCSF_BrainMetastases_TRAIN/100101A/100101A_T1pre.nii.gz \
           /data2/congcong/data/FM_data/UCSF-BMSR_cbb/UCSF-BMSR/UCSF_BrainMetastases_TRAIN/100101A/100101A_T1post.nii.gz \
           /data2/congcong/data/FM_data/UCSF-BMSR_cbb/UCSF-BMSR/UCSF_BrainMetastases_TRAIN/100101A/100101A_FLAIR.nii.gz \
  --anatomy /data2/congcong/data/FM_data/derived/synthseg/bmsr/seg_native/100101A_T1pre_seg.nii.gz \
  --threshold 0.65 --out /data2/congcong/data/FM_data/derived/brain_disease/infer_smoke/metastasis --folds 0 --gpu 0

PYTHONNOUSERSITE=1 PYTHONPATH=. nice -n 19 ~/anaconda3/envs/nvgen/bin/python scripts/infer_brain_disease.py --disease infarct \
  --images /data2/congcong/data/FM_data/ISLES_ltr/ISLES-2022/sub-strokecase0001/ses-0001/dwi/sub-strokecase0001_ses-0001_dwi.nii.gz \
           /data2/congcong/data/FM_data/ISLES_ltr/ISLES-2022/sub-strokecase0001/ses-0001/dwi/sub-strokecase0001_ses-0001_adc.nii.gz \
  --anatomy /data2/congcong/data/FM_data/derived/synthseg/isles/seg_native/sub-strokecase0001_ses-0001_dwi_seg.nii.gz \
  --threshold 0.5 --out /data2/congcong/data/FM_data/derived/brain_disease/infer_smoke/infarct --folds 0 --gpu 0
```

## Printed lines (the script's own; nnU-Net's lines end with `Segmentation export complete.` in each run)

```
channels expected: ('T1', 'T1c', 'T2', 'FLAIR')
1 lesions -> /data2/congcong/data/FM_data/derived/brain_disease/infer_smoke/glioma/record.json
右侧大脑白质存在肿瘤样异常，体积约 173 mL，累及大脑皮层。疑似胶质瘤。

channels expected: ('T1pre', 'T1post', 'FLAIR')
11 lesions -> /data2/congcong/data/FM_data/derived/brain_disease/infer_smoke/metastasis/record.json
左侧大脑皮层存在转移瘤样异常，体积约 1.7 mL；右侧大脑皮层存在转移瘤样异常，体积约 1.2 mL；右侧大脑皮层存在转移瘤样异常，体积约 605 mm³，累及大脑白质；左侧大脑皮层存在转移瘤样异常，体积约 372 mm³，累及大脑白质；左侧大脑白质存在转移瘤样异常，体积约 277 mm³；另有 6 处同类异常（还见于左侧小脑）。疑似脑转移瘤。

channels expected: ('DWI', 'ADC')
26 lesions -> /data2/congcong/data/FM_data/derived/brain_disease/infer_smoke/infarct/record.json
左侧大脑白质存在梗死样异常，体积约 744 mm³，累及大脑皮层；右侧大脑白质存在梗死样异常，体积约 696 mm³，累及大脑皮层；左侧基底节存在梗死样异常，体积约 384 mm³；左侧大脑皮层存在梗死样异常，体积约 280 mm³；左侧大脑皮层存在梗死样异常，体积约 272 mm³，累及大脑白质；另有 21 处同类异常（还见于右侧大脑皮层、左侧小脑、左侧丘脑）。疑似缺血性梗死。
```

The sentences are the `sentence` field of each `record.json`; the impressions are 疑似胶质瘤 / 疑似脑转移瘤 / 疑似缺血性梗死.
"疑似X" states which model ran, not a differential diagnosis (spec §6). The anatomy words come from SynthSeg pseudo-labels
(NOT_EVIDENCE).

## Check against the out-of-fold records (`checks/infer_smoke_consistency.py`, output in `checks/infer_smoke_consistency.txt`)

The five-fold evaluation wrote each case's record from fold 0's validation prediction at the same threshold
(`/data2/congcong/data/FM_data/derived/brain_disease/<disease>/records/<case>.json`, `model_folds` [0]). Same weights,
same threshold, so boxes and scores must agree to 1e-4:

| disease | case | lesions (smoke / record) | largest box difference | largest score difference | host fields differing | relative volume difference | sentences equal |
|---|---|---|---|---|---|---|---|
| glioma | UCSF-PDGM-0008 | 1 / 1 | 0 voxels | 1.0e-4 | 0 | 4.57e-4 (172755 vs 172676 mm³; both print as 173 mL) | yes |
| metastasis | 100101A | 11 / 11 | 0 | 0 | 0 | 0 | yes |
| infarct | sub-strokecase0001 | 26 / 26 | 0 | 0 | 0 | 0 | yes |

All three agree within the tolerance (`ALL AGREE`). The glioma case is the one difference: `nnUNetv2_predict` from the raw
NIfTI and the training run's validation prediction of the same case give the same 3D box, a mean probability different
in the fourth decimal (0.9889 against 0.9890, at the tolerance) and 79 voxels more in the 172755-voxel component
(under 0.05 %); the two other cases are identical to the voxel.

## Deviations

- The entry names the study after its first image file (`UCSF-PDGM-0008_T1.nii.gz`, `100101A_T1pre.nii.gz`,
  `sub-strokecase0001_ses-0001_dwi.nii.gz`); the evaluation's records use the case id. The check matches them by prefix.
  The entry has no `--study` option (the plan's command has none); whether it should carry one is left to the records review.
- The fold 0 model alone is used here (plan), while the default of the entry is the five-fold average, whose behaviour at
  these thresholds was not measured (see the README's known deviations).
