# nnDetection second arm on Task903, fold 0: record index

Spec `docs/superpowers/specs/2026-09-28-brain-nndet-design.md`, plan `docs/superpowers/plans/2026-09-28-brain-nndet.md`.
The arm asks whether nnDetection (Retina U-Net, `RetinaUNetV001_D3V001_3d`) finds more of S2's fastMRI+ small lesions
than S2's nnU-Net 2d detector on the same fold-0 validation cases (rule A, spec N3). Every number below is copied from
the files named beside it.

## Outcome

Rule A fails on fold 0 (`fold0/rule_a.json`): nnDetection with default postprocessing hits 17 of 280 lesions at its
operating threshold 0.50 (sensitivity 0.0607, 0.7255 false positives per volume); nnU-Net 2d on the same 51 volumes hits
92 at 0.60 (0.3286, 1.5490); required for a pass: 106 = 92 + 0.05 × 280. Per S2's D9 the arm stops here; no further fold
is trained. The swept postprocessing (parameters tuned on these same validation cases) is reported as NOT_GATE:
68 hits, 0.2429 at 1.8431 false positives per volume, threshold 0.85 (`fold0/REPORT.md`). Neither version reaches the
D1 gate (sensitivity 0.5 at ≤ 2 false positives per volume), which fold 0 alone could not decide anyway.

The 0.05 threshold grid (spec §7, as in S2) understates what nnDetection can do within the budget: the FROC has 60 hits at
0.45 (2.549 false positives per volume, over budget) and 17 at 0.50; 136 of the 25 881 boxes score in [0.45, 0.50), 35
score exactly 0.50 and none in (0.50, 0.55). On a 0.005 grid the best row within budget is 40 hits at 0.495 (1.765 per
volume). The decision does not change (40 is far from 106); the grid is kept for comparability with S2 (controller's
recount with the repository's `sweep` on a 0.005 grid, 2026-09-30; not a record).

## Records in this folder

| file | what it is | made by |
|---|---|---|
| `task_build.txt` | build log of Task903: 253 cases, 1297 instances, the one box changed by overwrite | `scripts/nndet_prepare.py --stage task` (plan Task 6 Step 2) |
| `plan.txt` | the resolved plan `D3V001_3d` (target spacing 5 / 0.6875 / 0.6875 mm, patch 12 × 256 × 224, batch 4), the `preprocessed` listing, the splits line (validation cases per fold 51, 51, 51, 50, 50) | pickle print of `preprocessed/D3V001_3d.pkl`, `ls`, `scripts/nndet_prepare.py --stage splits` (Task 6 Steps 3–4) |
| `gt_check/` | ground truth pushed through the runner's coordinate path and scored as if predicted: pass (criterion tightened afterwards, known deviation 7) | `scripts/nndet_runner.py gt` → `scripts/eval_brain_nndet.py --gt-check` (Task 6 Step 5) |
| `launch.md` | GPU snapshot, launch command, rate after 20 min, wall-clock projection | controller notes (Task 7 Steps 6–7) |
| `training.txt` | end-of-training confirmation: training-dir listing, 51 state files, head and tail of `train.log`, tail of the launcher log | plan Task 9 Step 1 command |
| `fold0/` | `REPORT.md`, `froc.csv`, `output.txt`, `rule_a.json` | `scripts/eval_brain_nndet.py --dets fold0_default.json --swept fold0_swept.json --folds 0` (Task 9 Step 3); inputs from `scripts/nndet_runner.py extract` (Step 2): default 25881 boxes, swept 1696 boxes, 51 cases each |
| `infer_smoke.md` | the inference entry on one unseen h5 | `scripts/infer_brain_lesions_nndet.py` (Task 9 Step 4) |
| `README.md` | this index | Task 9 Step 5 |

Outside the repository: the task `/data2/congcong/data/FM_data/derived/nndet/Task903_FastMRIBrainSmallLesion/`
(`dataset.json`, `instances.json`, `build_report.json`, `raw_splitted/`, `raw_cropped/`, `preprocessed/`), the training
directory `/data2/congcong/data/FM_data/derived/nndet_models/Task903_FastMRIBrainSmallLesion/RetinaUNetV001_D3V001_3d/fold0/`,
the runner outputs `/data2/congcong/data/FM_data/derived/nndet_runs/{gt.json, fold0_default.json, fold0_swept.json, infer_smoke/}`,
and, in the worktree and untracked, the launcher log `logs/brain_nndet/fold0.log` and
`logs/nndet_install/{02_smoke,03_runner_toy,04_prep903,05_unpack903}.log` (toy smoke, runner toy check, preprocessing,
unpack); the environment install logs are `~/logs/nndet_install/0{1..5}_env.log` (spec §3, `docs/nndet_install.md`).

## Known deviations

1. **One box changed by overwrite.** Instances are painted into one label map per case; where a later instance overwrote
   voxels of an earlier one, the earlier lesion's box in the task differs from its fastMRI+ box. This happened to one
   lesion: `file_brain_AXFLAIR_200_6002658`, instance 815 (`task_build.txt` last line; `build_report.json`
   `changed_boxes`).
2. **Eight lesions lost in resampling.** `gt_check/gt_check.json`: the ground truth pushed through the runner's own
   coordinate path (crop, resampling to the target spacing, and back) returns 1289 of 1297 lesions; 8 are lost
   (ids 1001, 1004, 1011, 1042, 1061, 1063, 1080, 1104). For the 1289 that return, IoU with the original box: minimum
   0.3156, median 1.0, 1151 at or above 0.99, none below 0.1 (`below_iou` empty); all 1289 are hits at threshold 0.05
   with 0 false positives. The eight lesions have no training label after resampling and do not return in the
   ground-truth check, but predictions are not bound by that: at threshold 0.05 the fold 0 predictions match two of them
   (1042 at IoU 0.10 by a box scored 0.12, 1104 at IoU 0.19 by a box scored 0.25; both are in fold 0's 280 lesions),
   at 0.50 neither. They stay in every denominator. All eight are single-slice lesions in volumes with 3 mm slices
   (registry strata `inplane_0.62_slice_3` and `inplane_0.86_slice_3`): the 7 lesion volumes with 3 mm slices hold 43
   lesions, 20 of them single-slice, of which these 8 are lost when resampled to 5 mm.
3. **Unpainted fastMRI+ labels (S2 D10).** Copied from `docs/verification/2026-09-28/brain_detector/README.md`: of the
   2604 fastMRI+ boxes in the 165 lesion volumes, 1785 pass Gate 0.5's filter (two small-lesion labels, side ≥ 3 px) and
   merge into the 1297 3D lesions; the other 819 boxes, in 61 volumes, are not painted. They are background in training,
   and a detection on them counts as a false positive. By label: Nonspecific lesion 152, Craniotomy 108, Dural
   thickening 86, Posttreatment change 85, Encephalomalacia 81, Possible artifact 69, Enlarged ventricles 48, Mass 47,
   Resection cavity 46, Craniectomy with Cranioplasty 31, Edema 31, Normal variant 17, Nonspecific white matter lesion 12,
   Paranasal sinus opacification 5, Lacunar infarct 1.
4. **Thresholds and sweep chosen on the validation cases.** Operating thresholds are picked on fold 0's own validation
   cases, as in S2; the swept postprocessing parameters were tuned by nnDetection on the same cases, which is why the
   swept version is NOT_GATE.
5. **Wall clock above the projection.** `launch.md` projected about 13 h; the run took 15 h 35 min because jobs of other
   sessions shared GPU 3 from the afternoon on (STATUS.md of 2026-09-29). The launcher log ends with batchgenerators'
   teardown `RuntimeError` / `OSError: [Errno 9] Bad file descriptor` after the analysis lines; as in the toy run of
   Task 1 this is printed after the work is done. The process exited on its own; nothing was killed.
6. **Patch larger than the data.** The launcher log warns twice `Found patch size which is bigger than data: data (10, 291,
   291) patch [12 256 224]` and `data (10, 320, 320) …`: volumes with 3 mm slices have 10 slices after resampling to 5 mm and
   are padded to the 12-slice patch. Harmless; it belongs beside the lost lesions of item 2.
7. **The ground-truth check's pass criterion was tightened after the run** (whole-branch review of 2026-09-30): besides no
   IoU under 0.1 and no false positive, at least 80 % of the returning instances must come back with IoU ≥ 0.99
   (`EXACT_SHARE` in `scripts/eval_brain_nndet.py`, test `test_gt_check_fails_when_the_margin_is_not_undone`). A margin
   error keeps every IoU above 0.1 (a 4 × 4 × 1 lesion gives 0.32 with the ±1 expansion not undone, 0.148 undone on the
   wrong ends) but almost none above 0.99, so the old criterion could not fail for the error it guards (spec N8). The
   recorded run satisfies the new criterion: 1151 of 1289 (0.893) at or above 0.99 (`gt_check/gt_check.json`, whose
   `pass: true` was computed under the old criterion).

## Timing (from the logs)

| stage | start | end | wall clock | source |
|---|---|---|---|---|
| task build (`nndet_prepare.py --stage task`) | not timestamped | — | — | `task_build.txt` |
| `nndet_prep 903` (preprocessing, CPU, `-np 4 -npp 4`) | 2026-09-29 03:55:35 | 03:59:21 | 3 min 46 s | `logs/nndet_install/04_prep903.log` first and last timestamps |
| `nndet_unpack` | 04:00:19 | not logged (one line) | — | `logs/nndet_install/05_unpack903.log` |
| training, 60 epochs (50 + 10 SWA) on GPU 3 | 04:24:50 | 19:39:01 (`model_last.ckpt` written) | 15 h 14 min | `train.log` first line; file time of `model_last.ckpt` |
| prediction of the 51 validation cases, default settings | 19:39:02 | 19:55:59 | 16 min 57 s | `train.log` "Predict cases with default settings..." → "Start parameter sweep..." |
| postprocessing sweep and analysis | 19:55:59 | 20:00:12 | 4 min 12 s | `train.log` "Start parameter sweep..." → last analysis line |
| launch → last write of the launcher log | 04:24:50 | 20:00:14 | 15 h 35 min | `train.log`; file time of `logs/brain_nndet/fold0.log` |
| extraction, report, inference smoke (2026-09-30) | 09:55 | 09:57:32 | under 3 min | scratch logs of Task 9 Steps 2–4 (not in the repo) |
