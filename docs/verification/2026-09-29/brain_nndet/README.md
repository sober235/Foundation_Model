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

## Records in this folder

| file | what it is | made by |
|---|---|---|
| `task_build.txt` | build log of Task903: 253 cases, 1297 instances, the one box changed by overwrite | `scripts/nndet_prepare.py --stage task` (plan Task 6 Step 2) |
| `plan.txt` | the resolved plan `D3V001_3d` (target spacing 5 / 0.6875 / 0.6875 mm, patch 12 × 256 × 224, batch 4), the `preprocessed` listing, the splits line (validation cases per fold 51, 51, 51, 50, 50) | pickle print of `preprocessed/D3V001_3d.pkl`, `ls`, `scripts/nndet_prepare.py --stage splits` (Task 6 Steps 3–4) |
| `gt_check/` | ground truth pushed through the runner's coordinate path and scored as if predicted: pass | `scripts/nndet_runner.py gt` → `scripts/eval_brain_nndet.py --gt-check` (Task 6 Step 5) |
| `launch.md` | GPU snapshot, launch command, rate after 20 min, wall-clock projection | controller notes (Task 7 Steps 6–7) |
| `training.txt` | end-of-training confirmation: training-dir listing, 51 state files, head and tail of `train.log`, tail of the launcher log | plan Task 9 Step 1 command |
| `fold0/` | `REPORT.md`, `froc.csv`, `output.txt`, `rule_a.json` | `scripts/eval_brain_nndet.py --dets fold0_default.json --swept fold0_swept.json --folds 0` (Task 9 Step 3); inputs from `scripts/nndet_runner.py extract` (Step 2): default 25881 boxes, swept 1696 boxes, 51 cases each |
| `infer_smoke.md` | the inference entry on one unseen h5 | `scripts/infer_brain_lesions_nndet.py` (Task 9 Step 4) |
| `README.md` | this index | Task 9 Step 5 |

Outside the repository: the task `/data2/congcong/data/FM_data/derived/nndet/Task903_FastMRIBrainSmallLesion/`
(`dataset.json`, `instances.json`, `build_report.json`, `raw_splitted/`, `raw_cropped/`, `preprocessed/`), the training
directory `/data2/congcong/data/FM_data/derived/nndet_models/Task903_FastMRIBrainSmallLesion/RetinaUNetV001_D3V001_3d/fold0/`,
the runner outputs `/data2/congcong/data/FM_data/derived/nndet_runs/{gt.json, fold0_default.json, fold0_swept.json, infer_smoke/}`,
and the logs `logs/brain_nndet/fold0.log` (launcher) and `logs/nndet_install/*.log` (install, preprocessing, unpack) in
the worktree, untracked.

## Known deviations

1. **One box changed by overwrite.** Instances are painted into one label map per case; where a later instance overwrote
   voxels of an earlier one, the earlier lesion's box in the task differs from its fastMRI+ box. This happened to one
   lesion: `file_brain_AXFLAIR_200_6002658`, instance 815 (`task_build.txt` last line; `build_report.json`
   `changed_boxes`).
2. **Eight lesions lost in resampling.** `gt_check/gt_check.json`: the ground truth pushed through the runner's own
   coordinate path (crop, resampling to the target spacing, and back) returns 1289 of 1297 lesions; 8 are lost
   (ids 1001, 1004, 1011, 1042, 1061, 1063, 1080, 1104). For the 1289 that return, IoU with the original box: minimum
   0.3156, median 1.0, 1151 at or above 0.99, none below 0.1 (`below_iou` empty); all 1289 are hits at threshold 0.05
   with 0 false positives. The eight lost lesions can never be hit by this arm; they stay in every denominator.
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

## Timing (from the logs)

| stage | start | end | wall clock | source |
|---|---|---|---|---|
| task build (`nndet_prepare.py --stage task`) | not timestamped | — | — | `task_build.txt` |
| `nndet_prep 903` (preprocessing, CPU, `-np 4 -npp 4`) | 2026-09-29 03:55:35 | 03:59:21 | 3 min 46 s | `logs/nndet_install/04_prep903.log` first and last timestamps |
| `nndet_unpack` | 04:00:19 | not logged (one line) | — | `logs/nndet_install/05_unpack903.log` |
| training, 60 epochs (50 + 10 SWA) on GPU 3 | 04:24:50 | 19:39:01 (`model_last.ckpt` written) | 15 h 14 min | `train.log` first line; file time of `model_last.ckpt` |
| prediction of the 51 validation cases, default settings | 19:39:02 | 19:55:59 | 16 min 57 s | `train.log` "Predict cases with default settings..." → "Start parameter sweep..." |
| postprocessing sweep and analysis | 19:55:59 | 20:00:12 | 4 min 13 s | `train.log` "Start parameter sweep..." → last analysis line |
| launch → last write of the launcher log | 04:24:50 | 20:00:14 | 15 h 35 min | `train.log`; file time of `logs/brain_nndet/fold0.log` |
| extraction, report, inference smoke (2026-09-30) | 09:55 | 09:57:32 | under 3 min | scratch logs of Task 9 Steps 2–4 (not in the repo) |
