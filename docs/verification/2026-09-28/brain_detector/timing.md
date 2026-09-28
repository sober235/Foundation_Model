# Brain detector training timing probe (D8)

GPU used: 5 (idle among the controller-directed candidates 5, 6, 7; GPUs 0-4 were busy with
other processes at probe time and were not touched). Re-checked with `nvidia-smi` immediately
before each launch.

Command:

```
source scripts/nnunet_env.sh
PYTHONNOUSERSITE=1 PYTHONPATH=. python scripts/brain_detector_train.py --config 2d --folds 0 --gpus 5 --trainer nnUNetTrainer_5epochs
PYTHONNOUSERSITE=1 PYTHONPATH=. python scripts/brain_detector_train.py --config 3d_fullres --folds 0 --gpus 5 --trainer nnUNetTrainer_5epochs
```

## 2d, fold 0, 5 epochs

Log: `$nnUNet_results/Dataset903_FastMRIBrainSmallLesion/nnUNetTrainer_5epochs__nnUNetPlans__2d/fold_0/training_log_2026_9_28_13_09_40.txt`
(mirrored to `logs/brain_detector/2d_fold0.log`).

Raw `Epoch time` lines:

```
2026-09-28 13:10:24.950426: Epoch time: 42.14 s
2026-09-28 13:11:05.447946: Epoch time: 39.39 s
2026-09-28 13:11:49.056612: Epoch time: 40.66 s
2026-09-28 13:12:32.381467: Epoch time: 40.43 s
2026-09-28 13:13:15.807297: Epoch time: 40.88 s
```

Mean epoch time = (42.14 + 39.39 + 40.66 + 40.43 + 40.88) / 5 = **40.70 s**

Projection: `250 * 40.70 * 2` (four folds in parallel, then fold 4 after) = 20350 s = **5.65 h**

## 3d_fullres, fold 0, 5 epochs

Log: `$nnUNet_results/Dataset903_FastMRIBrainSmallLesion/nnUNetTrainer_5epochs__nnUNetPlans__3d_fullres/fold_0/training_log_2026_9_28_13_14_43.txt`
(mirrored to `logs/brain_detector/3d_fullres_fold0.log`).

Raw `Epoch time` lines:

```
2026-09-28 13:15:32.630167: Epoch time: 47.18 s
2026-09-28 13:16:19.093577: Epoch time: 45.25 s
2026-09-28 13:17:07.786980: Epoch time: 45.73 s
2026-09-28 13:17:55.987105: Epoch time: 44.96 s
2026-09-28 13:18:44.388572: Epoch time: 45.47 s
```

Mean epoch time = (47.18 + 45.25 + 45.73 + 44.96 + 45.47) / 5 = **45.718 s**

Projection: `250 * 45.718 * 2` = 22859 s = **6.35 h**

## Total projected wall clock

2d (5.65 h) + 3d_fullres (6.35 h) = **12.00 h**, under the 24 h stop threshold (D7/D8).
The 250-epoch training (Task 7) is not launched here; the controller decides when to launch it.

The probe's own outputs (`nnUNetTrainer_5epochs__...` under `$nnUNet_results/Dataset903_FastMRIBrainSmallLesion/`)
are left in place, as instructed.
