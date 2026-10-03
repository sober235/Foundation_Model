# S4 trainings: launch record (plan Task 8, Steps 7–9; spec A9, A10, A15)

## Outline model (Dataset908, 2d, fold 0) — launched first, on the only idle card

GPU snapshot right before the launch (2026-10-03 22:22:37):

```
nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader
0, 44539 MiB, 97 %
1, 44539 MiB, 91 %
2, 44539 MiB, 97 %
3, 44539 MiB, 98 %
4, 44539 MiB, 89 %
5, 44539 MiB, 98 %
6, 44539 MiB, 91 %
7, 14 MiB, 0 %
```

Cards 0–6 are in use by other sessions (44.5 GB each); only GPU 7 was idle. The outline dataset was ready before the
simulated student dataset, so the outline model took the idle card first; the student starts when a card is idle again
(see below).

```
PYTHONNOUSERSITE=1 PYTHONPATH=. ~/anaconda3/envs/nvgen/bin/python scripts/brain_anatomy_train.py --jobs outline
launched outline on GPU 7 (pid 894550); log /data0/congcong/code/Project_Doing/foundation_model/logs/brain_anatomy/Dataset908_FastMRIBrainOutline_2d_nnUNetTrainer_250epochs_fold0.log; results /data2/congcong/data/FM_data/derived/nnunet/results/Dataset908_FastMRIBrainOutline/nnUNetTrainer_250epochs__nnUNetPlans__2d/fold_0
```

Rate after the first epochs (2026-10-03 22:38:09, from `training_log_*.txt` in the result folder): about 40 s per epoch
(pseudo Dice 0.97 after 21 epochs) → 250 epochs ≈ 2.8 h, expected end around 01:15 on 2026-10-04. Under 24 h: no question to the user.

## Student model (Dataset907, 3d_fullres, fold 0)

GPU snapshot right before the launch (2026-10-03 22:44:23); cards 4, 5 and 6 had become idle:

```
0, 24875 MiB, 100 %
1, 24773 MiB, 100 %
2, 25255 MiB, 99 %
3, 26239 MiB, 100 %
4, 14 MiB, 0 %
5, 14 MiB, 0 %
6, 14 MiB, 0 %
7, 7433 MiB, 62 %
```

```
PYTHONNOUSERSITE=1 PYTHONPATH=. ~/anaconda3/envs/nvgen/bin/python scripts/brain_anatomy_train.py --jobs student
launched student on GPU 4 (pid 1036131); log /data0/congcong/code/Project_Doing/foundation_model/logs/brain_anatomy/Dataset907_BrainAnatomyFLAIR_3d_fullres_nnUNetTrainer_250epochs_fold0.log; results /data2/congcong/data/FM_data/derived/nnunet/results/Dataset907_BrainAnatomyFLAIR/nnUNetTrainer_250epochs__nnUNetPlans__3d_fullres/fold_0
```

Rate after the first epochs (2026-10-03 22:48, from `training_log_2026_10_3_22_44_28.txt` in the result folder): epochs 0–3
took 49.0, 47.2, 47.1 and 48.4 s (3268 training and 872 validation samples in fold 0) → 250 epochs ≈ 3.3 h, expected end
around 02:05 on 2026-10-04. Under 24 h: no question to the user. The outline model was at epoch 34 at the same moment,
42 s per epoch, pseudo Dice 0.973.

## The processes to watch

The pid the script prints is the `setsid` wrapper, which exits at once (the process is already a session leader, so
`setsid` forks). The processes that stay are one above it:

| job | printed pid | shell (`bash -c …`) | trainer (`nnUNetv2_train`, holds the GPU) | GPU |
|---|---|---|---|---|
| outline | 894550 | 894551 | 894553 | 7 |
| student | 1036131 | 1036132 | 1036133 | 4 |

`ps -p 894553,1036133` tells whether the trainings are alive; `checkpoint_final.pth` in the two `fold_0` folders tells
that they ended normally.
