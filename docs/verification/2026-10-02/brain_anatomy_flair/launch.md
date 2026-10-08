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
42 s per epoch, pseudo Dice 0.9742.

## The processes to watch

The pid the script prints is the `setsid` wrapper, which exits at once (the process is already a session leader, so
`setsid` forks). The processes that stay are one above it:

| job | printed pid | shell (`bash -c …`) | trainer (`nnUNetv2_train`, holds the GPU) | GPU |
|---|---|---|---|---|
| outline | 894550 | 894551 | 894553 | 7 |
| student | 1036131 | 1036132 | 1036133 | 4 |

`ps -p 894553,1036133` tells whether the trainings are alive; `checkpoint_final.pth` in the two `fold_0` folders tells
that they ended normally.

## The first student run cannot learn the sides: mirroring (found 22:50, spec A17)

The student's classes have a side (white matter left / right, …). `nnUNetTrainer_250epochs` mirrors the image and the
labels together during training (and averages over mirrored copies at test time), so a left structure appears on either
side under the same label. Neither the spec nor the plan had a word on it. What showed it:

- `debug.json` of the first run: `"inference_allowed_mirroring_axes": "(0, 1, 2)"`.
- Its pseudo Dice (`training_log_2026_10_3_22_44_28.txt`; order: white matter L, R, cortex L, R, thalamus L, R, basal
  ganglia L, R, …, ventricles last). The two sides swing against each other, the ventricles (no side) are fine:

```
epoch 5  0.2379 0.4916 0.1358 0.4530 0.0    0.0    0.0164 0.0    …  0.7764
epoch 6  0.1761 0.4977 0.0935 0.4503 0.1523 0.0157 0.0864 0.0244 …  0.7371
epoch 8  0.5239 0.0617 0.4892 0.0135 0.3682 0.1710 0.1339 0.0760 …  0.7628
epoch 9  0.4609 0.2331 0.3012 0.3213 0.3953 0.0897 0.0225 0.2513 …  0.7724
```

- The inference chain run once with `checkpoint_best.pth` of both models on `file_brain_AXFLAIR_201_6002917.h5` (22:49,
  a probe, not a result; output `/data2/congcong/data/FM_data/derived/brain_anatomy/preflight_best_20261003_2249/svd`):
  white matter left 19.2 mL, right 268.9 mL; cortex left 0.1 mL, right 255.5 mL.

Decision (spec A17): the student is trained and run with nnU-Net's own `nnUNetTrainer_250epochs_NoMirroring` (no
mirroring in training, none at test time; everything else as before). The outline has no side and keeps its trainer;
its run is not touched. Code: commits dee2213 (spec, plan) and d96fca9 (launcher, inference chain, tests).

The first run was **not stopped**: the session's permission layer refused the signal, and it was not tried again. It
runs to its end on GPU 4 (about 02:05 on 2026-10-04) and its result is not used. To stop it by hand:
`kill -TERM -- -1036132`. Its result folder (`nnUNetTrainer_250epochs__nnUNetPlans__3d_fullres`) and log stay.

## Student model, second launch: without mirroring

GPU snapshot right before the launch (2026-10-03 22:56:30):

```
0, 26155 MiB, 96 %
1, 26053 MiB, 90 %
2, 25255 MiB, 100 %
3, 26239 MiB, 92 %
4, 8245 MiB, 72 %      <- the first student run
5, 14 MiB, 0 %
6, 14 MiB, 0 %
7, 7433 MiB, 65 %      <- the outline model
```

```
PYTHONNOUSERSITE=1 PYTHONPATH=. ~/anaconda3/envs/nvgen/bin/python scripts/brain_anatomy_train.py --jobs student
launched student on GPU 5 (pid 1078277); log /data0/congcong/code/Project_Doing/foundation_model/logs/brain_anatomy/Dataset907_BrainAnatomyFLAIR_3d_fullres_nnUNetTrainer_250epochs_NoMirroring_fold0.log; results /data2/congcong/data/FM_data/derived/nnunet/results/Dataset907_BrainAnatomyFLAIR/nnUNetTrainer_250epochs_NoMirroring__nnUNetPlans__3d_fullres/fold_0
```

Processes that stay: shell 1078278, trainer 1078279 (GPU 5). `debug.json`: `"inference_allowed_mirroring_axes": "None"`.

Pseudo Dice of the first epochs (`training_log_*.txt` in the new result folder), same order as above — both sides rise
together:

```
epoch 1  0.7495 0.7547 0.6140 0.6410 0.0    0.0    0.0    0.0    …  0.0
epoch 4  0.7939 0.7863 0.7305 0.7406 0.0009 0.0028 0.1486 0.1735 …  0.7855
epoch 6  0.7892 0.7975 0.7330 0.7314 0.7311 0.7020 0.6671 0.6168 …  0.8020
epoch 7  0.7953 0.8014 0.7378 0.7439 0.7681 0.7667 0.7007 0.6676 …  0.8222
```

Rate (three trainings share the CPU now): epochs 0–7 took 49.0–52.4 s → 250 epochs ≈ 3.5 h, expected end around 02:25 on
2026-10-04. Under 24 h: no question to the user.
