# Brain disease trainings: launch of the queue (plan Task 3, Steps 6–7; spec M4, M13, M14)

Launched by the controller after the task reviews of Task 2 (datasets on real data) and Task 3 (queue) approved.
Fifteen trainings: glioma, metastasis and infarct, folds 0–4, nnU-Net `3d_fullres`, `nnUNetTrainer_250epochs`.

## GPU snapshot just before launch (2026-09-29 13:46:11)

```
nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader
0, 14 MiB, 0 %
1, 14 MiB, 0 %
2, 14 MiB, 0 %
3, 18187 MiB, 45 %
4, 14 MiB, 0 %
5, 14 MiB, 0 %
6, 14 MiB, 0 %
7, 14 MiB, 0 %
```

Seven cards were idle (0, 1, 2, 4, 5, 6, 7). GPU 3 runs the nnDetection fold 0 of the second arm (pid 3334388, since
04:24). The queue starts at most six trainings at once (M14), so GPU 7 stayed free; another user took it between
13:48 and 13:56 (26 GB, 92 % at 13:56:07).

## Launch

```
bash -c 'source scripts/nnunet_env.sh && PYTHONPATH=. setsid nohup python scripts/gpu_queue.py --diseases glioma metastasis infarct --folds 0 1 2 3 4 > logs/brain_disease/queue.log 2>&1 < /dev/null &'
2026-09-29 13:46:13 queue of 15 jobs: [('glioma', 0), ('metastasis', 0), ('infarct', 0), ('glioma', 1), ('metastasis', 1), ('infarct', 1), ('glioma', 2), ('metastasis', 2), ('infarct', 2), ('glioma', 3), ('metastasis', 3), ('infarct', 3), ('glioma', 4), ('metastasis', 4), ('infarct', 4)]
2026-09-29 13:46:14 launched ('glioma', 0) on GPU 0 (pid 1091662)
2026-09-29 13:46:14 launched ('metastasis', 0) on GPU 1 (pid 1091663)
2026-09-29 13:46:14 launched ('infarct', 0) on GPU 2 (pid 1091665)
2026-09-29 13:46:14 launched ('glioma', 1) on GPU 4 (pid 1091666)
2026-09-29 13:46:14 launched ('metastasis', 1) on GPU 5 (pid 1091669)
2026-09-29 13:46:14 launched ('infarct', 1) on GPU 6 (pid 1091670)
```

Queue process pid 1091434 (its `bash -c` wrapper 1091432). Each job is a `bash -c` group whose child is
`nnUNetv2_train <id> 3d_fullres <fold> -tr nnUNetTrainer_250epochs --npz` under `nice -n 19` with
`nnUNet_n_proc_DA=6` and `CUDA_VISIBLE_DEVICES=<card>`. The dry run of the queue at 13:39:32 listed the same six
jobs on the same cards.

## CPU (13:48:19, 10 s window of `top`)

nnU-Net: 60 processes, 44.3 cores; nnDetection: 15 processes, 2.8 cores; together 47.2 cores (limit 48). Each training
holds its main process, six augmentation workers for training and three for validation; the validation workers idle
most of the time.

## Epoch times (2026-09-29 14:02:10, 16 min after launch)

Read from nnU-Net's own `training_log_*.txt` in each fold folder, not from `logs/brain_disease/Dataset90*_fold*.log`:
the standard output of a training is block-buffered when it goes to a file, so those logs lag by many minutes (at
13:56 they still ended at `using pin_memory on device 0` while the trainings had finished 5 to 12 epochs). This is a
deviation from the command in plan Task 3 Step 7; the numbers are the same lines, read where they arrive first.

```
for d in /data2/congcong/data/FM_data/derived/nnunet/results/Dataset90[456]_*/nnUNetTrainer_250epochs__nnUNetPlans__3d_fullres/fold_*; do l=$(ls -t "$d"/training_log_*.txt | head -1); n=$(echo "$d" | sed -E 's#.*/(Dataset90[0-9])_([A-Za-z]+)/.*/(fold_[0-9])#\1_\2 \3#'); grep 'Epoch time' "$l" | sed -E 's/.*Epoch time: ([0-9.]+) s.*/\1/' | awk -v n="$n" '{s+=$1; k++; if (k>1) {s2+=$1; k2++}; if (min==""||$1<min) min=$1; if ($1>max) max=$1} END {printf "%s: %d epochs, min %.1f max %.1f mean %.1f s (mean without the first %.1f s)\n", n, k, min, max, s/k, s2/k2}'; done
Dataset904_PDGMGlioma fold_0: 9 epochs, min 84.9 max 97.9 mean 93.1 s (mean without the first 93.4 s)
Dataset904_PDGMGlioma fold_1: 9 epochs, min 91.7 max 106.1 mean 99.0 s (mean without the first 99.6 s)
Dataset905_BMSRMetastasis fold_0: 11 epochs, min 73.0 max 82.4 mean 77.8 s (mean without the first 77.6 s)
Dataset905_BMSRMetastasis fold_1: 10 epochs, min 76.5 max 87.8 mean 83.7 s (mean without the first 83.4 s)
Dataset906_ISLESInfarct fold_0: 19 epochs, min 45.7 max 50.8 mean 46.4 s (mean without the first 46.2 s)
Dataset906_ISLESInfarct fold_1: 19 epochs, min 44.2 max 49.7 mean 45.4 s (mean without the first 45.2 s)
```

GPU use at the same time: 8.0–8.6 GB per training, utilisation between 0 % and 75 % from one second to the next; the
trainings wait for augmented batches part of the time (CPU-bound at six workers each).

## Projection

| disease | epoch (mean of the two folds) | 250 epochs | with the final validation (estimate) |
|---|---|---|---|
| glioma (Dataset904) | 93–99 s | 6.5–6.9 h | about 7.2 h |
| metastasis (Dataset905) | 78–84 s | 5.4–5.8 h | about 6.1 h |
| infarct (Dataset906) | 45–46 s | 3.2 h | about 3.4 h |

- The final validation predicts the fold's validation cases with `--npz` (99–101, 77–106 and 50 cases per fold); its duration is
  an estimate here (0.2–0.4 h) and is measured when the first fold ends.
- The epochs are longer than in the plan's dry run (67–73 s, 62–67 s, 23 s). For infarct the real plan uses batch 8
  where the mini dataset used batch 2; for all three the likely reason is that six trainings share the CPU budget
  (not measured separately).
- **Every job is far below the 24 h limit: no stop, no question to the user.**
- All fifteen: about 5 × (7.2 + 6.1 + 3.4) ≈ 84 GPU hours. With six cards that is about 14 h of wall clock, so the
  last fold would end in the morning of 2026-09-30 if idle cards stay available; the queue only ever takes idle cards,
  so other users' jobs can stretch this.
- First results: infarct folds 0 and 1 at about 17:10, metastasis folds 0 and 1 at about 19:50, glioma folds 0 and 1
  at about 21:00 on 2026-09-29. The fold 0 early reading of each disease (plan Task 11) follows its fold 0.

## Control files

`logs/brain_disease/skip_<dataset id>` stops new folds of one disease (M4, early stop); `logs/brain_disease/stop`
stops the queue from starting anything. Neither exists now. Running trainings are never signalled by the queue.
