# nnDetection fold 0 launch (plan Task 7, Steps 6–7; spec N9, N14)

Launched by the controller after the Task 6 (data) and Task 7 (launcher) reviews approved.

## GPU snapshot just before launch (2026-09-29 04:24:44)

```
nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader
0, 14 MiB, 0 %
1, 14 MiB, 0 %
2, 14 MiB, 0 %
3, 14 MiB, 0 %
4, 14 MiB, 0 %
5, 14 MiB, 0 %
6, 14 MiB, 0 %
7, 14 MiB, 0 %
```

## Launch

```
PYTHONNOUSERSITE=1 PYTHONPATH=. ~/anaconda3/envs/nvgen/bin/python scripts/nndet_train.py --fold 0 --gpus 3 2 0 4
launched fold 0 on GPU 3 (pid 3334387); log /data0/congcong/code/Project_Doing/foundation_model-nndet/logs/brain_nndet/fold0.log; training dir /data2/congcong/data/FM_data/derived/nndet_models/Task903_FastMRIBrainSmallLesion/RetinaUNetV001_D3V001_3d/fold0
```

pid 3334387 is the `setsid bash -c` wrapper; the training process is its child, pid 3334388 (`nndet_train 903 -o exp.fold=0 --sweep`). Log start: `Loading network patch size [ 12 256 224] and generator patch size [12, 365, 365]`, `CUDA_VISIBLE_DEVICES: [3]`.

## Rate (2026-09-29 04:45:05, 20 min 20 s after launch)

```
tail -c 4000 logs/brain_nndet/fold0.log | tr '\r' '\n' | grep -oE '[0-9]+/[0-9]+ \[[0-9:]+<[0-9:]+, *[0-9.]+(it/s|s/it)[^]]*\]' | tail -4
1733/2600 [07:56<03:58,  3.64it/s, loss=0.58, v_num=f027]
1734/2600 [07:56<03:58,  3.64it/s, loss=0.58, v_num=f027]
1734/2600 [07:56<03:58,  3.64it/s, loss=0.581, v_num=f027]
1735/2600 [07:57<03:57,  3.64it/s, loss=0.581, v_num=f027]
Epoch 1:  67%|██████▋   | 1736/2600 [07:57<03:57,  3.64it/s, loss=0.578, v_num=f027]
GPU 3: 18185 MiB, 57 %
```

## Projection

- One epoch = 2600 steps (2500 training batches + 100 validation batches) at 3.64 it/s = 714 s ≈ 11.9 min (the bar itself shows 07:57 elapsed + 03:57 remaining = 11.9 min).
- Training: 60 epochs (50 + 10 SWA) × 11.9 min ≈ 714 min ≈ 11.9 h.
- Sweep: the toy sweep took 60.6 s for 2 validation cases (Task 1: `Predict cases with default settings..` at 02:40:37.001 to the end of `train.log` at 02:41:37.558); scaled to 51 cases ≈ 26 min. The real volumes are larger than the toy cases, so 1 h is allowed for it.
- **Total ≈ 13 h, below the 24 h limit (N9): no stop, no question to the user.** Expected end around 17:30 on 2026-09-29.
- Known from Task 1: after the sweep and analysis, `nndet_train` may print batchgenerators' teardown `RuntimeError` and hang at exit; Task 9 Step 1 handles that.
