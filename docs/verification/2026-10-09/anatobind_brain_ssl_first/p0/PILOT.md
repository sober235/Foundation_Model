# Stage I pilot（T08 / Q19；2026-10-09）

目的：跑通两视图 MIM + 患者安全 InfoNCE 的真实数据训练，量显存与吞吐，看验证指标与恢复。只跑 mask 0.60 一组（Q19），8k 曝光。

## 运行

- 运行目录 `/data2/congcong/data/FM_data/derived/aur/ssl_runs/pilot_8k_mb12/`，日志 `…/pilot_8k_mb12.log`（不入库）。
- 卡：5、6、7（启动脚本等 150 s 内 7 号卡空出；0、1、3 被本机另一会话的 MC-GS 占着），world 3。
- 配置：microbatch 12 × 3 卡 = 全局 36 源裁块/步（两视图各 36 次曝光），不累积，每 rank 8 个 loader worker，lr 4.5e-4（= 3e-4 × sqrt(36/16)，PROPOSED），预热 50 步，验证与 resume checkpoint 每 50 步，验证 32 卷 × 2 裁块 = 64 裁块，mask 0.60、块 16–32 mm、τ 0.2、λ_c 0.1，bf16，无梯度检查点，`NCCL_P2P_DISABLE=1`。
- 启动脚本（scratchpad，内容等价于下面的命令）：

```bash
cd /data0/congcong/code/Project_Doing/foundation_model
export PYTHONNOUSERSITE=1 PYTHONPATH=. NCCL_P2P_DISABLE=1 TORCH_NCCL_ASYNC_ERROR_HANDLING=1 OMP_NUM_THREADS=8 CUDA_VISIBLE_DEVICES=5,6,7
~/anaconda3/envs/nvgen/bin/torchrun --standalone --nproc_per_node=3 scripts/aur_ssl_train.py \
  --samples /data2/congcong/data/FM_data/derived/aur/ssl_manifest_v1/samples_ssl.json \
  --out /data2/congcong/data/FM_data/derived/aur/ssl_runs/pilot_8k_mb12 --seen-crops 8000 --microbatch 12 --grad-accum 1 --workers 8 \
  --lr 4.50e-04 --warmup-steps 50 --val-every 50 --save-every 50 --log-every 10 --val-volumes 32 --mask-ratio 0.60 --contrast-weight 0.10
```

## 结果（`summary.json`、`val.jsonl`、`log_rank0.jsonl` 原文）

```
SUMMARY {"steps": 223, "seen_crops": 8028, "world": 3, "global_batch": 36, "step_s_mean": 2.793, "step_s_p95": 3.322,
         "data_wait_s_mean": 0.0145, "crops_per_s": 12.89, "best": {"mim": 0.06327, "step": 223, ...},
         "peak_alloc_gib": 50.89, "peak_reserved_gib": 58.40, "gpu": "NVIDIA A100 80GB PCIe"}   # rank 0 = 5 号卡（A100）
VAL step 150: mim 0.06413  baseline 0.36114  view_cosine 0.98465  effective_rank_z 37.24  contrast_acc 1.0  val_crops 64
VAL step 200: mim 0.06328  baseline 0.36114  view_cosine 0.98429  effective_rank_z 43.43
VAL step 223: mim 0.06327  baseline 0.36114  view_cosine 0.98665  effective_rank_z 43.86
train step 200: loss 0.1282  mim 0.0498  contrast 0.7846  contrast_acc 1.0  hidden_share 0.6024
```

读法：

- 吞吐 12.9 crops/s（3 卡），步时均值 2.79 s、p95 3.32 s，数据等待均值 0.015 s：8 个 worker 后 loader 不再拖后腿（探针里 4 个 worker 在 mb 12 时最大等 5 s）。按此外推 320k 曝光 3 卡约 6.9 h、4 卡约 5.2 h（NOT_MEASURED）。
- 显存峰值分配 50.9 / 预留 58.4 GiB，距 80 GB 卡顶有 21 GB 余量；验证批没有把峰值推高。
- 验证 masked Huber 0.063，远低于"不学习"的三线性插值基线 0.361（同一批裁块、同一遮挡）；训练 mim 在 0.05–0.07。
- 两视图投影余弦 0.985、对比准确率 1.0：64 个验证裁块里正对总是最近的，说明任务对当前批尺寸偏容易（主训批更大、负样本更多）；有效秩 37 → 44（上限 64 = 验证裁块数），没有塌缩迹象。8k 曝光只能看"有没有在学"，不能看收敛。
- 遮挡份额 0.60 与 mask 比率一致。
- 全程损失有限，无 NaN，3 卡 DDP 223 步不挂，每 50 步的 resume 与 best 导出都写出。

## G0.5 对照

| 项 | 结果 |
|---|---|
| 泄漏测试全过 | `tests/test_aur_ssl_masking.py` 等在全量 949 passed 里 |
| MIM + InfoNCE 有限有梯度 | 本次 223 步全程有限，loss 从 0.33 降到 0.12–0.15（train）；验证 Huber 0.064 → 0.063 |
| DDP 100 步不挂 | 四卡探针 100 步（`PROBES.md`）+ 本次三卡 223 步 |
| 恢复一致 | 见下 `resume150` 段 |

## 恢复一致（`pilot_8k_mb12_resume150_r2/`）

从 `pilot_8k_mb12/resume_step150.pt` 以同配置、同三张卡恢复（命令同上，加 `--resume …/resume_step150.pt --out …/pilot_8k_mb12_resume150_r2`）。第一次尝试 `pilot_8k_mb12_resume150/` 在启动时被一个 35 GB 的外来进程挤在 5 号卡上 OOM，无日志，目录保留。

原运行 vs 恢复运行逐记录步（`log_rank0.jsonl`）：

```
step  seen(orig, resumed)   lr(orig, resumed)           loss(orig, resumed)   mim(orig, resumed)
160   5760  5760            1.3187e-04  1.3187e-04      0.1498  0.1775        0.0666  0.0768
170   6120  6120            9.6411e-05  9.6411e-05      0.1314  0.1495        0.0532  0.0627
180   6480  6480            6.5180e-05  6.5180e-05      0.1515  0.1402        0.0678  0.0625
190   6840  6840            3.9206e-05  3.9206e-05      0.1436  0.1484        0.0706  0.0555
200   7200  7200            1.9342e-05  1.9342e-05      0.1282  0.1267        0.0498  0.0512
210   7560  7560            6.2406e-06  6.2406e-06      0.1184  0.1181        0.0502  0.0506
220   7920  7920            3.3381e-07  3.3381e-07      0.1248  0.1415        0.0582  0.0622
val   orig    step 200 mim 0.06328 rank 43.43 | step 223 mim 0.06327 rank 43.86
val   resumed step 200 mim 0.06341 rank 43.52 | step 223 mim 0.06341 rank 43.81
SUMMARY resumed: steps 223, seen_crops 8028, world 3, global_batch 36, final_export …/pilot_8k_mb12_resume150_r2/ssl_stage1_best.pt
```

读法：step、seen、学习率逐位相同，终点相同，验证 Huber 相差 1.4e-4；训练损失逐步不同是因为恢复后 loader 从新的 epoch 重新按权重抽样、遮挡随机数也按 (seed, step, rank) 重生成，不保存数据游标。**恢复一致：过。**

## G0.5 判定

四项全过（泄漏测试、损失有限有梯度、DDP 多步不挂、恢复一致）。主训按 `PROBES.md` 末段的配置起（microbatch 12、每 rank 8 worker、全局 12 × 卡数、lr 平方根缩放、预热 16000 / 全局 步），运行目录 `…/ssl_runs/stage1_320k_mb12/`。
