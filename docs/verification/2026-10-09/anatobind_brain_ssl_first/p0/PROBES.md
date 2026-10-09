# Stage I 探针记录（T07；2026-10-09）

每行一个探针运行；目录名 = 配置 + 卡号。数字取自各目录的 `probe.json`（均值、p95、峰值显存）与 `log_rank0.jsonl`（中位数按第 2 步起的记录算）。
所有探针：裁块 128×160×160、两视图、mask 0.60、bf16、不开梯度检查点、`NCCL_P2P_DISABLE=1`、`OMP_NUM_THREADS=8`、每 rank 4 个 loader worker（另注明者除外）。
"全空卡"= 启动时该卡 `nvidia-smi` 显示 1 MiB；"被挤"= 运行中有别的进程进入同一张卡（本机其他会话的 MC-GS `arc/cycle.py` 每个约 45 GB）。

## 有效数字

| 目录 | 卡 | world | microbatch × 累积 | 全局 batch | 步时均值 / p95 / 中位 (s) | 数据等待均值 / 最大 (s) | crops/s | 峰值分配 / 预留 (GiB) |
|---|---|---|---|---|---|---|---|---|
| `ddp_b2x2_g0567_r1` | 0,5,6,7 全空 | 4 | 2 × 2 | 16 | 1.75 / 2.89 / 1.46 | 0.12 / 0.82 | 9.15 | 8.8 / 10.5 |
| `ddp_b4x1_g0567` | 0,5,6,7 全空 | 4 | 4 × 1 | 16 | 1.58 / 2.89 / 1.22 | 0.33 / 1.36 | 10.11 | 17.2 / 20.3 |
| `single_b8_g1` | 1 全空（MC-GS 在结束后才进入） | 1 | 8 × 1 | 8 | 2.64 / 6.30 / 1.57 | 0.50 / — | 3.03 | 34.0 / 40.3 |
| `single_b12_g5` | 5 全空 | 1 | 12 × 1 | 12 | 3.65 / — / 2.06 | — / 5.00 | 3.29 | 50.8 / 60.7 |
| `single_b16_g6` | 6 全空 | 1 | 16 × 1 | 16 | 5.61 / 20.4 / 2.53 | 2.25 / 2.89 | 2.85 | 67.7 / 77.2 |

均值含第一个 epoch 边界（重建 loader）和偶发的数据等待，中位数更接近稳态；100 步探针的均值受首步影响小于 15–50 步探针。

读法：

- 显存随 microbatch 近似线性，约 4.2 GiB 分配 / 5 GiB 预留每源裁块（两视图）。microbatch 16 预留 77 GiB，贴着 80 GB 卡顶，验证批与分配器碎片随时会 OOM；**microbatch 12（预留 61 GiB）是能稳定占满显存的最大档**。
- 单卡 crops/s（按中位步时）：mb 8 约 5.1、mb 12 约 5.8、mb 16 约 6.3，增益递减；mb ≥ 12 时 4 个 worker 的 loader 开始拖后腿（mb 16 数据等待均值 2.25 s），主训把 worker 提到 8。
- 四卡全局 16 的两种拆法：2 × 2 累积 9.15 crops/s，4 × 1 不累积 10.11 crops/s；按后者 320k 曝光约 8.8 h（NOT_MEASURED，按探针外推）。

## 无效或中断的运行（保留目录以备查）

| 目录 / 日志 | 为什么无效 |
|---|---|
| `single_b2`、`single_b4`、`single_probe.log`（10-09 15:2x 本地时间） | 卡 0 当时被 MC-GS 占满 99%，步时 8.1 / 9.8 s 不可信；显存数字可信（8.7 / 17.1 GiB） |
| `ddp_b2x2`、`ddp_probe.log` | 四张卡都被占，第 5 步 154 s，15:35 被 SIGTERM 终止，无 probe.json |
| `ddp_b2x2_g0567`（空目录）、`ddp_probe_g0567.log` | 目录被预先 `mkdir`，训练器拒绝写入已存在目录，立即退出；空目录可删（只列不删） |
| `ddp_b12x1_g0567`、`ddp_probe_b12x1_g0567.log` | 运行中 MC-GS 进入卡 0（44.8 GB），rank 0 OOM；无 probe.json |
| `single_b12_g1`、`single_b16_g1`、`single_probe_b12_g1.log`、`single_probe_b16_g1.log` | 同上，卡 1 被 MC-GS 进入后 OOM |

## 命令

```bash
cd /data0/congcong/code/Project_Doing/foundation_model
export PYTHONNOUSERSITE=1 PYTHONPATH=. NCCL_P2P_DISABLE=1 TORCH_NCCL_ASYNC_ERROR_HANDLING=1 OMP_NUM_THREADS=8
S=/data2/congcong/data/FM_data/derived/aur/ssl_manifest_v1/samples_ssl.json
P0=docs/verification/2026-10-09/anatobind_brain_ssl_first/p0
# 四卡（目录必须不存在）
CUDA_VISIBLE_DEVICES=0,5,6,7 timeout 900 ~/anaconda3/envs/nvgen/bin/torchrun --standalone --nproc_per_node=4 scripts/aur_ssl_probe.py --samples $S --microbatch 2 --grad-accum 2 --steps 100 --out $P0/ddp_b2x2_g0567_r1
CUDA_VISIBLE_DEVICES=0,5,6,7 timeout 900 ~/anaconda3/envs/nvgen/bin/torchrun --standalone --nproc_per_node=4 scripts/aur_ssl_probe.py --samples $S --microbatch 4 --grad-accum 1 --steps 50 --out $P0/ddp_b4x1_g0567
# 单卡
CUDA_VISIBLE_DEVICES=1 timeout 600 ~/anaconda3/envs/nvgen/bin/python scripts/aur_ssl_probe.py --samples $S --microbatch 8 --steps 15 --log-every 5 --out $P0/single_b8_g1
CUDA_VISIBLE_DEVICES=5 timeout 600 ~/anaconda3/envs/nvgen/bin/python scripts/aur_ssl_probe.py --samples $S --microbatch 12 --steps 15 --log-every 5 --out $P0/single_b12_g5
CUDA_VISIBLE_DEVICES=6 timeout 600 ~/anaconda3/envs/nvgen/bin/python scripts/aur_ssl_probe.py --samples $S --microbatch 16 --steps 15 --log-every 5 --out $P0/single_b16_g6
```

原始输出：各目录的 `probe.json`、`log_rank0.jsonl`、`probe_config.json`，以及同名 `*.log`。

## 主训配置（由此推出，PROPOSED，待 pilot 与用户确认）

microbatch 12 不累积、每 rank 8 个 worker；全局 batch = 12 × 卡数（4 卡 48、3 卡 36、2 卡 24），比 Q11 的 16 大；学习率按平方根缩放 `3e-4 × sqrt(全局/16)`（48 → 5.2e-4、36 → 4.5e-4、24 → 3.7e-4），预热按曝光数折算 `16000 / 全局` 步；曝光预算 320k 不变。
