# AnatoBind-Brain SSL-first 执行计划（2026-10-09，紧凑版）

**状态：执行中。** 本文把 GitHub 方案 `2026-10-09-anatobind-brain-ssl-first-three-stage.md`（T00–T14）落到这台机器上：用户 2026-10-09 下午的全部决定、任务顺序、每个任务的文件与验收、命令、时间线。方案与审查文件保持原文（措辞已按决定 Q3 / Q7 / Q12 / Q13 修正）；冲突以本文为准。截止：**2026-10-15（周四）向专家汇报全项目结果**。

## 0. 决定记录（用户 2026-10-09 拍板；grill-me 两轮 + 截止日重排）

| # | 决定 | 内容 |
|---|---|---|
| Q16 | 顺序 | **严格 Stage I → II → III**，不先跑随机初始化；周四展示到哪算哪。C0（随机初始化 II → III）仍是必跑对照，排在 C2 之后 |
| Q2 | 基线 | 本地 `main` 为基，`git merge` GitHub 分支（提交 8781786）：冲突处留 main 的实现，移植 fa6e6fd 的新增项（a/r_supervised、affine 坐标、网格校验、裁块内宿主真值、`--disable-anatomy-for`）；直接在 main 上做；**推送已批准** |
| Q14 | 推送节奏 | 每过一门推一次：基线合并、G0.5 smoke、G1、G3；每次附记录与测试输出 |
| Q3 | 旋转增广 | 图像与标签同变换，**坐标网格不转**（RoPE 用相对位置；旋转 = 解剖在扫描架坐标系里的姿态变化） |
| Q4 | 裁块物理尺度 | **(b) 离线重采样到 1 mm 各向同性**：BMSR 459 例 + ISLES 250 例 → `derived/aur/resampled_1mm_v1/`，表 `samples_1mm_v1.json`；图像三线性、标签最近邻、RAS 对齐网格；原生层厚 > 3 mm（ISLES 54 例、BMSR 9 例）保留 U/S 监督、**关闭 A/R 监督**；按 spacing × source × sequence 报告 |
| Q5/Q17 | GPU | 4 张空闲卡（优先 A800，A100 可混），`NCCL_P2P_DISABLE=1`；MC-GS 跑完后 4 张卡不再续发（用户在另一会话关）；忙卡上只做 ≤ 10 min 烟雾探针 |
| Q6 | Stage I 数据 | 四来源 train 患者全部序列 + **HCP T1w/T2w**（1 mm 重采样，曝光上限 ≤ 25%，单独 source 分层）；**SSL 验证集 = Stage II 验证集**（训练患者按来源留 10%）；fastMRI 厚层、膝不进 |
| Q7 | 归一化与泄漏门 | 整卷百分位归一化保留；泄漏测试在归一化后的裁块上改隐藏体素；**Stage I 视图不做任何模糊**（`crops.augment` 的面内模糊会把被遮体素混入可见体素） |
| Q8 | G1 | 严口径：冻结 backbone 的 13 宿主浅读出宏 Dice 比随机初始化同配置高 **≥ 0.05**（探针头 3 种子，区间不含 0），且病灶 ROI 对配对正常 ROI 的可分性 **≥ 随机 − 0.02**；阈值标 PROPOSED，pilot 后不改；**最多 2 轮修复**后交用户 |
| Q11 | Stage I 超参 | global 16 源裁块/步（microbatch 2 × 4 卡 × 累积 2；显存允许则 4 × 4 不累积），2 万步 = 320k 曝光；AdamW lr 3e-4、1k 步线性预热后余弦、wd 0.05；MIM Huber，目标 = 该视图自己增强后、遮挡前的归一化强度；连续块遮挡边长 16–32 mm，比率 0.6；InfoNCE τ 0.2、λ_c 0.1；全部 PROPOSED |
| Q19 | pilot | 只跑 mask 0.6 一组 8k 曝光（跑通 + 显存 + 吞吐 + 恢复），不扫比率 |
| Q12 | Stage II/III lr | 规格 §6：II 5e-4（前 1k 步 backbone × 0.1）；III backbone 5e-5 / 解码器 2.5e-4 / R 5e-4 |
| Q13 | 命名 | 全线 **AnatoBind-Brain**；Stage I 叫"自监督预训练"；不出现 foundation model |
| Q18 | 流程 | 紧凑计划 + 每个函数先写测试，控制器直接实现，每批一个独立评审代理；不做子代理逐字节抄写 |
| Q20 | 汇报 | 全项目（含阴性结果）、伪标签数字标 NOT_EVIDENCE、中文 PPT；周三起草 |
| Q15 | Level R | 用户安排读片人；Stage III 后按测试病人准备导出与盲法清单 |

## 1. 任务（按依赖顺序；每个任务 = 测试先行 + 实现 + 评审）

| 任务 | 文件 | 验收 |
|---|---|---|
| E0 基线合并 | 合并提交 8781786；`requirements-ci.txt` 加 SimpleITK | 全量 `pytest tests/` 通过；推送 |
| E1 1 mm 重采样（Q4） | `anatobind/aur/resample.py`、`scripts/aur_resample.py`、`tests/test_aur_resample.py` | 709 例写入 `resampled_1mm_v1/`；`samples_1mm_v1.json` 4960 行；`scripts/aur_prepare.py --stage grids` 对新表 0 错；厚层行 A/R 关闭计数 = 54 ISLES + 9 BMSR 例 |
| T01 SSL 清单 | `anatobind/aur/ssl/samples.py`、`scripts/aur_ssl_prepare.py`、`tests/test_aur_ssl_samples.py` | `samples_ssl.json`（四来源 train 患者全部序列 + HCP 行）、`val_patients.json`（按来源 10%，与 Stage II 共用）、`split_leakage_report.json`（test/val 患者零曝光）、`data_inventory.csv`（source × sequence × spacing）；HCP 行经同一重采样器到 1 mm |
| T02 Stage I 数据 | `anatobind/aur/ssl/dataset.py`、`tests/test_aur_ssl_dataset.py` | 复用 `crops.py`：整卷归一化、裁块、两个强度视图（偏置场、gamma、噪声；**无模糊、无镜像**）、coords/local/valid 不随视图变；测试：两视图共享几何、无效体素恒 −1、无模糊 |
| T03 遮挡与可见性 | `anatobind/aur/ssl/masking.py`；`swin.py` 向后兼容加 `visible` 输入 | 块遮挡（16–32 mm，按 patch 取整，只在前景有效 patch 上，比率 0.6）；stem 前置零 + mask token；**泄漏测试**：改任意被遮体素，F1–F4 全部不变（shifted window、PatchMerging、多尺度） |
| T04 MIM 头 | `anatobind/aur/ssl/heads.py::MaskedPatchDecoder`、`tests/test_aur_ssl_heads.py` | 从 F1 解码每个被遮 patch 的 2×4×4 归一化强度；Huber；loss 只在 valid ∧ masked ∧ 前景；非遮挡处 loss = 0 |
| T05 对比头 | `anatobind/aur/ssl/contrast.py`、`tests/test_aur_ssl_contrast.py` | F4 全局池化 → 128-D projector；两视图 InfoNCE（τ 0.2），跨卡 all-gather，同患者不作负样本；无 DDP 时退化为单卡 |
| T06 包装与 checkpoint | `anatobind/aur/ssl/model.py`、`losses.py`、`checkpoint.py`、`tests/test_aur_ssl_model.py` | `L_I = L_MIM + 0.1 L_contrast`；`ssl_stage1_best.pt` 只含 `backbone_state_dict` + 元数据（config、清单 hash、代码 SHA、seen_crops、种子）；`AnatoBindBrain.backbone.load_state_dict(strict=True)` 成功 |
| T07 探针 | `scripts/aur_ssl_probe.py`、`tests/test_aur_ssl_probe.py` | CPU 小尺寸 smoke；单卡 microbatch 2/4 两视图峰值显存与步时；4 卡 DDP 100 步（`NCCL_P2P_DISABLE=1`，`timeout 600`）；data wait；记录到 `docs/verification/2026-10-09/anatobind_brain_ssl_first/p0/` |
| T08 Stage I 训练 | `anatobind/aur/ssl/train.py`、`scripts/aur_ssl_train.py`、`tests/test_aur_ssl_train.py` | DDP/AMP(bf16)/累积/恢复/精确曝光计数/患者-来源-序列三层采样/HCP 上限；每 1k 步（pilot）或 5k 步在验证患者上记 masked-MAE（对三线性插值基线）、表征有效秩、两视图一致性；pilot 8k 曝光 → 主训 320k |
| T09 冻结探针 | `anatobind/aur/ssl/eval.py`、`scripts/aur_ssl_eval.py`、`tests/test_aur_ssl_eval.py` | 13 宿主浅读出（训练患者校准子集训，验证患者测，3 种子）与病灶/正常 ROI 可分性，SSL 对随机初始化同配置；**G1 判定**写记录 |
| T10 Stage II | `anatobind/aur/train.py`、`scripts/aur_train.py`、`tests/test_aur_training_contract.py` | `--stage II --init-backbone` 严格载入；`model.relation` 冻结；`sample_points(..., instance=...)`；a/u/r 门控；跨 rank 极端监督组合 100 步测试；240k 裁块 |
| T11 Stage III | 同上 `--stage III --resume-stage2` | 整个 II 继承；R 独立优化器组；`r_supervised` 真正门控；80k 裁块 |
| T12 评估与推理 | `anatobind/aur/eval.py`、`infer.py`、`scripts/aur_eval.py`、`scripts/infer_anatobind_brain.py`、`tests/test_aur_eval.py` | 整卷滑窗；A 13 宿主 Dice；U 逐病灶灵敏度 @ S7 工作点；R controlled ABA vs B0/B0*、rescue/harm；全部 NOT_EVIDENCE；`anatomy.nii.gz` / `lesions.nii.gz` / `record.json` |

C0（随机初始化 II → III）用 T10/T11 的 `--init random`，在 C2 之后立刻跑单种子。

## 2. 命令（实际路径；尖括号 = 尚未建立的新目录）

```bash
cd /data0/congcong/code/Project_Doing/foundation_model
export PYTHONNOUSERSITE=1 PYTHONPATH=. NCCL_P2P_DISABLE=1 TORCH_NCCL_ASYNC_ERROR_HANDLING=1 OMP_NUM_THREADS=8
PY=~/anaconda3/envs/nvgen/bin/python
# E1（已在跑，2026-10-09 15:10 起，4 worker，约 20 min）
nice -n 19 $PY scripts/aur_resample.py --samples /data2/congcong/data/FM_data/derived/aur/samples.json \
  --out-root /data2/congcong/data/FM_data/derived/aur/resampled_1mm_v1 \
  --out-samples /data2/congcong/data/FM_data/derived/aur/samples_1mm_v1.json --workers 4 --resume
# T01
$PY scripts/aur_ssl_prepare.py --aur-samples .../derived/aur/samples_1mm_v1.json --out .../derived/aur/ssl_manifest_v1/
# T07 / T08 / T09
timeout 600 torchrun --standalone --nproc_per_node=4 scripts/aur_ssl_probe.py --samples .../ssl_manifest_v1/samples_ssl.json --microbatch 2 --grad-accum 2 --steps 100 --out <P0_DIR>
torchrun --standalone --nproc_per_node=4 scripts/aur_ssl_train.py --samples .../samples_ssl.json --seen-crops 8000 --mask-ratio 0.60 --contrast-weight 0.10 --out <PILOT_DIR>
torchrun --standalone --nproc_per_node=4 scripts/aur_ssl_train.py --samples .../samples_ssl.json --seen-crops 320000 --mask-ratio 0.60 --contrast-weight 0.10 --out <STAGE_I_DIR>
$PY scripts/aur_ssl_eval.py --checkpoint <STAGE_I_DIR>/ssl_stage1_best.pt --random-init --out <G1_DIR>
# T10–T12
torchrun --standalone --nproc_per_node=4 scripts/aur_train.py --stage II --init-backbone <STAGE_I_DIR>/ssl_stage1_best.pt --samples .../samples_1mm_v1.json --seen-crops 240000 --out <STAGE_II_DIR>
torchrun --standalone --nproc_per_node=4 scripts/aur_train.py --stage III --resume-stage2 <STAGE_II_DIR>/aur_stage2_best.pt --samples .../samples_1mm_v1.json --seen-crops 80000 --out <STAGE_III_DIR>
$PY scripts/aur_eval.py --checkpoint <STAGE_III_DIR>/aur_stage3_best.pt --samples .../samples_1mm_v1.json --split test --out <EVAL_DIR>
```

## 3. 门

| 门 | 核查 | 不过 |
|---|---|---|
| G0 | `samples_1mm_v1.json` 网格 0 错；`split_leakage_report.json` 测试/验证患者零曝光；厚层计数；4 卡探针 | 停 Stage I |
| G0.5 | 泄漏测试全过；MIM + InfoNCE 有限有梯度；DDP 100 步不挂；恢复一致 | 停 |
| G1 | Q8 严口径；表征有效秩不塌缩；masked-MAE 优于三线性插值 | 修 Stage I，最多 2 轮，再交用户 |
| G2 | A 13 宿主宏 Dice ≥ 0.80；U 每来源灵敏度 ≥ nnU-Net − 0.05 @ 同误报；小病灶与 64-query 饱和单列 | 记录、停、交用户 |
| G3 | R controlled ABA ≥ B0；rescue/harm；Level R 另算 | 报告不足 |

## 4. 时间线（周四汇报；GPU 自 2026-10-09 15:00 起 8 张全空）

| 日 | 内容 |
|---|---|
| 周五 10-09 | E0 ✔ 合并；E1 重采样（跑中）；T01–T03 |
| 周六 10-10 | T04–T07；G0.5；pilot；晚上起 Stage I 主训（估 8–12 h，NOT_MEASURED） |
| 周日 10-11 | T09 G1；T10–T11 代码；若 G1 过：Stage II（约 4 h）、Stage III（约 1.5 h） |
| 周一 10-12 | T12 评估；C0；修补 |
| 周二 10-13 | 评估收尾、图表、记录 |
| 周三 10-14 | 汇报 PPT 起草；用户审 |

记录目录：`docs/verification/2026-10-09/anatobind_brain_ssl_first/`（p0/、g0/、g05/、g1/、g2/、g3/、README.md）。每个数字附命令与原始输出；伪标签数字标 NOT_EVIDENCE；未实测的时间标 NOT_MEASURED。
