# AnatoBind-Brain Stage I→II→III：4×A800 80GB 训练就绪性审查（2026-10-09）

**Review 判定：CONDITIONAL / BLOCKED FOR FULL TRAINING。** 三阶段研究路线保持不变，但启动 320k Stage I 主训以前必须完成本文件 P0/P1。此文是实施方案的工程审查和修订记录，**不是**已经运行 Stage I/II/III 的实测报告。

- 本次审查的 GitHub 方案：`docs/superpowers/plans/2026-10-09-anatobind-brain-ssl-first-three-stage.md`，分支 `feature/aur-brain-ssl-first-2026-10-09`，审查起点 `521aa1b758352fc63af3ba0ac6fdc8d2c07564fc`。
- 同时核对 MRI-148 本地工作副本 `/data0/congcong/code/Project_Doing/foundation_model`、`STATUS.md` 与 `docs/verification/2026-10-08/anatobind_brain_aur/p0/`；**MRI-148 本地 HEAD 与 GitHub 新分支不是同一提交**。记录在本地、尚未通过这次提交引入 GitHub 的文件，均应标作“本地记录”。
- 本次提交是**文档与执行约束**，不混入未测试的代码合并；禁止将服务器本地 `main` 强推覆盖 GitHub 其他分支。Part 1 的完整测试记录不能替代新分支的回归测试。
- MRI-148 连接器仅提供文件/元数据读写，不提供 Shell、CUDA 作业提交或进程控制；本次无实际 GPU 作业，也未验证当前 GPU 是否空闲。

## 1. 已核实的证据和边界

| 项目 | MRI-148 本地记录 | 正确解释 |
|---|---|---|
| Part 1 模型规模 | `p0/model_params.txt`：backbone 12,929,792 参数；完整 A/U/S/R 22,719,561 参数 | 不能按总参数量单独预测训练显存 |
| 单卡探针 | `p0/probe_single.txt`：128×160×160，BF16，B4 无 checkpoint，18.4 GiB/卡、0.862 秒/步 | 合成输入、完整 A/U/S/R；不包含 Stage I 双视图和真实 I/O |
| 4 卡 DDP 探针 | `p0/probe_ddp.txt`：4 卡×B4、无 checkpoint，100 步，18.48 GiB/卡、约 0.90 秒/步 | 使用 `NCCL_P2P_DISABLE=1` 后才成功；不可从中推导 Stage I 耗时 |
| 默认 NCCL | 同一记录中默认 P2P 路径超时，退出码 124 | 必须作为执行前置条件，不可默认多卡通信正常 |
| 本地回归测试 | 验证 README：910 passed、1 skipped，87.78 s | 属于服务器后来本地 `main` 的测试记录；**不是新 GitHub 分支**测试结果 |
| 数据清单 | `p0/samples.txt`：总 4,960 行，训练 3,904 行，其中 1,359 行提供 U 监督 | 行是「患者×序列」，**不是独立患者数**；SSL 可用的无标注脑数据需重新审计 |
| 物理网格 | `p0/grids.txt`：4,960 行与对应伪解剖/病灶 mask 栅格检查一致 | 栅格相等不能替代真正的临床解剖/病灶正确性 |
| 空间尺度 | `p0/spacing.txt`：PDGM/SibBMS 为 1mm；ISLES 主要约 2mm，部分约 4.8mm 层厚；BMSR 多间距 | 不能将各来源的固定体素大小裁块视作同一物理 FOV |
| ISLES 伪解剖 | `p0/isles_ruling.md` 有 volume/montage 的初步 QC，维持 A 监督 | 仍是伪标签 QC 而非临床真值；`a_supervised`/`r_supervised` 要可关闭 |

**版本差异证据：** MRI-148 当前本地 `anatobind/aur/dataset.py` 存在 `nib.as_closest_canonical` 与 `entity_present`；本次审查时 GitHub 新分支 `dataset.py` 不具备这两项。应**逐文件三方审查**，选择性引入本地已经验证的 RAS/出现性逻辑；不得将新版源码与旧标签约定盲目混合。

## 2. Blocking P0：进 GPU 主训练之前必须修改

### P0-1：代码基线合并、RASH/LPS/LAS 方向和物理尺度

- 当前 MRI-148 本地 `main` 的 RAS 归一化及出现性逻辑属于后续实现，GitHub 新分支落后于它。先对比 `swin.py`、`dataset.py`、`crops.py`、`losses.py`、`labels.py`、`tests` 的差异，选择性移植并重新跑测试。**不可只 cherry-pick 一个文件而不更新测试/样本契约。**
- 处理顺序：每个来源读取 NIfTI affine/xyzt 单位 → `nib.as_closest_canonical` 或显式 RAS 重采样（仅当必要时）→ 同样变换 anatomy/lesion → 栅格一致性/左右身份测试 → crop/物理坐标/回写变换。unknown 单位必须单独判定，禁止静默按 mm 使用。
- 三种分辨率情形分层：① 1mm 3D 主 SSL；② BMSR 中间尺度按物理 FOV 采样/可选离线统一；③ 厚层 ISLES 作为适应性来源，不声称插值能产生丢失的真实高频解剖。务必报告 spacing × source × sequence 数量和拒绝率。主方案先冻结一套 crop 毫米视野策略，后做 native-grid 消融。
- **测试门**：LPS/LAS/RAS 同一体积变换后结构及侧别一致；图像/解剖/病灶 shape 相等但 affine 不同必拒绝；旋转后坐标和 mask 同步；输出逆变换恢复原始 affine。按物理间距分层核对小病灶 10mm³ 下限的最小体素数。

### P0-2：Stage I 真正无泄漏的 masked Swin

- 当前 `SwinBackbone.forward(image,valid,coords,local)` 的 stem 是 `Conv3d(kernel=stride=(2,4,4))`，尚无 `M_visible` / SSL decoder。遮挡必须在 stem 前发生，隐藏 patch 的原像素不得经 stem、归一化统计、任何 view augmentation 或残差路径流入重建特征。
- 独立维护 `M_valid`（真实有效/非 padding）和 `M_visible`（人为保留观测）；attention 的有效 key、merged token、F1–F4 均要在测试中区分二者。**对 masked patch 的预测不能用该 patch 的真实 image-derived feature 作跳连。**
- 对比任务可以使用独立的温和增强视图，但是需要标明是否无 mask / 共同 mask，避免通过另一视图在训练时向 MIM 直接传输目标 patch；严禁在“掩码不变性测试”的可见像素归一化中使用隐藏区域强度统计。
- Mask 0.6 是 pilot 初值；比较 0.4/0.6/0.7，并测试**物理尺度连续块遮挡**，不只是单个 2×4×4 patch 的邻近插值。MIM target 是前景有效隐藏体素，loss 边界与 padding 必须精确。
- **禁止信息泄漏测试**：固定可见原像素、mask、增强随机种子及坐标，只改变隐藏原像素；MIM prediction 在数值容差内不变；测试 shifted-window、patch merge、多尺度分支和整条前处理链。另测 loss 仅覆盖掩码前景。

### P0-3：修正 Stage I / II / III 的 batch 和执行预算定义

- **Stage I 2-view 不是 Part 1 的单-view probe**。建议从 microbatch=2/GPU，4 GPU、grad-accum=2 开始：每次 optimizer step 读取 **global 16 个 source-crops**，Backbone 对两个 view 分别计算（若为 2 路输入，则每步 32 次 view-exposure）。主预算 320k source-crops ≈ 20k optimizer steps。这里 `microbatch` 不应混淆“每视图”与“每原始裁块”。
- 如果 Stage I 4 GPU×2、grad-accum=1 则 global 8，320k 对应 40k optimizer steps；两种预算等效裁块数量相同，但每步优化与 LR schedule 不同，必须连同步数、warmup、日志统一校正，不能默认为旧方案 40k 固定步数。
- Stage II/III 使用已验证合成 B4/GPU 作为**候选**，不是承诺真实 B4 可用。候选 global16：Stage II 240k≈15k 步；Stage III 80k≈5k 步；GPU 空间足够也不能忽略 decoder 特征梯度和真实 loader 开销。
- Stage I 应开关 checkpoint / BF16 / microbatch/accum 并真实测量 CUDA peak allocated **和** reserved、CPU RSS、data wait、IOPS、吞吐(crops/s)、平均和 P95 step time、checkpoint-save 开销、可复现数据采样吞吐。
- 明确 `NCCL_P2P_DISABLE=1` 作为**本台 MRI-148 当前探针证实的兼容配置**，使用有界 `timeout`；诊断拓扑/驱动后可另测默认路径，但不得把默认 NCCL 当成当前可执行基线。`torchrun` 的 GPU 选择要求四张空闲且权限允许，不强占正在使用的设备。

### P0-4：DDP 无监督子批和 loss 梯度

- Stage II 必须冻结 `model.relation`；确认 DDP 包装前冻结，训练器不能暗中将 R 参数视为未用参数。
- 当某卡所有样本 `u_supervised=False`（SibBMS/T1/ADC 等）或 `r_supervised=False` / 当前批次没有有效实例时，确保前向/反向/梯度同步路径在所有 rank 上一致。实现要么有明确的 0*decoder-output 梯度锚点，要么启用并测量 `find_unused_parameters=True`；不得仅在“每张卡两枚合成病灶”上验证。
- Stage II 输入必须实际传 `instance` 给 `sample_points`（否则实例感知 quota 不生效），保持 Hungarian matching 与点 BCE/Dice 的 `point_weight` 一致；统计每病灶 sampled-positive 点数与覆盖率。
- Stage III 中 `r_supervised` 必须真实作用到 R loss 与其有效样本数，不能仅存在于 JSON 字段。对 no-lesion、no-host、non-supervised A/U、不同 rank 不同实例数及全 mask ignore 分别写测试。
- **验收门**：4 卡合成跨 rank 极端监督组合测试 100 步（含保存/恢复）；观察无挂死、参数/梯度同步、损失有限、被关闭任务梯度正确屏蔽和可恢复。

### P0-5：I/O / 物理坐标内存 / Stage I probe

- 现有 `AURDataset.__getitem__` 每次 `load_volume` 并读入整卷 NIfTI；Stage I 两视图 + 4 GPU 可能由 CPU/NFS IO 而非显卡计算决定吞吐。选择每 worker 有界 LRU、chunk/cache、明确预处理格式，避免 4×workers 并发反复读 gzip NIfTI；检查文件许可与输出目录空间，不复制身份敏感数据到 Git。
- 每个 128×160×160 裁块的 coords+local 是 6×3,276,800 个 FP32 值，单 crop 约 **75 MiB（十进制约78.6 MB）**，这是 CPU→GPU 传输而非显存估计的总量。考虑从 affine/crop window 在 GPU 上构建 token-level 物理坐标；但不许未经等价测试就替换原位置信息。
- 阶段间一致：train/val/test 必须对同一输入执行相同坐标规范化/图像归一化；STAGE I 使用原图的空间位置不代表患者间 alignment。以真实 NIfTI 做小样本 forward/bwd probe 和输出反投影冒烟检查。
- **验收门**：CPU loader 与四卡 GPU step 重叠合理，无重大 stall；混合分辨率下无 OOM/NaN；记录真实 GPU 型号、PCIe/NVLink 拓扑诊断输出（不含账户敏感数据）。

## 3. P1：研究有效性、Stage I 探针和 R 指标

1. **G1 不应仅写“未显著差于随机初始化”。** 预先锁定主冻结探针（例如 13 类宿主 shallow readout mean Dice、病灶区域与配对正常区可分性/小于5mm 的局部探针），报告 3 seeds/置信区间；至少主探针体现有意义的提升且小病灶表征不恶化，否则不得宣称完成有价值的 Stage I 自监督预训练。
2. **缩小跨模态-疾病 confounding。** Train manifest 要输出患者/source/sequence/spacing 交叉表，InfoNCE 跨卡 gather 避免同患者不同序列成为假负样本；多序列正样本仅在几何严格匹配时使用。对比损失温度 0.2、权重 0.1 是试验起点，检查全局一致性是否损害 U 的局部敏感性。
3. **R 需证明优于 B0，而非伪标签复刻。** coarse F1 几何当前是 event-centroid→host voxels 的最近距离，非病灶表面到宿主表面距离；`torch.cdist` 需复杂度 probe。测试 1–5mm/低分辨率体素，确认软 mask 和 side 在边界处的稳定性，做细尺度几何消融。人工独立 Level R 测试给出 rescue/harm，B0、B0*、end-to-end 与 controlled 口径不混淆。
4. **实际样本数与 Stage I 规模。** MRI-148 有 4,960 序列行而非 4,960 独立患者；Stage I 320k 是采样曝光，不是新的独立数据。额外 HCP / OASIS 等训练源需要完整授权、去重、空间/年龄群体核查，不能因服务器目录存在就自动纳入。
5. **A/U 评估的公正性。** A≥0.80 是 SynthSeg 伪参考工程门；U 小病灶敏感度在相同 FP/case/输入序列/切分上对比 nnU-Net 与 C0/C1/C2；单一序列不具 U 标注时不作为 U 阴性。64 query 饱和病例单列报告。
6. **资源整体预算。** C0/C1/C2 每个 3 seeds 时，若 C1/C2 分别训练 Stage I，则是 **6 次 Stage I + 9 次 Stage II + 9 次 Stage III**；C3 直接复用 C2 Stage III 的预测，无需重训 Stage I。短 pilot 只能决定配置，不能以单 seed 的差异论证泛化。

## 4. 更新后的分阶段落地顺序（先完成前一 gate）

| 门 | 先做任务 | 明确的停止条件 | 产物 |
|---|---|---|---|
| G0 / P0 | 当前 GitHub vs 服务器 local main 差异逐文件归并、RAS/spacing split、数据质量、单/四卡通信和真实 I/O | 未锁病人划分、左右侧/RAS 不一致、不能复现 DDP → 禁止主训 | 合并审计表，真实 NIfTI/grid/spacing/QC，GPU preflight |
| G0.5 / SSL smoke | masked Swin 输入无泄漏、MIM + InfoNCE 数值/梯度/恢复、跨 rank 去重、CPU 小尺寸+真实 crop probe | 任何 masked 内容泄漏、NaN、DDP hang、显存/数据管线不稳 → 停 | SSL smoke report + perf JSON |
| G1 / Stage I | 8k source-crop pilot → 锁定超参 → 320k 主训练 | 表征塌缩、冻结探针差或异常细节退化 → 不进入 II 主线 | backbone-only + full resume checkpoint |
| G2 / Stage II | Stage I strict load → A+U+S 训练，U 小病灶分层与同条件 baseline | 没有完整监督/负载证明或工程门不过 → 停 | Stage II checkpoint + 验证报告 |
| G3 / Stage III | Stage II strict load → R 训练，B0 vs learned R、人工 Level R | 若只拟合 SynthSeg 几何，不宣称关系推理超越 B0 | Stage III checkpoint + pseudo/human 分离的报告 |

**建议的第一项正式执行任务：** 先将 MRI-148 本地已经验证的 RAS/`entity_present` 变更与新分支做基线同步，并在 GitHub 提交 diff + 测试。随后处理 Stage I 的 mask 输入和双视图四卡真实探针。**不直接启动 320k Stage I**。

## 5. 实际 4 GPU 命令要求（执行环境须有 Shell；以下是接口示例，不是已执行命令）

~~~bash
# GPU device assignment only after verifying devices are idle and accessible.
export CUDA_VISIBLE_DEVICES=<four_idle_GPU_ids>
export NCCL_P2P_DISABLE=1
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export OMP_NUM_THREADS=8
export PYTHONNOUSERSITE=1
export PYTHONPATH=.

# After T01-T08 in the 2026-10-09 implementation plan are actually implemented:
timeout 600 torchrun --standalone --nproc_per_node=4 scripts/aur_ssl_probe.py \
    --samples <locked_train_manifest> --microbatch 2 \
    --grad-accum 2 --steps 100 --out <new_probe_dir>

# Do NOT run the 320k experiment until the probe and G0/G0.5 results are reviewed.
~~~

> 注意：`CUDA_VISIBLE_DEVICES` 的值必须由服务器当前空闲 GPU 状态确定；尖括号为待实现接口占位符。当前 MRI-148 文件连接器**无法执行这些命令**。本审查的“可执行”指写清未来脚本和验收合同，不代表脚本或 GPU 任务已运行。

## 6. 修改/未修改的范围

- **修改：** 新增本 Review；实施计划增加审查前置条件、修正 Stage I batch 定义及环境依赖；设计增补加审查入口。
- **未修改：** A32/U64/S6/R13+1 模型架构、I→II→III 研究主线、任何 Stage I/II/III 网络代码、训练脚本、GPU 服务、医学数据文件、`main` 分支。
- **状态：** 本审查结论属于工程分析；所有尚未实测的 Stage I 训练速度、性能、loss 增益都必须明确标记 **NOT_MEASURED**。
