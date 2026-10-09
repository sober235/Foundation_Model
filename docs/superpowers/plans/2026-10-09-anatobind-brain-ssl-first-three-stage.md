# AnatoBind-Brain：Stage I → II → III 三阶段实施计划（2026-10-09）

**状态：实施计划；尚未执行数据检查、预训练或正式训练。**
**代码基线：** `feature/aur-brain-part1-2026-10-09` 的 `fa6e6fd95be8e99e5b4ac1bcbd7fc13e54b0102b`；本计划在独立新分支制定，不重写历史代码。
**基于：** `docs/superpowers/specs/2026-10-08-anatobind-brain-aur-design.md` 与 `docs/superpowers/plans/2026-10-08-anatobind-brain-aur-part1.md`。
**决策变更：** 将原先“跳过 Stage I，直接 II→III”的工程验证路线改为**正式训练必须 I→II→III**。仅将 II→III（随机初始化）保留为消融基线；不把预训练损失收敛等同于通用视觉能力已经建立。

## 2026-10-09 四卡 A800 执行性 Review（优先阅读）

**结论：研究路线保留，正式 Stage I 主训暂缓。** MRI-148 本地后续 Part 1 记录已有 910 passed/1 skipped、128×160×160 BF16 合成数据 4×A800×B4 的 DDP 探针（0.90 s/step、18.48 GiB/GPU），**但是这些不是此 GitHub 新分支的重跑记录，也不是 Stage I 双视图/真实数据的测量**。默认 NCCL P2P 初始化曾超时，只有设置 `NCCL_P2P_DISABLE=1` 的四卡探针成功。

- 完整逐项阻断项、证据位置、验收门和四卡环境要求见：[2026-10-09 四卡执行就绪性审查](../reviews/2026-10-09-anatobind-brain-four-a800-execution-readiness.md)。以此文的 G0/G0.5/Stage I 预算定义优先于下文初始估计。
- **GitHub 基线差异必须先解决**：新分支从 `fa6e6fd` 继承，而 MRI-148 本地后续 `main` 已有 RAS 统一、`entity_present` 和更完整的 P0 记录。按文件核对、移植、运行新分支测试；不要直接将服务器本地分支强制推到远端。
- **重新测 Stage I**：建议 microbatch 2/GPU×4 GPUs，grad accumulation 2，global 16 source-crops/optimizer step（两个对比视图意味着 32 view-exposures/step）；320k 曝光对应 20k steps。若 grad accumulation 1，global 8 则为 40k steps。先 8k 曝光 pilot，不由 Stage II 探针外推速度。
- **修正 G1/G2 之前的实现门**：真正无泄漏 masked patch 输入；归一化/augment 不能使隐藏原图像素进入可见值；RAS/physical-FOV 一致；DDP 各 rank 可有不同 U/R 有效监督；`a_supervised`/`r_supervised` 在相应 loss 中实际生效。
- **明确用户当前工具权限**：MRI-148 连接器能检查数据/读取日志文件，但没有 shell/GPU 作业 API。文中所有 GPU 命令属于未来可执行的 CLI 合同，**尚未运行**。

## 0. 一句话、边界与实际交付

先在严格隔离下游测试病人的脑部 MRI 上自监督预训练**同一套可变尺寸 3D Swin Backbone**（Stage I）；以该权重初始化同一个 A/U/S 联合感知模型（Stage II）；再在预测解剖/异常的视觉表征上学习逐病灶 13+1 宿主竞争（Stage III）。最终检验“先预训练”是否提高小病灶发现、脑结构识别和绑定稳健性，而非仅提高 masked reconstruction 的 PSNR。

- **固定模型：** 单序列 3D MRI，Patch (2,4,4)，Swin 通道 (64,128,256,512)、深度 (2,2,6,2)、注意力头 (2,4,8,16)、窗口 (4,8,8)、物理 3D RoPE、局部位置、M_valid；A=32 固定身份 Query、U=64 匿名实例 Query、S=6 序列 Token、R=13 个分侧宿主 + no_host。绝不改回“三个疾病分类器”。
- **范围：** 脑部；T1/T1c/T2/FLAIR/DWI/ADC；跳过跨器官、k-space 干预、Stage IV、生成式诊断印象和“所有未知疾病均可识别”的主张。
- **训练产物：** Stage I `ssl_stage1_best.pt`（仅导出 backbone 权重和完整元数据）；Stage II `aur_stage2_best.pt`；Stage III `aur_stage3_best.pt`；逐病人分组评估、B0 对照、人工审核的独立 Level R 结果。
- **原有 Part 1 仍有效：** `anatobind/aur/{swin,rope,heads,relation,model,dataset,targets,losses}.py`；本计划新增 Stage I 部件、权重迁移入口和训练/评估管线，而不移除现有实现。
- **禁止将工程产物当成效果证据：** dry-run、GPU probe、SynthSeg 对 SynthSeg 的一致性与内部伪关系结果只作为 NOT_EVIDENCE；需独立标注与下游任务指标。
- **安全和执行规则：** 不删除已有数据或实验目录；新建输出前检查路径不存在；仅使用获许可数据；患者数据、NIfTI 内容、认证信息不进入 Git；源数据保留服务器，仅在仓库存配置、汇总统计、不可反推个人身份的检查报告。

## 1. 端到端架构及依赖

~~~text
数据登记 + patient-level split + NIfTI grid/QC + P0 hardware probe
                         |
     [Stage I] 3D MRI Unlabeled Volumes (train only)
       |--> masked 3D input / token observability
       |--> shared 3D Swin (64/128/256/512; 3D RoPE)
       |--> lightweight masked-patch reconstruction head -- L_MIM
       '--> image-level projector (paired views) -------- L_contrast
                         |
                  backbone_only.pt
                         |
     [Stage II] same Swin backbone (weights loaded, not frozen by default)
       |--> 32 fixed Anatomy Queries (A): presence + masks
       |--> 64 anonymous Event Queries (U): presence + instance masks
       '--> Sequence Head (S): T1/T1c/T2/FLAIR/DWI/ADC
                     L_A + L_U + L_S
                         |
                 stage2 checkpoint
                         |
     [Stage III] same Backbone + A/U/S, add Relation Head
       |--> each matched U event + A host candidates + S + G
       '--> 2-layer CandidateCompetition -> 13 host + no_host
                     L_A + L_U + L_S + λ_R(L_host + λ_h L_hard)
                         |
                  held-out evaluation
           A anatomy | U detection | R correct host
~~~

**Stage I 只训练 backbone + 两个辅助 Head，辅助 Head 不进入 Stage II；Stage II 继承 backbone；Stage III 继承整个 Stage II 并初始化 R。** 三段为顺序依赖，不是三个独立模型。

## 2. 数据与防泄漏协议（先于任何训练）

### 2.1 两套有区别的清单

1. **`samples_aur.json`：** 沿用现有四个主来源和 per-sequence 标志。UCSF-PDGM、UCSF-BMSR、ISLES-2022、SibBMS 的实际例数、文件/通道质量、可训练比例由服务器 inventory 计算，不从规格中的估计数推断最终训练集大小。PDGM 与 BMSR 采用 S4 锁定划分；ISLES 按患者划分；SibBMS 保持已有病人级划分。锁定 train/val/test 文件及 SHA256。
2. **`samples_ssl.json`：** Stage I 使用以上来源中的 **train 患者**全部合格 3D MRI（包括不具备 U 标注的序列），另外可纳入经过授权、身份可去重、满足网格/QC 的未标注 3D 脑 MRI（如已有 HCP T1）。无病灶标签也可参加 Stage I，且 Stage I 不能读取 U/R 标签作为训练输入。fastMRI 厚层 2D 脑卷只进入可选的 separate-domain ablation；SKM-TEA 是膝部，不纳入脑主预训练；TotalSegmentator-MRI 中非脑图像不纳入此轮。

### 2.2 锁定原则

- 先生成全项目统一 patient key（source + subject + session 并同时做跨来源重复性检查）；为同一患者的所有序列、增强视图、重复扫描分配同一 split。以图像 hash、元数据和可用的匿名体素指纹检查明显重复，疑似同一受试者进入人工核查清单。
- **所有下游 test/val 患者都不得用于 Stage I 的梯度更新。** 独立固定 Stage I 验证患者可用于训练过程选择但不能用于梯度更新；下游最终测试仅在三个阶段全部锁定后打开。对额外未标注数据只在确认不与锁定测试来源重复后使用，无法排除重复者不纳入主实验。
- 多序列配对只在同一患者、同一成像空间、验证 affine 后使用；绝不以同名序列推断已配准。
- NIfTI 以真实 affine 在 mm 空间处理（先确定单位）；image/anatomy/lesion 对齐需同时检查 shape、orientation 和 affine；空间单位 unknown 的病例单列核实，不能默认为可信物理空间。拒绝 silently resize/flip；若需配准/重采样则显式生成新版本及记录。增强时图像、valid、坐标、局部坐标必须采用同一空间变换。
- 数据平衡按**患者/来源/序列**三层采样，不能让序列=疾病来源的强相关性替代异常学习；记录各层有效采样曝光量。
- 阶段门 G0：生成 `cases_aur_locked.json`、`samples_ssl.json`、`samples_aur.json`、`data_inventory.csv`、`grid_report.csv`、`split_leakage_report.json`、ISLES 解剖 QC 可视化；无可核查的锁定划分及物理网格检查，不启动 Stage I。

## 3. Stage I：面向后续 A/U/R 的 3D 自监督预训练

### 3.1 模型兼容性（必须使用现有 Backbone）

- 直接复用 `anatobind/aur/swin.py::SwinBackbone` 的 **同名模块、同一参数形状**，以 `state_dict` 导入 II/III；不另外训练不同的 MAE-Transformer 再困难蒸馏。
- Stage I 输入仍为 `(image, valid, coords, local)`。在内部新增 **M_visible**（可观测 patch）与 **M_valid**（物理/填充有效 patch）区分：
  - `M_valid` 决定哪些体素真实存在；**不等价**于人为遮挡标志。
  - `M_visible` 决定哪些有效 patch 可把真实图像信息送入网络。遮挡在 patch embedding **之前**进行，不能让卷积 stem 首先看到目标 patch 的真实强度再 mask token，否则产生重建泄漏。
  - Mask token 可以作为 query 读取可见 patch 的上下文，但遮挡 patch 不能以自身真实强度成为 key/value；shifted-window、PatchMerging、残差都要覆盖防泄漏测试。不得直接把 `M_valid=0` 当作 masked patch：会改变 pooling/几何语义并可能让模型只在边界外“重建”。
  - 原生 NIfTI affine 与 RoPE 一致，禁止额外的镜像增强（防止左右标签/关系歧义）。主配置使用来自训练集中真实体素的有效前景块，确保 mask 前景不被纯背景占满。
- 建议的实现策略：两分支输入中，对 masked patch 预填可学习 `mask_token` 或零值、同时为 stem 建立输入可见性门控；对 masked query 在 token 空间插入 mask token。选择的具体实现须通过“任意改变被遮挡原像素，所有 visible context / MIM 预测保持不变”单测。

### 3.2 自监督任务与损失

**任务 I-A：3D masked image modeling (MIM)**

- 主训练使用 **三维连续块遮挡**，初始 mask 比率 0.60；pilot 比较 0.40 / 0.60 / 0.70。按实际 patch 网格取整（patch=2×4×4），保证每个窗口有足够可见 token；重建监督只在 `M_valid ∧ M_masked ∧ M_foreground` 上。
- 用轻量 patch decoder：从 F1 或浅层多尺度特征重建每个 masked patch 的 2×4×4 个归一化强度值；不要为 MIM 建立与 A/U Query 一样大的全分辨率解码器。损失以 **Huber/MAE** 为主，逐样本先归一化再平均，避免患者体积/场强权重失衡。
- 监测 masked foreground MAE 与随机/局部插值等简单基线；重建图片好看不是效果门槛。
- **病灶敏感性防护：** 不使用过强空间模糊、elastic 形变或“把亮点去掉”的视图生成；对有病灶标注的训练子集仅作**审计探针**，按体积与来源检查局部异常特征是否显著退化。不得把病灶掩码作为 Stage I 监督目标。

**任务 I-B：contrastive consistency**

- 每卷/裁块生成两种温和的强度视图（bias field、gamma、Rician/Gaussian 小噪声、小范围对比度变动；不做左右镜像和强模糊），经同一 Backbone 与 128-D projection head，训练同患者/同位置的两视图正对一致性。
- 对比损失优先 **患者去重的 NT-Xent/InfoNCE**（初始温度 τ=0.2，系数 λ_c=0.1），跨 4 卡 gather 表征；同患者其他序列/重复裁块不是负样本。若批内不同患者太少，首先调整 patient-balanced batch sampler；memory queue 作为另一个显式消融，而不是悄悄修改口径。
- 不强制 T1/FLAIR 同 voxel 特征相同：跨序列正对齐仅在配准及可见解剖一致时单独进行消融。全局对比不应压制细粒度病灶差异，需监测局部异常探针。

**阶段目标：**

\[
\mathcal L_{\mathrm{I}} =
\mathcal L_{\mathrm{MIM}} + 0.1\,\mathcal L_{\mathrm{contrast}}
\]

数值 0.60、τ=0.2、0.1 是**可执行初值，不是仓库已有实验验证值**，pilot 中冻结其选择后才能正式训练。

### 3.3 训练预算与 Stage I 验收

- 先做 8k crop 曝光 pilot：跑通 forward/backward、梯度、loss mask、双视图、DDP/all-gather、不同源采样、恢复训练、显存峰值、吞吐；需提交真实 GPU 探针后才定正式 batch/worker。
- 正式 Stage I 暂定 **320k source-crop 曝光**（建议先从 4 GPU×microbatch 2、gradient accumulation 2 = global batch 16 进行 probe：约 20k optimizer steps；若 accumulation=1、global batch 8 则约 40k steps；两个增强视图相当于约 640k encoder view-exposures）。4×A800 80 GB、AMP、梯度检查点和按空闲卡调度与原方案一致；绝不以预计秒数声明训练时间，使用真实单/四卡 P0 探针更新预算。GPU 不足时停并报告，不擅自占用他人任务。
- 每固定 5k 步（pilot 每 1k）在不参与梯度的 Stage I 验证患者上做 `masked-MAE`、表征有效秩/方差防塌缩、两视图一致性、序列/数据源分组指标；仅以训练集拆出的 calibration probes 测**冻结 Backbone 的 13 宿主解剖轻量读出、异常区域可分性（按病灶大小）**。
- **门 G1（先于 II 主训练）：** 完整 SSL smoke+unit tests；无 masked-content 泄漏；所有损失有限/有梯度；表示未塌缩；训练与 held-out SSL 指标优于简单基线；冻结探针至少不显著劣于随机初始化同配置，并披露样本数和置信区间；锁定 `ssl_stage1_best.pt`。未满足则只允许修复 Stage I 或提交偏离申请，**不自动跳过 Stage I**。
- Checkpoint 必备：`backbone_state_dict`、config、数据清单/hash、代码 SHA、epoch、seen_crops、optimizer/scheduler/scaler、随机种子、RNG 状态、数据拆分版本、validation 记录。对接 II 时加载 backbone 严格键/形状检查并打印 missing/unexpected keys；辅助头只存于 SSL 完整恢复 checkpoint，不载入下游模型。

## 4. Stage II：A+U+S 联合结构化感知

- 输入与原规格一致：一卷一个序列；裁块 128×160×160；Stage II 主实验初始化为 `ssl_stage1_best.pt` 的 **同结构 Swin**，A/U/S 解码器按现有代码随机初始化，并联合端到端微调（可在前 1k 步对 backbone 小学习率 warmup，不冻结整个阶段）。
- **保持 A=32 / U=64 / S=6，不添加疾病分类头。** U supervision 仅用于在当前序列有有效标注的病例；SibBMS 未标注 U 的样本始终遮罩 U loss，不能当作 U-negative。
- U 实例点采样在训练入口传入 `instance`，采用约 50% valid uniform / 30% per-instance positive / 20% peri-lesion hard negatives；统计每例每实例至少 1 点、目标点数、极小病灶覆盖率；Hungarian cost 与 event BCE/Dice 使用相同 `point_weight`；裁块 sliver 未达到最小体积时不强行作为实例。
- A 对 SynthSeg 伪标签监督只在 `a_supervised=True` 使用；ISLES QC 未过则同时关闭 `r_supervised` 的伪关系监督，但保留其有效 U/S。物理坐标由真实 NIfTI affine 给出，不把 shape 相等当作同空间；R 的训练目标按**最终裁块及几何增强后**的 A/U mask 计算，绝不照搬整卷 host。
- 两阶段传递：`L_II = L_A + L_U + L_S`；沿用 Stage II **240k crops** 预算与 AdamW/AMP/DDP 方案；必须通过 P0 probe 才能确定 batch 2/GPU 可行性。用训练病人中独立 validation 选择 `aur_stage2_best.pt`，不得对 test 调阈值。
- **门 G2：** A 13 宿主宏平均 Dice 对 SynthSeg 伪标签 ≥0.80（仅工程门），U 在每来源相同 FP/case 工作点的 lesion-level sensitivity ≥ nnU-Net 对照 − 0.05；额外强制报告小病灶按体积/等效直径分层灵敏度及 64-query 饱和病例（实例数 >64、漏掉个数），不把三类监督效果称作 unseen pathology generalization。

## 5. Stage III：基于自有视觉证据的关系绑定

- 加载**整个** Stage II checkpoint；加入 `CandidateCompetition` 两层关系模块，每个预测 U Query 对 13 个侧别宿主 + no_host 独立竞争，保留 S 与视觉 Embedding。
- 结构 `r_ij = φ(A_i, U_j, S, G_ij)`，当前 `G_ij` 含重叠、三轴物理位移、近邻距离、侧别和组织族；正式训练前审计 geometry() 的粗 F1 分辨率、`torch.cdist` 大体素复杂度、患者空间轴与“最近表面距离”是否一致。建议加入小病灶的细尺度几何消融，对不正确的物理距离实现必须修正并写测试。
- R 主监督来自训练裁块中的 SynthSeg + lesion masks 导出的伪关系，不是人工真值；如果 A QC 不可靠，则对应样本的 R loss 真正屏蔽（`r_supervised=False`），而不能只让 A loss 为零。
- `L_III = L_A + L_U + L_S + 1.0(L_host + 0.2L_hard)`；Stage III **80k crops**，backbone lr=3e-5、A/U/S lr=1.5e-4、R lr=3e-4（初始值）；须记录 U 匹配 query 生成 R 的有效训练样本数。
- **门 G3：** R controlled track 给定真实 lesion mask，与同一测试病例的 B0（预测 A + 几何查表）、B0*（SynthSeg + 几何查表，上限/伪参考）对比；host ABA(R) ≥ ABA(B0) 仍只是伪标签工程门。需分别报告 recognition–binding gap、R rescue、harm、side accuracy、各病灶体积层级和序列/来源。端到端轨道用 U 匹配上的病例单报。
- **真正研究结论必须依赖独立 Level R 人工审核**：双阅片/争议仲裁，标注主宿主和左右侧，包含 geometry-conflict 例与匹配 non-conflict controls；盲于模型 R 输出，并同时评估 R 与 B0。临床对错不可由同一 SynthSeg 自动规则自证。此部分可并行准备，但统计只能在 Stage III 锁定后开展。

## 6. 预注册对照、评估数据与研究问题

| 编号 | 条件 | 唯一改变项 | 主要用途 |
|---|---|---|---|
| C0 | 随机初始化 Swin → II → III | 无 Stage I | 证明 Stage I 是否带来因果可归因的增益 |
| C1 | MIM-only Stage I → II → III | 无 contrastive | 分解两种预训练目标的作用 |
| C2 | MIM+contrast Stage I → II → III | 主方案 | Foundation 主线 |
| C3 | C2 主方案但 R→B0 几何查表 | 不用学习关系竞争 | R 是否真正有价值 |
| C4 | C2 保留 R，但冻结或扰动 A 的可信度/几何证据 | 仅改关系证据 | 测试 rescue/harm、视觉与几何的互补性 |

- 同一病人拆分、相同 II/III crop budgets、相同目标定义/调参预算和相同推理后处理；否则不得宣称 Stage I 优于随机初始化。正式比较至少 3 个固定随机种子，patient-level bootstrap 置信区间；若算力无法覆盖，只能报告 pilot 不确定性。
- A：32 类分别报 Dice，13 宿主宏均值（分来源/序列），外加独立人工结构审核；0.80 只是 SynthSeg 蒸馏门，不是超越 SynthSeg。
- U：lesion-level sensitivity@matched FP/case、病人级复核、FROC、按等效球直径 <5mm / 5–10mm / >10mm 分层；来源留出的“新病种”/新中心需**额外独立数据**才称为 OOD 测试。不得以关闭疾病分类头作为泛化证明。
- R：controlled（固定真值 lesion）、end-to-end（预测 lesion）、B0/B0*、rescue/harm、几何冲突子集、Level R human truth。所有伪标签数字标记 NOT_EVIDENCE。
- 对输入条件用源×序列二维表：例如 PDGM T1c、BMSR T1c 同序列同疾病来源去混杂；不可比组合不硬凑，报告 no-overlap。
- **最终成功标准：** G0→G1→G2→G3 均通过 + 独立 Level R 证据支持主结论。若只通过 G0-G3，准确表述为“对已有伪标签和三类已标注异常的工程可行性”。

## 7. 分文件实施任务（按依赖顺序逐项完成）

**T00｜冻结旧规格并新增修订决议**
- 新文件：`docs/superpowers/specs/2026-10-09-anatobind-brain-ssl-first-addendum.md`。
- 不改旧 N1–N18 文档的历史叙述；修订 N11“跳过 Stage I”、N16 的 Stage I 算力和 §9 的 Stage I 文件路径；说明原 Part 1 的代码核查和探针依然是前提。

**T01｜数据 inventory 与锁定拆分**
- 新增 `anatobind/aur/ssl/samples.py`、`scripts/aur_ssl_prepare.py`、`tests/test_aur_ssl_samples.py`。
- 只引用已存在/授权的数据；输出 source/sequence/patient unique rows、partial modalities、单位/affine 信息、hash、split 与 duplicate audit；测试同一 patient 不跨 split、路径不存在时拒绝、测试病人零曝光、未知单位显式拦截。

**T02｜Stage I 图像数据管线**
- 新增 `anatobind/aur/ssl/dataset.py`、`tests/test_aur_ssl_dataset.py`。
- 重用 `anatobind/aur/crops.py`，整卷归一化、按 native grid 裁块、可选患者级配对、两个强度视图、coords/local/valid 一起变换。测试原始 affine 与转置顺序、全 padding、异常体素不被 mask 逻辑偷偷改动。

**T03｜遮挡策略和观测可见性**
- 新增 `anatobind/aur/ssl/masking.py`；必要时**向后兼容**扩展 `anatobind/aur/swin.py` 的 masked encoder 输入。
- 单测：被遮挡原像素任意变化，编码预测不得变化；测试窗口平移、下采样合并、多尺度 valid / visible mask 的传播；测试前景 mask 比例与 deterministic RNG。

**T04｜MIM reconstruction head**
- 新增 `anatobind/aur/ssl/heads.py::MaskedPatchDecoder` 和 `tests/test_aur_ssl_heads.py`。
- 解码 mask patch 的 32 个归一化 voxel intensities，输出 target 与 mask shape 一致；小输入可跑 forward/backward；验证除 masked valid foreground 外 loss 为 0。

**T05｜Contrastive projector & patient-safe positives/negatives**
- 新增 `anatobind/aur/ssl/contrast.py` 和 `tests/test_aur_ssl_contrast.py`。
- 128D projector、两视图 InfoNCE、4 卡 all-gather、按 patient ID 排除假负样本、数值防溢；同患者跨序列只是可选消融。

**T06｜Stage I wrapper、损失和 checkpoint**
- 新增 `anatobind/aur/ssl/model.py`、`anatobind/aur/ssl/losses.py`、`anatobind/aur/ssl/checkpoint.py`、`tests/test_aur_ssl_model.py`。
- 重用同一个 SwinBackbone；分开保存 resume checkpoint 和 backbone-only；配置、数据 hashes、预训练任务版本可回溯；严格兼容下游 `AnatoBindBrain.backbone.load_state_dict`。

**T07｜CPU/GPU 实测探针**
- 新增 `scripts/aur_ssl_probe.py`、`tests/test_aur_ssl_probe.py`。
- CPU 小尺寸 smoke；单卡 B1/B2 和 4 卡 DDP small steps；记录 GPU 名称、显存、梯度检查点、backbone 双视图吞吐、有效 crop/s；不可依据旧 Stage II 探针推算 Stage I，因为 Stage I 有双视图。

**T08｜Stage I 训练**
- 新增 `scripts/aur_ssl_train.py`、`anatobind/aur/ssl/train.py`、`tests/test_aur_ssl_train.py`。
- 支持 seed/DDP/AMP/grad-accum/sampler、resume、精确曝光量、按独立 val 选 ckpt、所有输出目录独立命名、日志不含患者明细。先 pilot，后正式 320k crops。

**T09｜Stage I 冻结表征探针及审计**
- 新增 `scripts/aur_ssl_eval.py`、`anatobind/aur/ssl/eval.py`、`tests/test_aur_ssl_eval.py`。
- probe 标签只来自训练拆分内的校准子集；分别报告正常宿主区分、病灶区域 vs matched normal ROI 可分性、体积层和序列；与随机 backbone 相同探针比较，避免将解码器训练收益误判为 SSL 收益。

**T10｜Stage II 加载与训练契约**
- 新增 `anatobind/aur/train.py`、`scripts/aur_train.py`、`tests/test_aur_training_contract.py`。
- `--stage II --init-backbone <StageI权重>` 必选（主线）；Stage I 未过 G1 拒绝主线训练；诊断性 `--init random` 仅用于 C0 标记消融。检查 A/U/S loss gating、匹配点采样实例覆盖与正确保留体素。

**T11｜Stage III 继承与 R 真值审计**
- 在上述训练入口增加 `--stage III --resume-stage2 <ckpt>`，测试完整参数继承、单独 R 优化器组、r_supervised gate、少于 10mm3 的 crop-sliver 排除、几何/侧别正确、无未来信息泄漏；一并修正物理近邻距离与错误单位。
- 对当前 coarse F1 relation geometry 与多尺度/细网格几何做独立可追溯消融，不在中途改变主要模型定义。

**T12｜完整推理/评估和人工真值接口**
- 新增 `anatobind/aur/eval.py`、`anatobind/aur/infer.py`、`scripts/aur_eval.py`、`scripts/infer_anatobind_brain.py`、`tests/test_aur_eval.py`。
- 整卷滑窗重建 A/U mask，严格 instance 去重/坐标反变换；输出 SynthSeg 值的 `anatomy.nii.gz`、实例 `lesions.nii.gz` 和 `record.json`；失败或未出现宿主时明确写 no_host。
- 增加 Level R review schema：每实例 source/patient_hash/lesion_id/侧别/宿主/两位阅片者/仲裁；受控与端到端独立指标、B0、rescue/harm。评估报告区分 pseudo 与 human evidence。

**T13｜完整验证与复现实验**
- 依次验证 `pytest -q tests/test_aur_*.py`、数据 split/affine/SSL 不泄漏、Stage I DDP probe、I→II checkpoint strict load、II→III checkpoint strict load、真实一卷推理烟雾测试、C0/C1/C2/C3 的 seed-matched 比较。
- 记录版本号、checkpoint hashes、日志路径、各门 G0/G1/G2/G3 的判定和失败原因；任何一门不过不悄悄启动下一阶段。

**T14｜审稿级独立验证与归档**
- 人工真值盲评与患者级统计，给出不能宣称的能力（未知病种、跨机构、跨器官）；更新 `docs/verification/2026-10-09/anatobind_brain_ssl_first/README.md` 和 `STATUS.md`。若模型成功，只能如实按现有病种/扫描器覆盖度限定主张。

## 8. 可执行命令接口（实现任务 T01–T12 后才存在；不得误称现在能运行）

~~~bash
# 0. 在服务器仓库根目录，确认当前分支与 GPU 资源，再运行。
PYTHONNOUSERSITE=1 PYTHONPATH=. python scripts/aur_ssl_prepare.py \
  --aur-samples /data2/congcong/data/FM_data/derived/aur/samples.json \
  --out /data2/congcong/data/FM_data/derived/aur/ssl_manifest_v1/

# 1. 先探针：保留 P0 输出，并确认 4 卡都空闲。
PYTHONNOUSERSITE=1 PYTHONPATH=. python scripts/aur_ssl_probe.py \
  --samples /data2/congcong/data/FM_data/derived/aur/ssl_manifest_v1/samples_ssl.json \
  --out <NEW_P0_DIR>

# 2. Stage I: pilot 8k crop 曝光，核查后主训 320k。
torchrun --nproc_per_node=4 scripts/aur_ssl_train.py \
  --samples <LOCKED_SSL_MANIFEST> --seen-crops 320000 \
  --mask-ratio 0.60 --contrast-weight 0.10 \
  --out <NEW_STAGE_I_DIR>

# 3. Stage II：继承 I 的 backbone。
torchrun --nproc_per_node=4 scripts/aur_train.py \
  --stage II --init-backbone <STAGE_I_DIR>/ssl_stage1_best.pt \
  --samples <LOCKED_AUR_MANIFEST> --seen-crops 240000 \
  --out <NEW_STAGE_II_DIR>

# 4. Stage III：继承整个 II。
torchrun --nproc_per_node=4 scripts/aur_train.py \
  --stage III --resume-stage2 <STAGE_II_DIR>/aur_stage2_best.pt \
  --samples <LOCKED_AUR_MANIFEST> --seen-crops 80000 \
  --out <NEW_STAGE_III_DIR>

# 5. 对锁定 test，仅在 III 完成后评价。
python scripts/aur_eval.py --checkpoint <STAGE_III_DIR>/aur_stage3_best.pt \
  --samples <LOCKED_TEST_MANIFEST> --out <NEW_EVAL_DIR>
~~~

说明：以上是**规定的 CLI 接口契约**，并非已有的可用命令；不要在尚无脚本时执行。所有尖括号路径由操作者替换为新建、尚不存在的路径。训练时使用 CPU nice、线程上限 48、4 张空闲 A800 的约束；不要在本地 Git 提交包含医学数据的 manifest 原始绝对路径及患者标识。

## 9. 严格的阶段验收和停止条件

| Gate | 输入 | 核心核查 | 不过怎么做 |
|---|---|---|---|
| G0 | 数据 inventory | 患者级隔离、真实 affine、数据许可、QC、显存预测 | 停止 Stage I |
| G1 | Stage I checkpoint | masked 泄漏检查、表征不塌缩、SSL + frozen probes、严格载入 | 修复 Stage I；不可默认改成随机初始化主线 |
| G2 | Stage II checkpoint | A 宏 Dice、U sensitivity@FP/case、小病灶覆盖与 64-query 负荷 | 记录失败；必要时提退路决议 |
| G3 | Stage III checkpoint | R vs B0 controlled ABA、端到端与 rescue/harm，独立 Level R | 报告不足，不以伪标签冒充临床有效性 |

**不得自动进入 Stage IV，亦不得因为模型成功输出字符串就声称发现所有异常。**

## 10. 近期最小执行清单（以交付先后而非估计工期排序）

1. 对当前 `fa6e6fd` 分支做测试基线和 P0 审计；先排查现存模型与 NIfTI grid 的潜在 bug；验证目前尚未证明的 A/U/R 效果。
2. T01–T03：锁定 manifest + 实现真正无泄漏的 masked Swin 输入。**这是 Stage I 最重要的第一项工程风险。**
3. T04–T07：完成 MIM、两视图对比学习与 4 卡 probe；对遮挡比率、可见 token、contrastive 负样本和显存做 pilot。
4. T08–T09：完整 Stage I 与 frozen probes，过 G1 导出权重。
5. T10–T11：执行 Stage II、Stage III，分别保存 checkpoint；训练时监控小病灶 U、R 的实体-宿主关联和失效门。
6. T12–T14：做 B0 对照、独立人工审核、统计检验、外部分布评估、记录归档。

**最终研究问题（避免大而空）：** 固定同一个三维 A/U/S/R 模型与同样的标注训练预算，先在无标签脑 MRI 上学习“细节敏感且物理空间一致”的视觉表征，是否能使小病灶实例检出和解剖宿主绑定优于随机初始化与几何查表？只有 C0/C1/C2/C3 + 独立 Level R 能回答这个问题。
