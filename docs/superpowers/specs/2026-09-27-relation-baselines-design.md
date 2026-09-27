# 脑侧关系基线 PR-C（子项目 S1）· 设计规格

> 状态：2026-09-27 与用户逐段讨论后定稿（四段设计各自点头，决定见 §1）。它记录**已经定下的设计**，实施计划另写。
> 上游：`docs/plans/2026-09-22-aur-v2.6-experiment-design-route.md` §3（本体）、§10 / §10.1（基线矩阵、初始配置、训练标签）、§11（几何对等）、§12（Gate R1 统计、嵌套 CV、封存）、§16（schema）、§17（代码结构）、§23（PR-C 清单）、§25 第 9 / 10 项。
> 依赖（全部已有）：病灶注册表 `docs/verification/2026-09-24/gate05/lesions.csv`（1297 条）；折表 `data/level_r/folds.json`；SynthSeg 分割 `derived/synthseg/fastmri_brain/seg_native/`（996 卷，覆盖全部 165 卷）；h5 里的 RSS；封存加载器 `anatobind/eval/level_r_labels.py`。

## 0. 一句话

在 fastMRI+ 脑 FLAIR 的 1297 个小病灶上，把 v2.6 §10 的前五个臂（B0 查表、Bprior 先验、Bgeo+ 强几何、B1 独立候选打分、B2 局部外观）连同外层五折 / 内层选择的评估骨架建起来：所有臂读同一张按版本落盘的特征表，同一候选集，同一份按患者分的折；阶段 1 用伪标签 C1 训练并出折外预测（全部数字标 NOT_EVIDENCE），医生标签回来后同一套代码换标签源做终测。S1 交付的是一条能跑、公平性由构造保证的流水线，不交付关于绑定质量的任何证据。

## 1. 决定记录

| # | 决定 | 一句话理由 |
|---|---|---|
| P1 | 顺序改为"先训练、后读片"：全部学习方法在医生标签存在之前冻结（用户 2026-09-27 拍板） | v2.6 §12.7 封存与 §12.8 "B4 在中期前冻结"自动满足，是最干净的预注册 |
| P2 | 医生阶段分两轮而不是"只做一次终测"：pilot 150 标签 → 判 §10.1 升级规则 → 全集读完封存 → 触发则用外层训练折的医生标签微调（Bgeo+ 同步重拟合）→ 封存折上终测一次 | 伪标签 C1 是输入几何的确定函数，阶段 1 模型约等于 B0；不留第二轮，Gate R1 按构造 NO-GO |
| P3 | 训练池 = 注册表的 1297 个病灶（用户选 A） | v2.6 §10 "same lesion instances"；扩到全部 FLAIR 框会混入另一种病灶群体 |
| P4 | 特征表先建、五个臂都读表（用户选方案 1） | 公平性由构造保证；建表一次 15 分钟 CPU，训练以秒计；医生标签只是换一列 |
| P5 | 新开 `anatobind/relation/` 包放特征表、基线、嵌套 CV、训练器；`lookup.py` 只加脑侧类级 B0，`geometry.py` 只加 16 维对量；指标按 §17 新建 `eval/relation_metrics.py`。这与 §17 "Bprior / Bgeo+ 写进 lookup.py" 不同 | sklearn 模型与训练循环放进查表模块会破坏其单一职责 |
| P6 | C1 是**类级**查表：7 个宿主类各自体素数取 argmax（左右合并、脑室 / CSF 不参与），全零取最小表面距离最近的类；建表时对照 `lesions.csv` 的标签级查表映射，一致率 ≥ 99%，差异逐条列出 | §10.1 的候选集就是 7 类；标签级 argmax 与类级可能在极少数病灶上不同，要看得见 |
| P7 | 候选集 = 最小表面距离 ≤ 15 mm 的宿主类槽（§10.1）；没有任何槽在 15 mm 内时取最近的一个槽；**所有臂**的预测限制在候选集内（候选集外概率置零后归一） | 一视同仁；B1 的 `present` 掩蔽本来就这么做 |
| P8 | 侧别按病灶体素上 SynthSeg 左右标签族的多数判 image_left / image_right / midline；`image_left` = 列号小的一侧；实现时用测试对齐读片工具的屏幕左右 | 不依赖中线估计；与 Level R 表单同一约定（记屏幕侧，不记患者侧） |
| P9 | Bprior 的"粗位置" = 侧别 × 行三分 × 层三分，均在图像坐标系里定义，不作前后 / 上下解剖断言 | 只供先验计数；解剖方向在 RSS 帧里未核实 |
| P10 | 图像小块按最大档存（36 mm × 36 mm @ 0.75 mm = 48 × 48，中心层 ±1 共 3 层），训练时裁到 22 / 32 / 42 px（名义 16 / 24 / 32 mm，实际 16.5 / 24 / 31.5 mm）；同窗口另存 int8 class-map 小块 | §10.1 的内层范围要三档边长与 {1, 3} 层；class-map 供 S3 的 B4 派生候选掩膜与距离图 |
| P11 | 内层选择的平局规则：阶段 1 内层准确率接近全 1 时，优先 §10.1 初始配置，再取更小的配置；平局过程写进报告 | 免得以后误读成"选出了最优" |
| P12 | 阶段 1 不做噪声视图增广、不做 B3 / B4、不做 B5 | 噪声视图要脑侧退化导出（S3）；B3 / B4 是 PR-D；B5 要医生标签与"最好的解剖"定义 |
| P13 | 阶段 2 钩子现在就进代码：`--labels R` 从封存加载器取训练折医生标签、剔除 not_a_lesion、集合标签走集合损失；B1 / B2 可从阶段 1 权重初始化；Bgeo+ 重拟合。现在只有测试走这条路 | P2 的第二轮不该等到医生标完再写代码 |
| P14 | §12.8 的 700 例中期无效性分析在"先训练后读片"下只对阶段 1 模型有意义；本规格不实现，留给 S3 在读片前决定做或不做 | 它是非约束性条款；升级触发后的模型在中期不存在 |
| P15 | 疾病层 D（"疑似 xxx 疾病"）定义为**整个检查一个印象**，真值用 fastMRI+ 研究级标签（正常 94 / 小血管病 150，均为 FLAIR）；不在 S1 范围，记在此处供 S5 | 用户 2026-09-27 拍板；有印象标签的检查与有框的检查几乎不重叠（两者都有的只 12 个），所以 D 层依赖 S2 的检测器 |
| P16 | 训练期六个子项目及顺序：S1 本规格 → S2 脑侧小病灶检测器 → S3 PR-D（B3 / B4）→ S4 脑侧解剖层（三件事：从 SynthSeg 输出蒸馏一个自有的 FLAIR 解剖分割器，推理时不再调外部工具、退化图上可测鲁棒性；SynthSeg 皮层分区 → 脑叶，侧别照旧；用医生的 A_local_quality 作 SynthSeg 在病灶处的验证，整卷级验证在 5 mm FLAIR 上没有真值，明说这是上限）→ S5 疾病印象（含拼句：输入 S2 的框、S1 / S3 的宿主、S4 的侧别与脑叶、医生微调后的类型头，输出"<侧别><脑叶><宿主>上出现<异常类型>，疑似<印象>"）→ S6 膝侧 nnDetection（独立，另写计划）；S1 定稿后立即写 S2，两者并行跑 | 用户 2026-09-27 批准；S4 的扩展与 S5 的拼句是同日第二轮目标对齐审查的结论：此前六段做完，脑侧"识别解剖"仍是外部工具，且没人负责把整句拼出来 |

## 2. 与核心目标的关系（2026-09-27 讨论结论）

用户的三个子目标是一条四层链：A 感知解剖结构；U 找到异常；R 异常绑定到解剖结构（"在 xxx 解剖结构出现 xxx 异常"）；D 由异常推疑似疾病。v2.6 第一篇主实验只严格验证 R（给定框、SynthSeg 解剖 → 宿主），A 脑侧外包、U 脑侧不在主实验、D 不存在。S1 只做 R 层的基线与评估骨架，其余偏离由 S2–S6 收回（§1 P16）。

S1 的数字在医生标签到来前没有证据价值：C1 由 §11 的几何量确定，Bgeo+ 会把它几乎原样拟合出来，B1 的图像分支拿不到梯度信号。这是 v2.6 §10.1 自己写明的风险，也是 P2 两轮制的理由。阶段 1 报告必须把这件事写成两项健全性检查（§6.7）。

## 3. 系统组成

```
scripts/build_relation_table.py      注册表 + SynthSeg + h5 RSS → derived/relation/v1/{table.csv, patches.npz, manifest.json}
scripts/run_relation_baselines.py    五个臂的外层五折 / 内层选择 → runs/relation/<run_id>/preds/<臂>.csv + run.json
scripts/eval_relation_baselines.py   折外预测 + 标签源（C1 | R）→ docs/verification/<日期>/relation_baselines/{REPORT.md, tables.csv, output.txt}
anatobind/relation/table.py          建表、读表、槽序与列名常量、manifest、建表检查
anatobind/relation/labels.py         标签源 C1 / R → {lesion_id: (可接受集合, 单元素, 分层)}
anatobind/relation/baselines.py      B0（表上的 C1 规则）、Bprior 四变体、Bgeo+ 三学习器与网格
anatobind/relation/encoder.py        E_u、小块裁剪、增广（翻转与 Δx 取反同步、±10% 强度）
anatobind/relation/models.py         B1（包装 model/relation.py 的 IndependentCandidateHead，M = 1）、B2（E_u + 线性头）
anatobind/relation/cv.py             外层折（folds.json）、内层按患者五折、配置范围、平局规则、折外预测存取
anatobind/relation/train.py          B1 / B2 训练循环（阶段 1；阶段 2 钩子 --labels R / --init-from）
anatobind/eval/geometry.py           + 16 维对量、地标距离、标签族侧别
anatobind/eval/lookup.py             + 脑侧类级查表（C1 规则）
anatobind/eval/relation_metrics.py   集合值正确性、准确率族、rescue / harm / net rescue、患者 bootstrap、McNemar、分层、Gate R1 函数
tests/test_relation_*.py             不读 /data2
```

## 4. 特征表（`scripts/build_relation_table.py` → `derived/relation/v1/`）

### 4.1 输入与网格约定

- 注册表 `lesions.csv` 用 `anatobind.level_r.registry.load_registry` 读（status = ok 的 1297 条，lesion_id 唯一）。逐层成员框沿 Level R 导出的同一路径取回：`level_r.export.small_lesion_rows` → `merged_lesions` → `match_registry`，保证与读片工具看到的是同一批框。
- 每卷：SynthSeg 分割 `seg_native/<file>_seg.nii.gz`，网格 (col, row, slice)；h5 `reconstruction_rss` 是 (slice, row, col)，转置到分割网格后使用；几何用 `fastmri_knee.volume_geometry(h5)`，并断言分割形状 = (n_cols, n_rows, slices)（与 `scripts/brain_frame.py` 相同）。
- 折：`data/level_r/folds.json` 的 `patient_fold` 就是外层五折；每名患者一卷。
- 病灶体素 = 逐层成员框的并集（`geometry.member_rects`），与 Gate 0.5 一致。

### 4.2 病灶级列（每病灶一行）

| 列 | 定义 |
|---|---|
| lesion_id, file, patient_id, fold | 注册表与折表 |
| stratum_geometry, stratum_series, band, is_3mm | 注册表；band 用 `registry.distance_band` |
| lesion_type | 数据集标签映射：Nonspecific white matter lesion → `nonspecific_wm_lesion`，Lacunar infarct → `lacunar_infarct` |
| n_slices, inplane_mm, x0, y0, x1, y1, z0, z1 | 注册表（RSS 帧） |
| spacing_col_mm, spacing_row_mm, spacing_slice_mm, slice_thickness_mm | 卷几何；slice_thickness_mm = spacing_slice_mm（§11 两列并存） |
| extent_x_mm, extent_y_mm, extent_z_mm | (x1 − x0)·spacing_col、(y1 − y0)·spacing_row、(z1 − z0 + 1)·spacing_slice |
| volume_mm3 | Σ 成员框面积 × 层厚 |
| centroid_col, centroid_row, centroid_slice | 病灶体素的均值（体素坐标，浮点） |
| side | P8：病灶体素上宿主类左右标签族的多数（左族 2, 3, 10, 11, 12, 13, 26, 7, 8, 17, 18, 28；右族 41, 42, 49, 50, 51, 52, 58, 46, 47, 53, 54, 60；脑干 16 不计）；两族各 ≥ 40% 记 midline；无宿主体素时按最近宿主体素的族 |
| row_third, slice_third | 中心层脑掩膜（seg > 0）行范围三分之 top / middle / bottom；卷内脑掩膜层范围三分之 inferior / middle / superior（图像坐标系，见 P9） |
| coarse_location | `f"{side}\|{row_third}\|{slice_third}"` |
| d1_mm, d_interface_mm, delta_d_mm | 用 `geometry.interface_margin` 重算，建表检查要求与注册表相等（|Δ| < 1e-6） |
| dist_cortex_mm, dist_ventricle_mm | 皮层槽的最小表面距离；到侧脑室标签 (4, 43, 5, 44) 的 EDT 最小值 |
| c1_class, c1_slot, c1_source, c1_overlap | P6：类名、槽号、`overlap` / `nearest`、argmax 类的体素占比 |
| registry_lookup_class | `host_lookup_parenchyma` 经 `HOST_CLASSES` 映射，供一致率检查 |

### 4.3 槽级列（7 个槽，槽序固定为 `geometry.CLASS_NAMES`，列名前缀 wm_ / cortex_ / thalamus_ / bg_ / brainstem_ / cerebellum_ / odg_）

每槽 10 列：`in_volume`（类在卷内）、`candidate`（min_surface ≤ 15 mm，P7）、`dx_mm`、`dy_mm`、`dz_mm`（病灶质心到该类最近体素的向量，mm，取自类距离图 `return_indices` 在质心体素处的索引）、`centroid_distance_mm`（该向量的模）、`min_surface_mm`（病灶体素上类距离图的最小值，重叠为 0）、`signed_surface_mm`（无重叠 = min_surface；有重叠 = −(病灶内属于该类体素到类边界距离的最大值)）、`ioa`（病灶体素属于该类的比例）、`soft_overlap`（病灶体素 exp(−d/1 mm) 的均值）。

距离一律封顶 30 mm。类不在卷内的槽：in_volume = candidate = False，距离 30，dx/dy/dz = 0，ioa = soft = 0。

§11 的 16 维 = 每槽 8 个对量（dx, dy, dz, centroid_distance, signed_surface, min_surface, ioa, soft_overlap）+ 8 个病灶级量（extent_x/y/z, volume, spacing_col/row/slice, slice_thickness）。脑侧附加 = d_interface, delta_d, dist_cortex, dist_ventricle, side。

### 4.4 图像小块（`patches.npz`）

- 键：`lesion_id` (N,)、`image` (N, 3, 48, 48) float16、`mask` (N, 3, 48, 48) bool、`classmap` (N, 3, 48, 48) int8、`meta`（JSON 字符串：window_mm 36、pixel_mm 0.75、slices "centre−1, centre, centre+1"）。
- 中心 = (centroid_col, centroid_row)，中心层 = round(centroid_slice)，相邻层越界时复制边缘层。
- image = 按卷 z-score（均值、标准差取 seg > 0 的体素）的 FLAIR，从本卷面内间距双线性重采样到 0.75 mm；mask = 该层成员框；classmap = `host_class_map(seg)`（0..7），二者最近邻重采样。
- 训练时裁剪 22 / 32 / 42 px（P10），中心对齐。

### 4.5 C1（P6）

类级查表：`c1_class = argmax_c count(病灶体素 ∈ HOST_CLASSES[c])`，`c1_source = overlap`；全零时 `c1_class = argmin_c min_surface_mm`，`c1_source = nearest`。可接受集合 = {c1_class}。`lookup.py` 新函数 `brain_class_lookup(seg, spacing, rects)` 实现，`table.py` 调用。

### 4.6 落盘与 manifest

- `table.csv`（病灶级 + 展平的槽级列）、`patches.npz`、`manifest.json`：version、built_at、git_commit、registry 路径与 sha256、folds sha256、seg_root、kspace_root、参数（cap 30、candidate 15、window 36、pixel 0.75、slices 3、soft σ 1）、计数（病灶、患者、每折患者 / 病灶、C1 分布、c1_source 分布、无候选取最近的病灶数）、`c1_agreement`（一致率 + 不一致的 lesion_id 列表）。
- 目标目录已存在就拒跑（用户硬规矩：不覆盖已有数据文件）。nice 19 单进程；预计 15 分钟 CPU。

### 4.7 建表检查（任一失败即中止，不写文件）

1297 行；lesion_id 集合 = 注册表集合；每名患者都在折表里且折按患者不重叠；d1 / d_interface / delta_d 与注册表相等；C1 一致率 ≥ 0.99；每个病灶 ≥ 1 个候选；每个小块 image 非常数。验证记录 `docs/verification/<日期>/relation_table.md`：计数、C1 分布、一致率与差异清单、每折患者数、命令与原始输出。

## 5. 五个臂

共享：同一张表、同一候选集（P7）、同一份折、同一标签源。输出统一为 8 个概率（7 槽 + none），候选集外置零归一。

### 5.1 B0 查表

就是 C1 规则；输出 one-hot。阶段 1 对 C1 准确率恒为 1，报告明说。

### 5.2 Bprior 先验（§10 四变体，全做）

在外层训练折上数频率、加一平滑（每类 +1）：`majority`；`type`（lesion_type）；`type_side`；`type_side_location`（coarse_location）。无超参，只跑外层。

### 5.3 Bgeo+ 强几何

- 每病灶一个向量（约 80 维）：7 槽 × [candidate, dx, dy, dz, centroid_distance, signed_surface, min_surface, ioa, soft_overlap] (63) + 病灶级 8 维 + 脑侧附加 4 维（d_interface, delta_d, dist_cortex, dist_ventricle）+ side one-hot (3) + lesion_type one-hot (2)。
- 标准化（均值 / 方差）只在外层训练折上拟合。
- 三个学习器与内层网格：多项 logistic 回归 C ∈ {0.01, 0.1, 1, 10}；`sklearn.ensemble.HistGradientBoostingClassifier` learning_rate ∈ {0.03, 0.1} × max_depth ∈ {3, 6}；`sklearn.neural_network.MLPClassifier` hidden (64, 64)，alpha ∈ {1e-4, 1e-3}，early_stopping。不加 XGBoost。
- 训练里没出现的类不可预测；候选掩蔽后归一。
- "Bgeo+ 最优学习器"（§6.3 的默认对照）= 三个学习器各自内层选完超参后，内层集合值准确率均值最高者；平局（差 < 1e-3）按 LR、HGB、MLP 的顺序取先者（简单优先），平局过程与 P11 一样写进报告。

### 5.4 共享病灶编码器 E_u（§10.1）

输入：3 层 × 2 通道（image, mask）合成 6 通道 2D 张量（1 层配置时 2 通道）；4 个卷积块 32-64-64-128（3×3 卷积 + BatchNorm + GELU + 2×2 池化）+ 全局平均池化 → 128 维。配置范围：边长 {22, 32, 42} px × 层数 {1, 3}，从初始 (32, 3) 单因素变化 → 4 个配置：(32, 3)、(22, 3)、(42, 3)、(32, 1)。

### 5.5 B2 局部外观

E_u + 线性头 → 8 logits。不接触表里的任何几何列。

### 5.6 B1 独立候选打分

复用 `IndependentCandidateHead`（M = 1，K = 7，d_model = 64）：`a_embed`[k] = MLP(类嵌入 32 维 ⊕ 该槽几何 g_k) → 64；`u_embed` = Linear(E_u 128 → 64)；`geo` = g_k（每槽 26 维：9 个槽量 + 8 个病灶级 + 4 个脑侧附加 + side 3 + type 2；与 Bgeo+ 看到的是同一组数）；`present` = candidate。none 头保留接口，阶段 1 没有正样本。B1 与 Bgeo+ 的几何对等在测试里断言（同一 lesion 的两种排列来自同一列）。

### 5.7 训练规则（B1、B2）

交叉熵；标签为集合时损失 = −log Σ_{c∈Y} p_c。AdamW lr 1e-3、wd 1e-4、批 64、最多 40 epoch；内层验证折上按集合值准确率早停（耐心 8）；外层重训时无验证折，训练轮数取内层各折最佳轮数的中位数。增广：左右翻转（小块翻转同时 dx 取反、side 的 left/right 互换）、强度 ×U(0.9, 1.1)。种子：每个 (外层折, 配置) 固定 seed 0。设备 `CUDA_VISIBLE_DEVICES=0`，CPU 亦可。

### 5.8 内层选择与平局（P11）

外层训练折内按患者分 5 折（种子 0 打乱患者后均分）；Bgeo+ 的网格与 B1 / B2 的 4 个配置都按内层集合值准确率均值选；平局（差 < 1e-3）按 P11。选定后在整个外层训练折重训、预测外层测试折。每臂配置数 ≤ 12（v2.6 上限；Bgeo+ 三个学习器合计 10 个，B1 / B2 各 4 个）。算力：B1、B2 各 5 × (5 × 4 + 1) = 105 次小训练，合计 210 次，GPU 0 一小时内；Bgeo+ 的 sklearn 拟合以秒计。

### 5.9 阶段 2 钩子（P13）

`run_relation_baselines.py --labels R`：训练标签来自 `level_r_labels.load_train_labels(k)`（裁定后 primary_host 与 acceptable_hosts；not_a_lesion 剔除；集合标签走集合损失）；`--init-from <run_id>` 用阶段 1 的 B1 / B2 权重初始化；Bgeo+ 与 Bprior 在医生标签上重新拟合。测试用合成封存文件覆盖。

## 6. 评估骨架

### 6.1 折外预测

每臂每病灶：8 个概率、所选配置、外层折号、run_id → `runs/relation/<run_id>/preds/<臂>.csv`；`run.json` 记 manifest 哈希、配置范围、平局记录、种子、git commit。`runs/` 已 gitignore。

### 6.2 标签源（`relation/labels.py`）

统一结构 `{lesion_id: {"acceptable": set, "singleton": bool, "strata": {...}}}`。
- `C1`：表的 c1_class；集合单元素；所有数字标 NOT_EVIDENCE。
- `R`：`load_test_labels(k, unblind=True)` 汇总五折，只允许 `scripts/eval_relation_baselines.py --labels R --unblind` 调用；not_a_lesion 剔出主终点、单独计数；封存清单不全则拒跑；每次访问由加载器写日志。开发期间任何脚本拿不到测试折医生标签。

### 6.3 指标（`eval/relation_metrics.py`，§12.2）

集合值正确性（argmax ∈ Y）；全体准确率；singleton rate；单元素子集的准确率、macro-F1、balanced accuracy；top-2 宿主准确率；混淆矩阵；rescue / harm / net rescue（默认对照 = 内层选出的 Bgeo+ 最优学习器；同时报对 B0、Bprior 最优变体、B2）。

### 6.4 推断（§12.3）

患者级聚簇 bootstrap：按患者有放回重采样 10000 次、种子 0（沿用 `eval/g2.py` 的写法），给准确率差与 net rescue 的 95% CI；辅助 paired McNemar（精确 / mid-p）。

### 6.5 分层

d_interface 四档；四个实测几何层；200/201 对其余系列；lesion_type；3 mm 层厚卷单列；net rescue 随 Δd 的分箱曲线（分箱边界 0 / 1 / 2 / 4 / 8 mm）带 bootstrap 带（§12.9）。按读者 A_local_quality 的 A-good / A-bad 分层只在 R 标签下可用，接口留着。

### 6.6 Gate R1 函数

`gate_r1(preds_model, preds_comparators, labels)` → §12.4 四条（net rescue 对 Bgeo+ 的 CI 下限 > 0；模型 > Bprior；> Bgeo+；> B2）及"全体人群成立"。S1 里对 B1、B2 各演练一次；S3 的 B4 直接调用。

### 6.7 阶段 1 报告必带的健全性检查

1. 每个臂与 C1 的一致率：预期 Bgeo+ 接近 1、B2 明显低——这是"伪标签是几何的函数"的书面证据。
2. 内层选择的平局记录：配置是按 P11 定的，不是选出来的。

### 6.8 脚本输出

`eval_relation_baselines.py --run <run_id> --labels C1|R [--unblind] --out <dir>` → `REPORT.md`（数据与折；各臂选定配置与平局记录；全体与单元素子集指标表；分层表；net rescue 与 CI；Gate R1 演练；健全性检查；每个数字附命令）、`tables.csv`、`output.txt`。`--labels R` 是预注册里"只跑一次"的那一步。

## 7. 测试（每个函数先写测试；不读 /data2）

- 几何：合成分割（沿用 `tests/test_geometry.py` 的 (col, row, slice) 小卷）上手算 dx/dy/dz、signed_surface、ioa、soft_overlap、侧别标签族、地标距离、封顶与缺类哨兵。
- 类级 C1：与标签级 argmax 相同与不同的构造例各一。
- 建表：合成分割 + 合成 RSS + 合成注册表 / 折表跑通建表；每条建表检查各一条失败用例；目标目录存在拒跑；小块中心与边缘层复制。
- 臂：Bprior 四变体的计数与平滑；Bgeo+ 向量排列与 B1 的 geo 张量来自同一列（对等断言）；候选掩蔽；翻转增广同步取反 dx 与互换 side；集合损失 = −log Σ p；none 头无正样本时训练不崩。
- CV：内层按患者不重叠；平局规则；折外预测覆盖全部病灶且每病灶恰一次；重训轮数 = 内层最佳轮数中位数；种子固定则结果可复现。
- 指标：rescue / harm / net rescue、集合值正确性、singleton rate、bootstrap 确定性、McNemar 的手算例子；分层键完整。
- 标签源：合成封存文件（复用 `tests/test_level_r_labels.py` 夹具）下 `--labels R` 走通；缺清单拒跑；不带 unblind 拿不到测试折；not_a_lesion 剔除计数。
- 端到端：几名患者、几十个病灶的合成表跑通 run → eval，产出 REPORT.md。

## 8. 分支、执行、算力

- 分支 `build/relation-baselines`（自 main 1f06d0f），工作树 `../foundation_model-relation`；S2 另开自己的分支与工作树。S1 结束合回 main、打 tag `handoff/<日期>-relation-baselines`；不 push 除非用户说。
- 执行方式沿用 PR-B：子代理逐任务、先写测试、每个任务独立评审、台账在工作树 `.superpowers/sdd/`（gitignore）。
- 提交作者用仓库本地配置，消息英文，不留 AI 痕迹。不删任何东西。
- 建表 nice 19 单进程；B1 / B2 钉 GPU 0；导出类任务并发 ≤ 4。

## 9. S1 完成的判据

1. 1297 个病灶的特征表建成，manifest 带 git commit 与注册表 sha256；`relation_table.md` 记 C1 一致率 ≥ 99% 与差异清单。
2. 五个臂在外层五折上对全部 1297 个病灶有折外预测；配置范围与平局记录在 `run.json` 与报告里。
3. 阶段 1 报告（C1，全部 NOT_EVIDENCE）含全部指标、分层、两项健全性检查；Gate R1 函数在 B1、B2 上演练过。
4. `--labels R` 路径在测试里用合成封存文件走通；缺清单拒跑；不带 unblind 拿不到测试折。
5. 全量测试通过（现有 612 加新增），`.github/workflows/tests.yml` 不改。
6. 文档：本规格、实施计划、`relation_table.md`、`relation_baselines/REPORT.md`、CLAUDE.md 代码地图、v2.6 §25 第 11 项（§1 的 P1–P3、P5、P11、P14–P16，即 §11 的偏离清单加训练期六个子项目）、STATUS；合回 main 并打 tag。
7. 不删任何东西；`runs/` 不入库；`derived/relation/v1` 存在就拒绝覆盖。

## 10. 明确不做

B3 / B4（S3）；噪声视图增广（S3）；B5 上限（终测前另定）；§12.8 中期分析（P14）；脑侧检测器（S2）；脑侧解剖层——自有解剖分割器、侧别 / 脑叶、SynthSeg 验证（S4）；疾病印象与整句拼装（S5）；膝侧任何事（S6）；任何关于绑定质量的主张。

## 11. 与 v2.6 的偏离清单

| 处 | v2.6 原文 | 本规格 | 依据 |
|---|---|---|---|
| §17 | Bprior / Bgeo+ 写进 `eval/lookup.py` | 新包 `anatobind/relation/` | P5 |
| §10.1 | 预训池 = 非测试患者的全部卷，含无小病灶标注的卷 | 只用注册表 1297 个病灶 | P3 |
| §19 步骤 6–7 | pilot 与升级规则判定在模型规格冻结之前 | 模型先冻结，pilot 与升级规则在医生阶段内判 | P1、P2 |
| §12.8 | 700 例中期无效性分析 | 不实现，S3 决定 | P14 |
| §10.1 | 内层按集合值准确率选配置 | 保留，但阶段 1 平局按 P11 定并记录 | P11 |

这些偏离在 S1 的文档任务里写进 v2.6 §25 第 11 项。
