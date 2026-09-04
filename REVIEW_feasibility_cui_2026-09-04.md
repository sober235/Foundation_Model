# AnatoBind-MRI(`AnatoBind-MRI_cui.md` 版)可行性评审

评审日期:2026-09-04(同日二次复核修订,修订记录见 §8)
评审依据:`AnatoBind-MRI_cui.md`(未提交)、已提交的 `RESEARCH_PLAN.md` v1.1、`/data0/congcong/data/FM_Data` 逐文件实测、本机算力实测。
评审口径:方案里的每个组件与每个实验,能否在**盘上现有数据 + 本机算力**下按原文执行;不能则给出最小修改。

---

## 0. 结论

**总判定:方向可行,但 `_cui` 文档不能照原样执行。** 核心科学赌注(显式关系建模 + k-space 物理干预 + 关系可观测性)在 SKM-TEA + fastMRI 上有完整闭环的数据基础,算力充足;但文档有 8 处与盘上数据或物理事实不符,其中前 4 处会直接让对应实验做不出来或做出来不可信。

照原样不成立的 8 处(按严重程度):

| # | 文档写法 | 实测事实 | 后果 | 修法 |
|---|---|---|---|---|
| 1 | 脑侧解剖实体 left frontal / brainstem / ventricle,来源 TotalSegmentator MRI | TotalSeg-MRI 脑只有 1 整类;PDGM 的 parenchyma seg 是二值 mask;HCP 无 aseg;OASIS freesurfer 目录为空。**全库脑细分解剖标签为零**,且本机未装 SynthSeg/FreeSurfer(ANTsPy 0.6.3 在 BBDM 环境,atlas 配准可用) | 脑侧所有 R、E 实验没有 A | 先建 SynthSeg 伪标签管线;解剖粒度降到 SynthSeg 能给的(半球 × 皮层/白质/脑室/深部核团/脑干/小脑);脑叶级只在各向同性库上用 atlas 配准补 |
| 2 | 运动仿真 `y'_t(k) = exp(-i2πk·Δr_t) · y_t(R_t k)` | 三个协议全是多线圈(脑 16–20、膝 15、SKM 8/16 线圈);物体动而线圈不动,该式只对单线圈刚体成立 | "物理干预"名不副实,审稿人一眼看穿 | 图像域逐 shot 刚体变换 → 用线圈图重编码 → 按 shot 拼 k-space(v1.1 §5.2 原写法);SKM-TEA 自带 ESPIRiT 图,fastMRI 用 BART 从 ACS 估 |
| 3 | 关系真值定义为重叠率 area(M_U ∩ M_A) / area(M_U),baseline 列表无 seg-then-lookup | SKM-TEA 每条标注**自带标注者指定的 tissue_id**,不是重叠率反推;476 例中积液 117 例 tissue_id = −1(无宿主),韧带 39 例的宿主 ACL/PCL **不在分割类里**,但 SKM-TEA 对 ACL/PCL 也没有独立解剖监督 | 重叠率定义把真值送给平凡管线,又浪费了现成的显式关系标签;分割外那 157 例既考不倒 baseline 也撑不起主张(积液只考"该不绑时不绑",韧带只有 39 例且关系模型自己也缺 ACL/PCL 实体) | R* 以 tissue_id 为主真值;seg-then-lookup 进 baseline;**主判据只用分割内 319 例、只在损坏条件下比**;积液作弃权分析,韧带作扩展分析(需外部韧带分割监督) |
| 4 | E 只给排序约束,无真值定义 | 若 E 训成"我的 R 会不会错",就是 learned failure prediction(ConfidNet 一族),Exp 5 只比 softmax/entropy 说明不了独立贡献 | E 贡献立不住 | 主真值用物理保真度(见 §3.7),模型相对口径作第二真值;baseline 加 learned failure prediction |
| 5 | 数据表:BraTS 2021、PI-CAI、MR-ART | 三者都不在盘。PDGM 501 例中 298 例带 BraTS21 ID(262 在 BraTS21 训练集、36 在验证集),203 例不在;PI-CAI/MR-ART 未下载 | 前列腺外测、脑侧 sim-to-real 外验做不了;任何 BraTS 预训练权重在 PDGM 上有 262 例污染 | BraTS→PDGM(如实写"重叠 298 例");前列腺→换 SPIDER 脊柱或砍;脑侧真实运动外验→改膝侧 KMAR;评估 PDGM 时不用 BraTS 训练过的权重,或剔除那 262 例 |
| 6 | KMAR-50K "用于真实 paired motion calibration"(参与训练) | Exp 4 的 held-out 组合正是 Knee+Motion | 训练泄漏,组合泛化结论作废 | KMAR 只做测试(v1.1 口径) |
| 7 | 变尺寸 3D Swin 直接吃 SKM-TEA 512×512×160 | patch (2,4,4) 后 131 万 token/卷,是 fastMRI 脑的 26 倍 | 显存与吞吐不可持续 | 面内温和重采样到 0.625 mm(文档允许"极端 spacing 温和 resampling"),训练裁块、推理滑窗 |
| 8 | fastMRI 脑按 3D 体处理 | 16 层 × 5 mm ≥ 80 mm 的部分覆盖(转换 DICOM 不带层间隙信息,80 mm 是下限),patch 2 再降采三次后 z 只剩 1 | 事实上是 2.5D;每卷解剖实体随机缺席 | 文档明说 2.5D;A 解码器带 no-object;实体存在性也算指标 |

KMAR 的配对对齐在首版评审里被列为第 9 处,复核后撤回:抽查 120 对仅约 5% 头信息不一致,多数配对可直接用(见 §1.1、§3.3)。

**核心可行路径**(数据与算力均已就位):SKM-TEA 为四齐考场(A 分割 + U 3D 框 + R tissue_id + raw k-space/ESPIRiT 图/Poisson 掩膜),fastMRI 为规模与 S 引擎(脑 5 对比度 4468 卷 raw,膝 1172 卷 raw,fastMRI+ 标注卷 100% 有 raw),KMAR 为膝侧真实运动测试端,PDGM/BMSR/ISLES 为脑侧第二战场(仅图像域干预)。

---

## 1. 评审依据:盘上数据与算力(实测)

### 1.1 数据

| 数据 | 状态 | 与方案变量的对应 |
|---|---|---|
| SKM-TEA raw | 155 个 h5 已解压;kspace 512×512×160×2echo×8coil(8 例 16 coil),ESPIRiT maps,Poisson 4x–16x 掩膜,target | T(干预)、S 退化(全 3T qDESS) |
| SKM-TEA 标注 v1.0.0 | 86/33/36 scans,242/104/130 = 476 个 3D 框,16 细类 4 族,每条带 tissue_id、confidence、labeler;155 scan = 155 个不同受试者,无跨 split 重复 | U_B、**R 显式真值** |
| SKM-TEA 分割 | 155 nii,标签 0–6(髌/股/胫内外侧软骨、内外侧半月板),与 raw target 同网格 | A(膝)、侧别 |
| fastMRI raw | 本地脑 train 4468、膝 train 973 + val 199;共享盘脑 val 1378/test 558/challenge 565 | T、S(AXT2 2678/T1POST 949/FLAIR 344/T1PRE 250/T1 248;1.5T/3T) |
| fastMRI DICOM | 脑 4469 卷(16 层,5 mm,FOV 220,矩阵 320/384,面内 0.69/0.57 mm),膝 1172 卷(33–38 层,3 mm,FOV 140,面内 0.44 mm) | 快速原型 |
| fastMRI+ | 脑 8213 个逐层 2D 框/997 卷/30 类(只标 FLAIR/T1/T1POST);膝 16167/974/22 类;**脑 85 卷带真实伪影标签**(Possible artifact 505 个定位框、Motion artifact 33 个 study-level),膝 13 卷;**全部标注卷有 raw** | U_B(2D)、U_Q 真实 |
| KMAR-50K | part2 已解压 641 对,part1 zip 698 对,测试 204 卷;典型 256×256×23,0.625×0.625×3.6 mm,PD TSE fs,1.5T/3T;抽查 120 对:6 对体素间距不一致、1 对矩阵不一致,逐层相关测试 3 对中 2 对同层对齐、1 对错开一层 | U_Q 真实、E 真实端(单档);多数配对可直接用,少数须配准 |
| UCSF-PDGM v5 | 501 例/495 完整,9 序列 + tumor seg(0/1/2/4),240×240×155 @1 mm,剥颅 BraTS 空间;WHO 4/3/2 级 = 402/43/56;298 例与 BraTS21 重叠 | U_B(脑),无 raw |
| UCSF-BMSR v1.3 | zip 未解压,461 例 TRAIN,多灶 seg | U_B 多实体,无 raw |
| ISLES-2022 | zip 未解压,250 例 DWI/ADC/FLAIR + mask | U_B(卒中),无 raw |
| HCP | 1113 例 T1w+T2w 0.7 mm 剥颅,无 aseg | SSL、正常谱 |
| TotalSeg-MRI v2.0.0 | zip 未解压,616 例 51 类(脑 1 类;有 prostate/femur/hip/vertebrae/discs;无膝软骨半月板) | A(躯干/骨盆/脊柱) |
| SPIDER | 447 图 + 447 mask + 逐椎间盘退变分级 | 候选 held-out 解剖格 |
| AMOS22 | 60 例 MRI 有标签 | A(腹部,小) |
| 不在盘 | MR-ART、BraTS 全量、PI-CAI、K2S、CMRxRecon | — |

### 1.2 算力与工具

- GPU:8 × 80 GB(A800 ×6,A100 ×2),评审时 5 张空闲;CPU 112 核;RAM 1 TB;/data0 余 6.1 T。
- 已有:BART(ESPIRiT/重建),sigpy,MONAI 1.2(GR_pt1.10)/1.5(nvgen),nnunetv2 与 dicom2nifti(nvgen),ANTsPy 0.6.3(BBDM,可做 atlas 配准),SimpleITK(多环境),pydicom。
- 缺:SynthSeg / FreeSurfer / FastSurfer,全机未装(find 深度 6 无结果,FREESURFER_HOME 未设)。

---

## 2. 逐组件可行性

判定符号:✅ 按原文可行;🔧 可行但须改;❌ 现有数据做不了。

| 组件 | 判定 | 依据与修法 |
|---|---|---|
| 变尺寸 3D Swin(不 resize,只 pad + valid mask,size bucket) | 🔧 | 几何跨度实测(见 §2.1)充分支持"不统一 resize";但 SKM-TEA 全分辨率 131 万 token 不可持续,须面内重采样到 0.625 mm + 裁块训练;fastMRI 脑 z 方向坍缩为 1,须承认 2.5D |
| 物理坐标 3D RoPE + 局部归一坐标 | ✅ | 所有库都有体素间距;mm 坐标以体积中心为原点即可;跨器官"尤其重要"是断言不是证据,须消融 |
| Patch stem (2,4,4) vs 4³ | ✅ | 5 mm/3 mm 层厚数据占多数,(2,4,4) 合理;消融可做 |
| A 膝 | ✅ | SKM-TEA 6 类分割含内外侧,直接支持侧别绑定;fastMRI 膝与 KMAR 无解剖标签,须用 SKM 训练的分割模型迁移(qDESS→PD TSE 域差,需人工抽检) |
| A 脑 | ❌→🔧 | 全库零标签;装 SynthSeg,粒度降级;fastMRI 脑至少 80 mm 的部分覆盖,SynthSeg 预期可跑但脑叶 atlas 配准不可靠(ANTsPy 在库) |
| 脑侧域差 | 🔧 | PDGM/BMSR/HCP 是剥颅配准域,fastMRI 带颅骨;解剖实体查询可能学到数据集身份而非解剖;保留 v1.1 §6.3 的域来源泄漏探针(Z_A→数据集来源 应接近 chance) |
| A 其他器官 | ✅ | TotalSeg-MRI 616 例、SPIDER 447、AMOS 60,均未解压 |
| S token | 🔧 | fastMRI 脑 5 对比度 + 1.5T/3T 是 S 的主监督;SKM-TEA 单协议,S 在膝侧退化;R(U,A\|S) 的条件化只能在脑侧检验 |
| U_B | ✅ | SKM 3D 框;fastMRI+ 2D 逐层框(关系按层用 IoA);PDGM/BMSR/ISLES 3D mask |
| U_Q(motion/noise/aliasing) | ✅ | 仿真:fastMRI + SKM raw;真实:fastMRI+ 85 卷、KMAR |
| R 真值 | 🔧 | 主真值改 tissue_id;主判据只用分割内 319 例;积液 117 例(无宿主)考弃权,韧带 39 例(ACL/PCL 未分割、亦无独立监督)作扩展分析 |
| Relation token + 关系 Transformer | ✅ | K×M 对数:膝 6×20 级,脑 SynthSeg 粒度约 30×20,均轻量 |
| Hard relational negatives | ✅ | 膝内外侧、脑左右半球都有标签支持 |
| E 可观测性 | 🔧 | 真值未定义;见 §3.7 |
| k-space 干预三协议 | 🔧 | 公式须改多线圈;shot 结构按 ISMRMRD 头的 ETL 取;SKM-TEA 3D qDESS 按 ky-kz 分段;本组 BART 管线可复用 |
| 四阶段训练 | ✅ | 算力够;SSL 库约 9k 卷(fastMRI 5.6k + HCP 1.1k + PDGM 0.5k + BMSR 0.46k + TotalSeg 0.6k + ISLES 0.25k + SKM 0.155k),对"结构化感知"够,对"foundation model"不够,措辞要收 |
| Prostate 跨器官外测 | ❌ | 只有 TotalSeg 的前列腺解剖,无病灶;换 SPIDER 或砍 |
| MR-ART 外验 | ❌ | 不在盘;脑侧真实运动只有 fastMRI+ 的 33 卷 study-level 标记 |

### 2.1 几何跨度与 token 数(patch stem 2×4×4,stage-1;stage-4 为三次 ×2 下采样后)

| 数据 | 矩阵(D×H×W) | 体素 mm | stage-1 token | stage-4 token |
|---|---|---|---|---|
| SKM-TEA 原始 | 160×512×512 | 0.8×0.31×0.31 | 1,310,720 | 2,560 |
| SKM-TEA 面内 0.625 mm | 160×256×256 | 0.8×0.625×0.625 | 327,680 | 640 |
| fastMRI 脑 | 16×320×320 | 5×0.69×0.69 | 51,200 | 100(z=1) |
| fastMRI 脑 AXT2 | 16×384×384 | 5×0.57×0.57 | 73,728 | 144 |
| fastMRI 膝 | 36×320×320 | 3×0.44×0.44 | 115,200 | 300 |
| PDGM | 155×240×240 | 1×1×1 | 280,800 | 640 |
| HCP | 260×311×260 | 0.7 各向同性 | 659,100 | 1,560 |
| KMAR | 23×256×256 | 3.6×0.625×0.625 | 49,152 | 128 |

跨度 27 倍,size bucket 必需;大体积 batch 只能 1–2。

---

## 3. 逐实验可行性

### 3.1 Exp 1 Recognition–Binding gap

- 数据:SKM-TEA(tissue_id 直接算 ABA)。
- 干净数据上分割内宿主(319 例)的 gap 按构造接近零:baseline 只要有 mask 头,重叠就能读出绑定。gap 只能在两种设定下测:(a)损坏条件下;(b)对没有 mask 头的纯分类/检测 baseline。
- 分割外宿主(157 例)不能拿来制造 gap:积液 117 例无宿主,只考"该不绑时不绑";韧带 39 例绑到未分割的 ACL/PCL,关系模型自己也没有独立监督的 ACL/PCL 实体。
- 报告口径:分割内主判据 + 弃权分析 + 韧带扩展分析,三者分开;混报会被审稿人指出 gap 来自探针不公平。

### 3.2 Exp 2 Relation-centric vs multi-task

- 数据:SKM-TEA 主;PDGM/BMSR 脑侧(伪解剖标签,须披露)。
- 必加 baseline:seg-then-lookup(nnU-Net 分割 + 检测 + 查表),这是 v1.1 的头号对照,`_cui` 漏了。
- 参数量匹配:四个对手须同 backbone 同参数量级。

### 3.3 Exp 3 Intervention vs augmentation

- 训练:fastMRI 脑/膝 raw + SKM raw 上物理仿真(可行,存储见 §4)。
- 真实测试端:fastMRI+ 85 卷伪影脑图(有 raw)、KMAR(先做头信息审计,约 5% 配对须配准,其余直接用)。
- 前提:多线圈仿真实现正确,否则"干预 ≠ 增广"的论证不成立。

### 3.4 Exp 4 组合泛化(Brain+Motion、Brain+Noise、Knee+Noise → held-out Knee+Motion)

- 数据完全就位:脑 motion/noise 出自 fastMRI 脑 raw,膝 noise 出自 fastMRI 膝 raw + SKM,held-out 膝 motion 用 SKM 仿真 + KMAR 真实。
- KMAR 必须只做测试。
- 病理 × 组织的组合(如 held-out 软骨病变 × 髌骨软骨)全库只有 56 例,统计力不足,只能做次级分析。

### 3.5 Exp 5 Observability failure prediction

- 仿真梯度 E^{q0} > E^{q1} > E^{q2} > E^{q3} 只能从 raw 生成(fastMRI、SKM);真实端 KMAR 单档、fastMRI+ 二值。
- 真值与 baseline 见 §3.7;不补则贡献不独立。

### 3.6 统计功效(SKM-TEA 赌注实验)

配对 McNemar,α = 0.05 双侧,功效 0.8,近似式(与 Connor 1987 精确式相差 <3%):

```
n ≈ (1.96 + 0.84)^2 × p_disc / d^2 = 7.84 × p_disc / d^2
d = 两法 ABA 之差,p_disc = 不一致率
```

| 想检出的 ABA 提升 d | 假设 p_disc | 需要实例数 |
|---|---|---|
| 0.05 | 0.15 | 470 |
| 0.08 | 0.18 | 220 |
| 0.10 | 0.20 | 157 |
| 0.15 | 0.25 | 87 |

可用实例:官方 test 分割内 84、test 全部 130;5 折 CV 分割内 319、全部 476(155 例扫描对应 155 个不同受试者,按 scan 分折无泄漏)。
两个折扣:实例按扫描聚簇(平均每例约 3 个),设计效应按 1.3–1.5 计,有效样本约为名义值的 70%;CV 汇总各折预测再做配对检验略偏乐观。折算后:官方 test 只能检出 ≥15 个百分点;全 155 例 CV 分割内 319 例(有效约 220–245)可检出 ≥9–10 个百分点。
结论:赌注实验主判据用全 155 例 CV、分割内实例、损坏条件,门槛定为 10 个百分点;官方 split 仅作报告口径。

### 3.7 E 的真值:建议定义

模型无关的物理口径(主):对关系对 (U_j, A_i) 取局部区域 Ω_ij = dilate(M_i^A ∪ B_j^U),定义

```
E*_ij(q) = exp( − NRMSE( X^q , X^0 ; Ω_ij ) / τ )    motion / noise / aliasing 通用,取值 (0, 1]
或
E*_ij(q) = 局部保留谱能量 / 局部 SNR                  aliasing / noise 可解析
X^q、X^0 为线圈合并后的幅值图;τ 按干净重复扫描的 NRMSE 分布定标
```

训练 E 回归 E*,评估时以"冻结干净模型在 X^q 上关系是否仍正确"为失效事件,报 AURC/ECE/Brier。模型相对口径(冻结模型对错)作第二真值。baseline 必含 learned failure prediction(ConfidNet 类)、softmax、entropy、全局质量分。

---

## 4. 算力与存储预算(粗估)

- 仿真产物存图不存 k-space:fastMRI 脑 4468 卷 × 9 变体 × 3.3 MB ≈ 132 GB;膝 1172 × 9 × 7.4 MB ≈ 78 GB;SKM(0.625 mm)155 × 9 × 42 MB ≈ 58 GB。合计 < 300 GB,盘够。
- SSL 预训练:约 9k 卷,3D Swin-T 级,4 张 A800 上数天到两周(粗估,依裁块尺寸而定)。
- 赌注实验(SKM-TEA 155 例 CV + 仿真):单卡即可,瓶颈是仿真管线与分割 baseline 的搭建,不是算力。

---

## 5. 必改清单

优先级 A(不改则对应实验不可信):
1. 运动仿真改多线圈图像域重编码路线;三协议各验一次(仿真 vs 真实伪影的 k-space 统计对比)。
2. R 主真值改 tissue_id;seg-then-lookup 进 baseline;主判据只用分割内实例与损坏条件,积液/韧带分开报。
3. 脑侧 SynthSeg 伪标签管线;解剖粒度降级并如实披露。
4. E 真值定义 + learned-failure-prediction baseline。
5. KMAR 只做测试;配对前做头信息审计,不一致的约 5% 配准或剔除。

优先级 B(不改则数据表与事实不符):
6. BraTS 2021 → UCSF-PDGM v5(与 BraTS21 重叠 298 例,非子集;评估 PDGM 不用 BraTS 训练过的权重);MR-ART → 移出必需项;PI-CAI → 换 SPIDER 或砍。
7. SKM-TEA 面内重采样 0.625 mm;fastMRI 脑标为 2.5D。
8. 赌注实验用全 155 例 CV,门槛 10 个百分点;脑侧保留域来源泄漏探针。

优先级 C(表述):
9. "foundation model"措辞收窄为"结构化感知编码器";SSL 库规模如实写 ~9k 卷。
10. 组合泛化基准定位为实体级(解剖 × 异常 × 伪影),引用 CrossMed 崩塌数据作动机(v1.1 §2)。

---

## 6. 最小可行实验(M1,建议 6–8 周)

目标:在进入四阶段全训练之前,用 SKM-TEA 回答"显式关系 + 物理干预是否胜过 seg-then-lookup"。

1. 数据引擎(1–2 周):SKM-TEA 重采样 0.625 mm;seg/bbox/tissue_id 对齐;多线圈运动/噪声/欠采仿真(用自带 ESPIRiT 图与 Poisson 掩膜);5 折 split 按 scan(155 scan = 155 subject)。
2. 两臂(2–3 周):臂 A = nnU-Net 分割 + 3D 检测 + 查表;臂 B = 同 backbone 的实体/事件 query + relation token。同监督、同参数量级。
3. 评估(1 周):干净 / 三级损坏 × ABA(分割内主判据、积液弃权、韧带扩展分层)、侧别准确率、RIC;instance 级 paired bootstrap + McNemar。
4. 判据:B 在损坏条件下、分割内实例上 ABA 提升 ≥ 10 个百分点且 CV 配对检验显著 → 进 M2;否则方向重议。

---

## 7. 需要用户拍板的事项(附推荐)

| # | 事项 | 选项 | 推荐 |
|---|---|---|---|
| Q1 | 主干文档 | _cui 版 / v1.1 / 合并 | 合并成 v2.0:_cui 架构叙事为骨,v1.1 数据方案、赌注实验、风险表为肉 |
| Q2 | 前列腺外测 | 下载 PI-CAI / 换 SPIDER 脊柱 / 砍 | 换 SPIDER(在库,有 mask 与退变分级);PI-CAI 留二阶段 |
| Q3 | 脑侧解剖粒度 | SynthSeg 粒度 + 侧别 / 脑叶级 | SynthSeg 粒度为主;脑叶级只在 PDGM/BMSR/HCP 上用 ANTsPy atlas 配准做扩展 |
| Q4 | SKM-TEA 分辨率 | 全分辨率 / 面内 0.625 mm / 裁块 | 0.625 mm 与 KMAR 对齐,训练裁块,推理滑窗 |
| Q5 | E 的真值 | 物理保真度 / 模型相对 / 两者 | 物理保真度为主,模型相对为第二口径,baseline 必含 learned failure prediction |
| Q6 | 赌注实验判据 | 官方 split / 全 155 例 5 折 CV | 5 折 CV 为主判据,分割内实例、损坏条件,门槛 10 个百分点(含设计效应折扣);官方 split 只作报告口径 |
| Q7 | KMAR 角色 | 训练校准 / 只做测试 | 只做测试;多数配对可直接用,约 5% 须配准 |
| Q8 | MICCAI 2027 | 赶 / 不赶 | 不以会议为纲;M1 按 §6 跑,10 月底看结果再定 |
| Q9 | 清理授权 | 删 2.4G truncated 残留与 820G 原 tar;解压 KMAR part1、BMSR、ISLES、TotalSeg、SPIDER | 建议做;删除不可逆,须用户点头 |

---

## 8. 复核记录

2026-09-04 二次复核(对首版评审逐条重验)后的修订:

- KMAR 配对:首版"层间距 3.6 vs 4.8,须配准"系从单个配对过度推广;抽查 120 对仅约 5% 不一致,改为"头信息审计 + 少数配准",并从"照原样不成立"表中撤回。
- PDGM 与 BraTS21:首版"子集"有误;实为 298/501 重叠(262 训练、36 验证),并新增 BraTS 预训练权重污染风险。
- 分割外宿主 157 例:首版当作 seg-then-lookup 的"天然盲区";实为积液 117 例无宿主 + 韧带 39 例无独立解剖监督,降为弃权分析与扩展分析。
- Exp 1:补"干净数据分割内 gap 按构造接近零"。
- 功效:补聚簇设计效应与 CV 折扣,门槛 8 → 10 个百分点。
- E*:1 − NRMSE 改为有界的 exp(−NRMSE/τ)。
- 工具:ANTsPy 0.6.3、SimpleITK、pydicom、dicom2nifti 在库;SynthSeg/FreeSurfer 仍缺。
- 新增风险:剥颅域 vs 带颅骨域的数据集身份泄漏。
- fastMRI 脑覆盖 80 mm 改为下限。
- 首版复核过后仍成立的:token 数与存储估算、SKM-TEA 各层实例数、fastMRI+ 标注卷全部有 raw、多线圈运动仿真论证、GPU 与 BART/MONAI/nnunetv2 可用性。
