# 2D/3D A/U/R 技术路线迭代与正式训练入口

更新：2026-10-19；起点：`d017fa7`。本轮交付为记录修订、源代码核查和训练就绪性分析，没有启动新主训练。历史事实标为 **已记录/已执行**，新选择标为 **建议/待冻结**；不把方案中的命令当成运行结果。

## 1. 固定目标与技术选择的区别

**用户已定目标：** 独立2D MRI图像和3D MRI体数据都要完整输出A32解剖mask、U异常实例mask、每个U的R13分侧宿主+no-host。S为内部序列上下文。2D输出二维mask/面积mm²，3D输出三维mask/体积mm³；两种模式分别验收。

**旧技术提案：** 3D沿用现有AUR；单张2D采用独立2D Swin和checkpoint，共享标签与结果语义。原产品规格的状态仍是“用户目标已并入；实现计划待用户审阅”。因此“必须支持2D/3D”是既有要求，“必须两个独立模型”不能冒充用户已经单独批准的架构决定。[原规格](../superpowers/specs/2026-10-14-anatobind-brain-dicom-web-design.md)

最初分析曾建议优先落地旧分离方案。**后续同日用户已明确：构建出的模型本身须兼容2D/3D，图像来自其提供的DICOM并保留全部信息。** 因此不再优先推进两个独立完整模型；新的[统一模型设计草案](../superpowers/specs/2026-10-19-aur-unified-2d-3d-dicom-design.md)建议维度适配stem/up、共享主体和单checkpoint，具体共享范围仍为技术提案，未把每个参数必须相同写成用户决定。下文独立2D训练建议为此前分析，需按新统一路线细化初始化、混合训练和两维回退检查。不得以复制/补零单张图像后调用3D模型来宣称完成独立2D能力。

## 2. 路线迭代记录

| 时间/依据 | 性质 | 路线变化与当前解释 |
|---|---|---|
| 2026-09-22文档，提交`227ff3b` | 原3D技术路线 | 可变尺寸3D Swin、物理坐标3D RoPE、实体/事件/关系；不是独立单张2D实现。[网络目标](2026-09-22-anatobind-mri-network-target.md) |
| 2026-10-08 | 简化实施方案 | 先实现3D A/U/R，曾简化为跳过Stage I；该训练顺序后被增补覆盖。[3D规格](../superpowers/specs/2026-10-08-anatobind-brain-aur-design.md) |
| 2026-10-09 | 正式3D训练路线修订 | 用户决定SSL-first：I→II→III；II严格继承同结构I backbone，随机初始化II/III属于C0控制。随后3D Stage I和G1已有执行记录。[增补](../superpowers/specs/2026-10-09-anatobind-brain-ssl-first-addendum.md)、[三阶段计划](../superpowers/plans/2026-10-09-anatobind-brain-ssl-first-three-stage.md) |
| 2026-10-13 07:48 UTC，Q27 | 用户目标重申 | 明确“一张2D或3D DICOM”均需解剖、异常和宿主绑定；交互URL为期望呈现方式。[归档Q27–Q29](../verification/2026-10-09/anatobind_brain_ssl_first/status_archive/STATUS_2026-10-14_before_task7_closeout.md) |
| 2026-10-13，提交`20fdec4` | 双输入规格和2D技术提案入库 | 文件名虽为10-14，实际提交为10-13 11:25:33+02:00。提出独立2D Swin、患者先划分再派生平面、专用2D参照与验收；2D模型尚未实施。[输入规格](../superpowers/specs/2026-10-14-anatobind-brain-dicom-web-design.md) |
| Q28/Q29及10-16补记 | 用户优先级与实施解释 | 用户要求先考虑模型成功、显示后做；2D、DICOM和离线验收保留。具体“先3D有效再2D”是后续工作线安排，不是删除2D目标；部分Q29技术顺序被明确记录为Codex裁定，不能改写成用户逐项批准。[执行范围](../superpowers/plans/2026-10-14-anatobind-brain-dicom-web-execution.md) |
| 2026-10-15 | 已执行，性能未达标 | 新U-loss C2训练/val/test完成；A历史test0.7988、U三源失败、controlled R0.5606<B0 0.7545。不能用“训练完成”代替“模型成功”。[C2结果](../verification/2026-10-09/anatobind_brain_ssl_first/u_empty_c2_20261014/RESULTS.md) |
| 2026-10-16–18 | 已执行诊断 | query_cc完整405val失败；排序、A监督/R几何与TRAIN U监督分析进一步定位缺口。它们不构成2D进展。[完整val](../verification/2026-10-17/aur_query_cc/RESULTS.md)、[U监督](../verification/2026-10-18/aur_u_train_supervision/RESULTS.md) |
| 2026-10-19 | 局部候选不获支持 | 环采点0.2→全局0的固定TRAIN切线对照不利，保留原配方，没有参数更新或新val/test。d017fa7修复入口只写3D的遗漏。[背景配额结果](../verification/2026-10-18/aur_u_background_allocation/RESULTS.md) |
| 本次迭代 | 记录完善/待决训练细化 | 显式分开六项能力覆盖、2D初始化缺口、训练准备与最终验收。独立2D训练不应未经新决定而无限依赖3D先达到最终质量门。没有宣布新架构或新超参已经成功。 |

## 3. 当前六项覆盖与可复用部分

| 模式 | A | U | R |
|---|---|---|---|
| 3D | 模型/训练/评价已执行；缺类与历史test缺口 | 模型/训练/评价已执行；误报及敏感度门失败 | 模型/训练/评价已执行；controlled低于B0 |
| 独立2D | 完整A32模型、训练、验收未完成 | 完整实例路径、训练、专用参照/验收未完成 | 完整R14训练/controlled/端到端验收未完成 |

不存在`anatobind/aur2d/`，旧计划的2D准备、训练、推理、评价入口和测试未实现。已有2.5D膝检测器/二值脑轮廓模型不是单张脑MRI完整A/U/R。现有3D标签映射、监督语义、匹配/损失、分阶段训练、恢复和评估经验可复用；`Conv3d/max_pool3d`、体素几何和FP/scan不能机械搬到2D。

## 4. 本次只读就绪性证据

证据：[匿名预检](../verification/2026-10-19/aur_dual_input_readiness/metadata_preflight.json)、[环境和源代码核查](../verification/2026-10-19/aur_dual_input_readiness/readiness_evidence.json)。预检只读清单和文件存在性，没有载入影像/标签数组、派生平面或运行新训练。

| 既有分区 | 来源限定患者 | 体/序列行 | U监督行 | loader启用A标志行 | R与U标志共同启用行 |
|---|---:|---:|---:|---:|---:|
| TRAIN | 894 | 3499 | 1216 | 3420 | 1177 |
| validation | 101 | 405 | 143 | 394 | 138 |
| 历史test | 249 | 1056 | 354 | 1032 | 342 |

来源限定身份的三对分区交集为0，不等于已完成跨来源生物学去重。4960条image路径引用、4960条anatomy引用和3887条lesion引用均存在于许可数据根；计数为逐行引用，可能重复，不是独立文件/患者数；这只证明文件可定位，不证明新的2D网格/QC正确。计划中的2D samples/splits和Dataset909 raw/preprocessed路径均不存在。

**监督字段兼容性：** 当前3D loader对旧清单缺失的A/R标志使用True默认值，显式False才关闭；不能用`row.get(flag) is True`把缺失字段误算成无监督。上表遵循`dataset.load_volume`口径，仍只是行级资格，不是实际平面内的正像素数或R实例数。新2D清单应显式物化标志及QC来源，避免继续依赖隐含默认。

当前nvgen的NumPy2.4.4、torch2.5.1+cu121、MONAI1.5.2、SimpleITK2.5.5及nnUNet trainer导入成功；nnunetv2包版本2.8.0。此为导入检查，不是完整新2D回归或训练兼容性证明。GPU快照有多张80GB A800/A100空闲，但未占用/预留，不构成batch或吞吐承诺；实际启动前重查分配、负载和进程。

## 5. 正式训练目前被什么阻挡

| 缺口 | 开训前必须形成的具体产物 |
|---|---|
| 架构/初始化未闭合 | 明确独立模型或共享权重；对选定2D backbone给出Stage I方案/兼容预训练来源、严格载入规则和2D表征探针。不能拿3D G1报告替代2D初始化证据。 |
| 2D数据与目标未实现 | 锁定患者/序列/平面清单和hash；共同变换image/A/U/ignore/valid；左/右与毫米坐标测试；监督开关显式化。 |
| 二维实例与面积规则未冻结 | 分开源3D资格门、2D连通、面积mm²门、裁块残片ignore、query超容量策略；不能把10mm³改写成10mm²或捏造层厚。 |
| 训练纵向路径未实现 | 原生2D backbone→A/U→R→loss/backward→保存/重载/恢复→普通推理→指标，而不是仅有forward或命令模板。 |
| 评估和参照未闭合 | 冻结每来源FP/image、2D匹配/ignore/阈值规则、同患者分区且同U监督资格子集的Dataset909及2D B0；受控与端到端R分开。A-only/U不可见平面不能被参照当作健康阴性。 |
| 新run配方未冻结 | 实测显存/吞吐后确定batch、累积、workers、I/II/III预算、LR/warmup、验证频率、恢复和停止条件。旧240k/80k seen slices、global batch24仍是PROPOSED。 |

### 初始化路线必须单独写明

已定3D主线是SSL-first，当前实现也把随机II/III标为C0控制；原2D计划却仅列II/III，没有同结构2D Stage I或兼容权重。这是需解决的缺口。

**建议：** 为选定2D结构补齐真正的2D Stage I，再严格I→II→III；原3D自监督的原则可复用，但其checkpoint、三维MIM实现和G1数值门不能直接算作2D证据。若希望随机初始化先跑监督学习，先标为工程pilot/C0；若要把它改成正式主线，必须显式修订训练协议。本文不将新2D Stage I说成已实施或已证明能修复性能。

### 3D经验必须进入2D设计，但不是已验证的2D修复

- A：共享病灶偏置采点+A的病灶ignore可让存在类别缺少有效正点。建议比较2D密集A监督或独立A采样；不把该机制直接认定为历史单侧缺类的唯一根因。
- U：C2已含`unmatched_empty_v1`。新run必须显式记录profile，因为兼容CLI默认仍为`legacy`；不能把再加空mask BCE当新修复。2D需保留matched/unmatched query及逐组件来源诊断，不能照搬不获支持的ring→global改动。
- R：3D中R读独立`embed_coarse`，并非直接读取已受A/U监督的full mask；共享像素金字塔确实受A/U训练。建议2D基线优先定义R几何来自哪一组**预测**mask，并考虑直接使用同一受监督mask分支；该选择需要验证梯度与回退，尚非已证明的性能修复。
- no-host/unknown：AUR零基`NO_HOST=13`，旧接口的`UNKNOWN_HOST=0`语义不同；AUR的`host_index=0`是有效宿主，不能按旧接口规则过滤。未知、无监督、背景和无宿主不能合并，需单独的资格/未知状态。单平面缺少另一侧宿主时也不能把“不可观测”当作中线或no-host事实。

## 6. 建议下一步：有限的训练准备闭环

此顺序为建议，不表示已经执行；它替代“直接复制旧命令长跑”或无截止地追加同类3D诊断。

1. **冻结2D训练细化。** 解决权重共享要求、初始化、平面/面积/坐标、A/U采样、R几何来源和评价分区。用户已要求双输入；这些是技术选择，不重新询问是否需要2D。
2. **实现一条可训练纵向路径。** 先合成fixture，再固定TRAIN-only小集；同时覆盖正/负U、关闭监督、无目标、缺少宿主、padding及多方向。训练数据可先来自有几何的现有NIfTI卷派生平面，不必等网页/API部署。
3. **核对真实2D数据。** 先分患者再切片/增强，全部序列/视图同split；冻结来源、序列、患者和平面曝光权重，区分原生采集平面与重采样/斜切平面，防止把相邻切片当独立患者。被动换平面/重采样与主动解剖增广的坐标约定分别写明，用左右不对称fixture验收。Dataset909只使用同一清单的U可监督子集；AUR额外A-only平面不变成参照阴性。
4. **完成限定预算的readiness pilot。** 依次检查CPU/GPU smoke、固定小集过拟合、真实loader短训练、验证、导出/重载、断点恢复和阶段衔接。只用TRAIN/validation；不以历史test挑设置。pilot的步数上限、数值目标及允许修复项在实际运行前写死。随机或不满足G1的试跑明确标pilot，不冒充主线。
5. **通过技术就绪检查即冻结正式run。** 冻结配置/清单/源码/初始化hash后开始正式Stage I；Stage I完成且2D G1通过后才能正式Stage II，再从II完整继承进入III。Dataset909参照可在同一冻结分区并行训练，比较结果需在最终U验收前就绪，不必成为第一个AUR optimizer step的前提。
6. **按预注册完成验证和测试。** validation选checkpoint/工作点；未见或已暴露的评价队列分别声明。已有历史test不能重新命名成新的独立测试。保留失败结果，不在test上改配方。

### 三种门不能混用

| 门 | 何时检查 | GO条件与解释 |
|---|---|---|
| 技术就绪 | 正式Stage I主训之前 | 数据/几何/分区、各任务真实梯度、loss有限、遮挡无泄漏、checkpoint恢复、推理/指标可运行；有限pilot可学习且资源实测。这不要求最终A/U/R已达标。 |
| 阶段迁移 | I→II、II→III | 同结构严格权重继承；2D专用G1及Stage II迁移/验证规则按新配方冻结。不能沿用3D probe成绩充当2D过门。 |
| 最终性能/发布 | 主训练和预定评价结束 | 原2D提案中的候选工程门，实施前需冻结：A13-host宏Dice≥0.80且全32类报告；U每来源在冻结FP/image参照点敏感度≥reference−0.05；R controlled≥同实例B0。两维分别报告，伪参照均NOT_EVIDENCE。原生2D DICOM外推与Level R人标仍需相应独立证据。 |

**不构成首次2D训练的技术前提：** 网页/API/公开URL已经完成、当前3D全部质量门已通过、Dataset909已经给出最终比较结果。它们各有产品/评价角色，不能形成“先训练成功才允许训练”的循环。DICOM解码与真实几何端到端验收仍须完成，派生2D平面过门不自动代表原生2D DICOM泛化。

## 7. 开训记录必须包含的项目

训练入口、mode/架构版本、初始化谱系、split/manifest/QC hash、监督映射、平面与实例规则、模型/loss/decoder/eval版本、每阶段seed与优化器配置、global batch计算、seen source-plane/crop与增强视图计数、step/warmup预算、数据采样曝光、验证/测试规则、GPU选择实测与内存/吞吐、停止/恢复条件、私有日志路径和匿名汇总。缺一个关键定义就标待冻结，不用默认值悄悄补齐。

外部实现核对：nnU-Net在没有自定义split时会生成自己的划分；必须把项目锁定划分写为其识别的`splits_final.json`，不复制文档里的默认随机做法。[官方自定义划分文档](https://github.com/MIC-DKFZ/nnUNet/blob/master/documentation/manual_data_splits.md) 空间几何由origin、spacing、direction共同定义，标签重采样使用不引入新标签的最近邻；这些事实支持前述几何验收，不证明本项目新管线已正确。[SimpleITK官方空间与重采样文档](https://simpleitk.readthedocs.io/en/master/fundamentalConcepts.html)

本环境无可调用的`artifact.science`，相应parsed/hypothesis/package-check证据以版本化JSON保存，不伪称已创建外部证据图。代码、数据或checkpoint未因本次分析改变；正式训练就绪状态仍为 **NOT READY**。
