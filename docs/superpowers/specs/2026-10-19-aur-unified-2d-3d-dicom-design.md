# 同一个AUR模型兼容2D/3D DICOM：设计草案

日期：2026-10-19；设计时代码基线：`4be75f1`。**用户随后已授权按此路线执行**；统一技术闭环和限定TRAIN pilot已实施，见[执行报告](../../verification/2026-10-19/aur_unified_input/RESULTS.md)。正式全模型联合训练、真实DICOM及性能验收仍未完成。下文保留设计依据；观察结果以执行报告为准。

## 1. 已确认目标和建议的边界

用户明确“构建出的模型能够同时兼容2d and 3d input”，并说明“使用的图来自于我给你的dicom，保留来自于给你的dicom数据的全部信息”。交付模型本身须同时接受独立单张2D MRI和3D体数据，两种模式均完成A32、U实例、R13+no-host。此前优先实施两个独立完整模型的建议不再适用。

全部信息保留覆盖原始DICOM像素、完整元数据和来源对应关系；不能只留截图或特征白名单。用户没有要求把所有字段都作为学习特征，也未逐层规定共享权重。**一个checkpoint保存共享主体及少量按维度区分的参数，是本草案的技术建议。** 既有3D结果保留为基线，不能改称统一模型已完成。

继续限定脑MRI，每次处理一个sequence；本轮不扩展到其他器官、CT或多序列融合。支持两种输入模式不要求同一次调用同时提供2D和3D。

## 2. 建议结构：共享主体，切换小型输入与上采样模块

继续使用当前PyTorch AUR结构：

- 真实2D平面内部表示为`B×1×1×H×W`，观测深度仍为1；3D为`B×1×D×H×W`。不复制单层伪造体积。
- 同一模型实例内，2D采用深度1的patch embedding及mask上采样，3D保留深度2路径；仅这两处先设置显式按维度区分的参数，以`input_mode`选择。
- 共享Swin注意力/合并层、物理坐标RoPE、A/U query解码器、mask投影、其余图像融合模块及R头。完整参数清单和梯度检查须核实实际共享范围，不能仅凭同名声明。
- 一个版本化checkpoint包含共享权重、两个适配路径、配置、标签/几何版本和训练状态；加载一次即可切换两种输入，不在后台选择两个独立完整AUR网络。
- batch内先使用同一种维度；训练step按冻结配额交替2D/3D，共享一个优化器，分别记录两维曝光量，避免长序列来源支配采样。

二维平面仍位于患者三维物理空间，每个token可携带三坐标，但只有两个被观测轴。沿法向没有额外影像证据。保留现有坐标参考约定并显式完成DICOM坐标转换；不能把不同扫描的原始坐标原点直接当跨患者解剖标签。

| 方案 | 优点 | 代价与选择 |
|---|---|---|
| 两维都用深度1分块 | 单一stem/up、结构改动少 | 原3D两处卷积形状改变；同体裁块的初始深度token约翻倍，需重测资源。保留为备选 |
| 按维度切换stem/up，其他主体共享 | 保留原3D路径，真实单层2D可运行，单模型/单checkpoint | 新2D参数必须初始化/训练，D=1合并分布仍需验证。**建议主方案** |
| 单层复制或补成现有3D输入 | 可绕过部分形状限制 | 不能建立独立2D训练/评价，不作为交付方案 |

已有研究通过切换2D/3D patch embedding后共享Transformer，支持这种结构方向；不证明本项目A/U/R迁移或性能。[UniMiSS原论文](https://arxiv.org/abs/2112.09356)

## 3. 现有代码与检查证据

当前默认patch=`(2,4,4)`，整除检查拒绝`D=1`；stem及坐标/有效性池化也使用该patch。窗口会随实际维度缩小，合并层处理奇数深度；A/U解码器把特征展平成token。mask上采样kernel/stride是另一处patch依赖。[Swin](../../../anatobind/aur/swin.py)、[任务头](../../../anatobind/aur/heads.py)

RoPE接受每token的三维物理坐标；R可在深度1网格产生14输出，但不自动解决平面内关系监督。[RoPE](../../../anatobind/aur/rope.py)、[模型](../../../anatobind/aur/model.py)、[R几何](../../../anatobind/aur/relation.py)

本轮子代理用现有代码、随机小模型和合成零张量作一次CPU前向：默认patch拒绝深度1；改成`(1,4,4)`时A=`(1,32,1,32,32)`、U=`(1,2,1,32,32)`、R=`(1,14)`。U query数2仅用于降低检查开销。两配置形状不同的参数只有`backbone.patch_embed.weight`与`masks.up.weight`。这是**缩小配置的形状证据**，不证明完整checkpoint数值等价、反向、恢复或真实2D质量。

检查使用nvgen、禁用user-site/CUDA、PyTorch线程2，约5.8秒；未使用要求的nice19，属于执行偏差。没有读取私有DICOM/清单/影像/checkpoint，没有训练或运行时代码编辑；后续检查须使用nice19及线程限制。当前普通推理入口仍只读3D NIfTI，DICOM端到端路径未实现。[推理入口](../../../anatobind/aur/infer.py)

匿名检查记录与复现源码见[合成形状检查](../../verification/2026-10-19/aur_dual_input_readiness/unified_shape_probe.json)，未独立重复执行。设计的fresh-context builder check未发现需要用户进一步裁定的实施阻塞问题；这不替代用户对设计的审阅，也不是模型性能审查。

## 4. DICOM完整保留与模型特征分开

原始DICOM在既有私有源中只读保留，作为完整信息权威源；不新增备份、复制或保存时长要求。派生张量记录源hash、帧映射、变换和预处理版本，可关联回完整原件。嵌套、逐帧和私有字段不因模型只用白名单而丢弃。旧网页临时上传清理方案不得删除唯一权威原件。

模型使用定义明确的影像、几何和序列特征；身份字段仅用于受控关联/患者分区，不进入学习特征，诊断文字或泄露目标的字段不默认输入。归一化、裁块和重采样视图不是无损原件替代。Git和工具输出只存规则、匿名计数和hash，不写原始字段值或医学数组。

- 单文件不等于2D，多帧/同series也不必然是一个3D体；先区分空间栈、时间、echo、diffusion等维度，再选真实平面或规则体。
- 2D保存`origin_patient`、`axis_h/w_patient`、`spacing_h/w`与有效覆盖，满足`P(h,w)=origin+h*spacing_h*axis_h+w*spacing_w*axis_w`。DICOM第一组IOP沿列索引增长，第二组沿行索引增长；PixelSpacing顺序为行、列。
- 3D保留逐帧位置及体素到患者空间映射，按位置构栈；SliceThickness不能冒充层中心间距。混合/不规则栈不得静默塞入规则affine；不可信几何显式拒绝对应物理量推理或标未支持，不捏造方向/间距。
- 明确LPS/RAS、主动增广/被动重格式化及裁块/padding/重采样变换。A/U输出回原始帧/体素网格，R关联原始U，保留源帧映射与有效覆盖。
- 实施前只读盘点实际提供的DICOM对象类型；classic与enhanced multi-frame分别解析顶层/shared/per-frame信息。不把实际未使用的对象类型强行设成首次模型训练的前提；最终只声明已验收对象类型的支持。

规则依据：[DICOM Image Plane Module](https://dicom.nema.org/medical/dicom/current/output/chtml/part03/sect_C.7.6.2.html)、[Functional Groups](https://dicom.nema.org/medical/dicom/current/output/chtml/part03/sect_C.7.6.16.2.html)、[Multi-frame Dimensions](https://dicom.nema.org/medical/dicom/current/output/chtml/part03/sect_C.7.6.17.html)。本轮没有核查用户DICOM字段是否齐全。

## 5. A/U/R和数据监督

共享标签语义：A32、U实例、R13分侧宿主+no-host；AUR的`host_index=0`有效，`NO_HOST=13`，未知另行表达。2D报告可见平面mask/面积mm²和关系；3D报告体mask/体积mm³。保留层厚不等于单层mask有完整病灶体积或平面外证据。

先锁患者/全部序列分区，再派生两维视图。同患者的相邻切片、平面、增强及原体保持同split。原生帧与重采样平面分别记录/评价，派生平面成绩不直接证明原生DICOM泛化。

共同变换image/A/U/ignore/valid。源3D U资格与2D可见性/连通/面积规则分开；U不可监督或不可见不当健康负例。2D R仅在原标签可靠且关系在该平面可监督时继承，否则关闭该实例R监督，不能把平面外宿主改成no-host。具体资格、面积门和query溢出策略由实施者在计划中冻结。

R可共享参数，但当前几何来自独立coarse mask分支，既有3D诊断风险保留。首个统一兼容基线不静默更换loss、R几何或背景采样；必要的维度语义适配显式记录。用受监督预测mask计算几何可作为后续预注册对照，不冒充已验证修复。A缺类、U误报及R低于B0不会因兼容两维自动消失。

## 6. 初始化与训练

保持SSL-first I→II→III。建议先核查既有3D **Stage I** 权重是否可严格映射到共享主体/3D路径，且来源分区符合要求；符合则初始化统一Stage I后加入2D联合学习，不符合则从头执行统一Stage I。不以C2监督末期权重加随机2D适配器冒充正式统一SSL主线。

2D新适配器先随机初始化，通过统一Stage I学习。深度核求和/平均仅为另行登记的候选，不默认视作数学等价。2D MIM头/遮挡须匹配真实平面，3D匹配真实体。共享主体从两维得到梯度，曝光/采样及loss归约分开记录；不强制两维特征/预测完全相同，因为观测信息不同。

迁移映射列出全部共享/3D参数来源及新增2D参数，拒绝非预期缺失/多余key。I→II严格继承同一统一backbone，II→III遵循冻结协议；optimizer、scheduler、RNG、sampler模式/位置及两维曝光量一并恢复。联合G1分别检查两维，旧3D G1不替代2D探针。

采样比例、batch/累积、步数、LR、G1数值门及资源预算由实施者实测后提出，写入实施计划和正式run配置，不借用旧PROPOSED数字或用旧test反向选择。

## 7. 验收和后续工作

先检验几何/模式/回映射、单次模型加载的两维推理、预期参数梯度、loss/ignore边界、严格载入和恢复。复用3D参数时先证明eval路径与冻结基线数值一致，声明容差/预处理；再进行限定预算的TRAIN-only过拟合和短pilot。不能要求最终质量达标才允许首次optimizer step。

最终按2D/3D×A/U/R分别验收：A全32类与host汇总、U各来源敏感度及FP/image或FP/scan、R controlled/端到端及同实例同观察范围B0。2D参照只用相同分区及U可监督子集，无U标签平面不作参照阴性。冻结validation选点和未暴露评价队列；历史test不重新命名为新独立测试。伪参照NOT_EVIDENCE，Level R人标是另一层证据。

这是项目级修改；用户已授权执行并完成初轮实现、有限TRAIN训练和验证，见[执行计划](../plans/2026-10-19-aur-unified-input-execution.md)。网页/API/URL继续后置。正式全模型run仍需联合G1、初始化、数据/几何、预算和独立评价检查，当前仍为**NOT READY**；不能把技术pilot升级为正式性能达标。

实施者负责类/配置/CLI命名、模块映射、单checkpoint序列化、实际DICOM对象清单、fixture、容差及上述数值细化，在计划中给出可执行检查。若主方案无法满足实测资源，提交统一深度1备选的预算/迁移比较，不能静默退回两个独立完整模型。“模型本身兼容两维”不被扩大为“每个参数都必须完全相同”；后者若被要求，是对本建议共享范围的进一步约束。
