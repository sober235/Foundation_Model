# 基于148实际训练数据的统一AUR缺口修复设计

日期：2026-10-19。代码起点：`6e5af4eac0d9ecae56076ce38943438f2ddd7ff9`。

## 目标、授权与状态

用户要求深入理解148上FM_data的数据构成并弥补此前缺口，已明确同一模型兼容独立2D/3D、保留DICOM全部信息、SSL-first I→II→III。沿用共享主体与维度适配器、单checkpoint，不扩展脑以外器官、多序列融合或网页。数据实际根为`/data2/congcong/data/FM_data`；只读，不接触冷备份。

本设计将**已执行的数据修复**与**后续训练建议**分开。完整审计、35例ISLES标签恢复和真实loader检查已执行；全尺寸统一训练、双模式G1、二维weak-R生成及医生R验收尚未执行。审计不能直接改变旧完整模型的A/U/R失败裁决。详细观察见[数据审计报告](../../verification/2026-10-19/aur_data_audit/RESULTS.md)。

## 三条可行路线

| 路线 | 可利用资产 | 代价与边界 | 选择 |
|---|---|---|---|
| 修复当前数据，warm-start统一Stage I | 已同格点的脑NIfTI、已验证Stage I骨干和完整SSL模型、既有患者分区 | 需新增两维数据资格、正式runner/G1；不能因此解决原生DICOM/人标 | **建议主线；本轮先完成数据修复** |
| 扩充脑数据后从头联合SSL | fastMRI脑5847卷、OASIS416来源限定患者、既有脑数据 | 先冻结新患者关联/曝光、格式/尺度/QC；更多训练预算；不会自动增加A32/U/R真值 | 作为登记后的后续对照，不默认全部混入 |
| 先完成原生DICOM与人标链，再训练 | 现有5640卷H5–DICOM候选关联、Level R待标库 | 当前IPP/IOP缺失、尺度解释不一致、人标0；需要上游材料及读者实际工作 | 并行补证，不作为现有NIfTI工程训练首次step的统一阻塞 |

不重新挑选旧test上的阈值、loss或训练方案。新增数据先查患者/来源关联和曝光，再定角色；相邻切片或同患者新序列不能创建独立评价患者。

## 数据组成与可用角色

| 数据源 | 实测规模 | 当前用途与限制 |
|---|---|---|
| PDGM | 495来源限定患者、501扫描 | T1/T1c/T2/FLAIR；A为SynthSeg伪标签；U按序列选择标签值，T1c仅1/4、FLAIR为1/2/4。54个TRAIN T1c目标值为空不等于健康 |
| BMSR | 314患者、461扫描 | T1pre/T1post/FLAIR；U只在T1post/T1c监督；subtraction不是额外患者 |
| ISLES | 250患者、250扫描 | 当前DWI/ADC，U仅DWI；另有250 FLAIR，其中180 TRAIN可作待QC的SSL补充，不直接继承DWI标签 |
| SibBMS | 193患者、372扫描/session | 当前模板网格作A/序列表征；当前清单不具U监督；8人未入已知监督/SSL清单，独立性未证明 |
| HCP | 1113患者、T1w/T2w | SSL TRAIN1001、validation112；已知Stage I曝光覆盖全部TRAIN，不是新独立队列 |
| fastMRI脑 | 5847卷、92590个RSS平面；FLAIR451卷/7150平面 | 8213框行涉及997卷，框不等于dense U；patient_id有5847个值，但不是已证明的生物学独立患者数 |
| OASIS | 416患者、436扫描 | 436对Analyze强度/fseg格点相符，fseg只有0/1/2/3；不当A32、健康阴性或医生R |

SKM-TEA、fastMRI膝以及AMOS/SPIDER/TotalSegmentator/KMAR等保留原角色，本轮不混入脑A32/R13监督。内部符号链接、硬链接、重复导出和压缩包不计为新患者。名称启发式器官字段仅用于初筛；fastMRI最终按实际acquisition分脑/膝。

当前监督TRAIN3499行/894来源限定患者、开发validation405/101、历史test1056/249；三者身份交集0，不构成跨来源生物学去重。SSL TRAIN5501行/1895患者，validation629行；原Stage I条件sampler回放覆盖全部1895 TRAIN身份，但不是完整历史逐患者日志。OASIS416与SibBMS8仅是未进入这两份清单的候选。

## 第一条已完成纵向修复：离散标签解码

35张TRAIN ISLES源标签的proxy scaling使正类略小于1，最大整数偏差仅2.3283064365386963e-10。直接`astype(int16)`使它们变成全零；180个TRAIN原重采样均能复现，根因不是affine错位。另3张原生本来全零，未证明是健康阴性。

统一接口为`anatobind.aur.labels.discrete_labels(values, dtype=np.int16)`：

- 实数、有限、浮点距整数绝对偏差≤1e-5；拒绝概率图、非有限值及输出dtype溢出。整数输入保留精度，范围检查使用Python整数，不经float舍入。
- 先解码类别再做nearest-neighbour重采样；连续强度仍用既有插值。生产dataset、外部U评价及fastMRI参考入口复用该边界。
- 新数据版本位于私有home，不覆盖已有/data2派生、历史清单、checkpoint、缓存或hash合同。`resume`旧导出会保留坏文件，不能作为修复。
- 新3499行TRAIN清单更新70条DWI/ADC病灶引用；35图恢复770019个体素、459个≥10mm³组件。3张原生空DWI仅在新清单关闭U/R资格，记录unknown原因；不是新增no-host或健康标签。
- 图像、anatomy文件、分区和A开关保持；恢复lesion会按原语义改变A-ignore，不能声称A训练目标完全相同。

回退只需重新选择旧清单。历史结果保留其原数据版本；在开发validation按同一预先冻结解码政策完成资格审计前，不释放新正式模型选点。历史test不为调参或扩大本次dense读取而重开。

## 训练接口与初始化

监督阶段保留`load_unified_volume → make_unified_crop → collate_unified → input_mode`的数据流。Stage I另外扩展`ssl.dataset.load_image/make_ssl_crop`的双维图像路径，仅读image/valid/几何/患者组，按图像前景选plane/crop；不能复用会读A/U标签或按病灶选plane的监督loader来冒充正式无标签SSL。正式数据合同新增版本、患者组、split/用途、图像/目标来源hash、单位、A/U/R资格原因、label provenance、观测mode/plane/FOV/变换及source instance映射；逐例内容仅私有保存。未知U不进入负例损失。

Stage I存在可信backbone-only导出：160 keys严格复用；也找到选中step8500的完整SSL recovery：176 keys、16个非骨干tensor严格载入，骨干与best export逐位相同。统一StageOne仅4个新增二维tensor待初始化/训练。**这是warm-start资格，不是正式初始化run或exact resume**：历史recovery缺CUDA/Python/NumPy完整RNG，新增参数也改变优化器结构。新run必须记录取backbone-only还是完整SSL状态、适配器种子、head初始化、重置optimizer/scheduler与新曝光计数；建议主线用已验证step8500完整SSL模型warm-start，新二维tensor固定种子初始化。

同一模型/optimizer，batch内mode一致，先固定2D/3D交替。每个mode内部先按source、再患者、再合法序列/观测采样，避免切片多的卷获得不受控权重；HCP曝光上限沿用既有可配置合同。mode与source曝光分别记录，恢复包含sampler/cursor/RNG与数值后端。CPU精确恢复；CUDA仅承诺已验证的状态恢复及声明容差，不承诺未来反向逐位确定。

`UnifiedPilot`保留工程pilot含义。新增正式编排、完整配置、验证/best选择及双mode G1，不能仅删除`pilot=True`限制。首个全尺寸资源probe用冻结TRAIN、每stage8step/每mode4step、batch1及既有完整3D裁块128×160×160/二维1×160×160；它仍是预算测量。实测时间/内存后单独冻结正式预算，不沿用旧PROPOSED batch/步数作为新观察。

## 二维R的可计算资格与证据边界

原清单显式二维R资格为0。当前`source_r_eligible`按源实例索引，不能一次设True后继承所有plane/crop。建议新增`qualify_planar_weak_relations`，按**变换后的每个观测组件**计算：

1. 仅TRAIN且A/U合格；源3D U沿用26连通≥10mm³，plane沿用8连通≥1mm²，身份追溯保持。源重叠必须从源instance与同格点解剖host-map直接统计或用有源hash的计数缓存；现有`host_targets`的one-hot分布不能区分真实重叠与nearest回退，不作为该条件的证明。
2. 源3D组件与实质host有正重叠、唯一主host h；h=0有效。nearest-only、并列、NO_HOST不进入此weak-positive政策。
3. 当前plane/FOV中h有有效、未被A-ignore遮住的可见解剖；另外独立计算U下SynthSeg伪图的重叠，要求plane唯一主host与h一致。不能用`U & ~a_ignore`定义重叠，否则病灶处会恒空。
4. 裁剪/旋转后重验；plane外宿主、歧义、过小/无效组件资格关闭，不能改成no-host。query超容量显式拒绝，不静默截断。
5. 合格目标用相容主host one-hot，provenance=`geometry_weak2d_v1`；首个兼容基线weak权重固定1.0，使用现有R归约，不通过val/test调权。它是训练伪参照，另行报告相对旧soft targets的变化，不是医生R。

医生positive/no-host走单独的已发布标签资格。现有Level R数据库1297 lesions/165登记患者、2594 orders、2读者账号，labels/adjudications/releases均0；不能填自动答案冒充人标。现有屏幕左右字段不自动转患者左右，须可信坐标/固定映射；不能将无医生标签的库当已完成人标。

## DICOM信息与几何

原字节只读保留，模型视图与完整来源分开。全部113409现存DICOM为classic单帧MR；raw fastMRI112609帧全部缺IPP/IOP。800个derived目录对象虽有几何，存在UID不合规、重复/不规则位置，不能充当合格原生脑栈。增强MR/SEG实际样本0，不扩大支持声明。

5640卷可建立H5路径/非空身份/shape/depth/索引候选关联；DICOM FOR与XML FOR全不匹配。PixelSpacing全匹配XML reconSpace，但76032帧与encodedSpace不符；保留两套解释及出处，不凭拟合修成“真实患者尺度”。H5 XML有矩阵/FOV和patientPosition，缺逐片位置/方向，不能反推出患者affine。现有NIfTI行无DICOM原帧链。

失败接口保持显式：缺患者几何不输出虚构mm³/患者侧别，非法UID/混栈拒绝；保留原字段，不静默改UID/补IPP。3D NIfTI派生平面训练与原生DICOM产品验收分开。先核本地可验证上游资料，再补原生来源；此次未流式审完171个tar.gz，所以不宣称所有容器都没有原始DICOM。

## 验收、失败处理和可执行边界

已完成的第一slice：原生标签→安全解码→新TRAIN派生/清单→35例实际loader/459实例→三源真实2D/3D crop/collate。没有新模型forward、GPU训练、val/test成绩或医生主张。

后续验收分别覆盖两mode×A/U/R：全32类及host汇总、U按source/sequence与FP/image或scan、R controlled/端到端与同观测B0。G1按患者分区、固定探针、两个mode单列，不能用旧3D PASS替代二维。A五类在有效TRAIN有标签，优先测试采样/类别竞争与优化机制；不直接宣称缺数据或单一loss已找到因果。

私有文件0700/0600、anonymous Git报告、通用错误不回显私有值。监督缺失、split冲突、源hash/decoder/数值环境变化、非法dtype/不可信几何均拒绝继续；任务失败保留已完成版本/原FAIL，不重写承诺。代码和数据变更有独立审阅，最多三轮自动修复。输出隐私有一次早期源码硬编码病例标识暴露，报告保留该偏差，不声称全会话匿名。

本设计不承诺现有资产即可补出医生标签、未知患者坐标或全新独立临床队列。下一实施计划应先完成双维数据资格与G1/runner，再按实测预算启动正式Stage I；当前数据修复可用，不代表正式训练已经完成。
