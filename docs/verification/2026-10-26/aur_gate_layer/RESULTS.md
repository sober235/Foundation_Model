# 双模式A/U/R判门层与fastMRI像素网格执行

本轮接续合并了远端Q31–Q34验收决定与本地已完成的LR5工程检查，合并提交`29d7917`。两者的口径分开：小集II/III已跑通工程链路；最终做成仍需双模式伪参照过门及指定fastMRI批处理。按用户已定顺序，先完成本块，再释放正式监督训练。所有伪参照数字均为NOT_EVIDENCE，医生关系标签属于独立论文证据线。

## 已实现的接口

- `gate_metrics.py`与`gate.py`提供32类出现、逐观察宿主Dice、最差宿主、精确同病灶2D/3D检出、同网格S7工作点、匹配实例掩膜Dice及部署关系实现的端到端一致率。完整阈值和head/lookup必须在评价前登记；缺支持为unavailable。CLI核对评价与参照的完成承诺及相同角色，不接受孤立结果表作为完整证据。
- `unified_evaluation.py`新增固定步长层与病灶质心层，保存逐观察支持、来源实例ID、匹配mask Dice和层内关系参照。原开发中央层口径不变。实际入口要求验证过的角色合同、完整正式权重及冻结登记；historical_test只允许认证StageIII，同一合格角色内按模型状态SHA独占一次评价。证书先于质心标签读取，失败尝试不自动重新评分。
- `reference_1mm.py`保留S7原生组件体积下限和组件平均概率工作点，将已选二值组件最近邻映到模型1mm网格，再重算组件与匹配。没有改为逐体素阈值或二次选择。S7缺失病例明确剔除，门比较相同row哈希交集。
- `test_role.py`建立独立historical_test后继合同：不改训练/validation父角色，验证完整源字节、离散A/U数组、原生病灶状态、源码及输出。ISLES修复写新私有文件；原生阳性变成全空时拒绝，未知不会转为健康负例。**本轮仅合成验证该模块，没有读取真实历史test数组或构建真实test后继。**
- `dicom_pixelgrid.py`是用户明确允许的独立假定像素网格入口；严格DICOM入口保持不变。两模式都按1mm输入推理并将标签/实例框映回原网格；只报告屏幕侧别，始终标记geometry=assumed、patient_coordinates=not_produced。正式批处理绑定登记的关系实现及三个疾病阈值的中位数。
- fastMRI续跑按完整文件集合、实际解码SourceRef、输入/权重/配置/源码/输出哈希核验。逐序列提交前源变化则只保留partial，不能在源文件恢复后误用另一版本的结果。fastMRI+转换固定沿用官方bottom→RSS；亮度比较只作诊断，未标注层检测不冒充健康阴性误报。

## 有限真实CPU验证

已在模型执行前冻结按字典序选取的3个序列。指定根目录共有4469个序列目录；本次3个均为AXFLAIR，原始层数16、14、16。原有format1 tiny pilot权重在两个模式上完成3/3序列：46个2D观察、3个3D观察，跳过0。worker wall42.107秒，各序列内部wall约12.028/11.548/12.234秒，输出完整性复核通过。此时长只属于该tiny CPU配置，不能外推正式模型或4469全量速度。

模型结果保存在私有`~/outputs/aur_gate_layer_20261026/fastmri_smoke_v1/`，未重复训练，未使用GPU。完成文件SHA256 `05b7a70b4cd2f664e60c9de4484e59d9df591c2c56c768c7adcc6bd203f1cbac`。原网格输出、假定几何及pilot状态都记录在产物中，不作A/U/R质量说明。

首次fastMRI+报告在创建输出目录前失败：7570条可解析框记录中有2条零面积框（zero_height2、zero_width1；原因可重叠）。原CSV、失败退出记录和私有复现保留。修复要求是排除并报告无效框，保留有效框原坐标，不把无效框或无框层当负例；无需重跑已完成的模型推理。

开发validation上的S7重评分已完成，wall632.285秒、退出0；513项源文件在评分前后校验，完成文件及输出哈希另行复核通过。完整完成文件SHA256 `7b7aa0f1c1256300d8011fa7163c57f261a389a37e09652fecab857ca9ad619a`。

|来源|同网格序列数|GT实例|命中|误报|灵敏度|FP/scan|缺折外预测剔除|
|---|---:|---:|---:|---:|---:|---:|---:|
|BMSR|41|470|318|36|0.676596|0.878049|0|
|ISLES|20|173|104|33|0.601156|1.650000|0|
|PDGM|41|53|48|15|0.905660|0.365854|0|

这是已保存S7预测在合格AUR网格上的参考工作点；无新模型前向、无S7补推理、无真实历史test访问，不等于AnatoBind已达到这些数值。各来源的exact row集合只留私有产物，供后续同集合比较。

修复后的标注报告`fastmri_plus_audit_v2`已完成，wall6.283秒、退出0，完成与输出哈希通过复核；未重跑3序列模型推理。全CSV两条无效框被明确剔除；20张有框图像的上下坐标诊断保留官方bottom规则，不依亮度改约定。3序列共有31个独立标注框、17个有框层；2D/3D各自命中0/31，各有17个未匹配检测。合计62框/34图是mode×slice计数，不是62个独立框。另58个未标注层检测单列，不记成健康负例误报；473个标注文件未纳入这次仅3序列批，不能解释为数据根缺失473个文件。这些tiny pilot结果只证明执行和报告链，不评价正式模型能力。完成文件SHA256 `2c8afd36e821b30db0ca8ed01fad94cc1c5a867324c716fa58bc88f38138ed32`。

## 代码检查与边界

判门纯函数/CLI45项、评价器与原开发回归32项、S7与角色相关51项、像素批处理及严格入口/统一推理相关145项均已有针对性通过记录。独立审阅发现的阈值覆盖、权重准入顺序、来源变化/续跑、部署关系选择、标签全丢失均已修复；失败测试和修复证据保留在私有implementation目录。

全套CPU回归实际**2069 passed、5 skipped，346.23秒，退出0**。命令为`env CUDA_VISIBLE_DEVICES= PYTHONNOUSERSITE=1 PYTHONPATH=. OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 nice -n 19 /home/congcongliu/anaconda3/envs/nvgen/bin/python -m pytest tests/ -q -rs -p no:cacheprovider`，在宿主机执行；日志为私有implementation/full_cpu.log。跳过的nndet专用模块及4项CUDA环境检查未冒充本轮通过。第一轮交叉集成曾有186通过/2失败：一项来自测试收集时实现仍在更新，另一项为下述fork超时；保留失败，不记为最终通过。

同时准备了正式II/III外部控制器，但尚未释放。其最初独立测试通过后，组合测试发现PyTorch已初始化线程池后的fork会卡在state_dict张量复制；进程栈定位后改为fresh interpreter阶段子进程，保持原20秒tiny检查上限，原故障组合102项通过。未以放宽超时隐藏问题。完整源码/请求哈希、父进程死亡保护、启动全量校验后的即时GPU检查均保留。正式训练的实际登记/启动与本块验收分别记录。

全套回归后，正式控制器仅追加请求/父登记的同一字节缓冲区哈希与解析修复，受影响的controller＋helper模块73 passed/30.49秒，独立增量复核0剩余；core和冻结runtime未变。旧registration_v1未执行且绑定旧控制器，后续需新v2登记/复核。启动cwd必须为冻结runtime，避免fresh `python -c`从当前checkout导入错误包。

按用户最后的交接要求，本轮不启动正式训练。以后只通过main的[PROJECT_CONTEXT.md](../../../../PROJECT_CONTEXT.md)和`work/aur-formal-supervised`增量恢复进展，不遍历全部历史分支。匿名汇总、命令和证据哈希见[RUN.json](RUN.json)，发布检查见[verification_anonymous.json](verification_anonymous.json)。

医学数组、逐患者/逐观察数据和原始DICOM路径仅在Git外私有目录。历史失败、原父合同与LR5 G1未改；本块没有最终test裁决、全4469正式推理或临床性能结论。
