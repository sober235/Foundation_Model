# 2026-10-28 当前接续：128-query probe完成，正式训练待启动

以[PROJECT_CONTEXT](PROJECT_CONTEXT.md)和[本次报告](docs/verification/2026-10-28/aur_query128_release/RESULTS.md)为准。128-query实现506dbdc完成，probe_v5 II/III各8步、双mode执行及只读产物回放通过；原StageI/G1不重跑。正式II4096→III2048使用v5登记，尚未启动。保留4h/2h阶段上限和8h控制器上限，probe不保证完整预算时长，当前128 CUDA恢复尚未单独验证。只做工程代码/训练；protocol暂不输入，论文推迟。下文均为历史记录，其中“待确认/未实现/运行中”不能当作当前状态。

# STATUS：2026-10-26（容量审计完成，128-query重新训练设计待确认）

## 最新：完整容量审计完成

[容量报告](docs/verification/2026-10-26/aur_query_capacity/RESULTS.md)：3499 TRAIN源行最多102实例；登记II的2D/3D各2048观察最多14/85，3D有3次超过64；III子集3D有2次超过64。固定周期validation每mode8观察最多3/22，已有1725步匹配计数全部一致。CPU4worker、157.914秒、无模型前向或训练；63项合成计数对照、48项方向几何检查及独立复核通过。

[128-query设计](docs/superpowers/specs/2026-10-26-aur-query-capacity-128-design.md)建议新版本从原LR5正式Stage I骨干重新开始II4096→III2048，不迁移1536的监督optimizer，不丢弃GT或改变裁块。原数据和损失保持，父子runtime身份必须显式转换；128覆盖本轮观察及当前三维源上界，不等于任意未来二维上界。具体架构/重新开始决定待用户确认；没有新生产实现或训练。下文“尚未审计”仅属此前停止诊断。

## 最新：容量失败已复现

run_v3终态FAILED：II已完成1725步，第1726步前3D裁块含65个病灶实例，超过64query。全部已保存loss有限；CPU按原seed/draw862复现容量拒绝，源码/角色/选中3源文件哈希通过。宿主机控制器退出，III未启动，closing完整校验通过。最近持久化及best checkpoint均为1536，文件哈希通过。详见[STOPPED.md](docs/verification/2026-10-26/aur_formal_supervised/STOPPED.md)。尚未审计完整容量分布或确定新策略，不自动重启；下面“运行中”为历史快照。


## 2026-10-26正式监督接续

当前run_v3于09:39:55 UTC启动，PID538100，空闲A100物理3；cwd/PYTHONPATH均为原LR5冻结runtime。v3登记及代码独立审阅0项，执行代码8868a8d，修复后完整CPU2080 passed/5 skipped（348.95秒）。预算仍为II4096→III2048，总上限8小时。09:50:40 UTC宿主机快照：Stage II已运行108/4096步，2D/3D各54次曝光，108条已记录loss全部有限，PID身份核对通过；III尚未开始。当前尚无正式监督完成证书或最终质量成绩。

原run_v2在Stage II初始化前GPU准入拒绝，step0、零checkpoint、III未启动，宿主机确认进程退出，closing完整输入校验通过。原通用错误不能定位拒绝原因；外部控制器现保存实际判定时匿名测量和固定原因码，GPU阈值和冻结训练配方不变。v1/v2及失败结果均保留；run_v3是新的有界启动，不是追加训练预算。

最新证据见[正式监督接续报告](docs/verification/2026-10-26/aur_formal_supervised/RESULTS.md)和[RUN](docs/verification/2026-10-26/aur_formal_supervised/RUN.json)。私有`~/outputs/aur_lr5_formal_supervised_20261026/run_v3/`是唯一当前运行，不要重复启动。先核对宿主机进程与匿名状态，再报告实际步骤和终态；下文“未启动/v2待生成”均为此前交接快照。

## 当前回顾入口：main＋一个工作分支

用户要求以后只回顾`main`和`work/aur-formal-supervised`，不完整读取所有分支。先读[PROJECT_CONTEXT.md](PROJECT_CONTEXT.md)，再读本节、[最新报告](docs/verification/2026-10-26/aur_gate_layer/RESULTS.md)和[RUN](docs/verification/2026-10-26/aur_gate_layer/RUN.json)；工作分支从本次归档main建立，作为唯一后续执行入口。旧分支是档案，不扫描、不删除。

- Q31–Q34判门层已实现：A32逐类出现与宿主Dice，U同网格S7工作点及匹配mask Dice，端到端R；development与historical角色严格区分，阈值和部署head/lookup预先绑定。真实historical test接口仅合成验收，未打开真实test数组。
- S7开发validation参考重评分已完成：BMSR/ISLES/PDGM分别41/20/41序列，缺折外预测剔除0；wall632.285秒、退出0、哈希复核通过。不是新模型成绩。
- fastMRI选定3个序列双mode执行成功，46个2D观察＋3个3D观察，wall42.107秒。使用原tiny pilot，仅验证假定像素网格与原网格输出，未执行4469全量。标注报告在修复2条零面积框后完成，6.283秒，无模型重跑；未标注层不当健康负例。
- 全套CPU2069 passed/5 skipped（346.23秒）；随后正式控制器单buffer校验修复的受影响模块73 passed（30.49秒），独立增量审阅0项剩余。代码、真实结果与原失败均已保留，详见新报告。
- 正式II/III控制器已准备，尚未开训；私有v1登记绑定旧控制器，保持未运行。下一次在唯一工作分支生成最终源码的v2登记并复核，启动cwd与PYTHONPATH都使用原LR5冻结runtime，检查GPU后才开始II4096→III2048。周期loss不是最终A/U/R成绩。

本次按用户要求归档并推送上下文，不启动新的训练。下面是历史记录；其中“最新/下一步”的措辞以本节和PROJECT_CONTEXT为准。

从2026-10-22本次合并起，`main`是代码、已发布实验结果和项目进展的查询入口。新会话先读本页顶部，再读[当前监督能力检查报告](docs/verification/2026-10-22/aur_lr5_supervised_capability/RESULTS.md)及同目录`RUN.json`；LR5/G1父证据见[此前候选报告](docs/verification/2026-10-21/aur_2d_candidates/RESULTS.md)。以下历史章节中的“main不直接推送”是当时的分支约定，已由本次用户要求合并到`main`取代。Git中的运行状态是注明时点的快照；后续实验状态须核对报告记录的私有完成证据后再更新。

## 2026-10-26接续：先完成已批准的验收层

已合并远端Q31–Q34与本地工程验收记录：小集工程检查通过不等于最终做成；最终做成按下述伪参照门及fastMRI像素网格任务判定。执行顺序为先完成[判门与批处理计划](docs/superpowers/plans/2026-10-22-aur-gate-layer-and-fastmri-batch.md)的合成/有限真实验收，再登记并启动完整数据池正式II/III。新正式控制器当前仅准备与测试，未开训。

## 最新（2026-10-22，Claude 会话）：三个子目标的做成口径与验收门已澄清

用户逐条拍板 Q31 到 Q34，全文在 [澄清设计](docs/superpowers/specs/2026-10-22-aur-three-subgoal-acceptance-design.md)。要点：做成 = 2D、3D 都在伪参照上过门，并在 fastMRI 那 4469 个 DICOM 序列上端到端跑通，医生标签另走论文线；用户原话"前序定的指标不一定对，你需要有自己的判断。只需要保证三个子目标一致即可"，据此重定三个门：A 加 32 类逐类存在与最差宿主下限，U 参照重采样到同一 1 mm 网格并加匹配实例掩膜质量，R 从"受控 R ≥ B0"改为端到端宿主一致率（用户"认"）；2D 同一口径，U 参照为同模型 3D 模式在病灶质心层的灵敏度，不训 Dataset909；门在历史 test 上判，评价合同新开一版加 test 角色；后续计划两块：判门层 + fastMRI 批处理，再从 LR5 Stage I 起跑正式 II/III。该设计替代 10-08 N14、10-14 §3.3、10-21 §6 的门条款，旧文不改。该远端规格编写时未启动训练、评估或推理；本次合并同时保留下面本地已完成的LR5监督工程检查记录。

## 最新用户澄清：本轮以跑通工程链路验收

用户进一步明确：“能跑通即可”，小数据量不能用于确认最终指标。按此范围，本轮**工程验收通过，最终性能待完整训练与独立评价确认**：II/III各512步、双模式曝光、有限loss、参数更新、完整CUDA恢复及收尾完整性均已有实际证据。此前为小集登记的质量/保持性阈值仍保留原FAIL数值与裁决，但不再作为后续正式训练的独立硬阻塞；不是将原实验改判PASS。

下一步可进入完整合格TRAIN数据的正式训练计划，固定预算、初始化、学习率和周期评价口径。保留原LR5双模式G1与正式阶段完成证书要求；资源检查checkpoint不能冒充正式阶段权重。本轮已充分提供工程执行证据，不重复小集试验来追逐阈值。小集A/U/R与冻结读出异常作为后续监控和配方检查项，不能仅归因于数据量，也不能据此认定最终模型失败。此次口径澄清没有启动新训练。

## 本轮执行记录：工程完成，原小集诊断阈值未通过

以下停止行为属于当时登记协议的执行记录；后续训练准入以本页最新用户澄清为准。原始完成文件、权重、数据和指标不变。

用户本次已明确要求继续执行。[检查设计](docs/superpowers/specs/2026-10-22-lr5-supervised-capability-design.md)和[实施计划](docs/superpowers/plans/2026-10-22-lr5-supervised-capability.md)已落实，完整结果见[本轮报告](docs/verification/2026-10-22/aur_lr5_supervised_capability/RESULTS.md)/RUN。原LR5正式Stage I/G1、数据合同与145份运行源码保持；两模式各固定16个TRAIN观察，均有全部32个A类和四来源支持。资源检查II/III各实际512步、各mode256次曝光，两次64步完整CUDA状态恢复通过，loss全有限。终态为`COMPLETE / gate=FAIL`，同宿主机确认控制器及worker均退出；这不是正式监督训练，没有正式监督完成证书，**不释放新的正式II/III，也不补训练改判定**。

|mode|A登记诊断Dice|U并集Dice|R受控宏准确率|冻结宿主读出：起点→终点|判定|
|---|---:|---:|---:|---|---|
|2D|0.036654|0.154875|1.000000|0.575331→0.284397|FAIL|
|3D|0.007002|0.153317|0.323819|0.760964→0.416228|FAIL|

U受控匹配mask Dice虽改善，但并集precision仅约0.085、预测区域约GT的9–10倍；不能由病灶聚焦采点的局部改善宣布完整U成功。2D R只在15个合格TRAIN目标/6宿主上达到登记条件，3D为82目标/10宿主，仍低于0.5。固定读出主要在II下降，未证明唯一因果。A是预先登记的presence×mask诊断，与生产纯mask阈值解码不同，不能称原网格A成绩；本轮不事后改口径。下一项先登记A解码/目标/采点覆盖检查与独立骨干保持性控制，不直接加预算或混合因素。

实现`3bb7589`最终CPU1890 passed/5 skipped（301.28秒）、针对性70项通过；代码与新登记独立审查0项待处理。忽略区池化/F2有效性和首次启动的非仓库home/.git误判均已补回归；v1在数据/权重加载前停止，失败与原登记保留。v2仅更新控制器哈希，不变数值协议。接手核对`~/outputs/aur_lr5_supervised_capability_20261022/run_v2/terminal.json`、保存计数与哈希，不重复已完成训练/拟合/前向。收尾完整性已通过；医学数组和逐观察记录不入Git。原G1通过仍成立，但本轮不是临床/独立队列或生产最终A/U/R验收。

## 此前：二维优化与结构候选

用户要求按建议继续后，已实现[两个独立候选](docs/verification/2026-10-21/aur_2d_candidates/RESULTS.md)：LR5只提高四个二维输入/重建tensor的学习率；PLANAR增加三层真正二维合并并保持原学习率。设计和边界见[候选设计](docs/superpowers/specs/2026-10-21-aur-lr-and-planar-merge-design.md)及[执行计划](docs/superpowers/plans/2026-10-21-aur-lr-and-planar-merge-execution.md)。两因素不混合，不改变原G1门槛或评价网格。

两臂正式Stage I均完成2048步，2D/3D各1024次曝光；按原八个验证边界与等权loss规则，均选择step1792。完整双模式G1及控制器最终核验均完成，结论如下（2026-10-22 05:35 UTC快照）：

|候选|2D G1 / 宿主Dice增益|3D G1 / 宿主Dice增益|流程状态|
|---|---|---|---|
|LR5|PASS / 0.077866|PASS / 0.351091|`STAGE_I_AND_G1_COMPLETE_PENDING_REVIEW`|
|PLANAR|FAIL / 0.043607|PASS / 0.352347|`STOPPED_AT_G1`|

LR5通过预先登记的二维增益≥0.05、三个seed区间下界>0和病灶AUC容差门；PLANAR的二维增益仍低于0.05。两臂均有13宿主与完整样本支持。报告、完成文件、所选权重绑定、逐mode判定和终态哈希已经只读复核，未重复训练、拟合或影像前向。**Stage II/III仍未启动**；下一步为审阅LR5的后续监督能力计划。G1是伪参照工程读出，尚不能认定最终A32/U/R、原生DICOM或独立临床评价达标，原失败记录保留。

生产提交`9bb5503`/`1cfdcd5`和整分支`d30947e`均通过独立审查。原完整CPU1820 passed/5 skipped（297.54秒），五项nndet/CUDA环境缺口未计通过；真实历史完成证书兼容。运行树原5资格源码未改，75项独立运行树检查及保存→合成G1通过。真实初始化保留176+4、PLANAR新增9切片一致；两次合成CPU前向的18项3D输出逐位相同。每臂16步TRAIN、step8完整状态恢复及非活动分支检查通过。最终登记审阅0项待处理，105输入承诺/145运行源码/25控制器源码核验后才释放正式实验。详细验证和本次main合并回归见候选报告/RUN。

## 此前：已批准路线的首批实现与诊断

用户已认可[核心目标与路线设计](docs/superpowers/specs/2026-10-21-anatobind-core-goal-and-evidence-route-design.md)，并要求继续执行。[当前实施计划](docs/superpowers/plans/2026-10-21-aur-evaluation-and-g1-diagnostic-execution.md)落实首步：评价入口与有边界的二维诊断。此前未提交的同日短草稿保留，当前实施计划优先。

- 正式format2 II/III权重已接入统一loader，真实完成证书、精确数据身份、完整有限权重在设备转移前验证；format1仍为pilot。合成两模式A/U/R等价及拒绝路径验证通过，68项针对性测试；独立审阅0项。无真实统一监督checkpoint或新医学质量成绩。
- 新`unified_evaluation.py`/`scripts/aur_unified_eval.py`支持固定development_validation观察，分别报告2D/3D的A32/host13、U敏感度和误报、受控及端到端R与同解剖B0。2D面积/FP-image与3D体积/FP-scan分开；未知R不变no-host，TRAIN资格保持。149项相关测试通过，审阅发现的历史父合同环境字段兼容问题已修复（b4244dd，修复检查56通过）；两轮后0剩余。
- 数据校验新增显式evaluation来源路径：验证原合格源码及当前角色/源文件/环境，默认训练仍检查live代码。实际final_v3的5项源码均可由65eb7df精确还原；其中1项没有原本保存的精确副本，私有来源归档明确标为本次审计重建，原合同/哈希不改。
- [二维诊断](docs/verification/2026-10-21/aur_g1_diagnosis/RESULTS.md)与精确回放通过：145份原源码、13项输入承诺不变。实际step0存在；2D输入/输出适配器到所选1792步分别变化1.735%/8.394%，共享骨干4.898%，优化器计数支持确实训练。两份二维缓存有限且非恒定、13宿主均有支持；丘脑读出仍0。没有发现静态归一化/维度/坐标不一致，采样/增广/读出类别不平衡的效应尚未分离。独立证据审阅0项。
- [单次二维起点对照已完成](docs/verification/2026-10-21/aur_g1_step0_control/RESULTS.md)：原selected三个读出逐项精确复现（误差0），实际step0平均Dice0.248889→selected1792的0.279637，差+0.030747；三个48患者组bootstrap95%区间均包含0，不能认定稳定增益或训练不足的根因。144次起点前向/6头拟合、35.611秒，输入/骨干完整性验证通过，独立CPU计数/区间回放差异小于3.89e-8。原G1仍失败、II/III未释放；不要重跑此对照。
- 下一项仅为**PROPOSED**模型干预：四个新二维适配器的学习率倍率5，其余配方与2048预算不变；1792固定描述对比另列，官方G1保留原等权验证loss选点和门槛。尚需优化器分组/旧格式/恢复/源码资格检查和独立登记，未实现或启动。3:1曝光方案不是当前原样可跑的备选，正式runner只接受交替2d/3d。

[首批交付报告](docs/verification/2026-10-21/aur_evaluation_interfaces/RESULTS.md)：最终沙箱外CPU回归1768 passed/5 skipped（282.65秒），实际合同字节/角色/源码复核PASS（198.829秒、合同不变、无模型前向），整分支终审0项剩余。5项跳过为nndet专用环境和4项CUDA检查，未记作本轮通过。沙箱内两次停滞的全套已由各自会话终止并保留日志。原`STOPPED_AT_G1`、旧完整模型A/U/R FAIL、医生标签0、原生DICOM几何和独立队列缺口均保持。

## 此前：统一2D/3D正式训练

[最新验证报告](docs/verification/2026-10-20/aur_qualified_training/RESULTS.md)、[RUN.json](docs/verification/2026-10-20/aur_qualified_training/RUN.json)与[实施计划](docs/superpowers/plans/2026-10-19-aur-data-grounded-gap-closure.md)为当前入口。下面2026-10-19的“尚未实现/NOT READY”是历史状态。

- 执行代码`65eb7df`已完成正式数据合同、观察级2D弱R、纯图像双维SSL、严格warm-start、完整恢复/完成证书与双mode G1。数据/核心/启动器等独立审阅均无剩余实质发现，完整CPU1699 passed/5 skipped/243.92秒。
- 新合同`final_v3`已完成全部源/输出/执行链校验。监督TRAIN3499行/894来源限定患者，SSL TRAIN5501/1895；开发validation405/101，SSL validation629/213。开发ISLES另3个近整数截断mask已在新版本恢复，原生20/20阳性均保留；TRAIN819份补充病灶数组QC完成。患者角色身份交集0，不等于跨来源生物学去重。
- 2D弱R对最终plane/FOV/变换重验真实宿主重叠与可见性。有限TRAIN资格测量39组件中26合格，仍标NOT_EVIDENCE。Stage I严格只读图像；176旧张量复用、4新增二维张量固定seed初始化。
- 全尺寸GPU probe共24步/449.868秒（含首尾全量校验），最大allocated约5.25GiB，CUDA实际状态恢复通过；两mode均实际参与Stage III弱R计算。这是资源与执行证据，heldout模型前向0。
- 首轮正式Stage I已完成2048步，两mode各1024次曝光，loss全部有限；训练/周期验证2474.271秒、输入完整性验证397.913秒。best来自step1792，完整预算由单独证书证明。CPU只读证书、所选骨干、默认G1配方/报告hash及源码闭包复核全部通过。
- **双mode G1实际2D FAIL / 3D PASS。** 2D宿主Dice增益0.043545未达到0.05；CI下界0.009362>0、病灶AUC0.784362对random0.789782满足−0.02容差。3D增益0.349600、CI下界0.324584、AUC0.742439对0.661013，三条件均通过。两mode均有全部13宿主与完整样本支持；这些是伪参照工程读出，非最终A32/U/R精度。见[完整匿名G1](docs/verification/2026-10-20/aur_qualified_training/g1_anonymous.json)。
- `formal_pass_v2`已于03:12:50 UTC按登记规则停止，PID2348457退出，II/III未启动。原预算与全部失败结果保留，不自动降低阈值、换best或追加训练。GitHub新分支已发布实现与证据，远端main保持原版本。
- 最终评价仍有接口工作：正式format2导出尚未接入旧format1统一推理loader；旧3D evaluator不直接兼容新validation角色；周期验证只有loss，held-out二维R资格尚未定义。Task5的训练/G1实现完成，不等于双mode最终A/U/R评价完成。
- 原生DICOM缺IPP/IOP及尺度歧义仍在。Level R工作副本已准备并核验，1297病灶但labels/adjudications/releases仍0，未联系读者。独立评价与最终两mode A/U/R质量验收仍未完成，历史模型质量FAIL保留。

私有证据及运行根为`~/outputs/aur_qualified_training_20261019/`。`formal_pass_v2/pipeline_status.json`为STOPPED_AT_G1；不重新启动已完成Stage I或重跑G1。后续先固定二维表征失败诊断及最终评价接口方案，另立新版本；原数据、角色、预算、门槛与权重证据不覆盖。GitHub发布使用用户指定的新feature分支，远端main不直接推送。

## 此前：实际训练数据与缺口修复（2026-10-19）

用户要求深入理解148的FM_data并弥补缺口。[数据审计报告](docs/verification/2026-10-19/aur_data_audit/RESULTS.md)、[设计](docs/superpowers/specs/2026-10-19-aur-data-grounded-gap-closure.md)与[实施计划](docs/superpowers/plans/2026-10-19-aur-data-grounded-gap-closure.md)优先于下面的旧资产未知描述。

- 全目录342889文件前缀、113409 DICOM header、7174 H5 metadata/XML、55044原始/标注NIfTI header及全部当前引用已核；OASIS436 Analyze标签/强度对已核。内部别名与硬链接不计新增患者，未全审的legacy派生/压缩payload边界见报告。
- 发现35个TRAIN ISLES阳性mask被近整数proxy scaling后的直接int16截断变成空；180TRAIN原重采样逐体素复现。代码`6265015`统一安全离散标签解码，新私有35mask/3499行TRAIN清单恢复770019体素、459个合格组件，70条引用更新；3个原生空DWI仅在新清单关闭U/R。原数据/旧清单/缓存/hash不变；恢复lesion也修正A-ignore。后续须使用RUN.json指明的新manifest，不能继续用旧全零派生。
- 全TRAIN有效A监督1068图/3420行，32类均有有效体素；5/44/18/26/58并非无标签。二维观察级weak-R资格尚待实现，当前真实2D默认R继续关闭。
- Stage I完整176-key SSL权重已验证严格warm-start，骨干与step8500 best导出相同；统一模型4个新增二维tensor未训练。历史RNG不完整，不是exact resume。正式image-only双维SSL、双mode G1、runner/预算仍未完成。
- raw fastMRI112609帧全缺IPP/IOP，H5XML不能补患者affine；5640卷候选关联但FOR全不匹配，76032帧encoded/recon尺度解释不同。800个derived几何对象UID/构栈不合格；不能据此宣称原生脑DICOM验收完成。
- 找到真实Level R库：1297 lesions/165登记患者、2读者，labels/adjudications/releases均0。OASIS416、SibBMS8及fastMRI未入旧outline的5400个元数据ID只是候选，不是已证明的新独立A/U/R队列。
- 最终CPU1528 passed/5 skipped/202.88秒；独立代码review2轮后0项，设计review0项；本轮无GPU/新model forward/正式训练/新val或test成绩。私有数据入口与证据在`~/outputs/aur_data_audit_20261019/`，医学数组和清单不入Git。早期一次源码输出硬编码病例标识的偏差已记入报告，不声称全会话匿名。

下一步执行新计划Task2–6的数据资格/纯图像SSL/双mode G1与正式编排，先测完整模型预算后开Stage I；原生DICOM上游资料与医生R另行补证。旧完整模型质量FAIL仍保留。

## 此前同日：统一输入技术闭环

本轮用户授权执行后续建议，已完成[统一输入实施计划](docs/superpowers/plans/2026-10-19-aur-unified-input-execution.md)中的代码、有限TRAIN pilot、A/R诊断与评价准备。完整结果、原失败与数值限制见[本轮报告](docs/verification/2026-10-19/aur_unified_input/RESULTS.md)和同目录RUN.json。

- **实现：** opt-in `dual_input=True`，同模型/单checkpoint接真实depth1的2D与3D，主体共享、stem/up按维度适配；SSL也支持两维。旧3D默认调用、key和数值输出保持兼容。新增严格classic MR几何入口、源hash/帧映射与原网格推理；enhanced/缺几何/歧义时间或扩散维显式拒绝。
- **真实TRAIN技术pilot：** 三源各1名患者，tiny随机模型，I/II/III各24步，每阶段2D/3D曝光12/12；共72次optimizer step，训练段16.397秒，CUDA allocated峰值约137.4MiB。两维训练loss下降，但不是全模型/泛化/过门结果。二维R无可靠标签时资格关闭，三维合格R与合成二维R分别验证。
- **恢复：** CPU连续/断点完整state逐位相同；CUDA恢复瞬间完整state精确，后续mode/RNG/counters一致。相同CUDA连续训练也会漂移；strict模式受trilinear backward无确定性实现限制。新runner核对数值环境，不承诺未来CUDA参数逐位复现。
- **新A/R测量：** 同旧完整C2、三名TRAIN各一个标准裁块，各一次float32前向。五个历史缺失A类在GT处仍几乎不给概率且胜出率0，不能由全局均值宣称A32正常。R固定192query，仅替换预测A/U几何，2个答案翻转，非准确率。原native replay因距离数值消减FAIL保留；独立版本的稳定距离CPU派生通过，原缓存与容差未改。
- **评价准备：** 重核TRAIN3499行/894来源限定患者、val405/101、历史test1056/249，清单身份交集0；不是跨来源生物学去重证明。未建立新独立队列。默认Level R数据库不存在，未审计其它路径或联系读者。当前抽样DICOM缺IPP/IOP，原生脑DICOM路径/关联仍需核实。
- **科学裁决不变：** 原完整模型的A/U/R测试门仍未过，默认legacy_max不变；无新val/test成绩、Level R人标或临床成功主张。正式训练前仍须全模型初始化/预算与双mode G1、可靠二维R资格及独立评价；网页/API/URL后置。

验证命令、最终CPU/CUDA计数、审阅修复与恢复方式以本轮RUN.json为准。下面保留执行前的历史记录，不能把其中“尚未实现统一模型/没有新训练”的句子当成本轮状态。

---

## 执行前记录：2026-10-19（双输入训练准备分析）

当前阶段：用户已明确构建出的同一个模型兼容2D/3D，输入来自其提供的DICOM且保留全部信息。[统一模型设计草案](docs/superpowers/specs/2026-10-19-aur-unified-2d-3d-dicom-design.md)建议共享主体、维度适配stem/up和单checkpoint，具体结构待审；正式训练仍为NOT READY，尚未启动新训练。下文此前独立2D建议须按此新方向细化。

最近已执行的3D实验：按已批准方案完成原六名TRAIN患者/十二裁块的配对背景采点实验，执行代码73aa2c7。只在诊断U点集把boundary_share0.2改为0，保留16384点、focus_share0.5和原匹配，测量原生值锚定的float32共享掩膜头切线。ISLES主面板8/8裁块×draw的combined单位方向差global−control均为正（均值+1.438208），未采到FP背景的局部抑制更弱，四裁块跨draw均值也4/4同向。当前结果不支持把该改动推进为修复或据此加训；正式采样器/损失/权重未改，没有val/test前向。默认legacy_max、完整405val FAIL及A/R缺口保留。

最新路线记录 `docs/plans/2026-10-19-aur-dual-input-route-and-training-readiness.md`；最近3D实验报告 `docs/verification/2026-10-18/aur_u_background_allocation/RESULTS.md`；同目录RUN.json、aggregate_anonymous.json、supplement_anonymous.json、numeric_audit.json。目录日期来自2026-10-18冻结方案，实际执行于2026-10-19。上一轮U监督报告仍为 `docs/verification/2026-10-18/aur_u_train_supervision/RESULTS.md`。更早分数排序上界继续限制既有最终标量的单调校准拟合，不排除新特征、改变排序或更好掩膜。

## 输入范围与覆盖状态

**已记录的最终目标包括2D MRI图像和3D MRI体数据的完整A/U/R输出。** 双输入要求至少已在2026-10-13提交`20fdec4`入库；[原规格](docs/superpowers/specs/2026-10-14-anatobind-brain-dicom-web-design.md)明确区分独立单张2D与3D series，提议独立checkpoint并分别验收。双输入是既有要求，不是2026-10-19新增范围；checkpoint策略属于技术提案。此前入口摘要只写3D，现已纠正。

| 输入模式 | 当前覆盖 | 未完成内容 |
|---|---|---|
| 3D | C2模型训练、既有val/test和最近机制诊断已执行 | A/U/R性能关卡尚未全部通过；AUR DICOM几何读入及离线端到端验收仍待完成 |
| 独立2D | 既有规格和实施任务已记录；已有独立2D检测组件不等同于完整2D AUR | 专用数据/参照、完整模型/checkpoint、训练与A/U/R验收仍未完成 |

当前C2和本页最新实验结果均属于3D工作线，不代表双模式目标完成。Q28优先模型成功、后做显示；[计划中的范围保留说明](docs/superpowers/plans/2026-10-14-anatobind-brain-dicom-web-execution.md)保留DICOM、离线验收和独立2D任务，网页/API/公开URL后置。

## 技术路线迭代与正式训练入口（2026-10-19）

[完整迭代记录与就绪性分析](docs/plans/2026-10-19-aur-dual-input-route-and-training-readiness.md)已补齐：原3D方案→SSL-first修订→Q27双输入要求→2D独立checkpoint技术提案→3D失败与诊断→当前2D训练缺口→同日用户明确统一模型与DICOM完整信息保留。旧独立模型提案不再优先推进；共享范围、初始化和联合训练配方仍须审阅冻结。

- **只读数据核对：** TRAIN3499行/894来源限定患者，validation405/101，历史test1056/249；三对身份交集为0。4960行的image/anatomy及3887条lesion路径引用存在且在许可根内；未载入影像数组。旧清单缺失的A/R标志按实际loader默认True，2D新清单应显式化。来源限定身份不等于跨来源生物学去重。
- **当前NOT READY：** `aur2d`完整代码、2D清单/splits、Dataset909、2D训练/恢复/评价均未落地。旧2D计划只有II/III而无兼容初始化；是否以及如何补齐2D Stage I须在新训练设计中冻结。现有3D G1和C2权重不是2D初始化证据。
- **建议的下一执行块：** 冻结架构/初始化/二维目标与评估规则，建立真实2D纵向训练闭环，通过有限smoke/小集过拟合/短pilot后冻结正式配置，开始Stage I，满足2D阶段门再进入II/III。原batch/LR/预算尚未实测；不要求最终A/U/R已达标才允许首次训练，也不继续无截止地重复同类3D探针。
- **状态边界：** 本轮完成记录与就绪性分析，没有新训练、2D平面生成、权重转换或新模型性能结论。历史test已经暴露；新的二维切片不能包装成新的独立测试患者。DICOM、原生2D泛化与最终产品验收仍保留，网页/API继续后置。

## 0. 最新背景配额对照与复用

- 主面板排除两臂采样点并集，只测原匹配query的官方FP有效真背景；先query内合并组件，再按crop/draw求均值。ISLES56条query×draw、8/8crop×draw有支持，control−0.859799→global+0.578410；原始combined及U-only单位方向差也8/8不利。仅局部切线，不是FP数量变化或完整网络/历史训练梯度。
- ISLES远处背景采点38536→64225，但已匹配FP真背景唯一覆盖体素×draw3085→1729；PDGM覆盖33→52而方向仍不利（4/8crop×draw有支持）。BMSR已匹配FP面板无支持，未匹配query路径仍独立保留，不能当成零效果。
- ISLES分配目标logit方向差均值+1.224134；A损失方向差均值−0.004193但5/8较好、3/8较差。PDGM/BMSR的A差均值为正，即较少下降；两臂combined单位方向下的全部A总损失方向仍为负，不是实测A Dice损害或修复。原固定query R logits、十二裁块完整模型状态完全相同，这是受限路径控制而非R性能提升。
- 重新按干预点官方匹配后5/106目标×draw换query（均ISLES，5/86）；不是前轮两个原draw之间的5/53，主分析保持原匹配。106条两臂均无有效正点缺失，无容量溢出。
- 完整CPU1345passed/2skipped180.61s；独立任务/整分支代码审阅0项问题。实际A800物理7两次空闲快照/兼容性通过后运行161.126s、退出码0，最大allocated2.510GiB/reserved3.977GiB、RSS2.860GiB。先前候选物理5在启动前已忙，保护检查在模型/影像加载前停止；既有任务保留。
- 纯CPU完整回放15.095s、退出码0，27成员字节不变，显式阻止模型/影像/checkpoint加载。独立脚本核对12crop/24draw/680ROI，69297项检查通过；未独立重新生成密集特征均值，其来源另由runner密集流式约简和输入/状态控制核验。TRAIN资格/完整源承诺由controller固定脚本核验，不混称独立溯源。
- 私有完整缓存 `/home/congcongliu/outputs/aur_u_background_allocation_20261018/run_73aa2c7`，只用隐藏CUDA的 `scripts/aur_diagnose_u_background.py --out <该目录> --replay-only` 复算，不重复GPU测量。128项原依赖、checkpoint和旧29成员参考存档不变；新医学派生数组/逐例记录留在0700/0600目录，不入Git。当前新增公开文件和实际固定stdout另行扫描，不能扩展为历史会话匿名保证。

## 0.1 上一轮TRAIN U监督诊断的读法与复用

- 来源异质：BMSR的6个FP在两次采点均来自未匹配query；ISLES100个FP对应176匹配/24未匹配query的FP组件×采样记录；PDGM7个FP对应6/8。不能把ISLES主导的全局模式直接概括到BMSR。
- 106个目标×采样均有有效正点，未遇query容量溢出；5/53裁块目标换了query（都在ISLES）。这些是新固定TRAIN裁块/CPU点draw，非历史分布式训练流或参数梯度重放。
- 私有缓存 `/home/congcongliu/outputs/aur_u_train_supervision_20261018/run_1a7de61` 已完成；只需隐藏CUDA运行 `scripts/aur_diagnose_u_supervision.py --out <该目录> --replay-only` 做只读完整复算。当前source/spec/原始输入/选择/权重承诺必须一致；报告提交导致HEAD前进但代码闭包不变可复用，不要重跑GPU补已完成的测量。
- 模型状态锚点5fbb0d6f9513fdebf66831033c6a8d32710a05ce2ec900656c062a61546e482f，12裁块前后均一致；原生产Python文件未改。启动至完成标记约111.467秒上界，allocated峰值2.609GiB、reserved3.977GiB，实际CPU回放29.87秒。不是推理速度比较。
- 两次子任务工具输出误带私有标识（上下文选择记录、审计清单路径键），第二审计任务已停止。新/改文件、匿名汇总与实际runner日志另行核验；不能把这些检查写成整个会话终端匿名。私有医学派生数组与逐query/组件记录继续留在0700目录，不入Git。

## 1. 三个目标及当前裁决

**A32解剖掩膜、统一异常U实例掩膜、每个U的R13分侧主宿主+no-host（14概率）不变。** S仅为内部序列上下文。默认解码仍为legacy_max；可选query_cc_v1只保留为可复现实验，不能当作已通过的模型修复。

| 目标 | 既有完整val结果（本轮未重跑） | 状态 |
|---|---|---|
| A | 394个A监督行的13-host Dice0.803457；405行保存分数逐项完全不变 | val宏门通过；五类缺失仍未解决，最近历史test0.7988仍不过 |
| U | BMSR/ISLES/PDGM-FLAIR检出2→78、4→60、27→45；FP/scan变为17.0、540.1、4.7317 | 三源均超S7预算，也均低于参考敏感度减0.05；该固定profile不采用 |
| R | 716个controlled实例的共享保存字段完全不变；ABA0.568436对B00.698324 | controlled门仍失败，未修复R |

前轮排序报告 `docs/verification/2026-10-18/aur_score_order/RESULTS.md`，同目录 `RUN.json`、`aggregate_anonymous.json`、`manifest_exposure_anonymous.json` 保存命令/哈希/匿名结果与患者清单资格审计。置信度测量仍见 `docs/verification/2026-10-17/aur_residual_confidence/RESULTS.md`；完整405行对照仍见 `docs/verification/2026-10-17/aur_query_cc/RESULTS.md`。那六例为已暴露开发val，与本轮固定TRAIN六例不同；均仅机制诊断。A/R采用伪参照，整体NOT_EVIDENCE。没有Level R人标、独立校准/总体过门或临床成功主张。

原六例固定阈值下，parent→子raw均值的事后过滤结果：BMSR命中2/13不变、FP16→23；ISLES命中11/40→10/40、FP1511→171；PDGM命中2/2不变、FP6→0。三种分数校准不同，反事实保留全部原mask/NMS/ownership，仅过滤后重匹配，不是新的正式解码性能结果。

本轮使用同六例S7预算0.5/1.5/0 FP/scan（总FP≤1/3/0），参考命中8/13、18/40、2/2；不能混用完整405预算0.8780/1.6/0.3659。三个分数上界相同不代表全部排序相同；单调最终标量映射不增加可保留集合，但此结论不覆盖体素变换或新特征。S7native/AUR1mm差异仍在，不作新正式门裁决。

## 2. 已完成的实现与控制

- 最新排序分析代码88c38b6，38个冻结依赖保持一致；原固定工作点和A/U/R存档控制逐项复现，17,350个集合全部按原Hungarian+ignored规则重匹配，不保存最大化阈值。无模型前向，存档不变不等于新A/R质量提升。
- 前轮置信度测量代码6b26137，原推理/模型/训练36个依赖文件与冻结1874源码一致。7528个子实例测量parent/raw/fused三项分数；原U行、GT、解码统计、legacy检测、六例A保存分数、55个R共享字段及query/window来源精确复现。两遍当前A/U图和上下文精确相同，历史未保存图不作跨轮声明。隐藏GPU的完整缓存恢复核验通过，未加载模型。
- 冻结实现代码1874af2，checkpoint SHA256 `38f6376333041a75be1cce83e927bc221468761f80ae952e2a3020805f7dcbde`；实际best选自Stage III step3000/72,000crops，80,016crops为已完成训练预算，权重本轮未改。
- query_cc_v1：独立窗口/query的26CC，presence>0.5、mask>=0.3、共享10mm³floor、原mask IoU>=0.5贪心NMS、按固定优先级归属体素后再拆残余组件，子组件继承父分数；无GT挑选、无候选上限、无回填。detector来源和重裁块后实际R query来源分别记录。设计与计划在 `docs/superpowers/{specs,plans}/2026-10-17-aur-query-cc-decoder*`。
- 405/405保存A分数严格相等；716/716controlled R共享字段严格相等。旧缓存未保存A体素图、controlled query/window来源，不能声称这些未保存对象也已逐项验证。
- 旧端到端缓存先匹配再过滤分数的问题已在本对照中统一为先固定阈值过滤、后一对一bbox IoU>=0.1匹配；2个保留检测恢复GT关联，旧匹配池38→40，预测未改变。原始缓存保留。
- 716个controlled合格GT的正确端到端宿主27→128；共同匹配的40个GT正确数27→26（0rescue、1harm）。新增成功主要来自检测覆盖，不能说R头质量提高。U共718个非忽略GT，有2个没有controlled记录；分母分开报告。
- 上一轮TRAIN U监督诊断完整CPU测试1244 passed、2 skipped，171.70秒；新四模块集成86项通过，独立代码/数值审计0项未解决。隐藏GPU、nice19、线程2，沙箱外解决DataLoader多进程环境限制。前轮排序分析的1170/2（163.48秒、19项针对性）、更早1151与1111项及A/R审计保留为对应版本历史，不能混写为本轮新性能结果。
- bf2a924历史CPU跳过项曾另行补测：nndet环境坐标模块2 passed（2.39秒），nvgen CUDA混合精度单测1 passed（19.06秒），均0失败/0跳过。历史CPU原计数不改；这是一个模块加一个单测，共3条用例，不是本轮新补测。nnDet6条第三方依赖警告保留；合成数据测试不构成新的A/U/R性能结果。证据 `docs/verification/2026-10-18/aur_score_order/supplemental_tests.json`。

## 3. 误报与 A/R 机制线索

- 数值源分数固定BMSR0.70、ISLES0.60、PDGM0.75；raw父组件均值与legacy融合图均值校准不同。本轮比较不是等观测FP，也没有搜索阈值；不能由此否定所有query保留方案。
- 完整val的既有事后形态分析：ISLES10,802个FP中，84.06%子组件保留父体积<=10%，89.98%来自多子组件父组件，子体积中位数36mm³；该完整集当时没有保存子概率。本轮六例直接测量：ISLES原1511个FP的parent/raw均值为0.63888/0.46675，1350个raw低于0.60；但BMSR原16个FP中14个保留比例>0.5且raw中位数0.71872，自身仍过0.70。直接换raw不能统一修复误报；没有把六例统计外推为完整val统计。
- A标签覆盖审计：8个TRAIN行、16裁块、32点采样；共享采样42.282%点落在A忽略区，uniform为1.350%。密集存在但采样无正点时，实际mask损失推动空预测；18/54在病灶裁块的正点均值及零正点频次相同，尚不能据此建立单侧18缺失的因果解释。见 `a_supervision_audit/REPORT.md`。
- R粗投影与dense投影独立，无直接A/U掩膜或一致性监督；Stage II粗投影与seed0初始化逐位相同，Stage III后改变。共享coarse像素金字塔接受A/U训练，不能称整个粗分支未训练。CPU权重/config/历史源码验证已由root和独立审阅者重跑。见 `r_coarse_supervision/REPORT.md`。

## 4. 下一项模型工作

1. **先收敛2D训练准备方案（建议，尚未实施）**：明确是否同权重、兼容初始化/Stage I、平面与目标、模型/loss/恢复/评价闭环；按上面的有限readiness流程准备正式run。不要把旧提案当用户已定配方，也不要以3D已最终达标作为未经决策的2D开训前提。
2. **背景配额对照也已完成，保留原采样，不据此加训**：当前ring0.2→global0候选在固定ISLES主面板的局部方向不获支持；不要在这些裁块上挑选中间配额或直接追加训练。下一项U干预须重新冻结要区分的空间权重/表示/优化机制及A/U/R回退检查；BMSR未匹配query路径与A/R独立任务仍保留。清单U-supervised TRAIN761来源限定患者/1216行，val86/143，历史test212/354；基模型已见TRAIN，未建立新独立评价队列。当前实验不证明U优先级全局最优。默认legacy_max和完整405行FAIL不变。
3. **R单独做固定token/query/crop/权重的几何来源对照**：仅将预测coarse A/U几何换为同一F1网格上的预测dense A/U几何，预先固定池化、valid、gate和坐标规则，记录逐特征和宿主logit变化。联合A/U替换不能区分二者贡献；冻结R头的分布变化也限制阴性结果解释。该实验尚未执行。
4. **A用同一冻结TRAIN裁块审计全部32query的dense概率和胜出类别margin**，分开检查正点遗漏、身份/优化塌缩及类别竞争，再设计单一采样或损失变量。不要仅据18/54相同的正点均值及零点频次直接跳到改loss。

这些是下一步诊断/实验，不是已完成的修复。三个模型目标始终优先；N7、DICOM/独立2D、网页/API/URL及Level R实际读者工作仍按既有顺序处理。

## 5. 资源与别重做

- 新排序分析仅CPU：17,350集合、wall55.38秒、脚本49.68秒、RSS847320KiB，CUDA隐藏/nice19/线程1，600秒内外限时；不重复拟合仅改变同一标量刻度的模型来期待改善可达排序上界。已规划SibBMS10人子集含7TRAIN/3历史test，不能包装成新评价；来源限定清单互斥不代表已证明跨来源生物学去重。
- 新六例回放已在空闲A100物理卡3完成：成功wall186.79秒、行时间合计175.66秒、峰值CUDA allocated9.0176GiB，CPU nice19/数值线程1/decoder4。首次外部启动器参数转发失败发生在模型加载前，2.57秒；修正入口并核验help后重启，日志保留。不要重新跑这六例来补已完成的数值测量；相同冻结合同可直接验证恢复。
- 完整val4302自然窗口，累计冻结单行推理4051.38秒，峰值已记录单行CUDA分配9.017GiB；并行墙钟跨度约4282.42秒含调度/空闲，不能与缓存legacy时间作速度对比。六例NMS加速前后全部保存结果相同，实测507.78→406.60秒为共享主机观测。
- 先检查分片的source/sequence/U监督分布。排序后步长2使全部20个DWI落一端；本次保留已完成odd202行，验证原始全部输入哈希后复用even68行，最终以1/2、0/4、2/4三完整分片合并202+102+101=405，原0/2partial不入结果。见 `scheduling/repartition.json`。
- 恢复必须保持code/checkpoint/所有原始输入/规则/参考哈希；重新分片还必须核对未完成行的原冻结输入哈希。逐例/逐query表、影像与Level R盲读材料只留本地private目录，不能入git。
- 数据只读/data2/congcong/data/FM_data；nvgen绝对路径、PYTHONNOUSERSITE=1；CPU测试隐藏GPU。GPU按用户常设授权，先查实时负载/显存/活跃任务及调度可见性，选兼容空卡并保留既有工作负载。
