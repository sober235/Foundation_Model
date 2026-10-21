# 统一2D/3D训练的数据资格与执行闭环

基线：`04638a9b45ceef82e2899e9a7cc575c914681661`。执行分支：`feature/aur-qualified-unified-training-2026-10-19`。任务来自[已审实施计划](../../../superpowers/plans/2026-10-19-aur-data-grounded-gap-closure.md)，原数据审计和35例TRAIN标签恢复见[此前报告](../../2026-10-19/aur_data_audit/RESULTS.md)。

## 已完成的数据资格

新合同使用私有`~/outputs/aur_qualified_training_20261019/data_contract/final_v3/contract.json`，SHA256 `a839cfc9c9213b5b376eac4bc4a8219506175fdbe3620bc36cb2056f602b6060`。完整输入、输出和冻结执行链已重验。原始数据、历史清单、修复版本与旧缓存保持原文件；新医学清单、掩膜、逐患者记录均保存在私有目录。

| 角色 | 行数 | 来源限定患者 |
|---|---:|---:|
| 监督TRAIN | 3499 | 894 |
| 图像SSL TRAIN | 5501 | 1895 |
| 开发validation | 405 | 101 |
| 图像SSL validation | 629 | 213 |

患者角色身份交集为0，未证明跨来源生物学去重。Stage I清单严格只保留图像与必要元数据，不携带A/U目标路径。监督TRAIN有效A/U/三维伪R资格行分别3420/1213/1174；开发validation分别394/143/138。这些R资格不是医生真值，也不自动授权所有二维平面R。

本轮对819份去重、尚未由既有修复覆盖的TRAIN病灶图完成数组检查：BMSR318、PDGM359、ISLES142，涉及2674行；验证有限值、离散类别、格点与序列可见值，并绑定实际字节哈希。另35份修复图和3份已知原生空图沿用冻结证据，不重复转换。2620行目标状态为源阳性，54行PDGM T1c为序列目标值缺失；后者保留既有序列参照资格，不能称健康患者。先前anatomy QC与当前3499条引用逐字节对应；不误用anatomy哈希充当lesion证据。

开发validation的20名ISLES患者在读取原生标签前已冻结全体成员及规则：原生20阳性、旧派生17阳性；另3图受近整数截断影响，新私有版本恢复70735体素、35个≥10mm³组件。405行开发标签资格均已检查，没有读取模型预测或历史test密集标签。

初版资格执行覆盖8405个源文件、84967568534字节，耗时748.05秒；819份补充数组QC耗时625.985秒。初版存在legacy缺省R开关误置false的问题，保留原产物；后继元数据版本恢复loader缺省True并与A/U资格取交集。`final_v1`、`final_v2`及其失败/历史验证记录保留；最后三轮审阅修复完成后发布`final_v3`，不重写旧代码承诺。上述两个耗时属于各自实际阶段，不拼接为未经计时的总wall。

## 二维弱关系与纯图像SSL

`planar_relation_policy="geometry_weak2d_v1"`按源完整实例的真实解剖重叠和最终plane/FOV/变换重新确定资格。源宿主图0=非宿主、1..13=宿主，R目标0..12有效、13为NO_HOST。最近距离回退、并列、源与plane主宿主不符、不可见或未知监督均关闭资格，不改成no-host。病灶下伪图重叠与A-ignore外可见宿主两个条件独立统计；源实例拆分后的身份仍可追溯。合格目标one-hot、固定权重1.0，标明NOT_EVIDENCE；默认未配置时仍不启用二维R。

冻结6名TRAIN患者、18个header四分位平面，原始/增强共36个观测，得到39个可见组件、26个合格weak-R。BMSR4/10、ISLES16/23、PDGM6/6；拒因是源主宿主并列2、plane与源不一致11。11个合格目标相对旧源soft target发生变化。这是有限TRAIN资格覆盖测量，不是精度或原生二维泛化结果。耗时12.893秒，无模型前向。

新增SSL loader只读取影像；二维由完整未裁剪原平面归一化、真实depth1，三维复用旧数值路径。CPU实际迁移检查确认176个旧Stage I张量逐位复用、4个二维张量按固定种子初始化，160个骨干张量与独立best导出一致。五来源各一份冻结TRAIN影像，两mode共10个crop有效、有限，标签访问0。此处无GPU或真实模型前向；warm-start重置优化器/调度器/新曝光，不冒充旧RNG完整恢复。

## 训练、恢复与G1实现

新增正式单进程编排、梯度累积、source→patient→sequence→session抽样、按mode记录来源曝光、周期验证、best/last与不可覆盖的checkpoint。HCP cap约束每个source draw的概率，不能将小batch的实测比例冒充精确配额；非HCP患者不足以保持唯一身份抽样时预先拒绝。

验证保持训练模型状态及全部训练RNG/计数不变；恢复核对数据/源码/后端、模式cursor和实际曝光。best可以早于最后训练步；单独完成证书绑定真实selected/final文件、同一执行ID/初始化lineage/数值环境、完整优化器与RNG，以及已执行预算。导出已写而完成记录尚未写、末步超时漏验证、best指针写入中断和日志未提交尾部均有恢复路径；不会重做已完成optimizer step。

双mode G1对照同架构随机骨干，冻结共享观测，仅训练读出头。新工程配方固定每mode96校准、48验证、48+48病灶观测，seeds0/1/2、300读出epoch、1000次患者组bootstrap；13宿主均须有验证支持。host SSL−random≥0.05、bootstrap下界最小值>0、lesion AUC_SSL≥AUC_random−0.02，两臂AUC须有限且有正负例，缺失为INSUFFICIENT。二维使用面积等效圆直径，三维使用体积球径。host readout排A-ignore；病灶positive保留feature-valid，正常ROI排除任何A-ignore体素。报告绑定合同与所选骨干hash，重算判门且核对seed/均值一致性。

较小合成配方不能释放正式I→II；resource probe也不能作为正式阶段完成。正式Stage I采用已验证warm-start，两个mode G1过门后才能进入II。医学R真值和独立临床队列仍依赖各自资料，不由工程门替代。

## 审阅及当前验收边界

Task3二维R与Task4 SSL最初独立审阅均0项。训练核心发现2项P2（单模式曝光预算、HCP概率重归一化），修复后增量审阅0项。数据合同发现3项P2（角色输出缺少强制承诺、补充QC执行链不完整、角色链接偏离规范路径），三轮后0项；G1独立审阅0项剩余。启动器发现P1导出/完成记录崩溃窗口和P2完成证书状态不完整，修复后第三轮0项。审阅不替代实际训练或性能测量。

私有证据根：`~/outputs/aur_qualified_training_20261019/`，包含分任务源码快照、合成RED/GREEN结果、真实数据合同、冻结资格观测与审阅报告。公开提交只含规则、匿名计数和hash。旧test失败裁决、DICOM缺IPP/IOP及尺度歧义、Level R空标注库的限制继续有效。

## 全尺寸资源测量与正式预算

执行代码为 `65eb7df8bd15d8155a6df389f525354020cd4dab`。全尺寸probe使用实际共享模型：Stage I有13,328,240参数，II/III有22,753,513参数；2D裁块为1×160×160，3D为128×160×160，batch1、累积1、float32。在检查可见性、调度变量、显存、利用率和活跃compute进程后，选择空闲A800 80GB物理卡0，并在完整输入校验后再次检查。

I/II/III各8步，每阶段两个mode各4步。24步optimizer合计43.282秒，包含首尾全部数据字节校验的wall为449.868秒，低于1800秒上限；首尾校验分别194.656/193.817秒。最大CUDA allocated为5,637,044,224字节，约5.25GiB。

| 阶段 | 2D平均step秒 | 3D平均step秒 | 正式登记steps | 每mode观测预算 | 训练段上限秒 |
|---|---:|---:|---:|---:|---:|
| I | 0.7453 | 1.6672 | 2048 | 1024 | 5400 |
| II | 1.4032 | 2.7519 | 4096 | 2048 | 14400 |
| III | 1.4628 | 2.7901 | 2048 | 1024 | 7200 |

Stage I第4步保存后在CUDA实际恢复，模型、optimizer、scheduler、RNG、cursor、曝光、来源及执行身份逐项一致，再继续至第8步。不承诺之后CUDA更新逐位确定。Stage III实际有2D/3D弱R监督实例3/2，证明两维关系路径参与计算，不是关系准确率。

probe只使用TRAIN观测与TRAIN控制验证，heldout模型前向为0。首个`full_model_probe_v1`因重复全量校验造成预计wall不合适，在任何训练步之前停止并保留ABORTED记录；成功的v2直接使用已审`UnifiedRun`核心，所有阶段共用一次完整前后校验。正式流程使用经过恢复/完成证书测试的`run_stage`启动器；不能将核心probe写成正式阶段已经完成。

第一轮正式预算在新G1结果产生前登记。Stage I为LR1e-4、warmup128；Stage II沿用backbone/heads LR5e-4、warmup1000、backbone前1000步scale0.1；Stage III为backbone5e-5、heads2.5e-4、relation5e-4、warmup200。Stage I每256步验证，II/III每512步验证，各mode8个验证batch；所有阶段每256步存checkpoint。contrast_weight=0与microbatch1一致。预计纯optimizer约4.3小时，验证、完整文件校验与共享机负载另计；此估时只依据各阶段8步测量。这是首次有界全尺寸pass，不等于复现旧240k/80k曝光预算。

私有控制器v1及未启动登记保留。v2修复重复恢复时已有G1判定的写入：内容相同复用、不同拒绝；重复进程锁拒绝不会覆盖正在运行的状态。三项控制检查通过，除controller hash外两版登记完全相同。独立预算/控制器静态审阅0项可操作问题，145个源码hash与执行代码一致；此审阅没有重跑GPU或读取医学数组。当前登记为`formal_pass_v2/registration.json`，SHA256 `fd21ff18abff1ca33599d97413ff513fb774394cad429f1aa2306d354b670cb8`。

正式流程于2026-10-21 01:53:39 UTC脱离会话启动，PID2348457，先执行Stage I输入完整性检查和2048步训练。它从已核旧Stage I step8500重新warm-start，不使用probe的8步权重。双mode G1按默认完整配方运行，wall上限7200秒；任一模式FAIL/INSUFFICIENT即保存结果并停止，不调整门槛或自动追加训练。运行进度是带时间戳的快照，见[RUN.json](RUN.json)和[匿名汇总](aggregate_anonymous.json)；启动事实不代表阶段或质量完成。

## 最终检查及外部补证

完整CPU套件实际结果为**1699 passed、5 skipped，243.92秒，退出0**。命令为`env CUDA_VISIBLE_DEVICES= PYTHONNOUSERSITE=1 PYTHONPATH=. OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 nice -n 19 /home/congcongliu/anaconda3/envs/nvgen/bin/python -m pytest tests/ -q -rs -p no:cacheprovider`，在沙箱外运行以支持DataLoader多进程。跳过项为nndet专用环境模块、1条CUDA混合精度检查及3条CUDA恢复检查；本轮未把历史补测记为新执行。上述实际float32 CUDA恢复probe另行记录。此前最终套件尝试曾因G1合成fixture缺少新强制合同字段出现1 failed/1698 passed/5 skipped；补齐fixture后取得这里的最终结果，没有隐藏该失败。

Level R原SQLite只读，使用SQLite backup创建新私有工作副本并通过integrity_check，原文件hash不变。现有1297病灶、2594条orders、3个账户（其中2读者）；labels/adjudications/releases仍全部0。未启动标注服务、未发送读者消息、未生成医生答案。已有5640卷H5–DICOM候选关联及其hash已冻结复核；它不提供缺失的患者IPP/IOP，原生脑DICOM验收仍未完成。

尚未完成的是正式阶段及双mode质量判门、独立评价患者资格、原生DICOM可核几何和真实医生R标注。历史完整模型的A/U/R失败裁决保留；新2D weak-R、训练loss或工程G1都不能替代这些证据。

公开新增内容经过身份/path交叉扫描、JSON解析、Markdown本地链接及冻结源码检查，结果见[匿名发布检查](verification_anonymous.json)。检查仅覆盖本次新增内容，不扩大为历史会话全匿名保证；源医学数组、逐患者/逐观察结果与工作数据库未入Git。
