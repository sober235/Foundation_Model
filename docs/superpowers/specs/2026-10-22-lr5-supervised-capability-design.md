# LR5：有界 TRAIN 监督能力检查

日期：2026-10-22。用户在已了解 LR5 双模式 G1 通过、II/III 尚未启动后明确要求继续执行。本设计细化已认可的[核心路线第三步](2026-10-21-anatobind-core-goal-and-evidence-route-design.md)，数值选择是本轮实施决定，不冒充用户指定配方。

## 目标与证据边界

从 execution_v5/A_LR5 的正式 Stage I selected_export 严格继承同一全尺寸骨干，在冻结的合格 TRAIN 小集合上依次执行有限 II、III，分别检查 2D/3D 的 A32、U 实例和合格 R 是否能够学习。Stage II 冻结关系模块，所以 R 的学习判定只来自 III。本轮为 TRAIN 工程优化检查，全部结果标记 NOT_EVIDENCE；不签发正式监督完成证书，不使用开发验证/历史 test，不自动接正式 II/III。

原 Stage I、G1、数据合同、初始化源码与历史失败均不可更改。LR5 的 5 倍倍率只属于 SSL 的四个 IO tensor，不能迁移为监督阶段的新倍率。保持模型、损失、点采样和默认解码不变，不组合 PLANAR。

## 源码与初始化

外部控制脚本和度量工具放在 scripts/，单独归档并记录哈希，在原 execution_v5/runtime 中运行。完整原运行树 source_snapshot 必须等于 LR5 身份 runtime_sha256；五项资格源码依然由原 read_contract 做 live 校验。当前工作树不能冒充旧运行树。严格核验 Stage I 完成证书、G1 完成/报告哈希、双 mode 门与骨干绑定，然后才构建 II。

II 使用 purpose=resource_probe 并保留实际正式父证据；III 从本轮 II 最终完整模型继承，二者都不通过正式推理 loader 发布。能力检查 checkpoint、模型状态和恢复试验继续使用 UnifiedRun，无新参数转换。

## 冻结观察

只从 read_contract 返回的 train_supervised/train 行生成 QualifiedViews(augment=False, validation=False)，使用父 run 的 seed、模型、crop、16384 训练采样点和 geometry_weak2d_v1。候选扫描每 mode 最多 64 个确定性 draw，每 mode 选最多 16 个观察。先按真实标签覆盖确定集合，再进行任何模型前向；不按模型分数选例。

选择为确定性的贪心覆盖：每轮优先新增有效 A 类，其次新增合格 R 宿主、来源和 U 实例支持，平局取较小 draw。来源计数只用于覆盖检查，不进行患者独立性主张。每 mode 必须覆盖全部 32 个非忽略 A 类，至少 4 个有效 U 实例、两个合格 R 宿主和三个监督来源；所有观察需有非空有效背景。若有限池无法满足，写 INSUFFICIENT 并停止，不把缺失类视为成功。

候选批次、draw、张量哈希、行身份承诺仅保存在 Git 外 0700/0600 私有目录；公共记录仅匿名计数、聚合指标和哈希。训练按固定列表循环，2D/3D 严格交替；两阶段用相同观察。任何私有路径、patient/case 字段、清单键值或逐观察结果不输出到工具。

## 预先固定的预算与配方

- II：512 optimizer steps；backbone=5e-5，heads=5e-4；warmup=32；backbone_warm_steps=0、scale=1，不在第1000步释放到5e-4。
- III：512 steps；backbone=1e-5，heads=2.5e-4，relation=5e-4；warmup=32；其余保留父数据/损失约定。没有学习率扫参。
- 每阶段分别在 0、128、256、512 步测量固定 TRAIN 集；最终步为唯一结果，不按检查点挑 best。每64步保存完整状态，首个64步边界执行一次销毁/恢复与完整序列化状态相等检查；后续 CUDA 更新不承诺逐位确定。
- 标签扫描上限1800秒，每阶段训练/测量段上限3600秒，整个控制任务上限10800秒。外部 watchdog 覆盖阻塞调用，并在父死亡时结束子进程。超时/非有限值/输入或源码变化即停止，保存原失败，不自动续跑或加预算。
- GPU 按现有授权选择兼容空闲 UUID，保留调度/可见性约束和原有任务；CPU nice19、数值线程2。全尺寸原 crop 保持 2D=1×160×160、3D=128×160×160。

## 度量与继续条件

A：有效网格上32类的 truth/prediction/intersection、逐类 Dice 和支持类宏均值。用 presence 与 mask 概率乘积、阈值0.5和逐体素最大类得到预测；忽略 A-ignore 与无效网格。U：固定均匀+病灶采点的 Hungarian 对应下，mask sigmoid>0.3 的硬 Dice 按真实实例等权求均值，匹配 query 的 presence>0.5 比率、未匹配活跃 query 数另列。全网格并集由 presence>0.5 的 query 的 mask>0.3 合并，报告其 Dice；不使用 presence 过滤受控匹配 Dice。它们是裁块/受控 query 指标，不称 FP/image、FP/scan 或完整解码检出率。R：只计 r_eligible、r_supervised 的真实匹配目标，报告交叉熵、正确/合格数、宏宿主准确率及多数宿主基线；未知目标永不改成 no-host。固定度量点独立于训练 RNG，度量前后恢复完整 RNG 与 train/eval 状态。

保持性：在 II step0 的同一 TRAIN 观察 F2+F3+F4 特征上，每 mode 只拟合一次类别均衡 ridge 宿主读出（ridge=0.01，标准化参数固定，最多每观察512个有效格点，按格点索引奇偶拆拟合/观察）。格点宿主真值沿用 G1 的有效区域多数池化，忽略非宿主；在各宿主内按格点索引均匀选点。每个 mode 的全部13宿主在拟合/观察两侧都必须各有至少2个格点，否则 INSUFFICIENT。之后冻结读出和格点，在所有边界复用；无原 G1 重拟合，也无 heldout 前向。报告固定13宿主支持的读出 Dice 与特征相对变化。这是 TRAIN 内读出保持性检查，不证明泛化；不以漂移量冒充解剖保持。

每 mode 的继续条件在结果产生前固定：II 后 A 宏 Dice 至少较 step0 增加0.05，U 匹配 mask Dice 至少增加0.10、匹配 presence 比率至少0.5、全网格异常并集 Dice 至少0.25；III 后仍满足这些条件，R 宏宿主准确率至少 max(0.5, 多数宿主基线的宏准确率+0.10)，且交叉熵相对 III step0 下降至少20%；冻结读出 Dice 对 II step0 的下降不超过0.05。多数宿主基线按同一固定面板的宿主分布求出，不能只靠恒定宿主输出或大面积异常掩膜过门。所有类支持与指标必须有限、全部预算完成；任何一项不满足记 FAIL/INSUFFICIENT，不发布正式监督就绪。即使通过也只交付正式 II/III 的下一阶段登记依据，不自动释放新完整训练。

## 交付与复核

实现两个小型脚本及针对角色、目标分母、严格继承、覆盖不足、冻结读出、恢复、隐私和超时的合成回归。执行真实 TRAIN 前，对准确源码、登记和父证据做一次独立审阅；所有实质问题解决后才释放有限 GPU 检查。完整 CPU 回归按项目命令执行。最终匿名报告区分代码检查、实际训练更新和医学质量；更新 main 的进展入口，保留未通过与未验证项目。
