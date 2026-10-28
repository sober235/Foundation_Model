# Foundation_Model：项目进展与接续入口

更新时间：2026-10-28。用户要求：以后回顾项目只查看`main`和这里登记的一个工作分支，不完整读取所有分支。

## 只读这两个入口

|入口|用途|
|---|---|
|`main`|已完成实现、验证证据、范围决定及当前进展的权威入口|
|`work/aur-formal-supervised`|唯一登记的后续工作分支：LR5正式II/III、最终评价及全量fastMRI执行|

工作分支最初从归档后的main建立；2026-10-26用户要求继续后，已从该分支接续正式监督启动。完成的里程碑合回main，并更新本文件和STATUS。旧分支保留作历史档案，不删除、不扫描；仅当本入口明确引用某个旧提交、证据缺失或用户明确要求时，定点读取所需内容。

回顾顺序：本文件→[STATUS顶部](STATUS.md)→[最新容量审计](docs/verification/2026-10-26/aur_query_capacity/RESULTS.md)及[已批准128-query设计](docs/superpowers/specs/2026-10-26-aur-query-capacity-128-design.md)→[正式监督接续](docs/verification/2026-10-26/aur_formal_supervised/RESULTS.md)→上述工作分支相对main的增量。不要从历史分支重新推断当前状态，不重跑已完成训练、G1、诊断或模型推理来恢复上下文。

## 当前目标和已确认决定

同一个模型、一个checkpoint处理2D/3D MRI，分别完成解剖A32、异常U和异常所在结构R。按[Q31–Q34验收规格](docs/superpowers/specs/2026-10-22-aur-three-subgoal-acceptance-design.md)，最终做成要求两模式在已登记的伪参照门上通过，并在指定4469个fastMRI序列上端到端执行；所有数字NOT_EVIDENCE。医生LevelR属于论文证据线，不作为这项工程目标的先决条件。

缺IPP/IOP的fastMRI允许走独立像素网格入口，始终标记`geometry=assumed`、`patient_coordinates=not_produced`，只报告屏幕侧别。严格DICOM入口不改变，假定几何不能写成患者坐标恢复。MRI采集欠采样若后续涉及，服从用户BART Poisson-disc＋中央24×24 ACS约束；本轮没有生成或更改采集mask。

## 已完成，不要重复

1. 数据资格与标签修复：合格监督TRAIN3499行/894来源限定患者、开发val405行/101患者；原合同和旧失败不覆盖。
2. 原统一StageI的2D G1失败已诊断；LR5候选已通过2D/3D G1，正式StageI2048步，按登记规则选step1792。PLANAR候选2D未过，原结果保留。
3. LR5小集监督工程检查II/III各512步、完整CUDA恢复与收尾校验完成。原质量诊断FAIL保留；用户明确“能跑通”用于该小集工程验收，不能用它推断最终性能。
4. Q34第一块已实现：A/U/R判门、同网格S7参照、合格historical_test后继接口、真实像素网格fastMRI批处理和标注报告。真实test接口目前只做合成验收，尚未构建真实test角色或读取其数组。
5. 实际CPU验证：S7开发val参考重评分完成，BMSR/ISLES/PDGM分别41/20/41序列；3个fastMRI序列双mode全部跑通（46个2D观察＋3个3D观察），约42秒。仅tiny pilot接口证明，尚未执行4469序列全量正式模型。
6. 标注CSV两条零面积框已修为明确排除/计数，新报告完成；不把退化框或未标注层当健康阴性。没有重跑3序列模型。
7. 全套CPU2069 passed/5 skipped（346.23秒）；随后正式控制器两处JSON单buffer校验修复，受影响模块73 passed（30.49秒），独立增量审阅0剩余。两次验证分别记录，不相加冒充一个全套计数。

## 当前执行点与下一步

**128-query实现与真实资源probe已完成，正式运行待启动。** 用户已确认从原LR5 Stage I骨干重新开始；实现506dbdc已完成，probe输出目录修复及回归待本次提交。probe_v5的II/III各8步、每mode各4次曝光、loss全部有限，完整U匹配和双mode R资格通过；2026-10-28只读回放核验checkpoint/export/完成记录和首尾承诺通过，没有重跑训练。最新[接续报告](docs/verification/2026-10-28/aur_query128_release/RESULTS.md)与[RUN](docs/verification/2026-10-28/aur_query128_release/RUN.json)优先于旧状态。

完整容量审计保持：TRAIN源max102，登记II的2D/3D max14/85。128覆盖当前三维源上界及登记观察，不宣称覆盖任意未来二维分裂。正式预算II4096→III2048、时间上限4h/2h和控制器8h不变。8个特选probe观察不能可靠外推全预算耗时；独立验证/保存时间和当前128-query CUDA恢复尚无单独证明，历史证据不冒充当前验证。运行达到上限必须停止，不自动加预算。

本轮继续代码和训练，不推进论文；扫描protocol暂不作为显式模型输入，保留DICOM几何处理。最终仍须2D/3D分别验收A/U/R及全量fastMRI，probe成功不是质量达标。

**当前run_v3已停止，Stage II终态FAILED。** 完成1725步（2D863、3D862次曝光），第1726步前的3D裁块有65个实例，超过登记的64query；同一冻结源码/seed/draw862已在CPU复现。宿主机控制器退出，III未启动，closing完整校验通过。最近持久化checkpoint是1536，文件哈希核对通过；不能从1725精确恢复。详见[停止诊断](docs/verification/2026-10-26/aur_formal_supervised/STOPPED.md)。原样恢复会重复该观察；须先制定独立容量处理方案，不直接重启，不修改旧冻结runtime或截断病灶。

原训练执行代码仍为`8868a8d`，v3登记SHA256为`8c1c359ea2319403652b60927c711c12b4614c9df18bc97997fea702f6346bc2`；完整CPU2080 passed/5 skipped、独立代码/登记审阅0项均为此前证据。首次停止诊断只复现一个观察；随后上述完整登记容量审计已完成，不重跑。

**此前run_v2已失败并保留：** Stage II初始化前GPU准入拒绝，step0、零checkpoint，III未启动，宿主机进程已退出；收尾完整输入校验通过。原错误没有保存拒绝时测量，具体原因仍未知；后来的空卡快照和仅导入探针不能证明原因为抢占或自身CUDA。新的外部控制器记录实际判定时的匿名快照及固定原因码，未放宽准入；这是新的有界启动，不是追加训练预算或恢复已训练权重。

在`work/aur-formal-supervised`按[正式监督计划](docs/superpowers/plans/2026-10-26-lr5-formal-supervised.md)接续：

1. 容量审计、128-query实现和有限probe已完成；按registration_v5核验后启动新正式运行。旧run_v3不原样重启，所有旧登记和失败记录保留。
2. 宿主机约束、兼容性、空卡和导入位置已检查，后续新运行仍须在每阶段模型初始化前复查。**启动cwd和PYTHONPATH均为registration_v5指定的新128-query冻结runtime**；初始化来自通过G1的正式StageI，不使用小集resource-probe权重。
3. 首次有界正式预算II4096、III2048步，完整TRAIN池抽样；先II完整证书，再III。详细LR、warmup、验证/保存周期与8小时总上限在计划中。全池抽样不保证所有行都访问，也不代表充分收敛。
4. 周期best仍按两mode等权loss，二维validation的弱R保持关闭；它不是最终R验收。正式III完成后在development validation选定工作点及head/lookup，冻结登记，再构建真实历史test角色并只做一次最终判门。历史test已暴露，不能称新独立队列。
5. 按已登记实现和阈值执行全量fastMRI。G1通过、小集跑通、CPU测试通过都不能替代最终六项A/U/R门或全量执行。

## 私有恢复定位

只读取完成标记、登记和允许公开的汇总；医学清单键/值、患者文件名、数组与逐观察记录不能进入Git或工具输出。

|证据|私有位置|
|---|---|
|原数据合同|`~/outputs/aur_qualified_training_20261019/data_contract/final_v3/contract.json`|
|精确资格源码归档|`~/outputs/aur_qualified_sources_20261021/65eb7df/`|
|LR5冻结运行树|`~/outputs/aur_2d_candidates_20261021/execution_v5/runtime/`|
|LR5正式父权重/G1|`~/outputs/aur_2d_candidates_20261021/execution_v5/A_LR5/`|
|小集工程检查|`~/outputs/aur_lr5_supervised_capability_20261022/run_v2/terminal.json`|
|本轮已完成真实检查|`~/outputs/aur_gate_layer_20261026/`：`reference_val_v1`、`fastmri_smoke_v1`、`fastmri_plus_audit_v2`|
|正式控制器/v3登记/已停止运行|`~/outputs/aur_lr5_formal_supervised_20261026/`：`launch_record_v3.json`、`registration_v3.json`、`run_v3/`；v1未运行、失败run_v2保留|
|已完成容量审计|`~/outputs/aur_capacity_audit_20261026/full_v1/completion_anonymous.json`；脚本/合成检查/独立复核及完整私有记录在同根，勿重跑|

父合同SHA256：`a839cfc9c9213b5b376eac4bc4a8219506175fdbe3620bc36cb2056f602b6060`；LR5冻结runtime摘要：`5120f2040421f0a8f4c6b198cc20281adfdf1aa4b8204848c220d5dffe3f423a`。此前判门层有限真实检查已完成；当前正式监督状态以本页顶部和新RUN为准。不要凭沙箱内缺少PID判断其他任务退出，须按任务执行边界核实。
