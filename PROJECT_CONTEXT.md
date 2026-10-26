# Foundation_Model：项目进展与接续入口

更新时间：2026-10-26。用户要求：以后回顾项目只查看`main`和这里登记的一个工作分支，不完整读取所有分支。

## 只读这两个入口

|入口|用途|
|---|---|
|`main`|已完成实现、验证证据、范围决定及当前进展的权威入口|
|`work/aur-formal-supervised`|唯一登记的后续工作分支：LR5正式II/III、最终评价及全量fastMRI执行|

本次交接时，工作分支从归档后的main建立，起点相同；不能把相同起点写成已有未合入实现。以后在该工作分支继续，完成的里程碑再合回main，并更新本文件和STATUS。旧分支保留作历史档案，不删除、不扫描；仅当本入口明确引用某个旧提交、证据缺失或用户明确要求时，定点读取所需内容。

回顾顺序：本文件→[STATUS顶部](STATUS.md)→[本轮结果](docs/verification/2026-10-26/aur_gate_layer/RESULTS.md)及[RUN](docs/verification/2026-10-26/aur_gate_layer/RUN.json)→上述工作分支相对main的增量。不要从历史分支重新推断当前状态，不重跑已完成训练、G1、诊断或模型推理来恢复上下文。

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

## 当前停止点与下一步

**新正式II/III尚未启动。** 外部控制器为[scripts/aur_formal_supervised.py](scripts/aur_formal_supervised.py)，实现与测试已完成。其最终源码SHA256为`cbc8d40b3a7e0e4eb0801b50d2493f7c57892df52b4be5865d4b5f3a211731fc`；须在新登记版本绑定该源码，不直接运行旧v1登记。

下一次在`work/aur-formal-supervised`按[正式监督计划](docs/superpowers/plans/2026-10-26-lr5-formal-supervised.md)接续：

1. 从最终源码生成私有`registration_v2.json`并完成精确登记复核。v1已保留但未运行，它绑定较早控制器版本；不可改写v1或把它当作已释放。
2. 重新检查宿主机GPU可见性、调度约束、负载和活跃进程，选择兼容空闲卡。使用原LR5冻结运行树；**启动cwd设为该冻结runtime目录，同时PYTHONPATH指向它**，先验证子进程`python -c`的anatobind导入来源，避免checkout工作目录覆盖冻结源码。直接从通过G1的正式StageI初始化，不能使用小集resource-probe权重。
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
|正式控制器准备与旧未执行登记|`~/outputs/aur_lr5_formal_supervised_20261026/`|

父合同SHA256：`a839cfc9c9213b5b376eac4bc4a8219506175fdbe3620bc36cb2056f602b6060`；LR5冻结runtime摘要：`5120f2040421f0a8f4c6b198cc20281adfdf1aa4b8204848c220d5dffe3f423a`。本轮真实检查进程均已完成；不要凭沙箱内缺少PID判断其他任务退出，须按任务执行边界核实。
