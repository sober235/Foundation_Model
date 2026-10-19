# TRAIN U 背景采点配额对照（2026-10-19执行）

**裁决：本次结果不支持把20%邻域环采点改为全局均匀采点作为修复方向；保留原配置，不据此追加训练。** ISLES主面板的8个裁块×采样对照中，改动后的未采到误报背景平均logit方向均高于原配方，即局部抑制更弱。这个结论只针对本次固定TRAIN裁块、这一个配额改动和共享掩膜头切线；不是新增的模型性能裁决。

A32解剖身份掩膜、统一异常U实例掩膜、每个U的R13分侧主宿主+no-host仍是三个目标。S为内部序列上下文。没有参数更新、优化器、训练、校准拟合、阈值搜索或val/test影像前向。默认`legacy_max`、完整405val的`query_cc_v1` FAIL、A缺失类别/历史test缺口和controlled R FAIL均保留。全部结果 **NOT_EVIDENCE**。

[批准设计](../../../superpowers/specs/2026-10-18-aur-u-background-allocation-design.md)；[实施计划](../../../superpowers/plans/2026-10-18-aur-u-background-allocation.md)；[运行记录](RUN.json)；[完整匿名汇总](aggregate_anonymous.json)；[采点与控制补充](supplement_anonymous.json)；[独立数值检查](numeric_audit.json)。目录沿用2026-10-18冻结方案日期，实际运行日为2026-10-19。

## 固定比较及读数单位

执行代码`73aa2c730075894216570b63c833d500b90c744b`。复用上一轮六名TRAIN患者、十二个完全相同的增强裁块，BMSR/T1c、ISLES/DWI、PDGM/FLAIR各两人/四裁块。C2 StageIII step3000/72000crops权重不变，checkpoint SHA256 `38f6376333041a75be1cce83e927bc221468761f80ae952e2a3020805f7dcbde`，实际profile仍为`unmatched_empty_v1`。

每裁块一次冻结eval前向，全部64个U query和32个A query；原生CUDA bf16输出先转float32。每裁块两组固定CPU采样种子，两臂均16384点、focus_share0.5，仅boundary_share由0.2改为0。配对验证了相同正点配额槽、相同非环槽，以及环槽恢复为同一初始uniform抽样。**总正点数仍可改变**，uniform恢复点也可能落在GT上；没有补点或重新加权。

原采点、presence/mask logits、官方训练匹配及解码组件图/行均复现；logit最大绝对误差为0。原16TP/113FP是复用的身份控制，不是新预测改善。53是裁块内GT计数，可能跨裁块重复；两组采点、组件和query记录也不构成新增独立患者。

主分析固定各原draw的query/target匹配，保留完整U presence、matched BCE/Dice和unmatched-empty BCE0.1。A仍用原点。对共享`embed_full`的74016个参数建立**原生值锚定的float32冻结特征切线**，主方向为原权重A+U梯度，另报U-only及原始/单位L2方向。真实模型权重始终冻结。它不是完整网络、历史bf16训练或优化器梯度。

主面板只含原官方FP中、原训练匹配query对应的有效真背景体素，且排除两臂所有采样点的并集。先在同query内合并组件，再按query→crop/draw求均值；组件加权另报。这样检查的是未直接采到位置的局部传递。`D<0`表示沿该负梯度方向降低平均logit；表中差值为`global−control`，没有实际步长或更新后的解码。

## 主结果与A/U/R控制

| 来源 | 有主面板的crop×draw / 全部 | FP query×draw支持 | Δ未采到FP背景logit方向 | Δ全部分配目标logit方向 | Δ A损失方向 |
|---|---:|---:|---:|---:|---:|
| BMSR/T1c | 0/8 | 0 | 未定义 | +2.223243（4/8有目标） | +0.006022 |
| ISLES/DWI | 8/8 | 56 | **+1.438208** | +1.224134 | −0.004193 |
| PDGM/FLAIR | 4/8 | 4 | +4.029640 | +6.666660 | +0.056155 |

表为**combined单位方向**的source内crop/draw均值；不同面板的支持分母分别列出，不能当作同一批独立样本。A在12裁块均有监督。GT目标方向正值差意味着局部目标logit相对更高；A损失差的正值意味着相对较少的损失下降，不等于实测A Dice下降。两臂在combined单位方向下的全部24个A损失方向本身仍为负值。

- ISLES主面板control为−0.859799，global为+0.578410；配对差8/8为正，四裁块对两次draw求均值后也4/4为正。不是所有单裁块方向都由负变正。未归一化combined差均值+4.198242，U-only单位方向差+1.533125，也均8/8为正；不利方向并非只出现在一种归一化中。
- ISLES全部分配目标差8/8为正；A差5/8为负、3/8为正。观察到的目标增强/A平均方向没有弥补主面板的抑制不足；不能写成无条件A改善。
- PDGM只有四条受支持query×draw、四个crop×draw（两裁块），主面板配对差均为正；A平均方向也相对较少下降。该小支持仅作来源对照。
- BMSR原6个FP均来自训练未匹配query，故本次已匹配FP主面板未定义。不能把无支持当作零效果或成功；未匹配query路径仍单独保留。
- 固定query的R logits及真实模型完整状态在十二裁块前后完全相同。R使用独立coarse投影，这只是本次受限路径控制，不证明完整训练干预对R安全，更不是R质量提升。

## 采到更多远处背景不等于覆盖更多误报

| 来源 | 远处背景采点次数 control→global | 已匹配FP真背景唯一覆盖体素×draw control→global | 对应FP query×draw零覆盖数 |
|---|---:|---:|---:|
| BMSR/T1c | 85190→98289 | 无已匹配FP支持 | 无支持 |
| ISLES/DWI | 38536→64225 | **3085→1729** | 4/56→5/56 |
| PDGM/FLAIR | 38324→63892 | 33→52 | 0/4→0/4 |

这些是controller从固定numeric缓存复算的描述性计数，脚本/哈希见RUN。ISLES远处背景点增加，但这批已匹配FP背景的直接覆盖反而减少；PDGM覆盖增加也没有带来更有利的受限参数方向。两者均不支持“增加uniform即可解决”的推断，也不排除保留环配额的其他采点、空间权重、表示或优化机制。

两臂106条target×draw均有有效正点，无query容量溢出。重新运行官方匹配后有5/106条target×draw换query，全部在ISLES（5/86）；这是两种配额之间的变化，不是上一轮两组原采点间的5/53结果。主分析没有用新匹配替换原匹配。BMSR四个无环draw两臂点集相同，作为负控制；全部配对槽检查通过。

## 核验、资源及证据边界

- 完整CPU回归：**1345 passed、2 skipped，180.61秒，退出码0**。跳过专用nndet环境模块和原CUDA-only bf16单测；没有将旧补测改写成本轮零跳过。新101条测试覆盖数学、测量、存档及真实tiny-model集成。
- 三项独立任务审阅和整分支审阅均为0项可操作问题。额外独立检查包括74016参数/16384点数学fixture、真实tiny-model两裁块完整存档回放和16例ROI/来源汇总oracle。
- 实验退出码0，墙钟161.126秒；十二裁块内部计时合计42.565秒。最大CUDA allocated2.510GiB/reserved3.977GiB，CPU RSS2.860GiB。开销包含取特征、控制、切线和归档，不是推理速度比较。
- 原选空闲A100物理5在启动时被其他任务占用，保护检查在模型/影像加载前停止；未产生实验存档。随后兼容性检查和两次负载/进程快照确认A800物理7空闲，使用其logical0运行，保留其他任务。沙箱内PyTorch不能访问CUDA，实际GPU运行及DataLoader完整CPU检查采用获准的沙箱外执行。
- 隐藏CUDA的CPU完整回放15.095秒、退出码0，27个存档成员字节不变；显式阻止了模型、影像和checkpoint加载。旧参考存档29成员也保持原hash。
- 独立脚本核对十二裁块/24draw、680条ROI，共69297项结构/数值检查通过：原损失/匹配、固定匹配的原始损失、独立FP32梯度、解析GELU JVP、ROI集合、A响应和主面板source汇总。预先固定梯度rtol2e-5/atol2e-7、响应rtol1e-4/atol2e-5；没有事后放宽。
- 独立核验**未重新生成密集像素特征均值**：均值来源由runner的独立密集投影流式约简及模型/输入控制核验；核验脚本接受旧缓存的官方组件标签，不重复密集解码。外部TRAIN资格、原始输入/代码/checkpoint承诺另由controller固定脚本验证，不混称为独立溯源审计。
- 同一checkpoint、128项原依赖承诺及原生产Python文件集不变；controller另在CPU核对了checkpoint完整状态锚点及十二裁块缓存的四个head张量。该checkpoint读取是单独溯源步骤，与不加载checkpoint的CPU回放分开。
- 新私有存档目录0700、文件0600；逐例/query/组件、采点/特征/梯度及选择清单不入Git。实际stdout精确13条固定进度/完成行；新增公开文件与stdout另作所选标识扫描。这不扩展为历史会话或未改历史文件的匿名保证。

本次完成的是一个不获支持的局部候选实验，模型缺口没有修复。保留原采样配方；不在这批结果上挑选中间boundary_share、增加患者、追加训练或选新阈值。下一项干预须另行冻结问题和A/U/R回退检查，A32全query审计与R的dense/coarse几何来源对照仍独立待做。

已有完整缓存仅需只读回放，不重复GPU测量：

```sh
CUDA_VISIBLE_DEVICES='' PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=. OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 nice -n19 /home/congcongliu/anaconda3/envs/nvgen/bin/python scripts/aur_diagnose_u_background.py --out /home/congcongliu/outputs/aur_u_background_allocation_20261018/run_73aa2c7 --replay-only
```
