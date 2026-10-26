# LR5正式监督接续：run_v3正在执行Stage II

2026-10-26用户要求继续后，从`work/aur-formal-supervised`接续正式计划。当前run_v3于09:39:55 UTC（北京时间17:39:55）启动，宿主机控制器PID538100，独立session后台运行，使用空闲A100物理3。09:50:40 UTC宿主机快照：Stage II已运行108/4096步，2D/3D各54次曝光，108条已记录loss全部有限，PID身份核对通过；III尚未开始。这证明正式监督已开始，不代表阶段完成或最终A/U/R验收。

## 原run_v2：初始化前拒绝，失败保留

v2登记独立审阅通过，完整CPU2071 passed/5 skipped（356.08秒）。08:59:00 UTC首次启动PID447755；Stage II初始化前的GPU空闲检查在09:05:55拒绝，step0，零优化步骤、零checkpoint，Stage III未启动。09:09:18写入FAILED终态，宿主机确认控制器及worker退出。closing完整输入验证通过、父身份不变。原v1/v2登记和run_v2所有产物均保留。

先前的空卡检查通过，失败后的空卡快照与controller→torch→launcher仅导入探针也通过，未发现自身CUDA上下文；这些都不能还原拒绝瞬间。通用错误吞掉了实际测量，因此原因为未知，不能声称是抢占、自身CUDA或超时，也不能宣称诊断改动已修复历史原因。

## 有限诊断修复和审阅

代码`8868a8d`仅修改外部控制器及其helper：GPU拒绝保留实际查询的匿名快照、UTC时间、固定原因/阶段码、可用利用率/显存/工作负载计数；查询异常只保留允许的类别和返回码，不保存原始命令/输出/异常文本或工作负载PID。阶段前及模型初始化前都用独占0600文件记录。利用率必须为0、无活跃计算任务、空闲显存至少8192MiB的要求不变，冻结runtime、模型、配方和阶段衔接不变。

六条新回归先观察到失败（无diagnostics或拒绝文件），随后受影响模块79项通过（30.81秒）；最终增加缺设备/查询超时/显存边界覆盖。独立增量审阅0项，审阅者单独执行9项新检查通过（6.03秒）。修复后完整CPU **2080 passed、5 skipped，348.95秒，退出0**。跳过项为nndet专用模块1、CUDA推理1、CUDA恢复3，未计通过。CPU命令保持CUDA隐藏、nvgen Python、PYTHONNOUSERSITE=1、PYTHONPATH=.、nice19、数值线程2和`pytest tests/ -q -rs -p no:cacheprovider`。各次测试为独立记录，不相加。

## v3登记与实际启动

新v3 SHA256为`8c1c359ea2319403652b60927c711c12b4614c9df18bc97997fea702f6346bc2`。独立精确登记复核确认：与v2只差两个控制器源码承诺，五项父输入、145份冻结源码、学习率和全部预算保持。正式controller SHA256为`30efe6dfc74e6ae5dfd9d4bc7ccf980757b5f9ce41882857cd6d4797c7e4da7a`，helper为`12f53ce43f8cb84503ee9b74a7d98bdaf9d0b684b04c90e430934604ea92df54`。

cwd和PYTHONPATH均指向原`execution_v5/runtime`，新进程导入来源和单张CUDA设备兼容性已核实。宿主机无继承CUDA/Slurm设备约束；最新检查卡3利用率0、无活跃计算任务、显存2MiB，完整UUID见RUN。其余任务不受干扰。本次Stage II模型初始化前已再次通过空卡检查；III启动前仍须独立复查。

预算为Stage II4096、III2048，交替2D/3D、batch1/accumulation1、16384点。II学习率backbone5e-5/heads5e-4、warmup128；III为backbone1e-5/heads2.5e-4/relation5e-4、warmup64。每256步保存、每512步按每mode8批验证；先完成并核验II正式证书，再以II正式selected export初始化III。外部总上限28800秒。run_v3是从原正式Stage I开始的新有界启动，未使用小集或失败运行权重，不增加优化步数、不无限恢复。

## 接手与剩余工作

当前唯一运行目录为`~/outputs/aur_lr5_formal_supervised_20261026/run_v3/`。先从宿主机边界核对PID及start ticks，再读terminal、阶段status和steps的匿名字段；不要再次启动。运行中的status.json步骤不会逐步更新（当前仍可为0），实时进度来自steps.jsonl完整记录，完成结论仍需证书。失败/超时须保留终态与已保存checkpoint，不能自动改变预算。只有完整证书、closing commitments与实际曝光计数核验通过后才能报告训练完成。

正式III完成后，按原计划在development validation选择并冻结工作点与关系head/lookup，再创建合格historical_test后继角色并最终判门，全量4469正式推理尚未执行。2D周期验证弱R关闭，等权loss不是最终二维R成绩。历史test已暴露，不称新独立队列；所有伪参照数字NOT_EVIDENCE。

[RUN.json](RUN.json)保存原失败、诊断、登记、审阅、实际启动和CPU日志的哈希。医学清单、数组与逐患者/观察记录均在Git外私有目录。
