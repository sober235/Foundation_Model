# STATUS：2026-09-29 18:00（S7 脑部多病种：代码与全分支终审完成，第一份读数已出——梗死 fold 0 早读 0.568 @ 1.66 误报/例，不是达标结论；15 个训练里 2 个已完成、6 个在跑、7 个排队；句子有两处写法等用户确认；nnDetection 第二臂 fold 0 仍在训练；S4 脑部解剖层的设计概览已给用户、等点头）

每次交接前整体重写本文件。五段固定：已验证、待拍板、下一步、坑与别重做、为什么。

## 1. 已完成且已验证

**本轮：S7 脑部多病种检测，阶段 A（每个病种一个 nnU-Net，各用原生序列）。** 规格 `docs/superpowers/specs/2026-09-29-brain-multidisease-design.md`（决定 M1–M14），计划 `docs/superpowers/plans/2026-09-29-brain-multidisease.md`（13 个任务）。分支 `build/brain-multidisease`（工作树 `../foundation_model-multidisease`）。台账 `../foundation_model-multidisease/.superpowers/sdd/2026-09-29-brain-multidisease/progress.md`（gitignore，不入库；终审原文 `final-review-1-report.md` 也在那里）。

- **测试**（2026-09-29 17:49，分支头 c1b26e3 的代码；nndet 环境那一条是 15:51 跑的，之后没有改动它涉及的文件）：
  ```
  PYTHONNOUSERSITE=1 PYTHONPATH=. nice -n 19 ~/anaconda3/envs/nvgen/bin/python -m pytest tests/ -q -p no:cacheprovider
  831 passed, 1 skipped in 109.86s (0:01:49)
  bash -c 'source scripts/nndet_env.sh && nice -n 19 python -m pytest tests/test_nndet_runner.py -q -p no:cacheprovider --noconftest'
  2 passed, 5 warnings in 2.80s
  ```
- **第一份读数：梗死 fold 0 的早读**（不是达标结论；记录 `docs/verification/2026-09-29/brain_multidisease/infarct_fold0/`，提交 5ff45ef，用 001cfd1 的代码出的）：
  ```
  PYTHONNOUSERSITE=1 PYTHONPATH=. nice -n 19 ~/anaconda3/envs/nvgen/bin/python scripts/eval_brain_disease.py --disease infarct --folds 0 --workers 2 --out docs/verification/2026-09-29/brain_multidisease/infarct_fold0
  Disease: infarct; folds [0]; kind early_reading
  Scans: 50; lesions counted: 428; ignored: 33
  Sensitivity 0.5678 at threshold 0.6 with 1.66 FP per scan
  Pass: None; stop remaining folds: False; early stop undecided: False
  Dice mean 0.753277521477103 over 49 cases
  ```
  - 误报预算没有用满（不设阈值时每例 1.68），所以 0.568 就是这一折的上限；阈值 0.95 时 0.423 @ 0.74。
  - 按等效直径分层：< 5 mm 60/193 = 0.311；5–10 mm 107/152 = 0.704；≥ 10 mm 76/83 = 0.916。漏的主要是小病灶。
  - 每例误报：中位数 1，最大 12，14 例超过 2 个。
  - 绑定一致率（NOT_EVIDENCE）：主结构 0.947，侧别 0.988。
  - 控制方另写了一段只用 scipy 的代码独立计数：计入 428、忽略 33、保留的预测块 327（= 243 命中 + 84 误报），与报告一致。终审者在这一折的五例真实输出上逐例、逐阈值重算，也一致。
- **训练进度**（17:55；队列进程 pid 1091434，2026-09-29 13:46:13 启动，记录 `docs/verification/2026-09-29/brain_multidisease/launch.md`）：

  | 训练 | 状态 |
  |---|---|
  | 梗死 fold 0、fold 1 | 已完成（17:04、17:02，退出码 0；nnU-Net 自报 Dice 0.753、0.785） |
  | 转移瘤 fold 1 | 第 213 轮，约 35 分钟 |
  | 转移瘤 fold 0 | 第 134 轮，约 3 小时 |
  | 胶质瘤 fold 0、fold 1 | 第 89–90 轮，约 5–6 小时 |
  | 转移瘤 fold 2（17:04 启动） | 第 31 轮，约 7 小时 |
  | 胶质瘤 fold 2（17:02 启动） | 第 10 轮，每轮 233–346 秒，约 15–23 小时（见 §2 第 1 条） |
  | 其余 7 个 | 排队 |
- **三个数据集**（Task 2，记录 `docs/verification/2026-09-29/brain_multidisease/build/`）：

  | 数据集 | 通道 | 扫描 / 病人 | 无标签体素的例 | 规划（间距 mm，块，batch） | 每折例数 |
  |---|---|---|---|---|---|
  | Dataset904_PDGMGlioma | T1、T1c、T2、FLAIR | 501 / 495 | 0 | 1.0 各向同性，128×160×112，2 | 100 / 101 / 100 / 99 / 101 |
  | Dataset905_BMSRMetastasis | T1pre、T1post、FLAIR | 461 / 314 | 0 | 1.5 × 0.859 × 0.859，80×192×160，2 | 105 / 86 / 87 / 106 / 77 |
  | Dataset906_ISLESInfarct | DWI、ADC | 250 / 250 | 3 | 2.0 各向同性，80×96×80，8 | 50 × 5 |
- **代码**：`anatobind/nnunet/brain_disease.py`、`anatobind/eval/lesion_components.py`、`anatobind/eval/detection_metrics.py`（`scan_matches` 与"忽略"标记）、`anatobind/bind/brain_lookup.py`、`anatobind/infer/brain_disease.py`、`anatobind/eval/brain_disease.py`；脚本 `scripts/brain_disease_{prepare,crossrun}.py`、`gpu_queue.py`、`eval_brain_disease.py`、`infer_brain_disease.py`。十个代码任务由子代理实现并逐个评审（sonnet）；终审后的五轮修补由控制方直接落地（代码先在副本里验证，每个提交先看新测试在旧代码上失败）。分支上 22 个代码与测试文件与计划里的代码逐字节一致。
- **全分支终审（最强模型）共六次往返，最后结论："可以写逐例记录"，前提是用户确认句子的两处写法（§2）。** 要点：
  - 终审者用另写的代码独立重算（分母、命中、误报），与分支一致；确认读的是折外预测、15 个折里没有病人跨训练与验证。
  - 它读了本机 nnU-Net 2.8.0 的导出代码：标签图就是所存 float32 概率图的 argmax，所以"预测块里出现前景概率 < 0.5 的体素就报错"这项检查不会在正确输出上误报（防的是轴序错位）。
  - 它把三十多种"改坏代码"的方式逐一施加到副本上，除一种（git 调用是否加锁）外都有测试拦住。
  - 终审引出的改动：早停规则（见 §5）；队列对失败训练的处理（供下次启动）；句子的多处写法（见下一条）；体积下限对 float32 文件头的容差；记录加 `model_folds`、`anatomy_source`、`host_side`、`host_sides`、`host_distance_mm`；报告加每例误报分布、按最近距离定位的比例、"未能定位"的比例、主结构侧别一致率、代码提交号。
- **句子现在的写法**（`anatobind/infer/brain_disease.py`；三例真实梗死预测的句子见台账）：
  - 写体积最大的 5 处，从大到小；其余计数，并补上"前 5 处都没提到的位置"，如"另有 21 处同类异常（还见于右侧大脑皮层、左侧小脑、左侧丘脑）"。**这两点是建议，待用户确认。**
  - 主结构前的侧别按"病灶落在主结构里的那部分体素"算；脑干不写侧别。
  - "累及"后面的结构都与主结构同侧时不写侧别；有一个不同侧就全部写出，脑干排最前。
  - 不与任何结构重叠的病灶：离最近结构 ≤ 10 mm 写"邻近<侧><结构>（未与任何结构重叠）存在…"，超过 10 mm 写"未能定位的区域存在…"（10 mm 是用户定的）。
  - 没检出写"本模型未检出<类型>（阈值 0.xx）。"，印象写"未检出相关异常"。
- **真实数据上的五项只读核查**（脚本、命令与原始输出在 `docs/verification/2026-09-29/brain_multidisease/checks/`）：体积下限不会被文件头改动一格；1212 例的 SynthSeg 图都在且与标签同网格；3887 个通道文件与解剖图仿射完全相同；三组交叉运行的输入都在且同网格；没有一张解剖图缺宿主体素。
- **用户本轮拍板**：S7 走 C（先 A 后 B）；达标线与 S2 的 D1 同口径、三病种分别判；产出 = 检测器 + 查表绑定 + 结构化记录，脑叶留给 S4；五折全训、fold 0 先报（"B，不用留卡"）；方法 nnU-Net 3d_fullres；采纳两条建议（交叉误报只报告；S4 设计与训练并行）。算力原话："后续整个服务器的算力都要优先该任务使用，但是前提是只能是考虑或者占用空的 GPU"。S4 的老师用方案 B。2026-09-29 16:52 用户回复"按照你的建议"：距离上限 10 mm；控制方同时把它理解为此前列出的几项也按建议办（句子三处措辞、"未定"的早停不自动停、GPU 冲突先等、S4 输出粒度 A），并已在聊天里说明这一理解、请用户有出入就指出，用户未提出异议。

**nnDetection 第二臂（S2）**：fold 0 自 04:24 在 GPU 3 训练，17:55 在第 52 轮（共 60 轮），3.0 步/秒，预计 20:00 前后训练结束，之后是 sweep。计划 Task 1–8 已在 main。

**S4 脑部解剖层**：设计概览已在聊天里给用户（约 17:25），等点头。只读探查的新事实：`/data2/congcong/data/FM_data/SibBMS_ms/sibbms.zip`（11 GB，未解压）里有健康人 100 人（T1、T2、FLAIR）和多发性硬化 93 人 / 272 次检查（另有增强 T1），全部 197×233×189、1 mm、已去颅骨并在标准空间；本机所有带 FLAIR 的数据集都去过颅骨。

**之前（保留）**：S2 nnU-Net 检测器 D1 门不过（2d 五折 0.3662 @ 1.636 FP/卷，主因漏检），记录索引 `docs/verification/2026-09-28/brain_detector/README.md`。S1 关系基线（NOT_EVIDENCE）、Level R 读片工具（冒烟服务 8791，浏览器验收仍 USER_REPORTED）同前。

## 2. 待用户拍板

1. **其他会话持续在我们占着的卡上启动任务，胶质瘤 fold 2 按现在的速度会超过 24 小时。** 我们的训练每个只占约 8 GB，别的会话按剩余显存挑卡就会落到这些卡上。经过：14:31 GS 会话（`mcgs_ce_mask_wt`，`CE_retro_cssense_af16_vdpois`）在 GPU 0、3、4 起 MC-GS、GPU 1 起重建链（约 17:05 前结束）；17:02 膝关节筛选会话（`mcgs_nb_knee_wt`）在 GPU 6、7 起任务，GPU 6 与我们的胶质瘤 fold 2 在同两秒内抢到同一张卡；17:33 GS 会话又在 GPU 0、1、2、4 各起一个 25 GB 的任务。本会话没有动任何不是自己启动的进程。已给用户三个选项：A 维持现状；B 用户去告诉那两个会话只用没有任何进程的空卡；C 授权本会话给它们发消息。用户未答复，按 A 进行。
2. **句子按体积从大到小写前 5 处**（原规格是按分数）。理由：分数是块内概率均值，小块最高；真实一例 26 处病灶里，按分数只写出 16–64 mm³ 的五处，最大的 744 mm³ 被并进"另有 21 处"。代码已按建议改，规格里标为待确认。
3. **"另有 N 处"后面补位置**（"还见于…"）。同上，已按建议改，待确认。
4. **S4 设计概览**：方向对的话继续分段细化。
5. **push**：main 自 afe641e 起所有提交与 tag 都只在本地。
6. **nnDetection fold 0 出结果后**：规则 A（≥ 106/280）通过就要补五折，开跑前问；不通过就停。
7. **读片人**、Level R 读片说明里 "other" 的定义、伦理备案：同前。
8. **可删清单（只列，不删；删除由用户执行）**：S2 工作树 `../foundation_model-detector` 与分支 `build/brain-detector`（已合并）；`/data2/congcong/data/FM_data/derived/nndet_smoke/`；`~/logs/nndet_install/wheels/` 里 1.6 GB 的 torch 安装包；会话 scratch（`/tmp/claude-1002/-home-congcongliu--claude/614ccd9c-f8e8-435b-afc8-47ae37fe3721/scratchpad/` 下的 `plan_dryrun`、`s7_dryrun`、`s7_nnunet`、`s7_e2e*`、`s7_readings`、`s7probe`、`s4probe`、`s7_checks`、`review_*` 等）。
9. 旧遗留：Q9 删除授权、其余 4850 卷 SynthSeg、Redivis token、两条远端评审分支去留、RSS 左右手性换算规则。

**终审给用户的提醒**（已在聊天里转达）：达标数字取决于分数的分布，nnU-Net 的分数集中在 1 附近；"无工作点"与找到多少病灶无关；"疑似 X"只说明跑的是 X 模型；数据里几乎没有正常脑（1212 例里 3 例无病灶），"未检出"不能当阴性；解剖词来自伪标签上的体素计票；左右取决于文件头（未独立核实）；侧别按 40% 规则判，一个 65/35 的跨中线肿瘤会写成单侧；10 mm 在层厚约 5 mm 的扫描里只相当于两层。

## 3. 下一步

1. **每个 `finished` 都核对退出码**（正在跑的队列是旧代码，失败只记一行）。监视已挂，队列出事件会通知。
2. **计划 Task 11，转移瘤与胶质瘤 fold 0 的早读**（梗死已做）。前提：该折 `validation/summary.json` 存在，且 `logs/brain_disease/queue.log` 有 `finished ('<病种>', 0) … exit code 0`。命令（训练还在跑时只开 2 个进程）：
   ```
   PYTHONNOUSERSITE=1 PYTHONPATH=. nice -n 19 ~/anaconda3/envs/nvgen/bin/python scripts/eval_brain_disease.py --disease <病种> --folds 0 --workers 2 --out docs/verification/2026-09-29/brain_multidisease/<病种>_fold0
   ```
   读 `verdict.json`：`early_stop_undecided` 为 true 就不写 skip 文件，把 FROC 表和"刚超预算的那一行"报给用户；`stop_remaining_folds` 为 true 才写 `logs/brain_disease/skip_<数据集号>`（写了就删不掉，只有用户能撤）。提交读数（`git commit -m "Brain disease <disease>: fold 0 early reading (not the gate)"`），报告用户并写明"这是早读，不是达标结论"。
3. **写逐例记录之前**：用户确认 §2 第 2、3 条；另按终审建议，先用转移瘤和胶质瘤的真实输出在内存里各生成三句读一遍（重点看带卫星灶的胶质瘤和病灶很多的转移瘤，"还见于"后面的位置可能有十来个，要不要设上限到时再定）。
4. **计划 Task 12**：队列日志出现 `queue empty, nothing running: done`，且每个要评估的折都有 `finished … with exit code 0` 后：三个病种五折评估（逐例记录写到 `/data2/congcong/data/FM_data/derived/brain_disease/<病种>/records`）、三组交叉误报、每病种一例推理冒烟（每次用新的输出目录）、记录索引 README（写明匹配规则与 S2 相同、阈值是单折模型上测的）。
5. **计划 Task 13**：更新本文件与 CLAUDE.md、记录复核、合回 main、tag `handoff/<日期>-brain-multidisease`，不 push。
6. **nnDetection**：训练与 sweep 结束后做其计划 Task 9（提取、fold 0 报告、规则 A、推理冒烟）与 Task 10。进程 3334388 结束时可能卡在退出，输出齐了再 `kill -TERM`（那是本会话自己的进程）。
7. **S4 设计**：用户点头后分段细化（数据、仿真、验证与达标线、推理），再写规格与计划。SibBMS 要先解压、再用 CPU 跑 SynthSeg（372 次，约 6 小时），CPU 现在贴着上限，得等 S7 的训练跑完一批。
8. 阶段 B（只吃 FLAIR 的统一模型）等阶段 A 出数后再设计。

## 4. 坑与别重做

- **训练日志滞后**：训练的标准输出重定向到文件是块缓冲，`logs/brain_disease/Dataset90*_fold*.log` 会落后十几分钟。每轮耗时读 nnU-Net 结果目录里的 `training_log_*.txt`。
- **CPU 已贴着上限**：六个训练加 nnDetection 实测 47.2 核（上限 48）。训练期间任何评估都加 `--workers 2`，全量测试一次只跑一个。
- **别的会话会把任务放到我们占着的卡上**：队列只保证自己不抢别人的卡，防不住别人。发现变慢先看 `nvidia-smi --query-compute-apps`，不是自己启动的进程一律不动，报用户。
- **这里不能删任何东西**，所以半成品目录会一直留着并挡住下一次同名运行：脚本都是"先检查、后建目录"；推理冒烟失败后换一个新的 `--out`。早读先输出到 scratch 看一眼、再写进仓库，是因为仓库里的目录写了就不能重来。
- **队列重启不要覆盖日志**：13:46 那次启动写的是 `logs/brain_disease/queue.log`，之后任何一次启动都用新名字（`queue_$(date +%Y%m%d_%H%M%S).log`）。出现非零退出码先读该任务的日志，若原因会影响每次启动就建 `logs/brain_disease/stop` 并报用户。
- **句子的每一处写法都要拿真实输出读一遍**：终审五轮里每一轮都是在真实或接近真实的病灶上才发现句子的问题（按分数排序、侧别词串读、"另有 N 处"藏住位置）。单元测试守得住规则，守不住"读起来对不对"。
- **匹配规则与 S2 相同**：取总 IoU 最大的一对一指派，再去掉 IoU < 0.1 的对；拥挤时偶尔比最优的一对一匹配少一个命中，命中数也因此不一定随阈值单调。为了与 S2 可比不改，记录索引 README 里要写明。
- **分数的性质**：检测分数是连通块内前景概率的均值，块来自 argmax 图，所以分数都大于 0.5；FROC 表里阈值 ≤ 0.50 的各行相同；小块的分数最高。
- **绑定一致率不是证据**：真值和预测都用同一张 SynthSeg 伪标签图查表。
- **计划里的代码事先跑过**：计划的每段代码都在分支的临时副本里执行过；临时副本在会话 scratch 里，是一次性的，以仓库里的计划为准。
- 台账里 17:28–17:42 的几个时刻是估的（已改成"约"）；要准的时间以提交时间和队列日志为准。
- nnDetection 的环境、约定、下载的坑同上一版（`docs/nndet_install.md`）；zsh 的 `pgrep -f` 会匹配到自己，等待循环用 `ps -p <pid>`。
- 别在 nnU-Net 上调参救线（M5、D9）；C1 上的数字都不是证据。

## 5. 关键决定的为什么

- **先 A 后 B（M1）**：病种与数据集完全混杂（每个病种只来自一个数据集），统一模型可以只靠图像风格分辨病种。
- **按 10 mm³ 而不是按体素数定下限（M7）**：三个数据集体素大小差很多（转移瘤一个数据集内下限就从 5 格到 55 格）。
- **五折全训、fold 0 先报（M4）**：用户说不用留卡；五折折外预测让每个病例都有一条记录。
- **早停只在读数明确时触发**：工作点灵敏度 < 0.3，且"刚超预算的那一行"加上两个病灶也到不了 0.3。阈值网格步长 0.05，最后一格可能误判；skip 文件写了就删不掉；误停会错杀一个可能过线的病种，多训几折只多花空闲 GPU 时间。没有工作点时没有测到灵敏度，一律不自动停。
- **推理前核对网格**：体积和最近结构都按解剖图的体素大小算，形状相同而网格不同会悄悄算错。
- **句子按体积排序**：读的人先找最大的病灶，而分数最高的是最小的块。
- **终审提前到出数之前**：出数之后才发现问题就要重算，而且已经报出去的数字收不回来。
- **中途交接合入 main**：训练要到第二天，协作者只读 main，最新状态不能只停在侧分支。
