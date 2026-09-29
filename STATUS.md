# STATUS：2026-09-29（S7 脑部多病种：十个代码任务已实现并逐个评审，全分支终审结论"可以出数"、终审提出的修补已落地，15 个训练由队列在空卡上跑，预计 09-30 出齐；nnDetection 第二臂 fold 0 仍在训练；S4 脑部解剖层在设计中；本轮中途交接）

每次交接前整体重写本文件。五段固定：已验证、待拍板、下一步、坑与别重做、为什么。

## 1. 已完成且已验证

**本轮：S7 脑部多病种检测，阶段 A（每个病种一个 nnU-Net，各用原生序列）。** 规格 `docs/superpowers/specs/2026-09-29-brain-multidisease-design.md`（决定 M1–M14），计划 `docs/superpowers/plans/2026-09-29-brain-multidisease.md`（13 个任务）。分支 `build/brain-multidisease`（工作树 `../foundation_model-multidisease`）。执行 = 子代理逐任务：实现与逐任务评审用 sonnet，控制方重跑测试、检查提交信息、与事先验证过的代码逐字节比对；逐任务裁定在台账 `../foundation_model-multidisease/.superpowers/sdd/2026-09-29-brain-multidisease/progress.md`（gitignore，不入库）。

- **测试**（2026-09-29 16:25，分支头 ecb172d；nndet 环境那一条是 15:51 在 a8668f7 上跑的，之后没有改动它涉及的文件）：
  ```
  PYTHONNOUSERSITE=1 PYTHONPATH=. nice -n 19 ~/anaconda3/envs/nvgen/bin/python -m pytest tests/ -q -p no:cacheprovider
  824 passed, 1 skipped in 108.54s (0:01:48)
  bash -c 'source scripts/nndet_env.sh && nice -n 19 python -m pytest tests/test_nndet_runner.py -q -p no:cacheprovider --noconftest'
  2 passed, 5 warnings in 2.80s
  ```
- **三个数据集已建好**（Task 2，记录 `docs/verification/2026-09-29/brain_multidisease/build/`）：

  | 数据集 | 通道 | 扫描 / 病人 | 无标签体素的例 | 规划（间距 mm，块，batch） | 每折例数 |
  |---|---|---|---|---|---|
  | Dataset904_PDGMGlioma | T1、T1c、T2、FLAIR | 501 / 495 | 0 | 1.0 各向同性，128×160×112，2 | 100 / 101 / 100 / 99 / 101 |
  | Dataset905_BMSRMetastasis | T1pre、T1post、FLAIR | 461 / 314 | 0 | 1.5 × 0.859 × 0.859，80×192×160，2 | 105 / 86 / 87 / 106 / 77 |
  | Dataset906_ISLESInfarct | DWI、ADC | 250 / 250 | 3 | 2.0 各向同性，80×96×80，8 | 50 × 5 |

  图像是指向原文件的软链接，标签重写成二类并保留原文件头；折按病人（种子 0），评审者全量核对：3887 个软链接都能解析、1212 个标签与 `cases.json` 的体素数一致、0 个病人跨折。
- **训练队列已启动**（Task 3，记录 `docs/verification/2026-09-29/brain_multidisease/launch.md`）：2026-09-29 13:46:13 启动，队列进程 pid 1091434；首批六个训练在 GPU 0、1、2、4、5、6（三个病种各 fold 0、fold 1），其余九个等卡。启动后 16 分钟实测每轮：胶质瘤 93–99 s、转移瘤 78–84 s、梗死 45–46 s，单个训练约 7.2 h / 6.1 h / 3.4 h，都远低于 24 h。队列只在空卡上启动、从不向任何进程发信号；控制文件在 `logs/brain_disease/`：`skip_<数据集号>` 停某病种的新折，`stop` 停止启动新任务。
- **代码**（Task 1、4–10，全部评审通过）：`anatobind/nnunet/brain_disease.py`、`anatobind/eval/lesion_components.py`、`anatobind/eval/detection_metrics.py`（加 `scan_matches` 与"忽略"标记）、`anatobind/bind/brain_lookup.py`、`anatobind/infer/brain_disease.py`、`anatobind/eval/brain_disease.py`；脚本 `scripts/brain_disease_{prepare,crossrun}.py`、`gpu_queue.py`、`eval_brain_disease.py`、`infer_brain_disease.py`。分支自 main 起 22 个代码与测试文件，逐个与计划里的代码一致。
- **评审后追加的修补**（都已写回计划与规格）：
  1. `per_family` 也认"忽略"标记（4c55ca9）；不带标记时新旧函数在 8000 次随机比较中结果相同。
  2. 推理入口在预测之前逐通道核对：与解剖图同形状、仿射最大差 ≤ 1e-3；不符即拒绝，此时不写任何文件（56b930f）。
  3. **没有工作点时**（阈值网格上每例误报始终 > 2）：判定写 `operating_point: false`、灵敏度记空、**不自动停其余各折**，报用户定；五折齐全时无工作点即不过线（7b66d16）。
  4. 交叉误报脚本写明所用病例（数据方 fold 0 的验证病例）与计数规则，并在建目录之前检查模型权重与全部通道文件（5f38722、404eb9d）；评估脚本全部算完才建输出目录（404eb9d）。
- **全分支终审（最强模型，出数之前做，16:05 返回）：结论"可以出数"，没有严重问题。** 终审者用另写的代码独立重算，在预演小模型的真实 nnU-Net 输出上，三个病种、四个阈值的（分母、命中、误报）都与分支一致；确认评估读的是折外预测、15 个折里没有病人跨训练与验证、从标签文件到句子的 11 个交接环节约定一致。它提出 3 个重要问题和 9 个小问题，处理如下（提交 68a2017、1598009、39f2ded、de8fe83、ecb172d；代码事先在副本里验证，由控制方直接落地，每个提交先看新测试在旧代码上失败）：
  1. **早停规则**：阈值网格步长 0.05，最后一格可能误判。现在自动停要求工作点灵敏度 < 0.3 **且**"刚超预算的那一行"灵敏度也 < 0.3；其余偏低的读数（含无工作点）判定记 `early_stop_undecided`，各折照常训，报用户定。
  2. **队列**（供下次启动；正在跑的队列是旧代码）：训练启动 10 分钟内失败就不再启动新任务；最后一行给出成功、失败、被拒、未启动的计数，返回码随之；每次启动写带时间的日志名。
  3. **句子措辞**（用户可改）：不与任何结构重叠的病灶写"邻近<侧><结构>（未与任何结构重叠）存在…"；主结构是脑干时不写侧别词；没检出写"本模型未检出<类型>（阈值 0.xx）。"、印象"未检出相关异常"。
  4. 其余：体积下限对同一网格的两个 float32 文件头给出同一个值；预测块里出现前景概率 < 0.5 的体素就报错（防轴序错位）；记录带 `model_folds` 与 `anatomy_source`；报告里加每例误报的分布与"按最近距离定位"的比例；补了终审指出的"测试发现不了"的几类回归。
  5. 用预演小模型的真实输出端到端重跑了修补后的评估脚本：胶质瘤 1.0 @ 0.95；转移瘤 0.125 @ 0.90（该读数会触发停）；梗死无工作点（0.95 处灵敏度 0.5、每例误报 2.5，未定）。这些是 5 个 epoch、每病种 2 例的小模型，**NOT_EVIDENCE**，只说明代码链路通。
  终审全文存于台账目录 `final-review-1-report.md`。
- **终审修补的复审（16:38 返回）：结论仍是"可以出数"。** 复审者把 12 种改坏代码的方式（分数取最大值、体素体积写死、轴序反转、记录阈值写死、`>=` 改 `>`、队列忽略 skip 或 stop 文件、关掉概率检查、关掉早停的第二个条件、关掉快速失败规则等）逐一施加到副本上，每一种现在都至少有一个测试失败；读了本机安装的 nnU-Net 2.8.0 的导出代码，确认标签图就是所存 float32 概率图的 argmax（五折集成是先平均 logits 再走同一条路），所以新加的概率检查不会在正确输出上误报。它新提的 4 个小问题都不影响早读，见 §3 第 3 条。
- **真实数据上的五项只读核查**（脚本、命令与原始输出在 `docs/verification/2026-09-29/brain_multidisease/checks/`）：10 mm³ 下限不会被 float32 文件头改动一格；1212 例的 SynthSeg 图都在，且与标签同网格（仿射偏差 0）；3887 个通道文件与解剖图仿射完全相同；三组交叉运行的输入都在且同网格；没有一张解剖图缺宿主体素。
- **用户本轮拍板**：S7 走 C（先 A 后 B）；达标线与 S2 的 D1 同口径、三病种分别判；产出 = 检测器 + 查表绑定 + 结构化记录，脑叶留给 S4；划分改为五折全训、fold 0 先报（"B，不用留卡"）；方法 nnU-Net 3d_fullres；采纳两条建议（交叉误报检查只报告；S4 设计与训练并行）。算力原话："后续整个服务器的算力都要优先该任务使用，但是前提是只能是考虑或者占用空的 GPU"。S4 的老师用方案 B（外部数据集整头高分辨 T1 上的 SynthSeg 当老师，学生看同病人的 FLAIR 并处理成 fastMRI 的样子）。

**nnDetection 第二臂（S2）**：fold 0 自 04:24 在 GPU 3 训练，15:49 在第 48 轮（共 60 轮）。14:31 起被共用同一张卡的任务拖慢（见 §2），原先 3.6 步/秒，现在约 0.25 步/秒。计划 Task 1–8 已在 main（tag `handoff/2026-09-29-nndet-fold0-training`）。

**之前（保留）**：S2 nnU-Net 检测器 D1 门不过（2d 五折 0.3662 @ 1.636 FP/卷，主因漏检），记录索引 `docs/verification/2026-09-28/brain_detector/README.md`。S1 关系基线（NOT_EVIDENCE）、Level R 读片工具（冒烟服务 8791，浏览器验收仍 USER_REPORTED）同前。

**还没有任何 S7 的检测数字。** 第一份读数（梗死 fold 0 的早读，不是达标结论）要等该折训练与验证预测结束。

## 2. 待用户拍板

- **GPU 冲突**：2026-09-29 14:31:52，同一用户的另一个会话（GS 项目，工作树 `mcgs_ce_mask_wt`，实验 `CE_retro_cssense_af16_vdpois`）在我们训练占着的卡上启动了任务：GPU 0（pid 1333445）、GPU 3（1333443）、GPU 4（1333440）各一个 MC-GS（4000 步，15:23 时约 1350 步），GPU 1 上是一条经典重建链。后果：胶质瘤两折每轮 95 s → 约 290 s，转移瘤 fold 0 在 65–257 s 之间波动，nnDetection 3.6 → 0.25 步/秒。本会话没有动这些进程。已给用户三个选项（A 等它们跑完；B 用户去那个会话挪走；C 授权本会话处理并点名进程），用户回复"continue"，未选，按 A 进行。
- **早停的两种"未定"情形**（§1 修补 3 与终审第 1 条）：规格 M4 只说了"fold 0 灵敏度 < 0.3 就停"。没有工作点，或工作点 < 0.3 而刚超预算的那一行 ≥ 0.3 时，现在都不自动停、报用户定。用户可改成自动停。
- **句子措辞**（终审第 3 条）：三处改动按终审建议先落地，已在聊天里告知用户，用户未表态。逐例记录到计划 Task 12 才写，之前都可以改。
- **终审给用户的六条提醒**（原文在 `final-review-1-report.md`）：达标数字取决于分数的分布，nnU-Net 的分数集中在 1 附近；"无工作点"与找到多少病灶无关；"疑似 X"只说明跑的是 X 模型；数据里几乎没有正常脑（1212 例里 3 例无病灶），"未检出"不能当阴性；解剖词来自伪标签上的体素计票；左右取决于文件头（未独立核实）。
- **S4 第二问**：解剖模型输出到多细。A 分两步（现在先做大类加左右侧，分叶等权重到位再加，推荐）；B 等分叶权重到位一次做完；C 只做大类。SynthSeg 的分叶与 QC 权重本机没有，FreeSurfer 官方两个地址 2026-09-29 连不上。
- **push**：main 自 afe641e 起所有提交与 tag 都只在本地。
- **nnDetection fold 0 出结果后**：规则 A（≥ 106/280）通过就要补五折，开跑前问；不通过就停。
- **读片人**、Level R 读片说明里 "other" 的定义、伦理备案：同前。
- **可删清单（只列，不删；删除由用户执行）**：S2 工作树 `../foundation_model-detector` 与分支 `build/brain-detector`（已合并）；`/data2/congcong/data/FM_data/derived/nndet_smoke/`；`~/logs/nndet_install/wheels/` 里 1.6 GB 的 torch 安装包；会话 scratch（`/tmp/claude-1002/-home-congcongliu--claude/614ccd9c-f8e8-435b-afc8-47ae37fe3721/scratchpad/` 下的 `plan_dryrun`、`s7_dryrun`、`s7_nnunet`、`s7_e2e`、`s7probe`、`s4probe`、`s7_checks`、`review_*` 等）。
- 旧遗留：Q9 删除授权、其余 4850 卷 SynthSeg、Redivis token、两条远端评审分支去留、RSS 左右手性换算规则。

## 3. 下一步

1. **评估记录出来后**再补一次针对记录与文档的复核（终审已覆盖代码）。
2. **计划 Task 11，每个病种 fold 0 的早读**。前提：`<results>/<Dataset>/nnUNetTrainer_250epochs__nnUNetPlans__3d_fullres/fold_0/validation/summary.json` 存在，且 `logs/brain_disease/queue.log` 有 `finished ('<病种>', 0) … exit code 0`。命令（训练还在跑时只开 2 个进程）：
   ```
   PYTHONNOUSERSITE=1 PYTHONPATH=. nice -n 19 ~/anaconda3/envs/nvgen/bin/python scripts/eval_brain_disease.py --disease <病种> --folds 0 --workers 2 --out docs/verification/2026-09-29/brain_multidisease/<病种>_fold0
   ```
   读 `verdict.json`：`early_stop_undecided` 为 true 就不写 skip 文件，把 FROC 表和那两行报给用户；`stop_remaining_folds` 为 true 才写 `logs/brain_disease/skip_<数据集号>`（写了就删不掉，只有用户能撤）。提交读数，报告用户并写明"这是早读，不是达标结论"。
3. **写逐例记录之前（计划 Task 12 Step 1 之前）要先修的四处**，都来自复审，都不影响早读：
   - 侧别词按整个病灶统计，却写在主结构前面：55% 左丘脑 + 45% 右白质会写成"双侧丘脑"。改成绑定时另算"主结构那部分体素"的侧别，句子用它。
   - "邻近"不带距离：离最近结构 30 个体素也写"邻近"。记录里加 `host_distance_mm`；超过多远改写"未能定位的区域"由用户定。
   - "刚超预算的那一行"不是严格上界（匹配按总 IoU 最大做，每个阈值重做；随机拥挤场景 20000 次里 32 次沿阈值多出一个命中）：改措辞，并把离 0.3 不到两个病灶的那一行也算作"够得着"。
   - 无工作点且 0.95 那一行灵敏度已 < 0.3 时，读数其实已经有结论，报告里加一句；在此之前由控制方在给用户的消息里说明。
4. **计划 Task 12**：队列日志出现 `queue empty, nothing running: done` 后，三个病种五折评估（写逐例记录到 `/data2/congcong/data/FM_data/derived/brain_disease/<病种>/records`）、三组交叉误报、每病种一例推理冒烟（每次冒烟用新的输出目录）、记录索引 README。
5. **计划 Task 13**：更新本文件与 CLAUDE.md、记录复核、合回 main、tag `handoff/<日期>-brain-multidisease`，不 push。
6. **nnDetection**：训练与 sweep 结束后做其计划 Task 9（提取、fold 0 报告、规则 A、推理冒烟）与 Task 10。进程 3334388 结束时可能卡在退出，输出齐了再 `kill -TERM`（那是本会话自己的进程）。
7. **S4 设计**：等用户回答第二问后继续（训练数据、仿真 fastMRI 的几何、验证口径、去颅骨），再写规格与计划。
8. 阶段 B（只吃 FLAIR 的统一模型）等阶段 A 出数后再设计。

## 4. 坑与别重做

- **训练日志滞后**：训练的标准输出重定向到文件是块缓冲，`logs/brain_disease/Dataset90*_fold*.log` 会落后十几分钟。每轮耗时读 nnU-Net 结果目录里的 `training_log_*.txt`。
- **CPU 已贴着上限**：六个训练加 nnDetection 实测 47.2 核（上限 48）。训练期间任何评估都加 `--workers 2`，全量测试一次只跑一个。
- **别的会话会把任务放到我们占着的卡上**：队列只保证自己不抢别人的卡，防不住别人。发现变慢先看 `nvidia-smi --query-compute-apps`，不是自己启动的进程一律不动，报用户。
- **这里不能删任何东西**，所以半成品目录会一直留着并挡住下一次同名运行：脚本都改成"先检查、后建目录"；推理冒烟失败后换一个新的 `--out`。
- **队列重启不要覆盖日志**：13:46 那次启动写的是 `logs/brain_disease/queue.log`，之后任何一次启动都用新名字（`queue_$(date +%Y%m%d_%H%M%S).log`），否则退出码的唯一记录会被截断。正在跑的队列是旧代码：训练失败只记一行就继续，所以每个 `finished` 都要核对退出码；出现非零码先读该任务的日志，若原因会影响每次启动就建 `logs/brain_disease/stop` 并报用户。
- **匹配规则与 S2 相同**：取总 IoU 最大的一对一指派，再去掉 IoU < 0.1 的对；拥挤时偶尔比最优的一对一匹配少一个命中（终审随机实验 20000 次里 47 次），从不多。为了与 S2 可比不改，记录索引 README 里要写明。
- **分数的性质**：检测分数是连通块内前景概率的均值，块来自 argmax 图，所以分数都大于 0.5；FROC 表里阈值 ≤ 0.50 的各行相同，每例 2 个误报的预算可能压不到。
- **绑定一致率不是证据**：真值和预测都用同一张 SynthSeg 伪标签图查表。
- **计划里的代码事先跑过**：计划的每段代码都在分支的临时副本里执行过；临时副本在会话 scratch 里，是一次性的，以仓库里的计划为准。
- nnDetection 的环境、约定、下载的坑同上一版（`docs/nndet_install.md`）；zsh 的 `pgrep -f` 会匹配到自己，等待循环用 `ps -p <pid>`。
- 别在 nnU-Net 上调参救线（M5、D9）；C1 上的数字都不是证据。

## 5. 关键决定的为什么

- **先 A 后 B（M1）**：病种与数据集完全混杂（每个病种只来自一个数据集），统一模型可以只靠图像风格分辨病种。A 先给出每个病种在原生序列上的上限，B 需要的标注转换、按病人划分、评估代码 A 都先做出来。
- **按 10 mm³ 而不是按体素数定下限（M7）**：三个数据集体素大小差很多（转移瘤一个数据集内下限就从 5 格到 55 格），按 9 个体素会去掉 55% 的梗死。
- **五折全训、fold 0 先报（M4）**：用户说不用留卡；五折折外预测让每个病例都有一条记录，数字也更稳。
- **无工作点不自动停**：误停会错杀一个可能过线的病种，之后还要重启；多训四折只多花空闲 GPU 时间。
- **推理前核对网格**：体积和最近结构都按解剖图的体素大小算，形状相同而网格不同会悄悄算错，而这正是要交给用户的那句话。
- **终审提前到出数之前**：代码已全部写完，出数之后才发现问题就要重算，而且已经报出去的数字收不回来。
- **中途交接合入 main**：训练要到第二天，协作者只读 main，最新状态不能只停在侧分支。
