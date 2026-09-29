# STATUS：2026-09-29 19:40（用户要求暂停。S7 脑部多病种：代码与全分支终审完成；已有两份单折读数——梗死 fold 0 早读 0.568 @ 1.66 误报/例、转移瘤 fold 1 0.719 @ 0.45，都不是达标结论；15 个训练里 3 个已完成、6 个在后台继续跑、6 个排队；nnDetection 第二臂 fold 0 在最后一轮；S4 设计概览已给用户、未获确认）

每次交接前整体重写本文件。五段固定：已验证、待拍板、下一步、坑与别重做、为什么。

## 0. 暂停时什么还在跑（2026-09-29 19:30）

用户 19:25 说"先将当前状态记住，然后保存，我需要你先暂停这里的任务"。本会话停掉了自己的两个后台监视（队列监视、nnDetection 监视），不再做任何新动作。**下面这些进程是脱离会话的，仍在后台运行，没有动它们；要不要停由用户定：**

| 进程 | pid | 位置 | 状态 |
|---|---|---|---|
| 训练队列 `scripts/gpu_queue.py`（旧代码） | 1091434 | 工作树 `../foundation_model-multidisease` | 还有 6 个任务排队：胶质瘤 fold 3、4，转移瘤 fold 3、4，梗死 fold 3、4 |
| 胶质瘤 fold 0 | 1091664 | GPU 0 | 第 128 轮，约 5 小时 |
| 胶质瘤 fold 1 | 1091671 | GPU 4 | 第 132 轮，约 5 小时 |
| 胶质瘤 fold 2 | 2005044 | GPU 6 | 第 84 轮，约 3 小时 |
| 转移瘤 fold 0 | 1091667 | GPU 1 | 第 181 轮，约 2 小时 |
| 转移瘤 fold 2 | 2025834 | GPU 2 | 第 79 轮，约 7 小时 |
| 梗死 fold 2 | 2501282 | GPU 5 | 第 7 轮，每轮 286 秒（与别的会话同卡），约 19 小时 |
| nnDetection fold 0 | 3334388 | GPU 3 | 第 59 轮（最后一轮），之后是 sweep；结束时可能卡在退出 |

- **只想让队列不再启动新训练**（正在跑的让它跑完）：`echo "paused by the user $(date '+%F %T')" > /data0/congcong/code/Project_Doing/foundation_model-multidisease/logs/brain_disease/stop`。这个文件建了就删不掉（本项目不删任何东西），之后要接着训剩下的折，得用新代码重新启动队列，并给日志起新名字（见 §4）。
- **想立刻腾出 GPU**：停训练进程要由用户自己执行或明确授权，例如 `kill -TERM 1091664`。nnU-Net 每 50 轮存一次 `checkpoint_latest.pth`，之后可以用 `nnUNetv2_train … --c` 续训，但队列会把已有结果目录的任务拒掉，续训要手动起。
- 已完成的三个训练（梗死 fold 0、1，转移瘤 fold 1）退出码都是 0，验证预测齐全。

## 1. 已完成且已验证

**本轮：S7 脑部多病种检测，阶段 A（每个病种一个 nnU-Net，各用原生序列）。** 规格 `docs/superpowers/specs/2026-09-29-brain-multidisease-design.md`（决定 M1–M14），计划 `docs/superpowers/plans/2026-09-29-brain-multidisease.md`（13 个任务）。分支 `build/brain-multidisease`（工作树 `../foundation_model-multidisease`）。台账 `../foundation_model-multidisease/.superpowers/sdd/2026-09-29-brain-multidisease/progress.md`（gitignore，不入库；终审原文 `final-review-1-report.md` 也在那里）。

- **测试**（2026-09-29 18:05，代码同分支头；nndet 环境那一条是 15:51 跑的，之后没有改动它涉及的文件）：
  ```
  PYTHONNOUSERSITE=1 PYTHONPATH=. nice -n 19 ~/anaconda3/envs/nvgen/bin/python -m pytest tests/ -q -p no:cacheprovider
  831 passed, 1 skipped in 117.51s (0:01:57)
  bash -c 'source scripts/nndet_env.sh && nice -n 19 python -m pytest tests/test_nndet_runner.py -q -p no:cacheprovider --noconftest'
  2 passed, 5 warnings in 2.80s
  ```
- **两份单折读数（都不是达标结论；达标线按五折折外预测合并判）**：

  | | 梗死 fold 0（计划里的早读） | 转移瘤 fold 1（先跑完的一折，不是计划里的早读） |
  |---|---|---|
  | 记录 | `docs/verification/2026-09-29/brain_multidisease/infarct_fold0/` | `…/metastasis_fold1_preliminary/` |
  | 扫描 / 计入病灶 / 忽略 | 50 / 428 / 33 | 86 / 580 / 91 |
  | 灵敏度 @ 每例误报（工作点） | 0.5678 @ 1.66（阈值 0.60） | 0.7190 @ 0.453（阈值 0.80） |
  | 不设阈值 | 0.5678 @ 1.68 | 0.7190 @ 0.581 |
  | 阈值 0.95 | 0.4229 @ 0.74 | 0.5569 @ 0.209 |
  | < 5 mm | 60/193 = 0.311 | 188/336 = 0.560 |
  | 5–10 mm | 107/152 = 0.704 | 139/152 = 0.915 |
  | ≥ 10 mm | 76/83 = 0.916 | 90/92 = 0.978 |
  | 每例误报：中位数 / 最大 / 超过 2 个的例数 | 1 / 12 / 14 | 0 / 6 / 4 |
  | Dice（只报告） | 0.753（49 例） | 0.827（86 例） |
  | 绑定一致率（NOT_EVIDENCE）主结构 / 侧别 | 0.947 / 0.988 | 0.981 / 0.993 |

  命令（把 `<病种>`、折号、输出目录换掉即可重跑；训练在跑时用 `--workers 2`）：
  ```
  PYTHONNOUSERSITE=1 PYTHONPATH=. nice -n 19 ~/anaconda3/envs/nvgen/bin/python scripts/eval_brain_disease.py --disease infarct --folds 0 --workers 2 --out docs/verification/2026-09-29/brain_multidisease/infarct_fold0
  Disease: infarct; folds [0]; kind early_reading
  Scans: 50; lesions counted: 428; ignored: 33
  Sensitivity 0.5678 at threshold 0.6 with 1.66 FP per scan
  Pass: None; stop remaining folds: False; early stop undecided: False
  Dice mean 0.753277521477103 over 49 cases
  ```
  - 两个病种的误报预算都没用满，所以这两个灵敏度就是各自那一折的上限，调阈值不会再涨。短板都在 5 mm 以下的病灶。
  - 转移瘤做没做过手术差别不大：没做过 338/474 = 0.713，做过 79/106 = 0.745。
  - 控制方另写了一段只用 scipy 的代码独立数了梗死 fold 0：计入 428、忽略 33、保留的预测块 327（= 243 命中 + 84 误报），与报告一致。终审者在这一折的五例真实输出上逐例、逐阈值重算，也一致。
  - 转移瘤比梗死好，可能因为增强 T1 上小病灶很亮、而梗死用的 DWI/ADC 只有 2 mm 分辨率。这只是猜测，没有单独验证。
- **真实句子已读过**（只在内存里生成，不写记录；`docs/verification/2026-09-29/brain_multidisease/checks/sentence_preview.py`，输出在同目录 `sentence_preview_{infarct_fold0,metastasis_fold1}.txt`）：梗死 fold 0 每例病灶中位数 5、最多 32，22 例超过 5 处，计数后补位置的 13 例、最多 4 个位置，最长句 220 字；转移瘤 fold 1 中位数 4、最多 26，27 例超过 5 处，补位置的 21 例、最多 5 个位置，最长句 207 字。读下来没有问题，"还见于"后面不需要设上限。
- **三个数据集**（Task 2，记录 `docs/verification/2026-09-29/brain_multidisease/build/`）：

  | 数据集 | 通道 | 扫描 / 病人 | 无标签体素的例 | 规划（间距 mm，块，batch） | 每折例数 |
  |---|---|---|---|---|---|
  | Dataset904_PDGMGlioma | T1、T1c、T2、FLAIR | 501 / 495 | 0 | 1.0 各向同性，128×160×112，2 | 100 / 101 / 100 / 99 / 101 |
  | Dataset905_BMSRMetastasis | T1pre、T1post、FLAIR | 461 / 314 | 0 | 1.5 × 0.859 × 0.859，80×192×160，2 | 105 / 86 / 87 / 106 / 77 |
  | Dataset906_ISLESInfarct | DWI、ADC | 250 / 250 | 3 | 2.0 各向同性，80×96×80，8 | 50 × 5 |
- **代码**：`anatobind/nnunet/brain_disease.py`、`anatobind/eval/lesion_components.py`、`anatobind/eval/detection_metrics.py`（`scan_matches` 与"忽略"标记）、`anatobind/bind/brain_lookup.py`、`anatobind/infer/brain_disease.py`、`anatobind/eval/brain_disease.py`；脚本 `scripts/brain_disease_{prepare,crossrun}.py`、`gpu_queue.py`、`eval_brain_disease.py`、`infer_brain_disease.py`。十个代码任务由子代理实现并逐个评审（sonnet）；终审后的修补由控制方直接落地（代码先在副本里验证，每个提交先看新测试在旧代码上失败）。
- **全分支终审（最强模型）共六次往返，最后结论："可以写逐例记录"，前提是用户确认句子的两处写法（§2）。** 终审者用另写的代码独立重算（分母、命中、误报），与分支一致；确认读的是折外预测、15 个折里没有病人跨训练与验证；读了本机 nnU-Net 2.8.0 的导出代码，确认"预测块里出现前景概率 < 0.5 的体素就报错"不会在正确输出上误报；把三十多种"改坏代码"的方式施加到副本上，除一种（git 调用是否加锁）外都有测试拦住。
- **句子现在的写法**（`anatobind/infer/brain_disease.py`）：
  - 写体积最大的 5 处，从大到小；其余计数，并补上"前 5 处都没提到的位置"，如"另有 21 处同类异常（还见于右侧大脑皮层、左侧小脑、左侧丘脑）"。**这两点是建议，待用户确认。**
  - 主结构前的侧别按"病灶落在主结构里的那部分体素"算；脑干不写侧别。
  - "累及"后面的结构都与主结构同侧时不写侧别；有一个不同侧就全部写出，脑干排最前。
  - 不与任何结构重叠的病灶：离最近结构 ≤ 10 mm 写"邻近<侧><结构>（未与任何结构重叠）存在…"，超过 10 mm 写"未能定位的区域存在…"（10 mm 是用户定的）。
  - 没检出写"本模型未检出<类型>（阈值 0.xx）。"，印象写"未检出相关异常"。
- **真实数据上的五项只读核查**（`docs/verification/2026-09-29/brain_multidisease/checks/`，脚本、命令与原始输出都在）：体积下限不会被文件头改动一格；1212 例的 SynthSeg 图都在且与标签同网格；3887 个通道文件与解剖图仿射完全相同；三组交叉运行的输入都在且同网格；没有一张解剖图缺宿主体素。
- **用户本轮拍板**：S7 走 C（先 A 后 B）；达标线与 S2 的 D1 同口径、三病种分别判；产出 = 检测器 + 查表绑定 + 结构化记录，脑叶留给 S4；五折全训、fold 0 先报（"B，不用留卡"）；方法 nnU-Net 3d_fullres；采纳两条建议（交叉误报只报告；S4 设计与训练并行）。算力原话："后续整个服务器的算力都要优先该任务使用，但是前提是只能是考虑或者占用空的 GPU"。S4 的老师用方案 B。2026-09-29 16:52 用户回复"按照你的建议"：距离上限 10 mm；控制方同时把它理解为此前列出的几项也按建议办（句子三处措辞、"未定"的早停不自动停、GPU 冲突先等、S4 输出粒度 A），并已在聊天里说明这一理解，用户未提出异议。

**nnDetection 第二臂（S2）**：fold 0 自 04:24 在 GPU 3 训练，19:29 在第 59 轮（最后一轮），3.1 步/秒。计划 Task 1–8 已在 main。

**S4 脑部解剖层（设计中，未获用户确认，没有写任何代码）**。只读探查的事实（脚本与原始输出在 `docs/verification/2026-09-29/s4_probe/`，叠加图在 `~/figs/foundation_model/s4probe/skullstrip_*.png`）：
- `/data2/congcong/data/FM_data/SibBMS_ms/sibbms.zip`（11 GB，未解压）：健康人 100 人（T1、T2、FLAIR），多发性硬化 93 人 / 272 次检查（另有增强 T1），全部 197×233×189、1 mm、RAS、已去颅骨并在标准空间。本机所有带 FLAIR 的数据集都去过颅骨。
- SynthSeg 在 fastMRI FLAIR（447 卷）上的脑轮廓：中间层面（第 2–9 层）贴合脑表面；最低一两层不可靠（有的把脑后部约四分之一划在外面，有的把中央深部标成背景）；颅顶只覆盖一部分；14 卷的轮廓不到 300 mL，其中一卷是空的。**所以不能直接拿它去颅骨**，概览里"整体轮廓可用"的说法已向用户更正。

**之前（保留）**：S2 nnU-Net 检测器 D1 门不过（2d 五折 0.3662 @ 1.636 FP/卷，主因漏检），记录索引 `docs/verification/2026-09-28/brain_detector/README.md`。S1 关系基线（NOT_EVIDENCE）、Level R 读片工具（冒烟服务 8791，浏览器验收仍 USER_REPORTED）同前。

## 2. 待用户拍板

1. **暂停的范围**：本会话已停手；后台的队列与训练还在跑（§0）。要不要也停掉它们，等用户一句话。
2. **其他会话持续在我们占着的卡上启动任务。** 我们的训练每个只占约 8 GB，别的会话按剩余显存挑卡就会落到这些卡上。经过：14:31 GS 会话（`mcgs_ce_mask_wt`，`CE_retro_cssense_af16_vdpois`）在 GPU 0、3、4、1 起任务；17:02 起膝关节筛选会话（`mcgs_nb_knee_wt`）先后在 GPU 6、7、5 起任务，其中两次与我们的队列在同两秒内抢到同一张卡；17:33 与 18:50 前后 GS 会话又在 GPU 0、1、2、4 各起一个 25–33 GB 的任务。19:29 时只有 GPU 3（nnDetection）和 GPU 6 是我们独占。后果是被共用的训练慢 1.5–4 倍；梗死 fold 2 按现在的速度要 19 小时，胶质瘤 fold 2 曾被估到 23.5 小时。没有训练失败。本会话没有动任何不是自己启动的进程。已给用户三个选项：A 维持现状；B 用户去告诉那两个会话只用没有任何进程的空卡；C 授权本会话给它们发消息。用户未答复。
3. **句子按体积从大到小写前 5 处**（原规格是按分数）。理由：分数是块内概率均值，小块最高；真实一例 26 处病灶里，按分数只写出 16–64 mm³ 的五处，最大的 744 mm³ 被并进"另有 21 处"。代码已按建议改，规格里标为待确认。
4. **"另有 N 处"后面补位置**（"还见于…"）。同上，已按建议改，待确认。
5. **S4 设计概览**（2026-09-29 约 17:25 在聊天里给出，19:20 更正了去颅骨一条）：老师 = SynthSeg 在 1 mm T1 上出标签；学生看同病人的 FLAIR，处理成 fastMRI 的样子（每 5 层合 1 层、只取基底节到颅顶的 14–16 层、随机倾斜、亮度不均与噪声）；训练数据以 SibBMS 健康人和 MS 为主，BMSR、PDGM 也用但病灶区域不参与训练；nnU-Net；去颅骨改为自己训一个脑轮廓模型（标签用 SynthSeg 在 fastMRI 中间层面的轮廓，最低两层和颅顶不参与训练），备选是现成工具 HD-BET；S4 的数字都是"和老师有多一致"，不是"解剖有多对"，真正的证据要靠医生抽查。方向对的话再分段细化。
6. **push**：main 自 afe641e 起所有提交与 tag 都只在本地。
7. **nnDetection fold 0 出结果后**：规则 A（≥ 106/280）通过就要补五折，开跑前问；不通过就停。
8. **读片人**、Level R 读片说明里 "other" 的定义、伦理备案：同前。
9. **可删清单（只列，不删；删除由用户执行）**：S2 工作树 `../foundation_model-detector` 与分支 `build/brain-detector`（已合并）；`/data2/congcong/data/FM_data/derived/nndet_smoke/`；`~/logs/nndet_install/wheels/` 里 1.6 GB 的 torch 安装包；会话 scratch（`/tmp/claude-1002/-home-congcongliu--claude/614ccd9c-f8e8-435b-afc8-47ae37fe3721/scratchpad/` 下的 `plan_dryrun`、`s7_dryrun`、`s7_nnunet`、`s7_e2e*`、`s7_readings`、`s7probe`、`s4probe`、`s7_checks`、`review_*` 等）。
10. 旧遗留：Q9 删除授权、其余 4850 卷 SynthSeg、Redivis token、两条远端评审分支去留、RSS 左右手性换算规则。

**终审给用户的提醒**（已在聊天里转达）：达标数字取决于分数的分布，nnU-Net 的分数集中在 1 附近；"无工作点"与找到多少病灶无关；"疑似 X"只说明跑的是 X 模型；数据里几乎没有正常脑（1212 例里 3 例无病灶），"未检出"不能当阴性；解剖词来自伪标签上的体素计票；左右取决于文件头（未独立核实）；侧别按 40% 规则判，一个 65/35 的跨中线肿瘤会写成单侧；10 mm 在层厚约 5 mm 的扫描里只相当于两层。

## 3. 下一步（恢复时从这里接）

1. **先看后台**：`tail -20 ../foundation_model-multidisease/logs/brain_disease/queue.log`，每个 `finished` 都核对退出码（正在跑的队列是旧代码，失败只记一行就继续）；`nvidia-smi --query-compute-apps=gpu_uuid,pid,used_memory --format=csv` 看谁在用卡。出现非零退出码先读该任务的日志，若原因会影响每次启动就建 `logs/brain_disease/stop` 并报用户。
2. **nnDetection**：判据是训练目录 `/data2/congcong/data/FM_data/derived/nndet_models/Task903_FastMRIBrainSmallLesion/RetinaUNetV001_D3V001_3d/fold0` 的 `train.log` 出现 `Found 51 predictions for analysis`、`plan_inference.pkl` 存在、`sweep_predictions/` 有 51 个 `*_boxes.pt`。输出齐了而进程 3334388 还在，就是卡在退出：`kill -TERM 3334388`（本项目自己的进程），把进程号、时间、退出码记进 `training.txt`。然后做其计划 Task 9（提取、fold 0 报告、规则 A、推理冒烟）与 Task 10。
3. **计划 Task 11，转移瘤与胶质瘤 fold 0 的早读**（梗死已做）。前提：该折 `validation/summary.json` 存在，且队列日志有 `finished ('<病种>', 0) … exit code 0`。先输出到 scratch 看一眼，再写进仓库：
   ```
   PYTHONNOUSERSITE=1 PYTHONPATH=. nice -n 19 ~/anaconda3/envs/nvgen/bin/python scripts/eval_brain_disease.py --disease <病种> --folds 0 --workers 2 --out docs/verification/2026-09-29/brain_multidisease/<病种>_fold0
   ```
   读 `verdict.json`：`early_stop_undecided` 为 true 就不写 skip 文件，把 FROC 表和"刚超预算的那一行"报给用户；`stop_remaining_folds` 为 true 才写 `logs/brain_disease/skip_<数据集号>`（写了就删不掉，只有用户能撤）。提交读数（`git commit -m "Brain disease <disease>: fold 0 early reading (not the gate)"`），报告用户并写明"这是早读，不是达标结论"。
4. **写逐例记录之前**：用户确认 §2 第 3、4 条；用胶质瘤的真实输出先生成几句读一遍（`checks/sentence_preview.py glioma 0 <阈值> 3`），重点看带卫星灶的胶质瘤。
5. **计划 Task 12**：队列日志出现 `queue empty, nothing running: done`，且每个要评估的折都有 `finished … with exit code 0` 后：三个病种五折评估（逐例记录写到 `/data2/congcong/data/FM_data/derived/brain_disease/<病种>/records`）、三组交叉误报、每病种一例推理冒烟（每次用新的输出目录）、记录索引 README（写明匹配规则与 S2 相同、阈值是单折模型上测的）。
6. **计划 Task 13**：更新本文件与 CLAUDE.md、记录复核、合回 main、tag `handoff/<日期>-brain-multidisease`，不 push。
7. **S4 设计**：用户点头后分段细化（数据、仿真、去颅骨、验证与达标线、推理），再写规格与计划。SibBMS 要先解压、再用 CPU 跑 SynthSeg（372 次，约 6 小时），CPU 现在贴着上限，得等 S7 的训练跑完一批。
8. 阶段 B（只吃 FLAIR 的统一模型）等阶段 A 出数后再设计。

## 4. 坑与别重做

- **训练日志滞后**：训练的标准输出重定向到文件是块缓冲，`logs/brain_disease/Dataset90*_fold*.log` 会落后十几分钟。每轮耗时读 nnU-Net 结果目录里的 `training_log_*.txt`。
- **CPU 已贴着上限**：六个训练加 nnDetection 实测 47.2 核（上限 48）。训练期间任何评估都加 `--workers 2`，全量测试一次只跑一个。
- **别的会话会把任务放到我们占着的卡上**：队列只保证自己不抢别人的卡，防不住别人；空卡一出现，几个会话会在同一两秒内抢到同一张。发现变慢先看 `nvidia-smi --query-compute-apps`，不是自己启动的进程一律不动，报用户。
- **这里不能删任何东西**，所以半成品目录会一直留着并挡住下一次同名运行：脚本都是"先检查、后建目录"；推理冒烟失败后换一个新的 `--out`。读数先输出到 scratch 看一眼、再写进仓库，是因为仓库里的目录写了就不能重来。
- **队列重启不要覆盖日志**：13:46 那次启动写的是 `logs/brain_disease/queue.log`，之后任何一次启动都用新名字（`queue_$(date +%Y%m%d_%H%M%S).log`）。重启用的是新代码：训练启动 10 分钟内失败就不再启动新任务，最后一行给出成功、失败、被拒、未启动的计数。已有结果目录或日志的任务会被拒绝启动，这是有意的。
- **句子的每一处写法都要拿真实输出读一遍**：终审五轮里每一轮都是在真实或接近真实的病灶上才发现句子的问题（按分数排序、侧别词串读、"另有 N 处"藏住位置、括号套括号）。单元测试守得住规则，守不住"读起来对不对"。
- **先验证再下结论**：S4 概览里"SynthSeg 的整体轮廓可用"是没验证就写的，验证后发现最低层面和颅顶不可靠，已更正。
- **匹配规则与 S2 相同**：取总 IoU 最大的一对一指派，再去掉 IoU < 0.1 的对；拥挤时偶尔比最优的一对一匹配少一个命中，命中数也因此不一定随阈值单调。为了与 S2 可比不改，记录索引 README 里要写明。
- **分数的性质**：检测分数是连通块内前景概率的均值，块来自 argmax 图，所以分数都大于 0.5；FROC 表里阈值 ≤ 0.50 的各行相同；小块的分数最高。
- **绑定一致率不是证据**：真值和预测都用同一张 SynthSeg 伪标签图查表。
- **计划里的代码事先跑过**：计划的每段代码都在分支的临时副本里执行过；临时副本在会话 scratch 里，是一次性的，以仓库里的计划为准。
- 台账里 17:28–17:55 的几个时刻是估的（已写成"约"）；要准的时间以提交时间和队列日志为准。
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
- **暂停时不动后台训练**：用户说的是暂停"这里的任务"；训练已经跑了五个多小时，停掉是不可逆的损失，而队列和训练都脱离会话、不需要本会话看着也能跑完。要停由用户明确说。
- **中途交接合入 main**：协作者只读 main，最新状态不能只停在侧分支。
