# STATUS：2026-10-08 上午（S4 脑部解剖层训完并评估：按预先定好的 A11 判"不过"，诊断显示失败的那个数量的是 SynthSeg 参照缺深部结构；未调参，三条出路待用户拍板；主线在 main，tag `handoff/2026-10-08-brain-anatomy-flair`，未 push——push 被本会话的权限分类器拒绝，请用户自己推）

每次交接前整体重写本文件。五段固定：已验证、待拍板、下一步、坑与别重做、为什么。

## 1. 已完成且已验证

**S4 脑部解剖层（fastMRI FLAIR 上的学生模型）十个任务做完。** 规格 `docs/superpowers/specs/2026-10-02-brain-anatomy-flair-design.md`（决定 A1–A17，A17 是 10-03 增补），计划 `docs/superpowers/plans/2026-10-02-brain-anatomy-flair.md`。直接在 main 上做（用户同意不开工作树）；九个代码任务由子代理逐任务实现并评审（实现用最便宜的模型按计划原文抄写、控制器逐字节比对；Task 6/7/9 各一轮修补；A17 修补由控制器自己改、独立评审通过），台账 `.superpowers/sdd/2026-10-02-brain-anatomy-flair/progress.md`（gitignore）。记录索引 **`docs/verification/2026-10-02/brain_anatomy_flair/README.md`**（每个数字标了出处）。

- **测试**（main，2026-10-08）：
  ```
  PYTHONNOUSERSITE=1 PYTHONPATH=. nice -n 19 ~/anaconda3/envs/nvgen/bin/python -m pytest tests/ -q -p no:cacheprovider
  872 passed, 1 skipped in 85.72s (0:01:25)
  ```
- **数据**（`build/`）：三来源 1320 例（SibBMS 358 例 / 185 人、PDGM 501、BMSR 461），按病人留 20% 测试；仿真 22:08–22:34（8 进程）出 4140 训练样本（每训练例 K=4）+ 285 测试样本，23 GB；Dataset907（学生，3d_fullres，patch [16, 320, 320]，batch 2，spacing 5 / 0.6875 / 0.6875；按病人五折 872/856/836/788/788）；Dataset908（脑轮廓，2d，fastMRI 447 卷去掉 14 卷失败卷 → 345 训 / 88 测，五折 69×5）。
- **训练**（`launch.md`、`training.txt`；每个训练一张空卡，`scripts/brain_anatomy_train.py`）：轮廓模型 GPU 7，22:22 → 01:22，41.4 s/轮，fold 0 验证 Dice 0.9704；学生模型（无镜像）GPU 5，22:56 → 02:32，49.0 s/轮，fold 0 验证（仿真卷）13 类平均 0.7004（白质 0.877/0.878、皮层 0.810/0.815、丘脑 0.833/0.845、基底节 0.823/0.833、脑干 0.385、小脑 0.418/0.399、深部灰质 0.539/0.474、脑室 0.877）。
- **A17 镜像缺陷**：第一次学生训练用了默认训练器，nnU-Net 默认镜像把图像和标签一起翻转、标签值不变，分侧类别学不出来（前 9 轮白质伪 Dice 左右来回摆 0.18/0.50 → 0.52/0.06，不分侧的脑室正常 0.77；`debug.json` 镜像轴 (0,1,2)）。改用 nnU-Net 自带的 `nnUNetTrainer_250epochs_NoMirroring`（训练、推理都不镜像；代码 d96fca9，规格/计划 dee2213）后第 1 轮白质左右就到 0.75/0.75。第一次训练停不掉（停进程被权限分类器拒绝），跑到 02:21 自己结束，验证 Dice 0.5587、小脑 0/0，结果不用、结果夹留作证据。
- **A11 评估（2026-10-04 02:33–02:49，GPU 7，`eval/`；433 个真实 fastMRI 卷，参照 = 同卷的 SynthSeg 伪标签，NOT_EVIDENCE）**：

  | 门 | 数值 | 阈值 | 结果 |
  |---|---|---|---|
  | 病灶框宿主一致率（学生查表 vs SynthSeg 查表，`BRAIN_PARENCHYMA`） | 0.9215（881/956；注册表 1297 处病灶，30 处在 14 个排除卷，311 处有层落在可靠范围外） | ≥ 0.90 | 过 |
  | 可靠层分侧宿主 Dice（13 类等权平均） | 0.2665 | ≥ 0.80 | **不过** |
  | 轮廓 Dice（88 个测试卷） | 0.9781（最低 0.8272，4 卷 < 0.95） | ≥ 0.97 | 过 |

  **结论：不过。未调参。** 逐类：白质 0.8334/0.8360、皮层 0.7798/0.7787、丘脑 0.0240/0.0031、基底节 0.1345/0.0748、脑干 0、小脑 0/0、深部灰质 0/0；逐卷中位 0.418（P10–P90 0.298–0.631）。宿主一致率按 SynthSeg 宿主拆：白质 846/855、皮层 35/100（65 处 SynthSeg 说皮层、学生说白质）、基底节 0/1；没有左右对调。
  重跑命令（输出目录必须不存在，约 16 分钟）：
  ```
  PYTHONNOUSERSITE=1 PYTHONPATH=. nice -n 19 ~/anaconda3/envs/nvgen/bin/python scripts/eval_brain_anatomy.py --out <新目录> --work /data2/congcong/data/FM_data/derived/brain_anatomy/eval_<新戳> --gpu <空卡>
  stacks 433; host agreement 0.9215481171548117; mean host Dice 0.2664787808434745; outline Dice 0.9780942600738576; pass False
  ```
- **事后诊断（2026-10-08，`eval/README.md`、`eval/ref_vs_student.txt`、`eval/low_slices/`；只解释数字，不改门）**：可靠层里 SynthSeg 参照几乎没有深部结构——丘脑每卷均 0.06 mL（433 卷里 79 卷有），脑干/小脑/深部灰质只剩零星体素（这些卷覆盖头顶往下约 65–80 mm，本就不含它们），学生给丘脑 1.05 mL（233 卷）、脑室 15.2 mL 对参照 7.0 mL。13 类里 9 类因此 Dice≈0，门是等权平均。看图（USER_REPORTED）：最低两层（不判）影像上有侧脑室、丘脑、豆状核，学生标得形状和左右都对，参照那里没有脑室和丘脑、中心留空、皮层铺到后半；可靠层 2–5 上两图的白质/皮层形状一致，脑室窄时参照漏标。
- **仿真测试集（A13，只报告）**：13 类平均 0.6611（白质 0.850/0.847、皮层 0.789/0.791、丘脑 0.805/0.826、基底节 0.780/0.797、脑干 0.354、小脑 0.452/0.408、深部灰质 0.439/0.459、脑室 0.849）。
- **推理冒烟**（`infer_smoke.md`，GPU 7，各约 30 s）：S2 冒烟卷 `file_brain_AXFLAIR_201_6002917` 脑 728.1 mL、可靠层 [2, 12]；注册表第一个病灶（`file_brain_AXFLAIR_200_6002425` 病灶 0，框 91 182 2 98 188 3）学生图绑定 白质/重叠/右侧 0.714，同卷 SynthSeg 图 白质/重叠/右侧 0.667，注册表 `host_lookup` 41；两图在框内 42 个体素差 2 个。
- **用户本轮拍板**（沿用）：S4 概览点头；数据段按建议（三来源、fastMRI 只验证、按病人留 20%、粒度 A）；第一篇论文的数据范围维持 v2.6 §15（"approve 1"）；§3–5（自训轮廓模型、门 0.90/0.80/0.97、推理链）"yes"；"start" 开跑。

**之前（保留）**：S7 脑部多病种阶段 A 完成（胶质瘤 0.811 @ 0.35、转移瘤 0.747 @ 0.57、梗死 0.572 @ 1.56 FP/例，三病种五折判门都过；1212 份逐例记录交付版 `records_v2/`；两次终审 + 一次记录审阅通过；记录索引 `docs/verification/2026-09-29/brain_multidisease/README.md`）。nnDetection 第二臂 fold 0 规则 A 不过、已停（`docs/verification/2026-09-29/brain_nndet/README.md`）。S2 nnU-Net 检测器 D1 门不过（`docs/verification/2026-09-28/brain_detector/README.md`）；小病灶线经用户批准先停（`docs/verification/2026-09-30/small_lesion_line/README.md`）。S1 关系基线（NOT_EVIDENCE）、Level R 读片工具（冒烟服务 8791，浏览器验收仍 USER_REPORTED）同前。S4 前期探查：`docs/verification/2026-09-29/s4_probe/`、`2026-09-30/s4_sibbms_synthseg/`、`2026-09-30/s4_probe2/`、`2026-10-02/s4_probe3_frame/`。

## 2. 待用户拍板

1. **S4 的结论怎么用**（`eval/README.md` 末尾三条路）：(1) 按字面接受"不过"，回头改仿真（A6 定位 / A7 对比度）；(2) 深部类只在有参照的地方判，即 Level R 读片（A12 原本就定它是终审），我倾向这条——失败的数衡量的是参照缺结构，不是学生可见的缺陷；(3) 把 Dice 门限定到这些卷真正含有的类（白质+皮层四类均值 0.807，皮层本身没到 0.80）。(2)(3) 都是事后口径，要先写进规格增补才能引用任何"过"的数字。
2. **push**：本会话推送被权限分类器拒绝，main 与全部 tag 仍只在本地。请你自己执行：`git -C /data0/congcong/code/Project_Doing/foundation_model push origin main --tags`。
3. **可删清单（只列，不删；删除由你执行）**：
   - 第一次带镜像的学生训练：`/data2/congcong/data/FM_data/derived/nnunet/results/Dataset907_BrainAnatomyFLAIR/nnUNetTrainer_250epochs__nnUNetPlans__3d_fullres/`（约 1.4 GB）与 `logs/brain_anatomy/Dataset907_BrainAnatomyFLAIR_3d_fullres_nnUNetTrainer_250epochs_fold0.log`（证据已抄进 `launch.md`/`training.txt`，删了也不丢）；
   - 链路探针 `/data2/congcong/data/FM_data/derived/brain_anatomy/preflight_best_20261003_2249/`；
   - 评估中间预测 `/data2/congcong/data/FM_data/derived/brain_anatomy/eval_20261004_0233/`（507 MB；重跑 16 分钟）；
   - SDD 工作区 `.superpowers/sdd/2026-10-02-brain-anatomy-flair/`（gitignore；台账也在里面，想留就留）；误写到 `~/.claude/docs/` 的探针副本（`rm -r ~/.claude/docs`）；
   - 沿用：已合并的工作树与分支 `../foundation_model-detector`（`build/brain-detector`）、`../foundation_model-nndet`（`build/brain-nndet`）、`../foundation_model-multidisease`（`build/brain-multidisease`）；`/data2/congcong/data/FM_data/derived/nndet_smoke/`；`~/logs/nndet_install/wheels/`（1.6 GB）；`/data2/…/brain_disease/crossrun/*/{input,pred}`；会话 scratch。
4. **S7 记录第三次审阅的六条措辞小项**（都不挡交付；改的话出 `records_v3/`，一次做完）：a）10.17 mm 写成"约 10 mm"却归入"未能定位"——四舍五入等于 10 时写一位小数；b）"跨多个结构"括号里有脑干时前面的侧别会被读到脑干上——此时每个结构各带侧别、脑干在前；c）很小的病灶（93、139 mm³）也写"跨多个结构"——< 1 mL 仍写主结构；d）跨多个结构的病灶"还见于"只按占比最大的结构去重——括号里的结构都算已命名；e）左右各一个体素骑中线记为"右侧"——平局记 midline；f）两个括号相连——把"（未与任何结构重叠）"放到句末。建议全部采纳。
5. **读片人**、Level R 读片说明里 "other" 的定义、伦理备案：同前。S4 的输出现在能给读片人看的东西：每卷 `anatomy.nii.gz`（SynthSeg 值）与 `brain_mask.nii.gz`；读片时顺带看一眼第 14 层那块不像头皮的组织（§1 诊断）。
6. 旧遗留：Q9 删除授权、其余 4850 卷 SynthSeg、Redivis token、两条远端评审分支去留、RSS 左右手性换算规则。

## 3. 下一步

1. 用户对 §2 第 1 条拍板。走 (2)：把 S4 学生输出接进 Level R 读片的出图（学生图与 SynthSeg 图并排，读片人判深部结构与低层），规格增补写清"深部类以读片为准"；走 (1)：先解 `simulation/README.md` 的定位问题（真实卷可能比仿真低 5 mm），再考虑 A7 对比度，重新仿真、重训、重评估（约 5 h 机时），门不变；走 (3)：规格增补重定义门，评估脚本加一个"按存在类"的报告量，不重训。
2. S5：把 S4 的解剖图接进 S7 的记录格式（`anatobind/infer/brain_disease.py` 现在吃 SynthSeg 图，S4 输出同一取值空间，可直接替换 `--anatomy`），先在 fastMRI 卷上出几份对照记录看句子变化；阶段 B（只吃 FLAIR 的统一模型）仍等用户看过数后再设计。
3. S7 六条措辞（§2 第 4 条）拍板后改 → 测试 → 三病种重跑到 `_v3/` → 审阅读句子。不重训。
4. 医生标签（Level R）：训练都结束了，排读片。

## 4. 坑与别重做

- **分侧标签不能用 nnU-Net 默认训练器**（A17）：设计任何数据集先问"标签分不分侧"，分侧就写明 `*_NoMirroring`，开训前几轮看左右两类的伪 Dice 是否同步上升。
- **`brain_anatomy_train.py` 打印的 pid 是 `setsid` 外壳的**，它马上退出；要盯的是紧随其后的 `bash -c` 和 `nnUNetv2_train` 进程（`ps -u $USER -o pid,ppid,cmd | grep nnUNetv2_train`；`launch.md` 有表）。`Popen(start_new_session=True)` 之后 `setsid` 其实多余（终审小项）。
- **本会话不能 push、不能停别的进程**：`git push` 被判为对外发布；`kill` 自己启动的训练也被判为干扰工作负载。都不再尝试，交给用户。
- **SynthSeg 在 fastMRI 卷上不是真值**：最低两层和颅顶不可靠（A12），可靠层里也几乎没有深部结构；拿它当参照的任何 Dice 都只能说"两张伪标签像不像"。
- **仿真的真实对照只有伪标签统计**，定位可能差一层（`simulation/README.md`）；别拿这个去调参数，等读片。
- **数组坐标系**：fastMRI RSS 帧 = RAS 翻转轴 0（`s4_probe3_frame`）；仿真与推理都在这个帧里，左右跟文件头走。
- **ignore 标签 = 最高 id**（学生 15、轮廓 2）；映射回 SynthSeg 值时 ignore 不出现在输出里。
- **记录目录写了就不能重来**（不删任何东西）：所有脚本"先算后建目录"，重跑用新的 `--out`/`--work`。`sed '1,/re/d'` 在首行就匹配时会删光整个文件（本轮吃过一次亏，用 Write 重写）。
- **SibBMS 在模板空间、已去颅骨**，两个队列都从 sub-001 编号，用 `MS_`/`Norm_` 前缀。SibBMS 没有同网格病灶掩膜，病灶不设 ignore。
- **三个训练同时跑时 CPU 48 线程是上限**（每个 nnU-Net 训练 13 个进程）；评估等训练结束再跑。
- **训练日志滞后**，看结果夹的 `training_log_*.txt`；别的会话会把任务放到空卡上，不是自己启动的进程一律不动。
- **`pytest -q -q` 不打 "N passed" 行**；zsh 里 `echo ====` 会报 "=== not found"。
- 沿用：记录版本不覆盖（v1/v2/v3 各自目录）；别在 nnU-Net 上调参救线；绑定一致率不是证据；"疑似 X"不是鉴别；匹配规则与 S2 相同。

## 5. 关键决定的为什么

- **老师在 T1、学生在 FLAIR（A1–A3）**：SynthSeg 在 1 mm T1 上可靠，在 5 mm FLAIR 卷上不可靠；fastMRI 只有 FLAIR 卷，所以把老师标签经仿真搬到 FLAIR 上学。
- **离线仿真从颅顶往下定位（A6 修订）**：按面积份额定位在真实数据上给不出空的顶层；真实卷顶上有 1–5 个空层。
- **自训轮廓模型而不用 HD-BET（A10）**：fastMRI 卷在 SynthSeg 可靠层上就能自监督出轮廓，门 0.97 过了；HD-BET 留作备选。
- **16 类紧凑标签、推理映射回 SynthSeg 值（A4）**：下游 `BrainBinder`/记录格式不用改。
- **无镜像训练器（A17）**：标准补救；不在共享 conda 环境里放自定义训练器。
- **门不调、结论按字面报（A11）**：门是开跑前定的；事后的解释写在旁边、标明事后，由用户决定口径怎么改。
- **第一次训练不去硬停**：权限层拒绝后不绕；多占一张卡 3.5 小时，换来不碰别人进程这条线不破。
- **直接在 main 上做**：用户同意；协作者只读 main。
