# STATUS：2026-09-30 下午（用户批准全部建议后：记录 v2 已交付并经第三次审阅；小病灶线先停；S4 数据第一步已开跑；主线在 main，tag `handoff/2026-09-30-records-v2`，未 push——push 被本会话的权限分类器拒绝，请用户自己推）

每次交接前整体重写本文件。五段固定：已验证、待拍板、下一步、坑与别重做、为什么。

## 1. 已完成且已验证

**S7 脑部多病种检测，阶段 A（每个病种一个 nnU-Net 3d_fullres，各用原生序列）全部 13 个任务做完。** 规格 `docs/superpowers/specs/2026-09-29-brain-multidisease-design.md`（决定 M1–M14），计划 `docs/superpowers/plans/2026-09-29-brain-multidisease.md`。分支 `build/brain-multidisease`。十个代码任务由子代理逐任务实现并评审，两次全分支终审（最强模型）；台账 `.superpowers/sdd/2026-09-29-brain-multidisease/progress.md`（gitignore）。记录索引 **`docs/verification/2026-09-29/brain_multidisease/README.md`**（每个数字标了出处）。

- **测试**（main，记录 v2 之后）：
  ```
  PYTHONNOUSERSITE=1 PYTHONPATH=. ~/anaconda3/envs/nvgen/bin/python -m pytest tests/ -q -p no:cacheprovider
  839 passed, 1 skipped in 94.54s (0:01:34)
  bash -c 'source scripts/nndet_env.sh && python -m pytest tests/test_nndet_runner.py -q -p no:cacheprovider --noconftest'
  2 passed, 5 warnings
  ```
- **训练**：15 个 250 轮训练由 `scripts/gpu_queue.py` 在空卡上跑，2026-09-29 13:46:14 启动，2026-09-30 05:48:16 全部结束，退出码全为 0，85.8 GPU 时、墙钟 16.03 h（`logs/brain_disease/queue.log`，不入库；每折时长见 README 的 Timing 表）。早停规则（M4）一次都没触发。
- **五折判门（M2，与 S2 的 D1 同口径；代码 9a89ff1；`<病种>/REPORT.md`、`verdict.json`、`froc.csv`）**：

  | 病种 | 例数 | 计入病灶（忽略 <10 mm³） | 阈值 | 灵敏度 | 每例误报 | 结论 | <5 / 5–10 / ≥10 mm | Dice（只报告） |
  |---|---|---|---|---|---|---|---|---|
  | 胶质瘤 Dataset904 | 501 | 677（259） | 0.60 | 0.8109（549） | 0.349 | 过 | 0.063 / 0.139 / 0.936 | 0.928 |
  | 转移瘤 Dataset905 | 461 | 3809（531） | 0.65 | 0.7474（2847） | 0.575 | 过 | 0.572 / 0.909 / 0.971 | 0.807 |
  | 梗死 Dataset906 | 250 | 2111（238） | 0.50 | 0.5722（1208） | 1.556 | 过 | 0.336 / 0.745 / 0.881 | 0.785（247 例） |

  重跑命令（约 3 分钟一个，`--workers 8`；输出目录必须不存在）：
  ```
  PYTHONNOUSERSITE=1 PYTHONPATH=. nice -n 19 ~/anaconda3/envs/nvgen/bin/python scripts/eval_brain_disease.py --disease infarct --folds 0 1 2 3 4 --workers 8 --out <新目录> --records <新目录>
  Disease: infarct; folds [0, 1, 2, 3, 4]; kind gate
  Scans: 250; lesions counted: 2111; ignored: 238
  Sensitivity 0.5722 at threshold 0.5 with 1.556 FP per scan
  Pass: True; stop remaining folds: False; early stop undecided: False
  ```
  - 三个模型的误报预算都没用满（不设阈值时每例误报 0.39 / 0.58 / 1.56），所以这些灵敏度就是不设阈值的上限；短板全在 5 mm 以下。每例误报中位数/最大/超过 2 个的例数：0/7/10、0/8/26、1/15/54。转移瘤做没做过手术差别不大（0.750 对 0.741）。
  - 对照小病灶线（同口径）：S2 nnU-Net 2d 五折 0.366 @ 1.64（不过）；nnDetection fold 0 默认后处理 0.061 @ 0.73（规则 A 不过）。数据集不可比：fastMRI+ 的病灶多是单层小框，这三个数据集里 ≥10 mm 的病灶占大头。
- **交叉误报（M11，只报告，重叠计数不是灵敏度；`crossrun/<模型>_on_<数据>/`）**：梗死模型→胶质瘤 100 例：46 个检出、25 个落在瘤区、144 个瘤里 16 个被碰到；转移瘤模型→胶质瘤：167 / 153 / 95；胶质瘤模型→转移瘤 105 例：378 / 164 / 729 个里 222 个，每例都有检出。"疑似 X"只说明跑的是 X 模型。
- **1212 份逐例记录**（`/data2/congcong/data/FM_data/derived/brain_disease/<病种>/records/<病例>.json`，阈值同判门，`model_folds` 是留出该例的那一折）：每例病灶数最多 10 / 69 / 48；"还见于"出现在 1 / 127 / 62 份里，最多补 2 / 8 / 6 个位置；最长句 221 字；最近结构规则用了 6 / 9 / 20 次。绑定一致率（主结构 / 侧别）0.965 / 0.998、0.977 / 0.997、0.963 / 0.993，**NOT_EVIDENCE**（真值和预测查同一张 SynthSeg 伪标签）。
- **推理冒烟**（`infer_smoke.md`）：三个病种各一例 fold 0 验证例、fold 0 模型：1 / 11 / 26 处病灶；与折外记录的框、分数在 1e-4 内一致（胶质瘤例分数差 1e-4、体积差 79 体素，另两例逐体素相同；`checks/infer_smoke_consistency.txt`）。
- **两次全分支终审（最强模型）**：第一次在 a22da8b（出数前，六次往返；报告 `.superpowers/sdd/2026-09-29-brain-multidisease/final-review-1-report.md`），要求的修补都已进分支（早停按"刚超预算那一行"判、队列失败计数、句子写法、体积下限容差、概率检查、记录出处字段、`code_version`）。第二次在 1e2e95f（出数后，含 1212 份记录逐条核对）：无 Critical、无 Important，全部 Minor。审阅者用自己写的代码在 1212 例折外预测上逐例重算：三条 FROC 曲线的 19 行命中数与每例误报（规格字面的一对一匹配与分支的匈牙利变体都一致）、忽略数 238/259/531、分层、误报分布、Dice、15 段训练时长、三组交叉误报逐例（0 处不同）、1212 份记录的折号/阈值/病灶数/框/分数/体积与每一句话（按 §6 规则重建，0 处违反）；831 测试通过；26 个变异里 22 个被现有测试抓住，剩下 3 个已补测试（d25d4fc）。小项："还见于邻近X"在 X 已写过时仍会补（7 条记录，措辞判断见 §2）、README 一句措辞、`logs/` 未 ignore（已加）、第一次终审第 9 小项的两行报告量没加（审阅者算的分数 ≥ 0.95 的检出占比 0.626 / 0.726 / 0.795，非记录）。两份终审报告在 `docs/verification/2026-09-29/brain_multidisease/reviews/`。
- **记录 v2（2026-09-30 下午，用户批准）**：确认了按体积写前 5 处与"还见于"；新加四条写法：主结构占比 < 40% 写"跨<侧>多个结构（…）"、"双侧"要求少数一侧 ≥ 10 mm³、"还见于"去重忽略"邻近"、超过 10 mm 写"未能定位的区域（距最近的…约 N mm）"；推理入口加 `--study`。代码 c8b1a11，测试 839 passed；三个病种重跑到 `docs/verification/2026-09-29/brain_multidisease/<病种>_v2/` 与 `/data2/…/brain_disease/<病种>/records_v2/`（**交付用 records_v2**，v1 目录保留）。与 v1 比：判门与 FROC 逐字节相同，1212 份记录的病灶、框、分数、主结构全部相同，26 句变化，5 个病灶的侧别字段变化（3 个 14–18 mm³ 的小脑小病灶不再写"双侧"，2 个只在未写出的 host_sides 里）。第三次审阅（最强模型，`reviews/final-review-3-records-v2.md`）：用自己的实现重建 1212 句 0 处不符，26 处变化都合规，"可以替代 v1 交付"；提出 6 条措辞小项（§2 第 3 条）。审阅指出的一处潜在套括号（距最近结构的括号里用长名）已改（无记录受影响）。
- **句子现在的写法**（`anatobind/infer/brain_disease.py`）：写体积最大的 5 处、从大到小；其余计数并补"前 5 处没提到的位置"；主结构前的侧别按落在主结构里的体素算，脑干不写侧别；"累及"后有不同侧才全部写侧别；不与任何结构重叠：≤10 mm 写"邻近<侧><结构>（未与任何结构重叠）"，>10 mm 写"未能定位的区域"；没检出写"本模型未检出<类型>（阈值 0.xx）。"、印象"未检出相关异常"。前两条是建议实现，规格里标"待用户确认"（§2）。
- **nnDetection 第二臂（S2 的补充）已收尾并在 main**（769abec，tag `handoff/2026-09-30-brain-nndet-fold0`）：fold 0 默认后处理 17/280 @ 0.73 FP/卷，同折 nnU-Net 2d 92/280，规则 A 要求 106，不过；按 D9 停，不补五折。终审无代码缺陷；真值检查加了"回来的实例里 IoU ≥ 0.99 的占比 ≥ 0.8"的判据。记录索引 `docs/verification/2026-09-29/brain_nndet/README.md`。
- **用户本轮拍板**（沿用）：S7 走 C（先 A 后 B）；达标线与 D1 同口径、三病种分别判；产出 = 检测器 + 查表绑定 + 结构化记录，脑叶留给 S4；五折全训、fold 0 先报（"B，不用留卡"）；只用空卡；交叉误报只报告；最近结构规则 10 mm；"按照你的建议"（2026-09-29 16:52）覆盖当时待定的建议项。

**之前（保留）**：S2 nnU-Net 检测器 D1 门不过（2d 五折 0.3662 @ 1.636 FP/卷），记录索引 `docs/verification/2026-09-28/brain_detector/README.md`。S1 关系基线（NOT_EVIDENCE）、Level R 读片工具（冒烟服务 8791，浏览器验收仍 USER_REPORTED）同前。S4 脑部解剖层的只读探查在 `docs/verification/2026-09-29/s4_probe/`（SynthSeg 在 fastMRI FLAIR 上最低层面与颅顶的脑轮廓不可靠，14/447 卷基本失败；SibBMS 有 100 健康 + 93 MS 的 T1/T2/FLAIR，1 mm、去颅骨、标准空间）。

## 2. 待用户拍板

1. **push**：本会话推送被权限分类器拒绝（"Out-of-Place Publication"），main 与全部 tag 仍只在本地。请你自己执行：`git -C /data0/congcong/code/Project_Doing/foundation_model push origin main --tags`。
2. **可删清单（只列，不删；删除由你执行）**：已合并的工作树与分支 `../foundation_model-detector`（`build/brain-detector`）、`../foundation_model-nndet`（`build/brain-nndet`）、`../foundation_model-multidisease`（`build/brain-multidisease`）；`/data2/congcong/data/FM_data/derived/nndet_smoke/`；`~/logs/nndet_install/wheels/`（1.6 GB）；交叉误报的中间文件 `/data2/…/brain_disease/crossrun/*/{input,pred}`；会话 scratch。训练结果、记录（v1 与 v2）、报告都留着。
3. **第三次审阅提出的六条措辞小项**（都不挡交付；改的话记录写到 `records_v3/`，一次做完）：a）10.17 mm 写成"约 10 mm"却归入"未能定位"（100197A）——建议四舍五入等于 10 时写一位小数；b）"跨多个结构"括号里有脑干时前面的"左侧"会被读到脑干上（100300D、sub-strokecase0131）——建议此时每个结构各带侧别、脑干在前，与"累及"同规则；c）很小的病灶也写"跨多个结构"（93、139 mm³）——建议 < 1 mL 仍写主结构；d）跨多个结构的病灶"还见于"只按占比最大的结构去重（sub-strokecase0168）——建议括号里的结构都算已命名；e）左右各一个体素正好骑中线的病灶记为"右侧"（平局取右）——建议平局记 midline；f）sub-strokecase0201 两个括号相连（v1 就有）——可把"（未与任何结构重叠）"放到句末。我的建议：a–e 都采纳，f 采纳，一起出 v3。
4. **S4 分段设计**：概览已点头，第一段（数据）见本次汇报，逐段确认后写规格与计划。
5. **读片人**、Level R 读片说明里 "other" 的定义、伦理备案：同前。
6. 旧遗留：Q9 删除授权、其余 4850 卷 SynthSeg、Redivis token、两条远端评审分支去留、RSS 左右手性换算规则。

**终审给用户的提醒**：达标数字取决于分数的分布，nnU-Net 的分数集中在 1 附近，"无工作点"与找到多少病灶无关；"疑似 X"只说明跑的是 X 模型；1212 例里只有 8 例没检出、数据里几乎没有正常脑，"未检出"不能当阴性；解剖词来自伪标签的体素计票，大肿瘤的主结构可能只是 54% 对 46%；左右取决于文件头。

## 3. 下一步

1. 用户对 §2 第 3 条（六条措辞）拍板 → 改 `anatobind/infer/brain_disease.py`（及 `side_of` 的平局）→ 测试 → 三个病种重跑到 `<病种>_v3/` 与 `records_v3/`（各约 3 分钟，`--workers 8`）→ 审阅者读句子 → README 记一笔。不重训。
2. **S4 脑部解剖层（用户已点头概览）**：数据第一步已在跑——SibBMS 解压到 `/data2/congcong/data/FM_data/SibBMS_ms/sibbms/Output/{MS,Norm,Annotation}`（1425 个 NIfTI，11 GB；非增强 T1w 371 个：MS 271、健康 100），经 `derived/synthseg/sibbms/inputs/` 的 `MS_`/`Norm_` 前缀软链（两个队列都从 sub-001 编号，直接用会撞名）跑 SynthSeg 已完成（`scripts/run_synthseg_fastmri_brain.py --glob … --workers 3 --threads 12`，CPU，93 分钟，370/371 ok；记录 `docs/verification/2026-09-30/s4_sibbms_synthseg/`）：可用老师标签 362 张（MS 265 次检查 / 91 人 + 健康 97），排除 8 张失败图（输入几乎全零或 SynthSeg 找不到脑）和 1 个二维文件（MS sub-057 ses-001 的 T1w 是 256×256 的 2D 图）。接下来按段确认设计：数据 → 仿真（把 1 mm FLAIR 处理成 fastMRI 的样子）→ 去颅骨（自训脑轮廓模型，备选 HD-BET）→ 验证与达标线 → 推理，然后写规格与计划。
3. **小病灶线先停**（用户批准；收尾说明 `docs/verification/2026-09-30/small_lesion_line/README.md`）。再捡起来要新规格：标签处理（819 个未填的框）、按原生层厚分开训、或等 Level R 医生标签。
4. S5 疾病印象 + 整句：S7 的记录已带整句；剩下的是把 S2/S4 的输出接进同一记录格式，等 S4。阶段 B（只吃 FLAIR 的统一模型）等用户看过阶段 A 的数后再设计。
5. 医生标签（Level R）在训练结束后，同前。

## 4. 坑与别重做

- **本会话不能 push**：auto-mode 的权限分类器把 `git push` 判为对外发布并拒绝，之后同类动作不再尝试；push 由用户自己执行。
- **记录版本不覆盖**：v1 → v2 → v3 各自新目录（评估脚本拒绝已存在的 `--out`/`--records`），README 写明哪一版是交付。改句子规则不改判门数字，但绑定一致率会随侧别规则微变。
- **SibBMS 的 MS 与 Norm 两个队列都从 sub-001 编号**，stem 会撞名；用队列前缀的软链再喂 SynthSeg。SibBMS 已去颅骨、在标准空间（197×233×189，1 mm，RAS）。
- **两条分支同时改 CLAUDE.md/STATUS.md 必冲突**：本轮 nnDetection 先合 main，S7 再把 main 合进分支手工合并后合回。以后并行分支只让一条改这两份文件，或者交接时按顺序合。
- **记录目录写了就不能重来**（不删任何东西）：所有脚本都是"先算后建目录"，重跑用新的 `--out`/`--records`。
- **`pytest -q -q` 不打 "N passed" 行**，数测试用单个 `-q`。
- **`nnUNetv2_predict` 与训练时的验证预测有微小差别**（胶质瘤例分数差 1e-4、172755 体素差 79 个），框不变；核对时用 1e-4 容差。
- **训练日志滞后**（块缓冲，落后十几分钟），看结果目录的 `training_log_*.txt`；**CPU 上限 48**，训练期间评估用 `--workers 2`。
- **别的会话会把任务放到我们占着的卡上**，不是自己启动的进程一律不动、报用户；本轮胶质瘤各折时长 4.95–11.08 h 就是这么来的。
- **队列重启不要覆盖日志**，新启动用 `queue_$(date +%Y%m%d_%H%M%S).log`；新代码的队列：启动 10 分钟内失败就不再启动新任务，最后一行给计数，非全成功返回 1。
- **句子的每一处写法都要拿真实输出读一遍**（`checks/sentence_preview.py`）：单元测试守得住规则，守不住"读起来对不对"。
- **匹配规则与 S2 相同**（总 IoU 最大的一对一指派再去掉 <0.1）：拥挤时偶尔比最优匹配少一个命中；为可比不改，README 已写明。
- **分数的性质**：块内前景概率均值，都大于 0.5，阈值 ≤0.50 的 FROC 行相同，小块分数最高，大的误报任何阈值去不掉。
- **绑定一致率不是证据**；**"疑似 X" 不是鉴别**；**交叉误报是重叠计数**。
- nnDetection 的坑（两环境、约定、`EXACT_SHARE` 依赖数据集）见 main 上 nnDetection 收尾那版 STATUS（tag `handoff/2026-09-30-brain-nndet-fold0`）与 `docs/nndet_install.md`。
- 别在 nnU-Net 上调参救线（M5、D9）；C1 上的数字都不是证据。

## 5. 关键决定的为什么

- **先 A 后 B（M1）**：病种与数据集完全混杂，统一模型可以只靠图像风格分辨病种。
- **按 10 mm³ 定下限（M7）**：三个数据集体素大小差很多。
- **五折全训、fold 0 先报（M4）**：用户说不用留卡；五折折外预测让每个病例都有记录。早停只在读数明确时触发（工作点 <0.3 且刚超预算那行加两个病灶也到不了 0.3），本轮一次没触发。
- **句子按体积排序**：读的人先找最大的病灶，而分数最高的是最小的块。
- **终审提前到出数之前，出数后再审一次**：出数前的审查防止重算，出数后的审查核对写出去的 1212 份记录。
- **nnDetection 停在 fold 0**：规则 A 是用户开跑前拍的板；fold 0 的配对表（只 nnDetection 多找回 6 个、却漏掉 2d 找到的 81 个）说明问题在检出不在阈值。
- **两分支都合回 main**：协作者只读 main，最新状态不能停在侧分支。
