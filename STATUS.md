# STATUS：2026-09-30（S7 脑部多病种检测阶段 A 收尾：三个病种五折判门都过、1212 份逐例记录已写；nnDetection 第二臂 fold 0 规则 A 不过、已停并合入 main；本轮合入 main 并打 tag `handoff/2026-09-30-brain-multidisease`，未 push）

每次交接前整体重写本文件。五段固定：已验证、待拍板、下一步、坑与别重做、为什么。

## 1. 已完成且已验证

**S7 脑部多病种检测，阶段 A（每个病种一个 nnU-Net 3d_fullres，各用原生序列）全部 13 个任务做完。** 规格 `docs/superpowers/specs/2026-09-29-brain-multidisease-design.md`（决定 M1–M14），计划 `docs/superpowers/plans/2026-09-29-brain-multidisease.md`。分支 `build/brain-multidisease`。十个代码任务由子代理逐任务实现并评审，两次全分支终审（最强模型）；台账 `.superpowers/sdd/2026-09-29-brain-multidisease/progress.md`（gitignore）。记录索引 **`docs/verification/2026-09-29/brain_multidisease/README.md`**（每个数字标了出处）。

- **测试**（合并后的 main）：
  ```
  PYTHONNOUSERSITE=1 PYTHONPATH=. ~/anaconda3/envs/nvgen/bin/python -m pytest tests/ -q -p no:cacheprovider
  835 passed, 1 skipped in 87.97s (0:01:27)
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
- **句子现在的写法**（`anatobind/infer/brain_disease.py`）：写体积最大的 5 处、从大到小；其余计数并补"前 5 处没提到的位置"；主结构前的侧别按落在主结构里的体素算，脑干不写侧别；"累及"后有不同侧才全部写侧别；不与任何结构重叠：≤10 mm 写"邻近<侧><结构>（未与任何结构重叠）"，>10 mm 写"未能定位的区域"；没检出写"本模型未检出<类型>（阈值 0.xx）。"、印象"未检出相关异常"。前两条是建议实现，规格里标"待用户确认"（§2）。
- **nnDetection 第二臂（S2 的补充）已收尾并在 main**（769abec，tag `handoff/2026-09-30-brain-nndet-fold0`）：fold 0 默认后处理 17/280 @ 0.73 FP/卷，同折 nnU-Net 2d 92/280，规则 A 要求 106，不过；按 D9 停，不补五折。终审无代码缺陷；真值检查加了"回来的实例里 IoU ≥ 0.99 的占比 ≥ 0.8"的判据。记录索引 `docs/verification/2026-09-29/brain_nndet/README.md`。
- **用户本轮拍板**（沿用）：S7 走 C（先 A 后 B）；达标线与 D1 同口径、三病种分别判；产出 = 检测器 + 查表绑定 + 结构化记录，脑叶留给 S4；五折全训、fold 0 先报（"B，不用留卡"）；只用空卡；交叉误报只报告；最近结构规则 10 mm；"按照你的建议"（2026-09-29 16:52）覆盖当时待定的建议项。

**之前（保留）**：S2 nnU-Net 检测器 D1 门不过（2d 五折 0.3662 @ 1.636 FP/卷），记录索引 `docs/verification/2026-09-28/brain_detector/README.md`。S1 关系基线（NOT_EVIDENCE）、Level R 读片工具（冒烟服务 8791，浏览器验收仍 USER_REPORTED）同前。S4 脑部解剖层的只读探查在 `docs/verification/2026-09-29/s4_probe/`（SynthSeg 在 fastMRI FLAIR 上最低层面与颅顶的脑轮廓不可靠，14/447 卷基本失败；SibBMS 有 100 健康 + 93 MS 的 T1/T2/FLAIR，1 mm、去颅骨、标准空间）。

## 2. 待用户拍板

1. **句子的两处写法**：按体积写前 5 处；"另有 N 处（还见于…）"。已按建议实现并写进 1212 份记录，规格里仍标"待确认"。不同意就改代码重写记录（记录目录不能删，得换新目录）。
2. **推理入口的 `study` 字段**现在是第一个图像文件名（`UCSF-PDGM-0008_T1.nii.gz`），评估记录用病例号。要不要加 `--study` 参数。
3. **终审 2 提出的四处措辞判断（都不是代码错，改了就要把记录写到新目录）**：a）主结构按体素多数投票，大肿瘤会写到占比不到 30% 的结构上（UCSF-PDGM-0483："双侧丘脑存在肿瘤样异常，体积约 116 mL，累及左侧大脑白质、左侧大脑皮层、左侧基底节"，丘脑只占 28.6%），建议占比低于某个值（如 40%）时改写为"累及多个结构"或把占比写进句子；b）两三个体素跨在左右分界上的小病灶被写成"双侧"（100196A 小脑 14 mm³、sub-strokecase0244 16 mm³ 等），建议小病灶不写"双侧"；c）"还见于邻近左侧大脑皮层"在前五处已写了左侧大脑皮层时没有新信息（sub-strokecase0107/0153/0193/0208/0248、100154D、100180A），去掉"邻近"类位置还是去前缀再比较；d）"未能定位的区域"（100126B 一处 3.5 mL 离最近结构 29 mm）可改写成"距左侧小脑约 29 mm"。
4. **小病灶检测下一步**：S2 与 nnDetection 都没过门。接受结论先放一放、转 S4/S5，还是换思路（要新规格）。不建议调参救门。
5. **S4 脑部解剖层设计概览**（老师 SynthSeg on 1 mm T1；学生看处理成 fastMRI 样子的 FLAIR；SibBMS 为核心数据；自训脑轮廓模型去颅骨）等点头后分段细化。
6. **push**：main 自 afe641e 起所有提交与 tag 都只在本地。
7. **读片人**、Level R 读片说明里 "other" 的定义、伦理备案：同前。
8. **可删清单（只列，不删；删除由用户执行）**：已合并的工作树与分支 `../foundation_model-detector`（`build/brain-detector`）、`../foundation_model-nndet`（`build/brain-nndet`）、`../foundation_model-multidisease`（`build/brain-multidisease`）；`/data2/congcong/data/FM_data/derived/nndet_smoke/`；`~/logs/nndet_install/wheels/`（1.6 GB）；会话 scratch；交叉误报的工作目录 `/data2/…/brain_disease/crossrun/*/{input,pred}`（报告已入库）。训练结果、记录、冒烟输出留着。
9. 旧遗留：Q9 删除授权、其余 4850 卷 SynthSeg、Redivis token、两条远端评审分支去留、RSS 左右手性换算规则。

**终审给用户的提醒**：达标数字取决于分数的分布，nnU-Net 的分数集中在 1 附近，"无工作点"与找到多少病灶无关；"疑似 X"只说明跑的是 X 模型；1212 例里只有 8 例没检出、数据里几乎没有正常脑，"未检出"不能当阴性；解剖词来自伪标签的体素计票，大肿瘤的主结构可能只是 54% 对 46%；左右取决于文件头。

## 3. 下一步

1. 用户对 §2 第 1–4 条拍板；第 1 条若改，改 `anatobind/infer/brain_disease.py` 后用新目录重写记录（`scripts/eval_brain_disease.py --records <新目录>`，不重训）。
2. **S4 脑部解剖层**：概览点头后分段细化（数据、仿真、去颅骨、验证与达标线、推理）→ 规格 → 计划。SibBMS 要先解压、再用 CPU 跑 SynthSeg（372 次，约 6 小时）。
3. **S5 疾病印象 + 整句**：S7 的记录已带整句；S5 剩下的是把 S2/S4 的输出接进同一记录格式，等 S4。
4. **阶段 B**（只吃 FLAIR 的统一模型）等用户看过阶段 A 的数后再设计。
5. 医生标签（Level R）在训练结束后，同前。

## 4. 坑与别重做

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
