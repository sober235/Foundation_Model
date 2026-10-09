# STATUS：2026-10-09 下午（AnatoBind 脑部 A/U/R 第一部分完成：按 09-04 方案的 3D Swin 架构实现的代码包、样本表、训练目标、探针与时间表都已在 main；第二部分（训练/评估/推理）待写计划；主线在 main，tag `handoff/2026-10-09-anatobind-brain-aur-part1`；GitHub 上次推到 94c3b21，之后的提交请用户自己推）

每次交接前整体重写本文件。五段固定：已验证、待拍板、下一步、坑与别重做、为什么。

## 1. 已完成且已验证

**AnatoBind 脑部 A/U/R，第一部分（数据、目标、模型、损失、探针）做完。** 用户 10-08 定：按 09-04 方案 `AnatoBind-MRI_cui.md` 的架构实现三个子目标，先做脑；A 不变、U = 找出所有异常（不分病种）、R = 每个异常一个分侧主宿主；不做疾病头 D、不做 E 与 k 空间干预；输入单序列；A 用 SynthSeg 32 结构、R 用 13 类宿主；跳过 Stage I 自监督；4 张空闲 A800。规格 `docs/superpowers/specs/2026-10-08-anatobind-brain-aur-design.md`（N1–N18，§6/N16 已按实测增补），计划 `docs/superpowers/plans/2026-10-08-anatobind-brain-aur-part1.md`（14 任务），记录索引 **`docs/verification/2026-10-08/anatobind_brain_aur/README.md`**。做法与 S4 相同：全部代码先在草稿区验证，子代理逐批抄写，控制器逐字节比对，每批独立评审，终审用最强模型；台账 `.superpowers/sdd/2026-10-08-anatobind-brain-aur-part1/progress.md`（gitignore）。

- **测试**（main，2026-10-09）：
  ```
  PYTHONNOUSERSITE=1 PYTHONPATH=. nice -n 19 ~/anaconda3/envs/nvgen/bin/python -m pytest tests/ -q -p no:cacheprovider
  910 passed, 1 skipped in 87.78s (0:01:27)
  ```
  新增 38 个（`tests/test_aur_*.py`，12 个文件；终审修补后，首次文档提交时为 909）。
- **代码**（`anatobind/aur/`，12 个模块；`scripts/aur_prepare.py`、`scripts/aur_probe.py`）：标签空间（32 实体 / 13+1 宿主 / 6 序列）、样本表、实例与软宿主真值、zyx 裁块与坐标、物理坐标 RoPE、可变尺寸 Swin（RoPE 注意力、无效 token 不作键且归零）、实体/事件/序列/掩膜头、每病灶候选竞争、点采样损失、`AnatoBindBrain`、数据集。模型默认配置 = 规格 §5（embed 64、[2,2,6,2]、[2,4,8,16]、窗口 (4,8,8)、patch (2,4,4)、d_model 256、K 32、M 64），22 719 561 参数（骨干 12.9 M）。
- **终审修补（47355a7）**：所有卷读入时先 `nib.as_closest_canonical` 到 RAS（四个来源存储方向不同：PDGM LPS、ISLES LAS、SibBMS RAS、BMSR RAS+LAS，不统一等于隐式镜像；带 LPS 副本测试）、裁块字典带 `entity_present`、几处整理；记录补 `p0/spacing.txt`、`p0/model_params.txt`、`reviews/final-review-1.md`。
- **评审修补（两轮，全部已进代码、计划与草稿副本）**：关系几何的侧别改为宿主身份 + 左右宿主质量中点；骨干的 -1 填充约定写明；掩膜损失采样点一半按实例配额；缺席实体也学空掩膜；`bind` 按存在性门控；裁块里低于 10 mm³ 的碎片不算实例；探针测试去掉空断言。
- **P0 核查（`p0/`）**：
  ```
  scripts/aur_prepare.py --stage samples --out /data2/congcong/data/FM_data/derived/aur/samples.json
  pdgm: train 400 cases / 1600 rows / 800 with U; test 101 cases / 404 rows / 202 with U
  bmsr: train 359 cases / 1077 rows / 359 with U; test 102 cases / 306 rows / 102 with U
  isles: train 200 cases / 400 rows / 200 with U; test 50 cases / 100 rows / 50 with U
  sibbms: train 276 cases / 827 rows / 0 with U; test 82 cases / 246 rows / 0 with U
  wrote …/samples.json: 4960 rows
  scripts/aur_prepare.py --stage grids …   -> 4960 rows checked, 0 off their grid
  ```
  ISLES 的 SynthSeg(DWI) 各宿主中位体积为 PDGM 的 0.76–0.92，蒙太奇合理（USER_REPORTED）→ 裁定保留 A 监督（`p0/isles_ruling.md`）。SibBMS 斑块标注只有 10 人且在原生网格 → 所有 SibBMS 行不监督 U（`p0/sibbms_note.md`）。
- **探针（`p0/probe_single.txt`、`p0/probe_ddp.txt`；裁块 128×160×160，bf16，完整 A+S+U+R 损失）**：单卡 batch 1/2/4 带检查点 0.54/0.67/0.97 s、3.3/6.0/11.6 GiB；batch 2/4 不带 0.53/0.86 s、9.4/18.4 GiB；四卡 DDP batch 4 不带 0.90 s/步、18.5 GiB。四卡默认 NCCL 路径两次卡死（600 s 超时），`NCCL_P2P_DISABLE=1` 后正常。时间表（规格 §6）：4 卡 × batch 4，Stage II 15 k 步 ≈ 3.8 h，Stage III 5 k 步 ≈ 1.3 h，lr 5e-4、预热 1 k 步。
- **用户本轮拍板**（10-08）：方向 3（多部位）选"借现成解剖模型、新意在绑定"；按方案架构实现三子目标（方案 2，收口到 A/U/R）；U 找出所有异常；先做脑；三个默认（单序列、A 32/R 13、跳过自监督）；4 卡任取空闲；"approve push"（已推到 94c3b21）；两份原始方案入库；执行方式子代理驱动。

**之前（保留）**：S4 脑解剖学生模型（fastMRI FLAIR）训完评估完，A11 按规则"不过"（分侧宿主 Dice 0.27，根因是 SynthSeg 参照在可靠层几乎没有深部结构；一致率 0.92、轮廓 0.98 过），三条出路待定（`docs/verification/2026-10-02/brain_anatomy_flair/eval/README.md`）。S7 三病种阶段 A 五折全过、1212 份记录 v2（`docs/verification/2026-09-29/brain_multidisease/README.md`）。nnDetection 与小病灶线已停。Level R 读片工具已做好，读片未开始。

## 2. 待用户拍板

1. **裁块的物理尺度怎么定（终审 I1，写 Part 2 计划前要定）**：`p0/spacing.txt` 实测：PDGM、SibBMS 1 mm；ISLES 196 例层厚 2 mm（194 例各向同性）、54 例层厚 4.8 mm；BMSR 面内 0.43–1.17 mm、层厚 1–5 mm（中位 0.859×0.859×1.5）。固定 128×160×160 体素的裁块在各来源的物理范围是面内 69–188 mm、z 向 128–640 mm。三条路：(a) 维持体素裁块、原生间距（方案 §3 的字面做法，RoPE 用 mm，模型自己跨尺度泛化）；(b) 离线把 BMSR 与 ISLES 重采样到统一间距（如 1×1×1 mm 或 1×1×2 mm，方案 §3 允许“极端 spacing 温和重采样”），裁块变成固定毫米范围；(c) 按来源定裁块大小。我倾向 (b)：解剖标签本来就是 1 mm 的 SynthSeg，训练与推理都在同一尺度，RoPE 的频率设计也按 1 mm 调的。
2. **ISLES 保留 A 监督的裁定是临时的**：六张蒙太奇我只看了两张，你没看过（`p0/isles_check/`）。
3. **第二部分开工**：写 Part 2 计划（训练 DDP、评估三个门、推理、记录），按规格 §6–§9；训练预计一天内完成。第 1 条定了就写。
4. **push**：`git -C /data0/congcong/code/Project_Doing/foundation_model push origin main --tags`（本会话的推送只在用户明确批准时做过一次）。
5. **可删清单（只列，不删）**：实现者留在家目录的垃圾 `/home/congcongliu/aurfix.qF3B/`、`/home/congcongliu/.aurfix_dir_tmp`；草稿区 `/tmp/claude-1002/-home-congcongliu--claude/614ccd9c-f8e8-435b-afc8-47ae37fe3721/scratchpad/{aur_dryrun,s4_dryrun}`（Part 2 写计划时还会用 aur_dryrun，之后再删）；两个 SDD 工作区 `.superpowers/sdd/2026-10-0{2,8}-*`（台账在里面）；S4 的 `derived/brain_anatomy/{preflight_best_20261003_2249,eval_20261004_0233}`、第一次带镜像的学生训练结果夹与日志；沿用：三个已合并工作树与分支、`nndet_smoke/`、`~/logs/nndet_install/wheels/`、`crossrun/*/{input,pred}`、`~/.claude/docs`。
6. S4 结论的三条出路、S7 六条措辞小项、读片人与伦理备案、旧遗留：同前。

## 3. 下一步

1. Part 2 计划：`anatobind/aur/train.py`（DDP、两阶段、按裁块预算、验证集 10% 病人、checkpoint 选择；`sample_points(..., instance=...)` 由训练器传入；`entity_present` 来自裁块字典；`NCCL_P2P_DISABLE=1`；`use_checkpoint=False`；**DDP 未用参数（终审 I2）**：Stage II 冻结 `model.relation`，某卡整批无 U 监督时给事件输出加零权重触碰，或 `find_unused_parameters=True` 并重测速度）、`eval.py`（整卷滑窗、A 13 类 Dice、U 逐病灶灵敏度 @ nnU-Net 工作点、R 受控轨道 ABA vs B0 几何查表、方案实验 1 的 gap 与 rescue/harm、fastMRI 外部一致率）、`infer.py` + `scripts/infer_anatobind_brain.py`（`anatomy.nii.gz`、`lesions.nii.gz`、`record.json`、简化句子）、记录与门。门不过就停、报告、交用户。
2. 第二部位膝（SKM-TEA）另开规格；专家演示 URL（DICOM 进）待定数据来源与时间。
3. 规格 §12 还没做的两项：P4（SibBMS 10 例标注子集只核了 1 例网格）、P5（U 对照用的 S7 折外预测位置与匹配规则对账）——放进 Part 2 计划的第一个任务。
4. 推理（Part 2）读入也要 `nib.as_closest_canonical`，和训练一致；输出写回原方向。
5. Level R 读片：同前。

## 4. 坑与别重做

- **四个来源的存储方向不同**（PDGM LPS、ISLES LAS、SibBMS RAS、BMSR RAS 452 + LAS 9，`p0/spacing.txt`）：不统一到 RAS 就是来源之间的隐式镜像，分侧标签学不出来；`dataset.load_volume` 现在先 `nib.as_closest_canonical`，推理也要。
- **本机四卡 NCCL 默认 P2P 路径会在初始化处卡死**（两次复现），`NCCL_P2P_DISABLE=1` 后正常；任何多卡命令都加 `timeout`。
- **数组顺序**：模型吃 (z, y, x)，nibabel 读出来是 (x, y, z)，`crops.to_zyx` 转；裁块不 resize，窗外填 -1 并标无效；RoPE 用 mm 坐标，相对位置不变，裁块内旋转不转坐标网格。
- **无效 token 不作注意力的键、进 stage 前和合并前归零**；部分有效的 patch 是有效 token，看到的是 -1 常量。
- **宿主侧别用身份，不用图像里的中线**：单侧裁块会让质心中线失效。
- **采样点**：均匀采样会让小病灶一个正点都没有，`sample_points` 要传实例图（训练器负责）。
- **钩子会拦下命令文本里含删除/改名调用字样的 heredoc**（即使只是散文），这类文件用 Write 工具写。
- **zsh 不对 `$var` 分词**：循环里传多个参数用 `${=var}`。
- **实现者会在家目录留垃圾、会用 `cat >` 覆盖已有报告**：派发时写明 scratch 只能放在 scratchpad、追加用 `>>`；本轮 task-12-report.md 被覆盖一次，已按交接内容补注。
- **不并行派两个实现者**；评审可以和实现者并行。
- **SibBMS 两个队列都从 sub-001 编号**（用 MS_/Norm_ 前缀）；SibBMS 没有同网格病灶标注。
- 沿用：本会话不能停别人的进程（自己启动的也可能被拒）；记录目录写了就不能重来；`pytest -q -q` 不打总数行；分侧标签不能用 nnU-Net 默认镜像（S4 A17）。

## 5. 关键决定的为什么

- **方案架构而不是 nnU-Net（N5–N8）**：用户要求用自己方案的模型；nnU-Net 链降为对照组 B0。
- **单序列输入 + S token（N3）**：任何单个序列都能进模型，对应"general input"；代价是不做序列融合。
- **A 32 实体、R 13 宿主 + 地标（N2）**：解剖"尽可能多"，宿主按 V7 的两级 ontology 避免"宿主 = 脑室"。
- **R 真值 = mask 重叠比（N9）**：方案 §13 的做法，不需要人工关系标注；临床对错仍靠 Level R。
- **跳过 Stage I（N11）**：用户的默认；省几周算力。
- **裁块不 resize（N4）**：方案 §3；物理坐标保留。
- **每病灶候选竞争而不是全局 K×M（N8）**：V7 §8；全局版作消融。
- **实例配额采样、缺席实体 BCE、存在性门控、碎片规则**：评审发现的训练质量风险，各自一行代码级改动，都有测试。
- **4 卡 × batch 4 不用检查点**：实测显存只用 18.5 GB，检查点白花 15% 时间。
- **ISLES 保留 A 监督**：体积在 PDGM 的 0.76–0.92、蒙太奇合理；2 mm 标签粗但不错位。
- **直接在 main 上做**：用户同意；协作者只读 main。
