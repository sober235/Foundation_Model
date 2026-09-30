# STATUS：2026-09-30（nnDetection 第二臂收尾：fold 0 训完，规则 A 不过，臂到此停止；全分支终审已过并修补；合入 main 并打 tag `handoff/2026-09-30-brain-nndet-fold0`，未 push）

每次交接前整体重写本文件。五段固定：已验证、待拍板、下一步、坑与别重做、为什么。
同日并行的 S7 脑多病种检测器（分支 `build/brain-multidisease`，工作树 `../foundation_model-multidisease`）有自己的 STATUS；两条线合回 main 时以合并后的 STATUS 为准。

## 1. 已完成且已验证

**本轮：S2 的第二臂 nnDetection（Retina U-Net）全部 10 个任务做完。** 规格 `docs/superpowers/specs/2026-09-28-brain-nndet-design.md`（决定 N1–N14），计划 `docs/superpowers/plans/2026-09-28-brain-nndet.md`。分支 `build/brain-nndet`。任务 1–8 由子代理逐任务实现并评审（台账 `.superpowers/sdd/2026-09-28-brain-nndet/progress.md`，gitignore），任务 9–10 由控制方执行。记录索引 `docs/verification/2026-09-29/brain_nndet/README.md`（每个文件是什么、哪条命令产生、已知偏差、实测耗时）。

- **测试**（终审修补后的代码）：
  ```
  PYTHONNOUSERSITE=1 PYTHONPATH=. ~/anaconda3/envs/nvgen/bin/python -m pytest tests/ -q -p no:cacheprovider
  749 passed, 1 skipped in 81.80s
  bash -c 'source scripts/nndet_env.sh && python -m pytest tests/test_nndet_runner.py -q -p no:cacheprovider --noconftest'
  2 passed, 5 warnings
  ```
- **fold 0 训练**（`launch.md`、`training.txt`、README 的 Timing 表）：2026-09-29 04:24:50 在 GPU 3 启动，60 epoch 到 19:39:01（`model_last.ckpt`），51 例验证集默认参数预测到 19:55:59，sweep 与分析到 20:00:12，共 15 h 35 min（预估 13 h；下午起别的会话的任务与它同卡）。进程自己退出，只在启动器日志尾部打了 batchgenerators 的 teardown 错误（与玩具训练一样，不是失败），没有 kill 任何进程。
- **提取**（`scripts/nndet_runner.py extract`，nndet 环境，CPU）：默认后处理 25881 个框、sweep 调参版 1696 个框，各 51 例；写在 `/data2/congcong/data/FM_data/derived/nndet_runs/fold0_{default,swept}.json`。
- **规则 A 不过**（`fold0/rule_a.json`、`fold0/REPORT.md`）：
  ```
  PYTHONNOUSERSITE=1 PYTHONPATH=. ~/anaconda3/envs/nvgen/bin/python scripts/eval_brain_nndet.py \
    --dets /data2/congcong/data/FM_data/derived/nndet_runs/fold0_default.json \
    --swept /data2/congcong/data/FM_data/derived/nndet_runs/fold0_swept.json --folds 0 --out docs/verification/2026-09-29/brain_nndet/fold0
  Folds: [0]; scans 51; lesions 280
  Rule A pass: False
  nnDetection operating threshold 0.50: hits 17, sensitivity 0.0607, FP per volume 0.7255
  nnU-Net 2d operating threshold 0.60: hits 92, sensitivity 0.3286, FP per volume 1.5490
  ```
  要求 ≥ 106（92 + 0.05 × 280）。nnU-Net 2d 的同折数字与探针完全一致。FROC：阈值 0.05 找到 178/280（0.636）但每卷 29.9 个假阳，0.35 处 92 个命中要付每卷 5.0 个假阳；每卷 ≤ 2 的预算里最好的一行就是 0.50 的 17 个。配对表（各自工作点）：两者都中 11、只 nnDetection 6、只 nnU-Net 2d 81、都漏 182。分层：单层病灶 7/222、多层 10/58；面内最大三分位 11/91。sweep 调参版（参数在这 51 例上调的，NOT_GATE）：0.2429 @ 每卷 1.8431，阈值 0.85。
- **推理冒烟**（`infer_smoke.md`）：S2 冒烟同一卷 `file_brain_AXFLAIR_201_6002917.h5`，fold 0 模型、默认后处理、GPU 7 空卡：156 行，11 行 ≥ 0.50，每行恰好 `boxes, score, z0, z1` 四个键，退出码 0（该卷无真值，行数不是证据）。
- **已知偏差**（README）：病灶 815 的框被覆盖改变（唯一一个）；8 个病灶在重采样到 5 mm 后消失（1001、1004、1011、1042、1061、1063、1080、1104；7 个有病灶的 3 mm 层厚卷共 43 个病灶、其中单层 20 个，丢的就是这 8 个单层的），训练标签里没有它们、真值检查里也不回来，但预测仍可能碰到（fold 0 的 1042、1104 在阈值 0.05 下被碰到，0.50 下没有），永远算分母；61 个卷里 819 个未填进标签的 fastMRI+ 框在训练里是背景（S2 D10）；阈值与 sweep 参数都在验证例上选的。
- **全分支终审（最强模型，2026-09-30）**：无代码缺陷；审阅者用自己的代码（自写 IoU、匈牙利匹配、nnU-Net 基线的连通块解码）复现了规则 A 的全部数字（17/280、92/280、配对表 11/6/81/182、各分层、真值检查 1289/8/0.3156/1151）；坐标链逐段核对：1175 个在目标间距卷里的病灶中 1133 个回来 IoU 正好 1.0，其余 42 个都有解释（病灶 815 的覆盖、列间距 0.6858–0.6898 mm 被 nnDetection 按 0.6875 缩放）。要改的两条已改：README 里“八个丢失病灶永远不可能被命中”是错话（fold 0 的 1042、1104 在阈值 0.05 下已被预测框碰到），改成事实；真值检查的判据挡不住 N8 说的 ±1 外扩错误（外扩不撤销 IoU 0.32、撤到错误一端 0.148 都 ≥ 0.1），加了“回来的实例里 ≥ 0.99 的占比 ≥ 0.8”（`EXACT_SHARE`，本轮 0.893，规格 §5 已同步）与测试 `test_gt_check_fails_when_the_margin_is_not_undone`；另加 `paired_table` 两阈值各归各的测试。小项：sweep 时长 4 min 12 s、安装日志位置、`--noconftest` 统一、CLAUDE 日期、README 补阈值网格的影响（0.005 步长下预算内最好 40 个命中 @ 1.76，35 个框分数恰好 0.5；结论不变）与 patch 大于数据的警告。审阅者列的未加测试：runner `extract_cases` 的桩测试（T3）、SimpleITK/nibabel 轴序只能靠真值检查（T4）。
- **用户本轮拍板**（沿用）：第二臂 nnDetection；先只训 fold 0 + 规则 A；主线 97a58f3 + 独立 py3.8/torch 1.11 环境；只分一类；执行 = 子代理逐任务。

**之前（保留）**：S2 nnU-Net 检测器在 main（tag `handoff/2026-09-29-brain-detector`）：D1 门不过（2d 五折 0.3662 @ 1.636 FP/卷，主因漏检），记录索引 `docs/verification/2026-09-28/brain_detector/README.md`。S1 关系基线（tag `handoff/2026-09-28-relation-baselines`，NOT_EVIDENCE）、Level R 读片工具（冒烟服务 8791，浏览器验收仍 USER_REPORTED）同前。

## 2. 待用户拍板

- **规则 A 不过，臂停止（S2 的 D9：不调参救门）。** 小病灶检测下一步走哪条由用户定：a）接受 S2 与本臂的结论，小病灶检测暂放，转 S4/S5；b）换思路（如 2D 检测器、或先解决 3 mm/5 mm 层厚下单层病灶的表示）——都要新规格，不在本计划内。不建议用 sweep 版或降阈值救门。
- **push**：main 上所有提交与 tag（09-26 起）只在本地。
- **可删清单（只列，不删；删除由用户执行）**：S2 工作树 `../foundation_model-detector` 与分支 `build/brain-detector`（已合并）；本轮合并后的工作树 `../foundation_model-nndet` 与分支 `build/brain-nndet`；玩具冒烟根 `/data2/congcong/data/FM_data/derived/nndet_smoke/`；`~/logs/nndet_install/wheels/` 里 1.6 GB 的 torch 安装包；会话 scratch 里的计划预演副本。fold 0 训练目录与 `nndet_runs/` 是结果，留着。
- 旧遗留：读片人、Level R 说明里 "other" 的定义、伦理备案、Q9 删除授权、其余 4850 卷 SynthSeg、Redivis token、两条远端评审分支去留、RSS 左右手性换算规则。

## 3. 下一步

1. 本轮收尾：终审修补已提交 → 合回 main（`--no-ff`）→ tag `handoff/2026-09-30-brain-nndet-fold0`，不 push。
2. S7 脑多病种检测器（另一工作树）正在做计划 Task 12/13（五折判门已过：胶质瘤 0.811 @ 0.35、转移瘤 0.747 @ 0.57、梗死 0.572 @ 1.56 FP/例，交叉误报与冒烟进行中），其 CLAUDE/STATUS 与本分支都改了同一处，合 main 时手工合并。
3. 之后按用户对第 2 段第一条的决定；S4（脑侧解剖层，设计概览待点头）、S5（疾病印象 + 整句：S7 的记录已带整句）、S6（膝侧 nnDetection：可复用本臂工具，但本臂结果不支持优先做）顺序同前。

## 4. 坑与别重做

- **nndet 环境**：只经 `bash -c 'source scripts/nndet_env.sh && …'`；本机 `conda activate` 不把环境放到 PATH 最前，裸 `pip`/`python` 会落到系统 python 3.10 与 `~/.local`。不与 `scripts/nnunet_env.sh` 同 shell。
- **nnDetection 结束判据**：`train.log` 出现 `Found 51 predictions for analysis`、`plan_inference.pkl` 存在、`sweep_predictions/` 有 51 个 `*_boxes.pt`。本次进程自己退出了（玩具训练卡过），别为了 teardown 错误去 kill 别人的进程。
- **nnDetection 的约定**：框每边外扩 1 格，runner 在预处理空间把低端加 1 再恢复；`splits_final.pkl` 缺失时它会自己按 KFold 新建折（启动器拒绝不同的折）；默认 `train.mode=overwrite` 会复用已有训练目录（启动器拒绝）；验证集预测只在 `--sweep` 时生成；门只读默认后处理那份，sweep 版是 NOT_GATE。默认后处理留下大量低分框（fold 0：25881 个），工作点因此落在 0.50。
- **推理入口的病例名**取 h5 文件名；表格是 Level R 框格式（每层 `[row0, row1, col0, col1]`，RSS 帧）。
- **pytest `-q -q`** 不打 "N passed" 行，数测试要用单个 `-q`。
- **下载**：代理下 1.6 GB 会断；阿里云 PyTorch 镜像拦 aria2c 默认 User-Agent（`-U curl/7.81.0` 可过）；大包用官方 sha256 核验。
- **zsh**：`echo ======` 会报错；`pgrep -f` 会匹配到执行命令的 shell 自己，等待循环用 `ps -p <pid>`。
- 记录文件是证据：不往会被提交的记录里追加监控行。S2 与 S1 的坑同前：别在 nnU-Net 上调参救门（D9）；C1 上的数字都不是证据。

## 5. 关键决定的为什么

- **nnDetection 第二臂（N1）**：直接学"一个病灶一个框"；S2 主因是漏检，它必须多找回漏检才可能过门。结果：它只额外找回 nnU-Net 2d 漏掉的 6 个，却漏掉 2d 找到的 81 个。
- **先只训 fold 0 + 规则 A（N2、N3）**：单折 51 卷直接卡 0.5 会误杀或误放，同折配对比较才看得出有没有用；0.05 是判断线，不是显著性检验。规则 A 不过就停，不补五折，是用户开跑前拍的板。
- **默认参数判门（N7）**：sweep 在验证例上调参，同一批例子再判门就是自证。
- **主线 97a58f3 + 独立 py3.8/torch 1.11 环境（N4）**、**只分一类（N5）**、**撤销外扩（N8）**、**两环境只用 JSON 交接（N10）**、**推理与门同一条链（N11）**：见规格。
- **为什么不救门**：D9。fold 0 的 FROC 说明问题在检出而不在阈值：0.05 处才找到 178/280，代价是每卷 29.9 个假阳，任何阈值都到不了 0.5 @ 2。
