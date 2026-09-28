# STATUS：2026-09-29（nnDetection 第二臂：计划 10 个任务中 1–8 已实现并逐个评审；fold 0 于 04:24 在 GPU 3 开训，约 13 h，预计 17:30 前结束；本轮中途交接，合入 main 并打 tag `handoff/2026-09-29-nndet-fold0-training`，未 push）

每次交接前整体重写本文件。五段固定：已验证、待拍板、下一步、坑与别重做、为什么。

## 1. 已完成且已验证

**本轮：S2 的第二臂 nnDetection（Retina U-Net）。** 规格 `docs/superpowers/specs/2026-09-28-brain-nndet-design.md`（决定 N1–N14），计划 `docs/superpowers/plans/2026-09-28-brain-nndet.md`（10 个任务）。分支 `build/brain-nndet`（工作树 `../foundation_model-nndet`）。执行 = 子代理逐任务（全部用 sonnet），每个任务控制方重跑测试、检查提交信息、与计划代码逐字节比对，再由独立审阅者评审；逐任务裁定都在台账 `../foundation_model-nndet/.superpowers/sdd/2026-09-28-brain-nndet/progress.md`（gitignore，不入库）。

- **测试**：
  ```
  PYTHONNOUSERSITE=1 PYTHONPATH=. ~/anaconda3/envs/nvgen/bin/python -m pytest tests/ -q -p no:cacheprovider
  747 passed, 1 skipped
  bash -c 'source scripts/nndet_env.sh && python -m pytest tests/test_nndet_runner.py -q -p no:cacheprovider --noconftest'
  2 passed, 5 warnings（batchgenerators/nnunet 自身的 scipy 弃用警告）
  ```
- **环境 `nndet`**（Task 1）：python 3.8.20、torch 1.11.0+cu113、nvcc 11.3.122、PyTorch Lightning 1.4.2、SimpleITK 2.0.2、numpy 1.24.4、setuptools 59.5.0；nnDetection 97a58f3 的扩展用系统 gcc-10 编译。装了七次才成，每次原因与脚本在 `docs/nndet_install.md` 与 `~/logs/nndet_install/`。玩具数据 smoke + sweep 跑通（验证集预测生成、sweep 60 s、`plan.pkl` 不含 `inference_plan`）。
- **坐标链**（Task 3）：runner 在玩具数据上与 nnDetection 自己的 `val_predictions` 逐框对账：高端相等、低端恰好差撤销掉的 1 格；"真值当预测" 10/10 精确；单卷推理日志 `Found inference plan: {} for prediction`（默认参数）。
- **Task903 真实数据**（Task 6，`docs/verification/2026-09-29/brain_nndet/`）：253 例、1297 个实例（只有病灶 815 的框被覆盖改变）；规划：目标间距 [5, 0.6875, 0.6875] mm、不转置、块 [12, 256, 224]、batch 4；折 51/51/51/50/50，与 Dataset903 相同；**开训前坐标检查通过**：1289/1297 预处理后仍在，全部与自己的注册表框 IoU ≥ 0.1（最小 0.316、中位 1.0、≥ 0.99 的 1151 个），假阳 0；丢失的 8 个（1001、1004、1011、1042、1061、1063、1080、1104）全是 3 mm 层厚卷里的单层病灶，重采样到 5 mm 时消失，仍算分母。审阅者对 253 卷全量核查了实例图、类别 json、与 Dataset903 的标签和图像逐字节一致。
- **fold 0 训练**（Task 7，`launch.md`）：2026-09-29 04:24:44 启动，GPU 3（当时八卡全空），wrapper pid 3334387、训练 pid 3334388；3.64 it/s，每 epoch 2600 步 ≈ 11.9 min，60 epoch ≈ 11.9 h，加 sweep ≈ 13 h（< 24 h，按 N9 无需问用户）。日志 `../foundation_model-nndet/logs/brain_nndet/fold0.log`，训练目录 `/data2/congcong/data/FM_data/derived/nndet_models/Task903_FastMRIBrainSmallLesion/RetinaUNetV001_D3V001_3d/fold0`。
- **代码**：`anatobind/nndet/{brain_task,boxes}.py`、`anatobind/eval/brain_nndet.py`（+ `brain_detector.strata_maps`）、`anatobind/infer/brain_nndet.py`；脚本 `scripts/nndet_{env.sh,prepare,runner,train}.py`、`eval_brain_nndet.py`、`infer_brain_lesions_nndet.py`。
- **用户本轮拍板**：第二臂 nnDetection（按推荐）；先只训 fold 0 + 规则 A（fold 0 命中 ≥ nnU-Net 2d 同折 92 + 14 = 106/280 才补五折）；方案 1（主线 97a58f3 + 独立 py3.8/torch 1.11 环境）；只分一类；四段设计；规格与计划（执行 = 子代理逐任务）。

**之前（保留）**：S2 nnU-Net 检测器已在 main（tag `handoff/2026-09-29-brain-detector`）：D1 门不过（2d 五折 0.3662 @ 1.636 FP/卷，主因漏检，解码范围只碰到 0.497），记录索引 `docs/verification/2026-09-28/brain_detector/README.md`。S1 关系基线（tag `handoff/2026-09-28-relation-baselines`，NOT_EVIDENCE）、Level R 读片工具（冒烟服务 8791，浏览器验收仍 USER_REPORTED）同前。

## 2. 待用户拍板

- **push**：main 自 afe641e 起所有提交与 tag（09-26 两个、09-28 一个、09-29 两个）都只在本地。
- **fold 0 出结果后**：规则 A 通过就要补五折，墙钟约一天以上，开跑前问；不通过就停，交用户决定（不调参救门）。
- **读片人**、Level R 读片说明里 "other" 的定义、伦理备案：同前。
- **可删清单（只列，不删；删除由用户执行）**：S2 工作树 `../foundation_model-detector` 与分支 `build/brain-detector`（已合并）；玩具冒烟根 `/data2/congcong/data/FM_data/derived/nndet_smoke/`；`~/logs/nndet_install/wheels/` 里 1.6 GB 的 torch 安装包（环境已装好）；会话 scratch 里的计划预演副本。
- 旧遗留：Q9 删除授权、其余 4850 卷 SynthSeg、Redivis token、两条远端评审分支去留、RSS 左右手性换算规则。

## 3. 下一步

1. **等 fold 0 训练与 sweep 结束**：判据是训练目录的 `train.log` 出现 `Found 51 predictions for analysis`、`plan_inference.pkl` 存在、`sweep_predictions/` 有 51 个 `*_boxes.pt`。nnDetection 结束时会打 batchgenerators 的 teardown `RuntimeError`，并可能卡在退出（玩具训练见过）：输出齐了就 `kill -TERM 3334388`，把进程号、时间、退出码记进 `training.txt`（计划 Task 9 Step 1）。
2. **计划 Task 9**：runner 按默认参数与调参参数提取 → `scripts/eval_brain_nndet.py --folds 0` 出 fold 0 报告与规则 A；推理冒烟用 `file_brain_AXFLAIR_201_6002917.h5`（S2 冒烟同一卷），核对病例名 `case` 与 nndet 解释器；写记录索引 README（已知偏差、实测耗时）。
3. **计划 Task 10**：CLAUDE.md 代码地图、STATUS.md、两套测试、全分支终审（最强模型）、合回 main、tag `handoff/<日期>-brain-nndet-fold0`，不 push。
4. 之后按规则 A 结论：通过 → 报告五折时长并问用户；不通过 → 停。其余子项目 S3（B3/B4）、S4（脑侧解剖层，可与检测并行）、S5（疾病印象 + 整句）、S6（膝侧 nnDetection，可复用本臂工具）顺序同前。

## 4. 坑与别重做

- **nndet 环境**：只经 `bash -c 'source scripts/nndet_env.sh && …'`（它把环境 bin 显式放到 PATH 最前）；本机 `conda activate` 不会这样做，裸 `pip`/`python` 会落到系统 python 3.10 与 `~/.local`（本轮在下载阶段拦住，未装入任何东西）。不与 `scripts/nnunet_env.sh` 同 shell。
- **nnDetection 的约定**：框每边外扩 1 格（`[min−1, max+1]`），runner 在预处理空间把低端加 1 再恢复；`splits_final.pkl` 缺失时它会自己按 KFold 新建折（启动器会拒绝不同的折）；默认 `train.mode=overwrite` 会复用已有训练目录（启动器会拒绝）；验证集预测只在 `--sweep` 时生成；门只读默认后处理参数那份（sweep 调参版是 NOT_GATE）。
- **下载**：代理下 1.6 GB 会断；阿里云 PyTorch 镜像拦 aria2c 默认 User-Agent（`-U curl/7.81.0` 可过）；大包下载后用官方 sha256 核验。
- **zsh**：`echo ======` 会报错；`pgrep -f` 会匹配到执行命令的 shell 自己（本轮误杀过一次自己的 shell），等待循环用 `ps -p <pid>`。
- 记录文件是证据：不要往会被提交的记录里追加监控行（实现者说明已写入）。
- S2 与 S1 的坑同上一版：别在 nnU-Net 上调参救门（D9）；C1 上的数字都不是证据。

## 5. 关键决定的为什么

- **nnDetection 第二臂（N1）**：直接学"一个病灶一个框"；但 S2 主因是漏检，所以它必须多找回漏检才可能过门。
- **先只训 fold 0 + 规则 A（N2、N3）**：官方默认每折约一天；单折 51 卷，直接卡 0.5 会误杀或误放，同折配对比较才看得出有没有用；0.05 是判断线，不是显著性检验。
- **主线 97a58f3 + 独立 py3.8/torch 1.11 环境（N4）**：主线要求 PyTorch 1.X；nextrelease 未发布。
- **只分一类（N5）**：腔隙性梗死只有 57 个；NMS 按类别分开做，两类会重复出框。
- **默认参数判门（N7）**、**撤销外扩（N8）**、**两环境只用 JSON 交接（N10）**、**推理与门同一条链（N11）**：见规格。
- **中途交接合入 main**：训练要 13 小时，协作者只读 main，最新状态不能只停在侧分支；计划 Task 10 结束时再合一次。
