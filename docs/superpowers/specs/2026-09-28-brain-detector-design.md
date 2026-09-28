# 脑侧小病灶检测器（子项目 S2）· 设计规格

> 状态：2026-09-28 与用户逐段讨论后定稿（四段设计各自点头，决定见 §1）。它记录**已经定下的设计**，实施计划另写。
> 上游：S1 规格 `docs/superpowers/specs/2026-09-27-relation-baselines-design.md`（P15 疾病层 D、P16 子项目顺序）、v2.6 §25 第 11 项。膝侧先例：`docs/verification/2026-09-15/H1_verdict.md`（自写 2.5D 检测器塌陷）、`docs/verification/2026-09-24/REPORT.md` §3（修框后重跑仍不过）、`docs/verification/2026-09-23/knee_eval/VERDICT.md`（nnU-Net 框填检测 0.254 @ 1.57 FP，门不过）。

## 0. 一句话

在 fastMRI+ 脑 FLAIR 上训练一个小病灶检测器：把 1297 个注册病灶的逐层框填成掩膜，用 nnU-Net v2（2D 为主）分割，再取连通域出带分数的 3D 框；按患者五折留出，在每卷假阳 ≤ 2 的工作点上判病灶级灵敏度 ≥ 0.5。它补上三目标里"找到异常"这一层，并给 S5 提供推理入口：150 个有小血管病印象的检查里有 130 个没有框，只能靠检测器找病灶。不需要医生标签。

## 1. 决定记录

| # | 决定 | 一句话理由 |
|---|---|---|
| D1 | 门 = 按患者五折留出，3D IoU ≥ 0.1 一对一匹配，每卷假阳 ≤ 2 的工作点上病灶级灵敏度 ≥ 0.5（用户选 A） | 与膝侧 H1、检测门同一口径，可比 |
| D2 | 训练与评估用 165 个有小病灶框的卷 + 88 个只有"年龄相符正常"标签的卷（用户选 A）；不用 194 个带其他标签的卷 | 正常卷上的检出是干净的假阳；其他标签卷里未画的小病灶会同时污染负样本与假阳计数 |
| D3 | 方法 = nnU-Net v2 分割框填掩膜 → 连通域出框（用户选方案 1）；自写 2.5D CenterNet 不修；nnDetection 留作门不过时的第二臂 | 复用膝侧已测的管线；Dice + CE 与标准配方避开自写检测器"负样本压倒正样本、训不够"的坑 |
| D4 | 2D 配置为主并判门，3D fullres 五折只报告 | 79% 的病灶只占一层，层厚 5 mm，面内中位数 6.9 mm |
| D5 | 训练器 `nnUNetTrainer_250epochs` | 与膝侧 Dataset901/902 一致（用户 09-12 定） |
| D6 | 解码最小块 9 体素（3×3×1），26 连通；膝侧的 27 体素默认值不变 | 单层 4×4 的小病灶只有 16 体素，按 27 会被整个丢掉 |
| D7 | GPU：用户 2026-09-28 授权用 4 张 80 GB 卡；默认 GPU 0–3，每折钉一张，启动前用 nvidia-smi 确认空闲，被占就换一张空卡，不抢别人的卡 | 一折一卡，四折并行，墙钟约单折两倍 |
| D8 | 先做计时探针（fold 0，5 个 epoch），实测总时长超过一天先问用户 | 用户规矩；共用卡上的速度无法先验估计 |
| D9 | 门不过时不在 S2 里调参救门，停下交用户决定（例如 nnDetection 第二臂） | 与膝侧处理一致，防止在门上做选择 |
| D10 | 只填注册表的 1297 个病灶；病灶卷里其他标签的框（肿块等）不填也不排除，作为已知标签噪声写进报告 | 注册表是全项目一致的病灶定义（Gate 0.5、Level R、S1 共用） |

## 2. 数据集（Dataset903_FastMRIBrainSmallLesion）

- **位置**：已有 nnU-Net 根（`scripts/nnunet_env.sh`：`derived/nnunet/{raw,preprocessed,results}`）。
- **病例**：253 例 = 165 个病灶卷 + 88 个纯正常卷。case 名 = h5 文件名（去扩展名）。
- **图像**：h5 的 `reconstruction_rss`，经 `anatobind.data_engine.fastmri.rss_h5_to_nifti` 写成 NIfTI，网格 (col, row, slice)，与 Level R 导出、S1 特征表同一帧。若该函数会补层（`_pad_slices`），标签必须按同样方式补，并在测试里断言图像与标签形状一致、框坐标对齐。
- **标签**：每个注册病灶的逐层成员框（Level R 导出路径 `small_lesion_rows → merged_lesions → match_registry`）填成 1，其余为 0；正常卷全 0。构建时逐病灶核对：每个病灶的成员框在标签图中全部为 1，标签图中为 1 的体素都属于某个成员框。
- **折**：165 名病灶患者沿用 `data/level_r/folds.json`；88 名正常患者按 h5 `patient_id` 排序、种子 0 打乱后均分进五折；断言两组患者不重叠、每名患者只在一折。写成 nnU-Net 的 `splits_final.json`（每折 train/val 列表），其交叉验证输出即折外预测。
- 数据集目录已存在则拒绝重建（不覆盖）。

## 3. 训练与算力

- 预处理：`nnUNetv2_plan_and_preprocess -d 903 -c 2d 3d_fullres --verify_dataset_integrity -np 4`，nice 19。
- 训练：`nnUNetv2_train 903 2d <fold> nnUNetTrainer_250epochs --npz`，fold 0–4；之后 3d_fullres 同样五折。每个进程 `CUDA_VISIBLE_DEVICES` 钉一张卡；`nnUNet_n_proc_DA=8`，四折并行共 32 个增强进程，不超过 48 线程上限。
- 计时探针（D8）：fold 0 跑 5 个 epoch，记录每 epoch 秒数，推算五折墙钟；超过一天停下问用户。
- 长任务用 `setsid` 脱离会话，日志写到结果目录；进度按 nnU-Net 的 epoch 日志监看。

## 4. 解码与评估

- **解码**：`anatobind.eval.lesion_boxes.decode_boxes(label_map, probs, min_voxels=BRAIN_MIN_VOXELS)`，`BRAIN_MIN_VOXELS = 9`；分数 = 块内平均前景概率（`--npz` 概率）。
- **匹配与门**：复用 `anatobind.eval.detection_metrics`（`sweep`、`operating_point`、`gate`；IoU 0.1，FP_MAX 2，门 0.5）；只有一类，family 传同一值，"大类正确"即"命中"。
- **分母**：灵敏度分母 = 1297 个病灶；假阳在全部 253 卷上计，另报 88 个正常卷上的每卷假阳。
- **报告**（`docs/verification/<日期>/brain_detector/`）：2D 的门结论、FROC 阈值表、正常卷假阳；分层灵敏度（不作门）：d_interface 四档、单层 / 多层、面内尺寸三分位、实测几何层；3D fullres 的同一套数字作对照；每个数字附命令与原始输出。

## 5. 推理入口（给 S5）

`anatobind/infer/brain.py` + `scripts/infer_brain_lesions.py`：输入任意 fastMRI FLAIR h5 与一个训练好的 2D 模型折（或五折集成，由参数指定），输出病灶表：每个病灶的 z0/z1、逐层框（RSS 帧，行从上数）、分数，字段对齐 Level R 注册表与 `lesions.json` 的逐层框格式。输出目录已存在则拒绝。

## 6. 代码布局

```
anatobind/nnunet/brain_lesion.py     数据集构建：RSS → NIfTI、框填掩膜、case 命名、五折 splits、正常患者分折与不重叠断言
anatobind/eval/lesion_boxes.py       + BRAIN_MIN_VOXELS = 9（膝侧默认不变）
anatobind/eval/detection_metrics.py  原样复用
anatobind/infer/brain.py             推理：h5 → nnU-Net 预测 → 解码 → 病灶表
scripts/brain_detector_prepare.py    --stage raw | splits
scripts/eval_brain_detector.py       五折评估、分层、报告
scripts/infer_brain_lesions.py       推理命令行
tests/test_brain_detector_*.py       合成小卷，不读 /data2
```

## 7. 测试（每个函数先写测试）

框填掩膜与成员框逐体素一致；补层时图像与标签形状一致、框坐标对齐；正常卷全 0；折按患者、两组不重叠、每人一折、splits 覆盖全部 253 例；`BRAIN_MIN_VOXELS` 保住单层 4×4 病灶、膝侧默认仍为 27；评估在合成预测上给出手算的灵敏度与假阳；推理输出格式与注册表逐层框一致；输出目录存在即拒绝。

## 8. 分支与执行

- 分支 `build/brain-detector`（自 main 7796810），工作树 `../foundation_model-detector`。结束合回 main、打 tag `handoff/<日期>-brain-detector`；不 push 除非用户说。
- 执行沿用 S1：子代理逐任务、先写测试、逐任务评审、台账在工作树 `.superpowers/sdd/`。S1 的教训写进执行规则：控制方自己重跑每个任务的测试并检查 `git log -1 --format=%B`（实现者会编造输出、会听 harness 提示加 Co-Authored-By）；同一工作树只允许一个控制方。
- 提交作者用仓库本地配置，消息英文，不留 AI 痕迹。不删任何东西。

## 9. S2 完成的判据

1. Dataset903 建成：253 例，折按患者、病灶患者与正常患者不重叠，框填掩膜与注册表逐病灶核对一致。
2. 2D 五折训练完，折外预测覆盖全部 253 卷；3D fullres 五折作报告。
3. 评估报告给出门结论（过或不过都是有效结果）、FROC 表、正常卷假阳、分层灵敏度。
4. 推理入口在一个真实 h5 上跑通，输出格式对齐注册表。
5. 全量测试通过；文档（本规格、计划、验证记录、CLAUDE.md 代码地图、STATUS）齐全；合回 main 打 tag。

## 10. 明确不做

病灶类型分类（1240/1297 同一标签）；疾病印象（S5）；膝侧（S6）；门不过时的调参（D9）；194 个其他标签卷；自写检测器的修复。
