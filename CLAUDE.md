# Foundation_Model / AnatoBind-MRI:新会话入口

先读这个文件,再读 `STATUS.md`,然后才动手。上一个会话的结论只当线索,不当事实;关键数字能重跑的先重跑,重跑与记录不符时以重跑为准并写明。

## 项目一句话

面向 3D MRI 的解剖实体 A、病灶实体 U 与显式病灶–解剖绑定 R 的结构化感知(`X → (A, U) → R`)。不称 foundation model。权威执行计划是 `docs/plans/2026-09-22-aur-v2.6-experiment-design-route.md`(v2.6,冲突以它为准;v2.5/v2.4 文件只作审计记录);Gate 链 = Gate 0(fastMRI+ 框坐标、患者 ID 折)→ Gate 0.5(全集 1297 病灶几何盘点,冻结困难组阈值 t)→ A/U(膝侧)→ Level R 人标(Gate R0,两层一致率门)→ 公平关系基线(Gate R1)→ 鲁棒性/E。关系主战场是 fastMRI+ 脑 FLAIR 小病灶,SKM-TEA 只作几何容易的对照。主投 MedIA。

## 硬规矩

- 数据只从 `/data2/congcong/data/FM_data` 读。`/data0/congcong/data/FM_Data` 是冷备份,不读不写。当前导出是 `derived/skmtea/m1r/`(校正版分割),缓存 `m1r_cache/`,预测 `m1r_pred/`,nnU-Net 在 `derived/nnunet/`;`m1/` 只作历史,只读。
- Python:`~/anaconda3/envs/nvgen/bin/python`(torch 2.5.1+cu121、MONAI 1.5.2),命令前缀 `PYTHONNOUSERSITE=1 PYTHONPATH=.`(`~/.local` 里的 torch 2.11/cu130 会覆盖 env)。SynthSeg 用 env `synthseg`(py3.8/TF2.2,`~/src/SynthSeg`,权重 robust_2.0),全 CPU 跑。
- 测试：`PYTHONNOUSERSITE=1 PYTHONPATH=. ~/anaconda3/envs/nvgen/bin/python -m pytest tests/ -q -p no:cacheprovider`（2026-09-25 合入后：397 passed，约 40 s；机器有别的训练时 100 s）。`.github/workflows/tests.yml` 在 GitHub 上跑同一套 CPU 测试（装 CPU torch 2.5.1 + `requirements-ci.txt`）；评审 PR 时也可把分支导出到临时目录跑：`git archive <ref> | tar -x -C <dir>`。
- **当前状态（2026-09-25，tag `handoff/2026-09-25-gate0-h1-gate05`）：Gate 0 已落地（fastMRI+ 框翻转修进数据引擎，新导出根 `derived/fastmri_knee/leg2_gate0/`，旧根拒载）；膝 H1 用修好的框重跑五折，迁移判据不过（clean × 四共享族大类正确灵敏度 0.091 @ 1.53 FP/卷，门 ≥ 0.5），按预先登记推荐 VERDICT §4 的 ②（nnDetection）；Gate 0.5 盘点完成（1297 小病灶，距离第二近脑区 ≤ 2 mm 的占 58%），t 未冻结，等用户定分层。** 先读 `STATUS.md`，再读 `docs/superpowers/plans/2026-09-24-gate0-h1-rerun-gate05.md`（决定 D1–D13）与 `docs/verification/2026-09-24/REPORT.md`。计划 1（膝侧能力系统）的产物仍在：`docs/verification/2026-09-23/knee_eval/{REPORT,VERDICT}.md`、`scripts/infer_knee.py`。v2.6 的 Gate 链仍有效；别再在 SKM-TEA 膝关节上找"退化让绑定失效"（G2 裁决已测死），也别把 SynthSeg 查表当脑侧真值。**每个函数先写测试；结构性测试是"loss 在降但标签错了"的唯一自动防线。
- 提交:作者用仓库本地配置(Congcong Liu);消息英文、句首大写、像现有历史一样描述做了什么;**不写 Co-Authored-By、Generated with 等任何 AI 痕迹**。push 偶发 TLS 失败时加 `https_proxy=http://127.0.0.1:7897`;凭据走 gh(sober235)。
- 分支:`main` 是权威版本,协作者只读 main。会话边界 = 提交 + 合回 main + tag `handoff/YYYY-MM-DD`。不每个会话开新分支;分支只给真正并行的工作线,合完就删。
- 根目录两个未跟踪的原始文件(`AnatoBind-MRI_cui.md`、`粘贴的 markdown …`)保持 untracked,别 stage。`runs/` 在 .gitignore 里。
- 算力:共享机。CPU 任务 nice 19、不超过 48 线程;GPU 用 `CUDA_VISIBLE_DEVICES` 钉一张;预计超过一天的任务先问用户。2026-09-12 起获准用 GPU 0(上游)、GPU 1(nnU-Net);共用卡上每次算子下发排队约 1.2 ms,估时间要在目标卡上实测。长任务用 `setsid` 脱离会话;导出类任务并发 worker 不超过 4(内存看门狗会杀)。
- 图:每张图两行,先 `http://localhost:8765/<项目>/<图>.png`,再绝对路径;图放 `~/figs/<项目>/`。
- 回复里公式不用 LaTeX(终端不渲染),用代码块。

## 方法学红线(方案 §5.1、§9.1)

- 关系真值只来自标注者的 `tissue_id`(导出列 `host_label`),永远不从重叠率推导;`IoA` 只允许作输入特征。测试 `test_host_ce_target_uses_host_label_not_max_ioa` 守着这条。
- `UNKNOWN_HOST = 0` 表示侧别未解,是"未知"不是"无宿主",从主宿主 CE 与关系 BCE 里整体剔除。
- 最小通路 `ArmBMinimal` 用真值 mask、真值 presence、匹配后的真值框做几何,它的准确率**不是 M1 证据**;日志键名 `val_host_acc_NOT_EVIDENCE` 是故意的。正式跑分前这三处 teacher forcing 必须换成预测量。
- 公平对照臂(seg-then-lookup)必须是类别感知的查表:按病灶类别限定候选结构 + IoA argmax + 零重叠取最近结构。不看类别的 argmax IoA 是稻草人(见 `REVIEW_expert_comments_audit_2026-09-09.md` §2)。
- 与 SENSE 重建图配对的分割一律用官方校正版 `segmentation_masks/raw-data-track`(论文附录 A.3/A.6);dicom-track 只用于量化旧导出的错位。
- G2 的判门路径(2026-09-12 起):每个标注病灶以标注框和类别进查表,只预测解剖;nnU-Net 分割定 G2,本模型分割只报告;检测为附带报告。评估脚本 `scripts/eval_g1_g2.py`,`GATE_KEY = "given_bucket"`。
- **脑侧关系真值只能来自 Level R 放射科医生人标**;SynthSeg + 重叠查表只是 C1/C2 伪参照(与 B0 同源,良性预处理就让它改答 2.8–6.0%),只用于训练、调试与分层,不定义任何"更准"的主张。
- **fastMRI+ 的框 y 从 RSS 底部数起**（官方 README：转 DICOM 时上下翻转），框占行 `[nr − y − h, nr − y)`，转换只在 `anatobind/data_engine/fastmri.py::convert_box_csv_to_rss` 一处。**只认 RSS 帧导出（manifest `transform_version 2`）：膝用 `derived/fastmri_knee/leg2_gate0/`，旧根 `leg2/` 是镜像框，`load_manifest / load_lesions / load_folds` 会拒载，别绕开。** 脑侧读框走 `read_fastmri_plus_rows → rows_to_rss_frame → merge_boxes_3d`，`nr` 按卷从 `reconstruction_rss` 取（320/276/256/234/213 都有）。
- **fastMRI+ 脑的系列号不等于分辨率**：205/209/210 与 200 同为 320×320 @ 0.6875 mm，0.86 mm 面内的是 202（部分）/203/206；分层按实测间距（`volume_geometry`），不按系列号。
- **Gate 0.5 的几何量**（`anatobind/eval/geometry.py`）：宿主类按 v2.6 §3 合并左右，脑室/CSF 只作地标；`d_interface` = 病灶到第二近宿主类的距离，`Δd` = 第二近 − 第一近。困难组阈值 t **未冻结**（≤ 2 mm 已占 58%），定分层前别拿 t = 2 当定论。
- 关系基线矩阵固定为 B0 / Bprior / Bgeo+ / B1 / B2 / B3 / B4 / B5(v2.6 §10);Bgeo+ 与学习模型必须共用同一套 16 维扩展几何(§11)。`IndependentCandidateHead` 是 B1,不是候选竞争模型。B2 的图像小块与 B4 的病灶编码器输入完全相同(§10.1)。
- **折按 h5 `patient_id` 划分**并断言不重叠(v2.6 §4.4);外层五折只用一次,所有选模型、调参在内层;外层测试折的 Level R 标签封存,开发者不看(§12.6–12.7)。
- **困难组 H1 的定义与任何模型、任何宿主答案无关**:病灶表面到任意两块候选脑区交界面的距离 ≤ t,t 在 Gate 0.5 后、模型前冻结(§4.5、§7.3)。脑室与 CSF 只作地标,不作宿主(§3)。主终点是全部病灶的集合值正确性(§7.8)。
- 所有学习方法在同一阶段使用同一标签来源(先伪标签;医生标签微调只按 §10.1 的预注册规则升级),否则比较不公平。
- 不是证据的数字一律标明。每个数字附可粘贴的命令与原始输出;样板是 `docs/verification/2026-09-08/REPORT.md`。

## 阅读顺序

1. `STATUS.md`:当前状态、待拍板决定、下一步、坑。
2. `docs/superpowers/plans/2026-09-24-gate0-h1-rerun-gate05.md`（本轮计划：决定 D1–D13、任务与命令）→ `docs/verification/2026-09-24/REPORT.md`（Gate 0 审计、H1 重跑、Gate 0.5 三段，每个数字带命令）→ `docs/verification/2026-09-24/{gate0,h1_rerun,gate05}/`。然后 `docs/superpowers/specs/2026-09-23-aur-capability-system-design.md`（能力系统的决定记录、接口、验收口径）→ `docs/superpowers/plans/2026-09-23-knee-capability-system.md`（计划 1）→ `docs/verification/2026-09-23/knee_eval/REPORT.md` 与 `VERDICT.md`（五折数字与检测门裁决）。`docs/plans/2026-09-22-anatobind-mri-network-target.md` 是网络目标（FM_MRI 文档），证据规则仍以 v2.6 为准。
3. `docs/plans/2026-09-22-aur-v2.6-experiment-design-route.md`:权威执行计划(Gate 链含 Gate 0.5、Level R 协议与两层一致率门、抽样与功效模拟、基线矩阵与 §10.1 初始配置、嵌套 CV 与封存、十步顺序、§25 决定记录)。`docs/plans/2026-09-22-aur-v2.5-complete-technical-route.md` 与 v2.4 文件只作审计记录。`RESEARCH_PLAN.md` 的 §0、§1、§9.1、§9.6、§13 是摘要与决策记录,其余多为已标记的历史章节。
4. `docs/verification/2026-09-13/G2_verdict.md`、`docs/verification/2026-09-15/H1_verdict.md`、`docs/verification/2026-09-16-brain-probe/REPORT.md`:三条实验裁决(膝 G2 不过、膝检测 H1 不过但被翻转框污染、脑探针 q3 改答 7.4%)。
5. `REVIEW_v7_feasibility_2026-09-16.md`、`REVIEW_v7_feasibility_2026-09-19.md`、`REVIEW_core_target_AUR_2026-09-22.md`:三份评审;`REVIEW_v2.5_feasibility_2026-09-22.md`:v2.5 合入后的冷启动独立可行性评审(有条件 GO;先读其 §0、§2、§7、§9);`REVIEW_external_experiment_design_2026-09-22.md`:外部 review 原文 + 逐条核验 + 取舍(核验脚本与输出在 `docs/verification/2026-09-22-external-review-check/`),v2.6 的直接依据;PR #4/#5/#6 的评审在 GitHub(链接见 STATUS.md §1)。
6. `REVIEW_expert_comments_audit_2026-09-09.md`:类别感知查表天花板 0.968 的审计。
7. `docs/superpowers/specs/2026-09-14-leg2-fastmri-knee-detection-gate-design.md`、`docs/superpowers/plans/`:leg 1/leg 2 实施记录与执行发现。
8. `docs/data_engine_skmtea.md`、`docs/data_engine_synthseg.md`:数据引擎事实。`docs/architecture_and_novelty_2026-09-08.md`:旧架构说明(历史)。

## 代码地图

- `anatobind/data_engine/`：`skmtea.py`（标注筛选 D5、分割坐标系、宿主解析、逐卷导出）、`skmtea_recon.py`（adjoint SENSE、Poisson 嵌入、欠采、支撑上加噪）、`resample.py`、`splits.py`、`fastmri.py`（RSS→NIfTI 几何；**fastMRI+ 框约定：`convert_box_csv_to_rss`、`rss_spacing_mm`、`voxel_to_world`、`read_fastmri_plus_rows`、`rows_to_rss_frame`、`merge_boxes_3d`**）、`synthseg_pipeline.py`、`fastmri_knee.py`（leg 2：家族映射 `read_annotations`、`merge_to_3d` 委托、共享 RSS 算子 `reconstruct_rss`、七视图导出；**导出 manifest 与加载器 `manifest_row / write_manifest / write_lesions / load_manifest / load_lesions / load_folds`、`LegacyBoxConvention`、`volume_geometry`、`EXPORT_ROOT` = `leg2_gate0`**）。
- `anatobind/model/`:`backbone.py`(MONAI Swin,无 M_valid、无 RoPE,可开梯度检查点)、`decoders.py`(身份锚定 A、全分辨率 mask 头、可选像素解码器、DETR 式 U_B 仅供最小通路)、`dense_head.py`(密集中心热图 U_B)、`dense_head_2d.py` + `detector2d.py`(leg 2 的 2.5D 检测器)、`reliability.py`(leg 2 可靠性头)、`upstream.py` + `upstream_losses.py`(第一批上游)、`relation.py`(`IndependentCandidateHead` = B1;`RelationModule` = 旧全局 K×M,仅消融;B3 `CandidateCompetitionHead` 未实现)、`losses.py`、`armb.py`(最小通路,历史)。
- `anatobind/train/`:`dataset.py`(最小通路裁块,历史)、`cache.py`(float16 缓存)、`dataset_v2.py`(整卷、4 类、none/unknown 宿主、批内补零)、`dataset_knee.py`(leg 2 切片数据集,只有左右翻转增广)、`train_upstream.py`、`train_detector.py`(固定步数、可续跑)。
- `anatobind/eval/`：`predict.py`（缓存预测）、`lookup.py`（膝：类别感知 B0 + 零重叠最近兜底；脑：`BrainLookup` 33 类无类别查表，PR-C 在此扩展）、`matching.py`（IoU 匹配、分桶）、`g2.py`（配对差、扫描 bootstrap）、`detect3d.py` + `gate.py`（leg 2 的 3D 聚合与 H1/H2/H3 门）、`geometry.py`（Gate 0.5：宿主类映射、类距离图、`d_interface` / `Δd`）、`fastmri_knee_detection.py`（H1 重跑的检测指标：族过滤、mL、中心误差、IoU、大小三分位、患者覆盖、正常卷假阳）。`anatobind/nnunet/prepare.py`(nnU-Net 数据集与划分)、`anatobind/nnunet/lesion_labels.py`(箱填病灶标签)、`anatobind/nnunet/prepare_lesion.py`(Dataset902)、`anatobind/eval/lesion_boxes.py`(连通域取框、npz 概率读取)、`anatobind/eval/detection_metrics.py`(灵敏度/假阳/工作点/门)、`anatobind/infer/{canonical,knee}.py`(入口:帧规范化、nnU-Net 子进程、病灶表、叠图)。`anatobind/data_engine/rawtrack_fetch.py`、`seg_frames.py`(校正版分割获取与坐标系)。
- `scripts/`：`build_skmtea_m1.py`（历史）、`build_skmtea_m1r.py`、`build_m1r_cache.py`、`fetch_skmtea_rawtrack_api.py`、`discover_rawtrack_frame.py`、`train_upstream.py`、`cache_predictions.py`、`nnunet_prepare.py` + `nnunet_env.sh`、`nnunet_prepare_lesion.py`、`collect_nnunet_predictions.py`、`eval_g1_g2.py`、`eval_knee_folds.py`（五折检测/绑定/解剖三张表）、`infer_knee.py`（单命令入口）、`peek_fold.py`；leg 2 / Gate 0：`build_fastmri_knee.py`（RSS 帧导出）、`relink_fastmri_knee_gate0.py`（从旧导出建 `leg2_gate0`）、`audit_fastmri_plus_boxes.py`（转换前后亮度审计）、`train_detector.py`、`cache_detections.py --score-min`、`check_h1.py`、`eval_fastmri_knee_detection.py`（H1 重跑评估与迁移判据）、`brain_frame.py`（Gate 0.5 盘点）；旧的 `train_armb_minimal.py`、`render_binding_overlay.py`、`run_synthseg_fastmri_brain.py`、`verify_skmtea_frames.py`。
- 工作树：`../foundation_model-aur`（分支 build/aur-system，计划 1 与本轮 Gate 0 / H1 重跑 / Gate 0.5 在此完成并合入 main；其 `runs/detector_gate0_fold*` 是重跑的模型；`../foundation_model-armb` 已不存在）。
