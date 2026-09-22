# Foundation_Model / AnatoBind-MRI:新会话入口

先读这个文件,再读 `STATUS.md`,然后才动手。上一个会话的结论只当线索,不当事实;关键数字能重跑的先重跑,重跑与记录不符时以重跑为准并写明。

## 项目一句话

面向 3D MRI 的解剖实体 A、病灶实体 U 与显式病灶–解剖绑定 R 的结构化感知(`X → (A, U) → R`)。不称 foundation model。方案是 `RESEARCH_PLAN.md` v2.5,权威执行计划是 `docs/plans/2026-09-22-aur-v2.5-complete-technical-route.md`(冲突以它为准);Gate 链 = Gate 0(fastMRI+ 框坐标)→ A/U → Level R 人标(Gate R0)→ 公平关系基线(Gate R1)→ 鲁棒性/E。关系主战场是 fastMRI+ 脑 FLAIR 小病灶,SKM-TEA 只作几何容易的对照。主投 MedIA。

## 硬规矩

- 数据只从 `/data2/congcong/data/FM_data` 读。`/data0/congcong/data/FM_Data` 是冷备份,不读不写。当前导出是 `derived/skmtea/m1r/`(校正版分割),缓存 `m1r_cache/`,预测 `m1r_pred/`,nnU-Net 在 `derived/nnunet/`;`m1/` 只作历史,只读。
- Python:`~/anaconda3/envs/nvgen/bin/python`(torch 2.5.1+cu121、MONAI 1.5.2),命令前缀 `PYTHONNOUSERSITE=1 PYTHONPATH=.`(`~/.local` 里的 torch 2.11/cu130 会覆盖 env)。SynthSeg 用 env `synthseg`(py3.8/TF2.2,`~/src/SynthSeg`,权重 robust_2.0),全 CPU 跑。
- 测试:`PYTHONNOUSERSITE=1 PYTHONPATH=. ~/anaconda3/envs/nvgen/bin/python -m pytest tests/ -q -p no:cacheprovider`(2026-09-22 合并 PR #6 后:311 passed,约 34 s)。仓库没有 CI,评审 PR 时把分支导出到临时目录跑:`git archive <ref> | tar -x -C <dir>`。
- **当前状态:v2.5 路线已合入 main(`779123f`,2026-09-22),读片人已确认可找到,下一步是 PR-A Gate 0(fastMRI+ 框上下翻转修复)。先读 `STATUS.md`,再读 v2.5 计划 §2、§4、§7–§12、§19、§23、§25(待拍板的 5 项)。别再在 SKM-TEA 膝关节上找"退化让绑定失效"(G2 裁决已测死),也别把 SynthSeg 查表当脑侧真值。**每个函数先写测试;结构性测试是"loss 在降但标签错了"的唯一自动防线。
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
- **fastMRI+ 的框 y 从 RSS 底部数起**(官方 README:转 DICOM 时上下翻转),框占行 `[nr − y − h, nr − y)`。Gate 0 落地前,`fastmri_knee.py` / `dataset_knee.py` 与 `derived/fastmri_knee/leg2` 的 56 GB 导出都是原样框,不能用来训练、评估或给医生看。
- 关系基线矩阵固定为 B0 / Bprior / Bgeo+ / B1 / B2 / B3 / B4 / B5(v2.5 §10);Bgeo+ 与学习模型必须共用同一套 16 维扩展几何(§11)。`IndependentCandidateHead` 是 B1,不是候选竞争模型。
- 不是证据的数字一律标明。每个数字附可粘贴的命令与原始输出;样板是 `docs/verification/2026-09-08/REPORT.md`。

## 阅读顺序

1. `STATUS.md`:当前状态、待拍板决定、下一步、坑。
2. `docs/plans/2026-09-22-aur-v2.5-complete-technical-route.md`:权威执行计划(Gate 链、Level R 协议与统计、基线矩阵、PR 顺序)。`RESEARCH_PLAN.md` v2.5 的 §0、§1、§9.1、§9.6、§13 是摘要与决策记录,其余多为已标记的历史章节。
3. `docs/verification/2026-09-13/G2_verdict.md`、`docs/verification/2026-09-15/H1_verdict.md`、`docs/verification/2026-09-16-brain-probe/REPORT.md`:三条实验裁决(膝 G2 不过、膝检测 H1 不过但被翻转框污染、脑探针 q3 改答 7.4%)。
4. `REVIEW_v7_feasibility_2026-09-16.md`、`REVIEW_v7_feasibility_2026-09-19.md`、`REVIEW_core_target_AUR_2026-09-22.md`:三份评审;PR #4/#5/#6 的评审在 GitHub(链接见 STATUS.md §1)。
5. `REVIEW_expert_comments_audit_2026-09-09.md`:类别感知查表天花板 0.968 的审计。
6. `docs/superpowers/specs/2026-09-14-leg2-fastmri-knee-detection-gate-design.md`、`docs/superpowers/plans/`:leg 1/leg 2 实施记录与执行发现。
7. `docs/data_engine_skmtea.md`、`docs/data_engine_synthseg.md`:数据引擎事实。`docs/architecture_and_novelty_2026-09-08.md`:旧架构说明(历史)。

## 代码地图

- `anatobind/data_engine/`:`skmtea.py`(标注筛选 D5、分割坐标系、宿主解析、逐卷导出)、`skmtea_recon.py`(adjoint SENSE、Poisson 嵌入、欠采、支撑上加噪)、`resample.py`、`splits.py`、`fastmri.py`、`synthseg_pipeline.py`、`fastmri_knee.py`(leg 2:fastMRI+ 膝标注清洗、3D 合并、共享 RSS 算子 `reconstruct_rss`、七视图导出;**框仍按原样用,Gate 0 要在这里加 CSV→RSS 转换**)。
- `anatobind/model/`:`backbone.py`(MONAI Swin,无 M_valid、无 RoPE,可开梯度检查点)、`decoders.py`(身份锚定 A、全分辨率 mask 头、可选像素解码器、DETR 式 U_B 仅供最小通路)、`dense_head.py`(密集中心热图 U_B)、`dense_head_2d.py` + `detector2d.py`(leg 2 的 2.5D 检测器)、`reliability.py`(leg 2 可靠性头)、`upstream.py` + `upstream_losses.py`(第一批上游)、`relation.py`(`IndependentCandidateHead` = B1;`RelationModule` = 旧全局 K×M,仅消融;B3 `CandidateCompetitionHead` 未实现)、`losses.py`、`armb.py`(最小通路,历史)。
- `anatobind/train/`:`dataset.py`(最小通路裁块,历史)、`cache.py`(float16 缓存)、`dataset_v2.py`(整卷、4 类、none/unknown 宿主、批内补零)、`dataset_knee.py`(leg 2 切片数据集,只有左右翻转增广)、`train_upstream.py`、`train_detector.py`(固定步数、可续跑)。
- `anatobind/eval/`:`predict.py`(缓存预测)、`lookup.py`(类别感知 B0 + 零重叠最近兜底,PR-C 在此扩展)、`matching.py`(IoU 匹配、分桶)、`g2.py`(配对差、扫描 bootstrap)、`detect3d.py` + `gate.py`(leg 2 的 3D 聚合与 H1/H2/H3 门)。`anatobind/nnunet/prepare.py`(nnU-Net 数据集与划分)。`anatobind/data_engine/rawtrack_fetch.py`、`seg_frames.py`(校正版分割获取与坐标系)。
- `scripts/`:`build_skmtea_m1.py`(历史)、`build_skmtea_m1r.py`、`build_m1r_cache.py`、`fetch_skmtea_rawtrack_api.py`、`discover_rawtrack_frame.py`、`train_upstream.py`、`cache_predictions.py`、`nnunet_prepare.py` + `nnunet_env.sh`、`collect_nnunet_predictions.py`、`eval_g1_g2.py`、`peek_fold.py`;旧的 `train_armb_minimal.py`、`render_binding_overlay.py`、`run_synthseg_fastmri_brain.py`、`verify_skmtea_frames.py`。
- 另一个长期 worktree:`../foundation_model-armb`(分支 armb,已合入 main)。
