# Foundation_Model / AnatoBind-MRI:新会话入口

先读这个文件,再读 `STATUS.md`,然后才动手。上一个会话的结论只当线索,不当事实;关键数字能重跑的先重跑,重跑与记录不符时以重跑为准并写明。

## 项目一句话

面向 3D MRI 的解剖–异常关系绑定(R)与关系可观测性(E)的结构化感知编码器。不称 foundation model。方案是 `RESEARCH_PLAN.md` v2.1;第一道门是 M1 赌注实验,战场 SKM-TEA(§9.1)。主投 MedIA。

## 硬规矩

- 数据只从 `/data2/congcong/data/FM_data` 读。`/data0/congcong/data/FM_Data` 是冷备份,不读不写。当前导出是 `derived/skmtea/m1r/`(校正版分割),缓存 `m1r_cache/`,预测 `m1r_pred/`,nnU-Net 在 `derived/nnunet/`;`m1/` 只作历史,只读。
- Python:`~/anaconda3/envs/nvgen/bin/python`(torch 2.5.1+cu121、MONAI 1.5.2),命令前缀 `PYTHONNOUSERSITE=1 PYTHONPATH=.`(`~/.local` 里的 torch 2.11/cu130 会覆盖 env)。SynthSeg 用 env `synthseg`(py3.8/TF2.2,`~/src/SynthSeg`,权重 robust_2.0),全 CPU 跑。
- 测试:`PYTHONNOUSERSITE=1 PYTHONPATH=. ~/anaconda3/envs/nvgen/bin/python -m pytest tests/ -q -p no:cacheprovider`(2026-09-12:224 passed,约 45 s)。每个函数先写测试;结构性测试是"loss 在降但标签错了"的唯一自动防线。
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
- 不是证据的数字一律标明。每个数字附可粘贴的命令与原始输出;样板是 `docs/verification/2026-09-08/REPORT.md`。

## 阅读顺序

1. `STATUS.md`:当前状态、待拍板决定、下一步、坑。
2. `RESEARCH_PLAN.md` §0、§5.1、§9.1、§9.5、§9.7、§13:方案与决策记录。
3. `REVIEW_expert_comments_audit_2026-09-09.md`:最新审计,含类别感知查表天花板。
4. `docs/verification/2026-09-08/REPORT.md`、`docs/verification/2026-09-09/`:可重跑的证据。
5. `docs/superpowers/plans/2026-09-07-m1-skmtea-data-engine.md`、`…-m1-arm-b-minimal-path.md`:实施记录与执行发现 F1–F5。
6. `docs/data_engine_skmtea.md`、`docs/data_engine_synthseg.md`:数据引擎事实。
7. `docs/architecture_and_novelty_2026-09-08.md`:架构与 novelty 说明(PDF 源文)。

## 代码地图

- `anatobind/data_engine/`:`skmtea.py`(标注筛选 D5、分割坐标系、宿主解析、逐卷导出)、`skmtea_recon.py`(adjoint SENSE、Poisson 嵌入、欠采、支撑上加噪)、`resample.py`、`splits.py`、`fastmri.py`、`synthseg_pipeline.py`。
- `anatobind/model/`:`backbone.py`(MONAI Swin,无 M_valid、无 RoPE,可开梯度检查点)、`decoders.py`(身份锚定 A、全分辨率 mask 头、可选像素解码器、DETR 式 U_B 仅供最小通路)、`dense_head.py`(密集中心热图 U_B)、`upstream.py` + `upstream_losses.py`(第一批上游)、`relation.py`、`losses.py`、`armb.py`(最小通路,历史)。
- `anatobind/train/`:`dataset.py`(最小通路裁块,历史)、`cache.py`(float16 缓存)、`dataset_v2.py`(整卷、4 类、none/unknown 宿主、批内补零)、`train_upstream.py`(固定步数、可续跑)。
- `anatobind/eval/`:`predict.py`(缓存预测)、`lookup.py`(类别感知 B0)、`matching.py`(IoU 匹配、分桶)、`g2.py`(配对差、扫描 bootstrap)。`anatobind/nnunet/prepare.py`(nnU-Net 数据集与划分)。`anatobind/data_engine/rawtrack_fetch.py`、`seg_frames.py`(校正版分割获取与坐标系)。
- `scripts/`:`build_skmtea_m1.py`(历史)、`build_skmtea_m1r.py`、`build_m1r_cache.py`、`fetch_skmtea_rawtrack_api.py`、`discover_rawtrack_frame.py`、`train_upstream.py`、`cache_predictions.py`、`nnunet_prepare.py` + `nnunet_env.sh`、`collect_nnunet_predictions.py`、`eval_g1_g2.py`、`peek_fold.py`;旧的 `train_armb_minimal.py`、`render_binding_overlay.py`、`run_synthseg_fastmri_brain.py`、`verify_skmtea_frames.py`。
- 另一个长期 worktree:`../foundation_model-armb`(分支 armb,已合入 main)。
