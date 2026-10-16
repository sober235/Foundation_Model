# AnatoBind-Brain SSL-first：记录索引（2026-10-09 起）

执行计划：`docs/superpowers/plans/2026-10-09-anatobind-brain-ssl-first-execution.md`（决定 Q2–Q20、任务 E0/E1/T01–T12、门 G0–G3）。方案：`…/2026-10-09-anatobind-brain-ssl-first-three-stage.md`；增补：`docs/superpowers/specs/2026-10-09-anatobind-brain-ssl-first-addendum.md`。
数据根 `/data2/congcong/data/FM_data/derived/aur/`（不入库）：`samples_1mm_v1.json`、`resampled_1mm_v1/`、`ssl_manifest_v1/`、`ssl_runs/`。

**当前入口（2026-10-16）：** 三目标修正 Task 1–6 完成，既有全量测试记录 1056 passed、2 skipped（本轮未重跑）；Task 7 的 `unmatched_empty_v1` C2 训练、验证选阈值与 1,056 行 test 已在 2026-10-15 01:56:58 UTC 完成，A 0.7988、U 三源均不过、受控 R 0.5606 < B0 0.7545，最终三门均未过、全部 `NOT_EVIDENCE`。见 [本轮结果与证据边界](u_empty_c2_20261014/RESULTS.md)。与 legacy C2 的预登记描述性配对已完成（2,000 次患者 bootstrap），Task 7 已收尾；本轮不需要新训 C0，后续模型修改需新计划。G1 第二轮已通过，下面第一轮“不过/待定”为历史记录。DICOM 解码、离线 A/U/R 与独立 2D 保留；网页/API/公开 URL 按 Q28 后移。

| 目录 / 文件 | 内容 | 状态 |
|---|---|---|
| `g0/grids_1mm_v1.txt` | E1 之后 `scripts/aur_prepare.py --stage grids` 对 1 mm 表的网格检查：4960 行 0 错 | G0 网格项过 |
| （数据根）`ssl_manifest_v1/split_leakage_report.json` | T01：6130 行（train 5501 / val 629），test 患者 249 人在 SSL 里 0 行，val 患者在 SSL train 里 0 行，重复图像 0；`patients_under_two_sources` 列出 HCP 与 BMSR 同名 ID 100206、100307（不同人） | G0 泄漏项过（同名 ID 见下） |
| （数据根）`ssl_manifest_v1/data_inventory.csv` | source × sequence × split × spacing 交叉表（厚层与重采样行数） | — |
| `p0/PROBES.md` | T07 显存 / 吞吐探针全表（有效与无效运行都列），主训配置的推导 | G0 四卡探针项过 |
| `p0/PILOT.md` | T08 pilot 8k 曝光（3 卡）、验证指标、恢复一致性逐步对照、G0.5 判定 | G0.5 过 |
| `p0/ddp_*`、`p0/single_*`、`p0/*.log` | 探针原始输出（`probe.json`、`log_rank0.jsonl`、`probe_config.json`） | — |
| （数据根）`ssl_runs/pilot_8k_mb12/`、`pilot_8k_mb12_resume150_r2/` | pilot 与恢复运行（日志、`val.jsonl`、resume 与导出 checkpoint） | — |
| `stage1/STAGE1.md`（数据根 `ssl_runs/stage1_320k_mb12/`） | **Stage I 主训**：2026-10-09 11:14–18:36（BST），卡 5/6/7，全局 36，lr 4.5e-4（Q21 认可），8889 步 / 320k 曝光，验证 masked Huber 0.0366（基线 0.361），有效秩 56.8，导出 `ssl_stage1_best.pt` | 完成 |
| `g1/G1.md`（`g1/run_20261009_1405/g1_report.json`、`g1/diag/`） | T09 冻结探针：13 宿主线性读出 Stage I 0.286 对随机 0.383（差 −0.097，区间全负）、病灶可分性 0.640 对 0.680 → **G1 不过**；诊断 D1–D3 + 微调探针（Stage II 600 步从 Stage I 骨干 vs 随机骨干）与读法都在 G1.md | 第一轮历史 FAIL；已由下列第二轮 G1 通过接续 |
| `reviews/t10-t11-trainer-review.md` | T10/T11（Stage II/III 训练器 `anatobind/aur/train.py`、`scripts/aur_train.py`、`tests/test_aur_training_contract.py` 17 个测试）的独立评审与修补清单；双进程 gloo DDP 极端监督测试在套件里 | 历史代码评审；随后 Stage II/III 真实训练已完成 |
| `reviews/t12-eval-infer-review.md` | T12（整卷推理 `anatobind/aur/infer.py`、评估 `anatobind/aur/eval.py`、`scripts/infer_anatobind_brain.py`、`scripts/aur_eval.py`，23 个测试）的独立评审与修补清单（含一条阻塞：autocast 外跑头，已修并在卡上验证） | 历史代码评审；随后多臂整卷评估已完成 |
| `g1/r2_run_20261010_1942/g1_report.json`（第二轮，λ_c 0；经过与预先登记在 `g1/G1.md` 末尾） | 正式 G1：13 宿主读出 0.724 对随机 0.383（+0.342，三种子区间下界 0.320）；病灶可分性 AUC 0.667 对 0.680（容差 0.02 内） | **过**（阈值 PROPOSED）；C2 10-11 02:16 起 |
| `stage2/C0_A_DIAG.md`（`stage2/diag/` 四个诊断脚本与图） | C0 Stage II（随机初始化）A 头学得极慢：训练时 A 点 Dice 第 3000 步 0.111；整卷冒烟 13 宿主宏 Dice 0.066（第 2000 步，CPU，4 例）；不是阈值校准，组织分不开（右白质 74 % 判成右皮层），冻结特征的线性可读性只和灰度 + 位置相当 | 历史诊断，NOT_EVIDENCE；后续配方与全测见下列记录 |
| `stage2/STAGE2_V2.md`、`stage2/frozen_probe/c2v2_step1000_check.{sh,log}`、`stage2/frozen_probe/c2v2_step1000/g1_report.json` | 决定 Q25 的预先登记与新 C2 第 1000 步检查：冻结读出 0.6829 ≥ 0.65 过，验证 A 点 Dice 0.0443 不过门 > 0.045 → **VERDICT: FAIL - stop and report**；检查脚本只报告，当时无人值守，链继续跑完 | 检查不过、未停（10-12 披露） |
| `eval/c0_val_20261010_1533/`、`eval/c0_test_20261010_1533/`（`REPORT.md`、`aggregate.json`、`reference.json`，分片日志） | 现行配方 C0（随机初始化）：验证集选 U 阈值 → 全测 1,056 行；A 宏 Dice 0.6289、U 三源 0.000 / 0.035 / 0.169、ABA(R) 0.5007 对 B0 0.6789 | G2、G3 不过 |
| `eval/c2_val_20261011_0500/`、`eval/c2_test_20261011_0500/`（同上） | 新配方 C2（预训练、主干学习率十分之一；越过不过的第 1000 步检查）：A 宏 Dice 0.7961、U PDGM 0.7538 / BMSR 0.0049 / ISLES 0.0069、ABA(R) 0.5142 对 B0 0.6692；差距子集 53 例 R 救回 23 / 弄错 7；端到端 139 例 0.7050 对 0.4892 | G2、G3 不过 |
| `eval/c0_val_20261013_0259/`、`eval/c0_test_20261013_0259/`（同上） | 新配方 C0（随机初始化、主干学习率十分之一，与新 C2 逐项同设置）：A 宏 Dice 0.7506、U PDGM 0.6077 / BMSR 0.0061 / ISLES 0.0242、ABA(R) 0.5996 对 B0 0.7013；端到端 103 例 0.6796 对 0.5340 | G2、G3 不过 |
| `eval/compare_c2v2_c0v2_20261014/`（`COMPARE.md`、`compare.json`） | Q25 配对，新 C2 − 新 C0：A +0.0454 [0.0435, 0.0477]、U PDGM FLAIR +0.1462 [0.0720, 0.2250]（误报 0.158 对 0.604）、受控 ABA(R) −0.0853 [−0.1186, −0.0496]；读法见 `stage2/STAGE2_V2.md` 末节 | 只报告，NOT_EVIDENCE |
| `eval/compare_c0v2_c0_20261014/`（同上） | 配方对照，新 C0 − 现行 C0（只差 Stage II 主干学习率）：A +0.1217 [0.1186, 0.1243]、U PDGM FLAIR +0.4385 [0.3534, 0.5360]（误报 0.604 对 0.139）、受控 ABA(R) +0.0988 [0.0562, 0.1382] | 只报告，NOT_EVIDENCE |
| `u_empty_c2_20261014/README.md`、`RESULTS.md`、`code_sha.txt`、`input_sha256.txt` | 新 U 损失 C2 的启动前约定、代码/输入哈希与结果：profile=`unmatched_empty_v1`；无 legacy Q25 早停是预注册；val 阈值 BMSR/ISLES/PDGM 0.70/0.60/0.75 | 训练/val/test 与 legacy 描述性对照均完成；三门失败 |
| `eval/c2_val_20261014_1344/REPORT.md`、`eval/c2_test_20261014_1344/REPORT.md` | 本轮 test 1,056 行：A 0.7988；U PDGM 0.6538 / BMSR 0.0146 / ISLES 0.0242；受控 ABA(R) 0.5606 对 B0 0.7545 | G2-A/G2-U/G3 均 FAIL；NOT_EVIDENCE |
| `eval/compare_c2empty_c2legacy_20261016/COMPARE.md`、`compare.json` | 新损失 C2 − legacy C2，同 1,056 行，2,000 次患者 bootstrap：A +0.0027 [0.0016, 0.0040]；PDGM FLAIR U −0.1000 [−0.1575, −0.0462]；受控 R +0.0464 [0.0050, 0.0861]。U 各臂 val 工作点不同，历史臂有 Q25 执行偏离 | 描述性对照收尾；NOT_EVIDENCE |
| `docs/verification/2026-10-16/aur_saved_eval_diagnostic/REPORT.md`、`summary.json`；同目录 `NEXT_STEPS.md` | 保存的 val 输出匿名核查完成：BMSR/ISLES raw box 覆盖仅 16/470、6/138；另一 oracle 指标下 query mask IoU 也低，不能与整卷检测直接等同或定位根因。bbox 不证明 mask 吞没或根因；下一步每源 2 个 val 病例的 query/窗口诊断尚未执行 | 诊断为 NOT_EVIDENCE；根因未定，下一步见计划 |
| `status_archive/STATUS_2026-10-14_before_task7_closeout.md` | 上一版完整 STATUS；保留旧通过记录、当时待定、卡位与历史决定 | 历史归档，不能当当前状态 |

逐例 `cases.json` 与 Level R 的 `level_r_key.json` / `level_r_sheet.csv` 留在本地对应目录、不入库（仓库公开，密钥会破坏盲读）。

## T10–T12 的原始代码口径（训练与全测现已完成）

- Stage II：`--init-backbone` 严格载入 Stage I 导出，主线必须带通过的 `g1_report.json`（`--pilot` 只给冒烟）；`--init random` 是 C0 对照并写进记录；relation 头、Stage I 的 mask token、coarse mask embedding 在 Stage II 冻结。Stage III：`--resume-stage2` 严格继承整个 Stage II 导出，三组学习率 5e-5 / 2.5e-4 / 5e-4（Q12），预热 200 步（PROPOSED）。预算 240k / 80k 裁块。
- 评估（T12 原始行为，实际运行按 val 选阈值后固定到 test）：A 的门只算 `a_supervised` 行；U 按 (来源, 序列) 分组，门只读与 S7 标签同源的序列（PDGM FLAIR、BMSR T1c、ISLES DWI），阈值默认在测试集上选且报告里标"test-selected"，用 `--u-threshold 来源=阈值` 传验证集上选的阈值；BMSR/ISLES 的 AUR 行在 1 mm 重采样网格上而参考在原生网格上，报告里逐来源注明"approximate"；R 受控轨道的真值就是 SynthSeg 查表本身，没有 B0*；gap 条件 = 绑定窗内 13 宿主 Dice ≥ 0.8 且在所选阈值下被检出；端到端轨道 = 预测病灶在阈值下匹配到真值者；Level R 读片表（盲）与密钥分开导出。
- 推理：输出写回输入自己的方向与仿射；`record.json` 里 `box`（规范 RAS zyx）与 `box_input`（输出文件自身轴序）并列；部署工作点 = 评估选定的图阈值 0.3 + `--score-threshold`。

保存 val 核查：[REPORT.md](../../2026-10-16/aur_saved_eval_diagnostic/REPORT.md)；尚未执行的机制诊断：[NEXT_STEPS.md](../../2026-10-16/aur_saved_eval_diagnostic/NEXT_STEPS.md)。

## 最新 val 报告的已知措辞缺陷

`eval/c2_val_20261014_1344/REPORT.md` 的标题仍写 test-set，阈值选择列仍写 test-selected，属于原始报告模板措辞缺陷。链日志先记录 val 选阈值，再以固定阈值执行 test；test REPORT 也明确写 chosen on the validation split。原始报告不改写，这些模板字样本身不能证明 test 泄漏；详见本轮 `RESULTS.md`。

## 历史主训口径与注意

- 主训批与学习率偏离 Q11（全局 16、lr 3e-4）：用户 10-09 要求"占满显存"，改为 microbatch 12 × 卡数（探针见 `p0/PROBES.md`），学习率按平方根缩放、预热按曝光数折算；标 PROPOSED，曝光预算 320k 不变。
- 卡数是 3 不是 4：0、1、3 号卡在 10:37（BST）被本机另一会话的 MC-GS `arc/cycle.py` 占用（各 45.8 GB、4000 步），7 号卡一度也被占；主训起时只有 5、6、7 空。
- 探针与 pilot 期间三次被外来进程挤到 OOM（`p0/PROBES.md` 无效运行表、`p0/PILOT.md` 恢复段），都是同一台机器上别的会话往"空卡"上发任务；凡四卡以上的运行先与其他会话对齐卡位。
- 厚层规则口径：执行计划写"BMSR 9 例"，那是 `p0/spacing.txt`（2026-10-08 记录）按"某轴 ≥ 3 mm"数的；实现 `anatobind/aur/resample.py::THICK_MM = 3.0` 按"层厚严格大于 3 mm"（与 Q4 原文"> 3 mm"一致）只标 6 行，3 例层厚恰为 3.0 mm 的 BMSR 保留 A/R 监督。实现没错，计划里的"9 例"应改为"6 行（3 例 3.0 mm 不算厚层）"。
- HCP 与 BMSR 同名患者 ID（100206、100307）：是不同人；`anatobind/aur/ssl/dataset.py::patient_id` 以 `source:patient` 的 CRC32 作键，两家各自成一个 id，InfoNCE 不会把它们当同一患者；两家都在 train，无泄漏。泄漏报告的 `patients_under_two_sources` 只是提示。
