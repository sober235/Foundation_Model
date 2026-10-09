# AnatoBind-Brain SSL-first：记录索引（2026-10-09 起）

执行计划：`docs/superpowers/plans/2026-10-09-anatobind-brain-ssl-first-execution.md`（决定 Q2–Q20、任务 E0/E1/T01–T12、门 G0–G3）。方案：`…/2026-10-09-anatobind-brain-ssl-first-three-stage.md`；增补：`docs/superpowers/specs/2026-10-09-anatobind-brain-ssl-first-addendum.md`。
数据根 `/data2/congcong/data/FM_data/derived/aur/`（不入库）：`samples_1mm_v1.json`、`resampled_1mm_v1/`、`ssl_manifest_v1/`、`ssl_runs/`。

| 目录 / 文件 | 内容 | 状态 |
|---|---|---|
| `g0/grids_1mm_v1.txt` | E1 之后 `scripts/aur_prepare.py --stage grids` 对 1 mm 表的网格检查：4960 行 0 错 | G0 网格项过 |
| （数据根）`ssl_manifest_v1/split_leakage_report.json` | T01：6130 行（train 5501 / val 629），test 患者 249 人在 SSL 里 0 行，val 患者在 SSL train 里 0 行，重复图像 0；`patients_under_two_sources` 列出 HCP 与 BMSR 同名 ID 100206、100307（不同人） | G0 泄漏项过（同名 ID 见下） |
| （数据根）`ssl_manifest_v1/data_inventory.csv` | source × sequence × split × spacing 交叉表（厚层与重采样行数） | — |
| `p0/PROBES.md` | T07 显存 / 吞吐探针全表（有效与无效运行都列），主训配置的推导 | G0 四卡探针项过 |
| `p0/PILOT.md` | T08 pilot 8k 曝光（3 卡）、验证指标、恢复一致性逐步对照、G0.5 判定 | G0.5 过 |
| `p0/ddp_*`、`p0/single_*`、`p0/*.log` | 探针原始输出（`probe.json`、`log_rank0.jsonl`、`probe_config.json`） | — |
| （数据根）`ssl_runs/pilot_8k_mb12/`、`pilot_8k_mb12_resume150_r2/` | pilot 与恢复运行（日志、`val.jsonl`、resume 与导出 checkpoint） | — |
| （数据根）`ssl_runs/stage1_320k_mb12/` | **Stage I 主训**：2026-10-09 11:14（BST，本机时区）起，卡 5/6/7，全局 36，lr 4.5e-4（PROPOSED），预热 444 步，8889 步，验证与 resume 每 500 步 | 运行中 |
| `g1/` | T09 冻结探针与 G1 判定 | 待主训结束 |
| `reviews/t10-t11-trainer-review.md` | T10/T11（Stage II/III 训练器 `anatobind/aur/train.py`、`scripts/aur_train.py`、`tests/test_aur_training_contract.py` 17 个测试）的独立评审与修补清单；双进程 gloo DDP 极端监督测试在套件里 | 代码就绪；4 卡 100 步含保存/恢复的验证记录待空卡 |
| `reviews/t12-eval-infer-review.md` | T12（整卷推理 `anatobind/aur/infer.py`、评估 `anatobind/aur/eval.py`、`scripts/infer_anatobind_brain.py`、`scripts/aur_eval.py`，23 个测试）的独立评审与修补清单（含一条阻塞：autocast 外跑头，已修并在卡上验证） | 代码就绪；评估等 Stage III |

## T10–T12 的口径（代码层面，尚未跑真实训练）

- Stage II：`--init-backbone` 严格载入 Stage I 导出，主线必须带通过的 `g1_report.json`（`--pilot` 只给冒烟）；`--init random` 是 C0 对照并写进记录；relation 头、Stage I 的 mask token、coarse mask embedding 在 Stage II 冻结。Stage III：`--resume-stage2` 严格继承整个 Stage II 导出，三组学习率 5e-5 / 2.5e-4 / 5e-4（Q12），预热 200 步（PROPOSED）。预算 240k / 80k 裁块。
- 评估（T12）：A 的门只算 `a_supervised` 行；U 按 (来源, 序列) 分组，门只读与 S7 标签同源的序列（PDGM FLAIR、BMSR T1c、ISLES DWI），阈值默认在测试集上选且报告里标"test-selected"，用 `--u-threshold 来源=阈值` 传验证集上选的阈值；BMSR/ISLES 的 AUR 行在 1 mm 重采样网格上而参考在原生网格上，报告里逐来源注明"approximate"；R 受控轨道的真值就是 SynthSeg 查表本身，没有 B0*；gap 条件 = 绑定窗内 13 宿主 Dice ≥ 0.8 且在所选阈值下被检出；端到端轨道 = 预测病灶在阈值下匹配到真值者；Level R 读片表（盲）与密钥分开导出。
- 推理：输出写回输入自己的方向与仿射；`record.json` 里 `box`（规范 RAS zyx）与 `box_input`（输出文件自身轴序）并列；部署工作点 = 评估选定的图阈值 0.3 + `--score-threshold`。

## 本轮口径与注意

- 主训批与学习率偏离 Q11（全局 16、lr 3e-4）：用户 10-09 要求"占满显存"，改为 microbatch 12 × 卡数（探针见 `p0/PROBES.md`），学习率按平方根缩放、预热按曝光数折算；标 PROPOSED，曝光预算 320k 不变。
- 卡数是 3 不是 4：0、1、3 号卡在 10:37（BST）被本机另一会话的 MC-GS `arc/cycle.py` 占用（各 45.8 GB、4000 步），7 号卡一度也被占；主训起时只有 5、6、7 空。
- 探针与 pilot 期间三次被外来进程挤到 OOM（`p0/PROBES.md` 无效运行表、`p0/PILOT.md` 恢复段），都是同一台机器上别的会话往"空卡"上发任务；凡四卡以上的运行先与其他会话对齐卡位。
- 厚层规则口径：执行计划写"BMSR 9 例"，那是 `p0/spacing.txt`（2026-10-08 记录）按"某轴 ≥ 3 mm"数的；实现 `anatobind/aur/resample.py::THICK_MM = 3.0` 按"层厚严格大于 3 mm"（与 Q4 原文"> 3 mm"一致）只标 6 行，3 例层厚恰为 3.0 mm 的 BMSR 保留 A/R 监督。实现没错，计划里的"9 例"应改为"6 行（3 例 3.0 mm 不算厚层）"。
- HCP 与 BMSR 同名患者 ID（100206、100307）：是不同人；`anatobind/aur/ssl/dataset.py::patient_id` 以 `source:patient` 的 CRC32 作键，两家各自成一个 id，InfoNCE 不会把它们当同一患者；两家都在 train，无泄漏。泄漏报告的 `patients_under_two_sources` 只是提示。
