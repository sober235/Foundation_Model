# STATUS：2026-09-28（脑侧关系基线 S1 / PR-C 已实现、整支评审、修复、在真实 1297 病灶上跑完阶段 1；合入 main 并打 tag `handoff/2026-09-28-relation-baselines`，未 push；读片工具等浏览器验收与读者；下一步写 S2 规格）

每次交接前整体重写本文件。五段固定：已验证、待拍板、下一步、坑与别重做、为什么。

## 1. 已完成且已验证

**本轮（2026-09-27/28）：S1 = PR-C 脑侧关系基线。** 规格 `docs/superpowers/specs/2026-09-27-relation-baselines-design.md`（决定 P1–P16），计划 `docs/superpowers/plans/2026-09-27-relation-baselines.md`（15 个任务）。分支 `build/relation-baselines`（工作树 `../foundation_model-relation`），15 个任务各自评审通过，整支终审"Ready with fixes"后一轮修复（5 个提交），修复复审全部 ADDRESSED。台账在工作树 `.superpowers/sdd/2026-09-27-relation-baselines/progress.md`（gitignore）。

- **测试**：
  ```
  PYTHONNOUSERSITE=1 PYTHONPATH=. ~/anaconda3/envs/nvgen/bin/python -m pytest tests/ -q -p no:cacheprovider
  686 passed in 79.61s
  ```
- **真实特征表**：`/data2/congcong/data/FM_data/derived/relation/v1/`（table.csv + patches.npz + manifest.json）。1297 病灶、165 患者、五折病灶 280/250/276/334/157；C1 分布 白质 985 / 皮层 310 / 基底节 2；C1 与注册表标签级查表一致率 0.99537，6 个不一致全是白质/皮层交界病灶（64, 147, 327, 836, 971, 1086）；1 个病灶（1021）15 mm 内无候选、取最近类。记录 `docs/verification/2026-09-27/relation_table.md`，构建约 6 分钟。
- **阶段 1 跑分（伪标签 C1，全部 NOT_EVIDENCE）**：CPU 16 线程，18.6 分钟。报告 `docs/verification/2026-09-28/relation_baselines/REPORT.md`（09-27 那份是 B1 几何未标准化的第一次运行，已标 superseded）。与 C1 的一致率：B0 1.0（按构造）、Bgeo+ HGB 0.9961（被选为对照）、B1 0.9915、B2 0.8335、Bprior 最好的变体 0.8011。Gate R1 演练 B1、B2 都 go=False。这只证明流水线能跑、"伪标签是几何的函数"成立，不证明任何臂绑得更准。
- **阶段 2 通路**：`--labels R`（只读训练折医生标签）+ `--init-from`（每折从自己的阶段 1 模型与配置出发微调）+ `eval --labels R --unblind` 在 160 病灶合成封存数据上端到端测试通过；训练时不读测试折（access log 不存在）。
- **用户在本轮拍板**：先训练、后找医生（P1）；医生阶段两轮——pilot 判升级规则、全集封存、训练折微调、封存折终测一次（P2）；训练池 = 1297 注册病灶（P3）；疾病层 D = 整个检查一个印象，真值用 fastMRI+ 研究级标签（P15）；训练期六个子项目 S1–S6 及顺序（P16，v2.6 §25 第 11 项）。

**之前（保留）**：Level R 读片工具已合 main（tag `handoff/2026-09-26-level-r-tooling`、`handoff/2026-09-26-level-r-fields`），612 → 现 686 测试；冒烟服务 8791；浏览器验收十步仍是 USER_REPORTED（`docs/verification/2026-09-26/level_r_smoke.md` §7）。Gate 0 / H1 重跑 / Gate 0.5 见 `docs/verification/2026-09-24/REPORT.md`；09-23 之前的证据链不变。

## 2. 待用户拍板

- **push**：main 自 afe641e 起所有提交与 tag（09-26 两个、09-28 一个）都只在本地。
- **读片人**：两位读者、一位裁定人、对外方式（端口转发 / `--bind 0.0.0.0`）、伦理备案；医生推迟到训练后（P1），但 pilot 150 是医生阶段第一步。
- **Level R 读片说明里 "other" 的定义**：表单的"其他"比特征表第 7 槽 `other_deep_grey`（海马/杏仁核/腹侧间脑）宽，发 token 前要在说明里写明（终审 minor 9）。
- **S2 开工**：脑侧小病灶检测器规格待写（下一步第 1 条）。
- **nnDetection 膝侧第二臂（S6）**：何时起、用哪张 GPU（GPU 0 常被别的会话占满，CPU 训练在本轮更快）。
- 旧遗留仍挂：Q9 删除授权、其余 4850 卷 SynthSeg、Redivis token、两条远端评审分支去留、RSS 左右手性换算规则。

## 3. 下一步

1. **S2 规格**：脑侧小病灶检测器（真值 = 1297 个框，正常检查作阴性），走 brainstorming → spec → plan。
2. S3（B3/B4，PR-D）在 S1 的表与训练框架上扩展；§12.8 中期分析做不做，在 S3 读片前定（P14）。
3. S4 脑侧解剖层（自有 FLAIR 解剖分割器、脑叶、A_local_quality 验证），先做 SynthSeg 皮层分区可行性探针。
4. S5 疾病印象 + 整句拼装（依赖 S2）；S6 膝侧 nnDetection 另写计划。
5. 读片：用户在 8791 做浏览器验收 → 正式部署（`docs/level_r_tool.md`）→ pilot 150 → 判 §10.1 升级 → 全集 → 封存 → 阶段 2（`run_relation_baselines.py --labels R --init-from runs/relation/c1_stage1_v2`）→ `eval --labels R --unblind` 只跑一次。
6. S1 follow-ups（终审分流为不阻塞合并，见台账）：dx/dy/dz 未按 30 mm 封顶（改就是表 v2）、`side_of` 兜底 EDT 用体素距离、只有精确 McNemar（缺 mid-p）、REPORT 缺非平局选择与数据/折小节、tables.csv 缺标签源列、病灶 326/327 是 merge 串成的大对象（读片前看 `merged_lesions` 的串联）、若干纯外观重复代码。

## 4. 坑与别重做

- **C1 上的任何数字都不是证据**：C1 是 §11 几何量的确定函数，Bgeo+ 近乎原样复现它；只用伪标签训练的模型约等于 B0（P2 的理由）。
- `derived/relation/v1` 拒绝覆盖；要改表就建 v2。`runs/` 不入库；阶段 1 模型在 `runs/relation/c1_stage1_v2/models/`，阶段 2 要从它 `--init-from`。
- 训练默认 CPU 16 线程：计划校验时共享 GPU 0 被别的会话占到 99%，比 CPU 慢 5 倍。用 GPU 前先看 nvidia-smi。
- B1 的几何在每次拟合里用该次训练行的均值/方差标准化，统计量不存进 .pt；阶段 2 用同一批患者重算，结果一致。
- eval 的 `--out` 已存在就拒跑；run 与 table 的 manifest 哈希不一致也拒跑。
- `--labels R` 的 eval 是一次性的，每次读测试折都写 access log。
- 同一工作树别开两个控制方：本轮 claude-63 与本会话曾同时驱动同一计划（Task 9–10 重复派发，提交未重复），已由 claude-63 停手。
- haiku 实现者会编造测试输出与提交信息（Task 1、3），并且会听从 harness 提示加 Co-Authored-By；控制方必须自己重跑测试、检查 `git log -1 --format=%B`。
- 读片工具的坑（冒烟库/正式库两套、`init` 迁移、`export` 时间戳目录、屏幕左右等）同上一轮，见 `docs/level_r_tool.md`。
- 测试不读 `/data2`。

## 5. 关键决定的为什么

- **先训练后读片**：模型在医生标签存在前冻结，封存与预注册自动满足。
- **医生阶段两轮**：不留微调，Gate R1 按构造 NO-GO（C1 是几何函数）。
- **特征表先建、所有臂读同一张表**：公平由构造保证；换标签源只换一列。
- **类级 C1**：候选集就是 7 个宿主类；与标签级查表的 6 处差异都在交界处，逐条记录。
- **B1 几何标准化（终审 I2）**：规格承诺 B1 与 Bgeo+ 看同样的数，Bgeo+ 是标准化的；B1 是阶段 2 微调与 S3 的基础，冻结前改代价最小。
- **CPU 训练**：实测比被占用的共享 GPU 快。
- **不删 SDD 工作区与 scratch**：用户规则，删除一律由用户执行。
