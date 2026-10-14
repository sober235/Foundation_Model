# 新 U 损失训练 C2：运行约定（2026-10-14 14:5x CEST，启动前写定）

三目标修正计划 `docs/superpowers/plans/2026-10-13-anatobind-aur-three-goal-conformance.md` 的 Task 7。和新 C2（决定 Q25，`eval/c2_test_20261011_0500/`）逐项相同，只改一个变量：U 掩膜损失从 `legacy` 换成 `unmatched_empty_v1`（U 监督样本里每个未匹配查询也学空掩膜，× NO_OBJECT_WEIGHT 0.1）。C0 配对按 Q29 是可选的描述性分析，本轮不跑。

## 输入与代码

- 代码：分支 `codex/aur-three-goal-conformance-2026-10-13`，提交见 `code_sha.txt`（56f65f7：Task 1–4 + 评审修复，全量测试 1056 passed、2 skipped）。
- 输入的 SHA-256 见 `input_sha256.txt`：样本表 `samples_1mm_v1.json`、验证病人 `ssl_manifest_v1/val_patients.json`、Stage I 导出 `ssl_runs/stage1_r2_320k_20261010_1303/ssl_stage1_best.pt`（与新 C2 的 `run_config.json` 记录的同一文件）、G1 报告 `g1/r2_run_20261010_1942/g1_report.json`。

## 设置

```
ARM=c2 CARDS=0,2 U_MASK_PROFILE=unmatched_empty_v1 C2_STEP1000_GATE=0 II_BB=5e-5 SEED=0 GLOBAL=24 WORKERS=24
INIT_BACKBONE=/data2/congcong/data/FM_data/derived/aur/ssl_runs/stage1_r2_320k_20261010_1303/ssl_stage1_best.pt
G1_REPORT=docs/verification/2026-10-09/anatobind_brain_ssl_first/g1/r2_run_20261010_1942/g1_report.json
bash scripts/stage23_chain.sh      # 在工作树 foundation_model-conformance 里跑，tmux 会话 anatobind 窗口 c2u
```

两卡、每卡 12、累积 1、全局 24；Stage II 24 万裁块，主干 / 头学习率 6.124e-5 / 6.124e-4，预热 667 步；Stage III 8 万裁块，6.124e-5 / 3.062e-4 / 6.124e-4，预热 133 步；每 500 步在 32 卷上验证。卡间空档由 `ops/card_guard.py` 守（窗口 guardc2u）。

## 判定（预先登记）

- 验证集选 U 阈值，测试集 1,056 行只评一次。
- 最终门不变：G2-A 13 宿主宏 Dice ≥ 0.80；G2-U 各来源在 S7 参照的误报工作点上，灵敏度 ≥ 参照 − 0.05；G3 受控轨道 ABA(R) ≥ ABA(B0)。真值全是 SynthSeg 与病灶伪标签，数字标 NOT_EVIDENCE。
- 没有早停门：Q25 的第 1000 步阈值属于 legacy 配方，不套用；本轮也没有另行登记新门。
- 任一门不过：如实记录，按计划停下，另写新计划补缺口；不在测试集上调参，不追加未登记的改动。
- 描述性对照：用 `scripts/aur_compare.py` 与新 C2（legacy）在同一批测试行上比较，只差 U 损失这一个变量。新 C2 越过过一次不过的预先登记检查，那一臂的数字本来就只作探索性结果。
