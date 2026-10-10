# 评估栈真实数据冒烟（2026-10-10 14:10–15:32 本机，CPU）

NOT_EVIDENCE：模型是 C0 Stage II 第 2000 步导出（`ssl_runs/c0_stage2_240k_20261010_0903/aur_stage2_step2000.pt`，A 头未学好，见 `../../stage2/C0_A_DIAG.md`；R 头在 Stage II 冻结、未训练），数字只说明通路走通，不说明模型好坏。

目的：在 C0 链真正走到评估之前（约 10-11 上午），用真实数据把链脚本的评估路径完整走一遍，并顺带跑两个不在链里的报告。脚本 `eval_smoke_cpu.sh`（CPU、nice 19、每源 1 例），完整输出在 `/data2/congcong/data/FM_data/derived/aur/eval_smoke_20261010/`，这里只存四份报告与日志。

| 步骤 | 内容 | 用时 | 结果 |
|---|---|---|---|
| 验证集 | 2 片（`aur_eval.py --shard k/2 --limit 1`）+ 合并（`aur_eval_merge.py`），4 个体积 | 974 s | 走通；U 阈值 PDGM 0.65；序列类型 4/4 |
| 测试集 | 同上，带 `--u-threshold pdgm=0.65` | 1487 s | 走通；阈值从验证集传过去 |
| fastMRI | `aur_eval_fastmri.py --limit 1`，与 S4 学生模型并排 | 69 s | 走通 |
| SibBMS | `aur_eval_sibbms.py`，10 个 MS 受试者（3 个在测试集） | 2342 s | 走通 |

读法与要回头看的：

- A：验证集 13 宿主宏 Dice 0.066、测试集 0.073；fastMRI 白质、皮层近 0（S4 学生模型约 0.8）→ 诊断在 `stage2/C0_A_DIAG.md`。
- R：受控与端到端都是 ABA(R) 0 对 B0 1，与"Stage II 不训 R"一致，不是评估缺陷。
- U：SibBMS 斑块敏感度近 0（MS 斑块从来不是 U 的训练目标）。阈值 0.05–0.30 的扫描结果完全相同，可能是病灶概率图两极分布；正式模型出来后要核对。
- CPU 上整卷推理每例约 2.5–8 分钟；链里用 GPU，不受影响。
