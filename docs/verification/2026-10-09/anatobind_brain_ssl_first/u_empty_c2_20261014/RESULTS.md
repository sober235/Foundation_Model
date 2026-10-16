# 新 U 损失 C2：全测结果（2026-10-16 收尾）

**结论：G2-A、G2-U、G3 全部未通过。全部数字均为 `NOT_EVIDENCE`。** 本轮不在 test 上调参或追加未登记改动；进一步模型修改需要新计划。训练、验证选阈值、全测与预注册的 legacy C2 描述性配对比较均已收尾，Task 7 完成；本轮不需要新训 C0。

## 运行与依据

启动前约定和文件哈希分别保存在 [README.md](README.md)、[input_sha256.txt](input_sha256.txt)；代码 SHA 见 [code_sha.txt](code_sha.txt)：`56f65f71491cb1940c496f68c5427fc678cbe10d`。现有全量 CPU 测试记录为 1056 passed、2 skipped；此次文档收尾未重跑测试、训练或推理。

| 项目 | 实际产物 |
|---|---|
| profile / seed | `unmatched_empty_v1` / 0 |
| Stage II | `ssl_runs/c2_stage2_240k_20261014_1344`，10,000 步 / 240,000 crops |
| Stage III | `ssl_runs/c2_stage3_80k_20261014_1344`，3,334 步 / 80,016 crops |
| 验证集 | `eval/c2_val_20261014_1344/REPORT.md`，405 行 |
| 测试集 | `eval/c2_test_20261014_1344/REPORT.md`，全部 1,056 行 |
| 链日志 | `/data2/congcong/data/FM_data/derived/aur/ssl_runs/c2_chain_20261014_1344.log` |
| 完成时间 | 2026-10-15 **01:56:58 UTC**（链日志 02:56:58 BST，UTC+1） |

`ssl_runs/` 相对数据根 `/data2/congcong/data/FM_data/derived/aur/`；`eval/` 相对本记录索引目录。两张卡 0、2，每卡 microbatch 12，全局 batch 24；学习率、预算与清单按启动约定执行。链日志先记录 val 评估，随后记录验证选阈值 `bmsr=0.70 isles=0.60 pdgm=0.75`，再记录 test 使用固定阈值。原始公开汇总报告：

- [test REPORT](../eval/c2_test_20261014_1344/REPORT.md)。
- [val REPORT](../eval/c2_val_20261014_1344/REPORT.md)。

本轮不继承 legacy 的 Q25 step-1000 早停门，`C2_STEP1000_GATE=0` 是运行前约定；没有另外登记新早停规则。最终三门保持不变，不能把这个设定描述成越过失败检查。

## 最终三门

| 门 | test 结果 | 门槛 / 参照 | 判定 |
|---|---|---|---|
| G2-A：13 宿主宏 Dice | 0.7988，1,032 个 A 监督卷 | ≥ 0.80 | FAIL，差 0.0012 |
| G2-U：PDGM FLAIR | 灵敏度 0.6538；FP/scan 0.218；阈值 0.75 | 参照 0.8615，要求 ≥ 0.8115；FP budget 0.228 | FAIL |
| G2-U：BMSR T1c | 灵敏度 0.0146；FP/scan 0.098；阈值 0.70 | 参照 0.7316，要求 ≥ 0.6816；FP budget 0.510 | FAIL |
| G2-U：ISLES DWI | 灵敏度 0.0242；FP/scan 1.120；阈值 0.60 | 参照 0.6303，要求 ≥ 0.5803；FP budget 1.460 | FAIL |
| G3：受控 R | ABA(R) 0.5606，对 B0 0.7545；1,336 个实例 | ABA(R) ≥ ABA(B0) | FAIL，R − B0 = −0.1939 |

A 全实体宏 Dice 0.6460；13 宿主各来源宏 Dice 为 BMSR 0.8062、ISLES 0.7527、PDGM 0.8039、SibBMS 0.7957。来源分项不替代整体 A 门。U 门只读 BMSR T1c、ISLES DWI、PDGM FLAIR；PDGM T1c 不是额外判门来源。BMSR/ISLES 的 AUR 1 mm 重采样网格与原生参照不同，原报告标为 approximate。

受控 R：R 救回 202、弄错 461、两者均对 547、均错 126。识别–绑定 gap 子集 48 个实例，救回 9、弄错 6；端到端已检出子集 120 个病灶，ABA(R) 0.7417 对 B0 0.7250。这两个子集仅报告，不能改变受控 G3 的失败判定。序列类型准确率 0.9981/1,056 卷只报告。

验证集 A 0.8035 达到 0.80，不能据此覆盖 test A 0.7988；验证集 U、受控 R 仍未过。模型成功按最终 test 的既定三门判定。

## 与 legacy 新配方 C2 的描述性配对（2026-10-16）

历史臂为 `c2_test_20261011_0500`（`legacy`），不是新的同 profile C0 对照；本轮没有新训 C0。启动约定是保持其余 recipe，只改未匹配 query 的空 mask BCE。**历史臂曾在 Q25 step-1000 检查 FAIL 后继续运行，结果只能作探索性参照。** 配对使用既有 test 逐例结果在 CPU 计算，覆盖同一 1,056 行，2,000 次患者级 bootstrap（seed 0）。原始完整报告见 [COMPARE.md](../eval/compare_c2empty_c2legacy_20261016/COMPARE.md)，结构化汇总见 [compare.json](../eval/compare_c2empty_c2legacy_20261016/compare.json)。区间只描述这两个已训练模型的患者抽样差异，不能当作已验证的因果效应或跨种子稳定性。

| 指标 | 本轮新 U 损失 C2 | legacy 新配方 C2 | 配对差（本轮 − legacy） | 95% 患者 bootstrap 区间 |
|---|---|---|---|---|
| A 13 宿主宏 Dice | 0.7988 | 0.7961 | +0.0027 | [0.0016, 0.0040] |
| U PDGM FLAIR 灵敏度 | 0.6538 | 0.7538 | −0.1000 | [−0.1575, −0.0462] |
| U BMSR T1c 灵敏度 | 0.0146 | 0.0049 | +0.0097 | [−0.0016, 0.0242] |
| U ISLES DWI 灵敏度 | 0.0242 | 0.0069 | +0.0173 | [0.0038, 0.0347] |
| 受控 ABA(R) | 0.5606 | 0.5142 | +0.0464 | [0.0050, 0.0861] |
| 受控 ABA(B0)（只列点差） | 0.7545 | 0.6692 | +0.0853 | — |

U 两臂使用各自验证集选出的工作点；本轮阈值为 0.70/0.60/0.75（BMSR/ISLES/PDGM），legacy 三源均为 0.70。因此这些是各自工作点的描述性差异，不能直接当作等误报灵敏度提升；最终 G2-U 仍单独按既定 S7 参照 FP 工作点规则判定。端到端 R 的检出子集和数量也会变化，不能据其变化推断全体关系能力。

U 工作点的实际 FP/scan（本轮 / legacy）：BMSR 0.098 / 0.118、ISLES 1.120 / 0.180、PDGM FLAIR 0.218 / 0.158。ISLES 的灵敏度点差伴随明显更多误报，不能写成等误报性能提升。R 虽比历史臂高，但 B0 同时提高更多，当前 R 仍低于自己的 B0，G3 仍失败。PDGM T1c 的非判门比较见完整报告，不加入新门。

**描述性对照与保存 val 输出核查已完成；失败机制仍未定。** 不声称压空监督解决了原先吞并病灶的问题，也不因区间未跨零改变任何最终门结论。每臂只有一次运行，旧臂 Q25 偏离与阈值差异都保留披露。

## 证据界限与报告措辞

A 和 R 的内部参照来自 SynthSeg/重叠查表，不能作为医生关系真值；U 使用当前标签覆盖的异常，不能声称能识别任意未知异常。Level R 尚未人标，未通过 R0/R1 或临床验证。fastMRI 433 卷与 SibBMS 特定检查不在此次报告中，不能假定已经完成。

val REPORT 保留了旧模板措辞：标题写 “test-set evaluation”，选择列写 “test-selected”。这与它所在的 val 路径、405 行规模及链日志的先 val 后 test 流程不符；test REPORT 明确写 “fixed (chosen on the validation split)”。这是已知的报告模板措辞缺陷，保留原始报告并在此注释；**这些字样本身不是发生 test 泄漏的证据**。本轮只依据已有记录描述实际流程，不把模板修辞当作独立泄漏审计。

病例级表、病例标识及盲读材料保留在本地；不复制或提交 `cases.json`、`level_r_key.json`、`level_r_sheet.csv`。本记录只收汇总值与无病例标识的原始 REPORT。

## 保存的 val 输出核查及下一步边界

后续汇总核查见 [匿名 val 诊断](../../../2026-10-16/aur_saved_eval_diagnostic/REPORT.md) 和 [尚未执行的机制诊断安排](../../../2026-10-16/aur_saved_eval_diagnostic/NEXT_STEPS.md)。既有新 C2 输出在 score 过滤前仅有 BMSR 16/470、ISLES 6/138 个 GT 框被任一已存连通块以 box IoU ≥ 0.1 覆盖，说明仅降 score 无法修复主要漏检。最大连通块中位分别 473.7/350.4 mL，全部对应扫描均超过 100 mL；这些是描述统计，不是新门。框包含不能证明体素 mask 吞没病灶。另一 oracle 指标下，漏检病灶 query mask IoU 中位分别 0.00216/0.00319；它使用 GT 中心裁块、全部 query、无 presence 过滤、mask > 0.5，与整卷 mask > 0.3 后按 box IoU 匹配的正常检测不同，不能直接等同或据此定位具体根因。

该核查仅重算保存的验证结果，没有新推理或阈值搜索，根因尚未确定。拟固定每源 2 个 val 病例保留 query/窗口输出，以区分单 query、presence、合并的损失；该步骤尚未执行，不能把拟做的诊断写成已证实结论。

## 可粘贴的只读复核命令

```bash
E=docs/verification/2026-10-09/anatobind_brain_ssl_first/eval
cat "$E/c2_test_20261014_1344/REPORT.md"
cat "$E/c2_val_20261014_1344/REPORT.md"
cat "$E/c2_test_20261011_0500/REPORT.md"
tail -n 14 /data2/congcong/data/FM_data/derived/aur/ssl_runs/c2_chain_20261014_1344.log
cat "$E/compare_c2empty_c2legacy_20261016/COMPARE.md"
```

比较的完整执行来源、输入 SHA256 和复现命令见 [RUN.md](../eval/compare_c2empty_c2legacy_20261016/RUN.md)。隔离副本只含公开汇总，不能用其相对 `eval/` 路径寻找未复制的原始逐例输入。按 RUN.md 从仓库根使用原始绝对路径和新的临时输出目录复现（从既有结果计算，不做新推理）：

```bash
CUDA_VISIBLE_DEVICES='' PYTHONNOUSERSITE=1 PYTHONPATH=. OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 nice -n 19 /home/congcongliu/anaconda3/envs/nvgen/bin/python scripts/aur_compare.py \
  --a /data0/congcong/code/Project_Doing/foundation_model-conformance/docs/verification/2026-10-09/anatobind_brain_ssl_first/eval/c2_test_20261014_1344 \
  --b /data0/congcong/code/Project_Doing/foundation_model/docs/verification/2026-10-09/anatobind_brain_ssl_first/eval/c2_test_20261011_0500 \
  --name-a C2-empty --name-b C2-legacy \
  --out /tmp/compare_c2empty_c2legacy_20261016_reproduction --n-boot 2000
```

复现输出目录必须不存在；若该示例目录已存在，改用另一个新目录，不覆盖已有产物。原始运行的输出是 `eval/compare_c2empty_c2legacy_20261016/`，其中结果已完成并保留。

原始汇总输出由上述 REPORT 保留；链尾关键输出（时钟 BST）为：

```text
[2026-10-15 01:04:53] U thresholds from the validation split: --u-threshold bmsr=0.7 --u-threshold isles=0.6 --u-threshold pdgm=0.75
[2026-10-15 01:04:53] evaluation on the whole test split, 2 shards on cards 0,2 -> docs/verification/2026-10-09/anatobind_brain_ssl_first/eval/c2_test_20261014_1344
[2026-10-15 02:56:58] evaluation finished: docs/verification/2026-10-09/anatobind_brain_ssl_first/eval/c2_test_20261014_1344/REPORT.md
[2026-10-15 02:56:58] c2 chain done
```

## 下一步

当前预注册的描述性比较与保存 val 输出核查已完成。按 NEXT_STEPS 的边界提出进一步机制诊断和新的 A/U/R 修正计划并审阅；尚未执行 query/窗口级诊断。当前训练计划在失败结论处停止，不在 test 上选改动；新的诊断还没有完成。DICOM 几何解码与离线验收继续保留为模型任务，单张 DICOM 的独立 2D 模型、数据与验收不得遗忘；网页、API、公开 URL 根据 Q28 放在后续阶段，不因本轮三门失败转去继续做显示。
