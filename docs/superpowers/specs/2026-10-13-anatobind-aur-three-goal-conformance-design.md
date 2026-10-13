# AnatoBind-Brain A/U/R 三目标执行修正设计

**状态：2026-10-13 复核修订；待用户复核本次 Q25/profile 边界更改**

**范围：** 对齐已批准的脑部 A/U/R 规格和 SSL-first 执行决定；不扩展研究目标。

**当前代码基线：** `main` / `origin/main` at `d31dd5a`。

## 目标与决策优先级

本设计只执行三个既定输出目标：

1. **A — Anatomy：**输出 SynthSeg 32 个非背景解剖结构的 mask；主门仍为 13 个分侧宿主的平均 Dice ≥ 0.80。
2. **U — Abnormality：**将所有已标注病灶作为一个通用异常类别，输出实例 mask；保持按来源、在 S7 参照误报工作点评估灵敏度的既定门槛。
3. **R — Relation：**对每个 U 实例输出一个 13 类分侧主宿主或“无宿主”；保持现有主宿主任务和 G3 指标。

脑部规格 N2/N7/N8/N14 定义 A/U/R 的输出与局部门槛。2026-10-09 SSL-first 决定覆盖 2026-10-08 规格中跳过 Stage I 的旧顺序，继续遵循严格 `Stage I → II → III`。本设计不加入疾病分类 D、采集可靠性 E、邻近/累及等次级关系、其他器官或新的主要终点。

G1 表征探针与 Q25 step-1000 检查是训练质量/执行门，不是第四个模型目标；G1 通过不能替代 A 输出门。

## 现状与边界

- 新配方 C0 当前仍按旧 U mask loss 运行；检查时已有 step 3260 记录。该运行保持原状，作为旧目标配置的结果。
- Q25 新 C2 的 step-1000 条件是冻结宿主读出 ≥ 0.65 且 A point Dice > 0.045。历史检查脚本只打印 verdict，失败没有阻止训练链继续。C2 后续全测仍标为越过失败条件后的探索结果。
- U mask loss profile 与 Q25 gate 是分开的 run 配置：`U_MASK_PROFILE=legacy|unmatched_empty_v1` 控制 U loss，`C2_STEP1000_GATE` 控制 Q25 执行门。run config 必须同时记录二者；Q25 的 step-1000 数值只适用于 `U_MASK_PROFILE=legacy` 的原 Q25 C2 recipe。
- 现有 G2/G3 使用 SynthSeg 与病灶伪标签，所有此类指标继续标记 `NOT_EVIDENCE`。Level R 人工读片仍是独立验证依赖。
- 实现和后续训练使用隔离分支，不改 `/data0/.../foundation_model` 上正在运行的 C0，也不暂存或覆盖其未跟踪评估文件。

## 设计

### 1. Q25 C2 门控必须 fail-closed

只对显式启用 Q25 step-1000 gate、`ARM=c2`、`II_BB=5e-5` 且 `U_MASK_PROFILE=legacy` 的 Stage II run 使用该门，不把这两个数值暗中推广为所有模型的通用门。`U_MASK_PROFILE=unmatched_empty_v1` 是新的训练目标配置，不继承旧配方的早停阈值；该配对实验按既定 G2/G3 最终门评估。若要为新 profile 加早停门，必须另行预注册并审阅后再启动。

1. 仅在 `ARM=c2`、`U_MASK_PROFILE=legacy` 且显式启用 Q25 gate 时，Stage II 首次调用以 `--stop-after 1000` 运行，写出验证记录和 `resume_step1000.pt` 后暂停。
2. 参数化的 Q25 检查器从同一 checkpoint 计算冻结 13-host readout，并读取该步验证 A point Dice；脚本本身对 FAIL、缺文件、解析错误返回非零状态。它接收 Stage II run dir、样本清单、验证患者清单和探针卡号，检查产物写入该 run 的 gate 子目录，不依赖临时 scratch 路径。
3. 只有两项均通过才从该 checkpoint 恢复 Stage II。任何条件失败、文件缺失、评估错误或解析错误都返回非零状态；链立即停止，不进入 Stage III 或测试集评估。
4. 通过后第二次调用从 `resume_step1000.pt` 恢复，并省略 `--stop-after`；沿用同一 optimizer、scheduler、数据游标、global batch 和预算。`train.py` 的 schedule 不含 invocation-local `stop_after`，现有恢复路径已覆盖“有限步暂停后继续完整 schedule”。门结果、阈值和检查产物写入 run metadata。
5. 验证阈值只取自 validation。测试集不参与门判断、调阈值或修复选择。

验收：脚本级回归测试覆盖 PASS 恢复、FAIL 停止、缺少 checkpoint、读取失败；FAIL 路径断言 Stage III 和测试评估没有启动。

### 2. U 修复只补未匹配 query 的空掩膜监督

保留 64 个匿名 event queries、匈牙利匹配、当前实例输出格式和推理合并路径。`U_MASK_PROFILE=legacy` 完全保留旧 mask loss；新 profile `U_MASK_PROFILE=unmatched_empty_v1` 在 **U 已监督样本**中给未匹配 query 添加空掩膜 BCE：

- 匹配 query：现有目标 mask BCE + Dice 不变。
- 未匹配 query：目标 mask 为全零；每个 query 只在 `point_weight` 有效采样点上取平均 BCE，再对所有未匹配 query 取平均；将该项乘现有 `NO_OBJECT_WEIGHT = 0.1` 后加到现有 matched-mask loss，不新增可调超参数。
- `u_supervised=False` 的样本：整个 U loss 仍为零。
- 存在性 BCE、采样配额、匹配规则、推理阈值和跨窗合并在首轮修复中保持不变，避免把多个变化混在一起。

验收：loss 回归测试覆盖未匹配 query 向空 mask 学习、匹配 query 的既有监督、无 U 监督样本零 loss，以及 `point_weight=0` 点不产生梯度。

### 3. 重新评估必须保持 A/U/R 对照可解释

当前 C0 作为 `U_MASK_PROFILE=legacy` 运行继续完成。代码修正后，新的配对实验中 C0 和 C2 都显式使用 `U_MASK_PROFILE=unmatched_empty_v1`，并匹配数据清单、global batch、Stage II/III 预算和阈值选择协议。该新 profile 不复用旧 Q25 早停门；validation 选 U 操作点，最终按原 G2-A、G2-U、G3 门评估，封存 test 只评估一次。若希望给新 profile 加早停门，必须在启动前另行预注册并审阅。所有伪标签数字仍标 `NOT_EVIDENCE`。

R 的输出空间和 G3 指标不变。Level R 的读片工具/盲法清单可准备，但人工真值、reader agreement 与论文级 R1 结果必须等实际读者完成；本设计不生成或伪造人工标注。

## 文件边界

- 修改 `anatobind/aur/losses.py`：按 U loss profile 选择旧 loss 或 U 未匹配 query 空 mask BCE。
- 修改 `anatobind/aur/train.py` 与 `scripts/aur_train.py`：显式传递并记录 `U_MASK_PROFILE=legacy|unmatched_empty_v1`；默认保留旧行为。
- 修改 `scripts/stage23_chain.sh` 与 `docs/verification/2026-10-09/anatobind_brain_ssl_first/ops/c2v2_check.sh`：Q25 gate 只对 `U_MASK_PROFILE=legacy` 参数化 checkpoint 路径并 fail-closed；禁止新 U-loss profile 误用旧 gate。
- 修改对应现有测试文件：`tests/test_aur_losses.py`、`tests/test_aur_training_contract.py` 及脚本门控测试。
- 更新 `STATUS.md`：记录实际运行状态、代码门控语义、旧 C2 的协议偏差及伪标签边界；不编辑运行中的 checkpoint 或逐例盲读文件。

## 非目标

- 不修改 A/U/R 类别定义、模型输入、训练/测试 patient split、主要评估指标或输出接口。
- 不替换 U query 解码器，不改 score threshold，不针对 test 集调参。
- 不把旧 Q25 早停阈值套到 `U_MASK_PROFILE=unmatched_empty_v1`；不为新 profile 事后发明早停门。
- 不重写 R 架构、不新增病种头、E 或次级关系。
- 不停止、重启或覆盖当前 C0；不在无人审阅时启动新的 GPU 训练链。
- 不把伪标签结果写成临床验证或 v2.6 R0/R1 完成。

## 完成判据

1. Q25 gate 的所有失败路径均 fail-closed；通过路径保持原始训练 schedule 并可续跑。
2. U 未匹配 query mask 的梯度和监督门控符合本设计，已有 matched-query 行为不回退。
3. A/U/R 的类别、输出和任务门槛无变化；新旧运行及证据边界可区分。
4. 状态文档只反映已观测事实；当前 C0 结果及 Level R 人工依赖不被误报为完成。

## 依据

- `docs/superpowers/specs/2026-10-08-anatobind-brain-aur-design.md`：N2、N7–N14、§7、§11。
- `docs/superpowers/plans/2026-10-09-anatobind-brain-ssl-first-execution.md`：Q16、Q22、G1/G2/G3。
- `docs/verification/2026-10-09/anatobind_brain_ssl_first/stage2/STAGE2_V2.md`：Q25 step-1000 条件。
- `docs/verification/2026-10-09/anatobind_brain_ssl_first/ops/c2v2_check.sh`：历史检查器只报告 verdict 的实现。
- `anatobind/aur/losses.py`、`anatobind/aur/infer.py`：U loss 与 query mask 推理路径。
- `STATUS.md` 与 C0/C2 全测报告：运行状态、数值和 NOT_EVIDENCE 边界。
