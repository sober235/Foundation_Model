# A/U/R 建模机制诊断：六例验证集回放

**NOT_EVIDENCE。诊断阶段已完成模型回放；A/U/R 模型门仍未通过，没有改动模型或重新训练。** 本记录与下面的分析只使用冻结的 validation 病例及历史 validation 输出。六例不能代表人群性能。

## 运行与可复现性

- 固定 BMSR/T1c、ISLES/DWI、PDGM/FLAIR 各两位不同验证患者，按设计的身份元组 SHA256 排序选例；看图/推理前冻结。私有身份清单及输入文件哈希只保存在本地运行目录。
- 代码 `b240962`；checkpoint、样本表、验证患者、身份清单及保存评价输入的 SHA256 在 [RUN.json](RUN.json)。观察器只记录原计算，不修改模型输出。
- 一张空闲 A800（物理 GPU 2），batch 2，原 bf16、crop `[128,160,160]`、presence >0.5、mask ≥0.3、原 Gaussian 平均及组件评分。>0.5 只用于另列的历史 oracle 对照。
- **6/6 卷的原 U 组件数、框、体素数、体积及 score 复现；55/55 受控 R 实例集合、宿主、概率、query IoU 复现。** GT 实例/ignored 几何也已核对。282 项保存 A 分数与当前回放完全相同。
- 83 个自然滑窗，55 个 GT 病灶（其中 33 个来自一卷）；180.31 秒，峰值 CUDA 分配 9.018 GiB。硬超时 7200 秒未触发。局部原始数组约 13 GiB，留在 `/home/congcongliu/outputs/aur_model_diagnostic/20261017_replay_b240962/`，不提交逐例文件。
- 全部 query 的 presence、体积、GT 交集/IoU、病灶内平均概率与 0.3/0.5 的 packed hard masks 均保存；至多八个代表 query 的浮点缓存是 float16，仅供核查，不用于决定阈值或复现成绩。

## 工程核验

独立代码评审的三条 Important 已修复：完整 query 证据、valid-only 解剖评分、严格 GT/R 集合及输出核对。新增回归测试分别观察到 RED 后转 GREEN；原推理的观察器开/关在小型真实模型上输出一致。

完整 CPU 测试：**1065 passed、2 skipped，155.20 秒**，CUDA 隐藏。首次沙箱内测试因 DataLoader 共享张量所需的本地 socket 被禁止而卡住；同一测试沙箱外通过，随后完整套件也在沙箱外通过。没有改动训练器或多进程实现来绕过环境限制。

```bash
timeout 600 env CUDA_VISIBLE_DEVICES='' PYTHONNOUSERSITE=1 PYTHONPATH=. \
  OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 nice -n 19 \
  /home/congcongliu/anaconda3/envs/nvgen/bin/python -m pytest tests/ -q -p no:cacheprovider
```

## 保存的完整验证集给出的 A/R 线索

独立 CPU 重算 405 行保存输出（394 行 A 监督、716 个 controlled R 实例）与原 aggregate 的数值差不超过 `2.3e-16`，计数完全一致。完整匿名分项、复现脚本和输入哈希见 [保存 val 的 A/R 分析](../../2026-10-16/aur_model_diagnostic/saved_val_ar/REPORT.md)。

- 五个 A 标签 5/44/18/26/58 在全部有限 A 监督卷上 Dice 为零，不能用 13-host 平均接近 0.80 隐去该缺口。
- 604/716 的历史选中 query IoU <0.1，其中有 197/215 次 R harm。这个 IoU 是整个 query mask 的 oracle 指标；断开的假阳块可能压低它，不能直接推断 query 中不存在可用病灶组件。
- 全部 716 个 `in_crop_share=1`，没有观察到该字段定义下的 GT 实例截断。高整卷 local A Dice 也未消除 R harm，但它不是 R 实际粗网格输入的评分。
- R 把 158/215 次 harm 判为皮层；白质真值 180 个，R 正确 32 个、B0 正确 142 个。来源、大小、query 质量混杂，不能只据这个分组归因。

## 六例机制分析

### U：已经观察到实例在 max 合并中受损，但不能把其他漏检全归为训练失败

完整源级汇总、精确 CC oracle 与加权重建校验见 [U 缓存分析](u/REPORT.md)、[结构化汇总](u/public_summary.json)。

55 个 GT 都至少有一个自然窗口完整包含它，不能用“每个窗口都截断病灶”解释以下现象。主分析保持 mask ≥0.3；对所有 64 query 的连通块做精确局部 oracle，GT 只用于事后测量，不参与原模型解码。

| 在任一完整可见自然窗口找到 mask IoU ≥0.5 的病灶 | 全部 query 的单独连通块 | presence 保留 query 的单独连通块 | 窗内 max 图的连通块 |
|---|---:|---:|---:|
| BMSR（13 个 GT） | 5 | 4 | 2 |
| ISLES（40 个 GT） | 11 | 9 | 3 |
| PDGM（2 个 GT） | 2 | 2 | 2 |
| 合计（55 个 GT） | 18 | 15 | 7 |

- 12/55 个 GT 存在**同一完整窗口**中“保留 query 的组件 IoU ≥0.5，但 max 后最佳组件 IoU <0.5”的情况（45 个 GT-window 观察）。其中 8/55 在其他完整窗口也找不到好的 max 组件。
- 3/55 有好的全部-query 组件，但经过 presence 后，所有完整窗口都失去这样的组件。它是一个确实存在的次要瓶颈，不能据此把主要错误都归给 presence。
- 整张 query mask 的 IoU 会被断开的假阳块拉低。例如已观察到整 query IoU 0.104、其病灶组件 IoU 0.771，但 max 后组件只有 0.00403。因此“52/55 没有整 query IoU ≥0.5”**不证明模型完全没有病灶信号**，也不支持立即改训练或切换 N7。
- 在 ≥0.3 下，跨窗平均压掉 7,047/206,590 个“至少一个窗口支持”的 GT 体素（3.41%）。这是相对乐观 any-window 支持的体素量，不是实例灵敏度，不能把损失简单相加或认为平均没有实例影响。
- >0.5 另列历史桥接分析为 19/19/13（全部 query CC / 保留 query CC / max CC）；不据此把生产阈值改为 0.5。

这些 oracle 使用 GT 选择最佳组件，不能作为可部署输出或 G2-U 成绩。这里的 mask IoU ≥0.5 与正式 box IoU ≥0.1 一对一检测也不是同一指标。仍有许多实例缺少高质量局部组件；六例无法证明仅改解码即可过门。

### A：五个实体是当前实际输出缺失

在六例的 30 个 case × label 组合中，标签 5/44/18/26/58 的真值体素都非零，预测体素都为零。存在性往往很高；更关键的是，正式整卷 A 根本不以 entity presence 筛 mask，所以存在性过滤不是 A 丢掉这五类的直接原因。下一项 A 研究需要检查这些实体的监督点覆盖、mask 概率和身份竞争，不改词表、不删掉小结构。

### R：正式 dense 感知与关系头所见的粗几何不一致

相同网格、固定中心采样、软几何与病灶区域限制详见 [A/R 缓存分析](ar/REPORT.md)、[结构化汇总](ar/public_summary.json)。

55 个绑定裁块中的均值：

| A 输入/测量 | 宿主宏 Dice（仅诊断） |
|---|---:|
| 单裁块 dense A，完整分辨率 | 0.74518 |
| 相同 dense 硬图最近邻采到实际 coarse 网格 | 0.74401 |
| 独立 coarse A，未 gate | 0.04847 |
| 实际 R 使用的 coarse A，presence gate 后 | 0.04869 |

最近邻粗采样有四个小 GT 病灶消失，必须保留这个网格限制；但同网格 dense A 对照仍远好于 coarse A，不能用分辨率本身解释 A 输入差异。gating 平均只改变约 0.424% 的 coarse 硬标签，也不能解释这一量级差距。

三个 dense query IoU ≥0.5 的绑定中，相应 coarse U 的 >0.5 前景占有效 crop 约 99.93–100%。这说明“dense query 好”不能自动转移为“R 的 coarse U 几何也好”。实际粗宿主软 mask 还普遍重叠，进一步限制几何特征的语义解释。

55/55 的局部与整卷 GT 主宿主一致，没有在这六例观察到该目标差异。R 正确 32/55、B0 正确 35/55，12 次救回、15 次弄错；这是小样本描述，不替代 G3。coarse 语义失配是需要控制实验检验的输入问题，**不是已证明导致全部 R 错误的因果根因**。

## 下一项模型实验

优先做**冻结权重、保留 query 身份的 U 解码对照**，检验当前 max 合并丢掉的实例能否以可接受误报保留下来，再决定是否需要训练改动。具体约束见 [NEXT_EXPERIMENT.md](NEXT_EXPERIMENT.md)。该实验尚未实现或运行。

A 的五类缺失与 R 的粗几何失配分别保留为明确的待办；不同时改 A loss、U decoder 与 R geometry。下一项 U 实验必须保留 32 类 A 和每实例 14 类 R 的接口，报告三项结果及 R 的实例来源。当前没有依据宣称任何一项模型门已经通过。

## 目标与证据边界

A 仍必须输出 32 个结构，U 仍为统一异常的独立实例，R 仍为每实例 13 个分侧宿主 + no-host。保留既有 G2-A/G2-U/G3，不用六例、任何 oracle 或较好看的子集替代原门。历史 test 不参与本轮诊断选型，没有新 test 推理或阈值搜索。A/R 伪参照不升级为真人关系真值或临床证据。
