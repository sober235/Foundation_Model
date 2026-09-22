# STATUS:2026-09-22 晚(v2.6 实验设计定稿已合入 main = PR #7 merge `fb25da9`;本交接点 tag `handoff/2026-09-22-v2.6`,上一交接点 tag `handoff/2026-09-22`)

每次交接前整体重写本文件。五段固定:已验证、待拍板、下一步、坑与别重做、为什么。

## 1. 已完成且已验证

**主线:关系问题的第一道门是"独立人标真值上 learned R 是否超过强几何基线"。权威方案 = `docs/plans/2026-09-22-aur-v2.6-experiment-design-route.md`(v2.6);`RESEARCH_PLAN.md` 的 §0/§1/§9.1/§13 是摘要与决策记录;v2.5、v2.4 计划文件只作审计记录。**

- **今天 main 上的三个提交**(交接点 4f86316 之后):6848e1b 落实 PR #6 评审 6 条机械修改并列出 5 项待拍板;8a952ea 冷启动独立评审 `REVIEW_v2.5_feasibility_2026-09-22.md`(有条件 GO;计数脚本与输出在 `docs/verification/2026-09-22-feasibility-review/`);1b8a33e 修评审指出的文档错误(读片工时"各 2600"实为合计、AC1 0.6≈raw 0.65 与 raw 0.90 不等价、XGBoost 措辞)。
- **外部 review 及核验**:用户粘贴的外部 review 存为 `REVIEW_external_experiment_design_2026-09-22.md`(原文 + 逐条核验 + 取舍)。核验脚本与原始输出在 `docs/verification/2026-09-22-external-review-check/`:
  - `patient_id_check`:252 卷有框 FLAIR 全带 h5 `patient_id`,对应 252 个患者;165 卷小病灶卷 = 165 个患者;训练/验证集无重叠患者。
  - `interface_check`:"到任意两块候选脑区交界面的距离"与"到最近其他候选结构的距离"差值 100% 在一个体素对角线内(median 1.4 mm、max 5.09 mm);单卷 3.6 s,252 卷约 15 CPU 分钟;那一卷 89% 白质体素离交界面 < 3 mm(体素级)。
  - `power_sim`:只看全体 raw 的漏洞成立(白质全一致、皮层一半一致 → raw 0.893、κ 0.62、AC1 0.88 全过线,皮层 positive agreement 0.67、困难组 raw 0.52);患者聚簇模拟(400 sims × 300 boots,ICC≈0.02)80% 功效的最小可检 d:全标 1297 约 0.03 / 108 h,H1 全标+H2 600 约 0.035 / 72 h,H1 全标+H2 300 约 0.04 / 47 h,随机 700 约 0.045 / 58 h;解析 SE 比 1.38 / 1.12,Neyman 分配六成给 H2。
  - 四篇引文均核到(PMC10820331、PMID 31359448、PMC11041453、PMC10388213)。
- **v2.6 已合入 main**(PR #7 https://github.com/sober235/Foundation_Model/pull/7 ,merge commit `fb25da9`,2026-09-22 17:18;分支 `plan/aur-v2.6-experiment-design-2026-09-22` 未删):Gate 0.5、两层一致率门、排除单一留出 + 两设计试标后模拟、H1 = 交界面距离 + Δd、集合值主终点 + singleton rate、脑室不作宿主、嵌套 CV + patient_id 折断言 + 标签封存 + 700 例中期规则、读者字段 not_a_lesion / A_local_quality / time_seconds、脑侧 U 降为次要、B4/B2/Bgeo+ 初始配置与训练标签规则(§10.1)、200/201 与低分辨率分层、§25 决定记录、十步顺序。
- **测试**(合并后 main 实测,文档改动不影响代码):
  ```
  PYTHONNOUSERSITE=1 PYTHONPATH=. ~/anaconda3/envs/nvgen/bin/python -m pytest tests/ -q -p no:cacheprovider
  311 passed
  ```
- 09-13 之后的证据链不变:leg 2 膝检测 H1 不过但训在上下镜像的框上(`docs/verification/2026-09-15/H1_verdict.md`);脑探针噪声 q3 改答 7.4% [5.5, 9.5],干净参照自身在良性预处理下改答 2.8%–6.0%(`docs/verification/2026-09-16-brain-probe/REPORT.md`);fastMRI+ 框 y 从 RSS 底部数起,框占行 `[nr − y − h, nr − y)`;三份 PR 评审只在 GitHub:#4 https://github.com/sober235/Foundation_Model/pull/4#pullrequestreview-5274086790 、#5 https://github.com/sober235/Foundation_Model/pull/5#pullrequestreview-5274417275 、#6 https://github.com/sober235/Foundation_Model/pull/6#pullrequestreview-5274706631 。
- **读片人**:用户 2026-09-22 确认是医生、可以找到;人选、时间、裁定人、临床负责人、标注工具仍待填。
- 关键数字(判断余量用):SKM-TEA 类别感知查表天花板 0.958–0.968,余量约 3 点,311 例检不出;fastMRI+ 脑有框 476 卷,FLAIR 252 患者、1825 病灶、1297 小病灶(165 患者;200/201 系列 1077,低分辨率 220;79.3% 单层;每患者中位 3、最大 87、m_eff 29.45);探针 24 卷里 78.7% 小病灶查表宿主为白质(全集未测,Gate 0.5 补)。

## 2. 待用户拍板

- **已决定(2026-09-22,记录在 v2.6 §25,Gate 0.5 之前可推翻)**:两层 raw 一致率门;排除单一留出、全标与分层抽样试标后模拟择一;主终点全部病灶 + 集合值;脑室不作宿主;H1 = 交界面距离 + Δd;B4 两种证据都用并由 §10.1 初值起步;训练标签先用伪标签、医生标签微调为预注册升级路径;B2 与 B4 绑定。
- 读片人姓名、裁定人、临床负责人、标注工具、机构伦理备案确认。
- `summary/2026-09-22-v2.5-feasibility-review` 分支(大白话总结 `docs/handoff/2026-09-22-v2.5-review-summary.md`,f08b6a1)是否合回 main;它写于 v2.6 之前,若合回须注明 §25 五项已决定。
- **删已合并分支**(按"不删数据与代码"规矩,由用户手动执行):
  ```
  git push origin --delete review-core-target-a-u-r-2026-09-22 feature/aur-structured-perception-2026-09-22 plan/aur-v2.4-review-response-2026-09-22 plan/aur-v2.5-complete-route-2026-09-22 plan/aur-v2.6-experiment-design-2026-09-22
  ```
- 旧遗留(多次未答):Q9 删除授权(SKM-TEA 2.4G truncated 残留 + 820G 原 tar);fastMRI 其余 4850 卷是否跑 SynthSeg;Redivis token 事后删除。

## 3. 下一步

按 v2.6 §19 十步、§23 PR 顺序:

1. **PR-A Gate 0**(1–2 天):`anatobind/data_engine/fastmri_knee.py` 加 CSV→RSS 框转换 `[nr − y − h, nr − y)`,四类测试,manifest 记坐标约定与版本、旧版本拒绝加载;新导出根用链接指向旧卷目录,只换坐标文件(不重导 56 GB,旧导出不删);file → patient_id 映射与折断言;加 GitHub Actions 跑 CPU 测试。
2. **膝侧 H1 重跑**(1 天,8 卡空):五折 `anatobind/train/train_detector.py`,裁决写 `docs/verification/<日期>/H1_rerun.md`,顺手修正 spec:165 与 `scripts/check_h1.py` 的 H1 定义差异。
3. **PR-A′ Gate 0.5**(2–3 天,CPU 15 分钟):探针 `docs/verification/2026-09-16-brain-probe/probe_common.py` 的读框/合并/翻转/查表移入包内并加测试;`anatobind/eval/geometry.py`;`scripts/brain_frame.py` 对全部 1297 病灶算宿主分布、d_interface、Δd、t=2/3/4/5 mm 困难组占比、每患者病灶数、单层比例、两系列分开;冻结 t(占比 ≥ 15% 的最小 t);落盘 `docs/verification/<日期>/brain_frame.md`。占比不足 15% → 关系主线降级为退路。
4. **PR-B Level R**:协议 v1(§7.5 字段)、预注册骨架、盲化工具(3D Slicer + 表单先试 10 例并计时)、一致率统计、抽样权重与功效模拟模板、封存目录;pilot 100–150 例覆盖两系列(低分辨率 ≥ 30)。
5. **PR-C 基线**(3–5 周):B0 脑侧 33 类、Bprior、Bgeo+(含 d_interface、Δd)、B1、B2 = E_u + 线性头,外层五折 + 内层选择骨架;先用伪标签拟合。
6. **PR-D**:B3、B4 按 §10.1 初始配置;pilot 后按升级规则决定是否医生标签微调;700 例中期无效性分析;Gate R1。
7. PR-E 鲁棒性、脑侧检测器、E,只在 Gate R1 = GO 后。

估时:Gate 0 一两天;Gate 0.5 两三天;H1 重跑一天;pilot 取决于医生;PR-A 到 PR-D 合计 8–12 周工程;Gate R1 最早第五到六个月(受读片速度约束)。仍无正式时间表。

## 4. 坑与别重做

- 别再在 SKM-TEA 膝上找"退化让绑定失效"(G2:退化 0.46 mm、门要 5–8 mm),也别在 SKM-TEA 上做 clean 关系优越性主结论(余量 3 点)。它是几何容易的对照。
- 脑侧 clean 查表参照 = SynthSeg + 重叠 = B0 本身,不能当真值;良性预处理就让它改答 2.8–6.0%。主张"更准"只能在 Level R 人标上。
- fastMRI+ 框 y 从 RSS 底部数起;现有 leg 2 代码与 56 GB 导出都是原样框。Gate 0 完成前不引用 H1 结论,不给医生看未转换的框。
- 零填充 1D 欠采不能当几何退化;要欠采档必须先 ESPIRiT + SENSE/压缩感知正经重建。
- **分层抽样不是免费午餐**:主终点是全部病灶,简单组占八成权重;H1 全标 + H2 抽 300 的标准误比全标高 38%,每小时信息量四种设计相近;简单病灶恰是最快标完的。别预设"分层能省上百小时"。
- **只看全体 raw 一致率有漏洞**:白质全一致、皮层一半一致就能 raw 0.893、κ 0.62、AC1 0.88 全过线。困难组与少数类必须单独看。
- **层厚 5 mm 下 t = 2–4 mm 的困难组是面内准则**,跨层邻接进不来;t 必须看全集分布后再定(一卷里 89% 白质体素离交界面 < 3 mm,病灶若按体素分布困难组会是大半)。
- **只用伪标签训练的 B4 学的是模仿 B0**,在 B0 出错的困难病例上被教了错答案;pilot 上要按 v2.6 §10.1 的规则检查是否升级到医生标签微调。
- `IndependentCandidateHead` 是 B1,不是候选竞争;5 维几何只够原型,Bgeo+ 与学习模型必须共用同一套扩展几何。
- GLI-AL(arXiv 2607.22135):Synapse 受控访问、只有标签,需另取 BraTS 2023-GLI 影像;与 UCSF-PDGM 重叠(仓库记 298 例,须实测去重);只作 A/U 预训练。
- 仓库无 CI(PR-A 加)。评审 PR 时把分支导出到临时目录跑测试:`git archive <ref> | tar -x -C <dir>`,再在该目录 pytest;别 checkout 到工作树。
- GitHub 上 owner 账号不能对自己的 PR 选 Approve / Request changes,评审用 Comment 发,结论写在正文第一段。
- 方案文档里 §4.6 / §4.7 / §8 / §9.2–9.5 / §11 / §12 是历史章节(已打标记)。leg 2 规格 spec:165 的 H1 定义与 `scripts/check_h1.py` 不一致,H1 重跑时一并修。
- 共用 GPU 估时、`pkill -f` 误杀、nnU-Net 孤儿进程、m1r 头信息等旧坑见 tag `handoff/2026-09-13` 的 STATUS 与 git 历史 `29142bb`。

## 5. 关键决定的为什么

- **删掉 clean 伪参照第一道门**(v2.4):膝上类别感知查表 0.958–0.968,余量 3 点、311 例检不出;脑上参照与 B0 同源,Brel 只能打平或输。
- **Level R 前置**(v2.4/v2.5):关系模型的"更正确"只能在独立人标上定义;干净参照自身抖动 3–6% 与效应同量级。
- **脑 FLAIR 小病灶为主战场、SKM-TEA 降为对照**(v2.4):只有"小病灶 + 无类别限定 + 结构密集"的配置存在几何模糊(脑 2 mm 位移改答 5.7%,膝 2.4%)。
- **两层一致率门而不是单一 raw / κ / AC1**(v2.6):三种全局指标在"白质全对、皮层一半对"的场景全部失守,而 B4 唯一可能赢的正是皮层 / 困难组。
- **排除单一留出、两设计试标后模拟**(v2.6):留出约 250 例在任何假设下检不出 d ≤ 0.05;脑侧 m_eff 29.45 使膝侧沿袭的设计效应 1.3–1.5 失效;分层设计每小时信息量与全标相近,所以由实测读片时间与功效模拟定,不预设。
- **H1 用交界面距离而不是"到非宿主边界"**(v2.6):后者要先知道宿主,而宿主来自伪标签;交界面定义与任何模型、任何宿主答案无关,几何上是同一个量。
- **脑侧 U 降为次要**(v2.6):1240/1297 同一标签、79.3% 单层、样本比膝少四倍;检测器做得再好也救不了 B4 = Bgeo+ 的结局。
- **训练标签先用伪标签**(2026-09-22 用户拍板):便宜、全集可用;风险(模仿 B0)写明并设预注册升级路径。
- **B4 两种证据都用、B2 与 B4 绑定**(2026-09-22 用户拍板):图像小块与距离图互补;B2 小块与 B4 相同才是诚实的对照。
- **合并 #6 而非分别合 #3/#4/#5**;**不执行运动分支(09-13)** 不变。
