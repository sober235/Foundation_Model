# STATUS:2026-09-22(v2.5 路线已合入 main,读片人已确认;上一交接点 tag `handoff/2026-09-13`)

每次交接前整体重写本文件。五段固定:已验证、待拍板、下一步、坑与别重做、为什么。

## 1. 已完成且已验证

**主线变化:关系问题的第一道门从"退化让绑定失效"改成"独立人标真值上 learned R 是否超过强几何基线"。权威方案 = `RESEARCH_PLAN.md` v2.5 + `docs/plans/2026-09-22-aur-v2.5-complete-technical-route.md`(冲突以后者为准);`docs/plans/2026-09-22-aur-v2.4-review-response.md` 只作审计记录。**

- **PR #6 已合入 main = `779123f`**(2026-09-22 14:23,merge commit,与 PR #1 / plan-v5 系列同一方式)。它包含 #4(A/U/R 重构 + B1 头)与 #5(v2.4)的全部提交;GitHub 自动把 #4、#5 标为 MERGED,#3 已关闭。四条远端分支未删(见 §2)。
- **代码**:`anatobind/model/relation.py` 新增 `IndependentCandidateHead` = **B1**(逐候选独立 MLP 打分后对宿主 softmax,候选之间、病灶之间无交互),`HostCompetitionHead` 别名已删;旧 `RelationModule`(全局 K×M)保留为消融。`tests/test_independent_candidate.py` 7 个测试(掩码、无候选时 none、病灶置换等变、无跨病灶混合、局部证据形状、梯度)。**没有任何新头接进训练或评估脚本。**
- **测试**(合并后 main 实测):
  ```
  PYTHONNOUSERSITE=1 PYTHONPATH=. ~/anaconda3/envs/nvgen/bin/python -m pytest tests/ -q -p no:cacheprovider
  311 passed in 33.65s
  ```
- **09-13 之后的证据链**(都在仓库里):
  - leg 2 fastMRI+ 膝检测 **H1 不过**(09-15,`docs/verification/2026-09-15/H1_verdict.md`):留出 300 卷检出 208 个、对 2 个,任何阈值下扫描级正率 0.823。**但检测器训在上下镜像的框上**(见下一条),H1 的"损失量级失衡"归因作废,修框后必须重跑。
  - **脑侧探针**(09-15/16,`docs/verification/2026-09-16-brain-probe/REPORT.md`,24 卷 FLAIR、780 小病灶、SynthSeg 33 类无类别限定查表):噪声 q1/q2/q3 改答 3.1% / 4.6% / **7.4% [5.5, 9.5]**,白质 Dice 0.972 / 0.952 / 0.911;零填充 1D 欠采 4× 即整幅鬼影不可用;免训练去噪关不掉窗口(仅噪声底 6.9%,加 NLM 9.1%);**干净参照自身在良性预处理下改答 2.8%–6.0%**,与效应同量级。
  - **fastMRI+ 框上下翻转**(官方 README:转 DICOM 时像素上下翻转,CSV 的 y 从 RSS 底部数起,框占行 `[nr − y − h, nr − y)`);实测脑 22/24 卷、膝 30/30 卷翻转后框才落在病灶上。`anatobind/data_engine/fastmri_knee.py` 与 `anatobind/train/dataset_knee.py` 目前仍按原样用框。
  - 评审文件:`REVIEW_v7_feasibility_2026-09-16.md`(SynthSeg+ 就是已用的 --robust;DETR 应换热图头)、`REVIEW_v7_feasibility_2026-09-19.md`(Gate A′ / Gate B 两道生死门)、`REVIEW_core_target_AUR_2026-09-22.md`(A/U/R 核心目标)。
  - 三份 PR 评审只在 GitHub,仓库内无副本:#4 https://github.com/sober235/Foundation_Model/pull/4#pullrequestreview-5274086790 、#5 https://github.com/sober235/Foundation_Model/pull/5#pullrequestreview-5274417275 、#6 https://github.com/sober235/Foundation_Model/pull/6#pullrequestreview-5274706631 。#6 的 11 条内联评论是 v2.5 待改清单。
- **读片人**:用户 2026-09-22 确认 Level R 的读片人是医生、可以找到 → Gate R0 / R1 可执行。人选、时间、裁定人、临床负责人、标注工具仍待填(v2.5 §18 的 TBD)。
- 关键实测数字(判断余量用):SKM-TEA 类别感知查表 clean 天花板 0.958(m1r)–0.968(oracle),余量约 3 点,311 例检不出(§9.7:d=0.03 需约 700 例);fastMRI+ 脑有框 476 卷,FLAIR 252 患者、1825 病灶、1297 小病灶(165 卷),约 79% 小病灶完全在白质内。

## 2. 待用户拍板

- **一致率门槛的量纲**:v2.5 §7.7 写 raw agreement 80%。白质 79% / 皮层 19% 时两读者机会一致率 0.66,raw 0.80 对应 κ≈0.41。建议改为 Gwet AC1 ≥ 0.6(或 raw ≥ 0.90),且只在最终队列判,pilot 只估参数。
- **关系模型评估用患者级五折 CV 还是单一留出**:CV 让全部 1297 小病灶当测试,人标目标 = 全集(两读者各约 2600 例,每例 1.5 分钟约 32 小时,裁定约 260 例);单一 20% 留出只有约 260 病灶,全标也只能检出 6–8 个点。退路 = ≥700 例加权子集(p_disc 0.10、d 0.04 需 640–735)。
- **v2.5 三处口径矛盾**(PR #6 review 内联):§7.8 主终点只算"确定"病灶 vs §12.1 主人群全部病灶(建议全部病灶 + 集合值正确性);§3 把 ventricle 列为宿主 vs V7/v2.4 的"脑室与 CSF 只作地标";§7.3 的 H1 分层须用纯几何准则(到最近候选边界距离),不能用含被测模型的方法不一致。
- 读片人姓名、裁定人、临床负责人、标注工具、机构伦理备案确认。
- **删已合并分支**(按"不删数据与代码"规矩,由用户手动执行):
  ```
  git push origin --delete review-core-target-a-u-r-2026-09-22 feature/aur-structured-perception-2026-09-22 plan/aur-v2.4-review-response-2026-09-22 plan/aur-v2.5-complete-route-2026-09-22
  ```
- 旧遗留(多次未答):Q9 删除授权(SKM-TEA 2.4G truncated 残留 + 820G 原 tar);fastMRI 其余 4850 卷是否跑 SynthSeg;Redivis token 事后删除。

## 3. 下一步

按 v2.5 §23 的 PR 顺序:

1. **PR-A Gate 0**(约一两天):在 `anatobind/data_engine/fastmri_knee.py` 实现 CSV→RSS 框转换 `[nr − y − h, nr − y)`,单元测试(合成已知框、CSV→RSS→CSV 往返误差 0、体素×间距的物理坐标)、脑/膝各抽样 overlay、manifest 记 `box_coordinate_convention` / `transform_version`;**旧导出 `derived/fastmri_knee/leg2`(56 GB)标为旧版本并在加载时拒绝,不删**。
2. 修复后重跑 H1 五折检测器(`anatobind/train/train_detector.py`,GPU 空卡上单折约 1 小时)。
3. **PR-B Level R**:标注协议(primary_host 约 7 类 + topography + adjacency + ambiguity + confidence)、盲化标注工具、一致率统计(raw + κ + Gwet AC1/AC2)、抽样权重;先 150 例 H1 富集 pilot(每读者 3–5 小时)。
4. **PR-C 基线**:B0(`anatobind/eval/lookup.py` 已有类别感知查表 + 最近兜底,扩展它)、Bprior、Bgeo+(sklearn `HistGradientBoostingClassifier`,xgboost 未装)、B1、B2,共用 16 维扩展几何(v2.5 §11)。
5. PR-D B3 `CandidateCompetitionHead` / B4 + Gate R1;PR-E 鲁棒性(噪声视图训练 nnU-Net、ΔA/ΔU/ΔR)。

估时:Gate 0 一两天;H1 重跑两三天 GPU;pilot 取决于医生;基线两三周。仍无正式时间表。

## 4. 坑与别重做

- 别再在 SKM-TEA 膝上找"退化让绑定失效"(G2 裁决:退化 0.46 mm、门要 5–8 mm),也别在 SKM-TEA 上做 clean 关系优越性主结论(余量 3 点,样本检不出)。它是几何容易的对照。
- 脑侧 clean 查表参照 = SynthSeg + 重叠 = B0 本身,**不能当真值**;良性预处理就让它改答 2.8–6.0%。主张"更准"只能在 Level R 人标上。
- fastMRI+ 框 y 从 RSS 底部数起;现有 leg 2 代码与 56 GB 导出都是原样框。Gate 0 完成前不引用 H1 结论,不给医生看未转换的框。
- 零填充 1D 欠采不能当几何退化;要欠采档必须先 ESPIRiT + SENSE/压缩感知正经重建。
- `IndependentCandidateHead` 是 B1,不是候选竞争;5 维几何 `(Δz, Δy, Δx, ‖d‖, IoA)` 只够原型,Bgeo+ 与学习模型必须共用同一套扩展几何。
- GLI-AL(arXiv 2607.22135):Synapse 受控访问、只有标签,需另取 BraTS 2023-GLI 影像;与 UCSF-PDGM 重叠(仓库记 298 例,须实测去重);只作 A/U 预训练。
- 仓库无 CI。评审 PR 时把分支导出到临时目录跑测试:`git archive <ref> | tar -x -C <dir>`,再在该目录 pytest;别 checkout 到工作树。
- GitHub 上 owner 账号不能对自己的 PR 选 Approve / Request changes,评审用 Comment 发,结论写在正文第一段。
- 方案文档里 §4.6 / §4.7 / §8 / §9.2–9.5 / §11 / §12 是历史章节(已打标记);第 240 / 485 / 508 / 584 / 669 行仍残留 v2.4 措辞,v2.4 文档没有"已被取代"头。
- 共用 GPU 估时、`pkill -f` 误杀、nnU-Net 孤儿进程、m1r 头信息等旧坑见上一版 STATUS(tag `handoff/2026-09-13`)与 git 历史 `29142bb`。

## 5. 关键决定的为什么

- **删掉 clean 伪参照第一道门**(v2.4,PR #4 评审):膝上类别感知查表 0.958–0.968,余量 3 点、311 例检不出;脑上参照与 B0 同源,Brel 只能打平或输。
- **Level R 前置**(v2.4/v2.5):关系模型的"更正确"只能在独立人标上定义;干净参照自身抖动 3–6% 与效应同量级。
- **脑 FLAIR 小病灶为主战场、SKM-TEA 降为对照**(v2.4):G2 §9 与探针表明只有"小病灶 + 无类别限定 + 结构密集"的配置存在几何模糊(脑 2 mm 位移改答 5.7%,膝 2.4%)。
- **B1 命名与删别名**(v2.5):逐候选独立打分再 softmax 不是候选竞争,叫竞争会误导;真 B3 另建。
- **退化降为二阶段机制研究**(v2.3→v2.5):Gate A′(噪声视图训练分割器)只否定"退化救回必须靠关系",不否定 A/U/R 主任务。
- **合并 #6 而非分别合 #3/#4/#5**:#6 含前两者全部提交且吸收了三轮评审;单独合 #4 会把 v2.3 的矛盾门写进 main。
- 不执行运动分支(09-13)不变。
