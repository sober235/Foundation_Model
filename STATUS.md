# STATUS：2026-09-26（Level R 读片工具已实现、整支评审、冒烟通过；等浏览器验收与用户拍板；已合入 main（91526e0，tag `handoff/2026-09-26-level-r-tooling`，本地未 push）；follow-up A1/A2（`649e7f0`..`dfc3d6c`，含本次 STATUS 修正）在分支上，待控制方合入 main 并打第二个 tag，同样未 push；上一交接点 tag `handoff/2026-09-25-gate0-h1-gate05`）

每次交接前整体重写本文件。五段固定：已验证、待拍板、下一步、坑与别重做、为什么。

## 1. 已完成且已验证

本轮实施 Level R 读片工具（v2.6 PR-B）。设计规格 `docs/superpowers/specs/2026-09-25-level-r-annotation-tooling-design.md`（决定记录 R1–R10；末尾新增 `## 13. 实施修订（2026-09-26）`，见本文件 §5 前的说明）；实施计划 `docs/superpowers/plans/2026-09-26-level-r-annotation-tooling.md`（13 个任务）。分支 `plan/level-r-tooling-2026-09-25`，工作树 `/data0/congcong/code/Project_Doing/foundation_model-levelr`，基于 main `afe641e`；本次提交前 head `20fb4d6`。自计划提交 `e7ac80a` 起 25 次提交（12 个任务提交 + 4 轮修复提交 + 9 个终审修复提交），每一条都经过独立的新鲜评审；每个决定的台账在 `.superpowers/sdd/2026-09-26-level-r-annotation-tooling/progress.md`（本工作树内，已 gitignore，不入库）。**尚未 push、未合 main、未打 tag——由用户拍板，见 §2 (f)。**

- **工具全部落地（R1–R10）**：`anatobind/level_r/{schema,registry,blind,export,store,server,admin,pilot}.py`、`anatobind/level_r/app/{index.html,app.js,style.css,guide.html}`、`anatobind/eval/{level_r_stats,level_r_labels}.py`、`scripts/level_r_{export,admin,server,pilot_sample,report}.py`、入库数据 `data/level_r/{folds.json,pilot_150.json}`、运维文档 `docs/level_r_tool.md`、验证记录 `docs/verification/2026-09-26/{level_r_pilot_sample,level_r_smoke}.md`。

- **测试**：
  ```
  PYTHONNOUSERSITE=1 PYTHONPATH=. ~/anaconda3/envs/nvgen/bin/python -m pytest tests/ -q -p no:cacheprovider
  581 passed in 58.15s
  ```
  本分支之前基线 413 passed；新增 12 个测试文件 `tests/test_level_r_*.py`；测试不读 `/data2`；`.github/workflows/tests.yml` 未改。

- **真实导出**（`docs/verification/2026-09-26/level_r_smoke.md` Step 3）：
  ```
  PYTHONNOUSERSITE=1 PYTHONPATH=. nice -n 19 … scripts/level_r_export.py --out /data2/congcong/data/FM_data/derived/level_r
  exported 1297 lesions from 165 volumes
  ```
  `volumes/` 下 330 个文件，共 473M；不同 volume code 165 个。

- **五折**（`data/level_r/folds.json`，`scripts/level_r_admin.py folds`）：165 名患者，每折 33 人；每折病灶数 280/250/276/334/157（按患者均分，不按病灶均分——有一名患者带 87 个病灶）。

- **pilot**（`data/level_r/pilot_150.json`，`scripts/level_r_pilot_sample.py`）：150 个病灶，82 名患者，每人最多 3 个；大格子分配 36/14/15/22（合计 87）。3 mm 层厚的格子因为其 7 名患者都撞到人数上限，实取比分配少（如 `0|inplane_0.62_slice_3` 分配 8 只取到 3）；缺口 14 个补进 `0|inplane_0.69_slice_5`（分配 36，实取 50）。这是"格子下限"与"每患者最多 3 个"两个约束不能同时满足时的取舍——正确处理，但和 R8 原文有出入。

- **冒烟部署**：冒烟库 `/data2/congcong/data/FM_data/derived/level_r_smoke/level_r.sqlite`（读者 `smoke_r1`/`smoke_r2`，裁定人 `smoke_adj`；token 在 `/data2/congcong/data/FM_data/derived/level_r_smoke/tokens.txt`，不入库）；服务已用评审后的代码重启：PID 1380837，`127.0.0.1:8791`，pid 文件 `$S/server.pid`，日志 `$S/server.log`。curl 核对：`/` 200 text/html；`/guide.html` 200；错 token 403；读者 `/api/me` → total 1297、next 830、held false。停止命令 `kill $(cat /data2/congcong/data/FM_data/derived/level_r_smoke/server.pid)`（本轮未停，交接后由用户决定）。

- **整支评审**：opus 评审结论 "With fixes"；所有修复已落地并复审：C1 封存加载器拒绝非整数/未知折号；`seal` 要求完整无重复的 1297 条集合，折数取自 `folds.json`；导出改写到带时间戳的 `export/<时间戳>/` 目录；除 `init` 外的子命令一律只读写打开已有库（库名拼错报错，不留一个空库）；第三位读者被拒绝；pilot 结束后读者被扣住，直到 `level_r_admin.py release`；一条裁定只在比两位读者当时最新答案都新时才计入最终标签；AC1 固定用 K = 8 类；用时统计按 pilot 范围、跨改答复访求和；0 mm 档的 raw 一致率带 bootstrap 区间；`time_seconds`/窗宽窗位做校验（400）；未预期的错误统一返回 500 JSON；裁定人页面不再显示备注框；每个病灶的放大中心点固定；缩略图保持长宽比；读片说明不再写"3T"（165 卷里有 22 卷是 1.5T）。

- **09-24/25（上一轮）**：Gate 0 落地（fastMRI+ 框翻转修进数据引擎，膝侧系列级审计 89.4%，按系列门槛加叠图判通过）；膝 H1 用修好的框重跑五折，迁移判据不过（大类正确灵敏度 0.091 @ 1.53 FP/卷，门 ≥ 0.5）；Gate 0.5 盘点完成（1297 个小病灶，165 名患者，几何分层）。详见 `docs/verification/2026-09-24/REPORT.md`。
- 09-23 之前的证据链不变（膝 G2 不过、SKM-TEA 箱填检测门不过 0.254、脑探针 q3 7.4%、查表天花板 0.958–0.968）。

- **follow-up A1（病灶类型/侧别/脑叶字段，2026-09-26）**：`schema.py`、`store.py`、app 表单与读片说明、`level_r_stats.py`、`level_r_report.py`、`admin.py` 的 CSV 列、`level_r_labels.py` 全链路加了三个字段：`lesion_type`（4 类）、`side`（`image_left`/`image_right`/`midline`，记的是屏幕左右）、`lobe`（5 叶 + `not_applicable`，必填）。`not_a_lesion` 为真时三者都记 None。裁定触发条件加了 `lesion_type`、`side`（`lobe` 不一致不触发裁定，最终 `lobe` 只在两位读者一致时保留，否则记 None）。旧库要跑一次 `level_r_admin.py init` 做列迁移，之前 server、报告脚本、`add-reader`/`order`/`export`/`release` 一律拒绝打开旧库（`backup` 例外）。提交 649e7f0、2a762f7、70e6eb3（字段与迁移）、5b68d74（读片说明补两叶病灶/非叶白质/信心范围三条规则）、45fb29e（评审修复：不适用标签放宽、勾选框加提示、脑叶兜底改判病灶中心所在脑叶、runbook 升级顺序修正）。
  ```
  PYTHONNOUSERSITE=1 PYTHONPATH=. ~/anaconda3/envs/nvgen/bin/python -m pytest tests/ -q -p no:cacheprovider --color=no
  612 passed in 60.51s   # 5b68d74 处测得；45fb29e 只在已有测试里加断言、没加测试项，未在 45fb29e 重新跑全量
  node --check anatobind/level_r/app/app.js   # exit 0，649e7f0/2a762f7 之后测过；45fb29e 没改 JS
  ```

- **核心目标对齐审查（2026-09-26）**：用户重申三个核心目标——(1) 精准识别解剖位置；(2) 精准识别病灶名字；(3) 解耦地识别"某组织上存在某病灶"。评审结论一条一线：(1) 膝侧已对齐，脑侧只有粗宿主（7 类，左右合并，SynthSeg 未验证）；(2) 病灶名字此前全链路未采集，被 v2.6 §6.3 降级，是最大的偏移——fastMRI+ 原标签对读者隐藏，1297 个病灶里 1240 个共用一个标签；(3) 架构已解耦、R 真值已采，但陈述里的病灶名字来自数据集标签，绑定粗糙。用户拍板：A 现在做（给 Level R 表单加病灶类型/侧别/脑叶，已实现，见上一条 A1）；B 留后（命名自动化：脑侧靠计划 2 的带 3D 掩膜与类型的数据，膝侧靠 nnDetection）；C 兜底（A 做不成才把论文口径收窄为"给定病灶实例的粗宿主绑定"，目前不需要）。

## 2. 待用户拍板

- **(a) R7 门**：已定（用户 2026-09-26 拍板：不考虑安全问题，一切以实现目标为准）：pilot 上的 GATE_R7 行只作指示，门在全集上判（v2.6 §7.2/§7.7）；pilot 后是否继续看 raw 一致率与读片用时；`scripts/level_r_report.py` 输出不改，解读按此。
- **(b) 匿名码**：已定（用户 2026-09-26 拍板：不考虑安全问题，一切以实现目标为准）：保持 sha256 前 8 位，不换 HMAC。
- **(c) 邻接**：已定（用户 2026-09-26 拍板：不考虑安全问题，一切以实现目标为准）：不强制；空选记为"未答"，"无"是明确回答，统计时分开。
- **(d) pilot 停顿**：已定（用户 2026-09-26 拍板：不考虑安全问题，一切以实现目标为准）：软停即可（"下一个"停下、列表仍可点），不做服务端硬停。
- **(e)** 读者姓名、token 发放、裁定人、伦理备案（v2.6 §18）；服务对外方式 A 端口转发 / B `--bind 0.0.0.0`。
- **(f)** 本分支要不要 push——目前完全本地：R1–R10 已合入 main（91526e0，tag `handoff/2026-09-26-level-r-tooling`）；follow-up A1/A2（`649e7f0`..`dfc3d6c` 及本次修正）待控制方合入 main 并打第二个 tag；两次合并与两个 tag 都还没 push。
- **(g)** nnDetection 第二臂何时起（GPU 是否有空）。
- 已定（上一轮，2026-09-25）：Gate 0 通过（膝 89.4% 差 0.6 个百分点，记录在案）；Gate 0.5 判 GO，t 不冻结，d_interface 四档分层、全集 1297 标注（v2.6 §25 第 9 项）；分层按实测几何；第二臂已定为 nnDetection 直训 SKM-TEA（VERDICT §4 ②），在 GPU 空时并行，计划另写；npz 不删、已合并分支不删。
- 已定（用户 2026-09-26）：按 A/B/C 建议执行——A：现在做，Level R 表单加病灶类型/侧别/脑叶（已完成，见 §1 A1）；B：留后，命名自动化脑侧走计划 2（3D 掩膜 + 类型），膝侧走 nnDetection；C：兜底，若 A 未做成则论文口径收窄为"给定病灶实例的粗宿主绑定"（目前不需要）。
- 旧遗留仍挂：Q9 删除授权（2.4G truncated 残留 + 820G 原 tar）、其余 4850 卷未标注脑的 SynthSeg、Redivis token、`summary/2026-09-22-v2.5-feasibility-review` 与 `review-core-target-a-u-r-2026-09-22` 两条远端分支是否合回或删除、`plan/level-r-tooling-2026-09-25` 合回 main 的时机。

## 3. 下一步

1. 合并 A1/A2 后：停冒烟服务 → init → 重启（控制方已按此顺序做过一次，PID 2152185 在 8791）。
2. 冒烟库迁移（`init` 一次）与冒烟服务重启由控制方在合并后做。浏览器验收要点加三个新字段：`docs/verification/2026-09-26/level_r_smoke.md` §7 第 9–10 行（勾选"不是病灶"后三个下拉变灰；两位读者只在病灶类型上不一致时该病灶进裁定列表），都还是 USER_REPORTED。
3. 用户在 8791 上做浏览器验收（`docs/verification/2026-09-26/level_r_smoke.md` §7 的八步仍是 USER_REPORTED）。
4. `openLesion` 时序 bug 已在本次提交修复：`state.lesion`/`state.vol` 等字段只在 `loadVolume` resolve 之后才赋值，见新增测试 `test_open_lesion_assigns_state_only_after_the_volume_loaded`（`tests/test_level_r_app_static.py`）。
5. 定 §2 (a)–(d) 四个决定。
6. 按 `docs/level_r_tool.md` 正式部署：`$D/level_r.sqlite` 上 `init` → `add-reader` ×3 → `order --pilot` → 8790 起服务 → pilot 150 → `level_r_report.py` → continue/revise 决定 → 两位读者 `release` → 读全集 → 裁定 → `seal`。
7. PR-C 关系基线可以在 pilot 期间并行开工。
8. nnDetection 第二臂计划另写（GPU 空时并行）。
9. B：nnDetection 计划另写（同上一条）；脑侧命名数据走计划 2（3D 掩膜 + 类型，待写）。

## 4. 坑与别重做

- 冒烟库/服务和正式库/服务是两套，别搞混：`level_r_smoke/` + 8791 冒烟，`level_r/level_r.sqlite` + 8790 正式。
- 除 `init` 外都用 `Store(create=False)` 打开；一个建库早于 `releases` 表加入之前的库（冒烟库已经处理过）要先补跑一次 `init`（幂等）才能被新服务打开。
- `export` 写到 `export/<时间戳>/`，同一时间戳已存在会拒跑。
- `seal` 需要 `--folds`（取 k）与完整的 1297 条集合。
- `lesions.json` / `pilot_150.json` / `folds.json` 的生成脚本都拒绝覆盖已有文件。
- pilot 的扣留是软的（见 §2 (d)）。
- 同一秒的裁定与读者提交打平时裁定生效（`store.py` 的比较规则）。
- 前端用 `Date.now()` 计时；客户端时钟往回跳会让提交被 400 拒绝，要重新打开病灶再提交一次。
- `BlindingError`（一种 `ValueError`）在服务端会映射成 409，报文里只有 key 名，不泄漏值。
- `fastmri_knee` 的 import 链会把 torch 一起拉进 `export`/`admin`，启动慢属预期。
- 浏览器交互没有 CI；改了 `app.js` 之后要人工点一遍。
- 测试不读 `/data2`。
- SDD 的台账/任务简报/报告都在这个工作树的 `.superpowers/sdd/`，已 gitignore，不进仓库。
- 侧别记的是屏幕左右，不是患者左右；换算规则待定。
- 两位读者脑叶不一致不进裁定，最终 `lobe` 记 None（只有主结构/病灶类型/侧别不一致才进裁定）。
- 旧库必须 `init` 一次，否则服务拒绝打开。

## 5. 关键决定的为什么

- 用 stdlib `http.server` + sqlite，不加依赖：医生不用装软件，盲化、计时、数据收集都在自己手里。
- 图像用 uint16 数组而不是 PNG：窗宽窗位放到浏览器里现算，小病灶才看得清。
- 单一 JSON 出口配白名单 + 递归黑名单检查（`blind.py`）。
- 表只追加：历史即证据，有测试守着"不出现 UPDATE/DELETE"。
- pilot 缺口回填到最大的格子：格子下限和每患者上限两个约束不能同时满足，选了保证总数 150。
- 折按患者、种子 0 分（v2.6 §4.4），即使各折病灶数不均——折的定义是患者不跨折，不是病灶数配平。
- 一次性封存 + sha256 清单 + `unblind=True` + 访问日志（v2.6 §12.7）。
- AC1 固定用 K = 8：这是评分量表本身的类别数，不是数据里实际观察到的类别数，小样本或 pilot 子集类别不全时仍然可比。
- 导出到带时间戳的目录：用户的硬规矩——不覆盖已有数据文件。
- 交接前先做整支评审：STATUS 记录的是评审过的状态，不是评审前的状态。
- 侧别只记屏幕左右：RSS 数组的左右手性没定，医生按屏幕作答最快也最不容易错；等手性定了再一次性、全体转换成患者左右，不在读片当下做。
- 脑叶必填、留"不适用"：目标关系陈述要凑齐"<侧别><脑叶><宿主>上存在<病灶类型>"，脑叶留空等于陈述缺一块；深部灰质、脑干、小脑这些确实不属于任何脑叶，"不适用"是诚实的答案，不是跳过。
