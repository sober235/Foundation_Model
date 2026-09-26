# STATUS：2026-09-26（Level R 读片工具已实现、整支评审、冒烟通过；等浏览器验收与用户拍板；未 push、未合 main、未打 tag；上一交接点 tag `handoff/2026-09-25-gate0-h1-gate05`）

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

## 2. 待用户拍板

- **(a) R7 门判 pilot 还是判全集**：规格 §9 把它当 pilot 的过关门；v2.6 §7.2/§7.7 说 pilot 只用来估参数，门应该在读完全集后判。150 例 / 82 患者的样本量下，CI 下限 ≥ 0.80 这一层需要点估计到约 0.86 才稳。建议：pilot 的门线只作参考，continue/revise 的决定看 raw 一致率与用时；R7 正式判定放到读完全集之后；随后统一改 `pilot.py` 的 docstring、运维文档与读片说明。
- **(b) 匿名码**：`volume_code = sha256(stem)[:8]` 不带密钥；有心的读者拿公开的 fastMRI 文件列表逐个哈希，就能反推出 stem 进而对上标签。建议：对配合的读者可以接受；如果要堵上，换成服务器端密钥的 HMAC，并在发 token 前重新导出到新目录（码已经烧进当前导出的文件里了）。
- **(c) 邻接是否必填**：`adjacency: []` 会被接受，和"选了'无'"无法区分，尽管读片说明写着要选"无"。建议：非 `not_a_lesion` 的答案要求邻接至少选一项；这条校验要在 pilot 开始前定，读到一半改校验会把数据切成两截。
- **(d) pilot 结束的硬停**：目前是软停——"下一个"按钮不再给新病灶，但被扣住的读者仍能从列表点开正式集的病灶并提交。建议：服务端对被扣读者的非 pilot 提交直接拒绝（409）；如果要在发 token 前补上，是个小工作量。
- **(e)** 读者姓名、token 发放、裁定人、伦理备案（v2.6 §18）；服务对外方式 A 端口转发 / B `--bind 0.0.0.0`。
- **(f)** 本分支要不要 push、要不要合回 main、要不要打 tag `handoff/2026-09-26-level-r-tooling`。
- **(g)** nnDetection 第二臂何时起（GPU 是否有空）。

## 3. 下一步

1. 用户在 8791 上做浏览器验收（`docs/verification/2026-09-26/level_r_smoke.md` §7 的八步仍是 USER_REPORTED）。
2. 发 token 前先修一个时序 bug：`openLesion` 在 `loadVolume` resolve 之前就把 `state.lesion` 设成了新病灶；走"下一个"或从列表切换时如果卷加载失败，页面还留着旧图但 `state.lesion` 已经指向新病灶，此时提交会记错病灶——要改成加载成功之后再赋值。
3. 定 §2 (a)–(d) 四个决定。
4. 按 `docs/level_r_tool.md` 正式部署：`$D/level_r.sqlite` 上 `init` → `add-reader` ×3 → `order --pilot` → 8790 起服务 → pilot 150 → `level_r_report.py` → continue/revise 决定 → 两位读者 `release` → 读全集 → 裁定 → `seal`。
5. PR-C 关系基线可以在 pilot 期间并行开工。

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
