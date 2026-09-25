# Level R 读片协议与工具（v2.6 PR-B）· 设计规格

> 状态：2026-09-25 与用户讨论后定稿（三次拍板见 §1）。它记录**已经定下的设计**，实施计划另写。
> 上游：`docs/plans/2026-09-22-aur-v2.6-experiment-design-route.md` §3（本体）、§7（读片协议）、§12.7（封存）、§16（schema）、§18（工具最低功能）、§23（PR-B 清单）、§25 第 9 项（Gate 0.5 决定：四档距离分层、全集标注）。病灶清单：`docs/verification/2026-09-24/gate05/lesions.csv`（1297 条）。

## 0. 一句话

给两位放射科医生一个浏览器就能打开的读片工具，让他们对 1297 个 fastMRI+ 脑 FLAIR 小病灶各自独立回答"这个病灶在哪块脑区"（主结构、可接受集合、拓扑位置、邻接、不确定性、是不是病灶、局部图像质量、信心），第三位医生只裁分歧；答案存在服务器上，按患者分五折封存测试折；先做 150 例 pilot 看一致率和用时，过门再读完全集。工具里没有任何模型输出，也没有 SynthSeg。

## 1. 决定记录

| # | 决定 | 一句话理由 |
|---|---|---|
| R1 | 医生用浏览器访问服务器上的工具（用户 2026-09-25 选 A）；离线包（导 PNG 图板 + 表格）只作退路，不在本规格内 | 不用装软件；盲化、计时、回收都在我们手里 |
| R2 | 两位读者独立读，第三位医生只看分歧的病灶做裁定（用户选 A） | v2.6 §7.4 默认；裁定页做通用，事后协商也能用 |
| R3 | 工具形态：Python 标准库 `http.server` + `sqlite3` 起单文件服务，前端一页纯 HTML/JS，不加新依赖 | 三种做法里唯一同时满足"不用装、可盲化、可计时"的（3D Slicer 要装、PNG 图板看不清小病灶） |
| R4 | 图像以 16 位原始数组发给浏览器，浏览器端算窗宽窗位 | 8 位 PNG 调不了窗；小病灶要看得清 |
| R5 | 医生看不到：SynthSeg、模型输出、距离档、采集系列、患者号、fastMRI+ 原标签文字 | "非特异白质病灶"这几个字就是答案提示；v2.6 §18 "不显示 model prediction" |
| R6 | 病灶清单直接用 Gate 0.5 的 `lesions.csv`，ID 一致 | 后面按四档距离、采集分层做统计要对得上号 |
| R7 | 一致率门（v2.6 §7.7）保留两层，"困难组"换成 0 mm 那档（框内已跨两个脑区，42%）：全体 raw 的 95% 区间下限 ≥ 0.80，且 0 mm 档 raw ≥ 0.70 | H1/H2 已按 §25 第 9 项废止；0 mm 档是最像"难分"的那组 |
| R8 | pilot 150 例，四档距离 × 四个采集分层按比例抽，非空格子至少 8 个，每位患者最多 3 个；两位读者先读完 pilot 再决定是否继续 | v2.6 §7.2 只估参数；分层覆盖两层（§7.3） |
| R9 | 最终标签：两人主结构一致 → 一致答案，可接受集合取并集；主结构不一致、或一人判"不是病灶"、或并集超过 2 个 → 进裁定，以裁定人为准 | v2.6 §7.5 可接受集合 ≤ 2、§7.8 集合值主终点 |
| R10 | 封存：165 名患者按种子分五折，折表入库；每折最终标签写到服务器封存目录，仓库只放 sha256 清单；测试折只能显式带 `unblind=True` 读，每次读写访问日志 | v2.6 §12.6–12.7 |

## 2. 核心目标与本工具的关系（2026-09-25 讨论结论）

医生读片给出的是**真值**，不是能力。脑侧现有的参照（SynthSeg + 重叠查表）与要比较的基线同源，模型与查表不一致时没有东西能判谁对；Gate R1 只有拿到独立真值才能跑。42% 的小病灶在 5 mm 层上框内已跨两个脑区，"在哪块组织"本身就模糊，只有医生能定。就算关系模型最后没赢，标完的 1297 例本身是可发表的资产（v2.6 §8 退路）。

## 3. 系统组成

```
scripts/level_r_export.py        病灶清单 + 165 卷 FLAIR → derived/level_r/{volumes/, lesions.json}
scripts/level_r_admin.py         建库、发 token、生成每位读者的随机顺序、导出 CSV、封存
scripts/level_r_server.py        http.server 单文件服务（API + 静态页）
anatobind/level_r/app/           index.html + app.js + style.css（读者页与裁定页同一套代码）
anatobind/level_r/store.py       sqlite 读写（只追加）、最终标签规则、分歧列表
anatobind/level_r/blind.py       发给浏览器前的字段白名单（禁用字段测试守着）
anatobind/eval/level_r_stats.py  一致率、κ、AC1、positive agreement、集合值一致、分层、用时
anatobind/eval/level_r_labels.py load_train_labels(k) / load_test_labels(k, unblind) / 折表
scripts/level_r_pilot_sample.py  pilot 150 抽样 → data/level_r/pilot_150.json
scripts/level_r_report.py        → docs/verification/<日期>/level_r_pilot.md
data/level_r/folds.json          165 患者五折（入库）
data/level_r/sealed_manifest.json 每折标签文件的 sha256 与行数（入库）
derived/level_r/                 volumes/ lesions.json level_r.sqlite readers.json export/ sealed/ access_log.txt（服务器，不入库）
```

## 4. 导出（`scripts/level_r_export.py`）

- 病灶：读 `gate05/lesions.csv` 作注册表；用 `read_fastmri_plus_rows → rows_to_rss_frame → merge_boxes_3d` 按 `scripts/brain_frame.py` 同一路径重算每个病灶的逐层框（members），并断言与注册表逐条相等（file、z0、z1、x0、y0、x1、y1），保证 `lesion_id` 对齐。
- 图像：每卷 `reconstruction_rss` 转 `uint16`（按卷把 p99.9 映射到 65535），写 `volumes/<code>.u16`（小端，形状 slices×rows×cols）和 `volumes/<code>.json`（shape、spacing_row/col/slice mm、默认窗 = [p1, p99.5] 换算后的值）。`code` 是 stem 的稳定哈希前 8 位，医生看不到 stem。
- `lesions.json`：每条 `{lesion_id, code, volume_code, z0, z1, boxes: {z: [row0, row1, col0, col1]}}`（RSS 帧，行从顶部数），**不含**标签文字、距离、分层、系列、患者号。
- 顺序：每位读者 `orders` 表一份，种子 = 读者编号；pilot 150 例（`data/level_r/pilot_150.json`）打乱后排最前，其余打乱排后。

## 5. 读者界面（`anatobind/level_r/app/`）

- 入口 `/?token=<16 位十六进制>`；服务器按 token 找读者，token 只存哈希。
- 任务页：已完成 / 总数、"下一个"；已提交的可从列表回看并修改（改动追加新行，统计用最后一行）。
- 病灶页：
  - 看图：canvas 画当前层（16 位数组按当前窗宽窗位映射），框用绿色矩形；上下键或滑条翻整卷；放大按钮在框附近 2×/4×；拖动改窗宽（横）窗位（纵），"复位"回默认窗；右侧两张小图是上一层、下一层（同窗）。
  - 表单（下拉/勾选，中文界面，英文键值入库）：

```
primary_host       white_matter / cortex / thalamus / basal_ganglia / brainstem / cerebellum / other
acceptable_hosts   多选 ≤ 2，必须包含 primary_host
topography         periventricular / juxtacortical / cortical / deep_white_matter / infratentorial
adjacency          多选: adjacent_to_cortex / adjacent_to_ventricle / crosses_boundary / none
ambiguity          certain / two_host / multi_structure / insufficient_resolution
not_a_lesion       勾选后 primary_host 与 acceptable_hosts 可空
local_quality      good / fair / poor
confidence         1–5
comment            自由文字
```
  - 提交校验（前后端都做）：`not_a_lesion` 为否时 `primary_host` 必填且在 `acceptable_hosts` 内；`acceptable_hosts` ≤ 2；`confidence` ∈ 1..5。
  - 自动记录：`time_seconds`（页面打开到提交）、`ts`、`reader_id`、`window_json`（提交时的窗宽窗位）。
- 读者 token 访问不到其他读者的答案（API 按 token 限定）。

## 6. 裁定界面

- 裁定人 token（角色 adjudicator）。列表 = 两位读者都已提交且满足任一条件的病灶：主结构不同；一人 `not_a_lesion` 另一人不是；两人可接受集合并集 > 2。
- 同一看图组件；表单下方并排"读者 1 / 读者 2"的最后一次答案（匿名，顺序固定）。裁定人填同一套字段作最终答案 + `reason`。
- 最终标签（`store.final_labels()`）：不在裁定列表的 → 一致答案，`acceptable_hosts` 取并集；在列表的 → 裁定表最后一行；未裁定的记 `pending`，不进统计。

## 7. 存储与封存

- SQLite `derived/level_r/level_r.sqlite`：
```
readers(reader_id TEXT PK, role TEXT CHECK(role IN ('reader','adjudicator')), token_hash TEXT, display TEXT)
lesions(lesion_id INT PK, code TEXT, volume_code TEXT, z0 INT, z1 INT, boxes_json TEXT)
orders(reader_id TEXT, position INT, lesion_id INT, is_pilot INT)
labels(row_id INTEGER PK, reader_id, lesion_id, primary_host, acceptable_json, topography, adjacency_json,
       ambiguity, not_a_lesion INT, local_quality, confidence INT, comment, time_seconds REAL, window_json, ts)
adjudications(row_id INTEGER PK, adjudicator_id, lesion_id, primary_host, acceptable_json, topography,
              adjacency_json, ambiguity, not_a_lesion INT, reason, ts)
```
  `labels`/`adjudications` 只追加，没有 UPDATE/DELETE。
- 导出（`level_r_admin.py export`）：`derived/level_r/export/labels_<reader>.csv`、`adjudications.csv`、`final_labels.csv`（含 `status ∈ {agreed, adjudicated, pending}`）。
- 折：`data/level_r/folds.json` = 165 名患者（h5 `patient_id`）按种子 0 分五折，断言患者不跨折（复用 `assert_folds_by_patient`）。
- 封存（`level_r_admin.py seal`）：把 `final_labels.csv` 按折拆成 `derived/level_r/sealed/labels_fold{k}.csv`，写 `data/level_r/sealed_manifest.json`（每折 sha256、行数、时间）。`anatobind/eval/level_r_labels.py`：`load_train_labels(k)` 读另外四折并核对 sha256；`load_test_labels(k, unblind=False)` 不带 `unblind=True` 就抛错，带了则追加一行到 `derived/level_r/sealed/access_log.txt`（时间、折、调用栈顶层文件）。开发期只允许在内层用训练折。
- 备份：每次读片结束跑一次 `export`，并把 `level_r.sqlite` 复制一份带日期到 `derived/level_r/backup/`（只增不删）。

## 8. 一致率统计（`anatobind/eval/level_r_stats.py`）

对每个病灶取两位读者的最后一次答案（`not_a_lesion` 的按 `other` 之外的一个专门类 `not_a_lesion` 参与主结构统计）：

- raw 一致率 = 主结构相同的比例，95% 区间按患者 bootstrap（2000 次，种子 0）。
- Cohen κ、Gwet AC1（`AC1 = (p_a − p_e) / (1 − p_e)`，`p_e = Σ_c π_c (1 − π_c) / (K − 1)`，`π_c` 为两位读者合并的类频率）。
- 每类 positive agreement `2·n_agree(c) / (n_1(c) + n_2(c))`，混淆矩阵。
- 集合值一致率 = 两位可接受集合有交集的比例。
- 分层：距离四档（由注册表 `d_interface_mm`：0 / (0, 2] / (2, 4] / > 4）× 采集分层（`stratum_geometry`），每层各报上述量；3 mm 层厚卷（43 例）单列。
- 用时：每位读者 `time_seconds` 的中位数与四分位；换算全集工时。
- 门（R7）：全体 raw 的 95% 区间下限 ≥ 0.80 且 0 mm 档 raw ≥ 0.70 → 单一主结构可作次要终点；否则主终点只用集合值（v2.6 §7.8 本来就是集合值）。

`scripts/level_r_report.py` 把以上写成 `docs/verification/<日期>/level_r_pilot.md`（表格 + 命令 + 原始输出）。

## 9. pilot（`scripts/level_r_pilot_sample.py`）

- 输入注册表；格子 = 距离四档 × `stratum_geometry` 四层（16 格，允许空格）。
- 分配：150 按各格人口比例取整，非空格子至少 8（不足 8 的格子全取）；同一患者最多 3 例（超出的换同格下一个）；种子 0，可复现。
- 输出 `data/level_r/pilot_150.json`（lesion_id 列表 + 每格计数），入库。
- 流程：两位读者先完成 pilot → 跑报告 → 门过就在同一工具里继续（顺序已排好），门没过先回 v2.6 §3 本体与表单再议，不硬推。

## 10. 部署与安全

- 服务默认绑定 `127.0.0.1:8790`；医生通过用户提供的端口转发或内网地址访问（`--bind 0.0.0.0` 显式打开）。数据是公开的 fastMRI，无患者隐私字段进工具。
- token 16 位十六进制随机，`level_r_admin.py add-reader --role reader --display "读者 1"` 生成，明文只打印一次。
- 服务用 `setsid nohup` 起，日志 `derived/level_r/server.log`；崩溃不丢数据（sqlite 每次提交即落盘）。

## 11. 测试（每个函数先写测试）

- 导出：`.u16`/`.json` 能读回、形状与卷一致；`lesions.json` 与注册表逐条对齐；`blind.py` 白名单测试——发出的病灶/卷 JSON 里不含 `label / d_interface / delta_d / stratum / series / patient_id / stem / file` 任何键。
- 服务：测试里在随机端口起一个实例（线程），验证：无效 token 403；读者只拿到自己的顺序与答案；表单校验（缺主结构、集合超 2、集合不含主结构、信心越界）返回 400 且不入库；提交后 `labels` 多一行；再次提交同一病灶再多一行；裁定列表只含分歧（四种情形各一个用例）；裁定后 `final_labels` 取裁定。
- 存储：`labels` 无 UPDATE/DELETE 路径（静态检查 store.py 里不出现这两个词 + 行为测试）。
- 封存：sha256 清单对得上；改一个字节后 `load_train_labels` 报错；`load_test_labels(k)` 不带旗标抛错；带旗标读到并写访问日志一行。
- 统计：κ、AC1、positive agreement 用手算过的 3 类小表核对；bootstrap 区间包含点估计且长度随 n 缩小；分层表每层行数之和等于总数；集合值一致率在人造数据上正确。
- pilot 抽样：格子下限、患者上限、总数 150、同种子两次结果相同。

## 12. 明确不做

模型预测进工具；任何 B 基线；3D Slicer 模块；离线包；分割/掩膜标注（只标关系字段）；多语言界面（中文）；账号系统（token 即身份）。
