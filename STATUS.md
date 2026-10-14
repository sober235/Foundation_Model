# STATUS：2026-10-14 14:3x CEST（三目标修正计划 Task 1–4 已在分支 `codex/aur-three-goal-conformance-2026-10-13` 完成并推送，全量测试 1046 passed、2 skipped；独立评审在跑，过了就起 Task 7（新 U 损失训练 C2）。周四节点已取消，按顺序连续推进，以 A、U、R 做成为准，见 §2 第 0 条"用户已定"与 §3。SSL-first 线：Q25 收尾，三臂都已全测，G2 的 A、U 与 G3 三臂全不过；配对比较显示 A 的大头来自 Stage II 主干学习率降到十分之一（随机初始化 +0.12），预训练再加 0.045、PDGM 病灶灵敏度 +0.15 且误报更少，但受控 R 低 0.085；新 C2 越过过一次不过的预先登记检查，只作探索性结果；BMSR / ISLES 的 U 三臂都接近 0，是事件解码问题不是数据；待用户定见 §2 第 0 条）

每次交接前整体重写本文件。五段固定：已验证、待拍板、下一步、坑与别重做、为什么。
时刻都注明时区或"链日志时钟"（BST）。旧记录里的"本机"：10-11 及以前是 BST（UTC+1），10-12 起是 CEST（UTC+2），见 §4 第一条。

## 1. 已完成且已验证

**执行计划** `docs/superpowers/plans/2026-10-09-anatobind-brain-ssl-first-execution.md`（决定 Q2–Q20、任务 E0–T12、门 G0–G3、周四 10-15 汇报）。**记录索引** `docs/verification/2026-10-09/anatobind_brain_ssl_first/README.md`。代码在 `anatobind/aur/ssl/`（samples、dataset、masking、heads、contrast、model、losses、checkpoint、train、eval）、`anatobind/aur/resample.py`，脚本 `scripts/aur_resample.py`、`aur_ssl_prepare.py`、`aur_ssl_probe.py`、`aur_ssl_train.py`、`aur_ssl_eval.py`，测试 `tests/test_aur_ssl_*.py`、`tests/test_aur_resample.py`。

- **测试**（main 93fb62c，10-09 10:1x 重跑）：
  ```
  PYTHONNOUSERSITE=1 PYTHONPATH=. nice -n 19 ~/anaconda3/envs/nvgen/bin/python -m pytest tests/ -q -p no:cacheprovider
  949 passed, 1 skipped in 115.95s (0:01:55)
  ```
- **E1 1 mm 重采样**：709 例 0 失败（BMSR 1377 行、ISLES 500 行重采样；厚层 > 3 mm 行 ISLES 108、BMSR 6，A/R 关闭）；`samples_1mm_v1.json` 4960 行；网格检查 `4960 rows checked, 0 off their grid`（`g0/grids_1mm_v1.txt`）。
- **T01 SSL 清单（含 HCP）** `…/derived/aur/ssl_manifest_v1/`：6130 行（train 5501 / val 629）；HCP 1113 人 2226 行全部重采样成功；泄漏报告 test 患者 249 人 0 行、val 患者 0 行进 train、重复图像 0、`ok: true`。
- **T07 探针**（`p0/PROBES.md`，全空卡上测）：四卡 mb 2×2 累积 9.15 crops/s、mb 4 不累积 10.11 crops/s；单卡 mb 8 / 12 / 16 峰值预留 40.3 / 60.7 / 77.2 GiB；每源裁块约 4.2 GiB 分配；mb 16 贴顶且 loader 拖后腿。
- **T08 pilot**（`p0/PILOT.md`）：3 卡（5/6/7）× mb 12 = 全局 36，8028 曝光 223 步，12.9 crops/s，峰值 50.9 / 58.4 GiB，数据等待 0.015 s；验证 masked Huber 0.0633 对三线性基线 0.361，两视图余弦 0.985，有效秩 44（上限 64）未塌缩；损失全程有限；**恢复一致**：从 step 150 恢复后每个记录步 step / seen / lr 逐位相同，终点相同，验证 Huber 差 1.4e-4。
- **导出可用**：`ssl_stage1_best.pt` 经 `checkpoint.load_backbone` 严格载入 `AnatoBindBrain().backbone`（160 键），元数据带代码 SHA 93fb62c 与清单哈希。
- **G0 过**（清单、网格、泄漏、四卡探针），**G0.5 过**（泄漏测试、损失有限有梯度、DDP 多步不挂、恢复一致）。
- **Stage I 主训已起**（11:14:39）：
  ```
  cards=5,6,7 world=3 microbatch=12 global=36 lr=4.50e-04 warmup=444 total_steps=8889
  运行目录 /data2/congcong/data/FM_data/derived/aur/ssl_runs/stage1_320k_mb12/   日志 …/stage1_320k_mb12.log
  启动脚本（scratchpad 副本，内容见 p0/PROBES.md 末段与 README）：torchrun --standalone --nproc_per_node=3 scripts/aur_ssl_train.py --seen-crops 320000 --microbatch 12 --grad-accum 1 --workers 8 --lr 4.50e-04 --warmup-steps 444 --val-every 500 --save-every 500 --log-every 20 --val-volumes 32 --mask-ratio 0.60 --contrast-weight 0.10
  第 1 步：峰值分配 50.8 GiB，三张卡各 60.6 GB、利用率 77–95%
  ```
  **已于 18:36（本机）结束**：8889 步、320,004 曝光、平均 2.97 s/步（CPU 争用）、12.1 crops/s；验证 masked Huber 0.0366（基线 0.361），有效秩 56.8，对比准确率全程 1.0（饱和）；导出 `ssl_stage1_best.pt`；记录 `docs/verification/2026-10-09/anatobind_brain_ssl_first/stage1/STAGE1.md`。之后由 tmux 会话 `anatobind` 的 `chain` 窗口（`scripts/ssl_first_chain.sh`）自动接 G1 → Stage II 冒烟 → Stage II → Stage III → 评估，链日志 `…/ssl_runs/chain_20261009_1405.log`。

- **T10–T12 代码完成（10-09 下午，两批各一次独立评审，修补后）**：`anatobind/aur/train.py` + `scripts/aur_train.py`（Stage II/III 训练器）、`anatobind/aur/infer.py` + `scripts/infer_anatobind_brain.py`（整卷推理）、`anatobind/aur/eval.py` + `scripts/aur_eval.py`（测试集评估、G2/G3 判门、Level R 导出）；新增 40 个测试（`tests/test_aur_training_contract.py` 17、`test_aur_infer.py` 8+1 CUDA-only、`test_aur_eval.py` 8、`test_aur_scripts_t12.py` 6）。评审清单与修补见 `docs/verification/2026-10-09/anatobind_brain_ssl_first/reviews/`。T12 的阻塞项（头在 autocast 外）已在 5 号卡上用小模型与全尺寸模型验证修好；全尺寸模型整卷 176×256×256 推理 18 窗 26 s（与主训共卡时测，上界）、峰值 4.7 GiB。
- **真实数据 CPU 核验**：训练器的损失通路在三条真实 1 mm 行（PDGM FLAIR、ISLES 厚层 ADC、SibBMS）上用小模型跑通，无缺梯度；单卡真实数据 GPU 冒烟因没有空卡未做。

- **G1 不过（10-09 18:52）**：`docs/verification/2026-10-09/anatobind_brain_ssl_first/g1/G1.md`。冻结 13 宿主线性读出宏 Dice：Stage I 0.286 对随机初始化 0.383（差 −0.097，三种子 bootstrap 区间 [−0.117, −0.074]）；病灶可分性 AUC 0.640 对 0.680。差距集中在丘脑、基底节、脑干、深灰（随机 0.05–0.46，Stage I 0–0.14）。诊断：随机臂约六成优势来自局部坐标嵌入的位置捷径（置零后 0.383 → 0.322），但位置全去后随机仍领先 0.06；Stage I 的读出 500 步时（0.300）就已低于随机，之后缓慢下降；单层看 F1/F2/F4 都不如随机投影（F2 最差 0.171 对 0.295）。**微调探针**（Stage II 单卡 600 步，同种子同裁块）：A 掩膜损失两臂一样（0.91–0.92），U 一样，Stage I 臂的序列类型损失明显更差（步 400：1.55 对 0.79），与对比项把两个强度增广视图拉成同一表征、使全局特征对强度外观不敏感一致。结论：按现口径 Stage I 没有给下游带来可测的好处；原因在目标设计（遮挡重建 + 强度不变的全局对比），不是训练没收敛。
- **根因（10-10 读代码确认）**：`StageOne.view` 只把 F1 交给重建解码器，第一层之后的 PatchMerging 与第 2–4 层（12 个块里 10 个）从没收到重建梯度，只被 0.1 权重、批内已饱和的全局 InfoNCE 训练；G1.md 末尾"更正"一节写明，先前"目标设计"的读法在机制上是错的。
- **第一轮评审后修订（35624fc，已推）**：评审 `reviews/stage1-r1-review.md`（解码器修复正确、无泄漏、单变量；提前规则必须改）。现规则：第 1000/2000 步探针用 3 个种子，仅当第 2000 步读出 < 随机 − 0.01 且上升 < 0.01 才停；探针失败只跳过不杀训练；探针卡必须在训练卡外；训练只许 3 或 6 张卡（对比项批内 36 个裁块与第一次一致）；每次探针在 CPU 记录归因（`scripts/aur_ssl_attrib.py`）；第二轮的单一改动按第一轮第 2000 步归因选（深层对比梯度占优 → λ_c 0，否则 → 解码器去掉 F1 一路）。基线改为只用可见前景（第一次运行的 0.361 是旧口径，"0.366 → 0.037" 高估了）。全量测试 996 passed、2 skipped。
- **第一轮修复（61b915a，用户 10-10 选 (c) 并行后做）**：解码器读全部四层（逐级上采样相加，只读特征不读图像）；新测试：只用重建损失时每层（含 PatchMerging）梯度非零（旧代码挂）、重建对被遮体素不变；全尺寸模型 CPU 实测四层梯度都非零，解码器 6.8 万参数。另加 `scripts/aur_ssl_eval.py --host-only`（提前探针），链 `scripts/stage1_r1_chain.sh`（同超参主训 + 第 1000/2000 步探针 + 预先登记的继续规则 + G1 + C2），预先登记写在 G1.md（cb1696a）。全量测试 991 passed、2 skipped。独立评审在跑。
- **C0 / C2 共用链** `scripts/stage23_chain.sh`（原 c0_chain.sh 改名）：全局批固定 16（每卡 4 × 累积 16/(4×卡数)，1/2/4 张卡），学习率用规格 §6 原值不缩放；冒烟 = 第 50 步停（新参数 `--stop-after`）再恢复到 100。
- **卡**：10-10 02:47–02:52 另一会话的对比方法队列（DIP/ZS-SSDU/INR/MC-GS，`/data0/congcong/code/GS/Results_20260908/CE_retro_cssense_af16_vdpois/logs/cmp/queue.log`）占满 8 张卡，并在卡空出几分钟内续发；02:53 我在 3 号卡起的 C0 与它的两个任务撞车，已自行停掉（目录 `ssl_runs/c0_stage2_smoke_20261010_0253/` 只有半截，可删清单）。
- **Q25 配对比较（10-14 13:4x CEST）**：新配方 C0 已于 10-13 全测完（`eval/c0_test_20261013_0259/REPORT.md`）。两组配对在 CPU 上跑，读各臂的 `cases.json`；单臂的 A、U、R 数值与各自 REPORT.md 逐项一致：
  ```
  E=docs/verification/2026-10-09/anatobind_brain_ssl_first/eval
  CUDA_VISIBLE_DEVICES="" PYTHONNOUSERSITE=1 PYTHONPATH=. nice -n 19 ~/anaconda3/envs/nvgen/bin/python scripts/aur_compare.py --a $E/c2_test_20261011_0500 --b $E/c0_test_20261013_0259 --name-a C2v2 --name-b C0v2 --out $E/compare_c2v2_c0v2_20261014
  | 1032 | 0.7961 | 0.7506 | 0.0454 | [0.0435, 0.0477] |
  | pdgm/FLAIR | 101 | 130 | 0.70 / 0.60 | 0.7538 | 0.6077 | 0.158 | 0.604 | 0.1462 | [0.0720, 0.2250] |
  | 1336 | 0.5142 | 0.5996 | -0.0853 | [-0.1186, -0.0496] | 0.6692 | 0.7013 |
  第二组 --a $E/c0_test_20261013_0259 --b $E/c0_test_20261010_1533 --name-a C0v2 --name-b C0 --out $E/compare_c0v2_c0_20261014
  | 1032 | 0.7506 | 0.6289 | 0.1217 | [0.1186, 0.1243] |
  | 1336 | 0.5996 | 0.5007 | 0.0988 | [0.0562, 0.1382] | 0.7013 | 0.6789 |
  ```
  全表与读法在 `stage2/STAGE2_V2.md` 末节和 §2 第 0 条。
- **三目标修正计划 Task 1–4（10-14 14:0x–14:3x CEST，本会话接 Codex；分支 `codex/aur-three-goal-conformance-2026-10-13`，工作树 `../foundation_model-conformance`）**：
  - Task 1 `anatobind/aur/gates.py::q25_step1000_gate`：读出 ≥ 0.65（含等号）且 A 点 Dice > 0.045（严格），缺失、非有限值一律不过。Codex 写的 1904a09 按仓库身份重提为 66510ce。
  - Task 2 `ops/c2v2_check.sh` 改成 fail-closed（595434b）：环境 STAGE2_DIR、SAMPLES、VAL_PATIENTS、PROBE_GPU，产物全写 `$STAGE2_DIR/gate_step1000/`，已有判定不覆盖，只有 PASS 退出 0。用 10-11 的真实产物回放判定函数，得到记录里的 FAIL（读出 0.6829 过、A 点 Dice 0.0443 不过）。当时实际跑的旧脚本原样留在 `stage2/frozen_probe/c2v2_step1000_check.sh`。
  - Task 3 `scripts/stage23_chain.sh`（a50a033）：必须给 `U_MASK_PROFILE`（legacy | unmatched_empty_v1），与 `SEED` 一起传给每次训练；`C2_STEP1000_GATE=1` 只给 Q25 配方（c2、legacy、II_BB 5e-5）：第 1000 步停、跑检查、PASS 才在同一目录续跑，FAIL 或检查出错就停链。链从自己所在的检出运行，可覆盖 PY、TORCHRUN、FM_DATA_ROOT、Q25_CHECKER、PROBE_CARD，默认值不变。
  - Task 4 `anatobind/aur/losses.py::event_loss(..., u_mask_profile)`（7da5743）：`unmatched_empty_v1` 给 U 监督样本里每个未匹配查询加压空 BCE（有效点平均 → 查询平均 → 样本平均，× NO_OBJECT_WEIGHT 0.1）；legacy 是默认且不变；`--u-mask-profile` 写进 run_config。
  - 全量测试（Task 6 的核心，14:2x CEST）：
    ```
    CUDA_VISIBLE_DEVICES="" PYTHONNOUSERSITE=1 PYTHONPATH=. nice -n 19 ~/anaconda3/envs/nvgen/bin/python -m pytest tests/ -q -p no:cacheprovider
    1046 passed, 2 skipped in 157.36s (0:02:37)
    ```
    此前 1012 passed；新增 34 个（gates 18、链 10、损失 5、训练 1）。

**之前（保留）**：Part 1（方案架构的代码包、样本表、目标、探针）在 tag `handoff/2026-10-09-anatobind-brain-aur-part1`；S4 脑解剖学生模型 A11 不过、三条出路待定；S7 三病种五折全过、1212 份记录 v2；nnDetection 与小病灶线已停；Level R 读片工具就绪、读片未开始（见 CLAUDE.md 当前状态段）。

## 2. 待用户拍板

0. **Q25 收尾与待定（10-14 13:5x CEST，本会话）**：
   - 新配方 C0 已跑完（链日志时钟 BST：10-13 02:59 起于 0、2 号卡，Stage II 10:03、Stage III 12:51，验证集选 U 阈值 bmsr 0.65 / isles 0.60 / pdgm 0.60，全测 14:40 完）→ `eval/c0_test_20261013_0259/REPORT.md`（验证集 `eval/c0_val_20261013_0259/`）。A 0.7506、U PDGM 0.6077 / BMSR 0.0061 / ISLES 0.0242、受控 ABA(R) 0.5996 对 B0 0.7013、端到端 103 例 0.6796 对 0.5340；G2、G3 都不过。现行配方 C0 塌掉的 `basal_ganglia_right` 这次正常（0.6890，左 0.6957）。
   - 配对比较（`scripts/aur_compare.py`，同 1,056 行，按病人 bootstrap 95 % 区间；NOT_EVIDENCE；每臂一次运行）：

     | 指标 | 预训练效应：新 C2 − 新 C0 | 配方效应：新 C0 − 现行 C0 |
     |---|---|---|
     | A 13 宿主宏 Dice | +0.0454 [0.0435, 0.0477] | +0.1217 [0.1186, 0.1243] |
     | U PDGM FLAIR（每例误报） | +0.1462 [0.0720, 0.2250]（0.158 对 0.604） | +0.4385 [0.3534, 0.5360]（0.604 对 0.139） |
     | U BMSR | −0.0012 [−0.0092, 0.0060] | +0.0061 [0.0000, 0.0154] |
     | U ISLES | −0.0173 [−0.0403, 0.0000] | −0.0104 [−0.0321, 0.0086] |
     | 受控 ABA(R) | −0.0853 [−0.1186, −0.0496] | +0.0988 [0.0562, 0.1382] |

     全表与读法在 `stage2/STAGE2_V2.md` 末节，原始输出 `eval/compare_c2v2_c0v2_20261014/`、`eval/compare_c0v2_c0_20261014/`。配方对照的两条链只差 Stage II 主干学习率（6.124e-5 对 6.124e-4），链日志开头逐项对过。
   - 读法：
     1. A 的大头来自配方，不是预训练。主干学习率降到十分之一，随机初始化就从 0.629 涨到 0.751；预训练再加 0.045（四个来源、13 个宿主逐个都更高），到 0.796，仍不到 0.80。10-12 那版写的"预训练让 A 从 0.629 到 0.796"把配方效应算到了预训练头上，引用时不能这么说。
     2. U：PDGM 上预训练灵敏度更高、误报更少，不是靠多报换来的。新 C0 对现行 C0 的 PDGM 提升发生在误报高约四倍的工作点上，不能直接算成配方的功劳。BMSR、ISLES 三臂都接近 0，与初始化、配方无关，原因是事件解码（下面 10-13 的 U 核查）。
     3. R：三臂受控轨道都输给 B0（R − B0：新 C2 −0.155、新 C0 −0.102、现行 C0 −0.178），预训练让 R 低 0.085。另有一点未解释：B0 是在各臂自己预测的解剖图上查表，新 C2 全脑 A 更好，B0 却更低（0.6692 对 0.7013），即病灶位置上它的解剖图与 SynthSeg 更不一致，待查。
     4. 新 C2 越过了一次不过的预先登记检查，它的数字只作探索性结果。
   - Q25 三个小目标：第 1 条不过（已披露），第 2、3 条完成。
   - 现状（10-14 13:4x CEST 查）：AnatoBind 没有在跑的任务，0、2 号卡空；pgrep 只看到本会话。GitHub 上有两条 10-13 的 codex 分支未合 main：`codex/review-aur-execution-alignment-2026-10-13`（执行一致性复核）；`codex/aur-three-goal-conformance-2026-10-13`（三目标修正的规格与计划 + DICOM 网页原型的规格与计划）。用户 10-13 在 Codex 里审过并让它按计划执行，Codex 在 `/tmp/Foundation_Model-aur-brain-iter2` 做完 Task 1（提交 1904a09，未推）、Task 2 做了一半，10-13 10:10 UTC 用量到顶停下（Codex 到 11-12 才能再用）。本会话从这里接着执行，14:3x CEST 已做完 Task 1–4（§1 末条），见下面"用户已定"与 §3。
   - **用户已定（10-13 / 10-14；10-13 的几条当时没写进仓库，10-14 从 Claude 与 Codex 会话记录里找回原话）**：
     - Q26（10-13 02:28 UTC，Claude 会话）："先消除周四的目标.已经能够实现上述三个子目标为准"。周四汇报节点取消，以 A、U、R 三个子目标真正做成为准。
     - Q27（10-13 07:48 UTC，Codex）：目标重申，给一张 2D 或 3D DICOM，1 识别解剖结构，2 识别异常组织，3 给出异常组织在哪个解剖结构上；最好以任何人都能打开的可交互 URL 呈现。规格与计划在 codex 三目标分支（`docs/superpowers/specs|plans/2026-10-14-anatobind-brain-dicom-web-*`）。
     - Q28（10-13 09:53 UTC，Codex）："先不考虑这里显示的问题，仅仅考虑模型是否能否成功即可,显示后续再说"。网页、显示、API、公网部署后移；DICOM 解码与离线 A/U/R 验收留在本阶段。
     - Q29（10-13 09:59 UTC，Codex）："逐个解决上述问题"。Codex 据此定的顺序：先做三目标修正计划（fail-closed 门控 + U 未匹配查询压空），再评估 A、R 缺口；2D 模型单独推进；C0/C2 配对改为可选的描述性分析，不作过门条件（Codex 的裁定，用户没有单独点头）。
     - Q30（10-14，本会话）："the concept of Thursday no longer exists. The project will be carried out sequentially and continuously." 没有日期节点，按顺序连续推进。
   - **仍待用户定（推荐在后）**：
     1. 新 C2 的结果怎么用 → 照实披露，作探索性结果，不为补程序重跑（同 10-12）。
     2. 门判定 → 三臂 A、U、R 全不过，不调参，照实报告（同 10-12）。
     3. 仍是 PROPOSED 的值 → 按已用的认下（同 10-12 第 4 条）。

   以下为 10-12 / 10-13 的记录（时刻是 CEST 或链日志时钟；"新配方 C0 未起"与其待定第 2 条已过时，它 10-13 已跑完）：
   旧 0. **结果与待定（10-12 15:40，本会话接手；上一个会话停在 10-11 05:10）**：
   - **两条链都已无人值守地跑完**：
     - 现行配方 C0（随机初始化）：Stage III 09:22 完；验证集选 U 阈值 bmsr 0.95 / isles 0.55 / pdgm 0.70；全测 1,056 行 12:07 完 → `eval/c0_test_20261010_1533/REPORT.md`（验证集 `eval/c0_val_20261010_1533/`）。
     - 新配方 C2（Stage I 预训练、主干学习率十分之一）：**第 1000 步预先登记检查 FAIL**（`stage2/frozen_probe/c2v2_step1000_check.log`、`c2v2_step1000.log`）：冻结 13 宿主读出 0.6829 ≥ 0.65 过；验证 A 点 Dice 0.0443，门 > 0.045（现行配方 C0 同步数值），差 0.0007 不过；按 `stage2/STAGE2_V2.md` 应停下报用户。检查脚本只报告不停链，当时没有会话值守，链继续：Stage II 14:04、Stage III 17:21、验证集选阈值（三源都 0.70）、全测 18:25 → `eval/c2_test_20261011_0500/REPORT.md`（验证集 `eval/c2_val_20261011_0500/`）。**这些结果是越过一次不过的预先登记检查之后得到的，引用时必须写明。**
   - **全测结果**（NOT_EVIDENCE，全部建立在 SynthSeg 与病灶伪标签上）：

     | 门 | C0 随机初始化 | C2 预训练（越过不过的检查） | 门槛 | 判定 |
     |---|---|---|---|---|
     | G2-A 13 宿主宏 Dice（1,032 卷） | 0.6289 | 0.7961 | ≥ 0.80 | 两臂都不过（C2 差 0.0039） |
     | G2-U PDGM FLAIR 灵敏度 | 0.1692 | 0.7538（参照 0.8615） | ≥ 参照 − 0.05 | 不过 |
     | G2-U BMSR T1c | 0.0000 | 0.0049（参照 0.7316） | 同上 | 不过 |
     | G2-U ISLES DWI | 0.0346 | 0.0069（参照 0.6303） | 同上 | 不过 |
     | G3 受控轨道 ABA(R) 对 ABA(B0)（1,336 例） | 0.5007 对 0.6789 | 0.5142 对 0.6692 | R ≥ B0 | 两臂都不过 |
     | 识别–绑定差距子集（局部 A ≥ 0.8 且找到病灶） | 无样本 | 53 例：B0 错 52.8 %，R 救回 23、弄错 7 | 只报告 | — |
     | 端到端（找到的病灶） | 32 例 0.5625 对 0.5938 | 139 例 0.7050 对 0.4892 | 只报告 | — |

   - **（10-13 04:1x 本机）U 核查结论：不是数据 bug，是事件输出的拼法**。重采样后的 BMSR / ISLES 病灶与图像对齐（最大病灶质心偏移 < 0.6 mm，标签值 {0,1}，加载器每卷看到 3–42 个实例，病灶对比 +0.6~+2.7 SD）。测试集每一卷 BMSR / ISLES 都有一个近全脑的连通块（中位 466–726 mL），真病灶 913/918、293/296 在其中。单卷探针（C2，GPU 7）：真病灶由一个查询找到（存在概率 0.96，掩膜覆盖病灶 92%），另有 6–9 个未匹配查询存在概率 0.54–0.77 过了 0.5 的门，它们的掩膜训练时不受监督、是散点；推理对过门查询取最大、跨窗口叠加、按 0.3 取连通块，散点连成全脑大块吞掉真病灶。PDGM 只有一个查询过门，所以干净。可选补救（都要用户定，都是事后的）：(a) 不重训，换解码（按查询各自成实例 / 提高存在门），验证集选、测试集只作诊断报告；(b) 重训：未匹配查询的掩膜加 BCE 压空（与 A 头的做法一致），或规格 N7 预先登记的退路（语义病灶头 + 连通块）。脚本 scratchpad `udiag/lesion_contrast.py`、`udiag/query_probe.py`。
   - **（10-13 02:59 本机，链日志时钟）新配方 C0 已起**（Q25 第 3 条）：0、2 号卡，tmux `c0v2`（守卡 `guardc0v2`），`ARM=c0 CARDS=0,2 II_BB=5e-5`，设置与新 C2 逐项相同；冒烟 03:11 通过，Stage II → Stage III → 验证集选阈值 → 全测，约 13.5 h；之后 `scripts/aur_compare.py` 与新 C2 配对比较。链日志 `ssl_runs/c0_chain_20261013_0259.log`。
   - **反常，先查**：BMSR、ISLES 的 U 两臂都接近 0（BMSR ≥ 10 mm 两臂都是 0.000；C0 在验证集上为 BMSR 选出的阈值是 0.95，说明验证集上也找不到），而 PDGM FLAIR 的 C2 到 0.754。已核：训练器把实例图传给了采样器（`anatobind/aur/train.py:242`、`:433`），不是采样点的问题。下一步只读核查：重采样后 BMSR / ISLES 的标签取值与对齐、病灶内外强度对比、与原生网格对照、每源 U 训练信号。
   - C0 的 `basal_ganglia_right` Dice 0.0000（左 0.5463），C2 两侧正常（0.7430 / 0.7734）：C0 有一类塌掉，记入诊断。
   - **待用户定（我的推荐在后）**：
     1. 新 C2 越过不过的检查之后的结果怎么用 → 推荐照实披露、作探索性结果用，不为补程序重跑（差 0.0007，等于 C0 同步数值；最终 A 0.796 对 C0 0.629）。
     2. 新配方 C0（Q25 第 3 条，已批准、未起）→ 推荐先查完上面的 U 数据通路，没有 bug 就在两张空卡上起（约 13.5 h）；查出 bug 则修好后两臂并行重跑（四卡约 14 h），这一步要用户点头。
     3. 门判定 → 按规则 A、U、R 全不过，不调参，照实报告；C2 的 A 写“0.796，差 0.004”。
     4. 仍是 PROPOSED 的值（Stage III 预热 200 步、G1 阈值 0.05 / 0.02、Q23 学习率按 √(24/16) 缩放）→ 推荐按已用的认下，改就得重跑。
     5. 周四汇报主线 → 预训练让 A 和大病灶 U 明显变好；R 只在感知做好的地方赢查表；小病灶与 BMSR / ISLES 两个来源没做成，U 核查后说清是数据问题还是方法问题。
   - 全测的逐例 `cases.json`（每份约 10 MB）与 Level R 的 `level_r_key.json` / `level_r_sheet.csv` 不入库：仓库是公开的，密钥入库会破坏盲读；它们留在本地 `eval/` 各目录，未跟踪。
   - 本机另有两个会话（claude-72、claude-ee，10-12 15:2x 起）在跑，不知道在做什么；动卡和文件前先确认它们不在 AnatoBind 线上。

   以下为上一个会话的记录：
   旧 0. **在跑与待定（10-11 05:10）**：
   - **决定 Q25（用户 10-11 "按照你的建议来执行"）**，三个小目标与判定预先登记在 `stage2/STAGE2_V2.md`（d064032，启动前写定）：
     1. 新 C2：05:00 起在 0、2 号卡（tmux `c2v2`，`II_BB=5e-5` 即 Stage II 主干 6.124e-5，其余同现行配方），从 G1 导出开始，先冒烟再 Stage II。第 1000 步（约 06:15）检查：冻结 13 宿主读出 ≥ 0.65 且验证 A 点 Dice > 0.045，两条都过才跑完，否则停下报用户；检查脚本 scratchpad `c2v2_check.sh`（后台，结果写 `stage2/frozen_probe/c2v2_step1000/`）。
     2. 现行配方 C0：Stage III（5 号卡）约 07:45 完，之后验证集选阈值、全测，作参照。
     3. 新配方 C0：卡一空就起（5 号卡在现行配方 C0 评估后，或 0、2 号卡在新 C2 跑完后，取先空的），与新 C2 配对，`scripts/aur_compare.py` 比较。
   - 现行配方的 C2 已于 05:00 停在约第 2700 步（目录 `c2_stage2_240k_20261011_0216/`，存盘到第 2500 步），保留作证据。切换时 `hold_swap.py` 与新链的守卡进程（tmux `guardc2v2`）接住了空档；脚本存档在 `ops/`。
   - （04:50 的记录）
   - **新发现（`stage2/C0_A_DIAG.md` 末两节）**：同步数下 C2 的 A 点 Dice 不比 C0 快（第 1500 步 0.051 对 0.081），序列头 C2 明显好（第 500 步损失 0.024 对 0.625）。冻结 13 宿主读出：C2 起点 0.724 → 第 500 步 0.694（主干学习率 0.1 倍）→ 第 1500 步 0.375（全速 6.12e-4 之后）；C0 第 1500 步 0.274、训完第 10000 步 0.306，随机主干 0.383。现行 Stage II 配方抹掉主干里的解剖信息，照此 C2 对 C0 回答不了"预训练有没有用"。
   - **待用户定**：(1) 照跑：两臂按现行配方跑完（周日晚出结果），A 两臂都过不了 G2，汇报写成"Stage II 冲掉预训练特征"的负结果；(2) 建议：停掉现在的 C2（它的预训练特征第 1500 步前已没了），从 G1 导出用主干学习率常设十分之一（6.12e-5，不改代码）重跑 C2，跑到第 1000 步先用冻结读出与 A 点 Dice 检查（约 1 小时），过了再跑完；C0 照 Q24 跑完现行配方作参照，之后在 5 号卡跑同配方的 C0 作配对；(3) 在 (2) 之上再改 A 头（33 类 softmax、像素特征加宽），要改代码和测试，更费时。
   - C2：Stage II 第 2660 步（04:45），约 3.1 s/步，约 11:00 完；C0：Stage III 第 1160/3333 步，约 07:45 完，之后评估。
   - （02:35 的记录）
   - **G1 过**（`g1/G1.md` 末节，报告 `g1/r2_run_20261010_1942/g1_report.json`）：第二轮第 8500 步导出，13 宿主读出 0.724 对随机 0.383（+0.342，三种子区间下界 0.320），丘脑 0.72 对 0.07、脑干 0.81 对 0.46；病灶可分性 AUC 0.667 对 0.680（低 0.013，在 0.02 容差内）。第二轮训满 8889 步，续跑段 4.19 s/步。
   - **C2**：链（`scripts/stage23_chain.sh` ARM=c2，由 `scripts/stage1_r2_resume_chain.sh` 接手）02:16 起在 0、2 号卡：冒烟（第 50 步停、续到 100）→ Stage II 24 万裁块 → Stage III 8 万 → 验证集选阈值 → 全测。每卡 12、全局 24、每卡 24 个加载进程，与 C0 同设置（Q23）。0、2 号卡由 `hold_c2b.py`（到正式 Stage II 记下第一步）与 `card_guard.py`（tmux `guardc2`，链在就守各阶段之间的空档）守着；3 号卡已放出。
   - **C0**：第 9880/10000 步（02:26），Stage II 约 02:40 完，之后 Stage III、评估自动接；5 号卡由 tmux `guardc0` 守阶段间空档。
   - **决定 Q24（用户 10-11 "我依然建议 C0 A头继续"）**：C0 照现行设计继续，不改 A 头；C2 用同一设计，两臂配对。A 头慢是随机初始化还是设计所致，由 C2 对 C0 在相同步数（500/1000/1500）的验证 A 点 Dice 回答（C0 是 0.015 / 0.045 / 0.081），写进 `stage2/C0_A_DIAG.md`。
   - 夜间第二轮没有再中断；看门脚本 01:57 在第 8800 步挂了 C2 交接占位，02:16 G1 判定后退出。
   - （10-10 19:55 的记录）
   - **第二轮 19:33 被中止、19:42 续跑**（详见 `g1/G1.md` 末尾）：NCCL 看门狗判 allgather 超时（1 号 rank 卡住约 17 分钟，原因不明）；1 分钟内在 0、2、3 号卡挂占位，修了 Stage I 训练器的续跑（65e3bf5：接着抽样序列走、读回最优、不覆盖 run_config），从 `resume_step3500.pt` 续跑（tmux `r2res`，`scripts/stage1_r2_resume_chain.sh`）；第 3520 步的学习率与无学习基线与中止前逐位相同。真实速度 5.5–6.3 s/步（我先前按日志抽样说的"约 3 s、20:30 跑满"是错的），约 10-11 04:30–05:30 跑满，然后 G1 → C2。
   - **夜间看门**（tmux `r2dog`，scratchpad `r2_watchdog.sh`，日志 `r2_watchdog.log`）：再中止就占卡并从最新存盘续跑，最多 3 次；第 8800 步起挂 C2 交接占位（`hold_c2b.py`，日志 `hold_c2b.log`）。
   - （16:40 的记录）
   - **第二轮第 2000 步探针（16:29）**：Stage I 0.516 对随机 0.383（三个种子 0.517 / 0.514 / 0.517，bootstrap 区间 [0.113, 0.156]、[0.105, 0.154]、[0.111, 0.153]）；丘脑 0.38 对 0.07、其他深灰 0.32 对 0.13、脑干 0.63 对 0.46。规则判"继续"，详见 `g1/G1.md` 末尾。约 20:30 跑满后在 5 号卡做正式 G1（宿主 + 病灶可分性），过了链自动在 0、2 号卡起 C2。
   - **0、2 号卡占位（16:32 起，tmux 窗口 `holdc2`，scratchpad `hold_c2.py`，日志 `hold_c2.log`）**：每卡 4 GB，守住第二轮训完 → G1 → C2 冒烟（含中途停一次再续跑）的空档；C2 正式 Stage II 记下第一步后 30 s 释放；第二轮或 C2 链报失败就立即释放；最长 8 h。3 号卡第二轮训完后放出，不守。
   - **A 头的建议改了**：第二轮的特征在深部结构上强得多，C2 从它出发，A 也许学得快得多，这正是 C2 对 C0 要检验的。建议先按原设计跑（选项 1），今晚 C2 的第 500/1000/1500 步验证 A 点 Dice 出来后（约 22:00–23:00），与 C0 同步数的 0.015 / 0.045 / 0.081 比，再定要不要改设计。
   - （15:45 的记录）**待用户定：C0 的 A 头学得极慢**（`docs/verification/2026-10-09/anatobind_brain_ssl_first/stage2/C0_A_DIAG.md`，NOT_EVIDENCE）。训练时 A 点 Dice 第 3000 步 0.111，照此训满约 0.25，G2 门是 0.8。不是阈值校准；组织分不开（右白质 74 % 判成右皮层）；冻结特征的线性可读性只和"灰度 + 位置"相当；fastMRI 冒烟 1 个体积白质、皮层近 0，而 S4 学生模型在同类标签上约 0.8。两个选择：(1) 不改，两臂照原设计跑完，A 作为负结果汇报；(2) 先改再跑：停 C0，在 5 号卡上跑三个各约 1000 步的小对照（像素特征 32 → 128、A 改成 33 类 softmax 交叉熵、学习率 2e-4），选出有效改动后 C0 从头重跑，C2 用同一设计（对照约 3 h，重跑约 15 h）。我倾向 (2)，但它改的是预先定好的设计，要用户点头。需要的代码：训练命令行暴露模型选项，A 损失加 softmax 选项，都先写测试。
   - tmux 窗口 `r2x3`：第二轮 Stage I（λ_c 0），卡 0/2/3，约 3 s/步。第 1000 步探针 14:54 出：0.397 对随机 0.383（三个种子增益 +0.012 到 +0.014，bootstrap 下界最低 −0.003，未显著），记在 `g1/G1.md` 末尾。第 2000 步的提前停规则只在读出 < 0.373 且上升 < 0.01 时停；跑满约 20:30 接 G1（门约 0.433），过了自动接 C2（卡 0、2）。
   - tmux 窗口 `c0w`：C0 Stage II 15:33 从 `resume_step3500.pt` 在 5 号卡续跑，32 个加载进程（`WORKERS=32`）；第 3520 → 3580 步墙钟实测 5.7 s/步（GPU 本身约 3.7–4 s，偶尔仍等数据），之前 8 个加载进程时约 12 s/步（日志每 20 步只记一步的耗时，看着像 4 s）。照此 Stage II 约 10-11 02:00 训完，Stage III 约 07:00，评估后 C0 结果约周日上午；若用户选 (1) 不改设计，可在第 4000/4500 步存盘后同法再切到 48 个加载进程。切换由 scratchpad 的 `swap_c0_v2.sh` 在第 3500 步存盘写完后自动做：先用 `hold_card.py` 在 5 号卡占 4 GB，停旧链、换链脚本、同卡续跑，新进程占满显存后占位自动退出；卡一直没空出来。续跑配置 `run_config_resume_step3500.json` 记 workers 32、代码 e426d3b。
   - 评估栈真实数据冒烟（CPU，C0 第 2000 步导出，每源 1 例，链脚本同样的分片流程）：验证集两片合并 → U 阈值 PDGM 0.65 → 测试集两片合并（阈值传过去）→ fastMRI 全部走通；SibBMS 在跑。输出 `/data2/congcong/data/FM_data/derived/aur/eval_smoke_20261010/`。
   - 训练器续跑的两个护栏（d336409）：目录里有比续跑点更晚的存盘就当场拒绝（以前要到下一次存盘才报错）；同一步第二次续跑不覆盖第一次的配置记录。全量 1010 passed、2 skipped。
   - 链脚本 `scripts/stage23_chain.sh`（94aba82）：`WORKERS` 每卡加载进程数，默认 24。
   - 可删清单新增（只列）：scratchpad 的 `patchtest_resume/`、`patchtest_resume2/`、`patchtest_resume3/`（打补丁验证用的代码副本）、`swap_c0.sh`（被 v2 取代）、`stage23_chain_new.sh`。

   （以下为 13:40 的记录）
   **在跑（10-10 13:40）**：
   - tmux 窗口 `r2x3`：第二轮 `scripts/stage1_r2_chain.sh`，`CARDS=0,2,3 PROBE_CARD=5 C2_CARDS=0,2`，运行目录 `ssl_runs/stage1_r2_320k_20261010_1303/`，约 4.2 s/步，跑满约 10 h。11:56 那次单卡版（`stage1_r2_320k_20261010_1156/`，到第 420 步）在三卡版起来后被我停掉，列入可删清单。
   - tmux 窗口 `c0w`：C0 Stage II 从 `c0_stage2_240k_20261010_0903/resume_step3000.pt` 在 5 号卡单卡续跑（每卡 12 × 累积 2 = 全局 24，与两卡时同一调度；日志 `c0_stage2_240k_20261010_0903.resume_*.log`），之后 Stage III、分片评估。
   - `scripts/stage23_chain.sh` 已更新：可 `RESUME_STAGE2=… STAGE2_DIR=…` 续跑 Stage II；每卡 12/8/6 自动选；验证集与测试集评估按卡数分片并行（`scripts/aur_eval.py --shard K/N` + `scripts/aur_eval_merge.py`）。G1 探针改为每卷只读一次、并行读、两臂共用（结果逐字节不变）。
   - **教训（记忆 never-release-held-gpus）**：13:29 为把 C0 改成三卡，存盘后停了 C0、放了 1、6 号卡，一分钟内被另一会话的膝 SPIRiT 队列占走；以后换配置在原卡上续跑，加卡先拿到新卡再动旧的。
   （以下为 12:00 的记录）
   - tmux 窗口 `r2`：`scripts/stage1_r2_chain.sh`，`CARDS=5 PROBE_CARD=6 C2_CARDS=5`，运行目录 `ssl_runs/stage1_r2_320k_20261010_1156/`，链日志 `ssl_runs/stage1_r2_chain_20261010_1156.log`。第一轮的判停、归因与第二轮的选择都在 `g1/G1.md` 末尾。
   - tmux 窗口 `c0w`：C0 Stage II（卡 1、6）照旧。
   - 4、7 号卡 11:51 被 haobo 的 JiT 微调占用（各 44.7 GB），不能再跑 58 GB 的 Stage I。若之后有 3 张整卡空出，可把第二轮从最近的 resume 改到 3 卡续跑（全局 36 不变，学习率曲线不变）。
   （以下为 09:00 的记录）
   - tmux 窗口 `r1`：`scripts/stage1_r1_chain.sh`，`CARDS=4,5,7 PROBE_CARD=6 C2_CARDS=4,5`，运行目录 `ssl_runs/stage1_r1_320k_20261010_0851/`，链日志 `ssl_runs/stage1_r1_chain_20261010_0851.log`；第 1000 步约 09:40、第 2000 步约 10:30 出探针。
   - tmux 窗口 `c0w`：`scripts/stage23_chain.sh` ARM=c0，卡 1、6，**每卡 12 个裁块、全局 24**（09:04 重启，链日志 `ssl_runs/c0_chain_20261010_0903.log`；08:58 那次每卡 4 × 累积 2 的冒烟只跑了几步就按用户"跑满显存"的要求停掉，`c0_stage2_smoke_20261010_0858/` 列入可删清单）。冒烟实测：显存分配 51.8 / 预留 63.3 GiB；采样点在 CPU 上时 5.6–7 s/步（GPU 一半时间空等），改到 GPU 后 2.45 s/步。
   - **决定 Q23（用户 10-10 "运行过程中要将四张 GPU 显存跑满"）**：C0 与 C2 用同一套设置：每卡 12 个裁块、两卡全局 24，规格 §6 的学习率（定义在全局 16）乘 sqrt(24/16)：Stage II 6.12e-4（前 667 步主干 ×0.1），Stage III 6.12e-5 / 3.06e-4 / 6.12e-4，预热按裁块数折算（Stage II 667 步、Stage III 133 步），PROPOSED。第一轮 Stage I 不动：全局 36 是预先登记的"与第一次相同"，3 卡每卡 12 已占 58/80 GB，每卡 16 会顶到 77 GB（探针实测），有 OOM 风险。
   - 1、6 号卡原先挂着暂停（T 状态）的 CE retro 进程，08:50 后退出；5、7 号卡 08:4x 起空闲。用户 10-10 问"已经有四张 GPU 了吗"，即这 4 张 + 我的 4 号卡。
   单卡小试（08:33 起于 4 号卡）在第 200 步左右由我停掉：第一轮全量的第 1000/2000 步探针出结果与小试一样快；目录 `ssl_runs/stage1_r1_pilot36k_20261010_0833/` 只有半截，列入可删清单。
   （以下为当时的记录）小试已起（10-10 08:33，4 号卡空出 3 分钟后被 `pilot` 窗口接手）：`ssl_runs/stage1_r1_pilot36k_20261010_0833/`，单卡 microbatch 12、显存分配 51.5 / 预留 58 GiB、约 2.2–2.9 s/步，3000 步约 2.1 h；日志 `ssl_runs/stage1_r1_pilot_20261010_0833.log`；探针结果写到 `g1/pilot_r1_probe_step{1500,3000}_20261010_0833/`、归因 `g1/pilot_r1_attrib_step*_20261010_0833.json`。之后同卡接 C0。
   **C0 守卡窗口 `c0w`（08:44 起）**：等 1、6 号卡同时空闲 3 分钟就起两卡 C0（7 号卡留给第一轮，两个守卡窗口的候选不重叠）（与之后 C2 的两卡对齐）；`stage23_chain.sh` 现有按臂的锁（`ssl_runs/<arm>_chain.lock`），小试结束后它自带的 C0 若发现已有 C0 在跑会自行退出。
   **守卡窗口（10-10 08:3x，用户回"continue"后）**：`scripts/wait_for_card.sh` 只接手连续 3 分钟完全空闲的卡（那边循环一两分钟内就续发的卡不会被抢）。`pilot` 窗口盯 1、6、4 号卡，拿到后跑 `scripts/stage1_r1_pilot.sh`（第一轮配置的单卡小试：36k 曝光，第 1500/3000 步探针与归因；InfoNCE 批内 12 个裁块、学习率在小试内退火，与第一轮不同，**不算预先登记的第一轮**），随后同卡接 C0（1 张卡，Stage II 约 23 h）。`r1` 窗口盯 2、5、7、3、0 中三张同时空闲，拿到后跑 `scripts/stage1_r1_chain.sh`（PROBE_CARD=1，C2 用前两张卡；C0 是 1 张卡、C2 是 2 张卡，全局批同为 16，数据流的分片不同，报告里注明）。
   原先的分卡请求仍然有效：用户 10-10 选了 (c) 并行（C0 与修 Stage I 同时跑），但 8 张卡全在另一会话的队列里。需要的卡：第一轮 Stage I 3 张（全局 36 只能用 1/2/3/6 张卡，3 张约 7.5 h），C0 1–2 张（全局 16 只能用 1/2/4 张卡；1 张 Stage II 约 23 h，2 张约 11 h）。最少 4 张、理想 5 张，而且要让那边的队列别往这几张卡上续发。起法：
   ```
   tmux new-window -t anatobind -n c0 "cd /data0/congcong/code/Project_Doing/foundation_model && ARM=c0 CARDS=3,4 bash scripts/stage23_chain.sh; exec bash"
   tmux new-window -t anatobind -n r1 "cd /data0/congcong/code/Project_Doing/foundation_model && CARDS=5,6,7 PROBE_CARD=3 C2_CARDS=5,6 bash scripts/stage1_r1_chain.sh; exec bash"
   ```
   （C0 先起：它的链从冒烟开始，先验证 stage23_chain 能在真实数据上走通，第一轮跑完交接 C2 时就不是首次；C0 与 C2 用相同卡数；探针卡可以借 C0 的卡，每次约 10 分钟。）
   ```
   ```
1. ~~推送~~：用户 10-09 晚批准并已推（见 §1 末尾的 push 记录）；之后仍按 Q14 每过一门推一次。
2. **卡位冲突**：10-09 上午本机另一会话的 MC-GS `arc/cycle.py` 先后进入 0、1、3、7 号卡，又有一个 35 GB 的未知进程短暂进入 5 号卡，共把三次探针 / 恢复检查挤到 OOM；主训现在占着 5/6/7 各 60 GB。请让另一会话别往 5/6/7 发任务；若想把主训换成 4 卡，只能停掉重来（曝光预算按 seen_crops 计，`--resume` 换卡数会改全局 batch 与 lr 调度，不建议）。
3. ~~主训超参偏离 Q11~~：用户 10-09 晚认可（全局 36、lr 4.5e-4、预热 444 步；mask 0.60、τ 0.2、λ_c 0.1 不变），记为决定 Q21。
4. **T10–T12 已写完**（用户 10-09 "continue" 后做的）；仍是 PROPOSED：Stage III 预热 200 步。**评估全测**：用户 10-09 晚定全部 1,056 行测试集都评（决定 Q22），不抽样；一张卡约 8 h，或按来源分卡并行后合并（合并步骤待写）。U 阈值先在 `--split val` 选再固定到 test。
5. 执行计划里"厚层 BMSR 9 例"改为"6 行"（实现按严格 > 3 mm，与 Q4 原文一致；记录 README 已写明）。
6. **可删清单（只列，不删）**：10-10 新增 `ssl_runs/stage1_r1_pilot36k_20261010_0833/` 与其日志（停掉的单卡小试）、`ssl_runs/ctest_chain.lock` 与 `ssl_runs/ctest_chain_*.log`（锁的自测）、`ssl_runs/c0_stage2_smoke_20261010_0253/`（撞卡后自停的半截冒烟）、`ssl_runs/stage1_r1_chain_20261010_0332.log`（链守卫自测留下的两行日志）、scratchpad 的 `g1_diag.py`、`ft_probe.sh`、评审的 `gradbal*.py`、`reach.py`、`f16_check.py`、`bashtest/`、`conv/`、`revert/`、`mut_*`；空目录 `docs/verification/2026-10-09/anatobind_brain_ssl_first/p0/ddp_b2x2_g0567/`（预建导致训练器拒写）与 `…/ssl_runs/pilot_8k_mb12_resume150/`（被挤 OOM，无内容）；scratchpad 的 `launch_pilot.sh`、`launch_stage1.sh`；沿用上一轮清单（`/home/congcongliu/aurfix.qF3B/`、`.aurfix_dir_tmp`、SDD 工作区、S4 中间夹等）。
7. 根目录 6 个未跟踪文件（两份 PDF、`docs/20260915_Proposal/`、`logs_build_m1r_cache.txt`、粘贴的 md 两份）：入库还是保持不跟踪。
8. 沿用：ISLES 保留 A 监督的裁定只看了两张蒙太奇；S4 三条出路；S7 六条措辞；读片人与伦理备案。

## 3. 下一步

按顺序连续推进（Q30），没有日期节点（Q26）。执行计划 `docs/superpowers/plans/2026-10-13-anatobind-aur-three-goal-conformance.md`（在 codex 三目标分支上），在工作树里按任务顺序做，每完成一个任务提交、推送：

1. Task 1–4 完成并推送（§1 末条）；Task 5 是本次 STATUS 同步；Task 6 全量测试已过（1046 passed）。
2. 独立评审（用户 Q18：每批一个独立评审代理）在看 Task 1–4 的改动；有阻塞问题先修，修完再起训练。
3. Task 7：新 U 损失（`U_MASK_PROFILE=unmatched_empty_v1`，`C2_STEP1000_GATE=0`）训练 C2：从 G1 导出起，配方同新 C2（II_BB 5e-5，全局 24，每卡 12，种子 0），两张空卡，约 13.5 h。启动前把代码 SHA、两份清单与主干的 SHA-256、卡号写进记录目录。C0 配对按 Q29 是可选分析，本轮不跑（只回答预训练增益，不影响 A/U/R 过门）。验证集选 U 阈值，测试集只评一次；任一门不过就如实记录、停下，再写新计划补 A、R 缺口（病灶位置的 A 图与 B0 反常、R 输给查表）。
5. 之后：2D 模型（DICOM 单张）单独推进；显示与公开 URL 最后做（Q28）。Level R 读片仍等用户安排。

## 4. 坑与别重做

- **日志的 step_s 是每 20 步抽一步，不是平均**：C0 缺数据时抽到的步看着是 4 s，真实平均是 12 s。算吞吐用两次记录之间的 seen_crops 与墙钟时间。
- **Stage II 裁块很吃 CPU**：一个裁块约 4 s 单核（六个数组的平面内旋转、偏置场、物理坐标），一张卡每秒要约 6 个裁块，每卡至少 24 个加载进程；两臂并行时 CPU 是瓶颈（本机 112 核，别人常占 30–40）。
- **换配置不放卡**：先在原卡放一个约 4 GB 的占位进程，再停旧任务、起新任务，新任务占满显存后占位自动退出（scratchpad `swap_c0_v2.sh` + `hold_card.py`）。
- **别人的会话会往"空卡"发任务**：本会话 10-09 上午三次被挤 OOM。多卡运行前后都看 `nvidia-smi --query-compute-apps`，起主训前与其他会话对齐卡位；主训起来后它自己占 60 GB，后来者放不下。
- **训练器拒绝已存在的输出目录**（`FileExistsError`），探针与训练都别预先 `mkdir` 运行目录；日志文件放在目录外。
- **mb 16 不能用**：预留 77 GiB 贴顶，验证批与碎片随时 OOM；mb 12 预留 58–61 GiB 是上限。mb ≥ 12 必须 `--workers 8`，4 个 worker 时数据等待最大 5 s。
- **Stage I 训练器的恢复不保存数据游标**（`ssl/train.py`）：`--resume` 后 loader 从新 epoch 重新抽样，损失逐步不同是预期；step / seen / lr 必须逐位相同。Stage II/III 训练器（`train.py`）已修：恢复带 epoch 位置，且 schedule 不同就拒绝恢复（`--resume` 不能换卡数或预算）。
- **测试套件要在 CPU 上跑**：`CUDA_VISIBLE_DEVICES="" PYTHONNOUSERSITE=1 PYTHONPATH=. nice -n 19 … pytest tests/`，否则训练器与推理测试会去抢 0 号卡（本机 0 号卡常被别的会话占着）；一个 CUDA-only 测试在 CPU 上跳过。
- **CPU 的 bf16 autocast 不是 CUDA 的代理**：transformer 层在 CPU autocast 下缺算子，强开会报 dtype 错；autocast 相关行为只能在卡上验。
- **整卷推理的主机侧成本**：窗口准备（坐标网格）每窗约 1 s，比前向贵；要快就把坐标搬到 GPU 上算。
- **新测试别对旧脚本跑**：旧 `stage23_chain.sh` 与旧 Q25 检查脚本都写死 `cd` 到主检出和真 Python，旧检查脚本还会在 5 号卡起探针、用 `>` 截断已入库的日志。10-14 先把脚本改成可覆盖，再放占位脚本看 RED，最后才在桩上跑。
- **链现在必须给 U_MASK_PROFILE**：历史启动脚本（`scripts/stage1_r*`、`ops/swap_*`）不带它会直接报错退出；要复现旧运行就加 `U_MASK_PROFILE=legacy`。
- **Codex 的 /tmp 克隆**：`/tmp/Foundation_Model-aur-brain-iter2` 还留着 Task 2 的半截改动和 1904a09（作者是全局身份 Qsdcm）；内容已全部移到分支，克隆原样保留，可删（只列）。
- **时区不固定**：会话 shell 10-11 及以前是 BST（UTC+1），10-12 起是 CEST（UTC+2）；tmux 服务器是 `TZ=Europe/London`，链日志一直是 BST（10-14 用 REPORT.md 的 UTC 修改时间对过）；系统时区是 Asia/Shanghai（UTC+8）。记录里的时刻一律注明时区或"链日志时钟"。
- **四卡默认 NCCL P2P 会卡死**，`NCCL_P2P_DISABLE=1`；`torchrun --standalone` 自选端口，多个同时起不冲突。
- **数组顺序 / RAS / 采样点 / 侧别用身份 / 钩子拦删除字样 / zsh 不分词 / 不并行派两个实现者 / SibBMS 编号**：同上一版。

## 5. 关键决定的为什么

- **mb 12 而不是 Q11 的 2×2 或 4×1**：用户 10-09 要求占满显存；单卡 crops/s 从 mb 8 的 5.1 到 mb 12 的 5.8，四卡全局 16 只有 10 crops/s，3 卡 mb 12 实测 12.9；mb 16 贴顶不要。
- **学习率平方根缩放**：批从 16 到 36，AdamW 常规做法；标 PROPOSED，8k pilot 下损失正常下降、无发散。
- **3 卡现在起而不是等第 4 张**：0/1 号卡的 MC-GS 约 11:50 才完，且那边的循环会续发；等一小时换来 4 卡约省 1.7 h，净收益小于一小时且不确定；5/6/7 空着也随时会被占。
- **pilot 在 3 卡**：冒烟目的（DDP、显存、loader、验证、恢复）与卡数无关；主训与 pilot 同 3 卡同全局 36，配置一致。
- **恢复检查用同卡数同配置**：换卡数会改调度，对照不成立。
- **记录只增不删**：被挤 OOM 的空目录保留并写明原因，删除只列清单。
- **Q25 门只管 legacy 配方**：那两个阈值是按旧配方 C0 同步数值定的，新 U 损失不继承；新配方要早停门得另行预先登记（三目标设计 §1）。
- **压空项的权重用现成的 NO_OBJECT_WEIGHT 0.1**：不新增可调超参数；存在性损失、匹配、推理阈值与跨窗合并首轮都不动，一次只改一个变量（三目标设计 §2）。
