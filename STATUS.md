# STATUS：2026-10-10 09:30 本机（SSL-first 线：第一轮 Stage I 08:51 起在 4/5/7（每卡 58 GB），C0 09:04 起在 1/6（每卡 12 个裁块、约 65 GB，全局 24；冒烟通过后自动进 Stage II，约 7 h）；用户 10-10 要求"跑满显存"，C0/C2 改为每卡 12、学习率按平方根缩放（Q23）；损失采样点改在 GPU 上算，C0 每步从 5.7 s 降到 2.45 s）

每次交接前整体重写本文件。五段固定：已验证、待拍板、下一步、坑与别重做、为什么。
本机时钟是 BST（UTC+1），北京时间加 7 小时；下面的时刻都是本机时钟。

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

**之前（保留）**：Part 1（方案架构的代码包、样本表、目标、探针）在 tag `handoff/2026-10-09-anatobind-brain-aur-part1`；S4 脑解剖学生模型 A11 不过、三条出路待定；S7 三病种五折全过、1212 份记录 v2；nnDetection 与小病灶线已停；Level R 读片工具就绪、读片未开始（见 CLAUDE.md 当前状态段）。

## 2. 待用户拍板

0. **在跑（10-10 09:00）**：
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

1. 用户分卡后：tmux 窗口 `r1` 起第一轮 Stage I（`scripts/stage1_r1_chain.sh`，自动做第 1000/2000 步探针、按预先登记规则继续或停、跑完接 G1、过了接 C2），窗口 `c0` 起 C0（`scripts/stage23_chain.sh ARM=c0`）。命令在 §2 第 0 条。
2. 第一轮提前停或 G1 仍不过：第二轮 = λ_c 0（纯遮挡重建），其余不变；两轮都不过交用户（G1.md 预先登记）。
3. 两条臂的评估都是先 `--split val` 选 U 阈值、再全测 1,056 行（Q22）；之后汇总 C0 对 C2（同全局批 16、同预算、同验证集选阈值）。
4. 周三起草汇报 PPT（Q20：全项目、含 G1 失败与修复、伪标签数字标 NOT_EVIDENCE、中文）。
5. Level R：T12 已能导出盲读片表与密钥；读片等用户安排。

## 4. 坑与别重做

- **别人的会话会往"空卡"发任务**：本会话 10-09 上午三次被挤 OOM。多卡运行前后都看 `nvidia-smi --query-compute-apps`，起主训前与其他会话对齐卡位；主训起来后它自己占 60 GB，后来者放不下。
- **训练器拒绝已存在的输出目录**（`FileExistsError`），探针与训练都别预先 `mkdir` 运行目录；日志文件放在目录外。
- **mb 16 不能用**：预留 77 GiB 贴顶，验证批与碎片随时 OOM；mb 12 预留 58–61 GiB 是上限。mb ≥ 12 必须 `--workers 8`，4 个 worker 时数据等待最大 5 s。
- **Stage I 训练器的恢复不保存数据游标**（`ssl/train.py`）：`--resume` 后 loader 从新 epoch 重新抽样，损失逐步不同是预期；step / seen / lr 必须逐位相同。Stage II/III 训练器（`train.py`）已修：恢复带 epoch 位置，且 schedule 不同就拒绝恢复（`--resume` 不能换卡数或预算）。
- **测试套件要在 CPU 上跑**：`CUDA_VISIBLE_DEVICES="" PYTHONNOUSERSITE=1 PYTHONPATH=. nice -n 19 … pytest tests/`，否则训练器与推理测试会去抢 0 号卡（本机 0 号卡常被别的会话占着）；一个 CUDA-only 测试在 CPU 上跳过。
- **CPU 的 bf16 autocast 不是 CUDA 的代理**：transformer 层在 CPU autocast 下缺算子，强开会报 dtype 错；autocast 相关行为只能在卡上验。
- **整卷推理的主机侧成本**：窗口准备（坐标网格）每窗约 1 s，比前向贵；要快就把坐标搬到 GPU 上算。
- **本机时钟是 BST**，比北京慢 7 小时；记录里的时刻注明时区。
- **四卡默认 NCCL P2P 会卡死**，`NCCL_P2P_DISABLE=1`；`torchrun --standalone` 自选端口，多个同时起不冲突。
- **数组顺序 / RAS / 采样点 / 侧别用身份 / 钩子拦删除字样 / zsh 不分词 / 不并行派两个实现者 / SibBMS 编号**：同上一版。

## 5. 关键决定的为什么

- **mb 12 而不是 Q11 的 2×2 或 4×1**：用户 10-09 要求占满显存；单卡 crops/s 从 mb 8 的 5.1 到 mb 12 的 5.8，四卡全局 16 只有 10 crops/s，3 卡 mb 12 实测 12.9；mb 16 贴顶不要。
- **学习率平方根缩放**：批从 16 到 36，AdamW 常规做法；标 PROPOSED，8k pilot 下损失正常下降、无发散。
- **3 卡现在起而不是等第 4 张**：0/1 号卡的 MC-GS 约 11:50 才完，且那边的循环会续发；等一小时换来 4 卡约省 1.7 h，净收益小于一小时且不确定；5/6/7 空着也随时会被占。
- **pilot 在 3 卡**：冒烟目的（DDP、显存、loader、验证、恢复）与卡数无关；主训与 pilot 同 3 卡同全局 36，配置一致。
- **恢复检查用同卡数同配置**：换卡数会改调度，对照不成立。
- **记录只增不删**：被挤 OOM 的空目录保留并写明原因，删除只列清单。
