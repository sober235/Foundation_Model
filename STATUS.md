# STATUS：2026-10-09 傍晚（SSL-first 线：T01–T12 代码全部在 main，G0 与 G0.5 过；Stage I 主训 11:14（BST，本机时区；北京 18:14）起在 5/6/7 跑，约 17:45 本机时钟结束；T10–T12 经两轮独立评审修补、全量测试通过；待主训结束跑 G1；main 比 origin/main 多 7 个提交未推）

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

**之前（保留）**：Part 1（方案架构的代码包、样本表、目标、探针）在 tag `handoff/2026-10-09-anatobind-brain-aur-part1`；S4 脑解剖学生模型 A11 不过、三条出路待定；S7 三病种五折全过、1212 份记录 v2；nnDetection 与小病灶线已停；Level R 读片工具就绪、读片未开始（见 CLAUDE.md 当前状态段）。

## 2. 待用户拍板

1. ~~推送~~：用户 10-09 晚批准并已推（见 §1 末尾的 push 记录）；之后仍按 Q14 每过一门推一次。
2. **卡位冲突**：10-09 上午本机另一会话的 MC-GS `arc/cycle.py` 先后进入 0、1、3、7 号卡，又有一个 35 GB 的未知进程短暂进入 5 号卡，共把三次探针 / 恢复检查挤到 OOM；主训现在占着 5/6/7 各 60 GB。请让另一会话别往 5/6/7 发任务；若想把主训换成 4 卡，只能停掉重来（曝光预算按 seen_crops 计，`--resume` 换卡数会改全局 batch 与 lr 调度，不建议）。
3. ~~主训超参偏离 Q11~~：用户 10-09 晚认可（全局 36、lr 4.5e-4、预热 444 步；mask 0.60、τ 0.2、λ_c 0.1 不变），记为决定 Q21。
4. **T10–T12 已写完**（用户 10-09 "continue" 后做的）；仍是 PROPOSED：Stage III 预热 200 步。**评估全测**：用户 10-09 晚定全部 1,056 行测试集都评（决定 Q22），不抽样；一张卡约 8 h，或按来源分卡并行后合并（合并步骤待写）。U 阈值先在 `--split val` 选再固定到 test。
5. 执行计划里"厚层 BMSR 9 例"改为"6 行"（实现按严格 > 3 mm，与 Q4 原文一致；记录 README 已写明）。
6. **可删清单（只列，不删）**：空目录 `docs/verification/2026-10-09/anatobind_brain_ssl_first/p0/ddp_b2x2_g0567/`（预建导致训练器拒写）与 `…/ssl_runs/pilot_8k_mb12_resume150/`（被挤 OOM，无内容）；scratchpad 的 `launch_pilot.sh`、`launch_stage1.sh`；沿用上一轮清单（`/home/congcongliu/aurfix.qF3B/`、`.aurfix_dir_tmp`、SDD 工作区、S4 中间夹等）。
7. 根目录 6 个未跟踪文件（两份 PDF、`docs/20260915_Proposal/`、`logs_build_m1r_cache.txt`、粘贴的 md 两份）：入库还是保持不跟踪。
8. 沿用：ISLES 保留 A 监督的裁定只看了两张蒙太奇；S4 三条出路；S7 六条措辞；读片人与伦理备案。

## 3. 下一步

1. 主训结束（约 17:45 本机时钟）→ **T09 G1**：
   ```
   PYTHONNOUSERSITE=1 PYTHONPATH=. CUDA_VISIBLE_DEVICES=<空卡> ~/anaconda3/envs/nvgen/bin/python scripts/aur_ssl_eval.py \
     --checkpoint /data2/congcong/data/FM_data/derived/aur/ssl_runs/stage1_320k_mb12/ssl_stage1_best.pt \
     --samples /data2/congcong/data/FM_data/derived/aur/samples_1mm_v1.json \
     --val-patients /data2/congcong/data/FM_data/derived/aur/ssl_manifest_v1/val_patients.json \
     --out docs/verification/2026-10-09/anatobind_brain_ssl_first/g1/<新目录>
   ```
   （含随机初始化对照与 3 种子；Q8 严口径：13 宿主读出宏 Dice 高于随机 ≥ 0.05 且区间不含 0，病灶可分性 ≥ 随机 − 0.02；不过最多修两轮再交用户。）
2. G1 过 → Stage II（命令在执行计划 §2，已补 `--g1-report` 与 `--val-patients`；3 或 4 卡，microbatch 4/卡不累积 = 全局 12–16，lr 5e-4；约 240k/全局 步）→ Stage III → `scripts/aur_eval.py`（先 `--split val` 选 U 阈值，再 `--split test --u-threshold …`）→ C0（`--init random`）。开 Stage II 前补 T10 的 4 卡 100 步含保存/恢复验证记录（需要 4 张空卡；2 进程 gloo 版已在测试套件里）。
3. 推理冒烟：主训/Stage III 导出后对一卷真实 FLAIR 跑 `scripts/infer_anatobind_brain.py --gpu <空卡>`，看 `record.json` 与句子。
4. 主训期间看一眼 `val.jsonl`（每 500 步）：masked Huber 要持续低于基线、有效秩不掉、`contrast_acc` 不该一直是 1.0（批大了该掉一点）；出现 `non-finite` 训练器会自己抛错退出。
5. 周三起草汇报 PPT（Q20：全项目含阴性结果、伪标签数字标 NOT_EVIDENCE、中文）。
6. Level R 读片：同前；T12 已能导出盲读片表 `level_r_sheet.csv` 与密钥 `level_r_key.json`。

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
