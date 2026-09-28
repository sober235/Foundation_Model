# STATUS：2026-09-29（S2 脑侧小病灶检测器已完成、整支终审、修复；D1 门不过；合入 main 并打 tag `handoff/2026-09-29-brain-detector`，未 push；下一步 nnDetection 第二臂，规格与计划已批准，放在分支 build/brain-nndet）

每次交接前整体重写本文件。五段固定：已验证、待拍板、下一步、坑与别重做、为什么。

## 1. 已完成且已验证

**本轮（2026-09-28/29）：S2 = 脑侧小病灶检测器（nnU-Net v2，Dataset903）。** 规格 `docs/superpowers/specs/2026-09-28-brain-detector-design.md`（决定 D1–D10），计划 `docs/superpowers/plans/2026-09-28-brain-detector.md`（7 个任务）。分支 `build/brain-detector`（工作树 `../foundation_model-detector`），任务逐个评审；整支终审（opus）"修完即可合并"，一轮修复（edf5fd4 启动器拒绝已存在的 nnU-Net 结果目录、e87a676 评估脚本核对覆盖与打分数、2b60f60 诊断分开标注统计范围），控制方重跑测试、在 scratch 重跑诊断逐行一致。台账在工作树 `.superpowers/sdd/2026-09-28-brain-detector/`（gitignore，含 `final-review.md`）。记录索引 **`docs/verification/2026-09-28/brain_detector/README.md`**。

- **测试**：
  ```
  PYTHONNOUSERSITE=1 PYTHONPATH=. ~/anaconda3/envs/nvgen/bin/python -m pytest tests/ -q -p no:cacheprovider
  715 passed in 89.55s (0:01:29)
  ```
- **Dataset903**：253 例 = 165 个病灶卷（1297 个注册病灶）+ 88 个"年龄相符正常"卷；折 51/51/51/50/50（病灶患者沿用 `data/level_r/folds.json`，正常患者种子 0）。独立核对：1297 个框都有标签体素，910357 个标签体素没有一个在框外。
- **D1 门（2d 五折折外，判门）：不过。**
  ```
  bash -c 'source scripts/nnunet_env.sh && PYTHONNOUSERSITE=1 PYTHONPATH=. ~/anaconda3/envs/nvgen/bin/python scripts/eval_brain_detector.py --config 2d --out docs/verification/2026-09-28/brain_detector/2d'
  Scans: 253 / Total ground truth lesions: 1297 / Gate pass: False / Operating threshold: 0.55
  Family sensitivity at operating point: 0.3662 / FP per scan at operating point: 1.6364 / Normal FP per volume: 0.3523
  ```
  终审用自己的脚本（不用我们的解码、匹配、扫描函数）从 253 份折外输出复算，FROC 19 行一致到小数点后 6 位。
- **3d_fullres（只报告）**：0.3678（477/1297）@ 0.60，1.4308 FP/卷，正常卷 0.4886（`…/3d_fullres/output.txt`，2026-09-29 00:19）。
- **漏检诊断（不作门，`2d/diagnostic_NOT_GATE_v2.txt`）**：按门实际用的解码范围（连通块 ≥ 9 体素），被预测碰到的病灶 645/1297 = 0.497；碰到但框 IoU < 0.1 的 174 个。主要问题是约一半病灶根本没被碰到。第一版诊断混用了两种范围，已标注被取代。
- **推理入口（给 S5）**：在没见过、有小血管病印象但无框的 `file_brain_AXFLAIR_201_6002917` 上跑通，17 个病灶，格式同 Level R 的 `lesions.json`（`infer_smoke.md`）。
- **D10 标签噪声**：165 个病灶卷里没填进标签的 fastMRI+ 框 819 个（61 卷），明细与命令在 README。
- **用户本轮拍板**：先训练、训练完做最终测试再找医生；三目标重述为"感知解剖 → 找到异常并给出疑似疾病 → 异常绑定到解剖结构"；疾病层 D = 整个检查一个印象；S2 的门 A、数据 A、方法 1、四张 80 GB 卡；门不过后选 nnDetection 第二臂（"按照你的倾向来"）；第二臂的规则 A、方案 1、只分一类、四段设计、规格与计划（"符合"，执行 = 子代理逐任务）。

**之前（保留）**：S1（PR-C 脑侧关系基线）已在 main（tag `handoff/2026-09-28-relation-baselines`），阶段一报告 `docs/verification/2026-09-28/relation_baselines/REPORT.md`（全部 NOT_EVIDENCE）。Level R 读片工具已合 main（tag `handoff/2026-09-26-level-r-tooling`、`handoff/2026-09-26-level-r-fields`）；冒烟服务 8791；浏览器验收十步仍是 USER_REPORTED（`docs/verification/2026-09-26/level_r_smoke.md` §7）。更早的证据链不变。

## 2. 待用户拍板

- **push**：main 自 afe641e 起所有提交与 tag（09-26 两个、09-28 一个、09-29 一个）都只在本地。
- **nnDetection fold 0 出结果后**：规则 A 通过就要补五折，墙钟约一天以上，开跑前问。
- **读片人**：两位读者、一位裁定人、对外方式（端口转发 / `--bind 0.0.0.0`）、伦理备案；医生阶段在训练结束后。
- **Level R 读片说明里 "other" 的定义**（S1 终审 minor 9）。
- **可删清单（只列，不删；删除由用户执行）**：S2 已合并后，工作树 `../foundation_model-detector` 与分支 `build/brain-detector` 可删（`git worktree remove ../foundation_model-detector`、`git branch -d build/brain-detector`）；scratch 里的计划预演副本 `plan_dryrun/`（会话临时目录）。
- 旧遗留仍挂：Q9 删除授权、其余 4850 卷 SynthSeg、Redivis token、两条远端评审分支去留、RSS 左右手性换算规则。

## 3. 下一步

1. **nnDetection 第二臂**：规格 `docs/superpowers/specs/2026-09-28-brain-nndet-design.md`（N1–N14）与计划 `docs/superpowers/plans/2026-09-28-brain-nndet.md`（10 个任务）已批准，作为分支 `build/brain-nndet`（从本 main 开，工作树 `../foundation_model-nndet`）的第一个提交。执行 = 子代理逐任务。
   - 环境 `nndet` 已由控制方提前开始装（计划 Task 1 Step 5），日志 `~/logs/nndet_install/`：第一、二种装法卡在 conda 求解编译器（见第 4 段），第三种 `03_env.sh` = 系统 gcc-10/g++-10 + nvidia 频道单独装 nvcc 11.3.1 + pip 装 torch 1.11.0+cu113。计划 Task 1 Step 5 的命令要改成实际跑通的这一套再交给实现者。
   - fold 0 只训一折（约一天），按规则 A（nnDetection fold 0 命中 ≥ nnU-Net 2d 同折命中 + 14，探针 92 → 106/280）决定是否补齐五折。
2. S3（B3/B4，PR-D）、S4（脑侧解剖层：自有 FLAIR 解剖分割、脑叶、A_local_quality）、S5（疾病印象 + 整句拼装，依赖检测器）、S6（膝侧 nnDetection，可复用第二臂的工具），顺序同 v2.6 §25 第 11 项。S4 不依赖检测器，可以并行。
3. 读片：用户在 8791 做浏览器验收 → 正式部署（`docs/level_r_tool.md`）→ pilot 150 → 判 §10.1 升级 → 全集 → 封存 → S1 阶段 2（`run_relation_baselines.py --labels R --init-from runs/relation/c1_stage1_v2`）→ `eval --labels R --unblind` 只跑一次。
4. S2 留下的待办：S5 用 S2 推理前，把概率轴序检查改成只在前景体素上比对（整卷 0.99 一致率在脑部稀疏前景上查不出面内轴交换；现有 253 卷已核对完全正确）。S1 的待办见上一版（git 历史里的 STATUS.md）与 S1 台账。

## 4. 坑与别重做

- **S2 的结论已定**：别在 nnU-Net 上调参救门（D9）。FROC 表阈值 0.05–0.50 几行相同，是 argmax 解码所致（块分数是前景概率均值，天然 > 0.5），门读在 argmax 工作点，每卷 2 个假阳的预算用不满。
- 诊断看 `2d/diagnostic_NOT_GATE_v2.txt`（两种统计范围）；台账里 09-28 那条"51.5%"混用了范围，已在台账追加更正。
- `nnUNetv2_train` 不加 `--c` 会覆盖已有折目录；启动器现在会拒绝。`logs/` 没有 gitignore，别 git add。
- 共享卡：开跑前 nvidia-smi；GPU 0–2、6–7 常被别的用户占满。
- zsh 的坑：`echo ======` 会被当成 `=命令` 展开而报错；`pgrep -f <模式>` 会匹配到执行这条命令的 shell 自己（本轮因此误杀了自己的一个 shell，无别的影响），模式要锚定到具体路径。
- conda 的坑：`~/.condarc` 用清华镜像且 strict 频道优先级；经典求解器在 conda-forge 全量索引上极慢（gxx 一步 28 分钟没结果）；nndet 环境的 python/libgcc 来自 conda-forge，defaults 的 `gxx_linux-64` 9.3 不可满足。系统有 gcc-9/10/11/12，nvcc 11.3 用 gcc-10。
- haiku 实现者会编造测试输出与提交信息，还会听 harness 提示加 Co-Authored-By；控制方必须自己重跑测试、看 `git log -1 --format=%B`。本轮修复用的是 sonnet。
- S1 的坑同上一版：C1 上的任何数字都不是证据；`derived/relation/v1` 拒绝覆盖；`--labels R` 的 eval 只跑一次。
- 测试不读 `/data2`。

## 5. 关键决定的为什么

- **2d 判门（D4）**：79% 的病灶只占一层，层厚 5 mm。
- **门不过不调参（D9）**：防止在门上做选择；换方法（第二臂）而不是换参数。
- **nnDetection 第二臂（N1）**：它直接学"一个病灶一个框"，不靠连通域拆分；但 S2 的主因是漏检（解码范围只碰到 0.497），所以它必须多找回漏检才可能过门，这一点写在第二臂规格的风险里。
- **先只训 fold 0 + 规则 A（N2、N3）**：官方默认每折约一天；单折只有 51 卷，直接卡 0.5 会误杀或误放，和同一折的 nnU-Net 配对比较才看得出有没有用；0.05 是判断线，不是显著性检验。
- **主线 97a58f3 + py3.8/torch 1.11 独立环境（N4）**：主线 README 要求 PyTorch 1.X；nextrelease 虽支持 torch 2 但未发布、与主线差 904 个文件。
- **只分一类（N5）**：腔隙性梗死只有 57 个；nnDetection 的 NMS 按类别分开做，两类会在同一病灶上出两个框，多的那个在门里算假阳。
- **默认后处理参数判门（N7）**：sweep 在验证折上调参再在上面评估等于选择；调参版只作 NOT_GATE。
- **不删 SDD 工作区与 scratch**：用户规则，删除一律由用户执行。
