# STATUS:2026-09-12（M1 第一批执行中；上一交接点 tag `handoff/2026-09-09`）

每次交接前整体重写本文件。五段固定:已验证、待拍板、下一步、坑与别重做、为什么。

## 1. 已完成且已验证

- **方案 v2.2**(`RESEARCH_PLAN.md` §13.6):M1 判据改为 G1/G2/G4 三道门(G3 只报曲线),G2 判据事先写死;2026-09-12 修订:判门路径以**标注的病灶框和类别**进查表,只预测解剖(Q19,见下)。实施计划 `docs/superpowers/plans/2026-09-11-m1-batch1-g1-g2.md`(含执行记录与修订)。
- **分割换成官方校正版**(`SKM-TEA_ltr/segmentation_masks/raw-data-track/`,155 个,md5 全对,来源 Redivis `aimi.skm_tea:5r8z`)。实测坐标系为恒等(155/155);旧 dicom-track 分割错位中位 0.3–0.9 mm、最大 3.8 mm,髌骨软骨 18 卷 Dice<0.5。**MTR_150 旧帧门槛失败的真正原因是旧分割髌骨软骨错位 3.84 mm,校正版得分 3.386**(推翻 09-07 的"扫描异常"结论)。`docs/verification/2026-09-11/REPORT.md`。
- **新导出 m1r**(`derived/skmtea/m1r/`,图像硬链接 m1,只重做 seg 与宿主;seg 头信息抄图像的层距,否则 nnU-Net 完整性检查拒收)。宿主标签只变 1 行(MTR_110 ann 15 → unresolved)。**查表天花板复核**:类别感知查表组织族级 m1 0.968 → m1r 0.958(n=308),软骨病变 0.952 → 0.938,结论不变。
- **训练缓存** `derived/skmtea/m1r_cache/`(155 卷 × 7 视图,float16,23 GB)。
- **代码**:整卷数据集(4 类病灶,积液 none/韧带 unknown)、上游(Swin + 身份锚定 A + 全分辨率 mask 头 + 密集中心热图 U_B)、固定步数训练器(每步 4 个整卷)、预测缓存、B0 查表、IoU 匹配与分桶、G2 统计、评估脚本、单折速览。224 个测试通过。
- **实测**(`docs/verification/2026-09-12/fold0_pilot.md`):fold-0 试跑 7500 步 4.5 h,损失全降;但**从零训的检测头在留出扫描上不泛化**(训练扫描全槽位召回 0.96,留出 0.57,留出上命中/落空槽位分数 0.17/0.15 分不开,按 ≥0.5 规则检出 0/84);**mask 头欠拟合**(训练与留出 Dice 都只有 0.28–0.63,nnU-Net 0.81–0.88)。两个对照实验都没能改善 mask:热图项降权 + 放宽裁剪(0.713 vs 0.699),自顶向下像素解码器(0.831 vs 0.699,500 步)。
- **nnU-Net**:`Dataset901_SKMTEAm1r`,1860 例(干净 ×6 + 六档退化),`nnUNetTrainer_250epochs`,3d_fullres patch [96,160,160];fold 0 完成(250 epoch,伪 Dice 0.81–0.88,留出 217 例分割已出),fold 1–4 在 GPU 1 自动串行。

## 2. 待用户拍板

- 无新的待拍板项。已拍板:Q1–Q19(见计划文档),其中 Q18 借 GPU 0、1,Q19 判门路径用标注框。
- 旧遗留(多次未答):Q9 删除授权(SKM-TEA 2.4G truncated 残留 + 820G 原 tar);fastMRI 其余 4850 卷是否跑 SynthSeg;运动仿真谁写;是否删除已合并的旧分支。
- 提醒:对话里出现过两个 Redivis token,事后请在 https://redivis.com/workspace/settings/tokens 删除;`~/.redivis_token` 用完可删。

## 3. 下一步(自动进行中)

1. 上游五折在 GPU 0 串行(试跑配置,按收敛规则延到 11250 步),每折训完即缓存预测(`derived/skmtea/m1r_pred/ours/fold{f}/`)。链脚本与日志在本会话 scratchpad(`upstream_folds.sh/.log`);断了就按 `docs/superpowers/plans/…` Task 10/12 的命令续跑(`--resume`)。
2. nnU-Net fold 1–4 在 GPU 1 串行(`nnunet_folds_1_4.sh/.log`),每折约 20 h。
3. 两者都完成后:`scripts/collect_nnunet_predictions.py` → `scripts/eval_g1_g2.py --out docs/verification/<日期>` → 报告(Task 15),G2 裁决按 §13.6。
4. G2 过 → 第二批(B1–B3、E、G4 门槛先问用户);不过 → 运动仿真,再判。

## 4. 坑与别重做

- 与 SENSE 图配对的分割一律用 raw-data-track;m1 只作历史,别再在它上面出数。
- m1r 的 seg 头信息必须与图像层距完全一致(nnU-Net 拒收 1e-4 mm 的差)。
- 共用 GPU 上每次下发算子排队约 1.2 ms:上游 19.6 s/步、nnU-Net 1200–1500 s/epoch;空卡上 4.7 s/步、277 s/epoch。别在共用卡上估时间。
- `pkill -f <模式>` 会连自己的 shell 一起杀(退出码 144);nnU-Net 主进程被杀后 12 个数据增广子进程会变孤儿,要按 PID `kill -9`。
- 检测头(密集中心热图)在 124 卷上过拟合,峰值分数不可用;判门不再依赖它。
- 收敛规则:最后 750 步比前 750 步再降 >2% 就延到 11250 步(fold 0 触发,3.45%)。
- 轴约定、强度归一、导出续跑判据、fastMRI 手性等旧坑见上一版本文件(git 历史 `29142bb`)。

## 5. 关键决定的为什么

- 判据从"高 10 个点"改为 G1/G2/G4:类别感知查表干净数据上 0.968(m1r 0.958),没有 10 个点可赢。
- 判门路径用标注框:从零训的检测头在留出扫描上给不出框,G2 的配对集合会是空集;关系问题本来就以"已有一个异常"为前提。这是看过 fold 0 之后改的,论文里明写。
- 检测头换成密集中心热图:DETR 解码器在合成实验里 2.4 万样本后仍定不了位(IoU 0.01),热图头 8000 样本 0.89。
- 借 GPU 0、1:共用卡上 nnU-Net 一折 14–18 天。
- nnU-Net 250 epoch:1000 epoch 在空卡上也要 3 天/折。
- 上游 mask 头留在试跑配置:两个改进假设都被 500 步对照否定;G2 不依赖它,第二批的几何可以直接用 nnU-Net 的分割。
