# STATUS:2026-09-23 晚(计划 1"膝侧能力系统"交付;检测门不过,停下待用户定第二臂;本交接点 tag `handoff/2026-09-23-knee-capability`,上一交接点 tag `handoff/2026-09-22-v2.6`)

每次交接前整体重写本文件。五段固定:已验证、待拍板、下一步、坑与别重做、为什么。

## 1. 已完成且已验证

**主线改了口径(2026-09-23 拷问式设计会,用户拍板)**:先把三条能力做出来(解剖分割图、病灶 3D 框加大类、病灶所在结构主结构/占比/侧别),再谈网络;膝先脑后;S/E/k 空间干预不做;FM_MRI 文档定网络、v2.6 定证据规则。决定记录在 `docs/superpowers/specs/2026-09-23-aur-capability-system-design.md`(Q1–Q12、N1–N8),实施计划 `docs/superpowers/plans/2026-09-23-knee-capability-system.md`。v2.6 的 Gate 链没有作废,Gate 0 被并入计划 3。

- **代码(分支 `build/aur-system`,已合 main)**:`anatobind/nnunet/lesion_labels.py`(箱填病灶标签图)、`anatobind/nnunet/prepare_lesion.py`(Dataset902,与 901 同视图同折同名)、`anatobind/eval/lesion_boxes.py`(连通域取框 + nnU-Net npz 概率读取,轴序断言)、`anatobind/eval/detection_metrics.py`(灵敏度、每卷假阳、工作点、门)、`anatobind/eval/lookup.py` 加 `host_fractions/describe_host`、`scripts/eval_knee_folds.py`、`anatobind/infer/{canonical,knee}.py` + `scripts/infer_knee.py`(单命令入口,`--frame h5|world`)。
- **测试**:
  ```
  PYTHONNOUSERSITE=1 PYTHONPATH=. ~/anaconda3/envs/nvgen/bin/python -m pytest tests/ -q -p no:cacheprovider
  342 passed
  ```
- **Dataset902 五折训练**(nnU-Net 3d_fullres,250 轮,每折约 3.7 h,GPU 0/1/4/6):模型与折末验证输出(含 npz 概率)在 `/data2/congcong/data/FM_data/derived/nnunet/results/Dataset902_SKMTEAlesion/`(179 GB)。
- **五折数字(clean 视图,155 卷、465 框,`docs/verification/2026-09-23/knee_eval/`)**:
  ```
  目标 1 解剖 Dice(Dataset901 六类): 0.886 / 0.873 / 0.852 / 0.858 / 0.845 / 0.839
  目标 2 病灶: 工作点 thr 0.75, 定位灵敏度 0.282, 大类正确灵敏度 0.254, 每卷假阳 1.57  -> 门(>=0.5)不过
             每族: 软骨 25/208, 积液 64/116, 韧带 9/38, 半月板 20/103; 阈值上限 0.394 @ 4.3 FP/卷
  目标 3 所在结构(命中上): 组织族正确 0.796 (n=54; 大类判对时 0.96), 侧别 0.81 (n=26)
             给定标注框对照: 0.961 / 0.979 (n=311/143)
  ```
  裁决与原因在 `VERDICT.md`(小病灶几乎全漏;解码变体不改善,上限 0.458 @ 6 FP/卷;缺的是模型召回)。全文报告 `REPORT.md`。
- **入口核对**:`scripts/infer_knee.py` 在 MTR_010 上与折末验证逐体素一致(`infer_regression.txt`);DICOM 转出的世界帧体积经 `--frame world` 跑通,软骨亮度分数 3.895、病灶与 h5 帧逐条对应(`dicom_world_frame.md`);图 `~/figs/anatobind_knee/`。
- 09-13 之前的证据链不变(G2 不过、H1 不过但框镜像、脑探针 q3 7.4%、查表天花板 0.958–0.968)。

## 2. 待用户拍板

- **第二臂选哪条**(`VERDICT.md` §4):① fastMRI+ 膝预训练 2.5D 检测器再迁 SKM-TEA(前提 Gate 0 + H1 重跑,约一周);② nnDetection 直训 SKM-TEA(2–3 天);③ nnU-Net 换目标继续挤(一天,上限不高);④ 接受现状转入脑侧计划 2。我的推荐:先做 ① 的前半段(Gate 0 + H1 重跑),再定。
- 计划 2(脑掩膜三大类)与计划 3(fastMRI+ 小病灶)何时开写。
- `derived/nnunet/results/Dataset902_SKMTEAlesion/` 里 5 × 217 个 npz 概率(约 60 GB)评估完是否删除(按规矩由用户手动删)。
- 删已合并分支(由用户手动执行):`git branch -d build/aur-system && git push origin --delete build/aur-system`;上次交接列的四条远端分支与 `summary/2026-09-22-v2.5-feasibility-review` 仍未删。
- 旧遗留(多次未答):Q9 删除授权(SKM-TEA 2.4G truncated 残留 + 820G 原 tar);fastMRI 其余 4850 卷是否跑 SynthSeg;Redivis token 事后删除;读片人姓名/裁定人/伦理。

## 3. 下一步

1. 用户定第二臂 → 另写实施计划(不改 spec 的验收口径)。
2. 计划 2 脑掩膜三大类:先取 `synthseg_parc_2.0.h5`(只在 FreeSurfer 7.4.1 tarball 里,9.5 GB),SynthSeg robust + parc 重跑 PDGM/BMSR/ISLES;三个 nnU-Net(PDGM 四序列、BMSR 三序列、ISLES 只用 DWI+ADC),按患者留 20%;脑侧入口与一致率报告(NOT_EVIDENCE)。
3. 计划 3 fastMRI+ 小病灶:Gate 0 翻转修复(`fastmri_knee.py` 加 `[nr − y − h, nr − y)`,旧 56 GB 导出只换坐标文件)+ 患者 ID 折断言 + H1 重跑,再做脑 FLAIR 小病灶检测。
4. (b) 阶段(关系模块)按决定 N8 等 (a) 阶段两部位出数后再拷问。

## 4. 坑与别重做

- **箱填 nnU-Net 对小病灶不行**:软骨 < 1.8 mL 2/69、半月板 < 0.4 mL 1/34;别再调阈值或解码(都试过,`decode_variants.py`)。
- nnU-Net `--npz` 的 `probabilities` 是 (C, Z, Y, X)(SimpleITK 轴序),`load_nnunet_probabilities` 转到 (C, X, Y, Z) 并断言与标签图 ≥ 99% 一致;别绕开它。
- 已训模型只认 SKM-TEA 导出帧 (I, P, R);外来 NIfTI 走 `--frame world`,导出/缓存文件走 `--frame h5`。SKM-TEA DICOM 两个回波共用一个 SeriesInstanceUID,dicom2nifti 须先按 EchoNumbers 拆开。
- zsh 数组下标从 1 起数:`${GPUS[0]}` 为空,fold 0 首次没拿到卡。命令里写显式卡号。
- nohup 的 stdout 缓冲让 `logs/nnunet902_fold*.log` 滞后几十轮;看 `results/fold_k/training_log_*.txt`。Monitor 脚本里 pgrep 会匹配自身命令行。
- 训练同时最多 4 个 nnU-Net(每个 8 个数据加载进程,CPU ≤ 48 线程)。
- 老坑不变:fastMRI+ 框 y 从 RSS 底部数起,Gate 0 前不引用 H1;脑侧 SynthSeg 查表不是真值;别在 SKM-TEA 上找"退化让绑定失效"。

## 5. 关键决定的为什么

- 先能力后网络(Q1):膝上目标 1、3 早已有数,自研 mask 头留出 Dice 0.38 对 nnU-Net 0.86,从头训网络没有胜算;缺的只是病灶检测。
- 箱填 nnU-Net 作第一臂(Q9):复用已跑通的 Dataset901 流程,一天出结果;它输了,输在召回,证据齐全(§1)。
- 达标线只设在目标 2(Q12):目标 1、3 只报数;门是里程碑门不是科学主张(文献无 SKM-TEA 检测基线)。
- 膝真值用 tissue_id、脑只报一致率(Q4):重叠率当真值会让查表按定义满分(v2.6 红线)。
- 所在结构报集合值(N4):脑肿瘤跨脑叶时单一主结构说不清;膝上主结构就是答案。
