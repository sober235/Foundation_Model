# S2 脑侧小病灶检测器：验证记录索引(2026-09-28/29)

**结论:D1 门不过。** 2d 五折折外(判门):灵敏度 0.3662(475/1297),阈值 0.55,每卷假阳 1.636,88 个正常卷上每卷假阳 0.352。3d_fullres 五折(只报告):0.3678(477/1297),阈值 0.60,每卷假阳 1.431,正常卷 0.489。终审(opus)用独立脚本从 253 份折外输出复算,2d 的 FROC 19 行逐行一致到小数点后 6 位(`.superpowers/sdd/2026-09-28-brain-detector/final-review.md`,不入库)。按 D9 不调参救门;下一步是 nnDetection 第二臂(规格 `docs/superpowers/specs/2026-09-28-brain-nndet-design.md`)。

## 记录

| 文件 | 内容 |
|---|---|
| `dataset.md` | Dataset903 构建、预处理、折的汇总(253 例 = 165 病灶卷 + 88 正常卷) |
| `prepare_raw.txt` | 逐例构建输出(每例标签体素数) |
| `splits.txt` | 折:每折验证例数 51/51/51/50/50 |
| `label_check.txt` | 逐卷核对:标签图与按注册表重画的结果一致 |
| `label_bbox_check.txt` | 独立核对:1297 个注册表框都有标签体素,910357 个标签体素没有一个落在框外 |
| `plan_preprocess_start.txt`、`plan_preprocess.txt` | nnU-Net 规划与预处理的开始时间和输出 |
| `timing.md` | 计时探针(D8):2d 40.70 s/epoch,3d_fullres 45.72 s/epoch,推算约 12 h |
| `2d/REPORT.md`、`2d/froc.csv`、`2d/output.txt` | **门结论**(见下方命令) |
| `2d/diagnostic_NOT_GATE_v2.txt` | 漏检诊断,两种统计范围分开标注(不作门);`2d/diagnostic_NOT_GATE.txt` 是第一版,范围混用,已标注被取代 |
| `3d_fullres/REPORT.md`、`froc.csv`、`output.txt` | 3d_fullres 同一套数字,只报告 |
| `infer_smoke.md` | 推理入口在一个没见过、有小血管病印象但无框的检查上跑通(17 个病灶) |

命令(在仓库根目录):

```
bash -c 'source scripts/nnunet_env.sh && PYTHONNOUSERSITE=1 PYTHONPATH=. ~/anaconda3/envs/nvgen/bin/python scripts/eval_brain_detector.py --config 2d --out docs/verification/2026-09-28/brain_detector/2d'
bash -c 'source scripts/nnunet_env.sh && PYTHONNOUSERSITE=1 PYTHONPATH=. ~/anaconda3/envs/nvgen/bin/python scripts/eval_brain_detector.py --config 3d_fullres --out docs/verification/2026-09-28/brain_detector/3d_fullres'
bash -c 'source scripts/nnunet_env.sh && PYTHONNOUSERSITE=1 PYTHONPATH=. ~/anaconda3/envs/nvgen/bin/python scripts/diagnose_brain_detector.py --config 2d --out docs/verification/2026-09-28/brain_detector/2d/diagnostic_NOT_GATE_v2.txt'
```

3d 的评估(2026-09-29 00:19)用的是终审修复后的脚本(e87a676):它先断言各折验证集正好覆盖 253 例,打分后再核对 253 卷、1297 个病灶。

## 读 FROC 表要注意

解码只取 argmax 前景的连通块,块的分数是块内前景概率的均值,天然大于 0.5。所以阈值 0.05 到 0.50 这几行完全相同,2d 曲线最多只到每卷 1.676 个假阳(3d_fullres 1.474),每卷 2 个假阳的预算根本用不满。门是在 argmax 工作点上读的:"0.366"不是"每卷 2 个假阳时的灵敏度",而是这个模型能到的最高灵敏度。

## 已知标签噪声(规格 D10)

165 个病灶卷里的 fastMRI+ 框共 2604 个。按 Gate 0.5 的筛选(两种小病灶标签、边长 ≥ 3 像素)填进标签图的有 1785 个,合并成 1297 个三维病灶;**没有填的 819 个,分布在 61 个卷里**。它们在训练里是背景,模型若在那里检出,记为假阳。按标签:

```
Nonspecific lesion 152, Craniotomy 108, Dural thickening 86, Posttreatment change 85, Encephalomalacia 81,
Possible artifact 69, Enlarged ventricles 48, Mass 47, Resection cavity 46, Craniectomy with Cranioplasty 31,
Edema 31, Normal variant 17, Nonspecific white matter lesion 12, Paranasal sinus opacification 5, Lacunar infarct 1
```

其中最后两类的 13 个框是小病灶标签,但有一边不到 3 像素,被 Gate 0.5 的边长下限筛掉了。计数命令(只读):

```
PYTHONNOUSERSITE=1 PYTHONPATH=. ~/anaconda3/envs/nvgen/bin/python - <<'EOF'
import collections, json
from pathlib import Path
from anatobind.data_engine.fastmri import read_fastmri_plus_rows
from anatobind.level_r.export import small_lesion_rows
CSV = Path("/data2/congcong/data/FM_data/fastMRI_lh_brain_knee/Annotations/brain.csv")
info = json.loads(Path("/data2/congcong/data/FM_data/derived/nnunet/raw/Dataset903_FastMRIBrainSmallLesion/cases.json").read_text())
lesion_files = {c for c, v in info.items() if v["kind"] == "lesion"}
rows = [r for r in read_fastmri_plus_rows(CSV) if r["file"] in lesion_files]
kept = {id(r) for r in small_lesion_rows(rows)}
other = [r for r in rows if id(r) not in kept]
print(len(rows), len(kept), len(other), len({r["file"] for r in other}))
print(dict(collections.Counter(r["label"] for r in other).most_common()))
EOF
```

输出:`2604 1785 819 61`,标签分布同上。

## 漏检诊断(不作门)

按门实际用的解码范围(连通块 ≥ 9 体素):被任何预测块碰到的病灶 645/1297 = 0.497;碰到了但所在预测块的框 IoU < 0.1 的 174 个;每个预测块碰到的病灶数 {1: 479, 2: 59, 3: 18, 4: 6, 5+: 6}。不设大小下限时是 668(0.515)、189、{1: 528, 2: 60, 3: 18, 4: 6, 5+: 6}。也就是说,主要问题是约一半病灶根本没被预测碰到(漏检),框不准和相邻病灶粘连是次要的。

## 终审与修复

终审(opus)结论"修完即可合并"。修复:启动器在 nnU-Net 结果目录已存在时拒绝启动(edf5fd4);评估脚本检查各折覆盖与实际打分数(e87a676);诊断分开标注两种范围并提交脚本(2b60f60)。本页补上了 D10 说明和 FROC 读法。留作以后:单卷推理时概率轴序的检查太弱(整卷 0.99 一致率在脑部这么稀疏的前景上查不出面内轴交换),S5 用之前应改为只在前景体素上比对;现有 253 卷的轴序已核对完全正确。
