# 2026-09-14 fastMRI+ 膝关节未标注核查：卷级零标签 = 读片正常，层级逐层审阅文献未解决

对应 spec `docs/superpowers/specs/2026-09-14-leg2-fastmri-knee-detection-gate-design.md` §1.7 的两处未决事实。结论供 Task 7（数据集取负样本）直接使用。

## 0. 结论

```
UNANNOTATED_VOLUMES_ARE_NEGATIVE = True
UNANNOTATED_SLICES_IN_ANNOTATED_VOLUMES_ARE_NEGATIVE = False
```

- 198 卷完全无标注的卷：作为全负样本（整卷所有层、所有类别族）纳入。
- 已标注卷内没有落框的层：从样本索引中整体剔除，既不算正类也不算负类，不参与训练。

依据见第 1 节（文献）与第 2 节（数据交叉验证）。

## 1. 文献依据

论文：Zhao R, Yaman B, Zhang Y, Stewart R, Dixon A, Knoll F, Huang Z, Lui YW, Hansen MS, Lungren MP. "fastMRI+, Clinical pathology annotations for knee and brain fully sampled magnetic resonance imaging data." *Scientific Data* 9, 152 (2022). DOI 10.1038/s41597-022-01255-z，PMID 35383186，PMCID PMC8983757。

nature.com 与 pubmed 在本机直接访问受阻，改用任务说明里给的 Europe PMC REST API：

- 检索确认这是目标论文：`https://www.ebi.ac.uk/europepmc/webservices/rest/search?query=fastMRI%2B%20clinical%20pathology%20annotations%20knee%20brain&format=json`
- 全文 XML（下面所有引用的出处）：`https://www.ebi.ac.uk/europepmc/webservices/rest/PMC8983757/fullTextXML`
- 人类可读镜像：`https://www.ncbi.nlm.nih.gov/pmc/articles/PMC8983757/`；DOI 链接：`https://doi.org/10.1038/s41597-022-01255-z`

### 问题 1：标注者是否逐层看完整卷？未标注层是"确认无异常"还是"未审阅"？

Methods → Annotations 段，原文（逐句引用）：

> "Annotation was performed with bounding box annotation to include the relevant label for a given pathology on a slice-by-slice level."

> "Each knee examination consisted of a single series (either proton density (PD) or T2-weighted) of coronal images where bounding box labels were placed on each slice where representative pathology was identified."

> "If no relevant pathology was identified on an examination, no labels were provided."

**round 1 复核后更正**：第三句是 examination（整卷）级别的规则，不能直接拿来当"某一层没发现病理，这一层就不给标签"的 slice 级证据——原稿这样用是错的。第二句里的 "representative" 也留了口子：它同样可以读成"只标有代表性的那几层"（严重程度/显著性筛选，或者一个跨多层的病灶只挑代表层打框），不必然是"每一层都做过有/无病理的判断"。评审意见原话："'representative' is at least as consistent with a severity/salience filter (or marking only representative slices of a multi-slice lesion) as with 'every slice was adjudicated present/absent.'"

三句合起来，文献能确认的只有：标注粒度是"层"（第一句），标注在序列层面按 slice 记录框（第二句）。但没有一句直接给出 slice 级的"无发现→不给该层标签"规则；第三句给的是 exam 级规则，套不到 slice 级。

回答：**文献未能解决这个问题。** 项目不依赖这一推断：已标注卷内没有落框的层，既不当"确认无异常"，也不当负样本用（见第 0 节 `UNANNOTATED_SLICES_IN_ANNOTATED_VOLUMES_ARE_NEGATIVE = False`），第 2 节末尾另有一条支持性但非结论性的数据观察。

### 问题 2：完全没有标注的卷是"读片为正常"还是"未标注"？

Methods → Annotations 段，原文：

> "All 1172 fastMRI knee MRI raw dataset studies were reconstructed and clinically annotated for fastMRI+."

> "If no relevant pathology was identified on an examination, no labels were provided."

第一句明确：全部 1172 例是普查式全覆盖，不是抽样——每一卷都走过了"clinically annotated"的流程，不存在"跳过某些卷不看"的情况。第二句直接把"零标签"翻译成"没有发现相关病理"（no relevant pathology was identified），而不是"没有被标注/没时间看"。

回答：**是"读片为正常"，不是"漏标"；198 卷可视为全负样本纳入（`UNANNOTATED_VOLUMES_ARE_NEGATIVE = True`）。**

### 如实披露的限制（不影响上面的结论，但影响假阳性率解读）

Methods 原文承认单一标注者、无复核：

> "Note there are several limitations to this dataset that bear acknowledgement. First, while the annotators are subspecialist radiologists in practice at leading academic medical centers, the lack of multiple annotators/repeated annotations to determine inter-rater/intra-rater reliability metrics or ensure consensus agreement is a limitation and should be considered in the use of these labels."

即：exam 级协议已确认为"零标签=正常读片"，但协议不保证零漏诊——单一放射科医生仍可能漏掉真实存在的病灶。这是标注质量问题，独立于"某一层是否被逐层审阅过"（后者见问题 1，round 1 复核后判定文献未解决，本项目已不再依赖它）。这一残余不确定性在 `UNANNOTATED_VOLUMES_ARE_NEGATIVE = True` 之下依然存在，消不掉。

另外，"Artifact" 是唯一的膝关节 study-level 标签（Table 1 脚注：*Artifact is study-level label*，13 例），在 `knee.csv` 里以 slice 0、无 x/y/width/height 坐标的形式写入（Data Records 段："Study-level labels are marked as 'Yes' in column 'Study Level' for slice 0 of the corresponding subjects with no specified bounding box information."）。本任务对"完全无标注"的判定标准是"`knee.csv` 里连一行都没有"，13 例 Artifact 只影响 974 例"已标注"卷内部的标注形式，不影响 198 例的认定，故不改变结论。

## 2. 数据交叉验证

命令（按任务对 Step 2 的澄清，删掉了原片段里那个无操作的 `with h5py.File(...) as h: pass` 空块，逻辑保留：数一遍无标注卷、按 `acquisition` 属性分组）：

```bash
PYTHONNOUSERSITE=1 ~/anaconda3/envs/nvgen/bin/python - <<'PY'
import csv, glob, h5py, os
from collections import Counter

ann = list(csv.DictReader(open("/data2/congcong/data/FM_data/fastMRI_lh_brain_knee/Annotations/knee.csv")))
vols = {r["file"] for r in ann}
fs = {os.path.basename(f)[:-3] for f in glob.glob(
    "/data2/congcong/data/FM_data/fastMRI_lh_brain_knee/kspace/knee/multicoil_*/*.h5")}
missing = sorted(fs - vols)
print(f"total volumes on disk: {len(fs)}")
print(f"total annotated volumes (>=1 row in knee.csv): {len(vols)}")
print(f"volumes with no annotation: {len(missing)}")

acq = Counter()
for v in missing:
    p = glob.glob(f"/data2/congcong/data/FM_data/fastMRI_lh_brain_knee/kspace/knee/multicoil_*/{v}.h5")[0]
    with h5py.File(p) as h:
        acq[h.attrs.get("acquisition", "?")] += 1
print("acquisition of unannotated volumes:", dict(acq))

acq_all = Counter()
for v in fs:
    p = glob.glob(f"/data2/congcong/data/FM_data/fastMRI_lh_brain_knee/kspace/knee/multicoil_*/{v}.h5")[0]
    with h5py.File(p) as h:
        acq_all[h.attrs.get("acquisition", "?")] += 1
print("acquisition of ALL volumes:", dict(acq_all))
PY
```

原始输出：

```
total volumes on disk: 1172
total annotated volumes (>=1 row in knee.csv): 974
volumes with no annotation: 198
acquisition of unannotated volumes: {'CORPD_FBK': 130, 'CORPDFS_FBK': 68}
acquisition of ALL volumes: {'CORPD_FBK': 584, 'CORPDFS_FBK': 588}
```

解读：

- 974 已标注 + 198 未标注 = 1172，与 spec §1.7、任务背景给出的数字一致。
- 全体 1172 卷两种采集几乎对半（584 / 588，即 49.8% / 50.2%）；未标注的 198 卷里 CORPD_FBK（不压脂）130 例（65.7%）、CORPDFS_FBK（压脂）68 例（34.3%）。换成"该采集类型里零标注的比例"：CORPD_FBK 130/584 = 22.3%，CORPDFS_FBK 68/588 = 11.6%——不压脂序列的零标注比例约是压脂序列的 2 倍，不是完全均衡。
- 按任务简介自带的判据（"若无标注卷在两种采集上分布均衡，更像'读片正常'；若集中在一种，更像漏标"），2 倍差异不是均衡，但也远不是集中到一种（不是那种 90%+ 一边倒的偏斜）。这个幅度的差异有不需要"漏标"就能解释的临床原因：压脂 PD 序列常因怀疑骨髓水肿/软骨等病变而加做或被优先解读，阳性率天然更高；不压脂 PD 覆盖更多常规送检，读出阴性的比例更高完全合理。这一数据模式与文献结论（零标签=读片正常）不矛盾，是次要的支持证据，本任务的结论仍以第 1 节的文献原文为主要依据。
- 交叉验证的副产品：也核实了任务背景给出的"16154 个 2D 框"这个数字。命令：

  ```bash
  PYTHONNOUSERSITE=1 ~/anaconda3/envs/nvgen/bin/python -c "
  import csv
  rows = list(csv.DictReader(open('/data2/congcong/data/FM_data/fastMRI_lh_brain_knee/Annotations/knee.csv')))
  print('total rows', len(rows))
  print('study_level==Yes rows', len([r for r in rows if r['study_level'] == 'Yes']))
  print('bounding-box rows', len([r for r in rows if r['study_level'] != 'Yes']))
  "
  ```

  输出：`total rows 16167` / `study_level==Yes rows 13` / `bounding-box rows 16154`。`knee.csv` 共 16167 行，其中 13 行是 study-level 的 "Artifact" 标签（slice=0，无坐标，即 Table 1 脚注和 Data Records 段描述的那种记录），其余 16154 行才是真正的逐层边界框，与任务背景给出的数字一致。

### 为什么仍然认为病灶是逐层标满的（支持性观察，非结论依据）

命令：

```bash
PYTHONNOUSERSITE=1 ~/anaconda3/envs/nvgen/bin/python - <<'PY'
import csv
from collections import defaultdict
rows = list(csv.DictReader(open("/data2/congcong/data/FM_data/fastMRI_lh_brain_knee/Annotations/knee.csv")))
by = defaultdict(list)
for r in rows:
    if r["study_level"].strip() == "Yes":
        continue
    by[(r["file"], r["label"].strip())].append(int(r["slice"]))
runs = [sorted(v) for v in by.values()]
gaps = [b - a for v in runs for a, b in zip(v, v[1:])]
n = len(runs)
print(f"(volume,label) groups {n}")
print(f"single-slice groups {sum(1 for v in runs if len(v) == 1)} ({sum(1 for v in runs if len(v) == 1)/n:.0%})")
print(f"consecutive gaps {sum(1 for g in gaps if g == 1)}, non-consecutive gaps {sum(1 for g in gaps if g > 1)}")
print(f"slices per group: median {sorted(len(v) for v in runs)[n//2]}, max {max(len(v) for v in runs)}")
PY
```

输出：

```
(volume,label) groups 2585
single-slice groups 156 (6%)
consecutive gaps 11348, non-consecutive gaps 436
slices per group: median 5, max 33
```

含义：一个病灶一旦被发现，几乎总是被逐层连续标满其空间范围，而不是只挑一张代表层——单层病灶只占 6%，层间距里 96%（11348/11784）是连续相邻层。这与"逐层标注"的操作习惯一致，但它证明的是"找到病灶之后怎么标"，不能反过来证明"没打框的层确实被看过、确认无病灶"，所以只是支持性观察，不是结论依据。

## 3. 决策（供 Task 7 使用）

```
UNANNOTATED_VOLUMES_ARE_NEGATIVE = True
UNANNOTATED_SLICES_IN_ANNOTATED_VOLUMES_ARE_NEGATIVE = False
```

- 198 卷完全无标注的卷：作为全负样本（整卷所有层、所有类别族）纳入训练/评估集，不排除；约 198×35≈6900 张负样本层，是本数据集负样本的来源。
- 已标注卷内没有落框的层：从样本索引中整体剔除，既不算正类也不算负类，不参与训练/评估。

决策代价：排除这些层，损失的是已标注卷内本可以多拿到的一批负样本；这个代价可恢复——以后要改，重新跑一次导出脚本就行，不会引入静默污染。而如果这里判断反了、把这些层当负样本纳入训练（即误判为 True），后果是训练检测器压制真实存在的病灶，这在重采闸门场景下不可接受。
