# 2026-09-08 验证报告：arm-B 首跑、重叠基线口径、MTR_110 ann 15

这份文档的目的是**让每一个数字都能被独立重跑**。每一节的结构是：结论 → 怎么算的（可直接粘贴的命令）→ 原始输出（未加工）→ 这个数支撑了哪个决策。

## 复现环境

```bash
# 解释器（必须带 PYTHONNOUSERSITE，~/.local 里有 torch 2.11/cu130 会覆盖 env）
PY=~/anaconda3/envs/nvgen/bin/python          # Python 3.11.15, torch 2.5.1+cu121

# 代码
cd /data0/congcong/code/Project_Doing/foundation_model-armb   # 分支 armb @ bce4219
# 或 /data0/congcong/code/Project_Doing/foundation_model      # 分支 plan-v5，同一 .git

# 数据（只读）
/data2/congcong/data/FM_data/derived/skmtea/m1/               # 155 卷，44 GB

# 训练产物（本地，runs/ 已在 .gitignore 中，不入库）
runs/armb_fold0_first/{config.json,metrics.jsonl,last.pt}
```

所有命令都从仓库根目录执行，前缀统一为 `PYTHONNOUSERSITE=1 PYTHONPATH=. $PY`。

---

## 1. 训练验收门：无 NaN，八项损失全部下降

**结论**：300 步、682 秒、单卡 A800（与另一用户的任务共享），全程无非有限值，八个损失项在 281–300 步的均值都低于 1–20 步。**门通过**。

**怎么跑的**

```bash
PYTHONNOUSERSITE=1 PYTHONPATH=. CUDA_VISIBLE_DEVICES=0 $PY \
  scripts/train_armb_minimal.py --steps 300 --batch 2 --workers 4 \
  --eval-every 50 --out runs/armb_fold0_first
```

**怎么验的**

```bash
PYTHONNOUSERSITE=1 $PY - <<'PY'
import json
import numpy as np
rows=[json.loads(l) for l in open('runs/armb_fold0_first/metrics.jsonl')]
keys=["loss","mask","presence","cls","l1","giou","host_ce","rel_bce"]
print(f"steps logged: {len(rows)}   wall: {rows[-1]['seconds']:.0f}s   {rows[-1]['seconds']/len(rows):.2f}s/step")
nan=[r["step"] for r in rows if any(not np.isfinite(r[k]) for k in keys)]
print("non-finite steps:", nan or "NONE")
def win(lo,hi):
    sel=[r for r in rows if lo<=r["step"]<=hi]
    return {k: float(np.mean([r[k] for r in sel])) for k in keys}
a,b=win(1,20),win(281,300)
print(f"\n{'term':10}{'steps 1-20':>12}{'steps 281-300':>15}{'change':>10}  gate")
allpass=True
for k in keys:
    d=b[k]-a[k]; ok=d<0; allpass &= ok
    print(f"{k:10}{a[k]:12.4f}{b[k]:15.4f}{d:+10.4f}  {'PASS' if ok else 'FAIL'}")
print(f"\nGATE (every term lower): {'PASS' if allpass else 'FAIL'}")
ev=[r for r in rows if "val_loss" in r]
print("\nvalidation (host acc = NOT EVIDENCE):")
for r in ev:
    print(f"  step {r['step']:>3}  val_loss {r['val_loss']:6.3f}   host_acc {r['val_host_acc_NOT_EVIDENCE']:.3f}  over {r['val_matched']} matched")
print("\ngrad norm: max %.1f  final-20 mean %.1f" % (max(r["grad_norm"] for r in rows), np.mean([r["grad_norm"] for r in rows[-20:]])))
PY
```

**原始输出**

```
steps logged: 300   wall: 682s   2.27s/step
non-finite steps: NONE

term        steps 1-20  steps 281-300    change  gate
loss           11.9399         8.4282   -3.5117  PASS
mask            1.0741         0.9382   -0.1358  PASS
presence        0.5923         0.5217   -0.0706  PASS
cls             1.1621         1.0029   -0.1592  PASS
l1              0.9804         0.5378   -0.4426  PASS
giou            1.3830         1.1625   -0.2205  PASS
host_ce         0.9595         0.6149   -0.3446  PASS
rel_bce         0.4837         0.3364   -0.1474  PASS

GATE (every term lower): PASS

validation (host acc = NOT EVIDENCE):
  step  50  val_loss  9.678   host_acc 0.652  over 46 matched
  step 100  val_loss  8.868   host_acc 0.652  over 46 matched
  step 150  val_loss  8.507   host_acc 0.652  over 46 matched
  step 200  val_loss  8.673   host_acc 0.630  over 46 matched
  step 250  val_loss  8.682   host_acc 0.674  over 46 matched
  step 300  val_loss  8.636   host_acc 0.652  over 46 matched

grad norm: max 75.8  final-20 mean 22.7
```

**注意**：`host_acc` 键名里带 `NOT_EVIDENCE` 是故意的。这一版模型的几何特征吃的是**真值** mask、presence 和匹配框，而 IoA 正是几何通道之一，所以关系头学 `argmax IoA` 就能拿分——那正是它要打败的基线本身。这个数不能作为 M1 命题的证据。见计划文档的 ⚠ 段与 `anatobind/model/armb.py` 顶部注释。

**支撑的决策**：计划文档第 4 步验收门通过；可以进入第 5 步（人工看图）。

---

## 2. 模型 vs 重叠基线（裁块口径，n=46）

**结论**：模型 0.652，**输给了零学习的 argmax-IoA 基线 0.804**。这与方案 §9.1 自述的「干净数据上 seg-then-lookup 几乎必然不输」一致。

**怎么验的**

```bash
PYTHONNOUSERSITE=1 PYTHONPATH=. CUDA_VISIBLE_DEVICES=0 $PY - <<'PY'
import collections, torch, numpy as np
from pathlib import Path
from torch.utils.data import DataLoader
from anatobind.model.armb import ArmBMinimal
from anatobind.train.dataset import SkmteaArmBDataset, collate, load_fold, SEG_NAMES
R=Path('/data2/congcong/data/FM_data/derived/skmtea/m1'); dev=torch.device('cuda')
ck=torch.load('runs/armb_fold0_first/last.pt', map_location='cpu', weights_only=False)
m=ArmBMinimal().to(dev); m.load_state_dict(ck['model']); m.eval()
_,val=load_fold(R,0)
dl=DataLoader(SkmteaArmBDataset(val,R,patch=(64,128,128),train=False,seed=0),batch_size=2,collate_fn=collate,num_workers=4)
gt,pred,ioa_pred=[],[],[]
with torch.no_grad():
    for b in dl:
        b={k:(v.to(dev) if torch.is_tensor(v) else ([t.to(dev) for t in v] if isinstance(v,list) and v and torch.is_tensor(v[0]) else v)) for k,v in b.items()}
        with torch.autocast('cuda',dtype=torch.bfloat16):
            o=m(b)
        gm=torch.stack([(b['seg']==k+1) for k in range(6)],1).float()
        for i,(pi,ti) in enumerate(o['matches']):
            if not pi.numel(): continue
            h=b['host_label'][i][ti]
            for q,t,hl in zip(pi.tolist(),ti.tolist(),h.tolist()):
                if hl==0: continue
                gt.append(hl)
                pred.append(int(o['host_logits'][i,q,:6].argmax())+1)
                z0,y0,x0,z1,y1,x1=b['boxes'][i][t].round().long().tolist()
                vol=max((z1-z0)*(y1-y0)*(x1-x0),1)
                ioa=gm[i][:,z0:z1,y0:y1,x0:x1].flatten(1).sum(-1)/vol
                ioa=ioa.masked_fill(~b['present'][i],-1)
                ioa_pred.append(int(ioa.argmax())+1)
gt,pred,ioa_pred=map(np.array,(gt,pred,ioa_pred))
print(f"scored instances: {len(gt)}")
print(f"model  host accuracy : {(pred==gt).mean():.3f}")
print(f"argmax-IoA baseline  : {(ioa_pred==gt).mean():.3f}   <- seg-then-lookup, no learning")
print(f"model agrees with argmax-IoA on {(pred==ioa_pred).mean():.3f} of instances")
print(f"majority-class baseline: {max(collections.Counter(gt.tolist()).values())/len(gt):.3f}")
print("\nGT host distribution:", {SEG_NAMES[k]:v for k,v in sorted(collections.Counter(gt.tolist()).items())})
print("model predictions    :", {SEG_NAMES[k]:v for k,v in sorted(collections.Counter(pred.tolist()).items())})
PY
```

**原始输出**

```
scored instances: 46
model  host accuracy : 0.652
argmax-IoA baseline  : 0.804   <- seg-then-lookup, no learning
model agrees with argmax-IoA on 0.717 of instances
majority-class baseline: 0.413

GT host distribution: {'patellar cartilage': 8, 'femoral cartilage': 19, 'tibial cartilage lateral': 4, 'meniscus medial': 10, 'meniscus lateral': 5}
model predictions    : {'patellar cartilage': 9, 'femoral cartilage': 18, 'tibial cartilage lateral': 2, 'meniscus medial': 11, 'meniscus lateral': 6}
```

**附带一条**：模型的预测分布几乎复刻了真值分布（股骨软骨 18 vs 19、内侧半月板 11 vs 10），说明它 300 步内学到的是边际分布，不是实例级绑定。它也没有塌成一个常数类（多数类基线 0.413，它 0.652），与 argmax-IoA 只有 71.7% 一致，所以并未简单抄袭重叠规则。

**口径说明（重要）**：这里的 n=46，比第 6 节的 n=57 少 11 个。原因是本节走的是训练用的 `SkmteaArmBDataset`，每卷只取一个 `64×128×128` 裁块，落在裁块外的框被丢弃、跨边界的框被裁短，**重叠率随之改变**。所以 0.804 是「模型在裁块条件下实际面对的难度」，第 6 节的 0.860 是「完整标注框上的数据性质」。引用任何一个都必须写明是哪个。

**支撑的决策**：确认最小通路的准确率不构成 M1 证据；同时给出重叠基线的量级。

---

## 3. 标注宿主与最大重叠结构不一致的实例（n=57 中 8 个）

**结论**：fold 0 验证集 57 个宿主已知的分割内实例中，**8 个（14.0%）的标注宿主不是其框重叠最多的结构**。八个全部是解剖上讲得通的邻接混淆，不是数据错误。

**怎么验的**

```bash
PYTHONNOUSERSITE=1 PYTHONPATH=. $PY - <<'PY'
import csv, numpy as np, nibabel as nib
from pathlib import Path
from anatobind.train.dataset import load_fold, SEG_NAMES
R=Path('/data2/congcong/data/FM_data/derived/skmtea/m1')
_,val=load_fold(R,0)
bad=[]; tot=0
for s in val:
    seg=np.asanyarray(nib.load(R/s/'seg.nii.gz').dataobj)
    for r in csv.DictReader(open(R/s/'boxes.csv')):
        if r['layer']!='in_seg' or not r['host_label'].strip(): continue
        tot+=1
        x0,y0,z0,x1,y1,z1=(int(r[k]) for k in ('x0','y0','z0','x1','y1','z1'))
        sub=seg[x0:x1,y0:y1,z0:z1]; vol=sub.size
        ioa={k:float((sub==k).sum())/vol for k in range(1,7) if (seg==k).any()}
        host=int(r['host_label']); top=max(ioa,key=ioa.get)
        if top!=host:
            bad.append((s,r['ann_id'],r['supercategory'],host,top,ioa[host],ioa[top],r['host_side']))
print(f"fold0 val in_seg instances with known host: {tot}")
print(f"annotated host != max-overlap structure:    {len(bad)}  ({100*len(bad)/tot:.1f}%)\n")
for s,a,sup,h,t,ih,it,side in bad:
    print(f"{s} ann{a:>4} {sup[:16]:16} host={h}({SEG_NAMES[h][:22]:22}) IoA={ih:.3f} | max-overlap={t}({SEG_NAMES[t][:22]:22}) IoA={it:.3f}  side={side}")
PY
```

**原始输出**

```
fold0 val in_seg instances with known host: 57
annotated host != max-overlap structure:    8  (14.0%)

MTR_015 ann  36 Cartilage Lesion host=2(femoral cartilage     ) IoA=0.017 | max-overlap=5(meniscus medial       ) IoA=0.078  side=single
MTR_052 ann  82 Cartilage Lesion host=2(femoral cartilage     ) IoA=0.034 | max-overlap=1(patellar cartilage    ) IoA=0.086  side=single
MTR_052 ann  83 Cartilage Lesion host=1(patellar cartilage    ) IoA=0.004 | max-overlap=2(femoral cartilage     ) IoA=0.044  side=single
MTR_069 ann  31 Meniscal Tear    host=6(meniscus lateral      ) IoA=0.210 | max-overlap=4(tibial cartilage later) IoA=0.300  side=lateral
MTR_104 ann   1 Meniscal Tear    host=6(meniscus lateral      ) IoA=0.357 | max-overlap=4(tibial cartilage later) IoA=0.597  side=lateral
MTR_110 ann  15 Meniscal Tear    host=5(meniscus medial       ) IoA=0.000 | max-overlap=2(femoral cartilage     ) IoA=0.091  side=medial
MTR_163 ann 132 Meniscal Tear    host=6(meniscus lateral      ) IoA=0.008 | max-overlap=2(femoral cartilage     ) IoA=0.049  side=lateral
MTR_236 ann 215 Meniscal Tear    host=6(meniscus lateral      ) IoA=0.075 | max-overlap=2(femoral cartilage     ) IoA=0.126  side=lateral
```

**解读**：半月板垫在胫骨平台上、顶着股骨髁软骨，所以框住半月板撕裂必然大片压到这两者（MTR_069/104/163/236）；髌骨软骨与股骨滑车软骨隔缝相对，跨缝的框会互相盖过（MTR_052 ann 82 与 83 恰是镜像的一对）。重叠率分不出「在里面」和「挨着」，医生分得出。

**支撑的决策**：这 14% 是关系建模要转化的空间；这些实例必须保留，剔掉就是在造偏袒基线的测试集。

---

## 4. MTR_110 全卷诊断

**结论**：这卷的内侧半月板分割完整可用（ann 16 同宿主、772 体素、零间隙）；**异常的只有 ann 15 一个框**，它与宿主零重叠，最近的内侧半月板在 1.25 mm 外。

**怎么验的**

```bash
PYTHONNOUSERSITE=1 PYTHONPATH=. $PY - <<'PY'
import csv, numpy as np, nibabel as nib
from pathlib import Path
R=Path('/data2/congcong/data/FM_data/derived/skmtea/m1/MTR_110')
img=nib.load(R/'image_clean_e1.nii.gz'); sp=np.array(img.header.get_zooms()[:3])
seg=np.asanyarray(nib.load(R/'seg.nii.gz').dataobj)
print("volume", seg.shape, "spacing (x,y,z) mm", np.round(sp,4))
NAMES={1:'patellar cart',2:'femoral cart',3:'tibial med',4:'tibial lat',5:'meniscus med',6:'meniscus lat'}
for k in range(1,7):
    m=seg==k
    if not m.any(): print(f"  {k} {NAMES[k]:15} ABSENT"); continue
    idx=np.argwhere(m); lo,hi=idx.min(0),idx.max(0)+1
    print(f"  {k} {NAMES[k]:15} {m.sum():>7} vox  bbox {tuple(lo)}-{tuple(hi)}")
for r in csv.DictReader(open(R/'boxes.csv')):
    x0,y0,z0,x1,y1,z1=(int(r[k]) for k in ('x0','y0','z0','x1','y1','z1'))
    sub=seg[x0:x1,y0:y1,z0:z1]
    host=r['host_label'].strip()
    inside={int(l):int(c) for l,c in zip(*np.unique(sub,return_counts=True)) if l>0}
    size_mm=np.round((np.array([x1-x0,y1-y0,z1-z0])*sp),1)
    print(f"  ann{r['ann_id']:>4} {r['layer']:8} {r['supercategory'][:15]:15} tid={r['tissue_id']:>2} host={host or '-':>2} side={r['host_side']:18} box{(x0,y0,z0)}-{(x1,y1,z1)} = {tuple(size_mm)} mm")
    print(f"        labels in box: {inside or 'BACKGROUND ONLY'}")
    if host:
        h=int(host); m=seg==h
        if m.any():
            idx=np.argwhere(m); lo=np.array([x0,y0,z0]); hi=np.array([x1,y1,z1])
            gap_mm=(np.maximum(np.maximum(lo-idx, idx-(hi-1)),0)*sp).max(1)
            j=int(gap_mm.argmin())
            print(f"        nearest host-{h} voxel at {tuple(idx[j])}, gap {gap_mm.min():.2f} mm")
PY
```

**原始输出（节选，完整见上游命令）**

```
volume (256, 256, 160) spacing (x,y,z) mm [0.625  0.625  0.8007]
  1 patellar cart      6443 vox  bbox (95, 61, 46)-(137, 83, 96)
  2 femoral cart      31625 vox  bbox (107, 72, 32)-(165, 169, 116)
  3 tibial med         7146 vox  bbox (156, 102, 83)-(174, 172, 114)
  4 tibial lat         6166 vox  bbox (150, 115, 31)-(168, 166, 67)
  5 meniscus med       6775 vox  bbox (151, 98, 82)-(171, 167, 113)
  6 meniscus lat       3781 vox  bbox (144, 114, 32)-(166, 160, 60)

  ann  15 in_seg   Meniscal Tear   tid= 1 host= 5 side=medial   box(152,109,89)-(160,137,104) = (5.0, 17.5, 12.0) mm
        labels in box: {2: 305}
        nearest host-5 voxel at (160, 107, 90), gap 1.25 mm
  ann  16 in_seg   Meniscal Tear   tid= 1 host= 5 side=medial   box(162,106,101)-(168,126,112) = (3.8, 12.5, 8.8) mm
        labels in box: {2: 133, 3: 63, 5: 772}
        nearest host-5 voxel at (162, 106, 101), gap 0.00 mm
```

**两条附带结论**

1. **Z 是内外侧方向**：内侧半月板 z 82–113，外侧半月板 z 32–60，两者在 z 上完全分开。ann 15 的 z 89–104 落在内侧区间，与 `side=medial` 自洽。
2. **ann 15 整个落在内侧半月板的三维包围盒 (151,98,82)-(171,167,113) 内部**，不是跑到别处去了。

**配图**：`~/figs/anatobind_m1_fold0/MTR_110_ann15_vs_ann16_diagnostic.png`（上排 ann 15 六层，每层宿主体素均为 0；下排 ann 16 六层，72/83/89/78/57/33）。

---

## 5. ann 15 框内到底装着什么

**结论**：框内 90.9% 是这套分割不标注的组织，9.1% 是股骨软骨，**半月板 0 体素**。

**怎么验的**

```bash
PYTHONNOUSERSITE=1 PYTHONPATH=. $PY - <<'PY'
import numpy as np, nibabel as nib
from pathlib import Path
R=Path('/data2/congcong/data/FM_data/derived/skmtea/m1/MTR_110')
seg=np.asanyarray(nib.load(R/'seg.nii.gz').dataobj)
img=nib.load(R/'image_clean_e1.nii.gz'); sp=np.array(img.header.get_zooms()[:3])
vol=img.get_fdata(dtype=np.float32)
NAMES={0:'未标注组织(骨/韧带/积液/肌肉等)',1:'髌骨软骨',2:'股骨软骨',3:'内侧胫骨软骨',
       4:'外侧胫骨软骨',5:'内侧半月板',6:'外侧半月板'}
x0,y0,z0,x1,y1,z1=152,109,89,160,137,104
sub=seg[x0:x1,y0:y1,z0:z1]; iv=vol[x0:x1,y0:y1,z0:z1]; n=sub.size
print(f"框尺寸 {x1-x0}×{y1-y0}×{z1-z0} 体素 = {tuple(np.round((np.array([x1-x0,y1-y0,z1-z0])*sp),1))} mm")
print(f"体素总数 {n}\n框内各类组织占比:")
for l,c in sorted(zip(*np.unique(sub,return_counts=True)), key=lambda t:-t[1]):
    print(f"  {NAMES[int(l)]:28} {int(c):>5} 体素  {100*c/n:5.1f}%")
whole=vol[vol>0]
print(f"\n框内信号强度: 中位 {np.median(iv):.3g}   全卷中位 {np.median(whole):.3g}   比值 {np.median(iv)/np.median(whole):.2f}")
print(f"框内亮体素(>全卷 75 分位)占比 {100*(iv>np.percentile(whole,75)).mean():.1f}%")
PY
```

**原始输出**

```
框尺寸 8×28×15 体素 = (5.0, 17.5, 12.0) mm
体素总数 3360

框内各类组织占比:
  未标注组织(骨/韧带/积液/肌肉等)            3055 体素   90.9%
  股骨软骨                           305 体素    9.1%

框内信号强度: 中位 2.84e+06   全卷中位 3.5e+06   比值 0.81
框内亮体素(>全卷 75 分位)占比 7.0%
```

**注意**：SKM-TEA 只分割 6 个结构（髌/股/内外侧胫骨软骨、内外侧半月板）。**骨、韧带、积液、肌肉、脂肪全部是标签 0**，所以「90.9% 未标注」不等于「空的」。信号强度也支持这点：框内中位强度是全卷的 0.81 倍，偏暗但非纯噪声，7% 的体素亮过全卷 75 分位。

---

## 6. 组织族级 vs 标签级：M1 判据该算在哪一层

**结论**：oracle 重叠绑定器的**组织族级与标签级准确率都是 0.860**。这说明只要它把族选对，侧别必然也对——侧别对它是白送的，不构成独立考验。**全部区分度落在组织族层，而该层是纯人工真值。**

**为什么要做这个测量**：并行会话（Codex）的架构评审指出，关系真值并非完全独立于重叠——组织族来自标注者的 `tissue_id`，但内外侧由 D5 规则借助分割 mask 解析（311 例中 144 例、46% 属此类）。若主判据算在标签级，就有一截是让重叠基线预测自己的构造规则。

**怎么验的**

```bash
PYTHONNOUSERSITE=1 PYTHONPATH=. $PY - <<'PY'
import csv, numpy as np, nibabel as nib
from pathlib import Path
from anatobind.train.dataset import load_fold
R=Path('/data2/congcong/data/FM_data/derived/skmtea/m1')
FAM={1:{5,6},4:{2},5:{1},6:{3,4}}          # tissue_id -> 该组织族的分割标签
_,val=load_fold(R,0)
rec=[]
for s in val:
    seg=np.asanyarray(nib.load(R/s/'seg.nii.gz').dataobj)
    for r in csv.DictReader(open(R/s/'boxes.csv')):
        if r['layer']!='in_seg' or not r['host_label'].strip(): continue
        x0,y0,z0,x1,y1,z1=(int(r[k]) for k in ('x0','y0','z0','x1','y1','z1'))
        sub=seg[x0:x1,y0:y1,z0:z1]; vol=sub.size
        ioa={k:float((sub==k).sum())/vol for k in range(1,7) if (seg==k).any()}
        rec.append(dict(host=int(r['host_label']),tid=int(r['tissue_id']),
                        side=r['host_side'],pred=max(ioa,key=ioa.get)))
def rate(rows,fn): return (sum(fn(r) for r in rows)/len(rows), len(rows)) if rows else (float('nan'),0)
lab=lambda r: r['pred']==r['host']
fam=lambda r: r['pred'] in FAM[r['tid']]
side_res=[r for r in rec if r['side'] in ('medial','lateral')]
single  =[r for r in rec if r['side']=='single']
print(f"fold0 验证集,宿主已知的分割内实例: {len(rec)}")
print(f"  其中侧别由重叠解析 (medial/lateral): {len(side_res)}")
print(f"  其中组织族 1:1 直接映射 (single)   : {len(single)}\n")
print("argmax-IoA 基线(用真值 mask):")
a,n=rate(rec,lab);   print(f"  标签级(含侧别)     {a:.3f}  n={n}")
a,n=rate(rec,fam);   print(f"  组织族级(纯人工真值) {a:.3f}  n={n}")
print()
a,n=rate(single,lab);   print(f"  只看 single 实例   {a:.3f}  n={n}")
a,n=rate(side_res,lab); print(f"  只看侧别解析实例   {a:.3f}  n={n}")
a,n=rate(side_res,fam); print(f"    …其中组织族对   {a:.3f}  n={n}")
PY
```

**原始输出**

```
fold0 验证集,宿主已知的分割内实例: 57
  其中侧别由重叠解析 (medial/lateral): 23
  其中组织族 1:1 直接映射 (single)   : 34

argmax-IoA 基线(用真值 mask):
  标签级(含侧别)     0.860  n=57
  组织族级(纯人工真值) 0.860  n=57

  只看 single 实例   0.912  n=34
  只看侧别解析实例   0.783  n=23
    …其中组织族对   0.783  n=23
```

**支撑的决策**：M1 判据的 10 个百分点门槛**算在组织族级 ABA 上**，标签级与侧别单列为次要口径并披露其真值来源。已写入 `RESEARCH_PLAN.md` §0、§9.1、§9.5、§9.7、§13.4（commit `bce4219`）。

---

## 7. 宿主判定用的是「框 + 4 体素邻域」，不是原始框

**结论**：`host_side` / `host_ratio` 由**外扩 4 个体素后的邻域**决定。这解释了 MTR_110 ann 15 为什么在原始框零重叠的情况下仍得到 `medial, ratio=1.000`——邻域里有 59 个内侧半月板体素、0 个外侧。**判定有据，label 没错。**

**这一节是对我此前表述的更正**：我在会话中两次把这条规则说成「内外侧由与内/外侧 mask 的**重叠比**决定」，漏掉了外扩这一步。方案 §5.1 的原文也只写了「重叠比」。实现里的 `pad=4` 应当在 §5.1 中补明。

**代码出处**：`anatobind/data_engine/skmtea.py:111-131`

```python
def host_seg_label(seg_h5, box, tissue_id, pad=4, ambiguous=(0.4, 0.6)):
    labels = TISSUE_TO_SEG.get(tissue_id, ())
    ...
    x0, y0, z0, x1, y1, z1 = box
    sub = seg_h5[max(x0-pad,0):x1+pad, max(y0-pad,0):y1+pad, max(z0-pad,0):z1+pad]   # ← 外扩
    counts = {l: int((sub == l).sum()) for l in labels}
    n = sum(counts.values())
    if len(labels) == 1:      # 单标签组织：tissue_id 定宿主，缺 mask 不删记录
        return {"label": labels[0], "side": "single" if n else "single_no_overlap", ...}
    if n == 0:                # 两标签组织且邻域内无 mask → unresolved（不是 none）
        return {"label": None, "side": "unresolved", ...}
    medial, lateral = labels                # (5,6) 或 (3,4)，内侧在前
    ratio = counts[medial] / n
    ...
```

注意 `pad=4` 作用在 **h5 网格**（面内 0.3125 mm）上，导出后面内做了 2 倍降采样，所以在导出网格（0.625 mm）上等效于面内 pad=2、z 方向 pad=4。

**怎么验的**

```bash
PYTHONNOUSERSITE=1 PYTHONPATH=. $PY - <<'PY'
import numpy as np, nibabel as nib
from pathlib import Path
R=Path('/data2/congcong/data/FM_data/derived/skmtea/m1')
for scan,ann,box in [('MTR_110','15',(152,109,89,160,137,104)),
                     ('MTR_040','74',(134,151,91,144,156,99)),
                     ('MTR_052','80',(165,86,88,184,115,122))]:
    seg=np.asanyarray(nib.load(R/scan/'seg.nii.gz').dataobj)
    x0,y0,z0,x1,y1,z1=box
    for name,(px,pz) in [('原始框',(0,0)),('外扩(等效 pad=4)',(2,4))]:
        sub=seg[max(x0-px,0):x1+px, max(y0-px,0):y1+px, max(z0-pz,0):z1+pz]
        m5,m6=int((sub==5).sum()),int((sub==6).sum()); n=m5+m6
        r=f"{m5/n:.3f}" if n else "—"
        print(f"{scan} ann{ann:>3} {name:18} 内侧半月板 {m5:>4}  外侧半月板 {m6:>4}  ratio={r}")
    print()
PY
```

**原始输出**

```
MTR_110 ann 15 原始框                内侧半月板    0  外侧半月板    0  ratio=—
MTR_110 ann 15 外扩(等效 pad=4)       内侧半月板   59  外侧半月板    0  ratio=1.000

MTR_040 ann 74 原始框                内侧半月板    0  外侧半月板    0  ratio=—
MTR_040 ann 74 外扩(等效 pad=4)       内侧半月板    0  外侧半月板    0  ratio=—

MTR_052 ann 80 原始框                内侧半月板    0  外侧半月板    0  ratio=—
MTR_052 ann 80 外扩(等效 pad=4)       内侧半月板    0  外侧半月板    0  ratio=—
```

三条记录与 `boxes.csv` 中的 `host_side` 完全一致（`medial` / `unresolved` / `unresolved`），**代码行为与设计相符**。

---

## 8. 尚未解决的一项，需要临床判断

**MTR_110 ann 15 的框位置。** 已排除的解释：

- 不是坐标系错误 —— 同卷其余 4 条标注全部正常压在各自宿主上；
- 不是分割缺失 —— ann 16 同为内侧半月板、772 体素、零间隙；
- 不是宿主判定错误 —— 第 7 节已验证 `medial` 判定有据；
- 不是内外侧搞反 —— z 89–104 落在内侧半月板的 z 82–113 区间内。

剩下三种可能，**图像上分不出，需要你看**：撕裂信号越出了被分割的半月板边界；框画得偏大或整体上移；原始标注确有偏差。

**我的建议是保留该实例**。它的宿主在 1.25 mm 外、框内只有股骨软骨，重叠率必然答「股骨软骨」而医生答「内侧半月板」——这正是关系模型要答对的题。把重叠率答错的实例逐个剔除，等于构造一个偏袒基线的测试集。单个实例统计上无影响，但处理原则会决定另外 7 个。

---

## 9. 待办

| # | 事项 | 状态 |
|---|---|---|
| 1 | MTR_110 ann 15 保留还是剔除 | 待用户判断 |
| 2 | 在 §5.1 补明宿主判定的 `pad=4` 邻域规则 | 待办（本报告第 7 节已记录） |
| 3 | 积液 116 例与韧带 38 例目前对 U_B 呈背景 | 待办（计划文档 F5） |
| 4 | 三处 teacher forcing 换成预测量 | 正式跑分前必做 |
| 5 | arm A（nnU-Net seg-then-lookup）尚未建立 | M1 判据的前提 |
