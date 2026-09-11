"""Oracle seg-then-lookup ceilings on the M1 export (read-only).

V0 = argmax IoA over all present labels (repo's oracle, whole box)
V1 = class-aware: candidates restricted by supercategory (Meniscal Tear -> {5,6}; Cartilage Lesion -> {1,2,3,4}),
     argmax IoA on the raw box, fallback = nearest candidate structure (mm) when all IoA == 0
V2 = class-aware, IoA on the D5-padded box (pad 2 in-plane, 4 in z on the export grid), fallback nearest
V3 = class-aware, nearest structure only (min box->mask distance in mm)
Accuracy at label level (1..6) and tissue-family level (family from tissue_id).
"""
import csv, json, sys
from pathlib import Path
import numpy as np, nibabel as nib

R = Path(sys.argv[1])
FAM = {1: {5, 6}, 4: {2}, 5: {1}, 6: {3, 4}}          # tissue_id -> seg labels of that family
CAND = {'Meniscal Tear': (5, 6), 'Cartilage Lesion': (1, 2, 3, 4)}
folds = json.loads((R / 'splits.json').read_text())['folds']
ready = [r['scan_id'] for r in csv.DictReader(open(R / 'manifest.csv')) if r['status'] == 'ok' and (R / r['scan_id'] / 'boxes.csv').exists()]
print('ready scans', len(ready))

def box_dist_mm(lo, hi, pts, sp):
    # distance from box [lo,hi) to voxel points, per axis max(0, lo-p, p-(hi-1)) * spacing
    d = np.maximum(np.maximum(lo - pts, pts - (hi - 1)), 0) * sp
    return np.sqrt((d ** 2).sum(1)).min() if len(pts) else np.inf

rec = []
for s in ready:
    seg = np.asanyarray(nib.load(R / s / 'seg.nii.gz').dataobj)
    sp = np.array(nib.load(R / s / 'seg.nii.gz').header.get_zooms()[:3], dtype=float)
    present = {k: (seg == k) for k in range(1, 7)}
    present = {k: m for k, m in present.items() if m.any()}
    pts = {k: np.argwhere(m) for k, m in present.items()}
    for r in csv.DictReader(open(R / s / 'boxes.csv')):
        if r['layer'] != 'in_seg' or not r['host_label'].strip():
            continue
        b = np.array([int(r[k]) for k in ('x0', 'y0', 'z0', 'x1', 'y1', 'z1')])
        lo, hi = b[:3], b[3:]
        sub = seg[lo[0]:hi[0], lo[1]:hi[1], lo[2]:hi[2]]; vol = max(sub.size, 1)
        ioa = {k: float((sub == k).sum()) / vol for k in present}
        plo = np.maximum(lo - np.array([2, 2, 4]), 0); phi = hi + np.array([2, 2, 4])
        psub = seg[plo[0]:phi[0], plo[1]:phi[1], plo[2]:phi[2]]; pvol = max(psub.size, 1)
        pioa = {k: float((psub == k).sum()) / pvol for k in present}
        dist = {k: box_dist_mm(lo, hi, pts[k], sp) for k in present}
        host = int(r['host_label']); tid = int(r['tissue_id']); sc = r['supercategory']
        cands = [k for k in CAND[sc] if k in present]
        v0 = max(ioa, key=ioa.get)
        def pick(score, cands):
            best = max(cands, key=lambda k: score[k]) if cands else None
            if best is None: return None
            if score[best] > 0: return best
            return min(cands, key=lambda k: dist[k])
        v1 = pick(ioa, cands); v2 = pick(pioa, cands)
        v3 = min(cands, key=lambda k: dist[k]) if cands else None
        rec.append(dict(scan=s, fold=folds[s], ann=r['ann_id'], sc=sc, tid=tid, host=host, side=r['host_side'],
                        v0=v0, v1=v1, v2=v2, v3=v3, ioa_host=ioa.get(host, 0.0), d_host=dist.get(host, np.inf)))

def acc(rows, key, level):
    ok = 0
    for x in rows:
        p = x[key]
        if p is None: continue
        ok += (p == x['host']) if level == 'label' else (p in FAM[x['tid']])
    return ok / len(rows), len(rows)

print(f"in_seg instances with known host: {len(rec)}")
for name, key in [('V0 argmax IoA all labels (repo oracle)', 'v0'), ('V1 class-aware IoA + nearest fallback', 'v1'),
                  ('V2 class-aware padded-IoA + nearest fallback', 'v2'), ('V3 class-aware nearest only', 'v3')]:
    al, n = acc(rec, key, 'label'); af, _ = acc(rec, key, 'family')
    print(f"{name:48s} label {al:.3f}  family {af:.3f}  n={n}")
print('\nper supercategory (family level):')
for sc in CAND:
    rows = [x for x in rec if x['sc'] == sc]
    print(f"  {sc:16s} n={len(rows):3d}  V0 {acc(rows,'v0','family')[0]:.3f}  V1 {acc(rows,'v1','family')[0]:.3f}  V2 {acc(rows,'v2','family')[0]:.3f}  V3 {acc(rows,'v3','family')[0]:.3f}")
print('\nper fold (family level, V0 vs V1 vs V2):')
for f in range(5):
    rows = [x for x in rec if x['fold'] == f]
    print(f"  fold {f} n={len(rows):3d}  V0 {acc(rows,'v0','family')[0]:.3f}  V1 {acc(rows,'v1','family')[0]:.3f}  V2 {acc(rows,'v2','family')[0]:.3f}")
print('\nremaining family-level errors under V2:')
for x in rec:
    if x['v2'] is not None and x['v2'] not in FAM[x['tid']]:
        print(f"  {x['scan']} ann{x['ann']:>4} {x['sc'][:15]:15} host={x['host']} pred={x['v2']} IoA_host={x['ioa_host']:.3f} d_host={x['d_host']:.2f}mm side={x['side']}")
print('\nremaining label-level errors under V2 that are family-correct (side errors):',
      sum(1 for x in rec if x['v2'] is not None and x['v2'] in FAM[x['tid']] and x['v2'] != x['host']))
json.dump(rec, open(sys.argv[2], 'w'), default=float)
