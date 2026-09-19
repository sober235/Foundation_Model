"""Step 0c: per-volume hyperintense-fraction test (as-is vs flip_y vs flip_x) and side-by-side crops."""
import json, numpy as np, h5py
from probe_common import *
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt

vols = (OUT / 'volumes.txt').read_text().split()
rows = [r for r in read_boxes() if r['file'] in vols and r['label'] in SMALL]
def box_rc(r, nr, nc, fx=False, fy=False):
    rr0, rr1, cc0, cc1 = r['y'], r['y'] + r['height'], r['x'], r['x'] + r['width']
    if fx: cc0, cc1 = nc - cc1, nc - cc0
    if fy: rr0, rr1 = nr - rr1, nr - rr0
    return max(0, rr0), min(nr, rr1), max(0, cc0), min(nc, cc1)
print(f"{'volume':38s} {'n':>4s} {'as_is':>7s} {'flip_y':>7s} {'flip_x':>7s}   (mean fraction of box pixels > 1.3 x slice WM median)")
tot = {'as_is': [], 'flip_y': [], 'flip_x': []}
for v in vols:
    with h5py.File(h5_path(v), 'r') as f:
        rss = f['reconstruction_rss'][()]
    seg, _ = load_seg(SEG_EXISTING / f'{v}_seg.nii.gz')
    res = {k: [] for k in tot}
    for r in [r for r in rows if r['file'] == v]:
        s = r['slice']; img = rss[s]; nr, nc = img.shape
        wm = np.isin(seg[:, :, s].T, [2, 41])
        if wm.sum() < 200: continue
        ref = np.median(img[wm])
        for k, kw in (('as_is', {}), ('flip_y', dict(fy=True)), ('flip_x', dict(fx=True))):
            r0, r1, c0, c1 = box_rc(r, nr, nc, **kw)
            if r1 > r0 and c1 > c0:
                res[k].append(float((img[r0:r1, c0:c1] > 1.3 * ref).mean()))
    line = f"{v:38s} {len(res['as_is']):4d}"
    for k in tot:
        m = float(np.mean(res[k])) if res[k] else float('nan'); tot[k].append(m); line += f" {m:7.3f}"
    print(line)
print('mean over volumes:', {k: round(float(np.nanmean(v)), 3) for k, v in tot.items()})
json.dump({k: v for k, v in tot.items()}, open(OUT / 'probe0c_hyperfrac.json', 'w'))

# crops: 12 boxes from 6 volumes, as-is vs flip_y
rng = np.random.default_rng(0)
pick = []
for v in vols[:6]:
    vr = [r for r in rows if r['file'] == v]
    pick += [vr[i] for i in rng.choice(len(vr), 2, replace=False)]
fig, axes = plt.subplots(4, 6, figsize=(15, 10.5))
for i, r in enumerate(pick):
    with h5py.File(h5_path(r['file']), 'r') as f:
        img = f['reconstruction_rss'][r['slice']]
    nr, nc = img.shape
    for j, (k, kw) in enumerate((('as_is', {}), ('flip_y', dict(fy=True)))):
        r0, r1, c0, c1 = box_rc(r, nr, nc, **kw)
        pad = 20; R0, R1, C0, C1 = max(0, r0 - pad), min(nr, r1 + pad), max(0, c0 - pad), min(nc, c1 + pad)
        ax = axes[(i // 6) * 2 + j, i % 6]
        ax.imshow(img[R0:R1, C0:C1], cmap='gray', vmin=0, vmax=np.percentile(img, 99.5))
        ax.add_patch(plt.Rectangle((c0 - C0, r0 - R0), c1 - c0, r1 - r0, fill=False, edgecolor='lime', linewidth=1.5))
        ax.set_title(f"{k} | {r['file'][-4:]} s{r['slice']}", fontsize=8); ax.axis('off')
plt.tight_layout(); plt.savefig(Path.home() / 'figs/anatobind_probe/probe0_crops_asis_vs_flipy.png', dpi=100)
print('saved crops')
