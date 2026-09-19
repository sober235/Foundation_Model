"""Step 0b: decisive orientation test. FLAIR WM lesions are hyperintense, so under the right
box->image mapping the box interior is brighter than a surrounding ring. Tests 8 hypotheses."""
import json, numpy as np, h5py
from scipy import ndimage
from probe_common import *
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt

vols = (OUT / 'volumes.txt').read_text().split()
rows = [r for r in read_boxes() if r['file'] in vols and r['label'] in SMALL]
hyps = {}
for T in (False, True):
    for fx in (False, True):
        for fy in (False, True):
            hyps[f"{'T' if T else '-'}{'fx' if fx else '--'}{'fy' if fy else '--'}"] = (T, fx, fy)
ratios = {h: [] for h in hyps}
for v in vols:
    with h5py.File(h5_path(v), 'r') as f:
        rss = f['reconstruction_rss'][()]  # (slice, row, col)
    ns, nr, nc = rss.shape
    for r in [r for r in rows if r['file'] == v]:
        img = rss[r['slice']]
        for h, (T, fx, fy) in hyps.items():
            x0, x1, y0, y1 = r['x'], r['x'] + r['width'], r['y'], r['y'] + r['height']
            if T:  # x indexes rows, y indexes cols
                rr0, rr1, cc0, cc1 = x0, x1, y0, y1
            else:
                rr0, rr1, cc0, cc1 = y0, y1, x0, x1
            if fx:
                cc0, cc1 = nc - cc1, nc - cc0
            if fy:
                rr0, rr1, rr1 = nr - rr1, nr - rr0, nr - rr0
            rr0, rr1, cc0, cc1 = max(0, rr0), min(nr, rr1), max(0, cc0), min(nc, cc1)
            if rr1 <= rr0 or cc1 <= cc0:
                continue
            inner = np.zeros((nr, nc), bool); inner[rr0:rr1, cc0:cc1] = True
            ring = ndimage.binary_dilation(inner, iterations=6) & ~ndimage.binary_dilation(inner, iterations=2)
            # top-quartile of the box vs median of the ring: robust to partial-volume boxes
            ratios[h].append(float(np.percentile(img[inner], 75) / (np.median(img[ring]) + 1e-9)))
print(f"{'hyp':8s} {'n':>5s} {'median ratio':>13s} {'frac>1.2':>9s}")
summary = {}
for h, vals in ratios.items():
    a = np.array(vals); summary[h] = dict(n=int(a.size), median=float(np.median(a)), frac_gt_1p2=float((a > 1.2).mean()))
    print(f"{h:8s} {a.size:5d} {np.median(a):13.3f} {(a > 1.2).mean():9.3f}")
json.dump(summary, open(OUT / 'probe0b_contrast.json', 'w'), indent=1)

# overlays (as-is mapping) with existing SynthSeg contours, 2 volumes x 2 slices
fig, axes = plt.subplots(2, 2, figsize=(11, 11))
for i, v in enumerate(vols[2:4]):
    with h5py.File(h5_path(v), 'r') as f:
        rss = f['reconstruction_rss'][()]
    seg, _ = load_seg(SEG_EXISTING / f'{v}_seg.nii.gz')  # (col,row,slice)
    vrows = [r for r in rows if r['file'] == v]
    slices = sorted({r['slice'] for r in vrows}, key=lambda s: -sum(1 for r in vrows if r['slice'] == s))[:2]
    for j, s in enumerate(slices):
        ax = axes[i, j]; img = rss[s]
        ax.imshow(img, cmap='gray', vmin=0, vmax=np.percentile(img, 99.5))
        segs = seg[:, :, s].T  # -> (row, col)
        ax.contour(segs == 2, levels=[0.5], colors='cyan', linewidths=0.5); ax.contour(segs == 41, levels=[0.5], colors='cyan', linewidths=0.5)
        ax.contour(np.isin(segs, [4, 43]), levels=[0.5], colors='yellow', linewidths=0.5)
        ax.contour(np.isin(segs, [3, 42]), levels=[0.5], colors='magenta', linewidths=0.4)
        for r in vrows:
            if r['slice'] == s:
                ax.add_patch(plt.Rectangle((r['x'], r['y']), r['width'], r['height'], fill=False, edgecolor='lime', linewidth=1.2))
        ax.set_title(f'{v}\nslice {s}: boxes (lime) as-is; WM cyan, ventricle yellow, cortex magenta', fontsize=8); ax.axis('off')
plt.tight_layout(); plt.savefig(Path.home() / 'figs/anatobind_probe/probe0_overlay_asis.png', dpi=110)
print('saved overlay')
