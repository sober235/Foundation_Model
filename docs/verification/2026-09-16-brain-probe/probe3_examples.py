"""Step 3: show lesions whose lookup host changes under a degraded view (clean vs degraded, with SynthSeg contours)."""
import sys, numpy as np, nibabel as nib
from probe_common import *
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
view = sys.argv[1] if len(sys.argv) > 1 else 'us16'
vols = (OUT / 'volumes.txt').read_text().split()
lesions = [L for L in merge_lesions(read_boxes()) if L['file'] in vols and L['small']]
flipped = []
for v in vols:
    a, sp = load_seg(WORK / 'clean' / 'seg_native' / f'{v}_seg.nii.gz'); b, _ = load_seg(WORK / view / 'seg_native' / f'{v}_seg.nii.gz')
    la, lb = Lookup(a, sp, ALL_CAND), Lookup(b, sp, ALL_CAND)
    for L in [L for L in lesions if L['file'] == v]:
        r = lesion_rects(L, a.shape); ha, hb = host_rects(la, r)[0], host_rects(lb, r)[0]
        if ha != hb:
            flipped.append((L, r, ha, hb))
print(view, 'flipped small lesions:', len(flipped))
pick = flipped[:: max(1, len(flipped) // 6)][:6]
fig, axes = plt.subplots(2, max(1, len(pick)), figsize=(3.2 * max(1, len(pick)), 6.6), squeeze=False)
for i, (L, r, ha, hb) in enumerate(pick):
    v = L['file']; x0, x1, y0, y1, s = r[0]
    for j, vw in enumerate(('clean', view)):
        img = np.asarray(nib.load(str(WORK / vw / 'stage' / f'{v}.nii.gz')).dataobj)[:, :, s].T  # (row, col)
        seg = load_seg(WORK / vw / 'seg_native' / f'{v}_seg.nii.gz')[0][:, :, s].T
        pad = 28; R0, R1, C0, C1 = max(0, y0 - pad), min(img.shape[0], y1 + pad), max(0, x0 - pad), min(img.shape[1], x1 + pad)
        ax = axes[j, i]; ax.imshow(img[R0:R1, C0:C1], cmap='gray', vmin=0, vmax=np.percentile(img, 99.5))
        sub = seg[R0:R1, C0:C1]
        for labs, col in (([2, 41], 'cyan'), ([3, 42], 'magenta'), ([4, 43, 5, 44, 24], 'yellow'), ([10, 49, 11, 50, 12, 51, 13, 52], 'orange')):
            m = np.isin(sub, labs)
            if m.any() and not m.all(): ax.contour(m, levels=[0.5], colors=col, linewidths=0.8)
        ax.add_patch(plt.Rectangle((x0 - C0, y0 - R0), x1 - x0, y1 - y0, fill=False, edgecolor='lime', linewidth=1.5))
        host = ha if vw == 'clean' else hb
        ax.set_title(f"{vw}: host = {LABEL_NAMES.get(host, host)}\n{v[-4:]} s{s}", fontsize=8); ax.axis('off')
plt.suptitle(f'small lesions whose category-free lookup host changes on {view} (WM cyan, cortex magenta, ventricle/CSF yellow, deep gray orange)', fontsize=9)
plt.tight_layout(); out = Path.home() / f'figs/anatobind_probe/probe3_flips_{view}.png'; plt.savefig(out, dpi=110); print('saved', out)
