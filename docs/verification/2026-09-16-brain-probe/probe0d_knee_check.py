"""Side check: does the fastMRI+ up/down flip also apply to the knee labels used by leg 2?
Fat-sat PD: joint effusion and subchondral edema are the brightest tissue, so the right mapping puts boxes on bright pixels."""
import csv, collections, numpy as np, h5py
from pathlib import Path
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
FM = Path('/data2/congcong/data/FM_data')
rows = []
for r in csv.DictReader(open(FM / 'fastMRI_lh_brain_knee/Annotations/knee.csv')):
    if r['label'].strip() in ('Joint Effusion', 'Bone- Subchondral edema') and r['study_level'].strip() != 'Yes':
        try: rows.append(dict(file=r['file'], slice=int(r['slice']), x=int(r['x']), y=int(r['y']), w=int(r['width']), h=int(r['height']), label=r['label'].strip()))
        except ValueError: pass
def path(stem):
    for sp in ('multicoil_train', 'multicoil_val'):
        p = FM / f'fastMRI_lh_brain_knee/kspace/knee/{sp}/{stem}.h5'
        if p.exists(): return p
per = collections.Counter(r['file'] for r in rows)
vols = []
for f, n in per.most_common(200):
    p = path(f)
    if p is None: continue
    with h5py.File(p, 'r') as h:
        if str(h.attrs.get('acquisition')) == 'CORPDFS_FBK': vols.append(f)
    if len(vols) == 30: break
wins = collections.Counter(); ratios = {'as_is': [], 'flip_y': []}
pick = []
for v in vols:
    with h5py.File(path(v), 'r') as h: rss = h['reconstruction_rss'][()]
    res = {'as_is': [], 'flip_y': []}
    for r in [r for r in rows if r['file'] == v]:
        img = rss[r['slice']]; nr, nc = img.shape; ref = np.mean(img[nr//2-100:nr//2+100, nc//2-100:nc//2+100])
        for k in res:
            r0, r1 = (r['y'], r['y'] + r['h']) if k == 'as_is' else (nr - r['y'] - r['h'], nr - r['y'])
            c0, c1 = r['x'], r['x'] + r['w']
            r0, r1, c0, c1 = max(0, r0), min(nr, r1), max(0, c0), min(nc, c1)
            if r1 > r0 and c1 > c0: res[k].append(float(np.mean(img[r0:r1, c0:c1]) / ref))
        if len(pick) < 8 and r['label'] == 'Joint Effusion' and v not in [p['file'] for p in pick]: pick.append(r)
    m = {k: float(np.mean(x)) for k, x in res.items()}
    for k in ratios: ratios[k].append(m[k])
    wins['as_is' if m['as_is'] > m['flip_y'] else 'flip_y'] += 1
print('knee CORPDFS volumes tested:', len(vols), '| per-volume winner:', dict(wins))
print('mean box/centre intensity ratio: as_is %.3f  flip_y %.3f' % (np.mean(ratios['as_is']), np.mean(ratios['flip_y'])))
fig, axes = plt.subplots(2, 8, figsize=(18, 5))
for i, r in enumerate(pick):
    with h5py.File(path(r['file']), 'r') as h: img = h['reconstruction_rss'][r['slice']]
    nr, nc = img.shape
    for j, k in enumerate(('as_is', 'flip_y')):
        r0, r1 = (r['y'], r['y'] + r['h']) if k == 'as_is' else (nr - r['y'] - r['h'], nr - r['y'])
        c0, c1 = r['x'], r['x'] + r['w']
        ax = axes[j, i]; ax.imshow(img, cmap='gray', vmin=0, vmax=np.percentile(img, 99.5))
        ax.add_patch(plt.Rectangle((c0, r0), c1 - c0, r1 - r0, fill=False, edgecolor='lime', linewidth=1.2))
        ax.set_title(f"{k} effusion {r['file'][-4:]} s{r['slice']}", fontsize=7); ax.axis('off')
plt.tight_layout(); plt.savefig(Path.home() / 'figs/anatobind_probe/probe0_knee_effusion_asis_vs_flipy.png', dpi=100); print('saved knee figure')
