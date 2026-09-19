"""Step 0: pick 24 volumes, then check the box->grid orientation on the existing clean SynthSeg labels."""
import json, collections
import numpy as np
from probe_common import *

rows = read_boxes()
per = collections.Counter(r['file'] for r in rows if r['label'] in SMALL)
cands = [f for f, n in per.most_common() if f.split('_')[2] in ('AXFLAIR',) and f.split('_')[3] in ('200', '201') and n >= 12]
vols = cands[:24]
OUT.mkdir(parents=True, exist_ok=True)
(OUT / 'volumes.txt').write_text('\n'.join(vols) + '\n')
print('selected', len(vols), 'volumes; small boxes', sum(per[v] for v in vols), 'range', min(per[v] for v in vols), max(per[v] for v in vols))

lesions = [L for L in merge_lesions(rows) if L['file'] in vols]
print('lesions in selected volumes:', len(lesions), 'small:', sum(L['small'] for L in lesions))

# orientation hypotheses on the existing clean labels
hyps = {'as_is': dict(), 'flip_y': dict(flip_y=True), 'flip_x': dict(flip_x=True), 'flip_xy': dict(flip_x=True, flip_y=True)}
tot = {h: collections.Counter() for h in hyps}
for v in vols:
    seg, sp = load_seg(SEG_EXISTING / f'{v}_seg.nii.gz')
    for L in lesions:
        if L['file'] != v or not L['small']:
            continue
        for h, kw in hyps.items():
            m = region_mask(L, seg.shape, **kw)
            vals, cnt = np.unique(seg[m], return_counts=True)
            for a, b in zip(vals, cnt):
                tot[h][int(a)] += int(b)
groups = {'WM': {2, 41}, 'cortex': {3, 42}, 'ventricle/CSF': NON_PARENCHYMA, 'deep gray': {10, 49, 11, 50, 12, 51, 13, 52, 17, 53, 18, 54, 26, 58, 28, 60}, 'background': {0}}
print('\nfraction of small-lesion box voxels by tissue group, per orientation hypothesis:')
print(f"{'hyp':10s}" + ''.join(f'{g:>15s}' for g in groups))
res = {}
for h in hyps:
    n = sum(tot[h].values())
    fr = {g: sum(tot[h][k] for k in ks) / n for g, ks in groups.items()}
    res[h] = fr
    print(f'{h:10s}' + ''.join(f'{fr[g]:15.3f}' for g in groups))
json.dump(dict(volumes=vols, n_lesions=len(lesions), n_small=sum(L['small'] for L in lesions), orientation=res), open(OUT / 'probe0_orientation.json', 'w'), indent=1)
