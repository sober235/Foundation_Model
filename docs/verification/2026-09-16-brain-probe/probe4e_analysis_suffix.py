"""Comparison B for a preprocessing suffix: prep(noisy) vs prep(clean), plus prep(clean) vs clean."""
import sys, json, numpy as np
from probe_common import *
suf = sys.argv[1]; noisy_views = sys.argv[2:] or ['noise_q2', 'noise_q3']
vols = (OUT / 'volumes.txt').read_text().split()
lesions = [L for L in merge_lesions(read_boxes()) if L['file'] in vols and L['small']]
views = ['clean', 'clean' + suf] + noisy_views + [v + suf for v in noisy_views]
segs = {vw: {v: load_seg(WORK / vw / 'seg_native' / f'{v}_seg.nii.gz') for v in vols if (WORK / vw / 'seg_native' / f'{v}_seg.nii.gz').exists()} for vw in views}
hosts = {rule: {vw: {} for vw in views} for rule in RULES}
for vw in views:
    for v, (seg, sp) in segs[vw].items():
        lk = {rule: Lookup(seg, sp, c) for rule, c in RULES.items()}
        for L in [L for L in lesions if L['file'] == v]:
            r = lesion_rects(L, seg.shape)
            for rule in RULES: hosts[rule][vw][L['lesion_id']] = host_rects(lk[rule], r)[0]
rng = np.random.default_rng(0)
def rate_ci(flags, vols_of, n=10000):
    flags, vols_of = np.asarray(flags, float), np.asarray(vols_of); uv = np.unique(vols_of); per = {u: flags[vols_of == u] for u in uv}; out = []
    for _ in range(n):
        pick = rng.choice(uv, len(uv), replace=True); out.append(np.concatenate([per[u] for u in pick]).mean())
    return float(flags.mean()), float(np.percentile(out, 2.5)), float(np.percentile(out, 97.5))
res = {}
for rule in RULES:
    print(f'=== rule {rule}, preprocessing "{suf}" ===')
    pairs = [(f'{nv} vs clean (raw)', nv, 'clean') for nv in noisy_views] + [(f'prep(clean) vs clean', 'clean' + suf, 'clean')] + [(f'B: prep({nv}) vs prep(clean)', nv + suf, 'clean' + suf) for nv in noisy_views]
    for label, test, ref in pairs:
        flags, vols_of = [], []
        for lid, h in hosts[rule][ref].items():
            if lid in hosts[rule][test]: flags.append(hosts[rule][test][lid] != h); vols_of.append(lid.split(':')[0])
        if not flags: print(f'{label:34s} (missing)'); continue
        m, lo, hi = rate_ci(flags, vols_of); res.setdefault(rule, {})[label] = dict(n=len(flags), flip=m, ci=[lo, hi])
        print(f'{label:34s} {len(flags):5d} {m:7.3f}   [{lo:.3f}, {hi:.3f}]  {"FIRES" if (m >= 0.05 and lo > 0) else "no"}')
groups = {'cerebral WM': [2, 41], 'cortex': [3, 42], 'deep gray': [10, 49, 11, 50, 12, 51, 13, 52]}
for label, test, ref in [('prep(clean) vs clean', 'clean' + suf, 'clean')] + [(f'prep({nv}) vs prep(clean)', nv + suf, 'clean' + suf) for nv in noisy_views]:
    dice = {g: [] for g in groups}; disp = []
    for v in vols:
        if v not in segs[test] or v not in segs[ref]: continue
        a, sp = segs[ref][v]; b, _ = segs[test][v]
        for g, labs in groups.items():
            ma, mb = np.isin(a, labs), np.isin(b, labs)
            if ma.sum() > 100: dice[g].append(2 * (ma & mb).sum() / (ma.sum() + mb.sum()))
        for lab in ALL_CAND:
            ma, mb = a == lab, b == lab
            if ma.sum() > 200 and mb.sum() > 200: disp.append(float(np.linalg.norm(np.array(np.nonzero(ma)).mean(1) * sp - np.array(np.nonzero(mb)).mean(1) * sp)))
    if disp: print(f"{label:26s} WM {np.mean(dice['cerebral WM']):.3f} cortex {np.mean(dice['cortex']):.3f} deep gray {np.mean(dice['deep gray']):.3f} | centroid {np.median(disp):.2f} / {np.percentile(disp, 90):.2f}")
json.dump(res, open(OUT / f'probe4e_analysis{suf}.json', 'w'), indent=1)
