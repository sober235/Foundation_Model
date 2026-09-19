"""Step 4b: does training-free preprocessing (noise-floor correction + NLM) remove the geometry conflicts?
A: SynthSeg(prep(noisy)) vs SynthSeg(clean)      -- deployment against the raw clean reference
B: SynthSeg(prep(noisy)) vs SynthSeg(prep(clean)) -- one consistent pipeline"""
import json, collections, numpy as np
from probe_common import *
vols = (OUT / 'volumes.txt').read_text().split()
lesions = [L for L in merge_lesions(read_boxes()) if L['file'] in vols and L['small']]
views = ['clean', 'clean_dn', 'noise_q1', 'noise_q2', 'noise_q3', 'noise_q1_dn', 'noise_q2_dn', 'noise_q3_dn']
segs = {vw: {v: load_seg(WORK / vw / 'seg_native' / f'{v}_seg.nii.gz') for v in vols if (WORK / vw / 'seg_native' / f'{v}_seg.nii.gz').exists()} for vw in views}
for vw in views: print(vw, len(segs[vw]))
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
result = {}
for rule in RULES:
    print(f'\n=== rule {rule} ===')
    print(f"{'comparison':34s} {'n':>5s} {'flip':>7s} {'95% CI':>17s}  gate(+0.05)")
    result[rule] = {}
    for label, test, ref in (('noise_q1 vs clean (raw, from probe)', 'noise_q1', 'clean'), ('noise_q2 vs clean (raw)', 'noise_q2', 'clean'), ('noise_q3 vs clean (raw)', 'noise_q3', 'clean'),
                             ('A: prep(q1) vs clean', 'noise_q1_dn', 'clean'), ('A: prep(q2) vs clean', 'noise_q2_dn', 'clean'), ('A: prep(q3) vs clean', 'noise_q3_dn', 'clean'),
                             ('B: prep(clean) vs clean (prep cost)', 'clean_dn', 'clean'),
                             ('B: prep(q1) vs prep(clean)', 'noise_q1_dn', 'clean_dn'), ('B: prep(q2) vs prep(clean)', 'noise_q2_dn', 'clean_dn'), ('B: prep(q3) vs prep(clean)', 'noise_q3_dn', 'clean_dn')):
        flags, vols_of = [], []
        for lid, h in hosts[rule][ref].items():
            if lid in hosts[rule][test]: flags.append(hosts[rule][test][lid] != h); vols_of.append(lid.split(':')[0])
        if not flags: print(f'{label:34s} (no labels yet)'); continue
        m, lo, hi = rate_ci(flags, vols_of); fires = m >= 0.05 and lo > 0
        result[rule][label] = dict(n=len(flags), flip=m, ci=[lo, hi], gate=bool(fires))
        print(f'{label:34s} {len(flags):5d} {m:7.3f}   [{lo:.3f}, {hi:.3f}]  {"FIRES" if fires else "no"}')
groups = {'cerebral WM': [2, 41], 'cortex': [3, 42], 'deep gray': [10, 49, 11, 50, 12, 51, 13, 52]}
print('\n=== structure Dice vs reference and centroid displacement (mm) ===')
result['structures'] = {}
for label, test, ref in (('q3 vs clean', 'noise_q3', 'clean'), ('prep(q3) vs clean', 'noise_q3_dn', 'clean'), ('prep(q3) vs prep(clean)', 'noise_q3_dn', 'clean_dn'), ('prep(clean) vs clean', 'clean_dn', 'clean'), ('prep(q2) vs prep(clean)', 'noise_q2_dn', 'clean_dn')):
    dice = {g: [] for g in groups}; disp = []
    for v in vols:
        if v not in segs[test] or v not in segs[ref]: continue
        a, sp = segs[ref][v]; b, _ = segs[test][v]
        for g, labs in groups.items():
            ma, mb = np.isin(a, labs), np.isin(b, labs)
            if ma.sum() > 100: dice[g].append(2 * (ma & mb).sum() / (ma.sum() + mb.sum()))
        for lab in ALL_CAND:
            ma, mb = a == lab, b == lab
            if ma.sum() > 200 and mb.sum() > 200:
                disp.append(float(np.linalg.norm(np.array(np.nonzero(ma)).mean(1) * sp - np.array(np.nonzero(mb)).mean(1) * sp)))
    if not disp: continue
    row = {g: float(np.mean(dice[g])) for g in groups}; row['centroid_median_mm'] = float(np.median(disp)); row['centroid_p90_mm'] = float(np.percentile(disp, 90)); result['structures'][label] = row
    print(f"{label:26s} WM {row['cerebral WM']:.3f} cortex {row['cortex']:.3f} deep gray {row['deep gray']:.3f} | centroid {row['centroid_median_mm']:.2f} / {row['centroid_p90_mm']:.2f}")
json.dump(result, open(OUT / 'probe4b_analysis.json', 'w'), indent=1)
