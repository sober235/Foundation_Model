"""Step 2b: after SynthSeg on all views: structure displacement and host-answer flips per view (vs the clean run)."""
import json, collections, numpy as np
from probe_common import *
vols = (OUT / 'volumes.txt').read_text().split()
lesions = [L for L in merge_lesions(read_boxes()) if L['file'] in vols and L['small']]
views = [v for v in VIEWS]
segs = {}
for view in views + ['existing']:
    d = SEG_EXISTING if view == 'existing' else WORK / view / 'seg_native'
    segs[view] = {v: load_seg(d / f'{v}_seg.nii.gz') for v in vols if (d / f'{v}_seg.nii.gz').exists()}
    print(view, 'labels available', len(segs[view]))
groups = {'cerebral WM': [2, 41], 'cortex': [3, 42], 'lateral ventricles': [4, 43], 'deep gray': [10, 49, 11, 50, 12, 51, 13, 52]}
rng = np.random.default_rng(0)
def boot_delta(a, b, vol_a, n=10000):
    a, b, vol_a = np.asarray(a, float), np.asarray(b, float), np.asarray(vol_a); uv = np.unique(vol_a); out = []
    per = {u: (a[vol_a == u], b[vol_a == u]) for u in uv}
    for _ in range(n):
        pick = rng.choice(uv, len(uv), replace=True)
        aa = np.concatenate([per[u][0] for u in pick]); bb = np.concatenate([per[u][1] for u in pick]); out.append(aa.mean() - bb.mean())
    return float(np.percentile(out, 2.5)), float(np.percentile(out, 97.5))
result = {}
# host answers per view
hosts = {rule: {view: {} for view in views + ['existing']} for rule in RULES}
for view in views + ['existing']:
    for v, (seg, sp) in segs[view].items():
        lk = {rule: Lookup(seg, sp, c) for rule, c in RULES.items()}
        for L in [L for L in lesions if L['file'] == v]:
            r = lesion_rects(L, seg.shape)
            for rule in RULES:
                hosts[rule][view][L['lesion_id']] = host_rects(lk[rule], r)[0]
for rule in RULES:
    print(f'\n=== rule {rule}: host flips of small lesions relative to the clean run ===')
    print(f"{'view':10s} {'n':>5s} {'flip rate':>10s} {'delta vs floor':>15s} {'95% CI (by volume)':>20s}  gate(+0.05)")
    ids_clean = hosts[rule]['clean']
    floor_flags, floor_vols = [], []
    for lid, h in ids_clean.items():
        if lid in hosts[rule]['existing']:
            floor_flags.append(hosts[rule]['existing'][lid] != h); floor_vols.append(lid.split(':')[0])
    floor = float(np.mean(floor_flags)) if floor_flags else float('nan')
    print(f"{'rerun-floor':10s} {len(floor_flags):5d} {floor:10.3f}")
    result[rule] = {'floor_flip_rate': floor, 'views': {}}
    for view in views[1:]:
        flags, vols_of, base = [], [], []
        for lid, h in ids_clean.items():
            if lid in hosts[rule][view]:
                flags.append(hosts[rule][view][lid] != h); vols_of.append(lid.split(':')[0])
                base.append(hosts[rule]['existing'].get(lid, h) != h)
        fr = float(np.mean(flags)); lo, hi = boot_delta(flags, base, vols_of)
        fires = (fr - floor) >= 0.05 and lo > 0
        result[rule]['views'][view] = dict(n=len(flags), flip_rate=fr, delta=fr - floor, ci=[lo, hi], gate=bool(fires))
        print(f'{view:10s} {len(flags):5d} {fr:10.3f} {fr - floor:15.3f}   [{lo:+.3f}, {hi:+.3f}]  {"FIRES" if fires else "no"}')
    # which structures do flips go to
    trans = collections.Counter()
    for lid, h in ids_clean.items():
        h2 = hosts[rule]['us16'].get(lid)
        if h2 is not None and h2 != h:
            trans[f'{LABEL_NAMES.get(h, h)} -> {LABEL_NAMES.get(h2, h2)}'] += 1
    print('  us16 transitions (top 8):', trans.most_common(8))
    result[rule]['us16_transitions'] = dict(trans.most_common(20))
# structure displacement per view
print('\n=== structure Dice vs clean and centroid displacement (mm) ===')
print(f"{'view':10s} " + ''.join(f'{g:>20s}' for g in groups) + f"{'centroid med/p90':>18s}")
result['structures'] = {}
for view in views[1:] + ['existing']:
    dice = {g: [] for g in groups}; disp = []
    for v in vols:
        if v not in segs[view] or v not in segs['clean']:
            continue
        a, sp = segs['clean'][v]; b, _ = segs[view][v]
        for g, labs in groups.items():
            ma, mb = np.isin(a, labs), np.isin(b, labs)
            if ma.sum() > 100: dice[g].append(2 * (ma & mb).sum() / (ma.sum() + mb.sum()))
        for lab in ALL_CAND:
            ma, mb = a == lab, b == lab
            if ma.sum() > 200 and mb.sum() > 200:
                ca = np.array(np.nonzero(ma)).mean(1) * sp; cb = np.array(np.nonzero(mb)).mean(1) * sp
                disp.append(float(np.linalg.norm(ca - cb)))
    row = {g: float(np.mean(dice[g])) for g in groups}; row['centroid_median_mm'] = float(np.median(disp)); row['centroid_p90_mm'] = float(np.percentile(disp, 90))
    result['structures'][view] = row
    print(f'{view:10s} ' + ''.join(f"{row[g]:20.3f}" for g in groups) + f"   {row['centroid_median_mm']:.2f} / {row['centroid_p90_mm']:.2f}")
json.dump(result, open(OUT / 'probe2b_views.json', 'w'), indent=1)
