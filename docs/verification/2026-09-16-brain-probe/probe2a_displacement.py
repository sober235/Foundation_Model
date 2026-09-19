"""Step 2a: how far must a real small lesion move before the category-free lookup changes its host? (clean labels)"""
import json, collections, numpy as np
from probe_common import *
vols = (OUT / 'volumes.txt').read_text().split()
lesions = [L for L in merge_lesions(read_boxes()) if L['file'] in vols and L['small']]
shifts = [0.5, 1, 2, 3, 5, 8]; angles = np.linspace(0, 2 * np.pi, 16, endpoint=False)
flips = {rule: {d: [] for d in shifts} for rule in RULES}; vol_of = {rule: {d: [] for d in shifts} for rule in RULES}
realized = {d: [] for d in shifts}; slice_flips = {rule: [] for rule in RULES}
host0 = {rule: collections.Counter() for rule in RULES}; frac0 = {rule: [] for rule in RULES}
for v in vols:
    seg, sp = load_seg(SEG_EXISTING / f'{v}_seg.nii.gz')
    lk = {rule: Lookup(seg, sp, c) for rule, c in RULES.items()}
    for L in [L for L in lesions if L['file'] == v]:
        base = lesion_rects(L, seg.shape)
        h0 = {}
        for rule in RULES:
            h, fr = host_rects(lk[rule], base); h0[rule] = h; host0[rule][LABEL_NAMES.get(h, str(h))] += 1; frac0[rule].append(fr)
        for d in shifts:
            for th in angles:
                dx = int(round(d * np.cos(th) / sp[0])); dy = int(round(d * np.sin(th) / sp[1]))
                realized[d].append(float(np.hypot(dx * sp[0], dy * sp[1])))
                r = lesion_rects(L, seg.shape, dx=dx, dy=dy)
                for rule in RULES:
                    flips[rule][d].append(host_rects(lk[rule], r)[0] != h0[rule]); vol_of[rule][d].append(v)
        for ds in (-1, 1):
            r = lesion_rects(L, seg.shape, ds=ds)
            if r:
                for rule in RULES:
                    slice_flips[rule].append(host_rects(lk[rule], r)[0] != h0[rule])
rng = np.random.default_rng(0)
def boot(flags, vols_of, n=2000):
    flags = np.asarray(flags, float); vols_of = np.asarray(vols_of); uv = np.unique(vols_of)
    per = {u: flags[vols_of == u] for u in uv}; out = []
    for _ in range(n):
        pick = rng.choice(uv, len(uv), replace=True); out.append(np.concatenate([per[u] for u in pick]).mean())
    return float(np.percentile(out, 2.5)), float(np.percentile(out, 97.5))
summary = {'n_small_lesions': len(lesions), 'n_volumes': len(vols), 'realized_shift_mm': {str(d): float(np.mean(realized[d])) for d in shifts}}
print(f'small lesions {len(lesions)} in {len(vols)} volumes')
for rule in RULES:
    print(f'\nrule = {rule}: clean host distribution (top 6):', host0[rule].most_common(6))
    print(f'  median overlap fraction with host: {np.median(frac0[rule]):.2f}')
    summary[rule] = {'host_distribution': dict(host0[rule]), 'median_overlap_fraction': float(np.median(frac0[rule])), 'flip_rate': {}}
    print(f"  {'shift mm':>9s} {'realized':>9s} {'flip rate':>10s} {'95% CI (by volume)':>20s}")
    for d in shifts:
        fr = float(np.mean(flips[rule][d])); lo, hi = boot(flips[rule][d], vol_of[rule][d])
        summary[rule]['flip_rate'][str(d)] = dict(rate=fr, ci=[lo, hi], realized_mm=float(np.mean(realized[d])))
        print(f'  {d:9.1f} {np.mean(realized[d]):9.2f} {fr:10.3f}   [{lo:.3f}, {hi:.3f}]')
    sf = float(np.mean(slice_flips[rule])); summary[rule]['one_slice_shift_flip_rate'] = sf
    print(f'  one-slice (5 mm through-plane) shift: flip rate {sf:.3f} (n={len(slice_flips[rule])})')
json.dump(summary, open(OUT / 'probe2a_displacement.json', 'w'), indent=1)
