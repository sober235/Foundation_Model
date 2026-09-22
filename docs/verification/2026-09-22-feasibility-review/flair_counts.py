"""Reviewer re-count of fastMRI+ FLAIR lesions with the probe's own merge rule, plus power / agreement arithmetic.
Read-only on /data2. Mirrors docs/verification/2026-09-16-brain-probe/probe_common.py::read_boxes / merge_lesions.
"""
import csv, re, sys
from collections import Counter, defaultdict
import numpy as np

sys.path.insert(0, '/data0/congcong/code/Project_Doing/foundation_model')
from anatobind.data_engine.fastmri_knee import _iou  # same IoU as the probe

CSV = '/data2/congcong/data/FM_data/fastMRI_lh_brain_knee/Annotations/brain.csv'
SMALL = {'Nonspecific white matter lesion', 'Lacunar infarct'}

rows = []
for r in csv.DictReader(open(CSV)):
    if r['study_level'].strip() == 'Yes':
        continue
    try:
        rows.append(dict(file=r['file'], slice=int(r['slice']), x=int(r['x']), y=int(r['y']),
                         width=int(r['width']), height=int(r['height']), label=r['label'].strip()))
    except ValueError:
        continue
rows = [r for r in rows if r['width'] >= 3 and r['height'] >= 3]
flair_rows = [r for r in rows if 'AXFLAIR' in r['file']]
print(f'FLAIR box rows after >=3px filter: {len(flair_rows)}')

by = defaultdict(list)
for r in flair_rows:
    by[(r['file'], r['label'])].append(r)
lesions = []
for (file, label), group in sorted(by.items()):
    parent = list(range(len(group)))
    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]; i = parent[i]
        return i
    for i, a in enumerate(group):
        for j, b in enumerate(group):
            if j <= i or abs(a['slice'] - b['slice']) != 1:
                continue
            if _iou(a, b) >= 0.3:
                parent[find(i)] = find(j)
    comps = defaultdict(list)
    for i in range(len(group)):
        comps[find(i)].append(group[i])
    for members in comps.values():
        lesions.append(dict(file=file, label=label, n=len(members), small=label in SMALL,
                            w=max(m['width'] for m in members), h=max(m['height'] for m in members)))

print(f'FLAIR merged 3D lesions: {len(lesions)}')
lab = Counter(L['label'] for L in lesions)
print('per-label merged counts:')
for k, v in lab.most_common(): print(f'   {v:5d}  {k}')
small = [L for L in lesions if L['small']]
vols_small = Counter(L['file'] for L in small)
print(f'small lesions (NSWML + lacunar): {len(small)} in {len(vols_small)} volumes')
flair_files = sorted(set(r['file'] for r in flair_rows))
print(f'FLAIR volumes with any box: {len(flair_files)}')
m = np.array(sorted(vols_small.values(), reverse=True))
print(f'small lesions per volume (volumes with >=1): min {m.min()} q25 {np.percentile(m,25):.0f} median {np.median(m):.0f} '
      f'q75 {np.percentile(m,75):.0f} max {m.max()}; top-10 volumes hold {m[:10].sum()} ({m[:10].sum()/m.sum():.1%})')
m_eff = (m**2).sum() / m.sum()
print(f'mean lesions/volume (over 165) {m.mean():.2f}; cluster-size m_eff = sum(m^2)/sum(m) = {m_eff:.2f}')
single = sum(1 for L in small if L['n'] == 1)
print(f'single-slice small lesions (thickness <= 5 mm): {single} ({single/len(small):.1%})')
px = 0.6875
tiny = sum(1 for L in small if max(L['w'], L['h']) * px <= 6.0)
print(f'small lesions with max in-plane extent <= 6 mm (at 0.6875 mm/px, 200/201 assumption): {tiny} ({tiny/len(small):.1%})')
ser = lambda f: re.search(r'AXFLAIR_(\d+)', f).group(1)
hi = sum(1 for L in small if ser(L['file']) in ('200', '201'))
print(f'small lesions in full-res series 200/201: {hi}; in low-res variants: {len(small)-hi}')

# ---- 20 % patient hold-out simulation (one FLAIR volume == one patient, per REVIEW 09-16 §4) ----
all_flair_patients = flair_files  # 252 patients with boxes
counts = np.array([vols_small.get(f, 0) for f in all_flair_patients])
rng = np.random.default_rng(0)
n_hold = int(round(0.2 * len(all_flair_patients)))
sims = []
for _ in range(5000):
    idx = rng.choice(len(counts), n_hold, replace=False)
    sims.append((counts[idx].sum(), (counts[idx] > 0).sum()))
sims = np.array(sims)
print(f'20% hold-out = {n_hold} patients: small lesions p5/p50/p95 = {np.percentile(sims[:,0],5):.0f}/{np.percentile(sims[:,0],50):.0f}/{np.percentile(sims[:,0],95):.0f}; '
      f'patients with >=1 small lesion p5/p50/p95 = {np.percentile(sims[:,1],5):.0f}/{np.percentile(sims[:,1],50):.0f}/{np.percentile(sims[:,1],95):.0f}')

# ---- power arithmetic (paired McNemar, alpha 0.05 two-sided, power 0.8) ----
za, zb = 1.96, 0.8416
def n_simple(p_disc, d): return 7.84 * p_disc / d**2
def n_exact(p_disc, d): return (za * np.sqrt(p_disc) + zb * np.sqrt(p_disc - d**2))**2 / d**2
print('\nrequired lesions n (simple / exact McNemar), before design effect:')
for p in (0.05, 0.08, 0.10, 0.15, 0.20):
    print(f'  p_disc {p:.2f}: ' + '  '.join(f'd={d:.2f}: {n_simple(p,d):5.0f}/{n_exact(p,d):5.0f}' for d in (0.02, 0.03, 0.04, 0.05)))
print('design effect DE = 1 + (m_eff - 1) * ICC with m_eff = %.2f:' % m_eff)
for icc in (0.02, 0.05, 0.10, 0.20):
    print(f'  ICC {icc:.2f}: DE = {1 + (m_eff-1)*icc:.2f}')
print('minimum detectable d = sqrt(7.84 * p_disc / n_eff):')
for n in (260, 700, 1297):
    for de in (1.0, 1.3, 1.5, 2.0):
        ne = n / de
        print(f'  n {n:5d} DE {de:.1f} n_eff {ne:6.0f}: ' + '  '.join(f'p_disc {p:.2f}: d_min {np.sqrt(7.84*p/ne):.3f}' for p in (0.05, 0.10, 0.20)))
print('net rescue d = p_disc * (2q - 1); q = share of disagreements where the learned model is right:')
for p in (0.05, 0.10, 0.20):
    print(f'  p_disc {p:.2f}: ' + '  '.join(f'q={q:.2f}: d={p*(2*q-1):.3f}' for q in (0.55, 0.60, 0.65, 0.70, 0.80)))

# ---- agreement arithmetic with the probe's clean host distribution (probe2a_displacement.json all_structures) ----
hosts = {'WM': 614, 'cortex': 146, 'CSF': 19, 'caudate': 1}
tot = sum(hosts.values()); pi = np.array(list(hosts.values())) / tot
pe = (pi**2).sum(); K = len(pi)
pe_gwet = (pi * (1 - pi)).sum() / (K - 1)
print(f'\nhost prevalence from probe: ' + ', '.join(f'{k} {v/tot:.3f}' for k, v in hosts.items()))
for pa in (0.80, 0.85, 0.90):
    print(f'  raw {pa:.2f}: kappa {(pa-pe)/(1-pe):.2f}  Gwet AC1 {(pa-pe_gwet)/(1-pe_gwet):.2f}   (p_e kappa {pe:.3f}, p_e Gwet {pe_gwet:.3f})')
for n in (100, 150, 260, 1297):
    print(f'  n {n}: 95% CI half-width of raw agreement at p=0.8: {1.96*np.sqrt(0.8*0.2/n):.3f}; of p_disc=0.10: {1.96*np.sqrt(0.1*0.9/n):.3f}')

# ---- Level R reader hours ----
print('\nreader hours (two readers, per-lesion minutes 1.5 / 2.5 / 4; adjudication 25% of lesions x 3 min):')
for name, N in (('pilot 150', 150), ('hold-out ~260', 260), ('700 subset', 700), ('CV full 1297', 1297)):
    print(f'  {name:14s}: ' + '  '.join(f'{mnt} min -> {2*N*mnt/60:5.1f} h' for mnt in (1.5, 2.5, 4.0)) + f';  adjudication {0.25*N*3/60:.1f} h')
