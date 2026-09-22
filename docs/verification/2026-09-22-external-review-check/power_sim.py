"""Spike: (1) re-check the external review's cluster arithmetic; (2) the 'overall raw agreement only' loophole;
(3) Monte-Carlo power of full-1297 annotation vs two-phase stratified designs, on the real per-volume
small-lesion counts (same merge rule as docs/verification/2026-09-22-feasibility-review/flair_counts.py).
Throwaway unless promoted to docs/verification."""
import csv, re, sys, time
from collections import Counter, defaultdict
import numpy as np

sys.path.insert(0, '/data0/congcong/code/Project_Doing/foundation_model')
from anatobind.data_engine.fastmri_knee import _iou

CSV = '/data2/congcong/data/FM_data/fastMRI_lh_brain_knee/Annotations/brain.csv'
SMALL = {'Nonspecific white matter lesion', 'Lacunar infarct'}
rows = []
for r in csv.DictReader(open(CSV)):
    if r['study_level'].strip() == 'Yes' or 'AXFLAIR' not in r['file']:
        continue
    try:
        rows.append(dict(file=r['file'], slice=int(r['slice']), x=int(r['x']), y=int(r['y']),
                         width=int(r['width']), height=int(r['height']), label=r['label'].strip()))
    except ValueError:
        continue
rows = [r for r in rows if r['width'] >= 3 and r['height'] >= 3]
by = defaultdict(list)
for r in rows:
    by[(r['file'], r['label'])].append(r)
lesion_vol = []
for (file, label), group in sorted(by.items()):
    if label not in SMALL:
        continue
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
        comps[find(i)].append(i)
    lesion_vol += [file] * len(comps)
vols = sorted(set(lesion_vol))
vol_index = {v: i for i, v in enumerate(vols)}
vol = np.array([vol_index[v] for v in lesion_vol])
N, K = len(vol), len(vols)
m = np.bincount(vol, minlength=K)
m_eff = (m ** 2).sum() / m.sum()
print(f'[1] lesions N={N} in K={K} volumes; m_eff={m_eff:.2f}; DE(ICC .02/.05/.10)='
      f'{1+(m_eff-1)*.02:.2f}/{1+(m_eff-1)*.05:.2f}/{1+(m_eff-1)*.10:.2f}; N/DE(.05)={N/(1+(m_eff-1)*.05):.0f}')

# ---------------- [2] overall-raw-only loophole ----------------
rng = np.random.default_rng(1)
prev = {'WM': 0.787, 'cortex': 0.187, 'other': 0.026}
n = 200_000
truth = rng.choice(list(prev), size=n, p=list(prev.values()))
# reader A labels truth; reader B: agrees on WM always, on cortex only half the time (else WM), on other half the time
A = truth.copy()
B = truth.copy()
flip = rng.random(n) < 0.5
B[(truth == 'cortex') & flip] = 'WM'
B[(truth == 'other') & flip] = 'WM'
raw = (A == B).mean()
cats = ['WM', 'cortex', 'other']
pA = np.array([(A == c).mean() for c in cats]); pB = np.array([(B == c).mean() for c in cats])
pe = (pA * pB).sum(); kappa = (raw - pe) / (1 - pe)
pe_g = (((pA + pB) / 2) * (1 - (pA + pB) / 2)).sum() / (len(cats) - 1); ac1 = (raw - pe_g) / (1 - pe_g)
pa_c = 2 * ((A == 'cortex') & (B == 'cortex')).sum() / ((A == 'cortex').sum() + (B == 'cortex').sum())
h1 = (truth != 'WM') | (rng.random(n) < 0.05 / 0.787 * 0.2)   # H1 ~ all non-WM plus a few WM
raw_h1 = (A[h1] == B[h1]).mean()
print(f'[2] scenario: perfect agreement on WM, 50% on cortex/other -> overall raw {raw:.3f} (passes 0.80), '
      f'kappa {kappa:.2f}, AC1 {ac1:.2f}, cortex positive agreement {pa_c:.2f}, raw within H1 ({h1.mean():.0%} of lesions) {raw_h1:.2f}')

# ---------------- [3] Monte-Carlo power: designs on the real cluster structure ----------------
def icc_anova(z, g, K):
    n_i = np.bincount(g, minlength=K).astype(float); keep = n_i > 0
    s_i = np.bincount(g, weights=z, minlength=K); ss_i = np.bincount(g, weights=z * z, minlength=K)
    Nn = n_i.sum(); k = keep.sum(); gm = s_i.sum() / Nn
    mean_i = np.where(keep, s_i / np.maximum(n_i, 1), 0)
    ssb = (n_i * (mean_i - gm) ** 2)[keep].sum(); ssw = (ss_i - n_i * mean_i ** 2)[keep].sum()
    msb = ssb / (k - 1); msw = ssw / (Nn - k)
    n0 = (Nn - (n_i ** 2).sum() / Nn) / (k - 1)
    return (msb - msw) / (msb + (n0 - 1) * msw)

def run(design, pH1, pd1, pd2, q, kq, n_sims=400, n_boot=300, seed=0, kp=10.0):
    rng = np.random.default_rng(seed)
    hits, iccs, n_annot = 0, [], []
    for s in range(n_sims):
        h1 = rng.random(N) < pH1
        # patient-level heterogeneity that leaves the marginal effect size p*(2q-1) unchanged:
        # discordance rate p_i ~ Beta(mean p_h, conc kp); model-win share q_i ~ Beta(mean q, conc kq), independent
        p_vol1 = rng.beta(pd1 * kp, (1 - pd1) * kp, K); p_vol2 = rng.beta(pd2 * kp, (1 - pd2) * kp, K)
        q_vol = np.full(K, q) if np.isinf(kq) else rng.beta(q * kq, (1 - q) * kq, K)
        pdisc = np.where(h1, p_vol1[vol], p_vol2[vol]); qq = q_vol[vol]
        u = rng.random(N)
        z = np.where(u < pdisc * qq, 1.0, np.where(u < pdisc, -1.0, 0.0))
        P1 = h1.mean(); P2 = 1 - P1
        if design == 'A_full':
            ann = np.ones(N, bool)
        elif design.startswith('B_H1all_H2'):
            n2 = int(design.split('_H2')[1])
            ann = h1.copy()
            h2_idx = np.flatnonzero(~h1)
            ann[rng.choice(h2_idx, size=min(n2, len(h2_idx)), replace=False)] = True
        elif design.startswith('D_srs'):
            n_s = int(design.split('srs')[1])
            ann = np.zeros(N, bool); ann[rng.choice(N, size=n_s, replace=False)] = True
        n_annot.append(ann.sum())
        # per-volume sums for the cluster bootstrap
        s1 = np.bincount(vol, weights=z * (ann & h1), minlength=K); c1 = np.bincount(vol, weights=(ann & h1), minlength=K)
        s2 = np.bincount(vol, weights=z * (ann & ~h1), minlength=K); c2 = np.bincount(vol, weights=(ann & ~h1), minlength=K)
        idx = rng.integers(0, K, size=(n_boot, K))
        S1 = s1[idx].sum(1); C1 = c1[idx].sum(1); S2 = s2[idx].sum(1); C2 = c2[idx].sum(1)
        if design == 'A_full' or design.startswith('D_srs'):
            theta_b = (S1 + S2) / np.maximum(C1 + C2, 1)
        else:   # stratified standardization with population weights from the full frame
            theta_b = P1 * S1 / np.maximum(C1, 1) + P2 * S2 / np.maximum(C2, 1)
        lo = np.percentile(theta_b, 2.5)
        hits += lo > 0
        if design == 'A_full':
            iccs.append(icc_anova(z, vol, K))
    d_true = pH1 * pd1 * (2 * q - 1) + (1 - pH1) * pd2 * (2 * q - 1)
    return hits / n_sims, d_true, (np.mean(iccs) if iccs else float('nan')), int(np.mean(n_annot))

designs = ['A_full', 'B_H1all_H2300', 'B_H1all_H2600', 'D_srs700']
print('\n[3] power (P[CI_lower>0], alpha .05 two-sided, patient cluster bootstrap), 400 sims x 300 boots; '
      'hours = 2 readers x N_annotated x 2.5 min')
print(f'{"pd_H1":>6} {"pd_H2":>6} {"q":>4} {"kq":>5} {"ICC_A":>5} {"d_true":>6} | ' + ' | '.join(f'{d:>14}' for d in designs))
t0 = time.time()
for pd1, pd2 in ((0.25, 0.03), (0.35, 0.05), (0.50, 0.08)):
    for q in (0.65, 0.70):
        for kq in (np.inf, 8.0, 2.0):
            cells = []
            for d in designs:
                pw, d_true, icc, na = run(d, 0.20, pd1, pd2, q, kq, seed=7)
                cells.append((pw, na))
                if d == 'A_full':
                    icc_a = icc
            print(f'{pd1:6.2f} {pd2:6.2f} {q:4.2f} {kq:5.1f} {icc_a:5.3f} {d_true:6.3f} | ' +
                  ' | '.join(f'{pw:4.2f} n={na:4d} {2*na*2.5/60:4.0f}h' for pw, na in cells))
print(f'(sim time {time.time()-t0:.0f}s)')

# analytic SE comparison, no clustering, for the p_disc .35/.05 q .65 case
pH1, pd1, pd2 = 0.2, 0.35, 0.05
v1, v2 = pd1, pd2                        # Var(Z) ~ p_disc when d is small
n1 = pH1 * N; n2 = (1 - pH1) * N
se_full = np.sqrt((pH1 * v1 + (1 - pH1) * v2) / N)
for n2s in (300, 600, n2):
    se_strat = np.sqrt(pH1 ** 2 * v1 / n1 + (1 - pH1) ** 2 * v2 / n2s)
    print(f'[3b] analytic SE (no clustering): full 1297 {se_full:.4f}; H1-all + H2 {n2s:.0f}: {se_strat:.4f} '
          f'(ratio {se_strat/se_full:.2f}); Neyman share of H2 would be {(1-pH1)*np.sqrt(v2)/((1-pH1)*np.sqrt(v2)+pH1*np.sqrt(v1)):.0%}')
