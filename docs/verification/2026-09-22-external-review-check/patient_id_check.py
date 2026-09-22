"""Spike: do fastMRI brain h5 files carry a patient_id attribute, and how many patients do the 252 FLAIR
volumes with fastMRI+ boxes map to? Read-only (h5 attrs only)."""
import csv, os, sys
from collections import Counter, defaultdict
import h5py

ROOT = '/data2/congcong/data/FM_data/fastMRI_lh_brain_knee'
CSV = f'{ROOT}/Annotations/brain.csv'
DIRS = [f'{ROOT}/kspace/brain/multicoil_train', f'{ROOT}/kspace/brain/multicoil_val']

box_files, small_files = set(), set()
for r in csv.DictReader(open(CSV)):
    if 'AXFLAIR' not in r['file']:
        continue
    if r['study_level'].strip() == 'Yes':
        continue
    try:
        w, h = int(r['width']), int(r['height'])
    except ValueError:
        continue
    box_files.add(r['file'])
    if r['label'].strip() in ('Nonspecific white matter lesion', 'Lacunar infarct') and w >= 3 and h >= 3:
        small_files.add(r['file'])
print(f'FLAIR files with >=1 box row: {len(box_files)}; with >=1 small-lesion box: {len(small_files)}')

def locate(stem):
    for d in DIRS:
        p = os.path.join(d, stem + '.h5')
        if os.path.exists(p):
            return p, os.path.basename(d)
    return None, None

attr_keys = Counter()
pid_of, split_of, missing = {}, {}, []
for stem in sorted(box_files):
    p, split = locate(stem)
    if p is None:
        missing.append(stem); continue
    with h5py.File(p, 'r') as f:
        attr_keys.update(f.attrs.keys())
        pid = f.attrs.get('patient_id', None)
        pid_of[stem] = None if pid is None else (pid.decode() if isinstance(pid, bytes) else str(pid))
        split_of[stem] = split
print(f'h5 located: {len(pid_of)}; missing: {len(missing)} {missing[:3]}')
print(f'attr keys seen: {dict(attr_keys)}')
have = {k: v for k, v in pid_of.items() if v}
print(f'files with a patient_id attr: {len(have)}/{len(pid_of)}')
if have:
    per_pid = defaultdict(list)
    for stem, pid in have.items():
        per_pid[pid].append(stem)
    sizes = Counter(len(v) for v in per_pid.values())
    print(f'unique patient_id among box FLAIR files: {len(per_pid)}; files-per-patient histogram: {dict(sorted(sizes.items()))}')
    multi = {pid: v for pid, v in per_pid.items() if len(v) > 1}
    for pid, v in list(multi.items())[:5]:
        print(f'  patient {pid[:12]}...: {v}')
    cross = [pid for pid, v in per_pid.items() if len({split_of[s] for s in v}) > 1]
    print(f'patients with volumes in both multicoil_train and multicoil_val: {len(cross)}')
    small_p = defaultdict(list)
    for stem in small_files:
        if stem in have:
            small_p[have[stem]].append(stem)
    print(f'small-lesion FLAIR volumes: {len(small_files)} -> unique patients: {len(small_p)}; '
          f'multi-volume patients among them: {sum(1 for v in small_p.values() if len(v) > 1)}')
    sample_pid = next(iter(have.values()))
    print(f'example patient_id value (format check): {sample_pid!r}')
