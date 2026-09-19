"""Step 1: seven views per volume from raw k-space through the repo's single RSS operator; NIfTIs for SynthSeg."""
import json, zlib, numpy as np, h5py
from probe_common import *
from anatobind.data_engine.fastmri_knee import _view_seed

vols = (OUT / 'volumes.txt').read_text().split()
WORK.mkdir(parents=True, exist_ok=True)
nrmse = {}
def nr(a, b): return float(np.linalg.norm(a - b) / np.linalg.norm(b))
for v in vols:
    h5 = h5_path(v)
    with h5py.File(h5, 'r') as f:
        k = f['kspace'][()]
        rss_ref = f['reconstruction_rss'][()]
    geo = geometry(h5)
    seed = zlib.crc32(v.encode()) & 0xFFFF
    views = {}
    for view in VIEWS:
        vol = np.stack([reconstruct_rss(degrade(k[s], view, _view_seed(view, seed, s)), 320) for s in range(k.shape[0])])
        views[view] = vol
        write_view_nifti(vol, geo, WORK / view / 'stage' / f'{v}.nii.gz')
    clean = views['clean']
    nrmse[v] = {'clean_vs_h5rss': nr(clean, rss_ref), **{vw: nr(views[vw], clean) for vw in VIEWS if vw != 'clean'}}
    print(v, {kk: round(x, 4) for kk, x in nrmse[v].items()}, flush=True)
json.dump(nrmse, open(OUT / 'probe1_nrmse.json', 'w'), indent=1)
med = {kk: float(np.median([nrmse[v][kk] for v in vols])) for kk in nrmse[vols[0]]}
print('median NRMSE per view:', {kk: round(x, 4) for kk, x in med.items()})
