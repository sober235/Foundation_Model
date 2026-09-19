"""Stage 1.5 control, variant 2: noise-floor correction only (no NLM), applied to clean, q2 and q3."""
import numpy as np, nibabel as nib
from concurrent.futures import ProcessPoolExecutor
from probe_common import *
vols = (OUT / 'volumes.txt').read_text().split()
def floor_correct(sl):
    corner = np.zeros_like(sl, bool); corner[:30, :30] = corner[-30:, :30] = corner[:30, -30:] = corner[-30:, -30:] = True
    return np.sqrt(np.maximum(sl ** 2 - float(np.mean(sl[corner] ** 2)), 0)).astype(np.float32)
def work(args):
    view, v = args
    img = nib.load(str(WORK / view / 'stage' / f'{v}.nii.gz')); arr = np.asarray(img.dataobj).astype(np.float32)
    out = np.stack([floor_correct(arr[:, :, s]) for s in range(arr.shape[2])], axis=2)
    od = WORK / f'{view}_fl' / 'stage'; od.mkdir(parents=True, exist_ok=True); nib.save(nib.Nifti1Image(out, img.affine, img.header), str(od / f'{v}.nii.gz')); return view
if __name__ == '__main__':
    with ProcessPoolExecutor(8) as ex: print('floor-only done:', len(list(ex.map(work, [(vw, v) for vw in ('clean', 'noise_q2', 'noise_q3') for v in vols]))))
