"""Same preprocessing (noise-floor correction + NLM) applied to the clean view, for comparison B."""
import numpy as np, nibabel as nib
from concurrent.futures import ProcessPoolExecutor
from skimage.restoration import denoise_nl_means, estimate_sigma
from probe_common import *
vols = (OUT / 'volumes.txt').read_text().split()
def floor_correct(sl):
    corner = np.zeros_like(sl, bool); corner[:30, :30] = corner[-30:, :30] = corner[:30, -30:] = corner[-30:, -30:] = True
    return np.sqrt(np.maximum(sl ** 2 - float(np.mean(sl[corner] ** 2)), 0)).astype(np.float32)
def work(v):
    img = nib.load(str(WORK / 'clean' / 'stage' / f'{v}.nii.gz')); arr = np.asarray(img.dataobj).astype(np.float32); out = np.empty_like(arr)
    for s in range(arr.shape[2]):
        xc = floor_correct(arr[:, :, s]); scale = float(np.percentile(xc, 99)) or 1.0; y = xc / scale; sg = float(estimate_sigma(y))
        out[:, :, s] = denoise_nl_means(y, h=0.8 * sg, sigma=sg, fast_mode=True, patch_size=5, patch_distance=6) * scale
    od = WORK / 'clean_dn' / 'stage'; od.mkdir(parents=True, exist_ok=True); nib.save(nib.Nifti1Image(out, img.affine, img.header), str(od / f'{v}.nii.gz')); return v
if __name__ == '__main__':
    with ProcessPoolExecutor(8) as ex: print('clean_dn done:', len(list(ex.map(work, vols))))
