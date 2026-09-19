"""Step 4 (Stage 1.5 first half): training-free preprocessing control before SynthSeg.
Per slice: (1) non-central-chi noise-floor correction I_c = sqrt(max(I^2 - b^2, 0)), b^2 = mean(I^2) over the four
30x30 image corners (outside the head); (2) non-local means, sigma estimated from the corrected image.
Nothing here uses the known noise level or the clean image."""
import json, numpy as np, nibabel as nib
from concurrent.futures import ProcessPoolExecutor
from skimage.restoration import denoise_nl_means, estimate_sigma
from probe_common import *

vols = (OUT / 'volumes.txt').read_text().split()
VIEWS_IN = ['noise_q1', 'noise_q2', 'noise_q3']

def floor_correct(sl):
    corner = np.zeros_like(sl, bool); corner[:30, :30] = corner[-30:, :30] = corner[:30, -30:] = corner[-30:, -30:] = True
    b2 = float(np.mean(sl[corner] ** 2))
    return np.sqrt(np.maximum(sl ** 2 - b2, 0)).astype(np.float32), b2

def work(args):
    view, v = args
    img = nib.load(str(WORK / view / 'stage' / f'{v}.nii.gz')); arr = np.asarray(img.dataobj).astype(np.float32)
    clean = np.asarray(nib.load(str(WORK / 'clean' / 'stage' / f'{v}.nii.gz')).dataobj).astype(np.float32)
    seg, _ = load_seg(WORK / 'clean' / 'seg_native' / f'{v}_seg.nii.gz'); brain = seg > 0
    corr = np.empty_like(arr); out = np.empty_like(arr)
    for s in range(arr.shape[2]):
        xc, _ = floor_correct(arr[:, :, s]); corr[:, :, s] = xc
        scale = float(np.percentile(xc, 99)) or 1.0; y = xc / scale; sg = float(estimate_sigma(y))
        out[:, :, s] = denoise_nl_means(y, h=0.8 * sg, sigma=sg, fast_mode=True, patch_size=5, patch_distance=6) * scale
    od = WORK / f'{view}_dn' / 'stage'; od.mkdir(parents=True, exist_ok=True)
    nib.save(nib.Nifti1Image(out, img.affine, img.header), str(od / f'{v}.nii.gz'))
    nr = lambda a, b: float(np.linalg.norm(a - b) / np.linalg.norm(b))
    return view, v, dict(nrmse_noisy=nr(arr, clean), nrmse_floor=nr(corr, clean), nrmse_dn=nr(out, clean),
                         brain_nrmse_noisy=nr(arr[brain], clean[brain]), brain_nrmse_dn=nr(out[brain], clean[brain]))

res = {}
with ProcessPoolExecutor(8) as ex:
    for view, v, r in ex.map(work, [(view, v) for view in VIEWS_IN for v in vols]):
        res.setdefault(view, {})[v] = r
for view in VIEWS_IN:
    m = {k: float(np.median([r[k] for r in res[view].values()])) for k in next(iter(res[view].values()))}
    print(f"{view}: NRMSE noisy {m['nrmse_noisy']:.3f} -> floor-corrected {m['nrmse_floor']:.3f} -> +NLM {m['nrmse_dn']:.3f} | brain-only {m['brain_nrmse_noisy']:.3f} -> {m['brain_nrmse_dn']:.3f}")
json.dump(res, open(OUT / 'probe4_denoise_nrmse.json', 'w'), indent=1)
