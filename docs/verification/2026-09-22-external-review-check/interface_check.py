"""Spike: is 'distance to the nearest anatomical interface' (external review) the same quantity as
'distance to the nearest other candidate structure' (cold-start review), and is it cheap to compute on a real
SynthSeg label map? Read-only; one clean FLAIR volume from the repo's derived SynthSeg outputs."""
import sys, time
import numpy as np, nibabel as nib
from scipy.ndimage import distance_transform_edt, generate_binary_structure, binary_dilation

p = sys.argv[1] if len(sys.argv) > 1 else \
    '/data2/congcong/data/FM_data/derived/synthseg/fastmri_brain/seg_native/file_brain_AXFLAIR_200_6002425_seg.nii.gz'
img = nib.load(p)
seg = np.asarray(img.dataobj).astype(np.int16)
sp = tuple(float(z) for z in img.header.get_zooms()[:3])
print(f'volume {p.split("/")[-1]} shape {seg.shape} spacing {sp}')
# SynthSeg (FreeSurfer aseg) candidate anatomy labels; background 0 and non-brain excluded on purpose
CAND = [2, 41, 3, 42, 4, 43, 5, 44, 7, 46, 8, 47, 10, 49, 11, 50, 12, 51, 13, 52, 16, 17, 53, 18, 54, 24, 26, 58, 28, 60]
lab = np.where(np.isin(seg, CAND), seg, 0).astype(np.int16)
labels = [int(l) for l in np.unique(lab) if l > 0]
print(f'candidate labels present: {len(labels)}; brain voxels {int((lab>0).sum())}')

t0 = time.time()
# (a) cold-start definition: distance from a voxel to the nearest voxel of a DIFFERENT candidate label
d_other = np.full(lab.shape, np.inf, dtype=np.float32)
for L in labels:
    other = (lab != L) & (lab > 0)
    dt = distance_transform_edt(~other, sampling=sp)
    sel = lab == L
    d_other[sel] = dt[sel]
t_a = time.time() - t0
# (b) external-review definition: distance to the nearest interface voxel (a candidate voxel with a 26-neighbour of a
# different candidate label); computed without naming any host
t0 = time.time()
st26 = generate_binary_structure(3, 3)
iface = np.zeros(lab.shape, bool)
for L in labels:
    mask = lab == L
    touch = binary_dilation(mask, structure=st26) & ~mask & (lab > 0)   # other-label voxels touching L
    iface |= touch
    iface |= binary_dilation(touch, structure=st26) & mask               # L voxels touching those
d_iface = distance_transform_edt(~iface, sampling=sp)
t_b = time.time() - t0
brain = lab > 0
diff = np.abs(d_other[brain] - d_iface[brain])
vox_diag = float(np.sqrt(sum(s * s for s in sp)))
print(f'(a) nearest-other-structure distance: {t_a:.1f}s; (b) nearest-interface distance: {t_b:.1f}s')
print(f'|d_other - d_iface| over brain voxels: median {np.median(diff):.3f} mm, p95 {np.percentile(diff,95):.3f} mm, '
      f'max {diff.max():.3f} mm; one voxel diagonal = {vox_diag:.2f} mm; share within one in-plane voxel ({sp[0]:.4f} mm): '
      f'{(diff <= sp[0] + 1e-6).mean():.1%}; within one diagonal: {(diff <= vox_diag + 1e-6).mean():.1%}')
wm = np.isin(lab, [2, 41])
for t in (2.0, 3.0, 5.0):
    print(f'WM voxels with d_iface <= {t:.0f} mm: {(d_iface[wm] <= t).mean():.1%}   (voxel-level, not lesion-level)')
print(f'extrapolation: {252 * (t_a + t_b) / 60:.1f} CPU-minutes for 252 FLAIR volumes (single thread)')
