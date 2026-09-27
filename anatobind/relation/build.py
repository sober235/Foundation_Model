"""Build the relation feature table (spec 2026-09-27 §4): per-lesion rows from SynthSeg geometry, image patches from
the RSS, the class-level pseudo label C1, the §4.7 checks and the manifest. Volumes come through an injected loader so
tests run on synthetic arrays."""
import numpy as np
from scipy import ndimage

from anatobind.relation.table import PATCH_PX, PATCH_SLICES, PIXEL_MM


def zscore_volume(rss, seg):
    """FLAIR z-scored over the brain voxels (seg > 0) of its own volume (spec §4.4)."""
    v = np.asarray(rss, np.float32)
    brain = np.asarray(seg) > 0
    mu, sd = float(v[brain].mean()), float(v[brain].std())
    if not sd > 0:
        raise ValueError("brain intensities are constant; refusing to z-score")
    return ((v - mu) / sd).astype(np.float32)


def lesion_patch(zvol, classmap, rects, spacing, centroid):
    """(image, mask, classmap) patches of PATCH_PX x PATCH_PX at PIXEL_MM centred on the lesion centroid, PATCH_SLICES
    slices around the centre slice with edge slices replicated. Axes: (slice, col, row). Image is bilinear,
    mask and classmap nearest-neighbour; outside the volume is zero."""
    nc, nr, ns = zvol.shape
    zc = int(np.clip(np.rint(centroid[2]), 0, ns - 1))
    u = (np.arange(PATCH_PX) - (PATCH_PX - 1) / 2) * PIXEL_MM
    cc, rr = np.meshgrid(centroid[0] + u / spacing[0], centroid[1] + u / spacing[1], indexing="ij")
    coords = np.stack([cc, rr])
    img = np.zeros((PATCH_SLICES, PATCH_PX, PATCH_PX), np.float32)
    msk = np.zeros((PATCH_SLICES, PATCH_PX, PATCH_PX), bool)
    cmp = np.zeros((PATCH_SLICES, PATCH_PX, PATCH_PX), np.int8)
    half = PATCH_SLICES // 2
    for k, dz in enumerate(range(-half, half + 1)):
        z = int(np.clip(zc + dz, 0, ns - 1))
        img[k] = ndimage.map_coordinates(zvol[:, :, z], coords, order=1, mode="constant", cval=0.0)
        plane = np.zeros((nc, nr), np.uint8)
        for c0, c1, r0, r1, s in rects:
            if s == z:
                plane[c0:c1, r0:r1] = 1
        msk[k] = ndimage.map_coordinates(plane, coords, order=0, mode="constant", cval=0) > 0
        cmp[k] = ndimage.map_coordinates(classmap[:, :, z].astype(np.int8), coords, order=0, mode="constant", cval=0)
    return img.astype(np.float16), msk, cmp
