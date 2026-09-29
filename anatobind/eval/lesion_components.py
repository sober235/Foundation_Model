"""Lesions as 26-connected components of a binary mask, with a physical volume floor (spec 2026-09-29 M7, §5).

A component under MIN_MM3 is kept but flagged "ignore": it never enters the denominator, and a detection matched to it
is neither a hit nor a false positive. The floor is a volume, not a voxel count, because the three datasets' voxels
differ eightfold (1 mm3 to 8 mm3)."""
import math

import numpy as np
from scipy import ndimage

from anatobind.eval.lesion_boxes import STRUCTURE

MIN_MM3 = 10.0
STRATA = ("<5", "5-10", ">=10")


def min_voxels_for(voxel_mm3, min_mm3=MIN_MM3):
    """Fewest voxels whose volume reaches min_mm3. The relative tolerance absorbs float32 headers: two headers of one
    grid (0.5 mm and 0.49999997 mm) give one floor."""
    return max(1, math.ceil(min_mm3 / float(voxel_mm3) * (1.0 - 1e-6)))


def equivalent_diameter_mm(mm3):
    return (6.0 * float(mm3) / math.pi) ** (1.0 / 3.0)


def size_stratum(mm3):
    d = equivalent_diameter_mm(mm3)
    return "<5" if d < 5.0 else ("5-10" if d < 10.0 else ">=10")


def components(mask):
    """(component map, number of components), 26-connectivity."""
    return ndimage.label(np.asarray(mask) > 0, structure=STRUCTURE)


def component_rows(comp, n, voxel_mm3, family, min_mm3=MIN_MM3):
    """One row per component: half-open box (x0, y0, z0, x1, y1, z1), voxel count, volume, and the ignore flag."""
    floor = min_voxels_for(voxel_mm3, min_mm3)
    sizes = np.bincount(comp.ravel(), minlength=n + 1)
    rows = []
    for k, sl in enumerate(ndimage.find_objects(comp), start=1):
        if sl is None:
            continue
        nv = int(sizes[k])
        rows.append({"component": k, "family": family,
                     "box": (sl[0].start, sl[1].start, sl[2].start, sl[0].stop, sl[1].stop, sl[2].stop),
                     "n_voxels": nv, "mm3": nv * float(voxel_mm3), "ignore": bool(nv < floor)})
    return rows


def component_mask(comp, row):
    """(slices of the row's box, boolean mask of the component inside the box)."""
    b = row["box"]
    sl = tuple(slice(b[i], b[i + 3]) for i in range(3))
    return sl, comp[sl] == row["component"]
