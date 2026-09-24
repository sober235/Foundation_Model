"""Model-free lesion geometry on a SynthSeg label map (v2.6 §4.5, §7.3): the distance from a lesion to each
candidate host class, the interface distance d_interface and the margin Δd that define the hard group H1.

Host classes follow v2.6 §3: cerebral white matter, cortex, thalamus, basal ganglia, brainstem, cerebellum and the
other deep grey structures (hippocampus, amygdala, ventral DC). Left and right are one class because the reader's
primary_host is side-agnostic. Ventricles and CSF are landmarks, not hosts (§25 item 4), so no interface is drawn
against them. Distances are Euclidean distance transforms in mm on the volume's own spacing; with 5 mm slices a
voxel one slice away is >= 5 mm, so t <= 4 mm is an in-plane criterion (§4.5).

Grid convention: label maps are (col, row, slice) like derived/synthseg/*/seg_native (fastmri.rss_h5_to_nifti);
lesion members are RSS-frame boxes (x = col0, y = row0 from the top, width, height, slice).
"""
import numpy as np
from scipy import ndimage

HOST_CLASSES = {
    "white_matter": (2, 41),
    "cortex": (3, 42),
    "thalamus": (10, 49),
    "basal_ganglia": (11, 50, 12, 51, 13, 52, 26, 58),
    "brainstem": (16,),
    "cerebellum": (7, 46, 8, 47),
    "other_deep_grey": (17, 53, 18, 54, 28, 60),
}
CLASS_NAMES = tuple(HOST_CLASSES)                 # class id = index + 1; 0 = not a host
LANDMARKS = (4, 43, 5, 44, 14, 15, 24)            # lateral / inferior-lateral / 3rd / 4th ventricles, CSF


def host_class_map(seg):
    """SynthSeg labels -> host class ids 1..7; ventricles, CSF, background and everything else -> 0."""
    out = np.zeros(np.shape(seg), np.int8)
    for i, labels in enumerate(HOST_CLASSES.values(), start=1):
        out[np.isin(seg, labels)] = i
    return out


def class_distance_maps(class_map, spacing):
    """{class id: distance in mm from every voxel to the nearest voxel of that class}, for the classes present."""
    return {int(c): ndimage.distance_transform_edt(class_map != c, sampling=spacing)
            for c in np.unique(class_map) if c > 0}


def member_rects(members, shape):
    """RSS-frame member boxes -> clipped (col0, col1, row0, row1, slice) rectangles on a (col, row, slice) grid;
    boxes that leave nothing on the grid are dropped."""
    nc, nr, ns = shape
    out = []
    for m in members:
        c0, c1 = max(0, m["x"]), min(nc, m["x"] + m["width"])
        r0, r1 = max(0, m["y"]), min(nr, m["y"] + m["height"])
        if c1 > c0 and r1 > r0 and 0 <= m["slice"] < ns:
            out.append((c0, c1, r0, r1, m["slice"]))
    return out


def lesion_class_distances(dist_maps, rects):
    """Minimum over the lesion's voxels of each class distance map (0 when the lesion touches the class)."""
    return {c: float(min(d[c0:c1, r0:r1, s].min() for c0, c1, r0, r1, s in rects)) for c, d in dist_maps.items()}


def interface_margin(dists):
    """(nearest class, d1, d2): d1 = distance to the nearest host class, d2 = to the second nearest.
    d_interface = d2 (the distance to the nearest interface with another class, within one voxel diagonal when
    the lesion lies inside one class; 0 when it straddles two) and Δd = d2 - d1. inf when only one class exists."""
    order = sorted(dists.items(), key=lambda kv: kv[1])
    c1, d1 = order[0]
    d2 = order[1][1] if len(order) > 1 else float("inf")
    return int(c1), d1, d2
