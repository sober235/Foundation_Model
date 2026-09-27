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


# --- relation baselines (spec 2026-09-27 §4.2–4.3): per-slot geometry, side, landmarks ---------------------------
CANDIDATE_MM = 15.0        # §10.1: host classes within 15 mm of the lesion surface are candidates
DIST_CAP_MM = 30.0         # every distance is capped here; absent classes carry the cap
SOFT_SIGMA_MM = 1.0
MIDLINE_SHARE = 0.4        # both families >= 40 % of the host voxels -> midline
LATERAL_VENTRICLES = (4, 43, 5, 44)
LEFT_LABELS = (2, 3, 10, 11, 12, 13, 26, 7, 8, 17, 18, 28)
RIGHT_LABELS = (41, 42, 49, 50, 51, 52, 58, 46, 47, 53, 54, 60)
SLOT_FIELDS = ("in_volume", "candidate", "dx_mm", "dy_mm", "dz_mm", "centroid_distance_mm", "min_surface_mm",
               "signed_surface_mm", "ioa", "soft_overlap")


def lesion_mask(rects):
    """Union of clipped rects -> (bbox (c0, c1, r0, r1, s0, s1), bool mask over that bbox). Overlapping members count once."""
    c0, c1 = min(r[0] for r in rects), max(r[1] for r in rects)
    r0, r1 = min(r[2] for r in rects), max(r[3] for r in rects)
    s0, s1 = min(r[4] for r in rects), max(r[4] for r in rects) + 1
    m = np.zeros((c1 - c0, r1 - r0, s1 - s0), bool)
    for a, b, c, d, s in rects:
        m[a - c0:b - c0, c - r0:d - r0, s - s0] = True
    return (c0, c1, r0, r1, s0, s1), m


def lesion_centroid(rects):
    """Mean (col, row, slice) voxel coordinate of the lesion's voxels (voxel centres, float)."""
    (c0, _, r0, _, s0, _), m = lesion_mask(rects)
    idx = np.argwhere(m).astype(float)
    return idx.mean(0) + np.array([c0, r0, s0], float)


def class_maps(class_map, spacing):
    """{class id: (distance to the class in mm, nearest-voxel index (3, ...), distance to the class boundary from inside)}
    for every host class present. Distances are EDTs on the volume spacing. The outside distance stays float64: it is
    the same EDT as class_distance_maps, so d1 / d_interface / Δd recomputed from it equal the Gate 0.5 registry
    exactly (the build checks this to 1e-6)."""
    out = {}
    for c in np.unique(class_map):
        if c <= 0:
            continue
        member = class_map == c
        dist, idx = ndimage.distance_transform_edt(~member, sampling=spacing, return_indices=True)
        inside = ndimage.distance_transform_edt(member, sampling=spacing)
        out[int(c)] = (dist, idx.astype(np.int32), inside.astype(np.float32))
    return out


def _absent():
    return {"in_volume": False, "candidate": False, "dx_mm": 0.0, "dy_mm": 0.0, "dz_mm": 0.0,
            "centroid_distance_mm": DIST_CAP_MM, "min_surface_mm": DIST_CAP_MM, "signed_surface_mm": DIST_CAP_MM,
            "ioa": 0.0, "soft_overlap": 0.0}


def slot_features(class_map, maps, rects, spacing):
    """The ten spec §4.3 quantities for every host class id 1..len(CLASS_NAMES). dx/dy/dz point from the centroid voxel
    to the class voxel nearest to it (mm); signed_surface is -(max inside depth) when the lesion overlaps
    the class, else the minimum surface distance. If no class is within CANDIDATE_MM the nearest one is the candidate."""
    (c0, c1, r0, r1, s0, s1), m = lesion_mask(rects)
    cls = class_map[c0:c1, r0:r1, s0:s1][m]
    centroid = lesion_centroid(rects)
    cv = np.minimum(np.rint(centroid).astype(int), np.array(class_map.shape) - 1)   # the centroid voxel
    sp = np.asarray(spacing, float)
    out = {}
    for c in range(1, len(CLASS_NAMES) + 1):
        if c not in maps:
            out[c] = _absent()
            continue
        dist, idx, inside = maps[c]
        d = dist[c0:c1, r0:r1, s0:s1][m]
        min_surface = min(float(d.min()), DIST_CAP_MM)
        ioa = float((cls == c).mean())
        soft = float(np.exp(-d / SOFT_SIGMA_MM).mean())
        signed = -float(inside[c0:c1, r0:r1, s0:s1][m][cls == c].max()) if ioa > 0 else min_surface
        near = idx[:, cv[0], cv[1], cv[2]].astype(float)
        delta = (near - cv) * sp                          # from the centroid voxel to the class voxel nearest to it
        out[c] = {"in_volume": True, "candidate": bool(min_surface <= CANDIDATE_MM),
                  "dx_mm": float(delta[0]), "dy_mm": float(delta[1]), "dz_mm": float(delta[2]),
                  "centroid_distance_mm": min(float(np.linalg.norm(delta)), DIST_CAP_MM),
                  "min_surface_mm": min_surface, "signed_surface_mm": signed, "ioa": ioa, "soft_overlap": soft}
    if not any(v["candidate"] for v in out.values()):
        nearest = min((c for c in out if out[c]["in_volume"]), key=lambda c: out[c]["min_surface_mm"])
        out[nearest]["candidate"] = True
    return out


def family_sides(seg):
    """Which image side (smaller / larger column index) each SynthSeg label family occupies in this volume (P8)."""
    left_cols = np.nonzero(np.isin(seg, LEFT_LABELS))[0]
    right_cols = np.nonzero(np.isin(seg, RIGHT_LABELS))[0]
    if not left_cols.size or not right_cols.size:
        raise ValueError("both label families are needed to orient the volume")
    left_first = left_cols.mean() < right_cols.mean()
    return {"left": "image_left" if left_first else "image_right", "right": "image_right" if left_first else "image_left"}


def side_of(seg, rects, families):
    """image_left / image_right / midline by the family majority of the lesion's host voxels; no host voxel -> the family
    of the nearest host voxel to the lesion centroid."""
    (c0, c1, r0, r1, s0, s1), m = lesion_mask(rects)
    vals = seg[c0:c1, r0:r1, s0:s1][m]
    left, right = int(np.isin(vals, LEFT_LABELS).sum()), int(np.isin(vals, RIGHT_LABELS).sum())
    if left + right == 0:
        host = np.isin(seg, LEFT_LABELS + RIGHT_LABELS)
        _, idx = ndimage.distance_transform_edt(~host, return_indices=True)
        cv = np.rint(lesion_centroid(rects)).astype(int)
        lab = int(seg[tuple(idx[:, cv[0], cv[1], cv[2]])])
        return families["left"] if lab in LEFT_LABELS else families["right"]
    if min(left, right) / (left + right) >= MIDLINE_SHARE:
        return "midline"
    return families["left"] if left > right else families["right"]


def landmark_map(seg, spacing, labels=LATERAL_VENTRICLES):
    """Distance in mm to the nearest voxel of ``labels`` (None when the volume has none of them)."""
    member = np.isin(seg, labels)
    if not member.any():
        return None
    return ndimage.distance_transform_edt(~member, sampling=spacing).astype(np.float32)


def min_in_lesion(dist_map, rects):
    if dist_map is None:
        return DIST_CAP_MM
    (c0, c1, r0, r1, s0, s1), m = lesion_mask(rects)
    return min(float(dist_map[c0:c1, r0:r1, s0:s1][m].min()), DIST_CAP_MM)


def third(index, lo, hi, names):
    """Which third of [lo, hi] (inclusive) ``index`` falls in; a degenerate range is the middle third."""
    if hi <= lo:
        return names[1]
    t = (index - lo) / (hi - lo)
    return names[0] if t < 1 / 3 else names[1] if t < 2 / 3 else names[2]
