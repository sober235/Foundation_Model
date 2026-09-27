"""Synthetic brains, registries and tables for the relation tests (nothing here reads /data2)."""
import numpy as np

from anatobind.eval.geometry import (
    CLASS_NAMES, SLOT_FIELDS, class_maps, host_class_map, interface_margin, lesion_class_distances, member_rects,
)
from anatobind.eval.lookup import BRAIN_PARENCHYMA, BrainLookup
from anatobind.level_r.registry import distance_band
from anatobind.relation.table import (
    LESION_TYPES, PATCH_PX, PATCH_SLICES, SIDES, SLOTS, Table, slot_col,
)

SPACING = (0.5, 0.5, 5.0)


def synthetic_volume(seed=0, shape=(64, 64, 4)):
    """Brain block cols 8..56 rows 8..56: left WM (2) cols 8..32, right WM (41) cols 32..56, cortex (3 / 42) rows 48..56,
    a lateral ventricle (4) cols 20..24 rows 20..24, a thalamus (10) cols 28..32 rows 28..32. RSS = noise + brain offset."""
    rng = np.random.default_rng(seed)
    seg = np.zeros(shape, np.int16)
    seg[8:32, 8:56, :] = 2
    seg[32:56, 8:56, :] = 41
    seg[8:32, 48:56, :] = 3
    seg[32:56, 48:56, :] = 42
    seg[20:24, 20:24, :] = 4
    seg[28:32, 28:32, :] = 10
    rss = rng.normal(0, 1, shape).astype(np.float32)
    rss[seg > 0] += 10.0
    rss[np.isin(seg, (3, 42))] += 3.0
    return seg, SPACING, rss


def synthetic_registry(files, per_file, seed=0):
    """Single-slice 4 x 4 boxes inside the brain, registry rows shaped like Gate 0.5's lesions.csv (after load_registry),
    and the member boxes per lesion. Registry geometry is computed with the same functions the builder uses."""
    rng = np.random.default_rng(seed)
    rows, members_of = [], {}
    lid = 0
    for fi, f in enumerate(files):
        seg, sp, _ = synthetic_volume(seed)
        cm = host_class_map(seg)
        maps = class_maps(cm, sp)
        look = BrainLookup(seg, sp, BRAIN_PARENCHYMA)
        members_of[f] = {}
        for _ in range(per_file):
            x, y, s = int(rng.integers(10, 50)), int(rng.integers(10, 50)), int(rng.integers(0, seg.shape[2]))
            member = {"x": x, "width": 4, "y": y, "height": 4, "slice": s}
            rects = member_rects([member], seg.shape)
            _, d1, d2 = interface_margin(lesion_class_distances({c: maps[c][0] for c in maps}, rects))
            rows.append({"lesion_id": lid, "file": f, "patient_id": f"patient_{fi}", "series": "200", "stratum_series": "200_201",
                         "stratum_geometry": "inplane_0.69_slice_5", "label": "Nonspecific white matter lesion" if lid % 5 else "Lacunar infarct",
                         "z0": s, "z1": s, "n_slices": 1, "x0": x, "y0": y, "x1": x + 4, "y1": y + 4, "inplane_mm": 2.0,
                         "spacing_row_mm": sp[1], "spacing_col_mm": sp[0], "spacing_slice_mm": sp[2],
                         "host_lookup_all": str(look.host(rects)[0]), "host_lookup_parenchyma": str(look.host(rects)[0]),
                         "host_class_nearest": "white_matter", "d1_mm": float(d1), "d_interface_mm": float(d2),
                         "delta_d_mm": float(d2 - d1), "status": "ok", "band": distance_band(float(d2))})
            members_of[f][lid] = [member]
            lid += 1
    return rows, members_of


def synthetic_table(n_patients=6, per_patient=4, seed=0):
    """A Table with random but self-consistent geometry: c1 is the argmax-ioa slot, candidates are the slots within 15 mm,
    patches are random. Enough structure for Bgeo+ to fit C1 and for the CV / metrics / training tests."""
    rng = np.random.default_rng(seed)
    rows = []
    lid = 0
    for p in range(n_patients):
        for _ in range(per_patient):
            ioa = np.zeros(7)
            k = int(rng.integers(0, 2))                                    # WM or cortex carries the overlap
            ioa[k] = float(rng.uniform(0.5, 1.0))
            ioa[1 - k] = 1.0 - ioa[k] if rng.uniform() < 0.5 else 0.0
            dist = np.where(ioa > 0, 0.0, rng.uniform(1.0, 40.0, 7))
            dist[2] = min(dist[2], 10.0)                                   # thalamus is always a candidate
            side = SIDES[int(rng.integers(0, 3))]
            ltype = LESION_TYPES[int(rng.integers(0, 2))]
            d_int = float(rng.uniform(0.0, 8.0))
            row = {"lesion_id": lid, "file": f"file_{p}", "patient_id": f"patient_{p}", "fold": p % 5,
                   "stratum_geometry": "inplane_0.69_slice_5", "stratum_series": "200_201", "band": distance_band(d_int),
                   "is_3mm": False, "lesion_type": ltype, "n_slices": 1, "inplane_mm": 3.0, "x0": 10, "y0": 10, "x1": 14, "y1": 14,
                   "z0": 1, "z1": 1, "spacing_col_mm": 0.6875, "spacing_row_mm": 0.6875, "spacing_slice_mm": 5.0,
                   "slice_thickness_mm": 5.0, "extent_x_mm": 2.75, "extent_y_mm": 2.75, "extent_z_mm": 5.0, "volume_mm3": 37.8,
                   "centroid_col": 11.5, "centroid_row": 11.5, "centroid_slice": 1.0, "side": side, "row_third": "middle",
                   "slice_third": "middle", "coarse_location": f"{side}|middle|middle", "d1_mm": 0.0, "d_interface_mm": d_int,
                   "delta_d_mm": d_int, "dist_cortex_mm": float(dist[1]), "dist_ventricle_mm": float(rng.uniform(0, 30)),
                   "c1_class": SLOTS[int(np.argmax(ioa))], "c1_slot": int(np.argmax(ioa)), "c1_source": "overlap",
                   "c1_overlap": float(ioa.max()), "registry_lookup_class": SLOTS[int(np.argmax(ioa))]}
            for i, s in enumerate(SLOTS):
                vals = {"in_volume": True, "candidate": bool(dist[i] <= 15.0), "dx_mm": float(rng.normal(0, 5)),
                        "dy_mm": float(rng.normal(0, 5)), "dz_mm": 0.0, "centroid_distance_mm": float(dist[i] + 1.0),
                        "min_surface_mm": float(dist[i]), "signed_surface_mm": float(-2.0 if ioa[i] > 0 else dist[i]),
                        "ioa": float(ioa[i]), "soft_overlap": float(np.exp(-dist[i]))}
                for f in SLOT_FIELDS:
                    row[slot_col(s, f)] = vals[f]
            rows.append(row)
            lid += 1
    n = len(rows)
    patches = {"lesion_id": np.arange(n), "image": rng.normal(0, 1, (n, PATCH_SLICES, PATCH_PX, PATCH_PX)).astype(np.float16),
               "mask": np.zeros((n, PATCH_SLICES, PATCH_PX, PATCH_PX), bool), "classmap": np.zeros((n, PATCH_SLICES, PATCH_PX, PATCH_PX), np.int8)}
    patches["mask"][:, :, 20:28, 20:28] = True
    return Table(rows), patches
