"""The relation feature table (spec 2026-09-27 §4): one row per lesion, seven fixed host-class slots, the §11 geometry,
candidate flags and the pseudo label C1. Every arm reads this table and nothing else, which is what makes the
comparison fair by construction (P4). Patches live next to it in patches.npz."""
import csv
import json
from pathlib import Path

import numpy as np

from anatobind.eval.geometry import CLASS_NAMES, SLOT_FIELDS

SLOTS = CLASS_NAMES
SLOT_PREFIX = {"white_matter": "wm", "cortex": "cortex", "thalamus": "thalamus", "basal_ganglia": "bg",
               "brainstem": "brainstem", "cerebellum": "cerebellum", "other_deep_grey": "odg"}
N_SLOTS = len(SLOTS)
NONE_SLOT = N_SLOTS
N_OUT = N_SLOTS + 1
PAIR_FIELDS = ("dx_mm", "dy_mm", "dz_mm", "centroid_distance_mm", "signed_surface_mm", "min_surface_mm", "ioa", "soft_overlap")
LESION_GEOMETRY = ("extent_x_mm", "extent_y_mm", "extent_z_mm", "volume_mm3", "spacing_col_mm", "spacing_row_mm",
                   "spacing_slice_mm", "slice_thickness_mm")
BRAIN_EXTRA = ("d_interface_mm", "delta_d_mm", "dist_cortex_mm", "dist_ventricle_mm")
SIDES = ("image_left", "image_right", "midline")
LESION_TYPES = ("nonspecific_wm_lesion", "lacunar_infarct")
LESION_TYPE_OF_LABEL = {"Nonspecific white matter lesion": "nonspecific_wm_lesion", "Lacunar infarct": "lacunar_infarct"}
ROW_THIRDS = ("top", "middle", "bottom")
SLICE_THIRDS = ("inferior", "middle", "superior")
PATCH_PX, PATCH_MM, PIXEL_MM, PATCH_SLICES = 48, 36.0, 0.75, 3
TABLE_VERSION = "v1"

_INT = ("lesion_id", "fold", "n_slices", "x0", "y0", "x1", "y1", "z0", "z1", "c1_slot")
_FLOAT = ("inplane_mm", "spacing_col_mm", "spacing_row_mm", "spacing_slice_mm", "slice_thickness_mm", "extent_x_mm",
          "extent_y_mm", "extent_z_mm", "volume_mm3", "centroid_col", "centroid_row", "centroid_slice", "d1_mm",
          "d_interface_mm", "delta_d_mm", "dist_cortex_mm", "dist_ventricle_mm", "c1_overlap")
_BOOL = ("is_3mm",)
_STR = ("file", "patient_id", "stratum_geometry", "stratum_series", "band", "lesion_type", "side", "row_third",
        "slice_third", "coarse_location", "c1_class", "c1_source", "registry_lookup_class")
LESION_COLUMNS = ("lesion_id", "file", "patient_id", "fold", "stratum_geometry", "stratum_series", "band", "is_3mm",
                  "lesion_type", "n_slices", "inplane_mm", "x0", "y0", "x1", "y1", "z0", "z1", "spacing_col_mm",
                  "spacing_row_mm", "spacing_slice_mm", "slice_thickness_mm", "extent_x_mm", "extent_y_mm", "extent_z_mm",
                  "volume_mm3", "centroid_col", "centroid_row", "centroid_slice", "side", "row_third", "slice_third",
                  "coarse_location", "d1_mm", "d_interface_mm", "delta_d_mm", "dist_cortex_mm", "dist_ventricle_mm",
                  "c1_class", "c1_slot", "c1_source", "c1_overlap", "registry_lookup_class")


def slot_col(slot_name, field):
    return f"{SLOT_PREFIX[slot_name]}_{field}"


SLOT_COLUMNS = [slot_col(s, f) for s in SLOTS for f in SLOT_FIELDS]
COLUMN_TYPES = {**{c: int for c in _INT}, **{c: float for c in _FLOAT}, **{c: bool for c in _BOOL}, **{c: str for c in _STR}}
for _s in SLOTS:
    for _f in SLOT_FIELDS:
        COLUMN_TYPES[slot_col(_s, _f)] = bool if _f in ("in_volume", "candidate") else float
ALL_COLUMNS = list(LESION_COLUMNS) + SLOT_COLUMNS
assert set(ALL_COLUMNS) == set(COLUMN_TYPES)


def _parse(value, typ):
    if typ is bool:
        return value in ("True", "true", "1")
    return typ(value)


def read_rows(csv_path):
    with open(csv_path, newline="", encoding="utf-8") as fh:
        return [{c: _parse(r[c], COLUMN_TYPES[c]) for c in ALL_COLUMNS} for r in csv.DictReader(fh)]


class Table:
    def __init__(self, rows):
        self.rows = list(rows)
        self.lesion_id = np.array([r["lesion_id"] for r in self.rows], dtype=np.int64)
        if len(set(self.lesion_id.tolist())) != len(self.rows):
            raise ValueError("duplicate lesion_id in table")
        self.index = {int(l): i for i, l in enumerate(self.lesion_id)}

    def __len__(self):
        return len(self.rows)

    def column(self, name):
        return np.array([r[name] for r in self.rows])

    def slot_matrix(self, field):
        return np.array([[r[slot_col(s, field)] for s in SLOTS] for r in self.rows])

    def candidates(self):
        return self.slot_matrix("candidate").astype(bool)

    def c1_slot(self):
        return self.column("c1_slot").astype(int)

    def folds(self):
        return self.column("fold").astype(int)

    def patients(self):
        return self.column("patient_id")

    def slot_block(self):
        cand = self.slot_matrix("candidate").astype(np.float32)[:, :, None]
        pair = np.stack([self.slot_matrix(f) for f in PAIR_FIELDS], -1).astype(np.float32)
        return np.concatenate([cand, pair], -1)

    def lesion_block(self):
        geo = np.array([[r[c] for c in LESION_GEOMETRY + BRAIN_EXTRA] for r in self.rows], np.float32).reshape(len(self), 12)
        side = np.array([[r["side"] == s for s in SIDES] for r in self.rows], np.float32)
        typ = np.array([[r["lesion_type"] == t for t in LESION_TYPES] for r in self.rows], np.float32)
        return np.concatenate([geo, side, typ], 1)

    def subset(self, idx):
        return Table([self.rows[i] for i in np.asarray(idx)])


def features_flat(table):
    n = len(table)
    return np.concatenate([table.slot_block().reshape(n, N_SLOTS * 9), table.lesion_block()], 1)


def features_per_slot(table):
    lb = table.lesion_block()
    return np.concatenate([table.slot_block(), np.repeat(lb[:, None, :], N_SLOTS, 1)], -1)


def mask_to_candidates(probs, candidates):
    """Zero every non-candidate slot (spec P7); the none column is kept; rows are renormalised. A row with no mass left
    on its candidates and none becomes uniform over its candidates (or all none if it has no candidate)."""
    p = np.array(probs, dtype=np.float64)
    cand = np.asarray(candidates, bool)
    keep = np.concatenate([cand, np.ones((len(p), 1), bool)], 1)
    p[~keep] = 0.0
    s = p.sum(1, keepdims=True)
    empty = s[:, 0] <= 0
    if empty.any():
        p[empty] = np.concatenate([cand[empty], ~cand[empty].any(1, keepdims=True)], 1)
        s[empty] = p[empty].sum(1, keepdims=True)
    return p / s


def write_table(out_dir, rows, patches, manifest):
    out = Path(out_dir)
    if out.exists():
        raise FileExistsError(f"{out} exists; a relation table is never overwritten (spec §4.6)")
    out.mkdir(parents=True)
    with open(out / "table.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=ALL_COLUMNS, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)
    np.savez_compressed(out / "patches.npz", **patches)
    (out / "manifest.json").write_text(json.dumps(manifest, indent=1), encoding="utf-8")
    return out


def load_table(dir, with_patches=True):
    d = Path(dir)
    table = Table(read_rows(d / "table.csv"))
    patches = None
    if with_patches:
        with np.load(d / "patches.npz") as z:
            patches = {k: z[k] for k in z.files}
        if patches["lesion_id"].tolist() != table.lesion_id.tolist():
            raise ValueError("patches.npz and table.csv disagree on lesion order")
    return table, patches
