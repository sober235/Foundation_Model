"""The Level R lesion registry is Gate 0.5's lesions.csv (docs/verification/2026-09-24/gate05/lesions.csv): 1297
fastMRI+ FLAIR small lesions in the RSS frame with patient_id, d_interface and the measured-geometry stratum.
lesion_id is the join key for orders, labels, statistics and sealing (spec R6). Readers never see this file."""
import csv
import hashlib
from pathlib import Path

REGISTRY = Path("docs/verification/2026-09-24/gate05/lesions.csv")
INT_FIELDS = ("lesion_id", "z0", "z1", "n_slices", "x0", "y0", "x1", "y1")
FLOAT_FIELDS = ("inplane_mm", "spacing_row_mm", "spacing_col_mm", "spacing_slice_mm", "d1_mm", "d_interface_mm", "delta_d_mm")
BANDS = ("0", "0-2", "2-4", ">4")          # v2.6 §25 item 9: d_interface 0 / (0, 2] / (2, 4] / > 4 mm


def distance_band(d_mm):
    if d_mm <= 0:
        return "0"
    if d_mm <= 2:
        return "0-2"
    if d_mm <= 4:
        return "2-4"
    return ">4"


def is_3mm(stratum_geometry):
    return stratum_geometry.endswith("_slice_3")


def load_registry(path=REGISTRY):
    with open(path, newline="") as fh:
        rows = list(csv.DictReader(fh))
    out = []
    for r in rows:
        if r["status"] != "ok":
            continue
        rec = dict(r)
        for k in INT_FIELDS:
            rec[k] = int(r[k])
        for k in FLOAT_FIELDS:
            rec[k] = float(r[k])
        rec["band"] = distance_band(rec["d_interface_mm"])
        out.append(rec)
    ids = [r["lesion_id"] for r in out]
    if len(ids) != len(set(ids)):
        raise ValueError(f"{path}: duplicate lesion_id")
    return out


def volume_code(stem):
    """Stable 8-hex code shown to readers instead of the h5 stem (spec §4)."""
    return hashlib.sha256(stem.encode()).hexdigest()[:8]


def lesion_code(lesion_id):
    return hashlib.sha256(f"lesion:{int(lesion_id)}".encode()).hexdigest()[:8]
