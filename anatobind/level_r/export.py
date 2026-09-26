"""Level R export (spec §4): per volume a little-endian uint16 array + JSON geometry, per lesion its per-slice boxes.
Boxes are recomputed along Gate 0.5's exact path (read_fastmri_plus_rows -> rows_to_rss_frame -> merge_boxes_3d) and
asserted equal to the registry so lesion_id lines up. Nothing written here carries a label string, a distance, a
stratum, a series or a patient id: those stay in the registry, which readers never see."""
import json
from pathlib import Path

import h5py
import numpy as np

from anatobind.data_engine.fastmri import MIN_BOX_SIDE, merge_boxes_3d, rows_to_rss_frame
from anatobind.data_engine.fastmri_knee import volume_geometry
from anatobind.level_r.registry import lesion_code, volume_code

SMALL_LABELS = ("Nonspecific white matter lesion", "Lacunar infarct")     # as scripts/brain_frame.py
U16_MAX = 65535
P_SCALE = 99.9
WINDOW_PERCENTILES = (1.0, 99.5)


def small_lesion_rows(rows):
    """The Gate 0.5 filter on fastMRI+ rows: FLAIR, the two small-lesion labels, both sides >= MIN_BOX_SIDE."""
    return [r for r in rows if "AXFLAIR" in r["file"] and r["label"] in SMALL_LABELS
            and r["width"] >= MIN_BOX_SIDE and r["height"] >= MIN_BOX_SIDE]


def merged_lesions(rows_of_file, n_rows):
    return merge_boxes_3d(rows_to_rss_frame(rows_of_file, n_rows), "label")


def boxes_by_slice(members):
    """{slice: [[row0, row1, col0, col1], ...]} in the RSS frame (rows from the top, half-open)."""
    out = {}
    for m in members:
        out.setdefault(str(m["slice"]), []).append([m["y"], m["y"] + m["height"], m["x"], m["x"] + m["width"]])
    return out


def _key(d):
    return (d["label"], d["z0"], d["z1"], d["x0"], d["y0"], d["x1"], d["y1"])


def match_registry(registry_rows, lesions):
    """Registry rows of one file <-> merged lesions of that file, one-to-one on (label, z0, z1, x0, y0, x1, y1)."""
    by_key = {}
    for L in lesions:
        by_key.setdefault(_key(L), []).append(L)
    out = {}
    for r in registry_rows:
        hits = by_key.get(_key(r), [])
        if len(hits) != 1:
            raise ValueError(f"{r['file']} lesion {r['lesion_id']}: {len(hits)} merged lesions match {_key(r)}; "
                             f"registry and CSV disagree, refusing to export")
        out[r["lesion_id"]] = hits[0]
    return out


def to_u16(rss):
    """float RSS volume -> (uint16 volume, info): p99.9 -> 65535, clipped; default window [p1, p99.5] in u16 units."""
    v = np.asarray(rss, np.float32)
    scale = float(np.percentile(v, P_SCALE))
    if not scale > 0:
        raise ValueError("volume has no positive intensities")
    u = np.clip(np.rint(v / scale * U16_MAX), 0, U16_MAX).astype("<u2")
    lo, hi = (int(np.clip(np.rint(np.percentile(v, q) / scale * U16_MAX), 0, U16_MAX)) for q in WINDOW_PERCENTILES)
    return u, {"scale_p999": scale, "window": [lo, hi]}


def write_volume(out_dir, code, u16, geometry, window):
    """volumes/<code>.u16 (little-endian, slices x rows x cols) and volumes/<code>.json."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / f"{code}.u16").write_bytes(np.ascontiguousarray(u16, dtype="<u2").tobytes())
    meta = {"shape": [int(s) for s in u16.shape], "spacing_slice_mm": geometry["spacing_slice_mm"],
            "spacing_row_mm": geometry["spacing_row_mm"], "spacing_col_mm": geometry["spacing_col_mm"], "window": list(window)}
    (out_dir / f"{code}.json").write_text(json.dumps(meta))
    return meta


def read_volume(out_dir, code):
    meta = json.loads((Path(out_dir) / f"{code}.json").read_text())
    u = np.frombuffer((Path(out_dir) / f"{code}.u16").read_bytes(), dtype="<u2").reshape(meta["shape"])
    return u, meta


def lesion_record(registry_row, lesion):
    return {"lesion_id": registry_row["lesion_id"], "code": lesion_code(registry_row["lesion_id"]),
            "volume_code": volume_code(registry_row["file"]), "z0": lesion["z0"], "z1": lesion["z1"],
            "boxes": boxes_by_slice(lesion["members"])}


def export_all(registry, csv_rows, out, h5_of, log=print):
    """registry: load_registry rows; csv_rows: read_fastmri_plus_rows(brain.csv); h5_of: stem -> h5 path."""
    out = Path(out)
    by_file_reg, by_file_csv = {}, {}
    for r in registry:
        by_file_reg.setdefault(r["file"], []).append(r)
    for r in small_lesion_rows(csv_rows):
        by_file_csv.setdefault(r["file"], []).append(r)
    records = []
    for i, f in enumerate(sorted(by_file_reg)):
        path = h5_of(f)
        g = volume_geometry(path)
        with h5py.File(path) as h:
            rss = h["reconstruction_rss"][()]
        matched = match_registry(by_file_reg[f], merged_lesions(by_file_csv.get(f, []), g["n_rows"]))
        u16, info = to_u16(rss)
        write_volume(out / "volumes", volume_code(f), u16, g, info["window"])
        records += [lesion_record(r, matched[r["lesion_id"]]) for r in by_file_reg[f]]
        log(f"{i + 1}/{len(by_file_reg)} {volume_code(f)}: {len(by_file_reg[f])} lesions, shape {u16.shape}")
    records.sort(key=lambda r: r["lesion_id"])
    out.mkdir(parents=True, exist_ok=True)
    (out / "lesions.json").write_text(json.dumps(records))
    return records
