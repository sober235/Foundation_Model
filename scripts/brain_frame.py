#!/usr/bin/env python
# scripts/brain_frame.py
"""Gate 0.5 (v2.6 §4.5): geometry of every fastMRI+ FLAIR small lesion (nonspecific white-matter lesion, lacunar
infarct; >= 3 px; the leg 2 merge rule) on the clean SynthSeg parcellation, before any reader sees an image.
Per lesion: the category-free lookup host (33 labels, with and without ventricles/CSF as candidates), the nearest
host class, d1, d_interface (= d2) and Δd (anatobind.eval.geometry), in-plane extent, slice count, series and
measured-geometry strata. Then the share of the hard group at t = 2/3/4/5 mm and the frozen t.

Strata measured on 2026-09-24 over the 165 volumes: 200/201/205/209/210 are 320x320 at 0.6875 mm; the 0.8594 mm
volumes are (part of) 202, 203 and 206; seven volumes have 3 mm slices. The series-number split of v2.6 §7.3
(200/201 vs the rest) is reported alongside the measured one.

  PYTHONNOUSERSITE=1 PYTHONPATH=. nice -n 19 ~/anaconda3/envs/nvgen/bin/python scripts/brain_frame.py \
      --out docs/verification/2026-09-24/gate05
"""
import argparse
import csv
import json
import re
import sys
from collections import Counter
from pathlib import Path

import nibabel as nib
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from anatobind.data_engine.fastmri import MIN_BOX_SIDE, merge_boxes_3d, read_fastmri_plus_rows, rows_to_rss_frame  # noqa: E402
from anatobind.data_engine.fastmri_knee import volume_geometry  # noqa: E402
from anatobind.eval.geometry import (  # noqa: E402
    CLASS_NAMES, class_distance_maps, host_class_map, interface_margin, lesion_class_distances, member_rects,
)
from anatobind.eval.lookup import BRAIN_ALL, BRAIN_PARENCHYMA, BrainLookup  # noqa: E402

FM = Path("/data2/congcong/data/FM_data")
CSV = FM / "fastMRI_lh_brain_knee/Annotations/brain.csv"
KROOT = FM / "fastMRI_lh_brain_knee/kspace/brain"
SEG_ROOT = FM / "derived/synthseg/fastmri_brain/seg_native"
SMALL_LABELS = ("Nonspecific white matter lesion", "Lacunar infarct")
T_GRID = (2, 3, 4, 5)
MIN_SHARE = 0.15
FIELDS = ["lesion_id", "file", "patient_id", "series", "stratum_series", "stratum_geometry", "label", "z0", "z1", "n_slices",
          "x0", "y0", "x1", "y1", "inplane_mm", "spacing_row_mm", "spacing_col_mm", "spacing_slice_mm",
          "host_lookup_all", "host_lookup_parenchyma", "host_class_nearest", "d1_mm", "d_interface_mm", "delta_d_mm", "status"]


def series_of(file):
    return re.search(r"AXFLAIR_(\d+)_", file).group(1)


def geometry_stratum(spacing_row_mm, spacing_slice_mm):
    return f"inplane_{spacing_row_mm:.2f}_slice_{int(round(spacing_slice_mm))}"


def h5_of(stem):
    for split in ("multicoil_train", "multicoil_val"):
        p = KROOT / split / f"{stem}.h5"
        if p.exists():
            return p
    raise FileNotFoundError(stem)


def analyse_volume(seg, zooms, rows, n_rows, meta):
    """seg: (col, row, slice) SynthSeg labels; zooms: (col, row, slice) mm; rows: this file's CSV-frame small-lesion
    boxes. One dict per merged lesion."""
    lesions = merge_boxes_3d(rows_to_rss_frame(rows, n_rows), "label")
    class_map = host_class_map(seg)
    dist = class_distance_maps(class_map, zooms)
    look_all, look_par = BrainLookup(seg, zooms, BRAIN_ALL), BrainLookup(seg, zooms, BRAIN_PARENCHYMA)
    out = []
    for L in lesions:
        rects = member_rects(L["members"], seg.shape)
        row = {"file": meta["file"], "patient_id": meta["patient_id"], "series": meta["series"],
               "stratum_series": "200_201" if meta["series"] in ("200", "201") else "other",
               "stratum_geometry": geometry_stratum(meta["spacing_row_mm"], meta["spacing_slice_mm"]),
               "label": L["label"], "z0": L["z0"], "z1": L["z1"], "n_slices": L["z1"] - L["z0"] + 1,
               "x0": L["x0"], "y0": L["y0"], "x1": L["x1"], "y1": L["y1"],
               "inplane_mm": max((L["x1"] - L["x0"]) * meta["spacing_col_mm"], (L["y1"] - L["y0"]) * meta["spacing_row_mm"]),
               "spacing_row_mm": meta["spacing_row_mm"], "spacing_col_mm": meta["spacing_col_mm"], "spacing_slice_mm": meta["spacing_slice_mm"]}
        if not rects or not dist:
            out.append({**row, "host_lookup_all": None, "host_lookup_parenchyma": None, "host_class_nearest": None,
                        "d1_mm": None, "d_interface_mm": None, "delta_d_mm": None, "status": "outside" if not rects else "no_host_class"})
            continue
        c1, d1, d2 = interface_margin(lesion_class_distances(dist, rects))
        out.append({**row, "host_lookup_all": look_all.host(rects)[0], "host_lookup_parenchyma": look_par.host(rects)[0],
                    "host_class_nearest": CLASS_NAMES[c1 - 1], "d1_mm": d1, "d_interface_mm": d2, "delta_d_mm": d2 - d1, "status": "ok"})
    return out


def freeze_t(share_at, min_share=MIN_SHARE):
    for t in T_GRID:
        if share_at[t] >= min_share:
            return t
    return None


def _quantiles(values):
    v = np.array([x for x in values if x is not None and np.isfinite(x)], float)
    return {"n": int(v.size), "p10": float(np.percentile(v, 10)), "median": float(np.median(v)),
            "p90": float(np.percentile(v, 90))} if v.size else {"n": 0}


def summarise(rows):
    ok = [r for r in rows if r["status"] == "ok"]
    d = np.array([r["d_interface_mm"] for r in ok], float)
    share_at = {t: float((d <= t).mean()) if d.size else 0.0 for t in (0, 1) + T_GRID}
    t = freeze_t(share_at)
    per_patient = Counter(r["patient_id"] for r in rows)
    m = np.array(sorted(per_patient.values()), float)
    out = {"n_lesions": len(rows), "n_ok": len(ok), "n_status": dict(Counter(r["status"] for r in rows)),
           "n_patients": len(per_patient),
           "per_patient": {"min": int(m.min()), "median": float(np.median(m)), "max": int(m.max()),
                           "m_eff": float((m ** 2).sum() / m.sum())} if m.size else {},
           "single_slice_share": float(np.mean([r["n_slices"] == 1 for r in rows])) if rows else 0.0,
           "share_at": share_at, "t_frozen": t, "hard_share": share_at[t] if t else None,
           "majority_flag": bool(t is not None and share_at[t] > 0.5),
           "d_interface": _quantiles([r["d_interface_mm"] for r in ok]), "delta_d": _quantiles([r["delta_d_mm"] for r in ok]),
           "host_class_nearest": dict(Counter(r["host_class_nearest"] for r in ok)),
           "host_lookup_all": dict(Counter(r["host_lookup_all"] for r in ok)),
           "host_lookup_parenchyma": dict(Counter(r["host_lookup_parenchyma"] for r in ok)),
           "strata": {}}
    for key in ("stratum_series", "stratum_geometry"):
        for s in sorted({r[key] for r in ok}):
            sel = np.array([r["d_interface_mm"] for r in ok if r[key] == s], float)
            out["strata"][f"{key}={s}"] = {"n": int(sel.size), **{f"share_le_{t}": float((sel <= t).mean()) for t in T_GRID}}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    rows = [r for r in read_fastmri_plus_rows(CSV) if "AXFLAIR" in r["file"] and r["label"] in SMALL_LABELS
            and r["width"] >= MIN_BOX_SIDE and r["height"] >= MIN_BOX_SIDE]
    by_file = {}
    for r in rows:
        by_file.setdefault(r["file"], []).append(r)
    files = sorted(by_file)[:a.limit] if a.limit else sorted(by_file)
    records = []
    for i, f in enumerate(files):
        g = volume_geometry(h5_of(f))
        img = nib.load(str(SEG_ROOT / f"{f}_seg.nii.gz"))
        seg = np.asarray(img.dataobj).astype(np.int16)
        zooms = tuple(float(z) for z in img.header.get_zooms()[:3])
        if seg.shape != (g["n_cols"], g["n_rows"], g["slices"]):
            raise ValueError(f"{f}: seg grid {seg.shape} != RSS (cols, rows, slices) {(g['n_cols'], g['n_rows'], g['slices'])}")
        meta = {"file": f, "patient_id": g["patient_id"], "series": series_of(f), "spacing_row_mm": g["spacing_row_mm"],
                "spacing_col_mm": g["spacing_col_mm"], "spacing_slice_mm": g["spacing_slice_mm"]}
        records += analyse_volume(seg, zooms, by_file[f], g["n_rows"], meta)
        print(f"{i + 1}/{len(files)} {f}: {len(by_file[f])} boxes -> {len(records)} lesions so far", flush=True)
    for k, r in enumerate(records):
        r["lesion_id"] = k
    with open(a.out / "lesions.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=FIELDS)
        w.writeheader()
        w.writerows(records)
    s = summarise(records)
    verdict = "DECIDE" if s["majority_flag"] else "GO" if s["t_frozen"] else "NO-GO"
    s["verdict"] = verdict
    (a.out / "gate05.json").write_text(json.dumps(s, indent=1))
    lines = ["# Gate 0.5 brain frame (fastMRI+ FLAIR small lesions on clean SynthSeg)", "",
             f"lesions {s['n_lesions']} (ok {s['n_ok']}, {s['n_status']}), patients {s['n_patients']}, per patient {s['per_patient']}",
             f"single-slice share {s['single_slice_share']:.3f}", "",
             "share of lesions with d_interface <= t (host classes of v2.6 §3, sides merged, ventricles/CSF landmarks):",
             "".join(f"  t={t}: {v:.3f}" for t, v in s["share_at"].items()), "",
             f"d_interface quantiles {s['d_interface']}", f"delta_d quantiles {s['delta_d']}", "",
             f"nearest host class: {s['host_class_nearest']}", f"lookup host (all 33): {s['host_lookup_all']}",
             f"lookup host (parenchyma only): {s['host_lookup_parenchyma']}", "", "strata:"]
    lines += [f"  {k}: {v}" for k, v in s["strata"].items()]
    lines += ["", f"t_frozen = {s['t_frozen']} (smallest t in {T_GRID} with share >= {MIN_SHARE}); hard share {s['hard_share']}; majority_flag {s['majority_flag']}",
              f"GATE05: {verdict}"]
    (a.out / "brain_frame.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
