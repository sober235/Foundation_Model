#!/usr/bin/env python
"""Re-export SKM-TEA with the gradient-warp-corrected segmentation (raw-data-track) into m1r.

Images are hard links to derived/skmtea/m1 (identical bytes). Only seg.nii.gz and boxes.csv are
regenerated, because host resolution (rule D5, part 3) reads the segmentation. m1 stays untouched
so the 2026-09-08 and 2026-09-09 reports remain reproducible.

  PYTHONNOUSERSITE=1 PYTHONPATH=. ~/anaconda3/envs/nvgen/bin/python scripts/build_skmtea_m1r.py --workers 4
"""
import argparse
import csv
import os
import shutil
import sys
import time
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import nibabel as nib

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from anatobind.data_engine.skmtea import (  # noqa: E402
    link_files, load_orientations, load_rawtrack_seg_h5_frame, rawtrack_spacing_h5_frame, write_seg_and_boxes,
)

FM = Path("/data2/congcong/data/FM_data")
M1, M1R = FM / "derived/skmtea/m1", FM / "derived/skmtea/m1r"
RAWSEG = FM / "SKM-TEA_ltr/segmentation_masks/raw-data-track"
ANN = FM / "SKM-TEA_ltr/annotations/v1.0.0"
IMAGES = ["image_clean_e1.nii.gz", "image_clean_e2.nii.gz", "image_noise_q1_e1.nii.gz", "image_noise_q2_e1.nii.gz",
          "image_noise_q3_e1.nii.gz", "image_us4_e1.nii.gz", "image_us8_e1.nii.gz", "image_us16_e1.nii.gz"]
FIELDS = ["scan_id", "out_dir", "n_boxes_kept", "n_ambiguous", "n_no_overlap", "n_unresolved",
          "spacing", "nrmse", "files", "status", "seconds"]
INT_KEYS = ("ann_id", "category_id", "tissue_id", "x0", "y0", "z0", "x1", "y1", "z1", "depth")


def screened_rows(path):
    by_scan = defaultdict(list)
    with open(path, newline="") as fh:
        for r in csv.DictReader(fh):
            by_scan[r["scan_id"]].append({**r, **{k: int(r[k]) for k in INT_KEYS}, "keep": r["keep"] == "True"})
    return by_scan


def job(args):
    scan, rows, orientation, m1_row = args
    os.nice(19)
    t0 = time.time()
    out = M1R / scan
    try:
        link_files(M1 / scan, out, IMAGES)
        seg_h5 = load_rawtrack_seg_h5_frame(RAWSEG / f"{scan}.nii.gz", orientation)
        sx, sy, sz = rawtrack_spacing_h5_frame(RAWSEG / f"{scan}.nii.gz", orientation)
        spacing = (sx * 2, sy * 2, sz)
        # The corrected NIfTI's zooms differ from the image's by up to 1e-4 mm (0.79988 vs 0.79998 on
        # MTR_112). Same grid, so the seg header takes the image's spacing exactly: nnU-Net's integrity
        # check compares the two headers and rejects even that difference.
        m1_spacing = tuple(float(v) for v in m1_row["spacing"].split(";"))
        if max(abs(a - b) for a, b in zip(spacing, m1_spacing)) > 1e-3:
            raise ValueError(f"spacing {spacing} differs from m1 {m1_spacing}")
        image_zooms = tuple(float(v) for v in nib.load(str(M1 / scan / "image_clean_e1.nii.gz")).header.get_zooms()[:3])
        counts = write_seg_and_boxes(seg_h5, image_zooms, rows, out)
        seg_zooms = tuple(float(v) for v in nib.load(str(out / "seg.nii.gz")).header.get_zooms()[:3])
        if seg_zooms != image_zooms:
            raise ValueError(f"seg zooms {seg_zooms} != image zooms {image_zooms}")
        row = {"scan_id": scan, "out_dir": str(out), **counts, "spacing": m1_row["spacing"], "nrmse": m1_row["nrmse"],
               "files": ";".join(IMAGES + ["seg.nii.gz", "boxes.csv"]), "status": "ok"}
    except Exception as exc:
        row = {"scan_id": scan, "out_dir": str(out), "status": f"error: {type(exc).__name__}: {exc}"[:200]}
    row["seconds"] = round(time.time() - t0, 1)
    return row


def in_seg_sides(root, scans):
    c = Counter()
    for s in scans:
        with open(root / s / "boxes.csv", newline="") as fh:
            c.update(r["host_side"] for r in csv.DictReader(fh) if r["layer"] == "in_seg")
    return dict(c)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=4)
    a = ap.parse_args()
    M1R.mkdir(parents=True, exist_ok=True)
    for name in ("splits.json", "annotations_screened.csv"):
        shutil.copy2(M1 / name, M1R / name)
    with open(M1 / "manifest.csv", newline="") as fh:
        m1_manifest = {r["scan_id"]: r for r in csv.DictReader(fh)}
    rows = screened_rows(M1 / "annotations_screened.csv")
    orient = load_orientations(ANN)
    scans = sorted(m1_manifest)
    with ProcessPoolExecutor(max_workers=a.workers) as ex:
        results = list(ex.map(job, [(s, rows.get(s, []), orient[s], m1_manifest[s]) for s in scans]))
    with open(M1R / "manifest.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=FIELDS, extrasaction="ignore")
        w.writeheader()
        w.writerows(results)
    bad = [r["scan_id"] for r in results if r["status"] != "ok"]
    print(f"{len(results)} scans, {len(bad)} failed: {bad}")
    if bad:
        for r in results:
            if r["status"] != "ok":
                print("  ", r["scan_id"], r["status"])
        sys.exit(1)
    print("in_seg host_side m1 :", in_seg_sides(M1, scans))
    print("in_seg host_side m1r:", in_seg_sides(M1R, scans))
    changed = []
    for s in scans:
        with open(M1 / s / "boxes.csv", newline="") as fh:
            old = {r["ann_id"]: r["host_label"] for r in csv.DictReader(fh)}
        with open(M1R / s / "boxes.csv", newline="") as fh:
            changed += [(s, r["ann_id"], old.get(r["ann_id"]), r["host_label"])
                        for r in csv.DictReader(fh) if old.get(r["ann_id"]) != r["host_label"]]
    print(f"host_label changed on {len(changed)} rows (scan, ann_id, m1, m1r):")
    for c in changed:
        print("  ", *c)


if __name__ == "__main__":
    main()
