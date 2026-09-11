#!/usr/bin/env python
"""Measure which frame puts the raw-data-track segmentation on the h5 grid, and how far the
dicom-track segmentation used by the m1 export was from it.

Writes docs/verification/2026-09-11/rawtrack_frame.csv and prints the per-orientation RULE.
  PYTHONNOUSERSITE=1 PYTHONPATH=. ~/anaconda3/envs/nvgen/bin/python scripts/discover_rawtrack_frame.py --workers 4
"""
import argparse
import csv
import json
import os
import sys
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import nibabel as nib
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from anatobind.data_engine.seg_frames import (  # noqa: E402
    FRAMES, apply_frame, frame_name, overlap_stats, rank_frames, zooms_in_h5_frame,
)
from anatobind.data_engine.skmtea import load_orientations, load_seg_h5_frame, load_target_magnitude  # noqa: E402

FM = Path("/data2/congcong/data/FM_data")
RAW = FM / "SKM-TEA/files_recon_calib-24"
RAWSEG = FM / "SKM-TEA_ltr/segmentation_masks/raw-data-track"
DCMSEG = FM / "SKM-TEA_ltr/segmentation_masks/dicom-track"
ANN = FM / "SKM-TEA_ltr/annotations/v1.0.0"
OUT = Path(__file__).resolve().parents[1] / "docs/verification/2026-09-11/rawtrack_frame.csv"
FIELDS = (["scan", "orientation", "shape_raw", "best", "score_best", "score_second", "score_identity", "spacing"]
          + [f"dice_{l}" for l in range(1, 7)] + [f"shift_mm_{l}" for l in range(1, 7)])


def one(args):
    scan, orientation = args
    os.nice(19)
    mag = load_target_magnitude(RAW / f"{scan}.h5", echo=0)
    img = nib.load(str(RAWSEG / f"{scan}.nii.gz"))
    arr = np.asarray(img.dataobj).astype(np.uint8)
    row = {"scan": scan, "orientation": "".join(orientation), "shape_raw": "x".join(map(str, arr.shape))}
    ranked = rank_frames(mag, arr)
    if not ranked:
        return {**row, "best": "shape-mismatch"}
    scores = dict(ranked)
    best, score = ranked[0]
    raw_h5 = np.ascontiguousarray(apply_frame(arr, best))
    dcm_h5 = load_seg_h5_frame(DCMSEG / f"{scan}.nii.gz", orientation)
    sp = zooms_in_h5_frame(img.header.get_zooms(), best)
    stats = overlap_stats(dcm_h5, raw_h5, sp)
    row.update({"best": frame_name(best), "score_best": f"{score:.3f}",
                "score_second": f"{ranked[1][1]:.3f}", "score_identity": f"{scores.get(FRAMES[0], float('nan')):.3f}",
                "spacing": ";".join(f"{v:.5f}" for v in sp)})
    for label in range(1, 7):
        d, s = stats.get(label, (float("nan"), float("nan")))
        row[f"dice_{label}"] = f"{d:.3f}"
        row[f"shift_mm_{label}"] = f"{s:.2f}"
    return row


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=4)
    a = ap.parse_args()
    orient = load_orientations(ANN)
    scans = sorted(p.name[:-7] for p in RAWSEG.glob("MTR_*.nii.gz"))
    with ProcessPoolExecutor(max_workers=a.workers) as ex:
        rows = list(ex.map(one, [(s, orient[s]) for s in scans]))
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=FIELDS, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)
    print(f"{len(rows)} scans -> {OUT}")
    rule = {}
    for o in ("LR", "RL"):
        group = [r for r in rows if r["orientation"].endswith(o)]
        counts = Counter(r["best"] for r in group)
        rule[o] = counts.most_common(1)[0][0]
        print(f"{o}: {len(group)} scans, best-frame counts {dict(counts)}, "
              f"deviating {[r['scan'] for r in group if r['best'] != rule[o]]}")
    print("RULE", json.dumps(rule))
    low = [(r["scan"], r["score_best"]) for r in rows if r["best"] != "shape-mismatch" and float(r["score_best"]) <= 1.5]
    print(f"score_best <= 1.5 (the dicom-track gate line): {low}")
    print("dicom-track vs raw-data-track, per label: median Dice, median / max centroid shift (mm)")
    for label in range(1, 7):
        d = np.array([float(r[f"dice_{label}"]) for r in rows if r.get(f"dice_{label}", "nan") != "nan"])
        s = np.array([float(r[f"shift_mm_{label}"]) for r in rows if r.get(f"shift_mm_{label}", "nan") != "nan"])
        print(f"  label {label}: n={len(d)}  Dice median {np.median(d):.3f} (min {d.min():.3f})  "
              f"shift median {np.median(s):.2f} / max {s.max():.2f} mm")


if __name__ == "__main__":
    main()
