#!/usr/bin/env python
# scripts/eval_fastmri_knee_detection.py
"""H1 rerun evaluation (v2.6 §6.2): the leg 2 knee detector trained on Gate-0 boxes, held-out patients, every fold.

  PYTHONNOUSERSITE=1 PYTHONPATH=. nice -n 19 ~/anaconda3/envs/nvgen/bin/python scripts/eval_fastmri_knee_detection.py \
      --out docs/verification/2026-09-24/h1_rerun
"""
import argparse
import csv
import json
import pickle
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from anatobind.data_engine.fastmri_knee import ANNOTATIONS, EXPORT_ROOT, VIEWS, load_folds, load_lesions, load_manifest  # noqa: E402
from anatobind.eval.detection_metrics import FP_MAX, GATE_SENSITIVITY, gate, operating_point, per_family, sweep  # noqa: E402
from anatobind.eval.fastmri_knee_detection import (  # noqa: E402
    SHARED_FAMILIES, THRESHOLDS, fp_per_normal_scan, hit_geometry, patient_coverage, scan_record, size_terciles,
)

FAMILY_SETS = {"shared4": SHARED_FAMILIES, "all5": None}
DECISION = ("clean", "shared4")


def load_view(export_root, det_root, view):
    """Raw material of one view: {file: (gt lesions, cached 3D detections, normal?)} over every held-out file
    (each file is held out in exactly one fold), plus spacing and patient per file and the files without a cache."""
    manifest, folds = load_manifest(export_root), load_folds(export_root)
    by_file = {}
    for r in load_lesions(export_root):
        by_file.setdefault(r["file"], []).append(r)
    with open(ANNOTATIONS, newline="") as fh:
        annotated = {r["file"] for r in csv.DictReader(fh)}        # every row, study-level ones included: a normal
    raw, missing = {}, []                                          # volume is one the radiologist wrote nothing about
    for f, k in sorted(folds.items()):
        p = Path(det_root) / f"fold{k}" / f"{f}__{view}.pkl"
        if not p.exists():
            missing.append(f)
            continue
        raw[f] = (by_file.get(f, []), pickle.load(open(p, "rb"))["dets"], f not in annotated)
    spacing_of = {f: (float(m["spacing_slice_mm"]), float(m["spacing_row_mm"]), float(m["spacing_col_mm"])) for f, m in manifest.items()}
    patient_of = {f: m["patient_id"] for f, m in manifest.items()}
    return raw, spacing_of, patient_of, missing


def write_csv(path, rows):
    with open(path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--export-root", type=Path, default=EXPORT_ROOT)
    ap.add_argument("--det-root", type=Path, default=None)
    ap.add_argument("--views", default=",".join(VIEWS))
    a = ap.parse_args()
    det_root = a.det_root or a.export_root / "detections"
    a.out.mkdir(parents=True, exist_ok=True)
    lines = ["# fastMRI+ knee detector after Gate 0 (held-out patients, five folds)", "",
             f"IoU >= 0.1, FP budget {FP_MAX}/scan, transfer gate on clean x shared4 family sensitivity >= {GATE_SENSITIVITY}", "",
             "| view | families | n_scans | n_gt | thr | sensitivity | sensitivity_family | fp_per_scan | fp_per_normal_scan | ceiling (sens_fam @ fp) |",
             "|---|---|---|---|---|---|---|---|---|---|"]
    decision = None
    for view in a.views.split(","):
        raw, spacing_of, patient_of, missing = load_view(a.export_root, det_root, view)
        for fam_name, fams in FAMILY_SETS.items():
            scans = [scan_record(f, gt, dets, normal, fams) for f, (gt, dets, normal) in raw.items()]
            rows = sweep(scans, THRESHOLDS)
            write_csv(a.out / f"froc_{view}_{fam_name}.csv", rows)
            op = operating_point(rows)
            ceiling = max(rows, key=lambda r: r["sensitivity_family"])
            n_gt = sum(len(s["gt"]) for s in scans)
            if op is None:
                lines.append(f"| {view} | {fam_name} | {len(scans)} | {n_gt} | - | - | - | - | - | {ceiling['sensitivity_family']:.3f} @ {ceiling['fp_per_scan']:.2f} |")
                continue
            fpn = fp_per_normal_scan(scans, op["thr"])
            lines.append(f"| {view} | {fam_name} | {len(scans)} | {n_gt} | {op['thr']:.2f} | {op['sensitivity']:.3f} | "
                         f"{op['sensitivity_family']:.3f} | {op['fp_per_scan']:.2f} | {fpn['fp_per_normal_scan']:.2f} (n={fpn['n_normal']}) | "
                         f"{ceiling['sensitivity_family']:.3f} @ {ceiling['fp_per_scan']:.2f} |")
            if (view, fam_name) == DECISION:
                fams_at_op = per_family(scans, op["thr"])
                terciles = size_terciles(scans, spacing_of, op["thr"])
                hits = [h for s in scans for h in hit_geometry(s, op["thr"], spacing_of[s["file"]])]
                cov = patient_coverage(scans, patient_of, op["thr"])
                decision = {**gate(rows), "view": view, "families": fam_name, "n_scans": len(scans), "n_gt": n_gt,
                            "missing_detection_files": len(missing), "fp_per_normal_scan": fpn,
                            "per_family": fams_at_op, "size_terciles": terciles, "patient_coverage": cov,
                            "n_hits": len(hits),
                            "centre_error_mm_median": float(np.median([h["centre_error_mm"] for h in hits])) if hits else None,
                            "iou_median": float(np.median([h["iou"] for h in hits])) if hits else None,
                            "family_correct_on_hits": float(np.mean([h["family"] == h["pred_family"] for h in hits])) if hits else None}
                if hits:
                    write_csv(a.out / f"hits_{view}.csv", hits)
    lines += ["", "## decision view (clean x shared4)", "", "```", json.dumps(decision, indent=1), "```", "",
              f"TRANSFER_GATE: {'PASS' if decision and decision['pass'] else 'FAIL'} "
              f"({json.dumps({k: decision[k] for k in ('thr', 'sensitivity_family', 'fp_per_scan')}) if decision else 'no operating point'})"]
    (a.out / "summary.md").write_text("\n".join(lines) + "\n")
    (a.out / "gate.json").write_text(json.dumps(decision, indent=1))
    print("\n".join(lines))


if __name__ == "__main__":
    main()
