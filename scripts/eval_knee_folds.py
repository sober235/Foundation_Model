#!/usr/bin/env python
"""Fold evaluation of the knee capability system on nnU-Net's held-out validation outputs (spec 2026-09-23 §4.1).

Per held-out scan and view: Dataset902's validation label map + npz probabilities give the detections,
Dataset901's validation label map gives the anatomy for the lookup, boxes.csv gives the truth.
Detection: sensitivity and false positives per scan over a score sweep, operating point at <= 2 FP/scan.
Binding: on the hits at the operating point (system path) and on the annotated boxes (given-box path,
a control, not a capability). Anatomy: Dice of Dataset901 against the export segmentation.

  D=docs/verification/$(date +%F)/knee_eval; mkdir -p $D
  source scripts/nnunet_env.sh
  PYTHONPATH=. python scripts/eval_knee_folds.py --out $D | tee $D/output.txt
"""
import argparse
import csv
import json
import os
import sys
from collections import defaultdict
from pathlib import Path

import nibabel as nib
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from anatobind.eval.detection_metrics import FP_MAX, GATE_SENSITIVITY, THRESHOLDS, gate, match_scan, operating_point, per_family, sweep  # noqa: E402
from anatobind.eval.lesion_boxes import decode_boxes, load_label_map, load_nnunet_probabilities  # noqa: E402
from anatobind.eval.lookup import FAMILY_OF_TISSUE, LabelIndex, describe_host  # noqa: E402
from anatobind.eval.matching import CORRECT, bucket  # noqa: E402
from anatobind.nnunet import prepare as anat  # noqa: E402
from anatobind.nnunet import prepare_lesion as les  # noqa: E402
from anatobind.nnunet.lesion_labels import LOOKUP_CLASS_OF_FAMILY, read_boxes_xyz  # noqa: E402
from anatobind.train.cache import VIEWS  # noqa: E402

FM = Path("/data2/congcong/data/FM_data/derived")
GATE_VIEW = "clean"
FIELDS = ["fold", "scan", "view", "ann_id", "layer", "gt_family", "tissue_id", "host_label", "host_side",
          "matched", "det_score", "det_family", "iou", "pred_host_label", "pred_side", "pred_host_fractions",
          "bucket", "side_correct", "given_host_label", "given_side", "given_bucket", "given_side_correct"]


def parse(argv):
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--export-root", type=Path, default=FM / "skmtea" / "m1r")
    ap.add_argument("--results-root", type=Path, default=Path(os.environ.get("nnUNet_results", FM / "nnunet" / "results")))
    ap.add_argument("--views", default=",".join(VIEWS))
    ap.add_argument("--folds", default="0,1,2,3,4")
    ap.add_argument("--allow-missing", action="store_true")
    a = ap.parse_args(argv)
    a.views = a.views.split(",")
    a.folds = [int(f) for f in a.folds.split(",")]
    return a


def iou_of(a, b):
    lo, hi = np.maximum(a[:3], b[:3]), np.minimum(a[3:], b[3:])
    inter = float(np.clip(hi - lo, 0, None).prod())
    va, vb = float(np.prod(np.subtract(a[3:], a[:3]))), float(np.prod(np.subtract(b[3:], b[:3])))
    return inter / (va + vb - inter) if va + vb - inter > 0 else 0.0


def load_scan(a, scan, fold, view, missing):
    paths = {"anat": anat.validation_path(a.results_root, fold, scan, view),
             "les": les.validation_path(a.results_root, fold, scan, view),
             "npz": les.validation_npz_path(a.results_root, fold, scan, view)}
    absent = [str(p) for p in paths.values() if not p.exists()]
    if absent:
        missing.extend(absent)
        return None
    seg_img = nib.load(str(a.export_root / scan / "seg.nii.gz"))
    spacing = tuple(float(v) for v in seg_img.header.get_zooms()[:3])
    lesion = load_label_map(paths["les"])
    dets = decode_boxes(lesion, load_nnunet_probabilities(paths["npz"], lesion))
    anatomy = load_label_map(paths["anat"])
    seg = np.asanyarray(seg_img.dataobj).astype(np.uint8)
    dice = {k: 2.0 * np.logical_and(anatomy == k, seg == k).sum() / ((anatomy == k).sum() + (seg == k).sum())
            for k in range(1, 7) if (seg == k).any()}
    gt = [{**r, "family": r["supercategory"]} for r in read_boxes_xyz(a.export_root / scan / "boxes.csv")]
    return {"scan": scan, "fold": fold, "view": view, "gt": gt,
            "dets": dets, "index": LabelIndex(anatomy, spacing), "dice": dice}


def side_ok(gt_side, pred_side):
    return "" if gt_side not in ("medial", "lateral") else str(int(gt_side == pred_side))


def _host_arg(h):
    return h["host_label"] if h["host_label"] is not None else h["host_name"]


def binding_records(scans, thr):
    records = []
    for s in scans:
        dets = [d for d in s["dets"] if d["score"] >= thr]
        pairs = match_scan(s["gt"], dets)
        for g, r in enumerate(s["gt"]):
            gt_cls = LOOKUP_CLASS_OF_FAMILY[r["supercategory"]]
            in_seg = r["layer"] == "in_seg" and r["tissue_id"] in FAMILY_OF_TISSUE
            rec = {"fold": s["fold"], "scan": s["scan"], "view": s["view"], "ann_id": r["ann_id"], "layer": r["layer"],
                   "gt_family": r["supercategory"], "tissue_id": r["tissue_id"],
                   "host_label": "" if r["host_label"] is None else r["host_label"], "host_side": r["host_side"],
                   "matched": "0", "det_score": "", "det_family": "", "iou": "", "pred_host_label": "", "pred_side": "-",
                   "pred_host_fractions": "{}", "bucket": "", "side_correct": ""}
            given = describe_host(s["index"], r["box"], gt_cls)
            rec.update({"given_host_label": "" if given["host_label"] is None else given["host_label"],
                        "given_side": given["side"],
                        "given_bucket": bucket(gt_cls, r["tissue_id"], gt_cls, _host_arg(given)) if in_seg else "",
                        "given_side_correct": side_ok(r["host_side"], given["side"])})
            if g in pairs:
                d = dets[pairs[g]]
                pred_cls = LOOKUP_CLASS_OF_FAMILY[d["family"]]
                h = describe_host(s["index"], d["box"], pred_cls)
                rec.update({"matched": "1", "det_score": round(d["score"], 4), "det_family": d["family"],
                            "iou": round(iou_of(np.array(r["box"]), np.array(d["box"])), 4),
                            "pred_host_label": "" if h["host_label"] is None else h["host_label"], "pred_side": h["side"],
                            "pred_host_fractions": json.dumps({str(k): round(v, 4) for k, v in h["host_fractions"].items()}),
                            "bucket": bucket(gt_cls, r["tissue_id"], pred_cls, _host_arg(h)) if in_seg else "",
                            "side_correct": side_ok(r["host_side"], h["side"])})
            elif in_seg:
                rec["bucket"] = "miss"
            records.append(rec)
    return records


def binding_summary(records, key_bucket, key_side):
    matched = [r for r in records if r[key_bucket] not in ("", "miss")]
    sides = [r for r in matched if r[key_side] != ""]
    return {"n": len(matched),
            "family_correct": sum(r[key_bucket] == CORRECT for r in matched) / len(matched) if matched else float("nan"),
            "n_side": len(sides),
            "side_correct": sum(r[key_side] == "1" for r in sides) / len(sides) if sides else float("nan")}


def main(argv=None):
    a = parse(argv)
    a.out.mkdir(parents=True, exist_ok=True)
    folds = json.loads((a.export_root / "splits.json").read_text())["folds"]
    missing, per_view = [], {}
    for view in a.views:
        scans = [load_scan(a, scan, fold, view, missing) for scan, fold in sorted(folds.items()) if fold in a.folds]
        per_view[view] = [s for s in scans if s is not None]
    if missing and not a.allow_missing:
        print(f"{len(missing)} validation files missing, e.g. {missing[:3]}; rerun with --allow-missing to evaluate the rest")
        return 2
    records, gate_out = [], {}
    lines = ["# Knee capability evaluation", "",
             f"IoU >= 0.1, FP budget {FP_MAX}/scan, gate on `{GATE_VIEW}` family sensitivity >= {GATE_SENSITIVITY}", "",
             "| view | n_scans | n_gt | thr | sensitivity | sensitivity_family | fp_per_scan |", "|---|---|---|---|---|---|---|"]
    detail = []
    for view, scans in per_view.items():
        rows = sweep(scans, THRESHOLDS)
        with open(a.out / f"froc_{view}.csv", "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)
        op = operating_point(rows)
        g = gate(rows)
        if view == GATE_VIEW:
            gate_out = {"view": view, **g, "n_scans": len(scans), "n_gt": rows[0]["n_gt"] if rows else 0}
        thr = op["thr"] if op else 1.0
        lines.append(f"| {view} | {len(scans)} | {rows[0]['n_gt'] if rows else 0} | {thr:.2f} | "
                     f"{(op or {}).get('sensitivity', 0.0):.3f} | {(op or {}).get('sensitivity_family', 0.0):.3f} | "
                     f"{(op or {}).get('fp_per_scan', float('nan')):.2f} |")
        recs = binding_records(scans, thr)
        records += recs
        fam = per_family(scans, thr)
        sysb = binding_summary(recs, "bucket", "side_correct")
        given = binding_summary(recs, "given_bucket", "given_side_correct")
        dice = defaultdict(list)
        for s in scans:
            for k, v in s["dice"].items():
                dice[k].append(v)
        detail += [f"## {view}", "", "per family at the operating point: " + ", ".join(
            f"{f}: {v['n_hit_family']}/{v['n_gt']} ({v['sensitivity_family']:.2f})" for f, v in sorted(fam.items())), "",
            f"binding on hits (system): n={sysb['n']}, family correct {sysb['family_correct']:.3f}, "
            f"side correct {sysb['side_correct']:.3f} (n_side={sysb['n_side']})",
            f"binding on annotated boxes (given-box control): n={given['n']}, family correct {given['family_correct']:.3f}, "
            f"side correct {given['side_correct']:.3f} (n_side={given['n_side']})",
            "anatomy Dice (Dataset901 vs export seg): " + ", ".join(f"{k}: {np.mean(v):.3f}" for k, v in sorted(dice.items())), ""]
    lines += [""] + detail
    lines.append(f"GATE: {'PASS' if gate_out.get('pass') else 'FAIL'} ({json.dumps(gate_out)})")
    if missing:
        lines += ["", f"{len(missing)} validation files were missing and skipped."]
    (a.out / "summary.md").write_text("\n".join(lines) + "\n")
    (a.out / "gate.json").write_text(json.dumps(gate_out, indent=1))
    with open(a.out / "records.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=FIELDS)
        w.writeheader()
        w.writerows(records)
    print("\n".join(lines))
    return 0


if __name__ == "__main__":
    sys.exit(main())
