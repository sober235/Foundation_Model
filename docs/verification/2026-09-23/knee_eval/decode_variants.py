#!/usr/bin/env python
"""How much recall does a cheaper decoder buy, without retraining?

The main evaluation forms candidates from nnU-Net's argmax label map. Here candidates are formed from the
per-family probability maps at a lower threshold p (a voxel can then belong to several families), scored
by the mean family probability, and swept exactly like the main evaluation (IoU >= 0.1, <= 2 FP/scan).
Uses the same held-out validation npz files; nothing is retrained. Throwaway analysis, kept as evidence.

  PYTHONNOUSERSITE=1 PYTHONPATH=. ~/anaconda3/envs/nvgen/bin/python docs/verification/2026-09-23/knee_eval/decode_variants.py --folds 0,1,2,3
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
from scipy import ndimage

sys.path.insert(0, str(Path(__file__).resolve().parents[4]))
from anatobind.eval.detection_metrics import THRESHOLDS, operating_point, per_family, sweep  # noqa: E402
from anatobind.eval.lesion_boxes import STRUCTURE, load_label_map, load_nnunet_probabilities  # noqa: E402
from anatobind.nnunet import prepare_lesion as les  # noqa: E402
from anatobind.nnunet.lesion_labels import FAMILY_OF_LABEL, read_boxes_xyz  # noqa: E402

EXPORT = Path("/data2/congcong/data/FM_data/derived/skmtea/m1r")
RESULTS = Path("/data2/congcong/data/FM_data/derived/nnunet/results")
P_LEVELS = (0.5, 0.3, 0.2, 0.1)
MIN_VOXELS = 27


def decode_at(probs, p, min_voxels=MIN_VOXELS):
    out = []
    for label, family in FAMILY_OF_LABEL.items():
        comp, n = ndimage.label(probs[label] >= p, structure=STRUCTURE)
        for k, sl in enumerate(ndimage.find_objects(comp), start=1):
            if sl is None:
                continue
            mask = comp[sl] == k
            nv = int(mask.sum())
            if nv < min_voxels:
                continue
            out.append({"family": family, "box": (sl[0].start, sl[1].start, sl[2].start, sl[0].stop, sl[1].stop, sl[2].stop),
                        "score": float(probs[label][sl][mask].mean()), "n_voxels": nv})
    return sorted(out, key=lambda d: -d["score"])


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--folds", default="0,1,2,3,4")
    ap.add_argument("--view", default="clean")
    a = ap.parse_args(argv)
    folds = json.loads((EXPORT / "splits.json").read_text())["folds"]
    wanted = [int(f) for f in a.folds.split(",")]
    scans = {p: [] for p in P_LEVELS}
    n = 0
    for scan, fold in sorted(folds.items()):
        if fold not in wanted:
            continue
        lab_path = les.validation_path(RESULTS, fold, scan, a.view)
        npz = les.validation_npz_path(RESULTS, fold, scan, a.view)
        if not lab_path.exists() or not npz.exists():
            continue
        lab = load_label_map(lab_path)
        probs = load_nnunet_probabilities(npz, lab)
        gt = [{**r, "family": r["supercategory"]} for r in read_boxes_xyz(EXPORT / scan / "boxes.csv")]
        for p in P_LEVELS:
            scans[p].append({"scan": scan, "gt": gt, "dets": decode_at(probs, p)})
        n += 1
    print(f"{a.view}: {n} scans, folds {wanted}")
    print("p_level | thr | sensitivity | sensitivity_family | fp_per_scan | ceiling(sens @ thr 0.05, fp/scan) | per family at op")
    for p in P_LEVELS:
        rows = sweep(scans[p], THRESHOLDS)
        op = operating_point(rows)
        fam = per_family(scans[p], op["thr"]) if op else {}
        low = rows[0]
        fams = ", ".join(f"{f[:4]} {v['n_hit_family']}/{v['n_gt']}" for f, v in sorted(fam.items()))
        if op:
            print(f"{p:5.2f} | {op['thr']:.2f} | {op['sensitivity']:.3f} | {op['sensitivity_family']:.3f} | {op['fp_per_scan']:.2f} | "
                  f"{low['sensitivity']:.3f} @ {low['fp_per_scan']:.2f} | {fams}")
        else:
            print(f"{p:5.2f} | no operating point under 2 FP/scan | ceiling {low['sensitivity']:.3f} @ {low['fp_per_scan']:.2f}")


if __name__ == "__main__":
    main()
