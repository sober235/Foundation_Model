#!/usr/bin/env python
"""Quick look at one fold's cached upstream predictions (checkpoint review, not a gate).

Held-out scans of the fold: detection recall/precision per view, B0 buckets on OUR masks,
mean Dice of our label maps on held-out and training scans. G2 itself uses nnU-Net masks and all folds
(scripts/eval_g1_g2.py); this is only to judge whether the upstream trains at all.

  PYTHONNOUSERSITE=1 PYTHONPATH=. ~/anaconda3/envs/nvgen/bin/python scripts/peek_fold.py --fold 0
"""
import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from anatobind.eval.lookup import LabelIndex, b0_host  # noqa: E402
from anatobind.eval.matching import bucket, detected, match  # noqa: E402
from anatobind.eval.predict import load_prediction  # noqa: E402
from anatobind.train.cache import VIEWS, load_array, load_meta  # noqa: E402
from anatobind.train.dataset_v2 import read_all_boxes  # noqa: E402

FM = Path("/data2/congcong/data/FM_data/derived/skmtea")


def dice(a, b):
    s = a.sum() + b.sum()
    return float(2.0 * np.logical_and(a, b).sum() / s) if s else float("nan")


def peek(fold, export_root=FM / "m1r", cache_root=FM / "m1r_cache", pred_root=FM / "m1r_pred", thr=0.1):
    folds = json.loads((Path(export_root) / "splits.json").read_text())["folds"]
    det = {v: Counter() for v in VIEWS}
    buckets = {v: Counter() for v in VIEWS}
    dices = defaultdict(list)
    for scan, f in sorted(folds.items()):
        group = "held_out" if f == fold else "training"
        rows = read_all_boxes(Path(export_root) / scan / "boxes.csv")
        gt = np.array([r["box"] for r in rows], dtype=float).reshape(-1, 6)
        seg = np.asarray(load_array(Path(cache_root) / scan, "seg"))
        spacing = load_meta(Path(cache_root) / scan)["spacing_mm"]
        for view in VIEWS:
            p = load_prediction(Path(pred_root) / "ours" / f"fold{fold}" / f"{scan}__{view}.npz")
            lm = p["label_map"]
            d = [dice(lm == k, seg == k) for k in range(1, 7) if (seg == k).any()]
            dices[(view, group)].append(float(np.mean(d)))
            if group != "held_out":
                continue
            idx = detected(p["cls_prob"])
            pboxes, pcls = p["boxes_vox"][idx], p["cls_prob"][idx].argmax(-1)
            m = match(gt, pboxes, thr)
            det[view]["detections"] += len(idx)
            det[view]["matched"] += len(m)
            det[view]["truth"] += len(rows)
            index = LabelIndex(lm, spacing)
            for j, r in enumerate(rows):
                if r["layer"] != "in_seg":
                    continue
                q = m.get(j)
                pc = None if q is None else int(pcls[q])
                label = None if q is None else b0_host(index, pboxes[q], pc)
                buckets[view][bucket(r["cls"], r["tissue_id"], pc, label)] += 1
    return {
        "held_out_scans": sum(1 for f in folds.values() if f == fold),
        "training_scans": sum(1 for f in folds.values() if f != fold),
        "detection": {v: {"recall": det[v]["matched"] / max(det[v]["truth"], 1),
                          "precision": det[v]["matched"] / max(det[v]["detections"], 1), **det[v]} for v in VIEWS},
        "buckets": {v: dict(buckets[v]) for v in VIEWS},
        "dice": {v: {g: float(np.mean(dices[(v, g)])) for g in ("held_out", "training")} for v in VIEWS},
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fold", type=int, required=True)
    a = ap.parse_args()
    out = peek(a.fold)
    print(f"fold {a.fold}: {out['held_out_scans']} held-out scans, {out['training_scans']} training scans (IoU 0.1)")
    print(f"{'view':9} {'recall':>7} {'prec':>6} {'det':>5} | {'correct':>8} {'wr_host':>8} {'wr_cls':>7} {'miss':>5} | {'Dice held':>9} {'Dice train':>10}")
    for v in VIEWS:
        d, b, dc = out["detection"][v], out["buckets"][v], out["dice"][v]
        n = max(sum(b.values()), 1)
        print(f"{v:9} {d['recall']:7.3f} {d['precision']:6.3f} {d['detections']:5d} | {b.get('correct', 0) / n:8.3f} "
              f"{b.get('wrong_host', 0) / n:8.3f} {b.get('wrong_class', 0) / n:7.3f} {b.get('miss', 0) / n:5.3f} | "
              f"{dc['held_out']:9.3f} {dc['training']:10.3f}")


if __name__ == "__main__":
    main()
