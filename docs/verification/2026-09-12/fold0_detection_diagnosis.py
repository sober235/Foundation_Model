"""Fold-0 detection diagnosis from the cached predictions (read-only). See fold0_pilot.md section 2."""
import json
from collections import defaultdict
from pathlib import Path
import numpy as np
from anatobind.eval.matching import iou3d, match
from anatobind.eval.predict import load_prediction
from anatobind.train.dataset_v2 import read_all_boxes

FM = Path("/data2/congcong/data/FM_data/derived/skmtea")
folds = json.loads((FM / "m1r/splits.json").read_text())["folds"]
stats = defaultdict(lambda: {"truth": 0, "matched_any": 0, "matched_scores": [], "unmatched_scores": [], "n_slots": 0, "best_iou": []})
for scan, f in sorted(folds.items()):
    group = "held_out" if f == 0 else "training"
    rows = read_all_boxes(FM / "m1r" / scan / "boxes.csv")
    gt = np.array([r["box"] for r in rows], dtype=float).reshape(-1, 6)
    for view in ("clean", "noise_q3", "us16"):
        p = load_prediction(FM / "m1r_pred/ours/fold0" / f"{scan}__{view}.npz")
        scores = p["cls_prob"][:, :4].max(-1)
        m = match(gt, p["boxes_vox"], 0.1)
        s = stats[(group, view)]
        s["truth"] += len(rows); s["matched_any"] += len(m); s["n_slots"] += len(scores)
        s["matched_scores"] += [float(scores[q]) for q in m.values()]
        s["unmatched_scores"] += [float(scores[q]) for q in range(len(scores)) if q not in m.values()]
        if len(gt):
            s["best_iou"] += iou3d(gt, p["boxes_vox"]).max(1).tolist()
for key in sorted(stats):
    s = stats[key]; ms = np.array(s["matched_scores"]); us = np.array(s["unmatched_scores"]); bi = np.array(s["best_iou"])
    print(f"{key[0]:9s} {key[1]:9s} truth {s['truth']:4d}  recall@IoU0.1 all slots {s['matched_any']/s['truth']:.3f}  "
          f"matched score median {np.median(ms):.3f}  unmatched median {np.median(us):.3f}  best IoU median {np.median(bi):.3f}")
    for thr in (0.5, 0.3, 0.2, 0.1):
        print(f"      score >= {thr}: matched {int((ms >= thr).sum())} / {s['truth']}   false slots {int((us >= thr).sum())} / {s['n_slots']}")
