"""Is the binding task sensitive to anatomy quality at all?

Same gate path (annotated box and class, class-aware lookup), three anatomy sources on the SAME
scans: nnU-Net (held-out Dice 0.86), our upstream (held-out Dice 0.38), and the ground-truth masks.
Folds 0-2 of our upstream are cached, so all three are scored on those 93 scans.
"""
import json
from collections import Counter
from pathlib import Path

import numpy as np

from anatobind.eval.lookup import LabelIndex, b0_host
from anatobind.eval.matching import bucket
from anatobind.eval.predict import load_prediction
from anatobind.train.cache import load_array, load_meta
from anatobind.train.dataset_v2 import read_all_boxes

FM = Path("/data2/congcong/data/FM_data/derived/skmtea")
folds = json.loads((FM / "m1r/splits.json").read_text())["folds"]
scans = sorted(s for s, f in folds.items() if f in (0, 1, 2))
out = {k: Counter() for k in ("truth", "nnunet", "ours")}
dice = {k: [] for k in ("nnunet", "ours")}
for scan in scans:
    rows = [r for r in read_all_boxes(FM / "m1r" / scan / "boxes.csv") if r["layer"] == "in_seg"]
    if not rows:
        continue
    seg = np.asarray(load_array(FM / "m1r_cache" / scan, "seg"))
    sp = load_meta(FM / "m1r_cache" / scan)["spacing_mm"]
    maps = {"truth": seg,
            "nnunet": np.load(FM / "m1r_pred/nnunet" / f"{scan}__clean.npz")["label_map"],
            "ours": load_prediction(FM / f"m1r_pred/ours/fold{folds[scan]}" / f"{scan}__clean.npz")["label_map"]}
    for k in ("nnunet", "ours"):
        d = [2.0 * np.logical_and(maps[k] == c, seg == c).sum() / ((maps[k] == c).sum() + (seg == c).sum())
             for c in range(1, 7) if (seg == c).any()]
        dice[k].append(np.mean(d))
    for k, lm in maps.items():
        index = LabelIndex(lm, sp)
        for r in rows:
            out[k][bucket(r["cls"], r["tissue_id"], r["cls"], b0_host(index, r["box"], r["cls"]))] += 1
n = sum(out["truth"].values())
print(f"{len(scans)} scans (folds 0-2), {n} in-segmentation lesions, clean view\n")
print(f"{'anatomy source':22}{'mean Dice':>10}{'correct':>10}{'wrong host':>12}")
for k, label in (("truth", "ground truth"), ("nnunet", "nnU-Net"), ("ours", "our upstream")):
    d = f"{np.mean(dice[k]):10.3f}" if k in dice else f"{1.0:10.3f}"
    print(f"{label:22}{d}{out[k]['correct']/n:10.3f}{out[k]['wrong_host']/n:12.3f}")
