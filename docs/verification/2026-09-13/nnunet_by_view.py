"""Did the degradations actually damage the anatomy? nnU-Net Dice and boundary shift per view."""
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from anatobind.train.cache import VIEWS, load_array, load_meta

FM = Path("/data2/congcong/data/FM_data/derived/skmtea")
folds = json.loads((FM / "m1r/splits.json").read_text())["folds"]
dice = defaultdict(list)
shift = defaultdict(list)
for scan in sorted(folds):
    seg = np.asarray(load_array(FM / "m1r_cache" / scan, "seg"))
    sp = np.array(load_meta(FM / "m1r_cache" / scan)["spacing_mm"], dtype=float)
    ref = None
    for view in VIEWS:
        lm = np.load(FM / "m1r_pred/nnunet" / f"{scan}__{view}.npz")["label_map"]
        if view == "clean":
            ref = lm
        for k in range(1, 7):
            t, p = seg == k, lm == k
            if not t.any():
                continue
            dice[(view, k)].append(2.0 * np.logical_and(t, p).sum() / (t.sum() + p.sum()))
            if view != "clean" and p.any() and (ref == k).any():
                c = lambda m: np.array(np.nonzero(m)).mean(1) * sp
                shift[view].append(float(np.linalg.norm(c(p) - c(ref == k))))
names = {1: "patellar", 2: "femoral", 3: "tibial med", 4: "tibial lat", 5: "menisc med", 6: "menisc lat"}
print(f"{'view':9}" + "".join(f"{names[k]:>12}" for k in range(1, 7)) + f"{'mean':>8}")
for v in VIEWS:
    row = [np.mean(dice[(v, k)]) for k in range(1, 7)]
    print(f"{v:9}" + "".join(f"{x:12.3f}" for x in row) + f"{np.mean(row):8.3f}")
print("\ncentroid shift of the predicted structure vs its clean prediction (mm, median / 90th pct):")
for v in VIEWS[1:]:
    s = np.array(shift[v])
    print(f"  {v:9} {np.median(s):.3f} / {np.percentile(s, 90):.3f}")
