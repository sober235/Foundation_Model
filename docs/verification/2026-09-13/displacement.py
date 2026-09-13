"""How much displacement would it take to make the gate fire?

The gate needs the wrong-host rate to rise 5 points above clean. Degradation moves the predicted
anatomy relative to the lesion; here that relative shift is applied directly, by translating each
lesion box by d mm in a fixed random direction on the ground-truth anatomy (16 directions per
lesion, seed 0, so the answer does not depend on one lucky direction).
"""
import json
from collections import Counter
from pathlib import Path

import numpy as np

from anatobind.eval.lookup import LabelIndex, b0_host
from anatobind.eval.matching import bucket
from anatobind.train.cache import load_array, load_meta
from anatobind.train.dataset_v2 import read_all_boxes

FM = Path("/data2/congcong/data/FM_data/derived/skmtea")
DISTANCES = (0.0, 0.5, 1.0, 2.0, 3.0, 5.0, 8.0, 12.0)
folds = json.loads((FM / "m1r/splits.json").read_text())["folds"]
rng = np.random.default_rng(0)
dirs = rng.normal(size=(16, 3))
dirs /= np.linalg.norm(dirs, axis=1, keepdims=True)
tally = {d: Counter() for d in DISTANCES}
for scan in sorted(folds):
    rows = [r for r in read_all_boxes(FM / "m1r" / scan / "boxes.csv") if r["layer"] == "in_seg"]
    if not rows:
        continue
    seg = np.asarray(load_array(FM / "m1r_cache" / scan, "seg"))
    sp = np.array(load_meta(FM / "m1r_cache" / scan)["spacing_mm"], dtype=float)
    index = LabelIndex(seg, list(sp))
    for r in rows:
        for d in DISTANCES:
            for u in (dirs if d else dirs[:1]):
                off = np.tile(u * d / sp, 2)
                host = b0_host(index, np.asarray(r["box"], dtype=float) + off, r["cls"])
                tally[d][bucket(r["cls"], r["tissue_id"], r["cls"], host)] += 1
base = tally[0.0]["wrong_host"] / sum(tally[0.0].values())
print(f"lesions: {sum(tally[0.0].values())}; ground-truth anatomy; wrong-host rate vs relative shift\n")
print(f"{'shift (mm)':>11}{'wrong host':>12}{'over clean':>12}   gate (+0.05)")
for d in DISTANCES:
    n = sum(tally[d].values())
    w = tally[d]["wrong_host"] / n
    print(f"{d:11.1f}{w:12.3f}{w - base:+12.3f}   {'fires' if w - base >= 0.05 else ''}")
