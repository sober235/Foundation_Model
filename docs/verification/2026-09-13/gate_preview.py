"""G2 gate path preview on nnU-Net masks: annotated box and class, predicted anatomy.

Uses the same library calls as scripts/eval_g1_g2.py, so the numbers are the gate's own;
it exists only because the "B0 on our masks" report column still waits for upstream folds 3-4.
"""
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from anatobind.eval.g2 import GATE_VIEWS, MIN_DELTA, g2
from anatobind.eval.lookup import LabelIndex, b0_host
from anatobind.eval.matching import bucket
from anatobind.eval.predict import load_prediction
from anatobind.train.cache import VIEWS, load_meta
from anatobind.train.dataset_v2 import read_all_boxes

FM = Path("/data2/congcong/data/FM_data/derived/skmtea")
folds = json.loads((FM / "m1r/splits.json").read_text())["folds"]
outcomes = defaultdict(dict)
by_view = {v: Counter() for v in VIEWS}
by_class = defaultdict(lambda: {v: Counter() for v in VIEWS})
for scan in sorted(folds):
    rows = [r for r in read_all_boxes(FM / "m1r" / scan / "boxes.csv") if r["layer"] == "in_seg"]
    if not rows:
        continue
    spacing = load_meta(FM / "m1r_cache" / scan)["spacing_mm"]
    for view in VIEWS:
        lm = np.load(FM / "m1r_pred/nnunet" / f"{scan}__{view}.npz")["label_map"]
        index = LabelIndex(lm, spacing)
        for r in rows:
            host = b0_host(index, r["box"], r["cls"])
            b = bucket(r["cls"], r["tissue_id"], r["cls"], host)
            outcomes[(scan, r["ann_id"])][view] = b
            by_view[view][b] += 1
            by_class[r["cls"]][view][b] += 1

res = g2(outcomes)
print(f"scorable in-segmentation lesions: {sum(by_view['clean'].values())}")
print(f"\n{'view':9} {'correct':>8} {'wrong host':>11}   n")
for v in VIEWS:
    n = sum(by_view[v].values())
    print(f"{v:9} {by_view[v]['correct']/n:8.3f} {by_view[v]['wrong_host']/n:11.3f}   {n}")
print(f"\nG2 (paired wrong-host rate, view minus clean; threshold +{MIN_DELTA}, gate views {GATE_VIEWS}):")
for v in GATE_VIEWS:
    d = res[v]
    print("  %-9s delta %+0.4f  95%% CI [%+0.4f, %+0.4f]  clean %.4f -> %.4f  n=%d lesions, %d scans"
          % (v, d["delta"], d["ci95"][0], d["ci95"][1], d["rate_clean"], d["rate_view"], d["n_lesions"], d["n_scans"]))
print(f"\nG2 VERDICT: {'PASS' if res['pass'] else 'FAIL'}")
names = {0: "Meniscal Tear", 1: "Cartilage Lesion", 2: "Effusion", 3: "Ligament Tear"}
print(f"\nper class, wrong-host rate:")
print(f"{'class':18}" + "".join(f"{v:>10}" for v in VIEWS))
for c in sorted(by_class):
    t = by_class[c]
    n = sum(t["clean"].values())
    print(f"{names[c]:14} n={n:<3}" + "".join(f"{t[v]['wrong_host']/max(sum(t[v].values()),1):10.3f}" for v in VIEWS))
