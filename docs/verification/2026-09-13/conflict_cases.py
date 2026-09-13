"""V5 section 15.2's key subset: geometry-conflict cases.

A net paired delta near zero can still hide flips in both directions. Count them per view and per
anatomy source: clean-correct -> degraded-wrong (what a relation model would have to rescue) and
clean-wrong -> degraded-correct.
"""
import json
import os
from collections import Counter, defaultdict
from pathlib import Path

import nibabel as nib
import numpy as np

from anatobind.eval.lookup import LabelIndex, b0_host
from anatobind.eval.matching import bucket
from anatobind.eval.predict import load_prediction
from anatobind.train.cache import VIEWS, load_array, load_meta
from anatobind.train.dataset import to_model_frame
from anatobind.train.dataset_v2 import read_all_boxes

FM = Path("/data2/congcong/data/FM_data/derived/skmtea")
PROBE = Path(os.environ["nnUNet_results"]) / "Dataset901_SKMTEAm1r/nnUNetTrainer_250epochs__nnUNetPlans__3d_fullres/fold_5/validation"
folds = json.loads((FM / "m1r/splits.json").read_text())["folds"]
outcome = {src: defaultdict(dict) for src in ("nnunet", "ours", "probe")}
for scan in sorted(folds):
    rows = [r for r in read_all_boxes(FM / "m1r" / scan / "boxes.csv") if r["layer"] == "in_seg"]
    if not rows:
        continue
    sp = load_meta(FM / "m1r_cache" / scan)["spacing_mm"]
    shape = np.asarray(load_array(FM / "m1r_cache" / scan, "seg")).shape
    for view in VIEWS:
        maps = {"nnunet": np.load(FM / "m1r_pred/nnunet" / f"{scan}__{view}.npz")["label_map"],
                "ours": load_prediction(FM / f"m1r_pred/ours/fold{folds[scan]}" / f"{scan}__{view}.npz")["label_map"]}
        p = PROBE / f"{scan}_{'clean0' if view == 'clean' else view}.nii.gz"
        if p.exists():
            maps["probe"] = to_model_frame(np.asanyarray(nib.load(str(p)).dataobj).astype(np.uint8))
        for src, lm in maps.items():
            index = LabelIndex(lm, sp)
            for r in rows:
                outcome[src][(scan, r["ann_id"])][view] = bucket(r["cls"], r["tissue_id"], r["cls"],
                                                                 b0_host(index, r["box"], r["cls"]))
print("geometry-conflict cases (V5 15.2): lesions whose host flips between clean and the degraded view\n")
print(f"{'anatomy source':34}{'lesions':>9}" + "".join(f"{v:>12}" for v in VIEWS[1:]))
for src, label in (("nnunet", "nnU-Net (degraded-trained)"), ("probe", "nnU-Net (clean-only)"), ("ours", "our upstream (Dice 0.40)")):
    o = outcome[src]
    n = len(o)
    broke, fixed = [], []
    for v in VIEWS[1:]:
        b = sum(1 for d in o.values() if d.get("clean") == "correct" and d.get(v) == "wrong_host")
        f = sum(1 for d in o.values() if d.get("clean") == "wrong_host" and d.get(v) == "correct")
        broke.append(b); fixed.append(f)
    print(f"{label + ', broke':34}{n:9d}" + "".join(f"{x:12d}" for x in broke))
    print(f"{label + ', fixed':34}{'':9}" + "".join(f"{x:12d}" for x in fixed))
