"""The clean-only nnU-Net probe: per-view Dice and the G2 gate path on its masks.

Fold 5 of Dataset901 trained on the 744 clean cases of fold 0's training scans and was validated on
the same 217 held-out cases (31 scans x 7 views), so it has never seen a degraded image.
"""
import os
import sys
from collections import Counter, defaultdict
from pathlib import Path

import nibabel as nib
import numpy as np

from anatobind.eval.g2 import g2
from anatobind.eval.lookup import LabelIndex, b0_host
from anatobind.eval.matching import bucket
from anatobind.train.cache import VIEWS, load_array, load_meta
from anatobind.train.dataset import to_model_frame
from anatobind.train.dataset_v2 import read_all_boxes

FM = Path("/data2/congcong/data/FM_data/derived/skmtea")
VAL = Path(os.environ["nnUNet_results"]) / "Dataset901_SKMTEAm1r/nnUNetTrainer_250epochs__nnUNetPlans__3d_fullres/fold_5/validation"
NNUNET = FM / "m1r_pred/nnunet"

scans = sorted({p.name.rsplit("_", 1)[0].replace("_noise", "").replace("_us", "")[:7] for p in VAL.glob("MTR_*.nii.gz")})
dice = defaultdict(list)
ref_dice = defaultdict(list)
shift = defaultdict(list)
outcomes = defaultdict(dict)
by_view = {v: Counter() for v in VIEWS}
n_scan = 0
for scan in scans:
    files = {v: VAL / f"{scan}_{'clean0' if v == 'clean' else v}.nii.gz" for v in VIEWS}
    if not all(f.exists() for f in files.values()):
        continue
    n_scan += 1
    seg = np.asarray(load_array(FM / "m1r_cache" / scan, "seg"))
    sp = load_meta(FM / "m1r_cache" / scan)["spacing_mm"]
    rows = [r for r in read_all_boxes(FM / "m1r" / scan / "boxes.csv") if r["layer"] == "in_seg"]
    clean_lm = None
    for v in VIEWS:
        lm = to_model_frame(np.asanyarray(nib.load(str(files[v])).dataobj).astype(np.uint8))
        assert lm.shape == seg.shape, (lm.shape, seg.shape)
        if v == "clean":
            clean_lm = lm
        ref = np.load(NNUNET / f"{scan}__{v}.npz")["label_map"]
        d, rd = [], []
        for k in range(1, 7):
            t = seg == k
            if not t.any():
                continue
            d.append(2.0 * np.logical_and(lm == k, t).sum() / ((lm == k).sum() + t.sum()))
            rd.append(2.0 * np.logical_and(ref == k, t).sum() / ((ref == k).sum() + t.sum()))
            if v != "clean" and (lm == k).any() and (clean_lm == k).any():
                c = lambda m: np.array(np.nonzero(m)).mean(1) * np.asarray(sp)
                shift[v].append(float(np.linalg.norm(c(lm == k) - c(clean_lm == k))))
        dice[v].append(np.mean(d))
        ref_dice[v].append(np.mean(rd))
        index = LabelIndex(lm, sp)
        for r in rows:
            b = bucket(r["cls"], r["tissue_id"], r["cls"], b0_host(index, r["box"], r["cls"]))
            outcomes[(scan, r["ann_id"])][v] = b
            by_view[v][b] += 1

print(f"{n_scan} held-out scans, {sum(by_view['clean'].values())} in-segmentation lesions\n")
print(f"{'view':9}{'Dice clean-only':>17}{'Dice degraded-trained':>23}{'shift vs own clean (mm)':>25}{'wrong host':>12}")
for v in VIEWS:
    n = sum(by_view[v].values())
    s = f"{np.median(shift[v]):.3f}" if shift[v] else "-"
    print(f"{v:9}{np.mean(dice[v]):17.3f}{np.mean(ref_dice[v]):23.3f}{s:>25}{by_view[v]['wrong_host']/n:12.3f}")
res = g2(outcomes)
print("\nG2 on the clean-only masks (same criterion: +0.05 at noise_q3 or us16, CI excluding 0):")
for v in ("noise_q3", "us16"):
    d = res[v]
    print("  %-9s delta %+0.4f  95%% CI [%+0.4f, %+0.4f]  clean %.4f -> %.4f  n=%d lesions, %d scans"
          % (v, d["delta"], d["ci95"][0], d["ci95"][1], d["rate_clean"], d["rate_view"], d["n_lesions"], d["n_scans"]))
print(f"\nVERDICT on this probe: {'PASS' if res['pass'] else 'FAIL'}")
