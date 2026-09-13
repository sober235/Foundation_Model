"""How much displacement flips the host in brain anatomy, against knee?

Same procedure as docs/verification/2026-09-13/displacement.py: host = the structure with the largest
intersection-over-box, then translate the box by d mm in 16 directions and count how often the host
changes. Brain boxes sit at random inside the parenchyma with the SKM-TEA lesion box sizes, so only
the anatomy differs. Brain has no class restriction, so knee is reported both ways.
"""
from collections import Counter
from pathlib import Path

import json
import nibabel as nib
import numpy as np

from anatobind.eval.lookup import CANDIDATES, LabelIndex, b0_host, clip_box
from anatobind.train.cache import load_array, load_meta
from anatobind.train.dataset_v2 import read_all_boxes

FM = Path("/data2/congcong/data/FM_data")
DIST = (0.0, 0.5, 1.0, 2.0, 3.0, 5.0, 8.0, 12.0)
rng = np.random.default_rng(0)
dirs = rng.normal(size=(16, 3))
dirs /= np.linalg.norm(dirs, axis=1, keepdims=True)
sizes = np.load(Path(__file__).with_name("box_sizes.npy"))


def argmax_host(seg, box):
    """Largest intersection-over-box over every label present; None if the box is all background."""
    b = clip_box(box, seg.shape)
    sub = seg[b[0]:b[3], b[1]:b[4], b[2]:b[5]]
    if not sub.size:
        return None
    lab, cnt = np.unique(sub[sub > 0], return_counts=True)
    return int(lab[cnt.argmax()]) if len(lab) else None


def curve(cases, host):
    tally = {d: Counter() for d in DIST}
    for seg, sp, boxes, cands in cases:
        for box, cand in zip(boxes, cands):
            ref = host(seg, box, cand)
            if ref is None:
                continue
            for d in DIST:
                for u in (dirs if d else dirs[:1]):
                    moved = box + np.tile(u * d / np.asarray(sp), 2)
                    tally[d]["flip" if host(seg, moved, cand) != ref else "same"] += 1
    return [tally[d]["flip"] / max(sum(tally[d].values()), 1) for d in DIST]


brain = []
for p in sorted((FM / "derived/synthseg/pdgm/seg_native").glob("*.nii.gz"))[:30]:
    im = nib.load(str(p))
    seg = np.asarray(im.dataobj).astype(np.int32)
    sp = [float(z) for z in im.header.get_zooms()[:3]]
    inside = np.array(np.nonzero(seg > 0)).T
    boxes = []
    for _ in range(20):
        c = inside[rng.integers(len(inside))]
        half = sizes[rng.integers(len(sizes))] / 2 / np.asarray(sp)
        boxes.append(np.concatenate([c - half, c + half]))
    brain.append((seg, sp, boxes, [None] * len(boxes)))

knee = []
for scan in sorted(json.loads((FM / "derived/skmtea/m1r/splits.json").read_text())["folds"]):
    rows = [r for r in read_all_boxes(FM / "derived/skmtea/m1r" / scan / "boxes.csv")
            if r["layer"] == "in_seg" and r["cls"] in CANDIDATES]
    if not rows:
        continue
    seg = np.asarray(load_array(FM / "derived/skmtea/m1r_cache" / scan, "seg"))
    sp = load_meta(FM / "derived/skmtea/m1r_cache" / scan)["spacing_mm"]
    knee.append((seg, sp, [np.asarray(r["box"], float) for r in rows], [r["cls"] for r in rows]))

restricted = lambda seg, box, cls: b0_host(LabelIndex(seg, [1, 1, 1]), box, cls) if False else None
rows = [
    (f"brain, 32 SynthSeg structures ({sum(len(c[2]) for c in brain)} boxes)", curve(brain, lambda s, b, c: argmax_host(s, b))),
    (f"knee, 6 structures, no class restriction ({sum(len(c[2]) for c in knee)} lesions)", curve(knee, lambda s, b, c: argmax_host(s, b))),
]
print(f"{'host changes when the box moves by':52}" + "".join(f"{d:>7.1f}" for d in DIST) + "  mm")
for name, c in rows:
    print(f"{name:52}" + "".join(f"{x:7.3f}" for x in c))
