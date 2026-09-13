"""Is the decision margin large because the lesions are large?

Displacement needed to flip the host, as a function of box size. Knee uses the real lesions binned by
size; brain uses PDGM SynthSeg anatomy with synthetic boxes swept from 3 mm to 24 mm a side, since
the question is whether small lesions sit in genuinely ambiguous places.
"""
import json
from collections import Counter
from pathlib import Path

import nibabel as nib
import numpy as np

from anatobind.eval.lookup import CANDIDATES, LabelIndex, b0_host, clip_box
from anatobind.train.cache import load_array, load_meta
from anatobind.train.dataset_v2 import read_all_boxes

FM = Path("/data2/congcong/data/FM_data")
DIST = (0.5, 1.0, 2.0, 3.0, 5.0, 8.0)
rng = np.random.default_rng(0)
dirs = rng.normal(size=(16, 3))
dirs /= np.linalg.norm(dirs, axis=1, keepdims=True)


def argmax_host(seg, box):
    b = clip_box(box, seg.shape)
    sub = seg[b[0]:b[3], b[1]:b[4], b[2]:b[5]]
    lab, cnt = (np.unique(sub[sub > 0], return_counts=True) if sub.size else (np.array([]), None))
    return int(lab[cnt.argmax()]) if len(lab) else None


def flip_rate(cases, host):
    out = []
    for d in DIST:
        t = Counter()
        for seg, sp, boxes, cands in cases:
            for box, cand in zip(boxes, cands):
                ref = host(seg, box, cand)
                if ref is None:
                    continue
                for u in dirs:
                    t["flip" if host(seg, box + np.tile(u * d / np.asarray(sp), 2), cand) != ref else "same"] += 1
        out.append(t["flip"] / max(sum(t.values()), 1))
    return out


# knee: real lesions, class-restricted, split by box volume
knee = []
for scan in sorted(json.loads((FM / "derived/skmtea/m1r/splits.json").read_text())["folds"]):
    rows = [r for r in read_all_boxes(FM / "derived/skmtea/m1r" / scan / "boxes.csv")
            if r["layer"] == "in_seg" and r["cls"] in CANDIDATES]
    if not rows:
        continue
    seg = np.asarray(load_array(FM / "derived/skmtea/m1r_cache" / scan, "seg"))
    sp = np.asarray(load_meta(FM / "derived/skmtea/m1r_cache" / scan)["spacing_mm"])
    for r in rows:
        b = np.asarray(r["box"], float)
        knee.append((seg, sp, [b], [r["cls"]], float(np.prod((b[3:] - b[:3]) * sp)) ** (1 / 3)))
edges = np.percentile([c[4] for c in knee], [33, 67])
host_r = lambda seg, box, cls: b0_host(LabelIndex(seg, [1, 1, 1]), box, cls)
print(f"{'knee, real lesions, class-restricted':46}" + "".join(f"{d:>7.1f}" for d in DIST) + "  mm")
for name, sel in (("small third", lambda s: s <= edges[0]), ("middle third", lambda s: edges[0] < s <= edges[1]),
                  ("large third", lambda s: s > edges[1])):
    sub = [c[:4] for c in knee if sel(c[4])]
    sizes = [c[4] for c in knee if sel(c[4])]
    r = flip_rate([(s, sp, bs, cs) for s, sp, bs, cs in sub],
                  lambda seg, box, cls: b0_host(LabelIndex(seg, [1, 1, 1]), box, cls))
    print(f"  {name} (side {np.median(sizes):.0f} mm, n={len(sub)}){'':<8}" + "".join(f"{x:7.3f}" for x in r))

# brain: synthetic boxes of increasing size on SynthSeg anatomy
print(f"\n{'brain, synthetic boxes, 32 SynthSeg structures':46}" + "".join(f"{d:>7.1f}" for d in DIST) + "  mm")
segs = []
for p in sorted((FM / "derived/synthseg/pdgm/seg_native").glob("*.nii.gz"))[:20]:
    im = nib.load(str(p))
    segs.append((np.asarray(im.dataobj).astype(np.int32), np.asarray([float(z) for z in im.header.get_zooms()[:3]])))
for side in (3, 6, 12, 24):
    cases = []
    for seg, sp in segs:
        inside = np.array(np.nonzero(seg > 0)).T
        boxes = []
        for _ in range(25):
            c = inside[rng.integers(len(inside))]
            half = np.full(3, side) / 2 / sp
            boxes.append(np.concatenate([c - half, c + half]))
        cases.append((seg, sp, boxes, [None] * len(boxes)))
    r = flip_rate(cases, lambda seg, box, cls: argmax_host(seg, box))
    print(f"  {side:2d} mm a side (n={sum(len(c[2]) for c in cases)}){'':<18}" + "".join(f"{x:7.3f}" for x in r))
