"""Read the nndet runner's JSON into the repository's boxes and lesion tables (spec 2026-09-28 brain-nndet §5, §8)."""
import json
from pathlib import Path

import numpy as np

from anatobind.nnunet.brain_lesion import FAMILY

LAYOUT = "nndet [s_lo, r_lo, s_hi, r_hi, c_lo, c_hi] on original array axes (slice, row, col), half-open"


def to_corner_box(b):
    """[s_lo, r_lo, s_hi, r_hi, c_lo, c_hi] -> (x0, y0, z0, x1, y1, z1) on the (col, row, slice) grid, half-open."""
    s0, r0, s1, r1, c0, c1 = (float(v) for v in b)
    return (c0, r0, s0, c1, r1, s1)


def load_runner_json(path):
    d = json.loads(Path(path).read_text())
    if d.get("layout") != LAYOUT:
        raise ValueError(f"{path}: layout {d.get('layout')!r} is not the runner's {LAYOUT!r}")
    return d


def detections(case_rec):
    boxes, scores = case_rec["boxes"], case_rec["scores"]
    labels = case_rec.get("labels", [0] * len(boxes))
    if len(boxes) != len(scores) or len(boxes) != len(labels):
        raise ValueError(f"boxes, scores and labels differ in length ({len(boxes)}, {len(scores)}, {len(labels)})")
    if any(int(v) != 0 for v in labels):
        raise ValueError(f"only class 0 is expected, found class {sorted({int(v) for v in labels})}")
    return [{"box": to_corner_box(b), "score": float(s), "family": FAMILY} for b, s in zip(boxes, scores)]


def _half_up(v):
    return int(np.floor(v + 0.5))


def _span(lo, hi, n):
    a, b = _half_up(lo), _half_up(hi)
    b = max(b, a + 1)
    a, b = max(0, a), min(n, b)
    return (a, b) if b > a else None


def lesion_rows(dets, shape):
    """Scored 3D boxes -> S2 lesions.json rows: z0..z1 inclusive, per-slice [row0, row1, col0, col1] = the box's
    in-plane rectangle on every slice; coordinates rounded half up, at least one voxel per axis, clipped to the
    (col, row, slice) grid; a box fully outside is dropped; rows sorted by score, highest first."""
    nc, nr, ns = shape
    out = []
    for d in sorted(dets, key=lambda d: -d["score"]):
        x0, y0, z0, x1, y1, z1 = d["box"]
        cs, rs, ss = _span(x0, x1, nc), _span(y0, y1, nr), _span(z0, z1, ns)
        if cs is None or rs is None or ss is None:
            continue
        out.append({"z0": ss[0], "z1": ss[1] - 1, "score": float(d["score"]),
                    "boxes": {str(s): [[rs[0], rs[1], cs[0], cs[1]]] for s in range(ss[0], ss[1])}})
    return out
