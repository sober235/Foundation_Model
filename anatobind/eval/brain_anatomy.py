"""Evaluation of the S4 anatomy model (spec 2026-10-02 §7, A11, A13).

All numbers are agreement with SynthSeg pseudo-labels, NOT_EVIDENCE. On the fastMRI stacks only the reliable slices
count (the same slices that supervise the outline model: from slice 2 up to the slice below the last one with brain);
the host agreement is the share of fastMRI+ lesion boxes for which the box lookup on the student's anatomy names the
same host class as the lookup on SynthSeg's anatomy."""
import numpy as np

from anatobind.anatomy.labels import HOST_IDS, NAMES, to_synthseg
from anatobind.anatomy.outline import dice as mask_dice, fill_and_keep_largest, supervised_slices
from anatobind.eval.geometry import CLASS_NAMES, host_class_map
from anatobind.eval.lookup import BRAIN_PARENCHYMA, BrainLookup

GATES = {"host_agreement": 0.90, "mean_host_dice": 0.80, "outline_dice": 0.97}      # A11 (1), (2), (3)


def reliable_slices(synthseg, voxel_area_mm2):
    """The slices of a fastMRI stack whose SynthSeg labels are trusted: outline rule of the skull-strip model."""
    return supervised_slices(fill_and_keep_largest(np.asarray(synthseg) > 0), voxel_area_mm2)


def class_dice(pred, ref, classes=HOST_IDS, slices=None):
    """{class id: Dice} over the given slices (all when None); None for a class empty in both maps."""
    p, r = np.asarray(pred), np.asarray(ref)
    if slices is not None:
        p, r = p[:, :, list(slices)], r[:, :, list(slices)]
    out = {}
    for c in classes:
        a, b = p == c, r == c
        s = int(a.sum()) + int(b.sum())
        out[int(c)] = None if s == 0 else 2.0 * float((a & b).sum()) / s
    return out


def summarize_dice(per_case):
    """per_case: list of {class id: Dice | None}. Mean per class over the cases where the class occurs, and the mean of
    those per-class means over HOST_IDS (mean_host_dice)."""
    per_class = {}
    for c in HOST_IDS:
        vals = [d[c] for d in per_case if d.get(c) is not None]
        per_class[NAMES[c]] = float(np.mean(vals)) if vals else None
    host = [v for v in per_class.values() if v is not None]
    return {"per_class": per_class, "mean_host_dice": float(np.mean(host)) if host else None, "n_cases": len(per_case)}


def rects_of(row):
    """Registry row (x0, x1 half-open columns; y0, y1 half-open rows; z0..z1 inclusive slices) -> lookup rectangles."""
    return [(int(row["x0"]), int(row["x1"]), int(row["y0"]), int(row["y1"]), s) for s in range(int(row["z0"]), int(row["z1"]) + 1)]


def box_host(lookup, rects):
    """Host class name of a box by argmax overlap over the parenchyma labels, nearest when no overlap; None off the grid."""
    label, _ = lookup.host(rects)
    if label is None:
        return None
    cls = int(host_class_map(np.array([label]))[0])
    return CLASS_NAMES[cls - 1] if cls > 0 else None


def host_agreement(student, synthseg, spacing, rows, reliable):
    """student: compact labels; synthseg: SynthSeg map on the same grid; rows: registry rows of this stack.
    Boxes with a slice outside `reliable` are counted but not evaluated."""
    a = BrainLookup(to_synthseg(student), spacing, BRAIN_PARENCHYMA)
    b = BrainLookup(np.asarray(synthseg), spacing, BRAIN_PARENCHYMA)
    n_eval = n_agree = n_outside = 0
    pairs = []
    for row in rows:
        if any(s not in reliable for s in range(int(row["z0"]), int(row["z1"]) + 1)):
            n_outside += 1
            continue
        rects = rects_of(row)
        hs, hr = box_host(a, rects), box_host(b, rects)
        n_eval += 1
        n_agree += int(hs == hr)
        pairs.append({"lesion_id": int(row["lesion_id"]), "student": hs, "synthseg": hr})
    return {"n_lesions": len(rows), "n_evaluated": n_eval, "n_agree": n_agree, "n_outside_reliable": n_outside,
            "rate": (n_agree / n_eval) if n_eval else None, "pairs": pairs}


def pool_agreement(parts):
    n_eval = sum(p["n_evaluated"] for p in parts)
    n_agree = sum(p["n_agree"] for p in parts)
    return {"n_lesions": sum(p["n_lesions"] for p in parts), "n_evaluated": n_eval, "n_agree": n_agree,
            "n_outside_reliable": sum(p["n_outside_reliable"] for p in parts), "rate": (n_agree / n_eval) if n_eval else None}


def outline_dice(pred_mask, synthseg, voxel_area_mm2):
    """Dice of a predicted brain mask against the SynthSeg outline on the reliable slices."""
    ref = fill_and_keep_largest(np.asarray(synthseg) > 0)
    sl = list(reliable_slices(synthseg, voxel_area_mm2))
    if not sl:
        return None
    return mask_dice(np.asarray(pred_mask)[:, :, sl] > 0, ref[:, :, sl])


def verdict(host_rate, mean_host_dice, outline):
    """A11: the three lines must all pass; a missing number fails its line."""
    checks = {"host_agreement": host_rate is not None and host_rate >= GATES["host_agreement"],
              "mean_host_dice": mean_host_dice is not None and mean_host_dice >= GATES["mean_host_dice"],
              "outline_dice": outline is not None and outline >= GATES["outline_dice"]}
    return {"pass": all(checks.values()), **checks, "values": {"host_agreement": host_rate, "mean_host_dice": mean_host_dice,
                                                                "outline_dice": outline}, "gates": dict(GATES)}
