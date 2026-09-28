#!/usr/bin/env python
# scripts/diagnose_brain_detector.py
"""Read-only diagnostic of the brain small-lesion detector's misses (final review Important 2). NOT the D1 gate
(see scripts/eval_brain_detector.py for that): this only characterises, for each registry lesion, whether any
predicted component overlaps its box, and how often one predicted component spans several lesion boxes.

Two populations of predicted components are reported side by side, since they answer different questions:
'all' is every 26-connected component of the argmax foreground (what the network predicted before any size
filtering); 'decoded' is only components with >= BRAIN_MIN_VOXELS voxels (what scripts/eval_brain_detector.py
actually scores as detections).

  source scripts/nnunet_env.sh
  PYTHONPATH=. python scripts/diagnose_brain_detector.py --config 2d --out /path/to/file.txt
"""
import argparse
import json
import os
import sys
from collections import Counter
from pathlib import Path

import numpy as np
from scipy import ndimage

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from anatobind.eval.lesion_boxes import BRAIN_MIN_VOXELS, STRUCTURE, load_label_map  # noqa: E402
from anatobind.level_r.registry import load_registry  # noqa: E402
from anatobind.nnunet.brain_lesion import DATASET_NAME, gt_boxes, validation_path  # noqa: E402

IOU_THRESHOLD = 0.1
EXPECTED_N_GT = 1297
POPULATIONS = (("all", 1), ("decoded", BRAIN_MIN_VOXELS))


def label_map_components(label_map):
    """26-connected components of the argmax foreground (label_map == 1): voxel count and bounding box per id."""
    comp, n = ndimage.label(label_map == 1, structure=STRUCTURE)
    sizes = np.bincount(comp.ravel(), minlength=n + 1)
    boxes = {}
    for k, sl in enumerate(ndimage.find_objects(comp), start=1):
        if sl is not None:
            boxes[k] = (sl[0].start, sl[1].start, sl[2].start, sl[0].stop, sl[1].stop, sl[2].stop)
    return comp, sizes, boxes


def box_iou(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    lo, hi = np.maximum(a[:3], b[:3]), np.minimum(a[3:], b[3:])
    inter = float(np.clip(hi - lo, 0, None).prod())
    union = float((a[3:] - a[:3]).prod() + (b[3:] - b[:3]).prod() - inter)
    return inter / union if union else 0.0


def touching_components(comp, sizes, box, min_voxels):
    """Ids of components with >= min_voxels voxels that have >= 1 voxel inside box."""
    x0, y0, z0, x1, y1, z1 = (int(v) for v in box)
    sub = comp[x0:x1, y0:y1, z0:z1]
    return {int(k) for k in np.unique(sub) if k and sizes[k] >= min_voxels}


def diagnose_case(label_map, lesion_boxes, min_voxels):
    """One case's registry lesion boxes against its own label map.

    Returns (per_lesion, component_hits): per_lesion is a list parallel to lesion_boxes, each entry
    {"touched": bool, "iou_below": bool}. component_hits maps component id -> number of this case's
    lesion boxes it touches (component ids are only meaningful within this one case)."""
    comp, sizes, boxes = label_map_components(label_map)
    per_lesion = []
    component_hits = Counter()
    for box in lesion_boxes:
        ks = touching_components(comp, sizes, box, min_voxels)
        per_lesion.append({
            "touched": bool(ks),
            "iou_below": any(box_iou(box, boxes[k]) < IOU_THRESHOLD for k in ks),
        })
        for k in ks:
            component_hits[k] += 1
    return per_lesion, component_hits


def summarize_population(lesion_boxes_of_case, label_map_of_case, min_voxels):
    """Aggregate diagnose_case over all cases for one population (a min_voxels threshold)."""
    touched = iou_below = 0
    hist = Counter()
    for case, lesion_boxes in lesion_boxes_of_case.items():
        per_lesion, component_hits = diagnose_case(label_map_of_case[case], lesion_boxes, min_voxels)
        touched += sum(1 for r in per_lesion if r["touched"])
        iou_below += sum(1 for r in per_lesion if r["iou_below"])
        for v in component_hits.values():
            hist[min(v, 5)] += 1
    return {"touched": touched, "iou_below_0.1": iou_below, "lesions_per_component": dict(sorted(hist.items()))}


def main(argv=None):
    ap = argparse.ArgumentParser(description="Read-only diagnostic of the brain small-lesion detector's misses")
    ap.add_argument("--config", choices=("2d", "3d_fullres"), required=True, help="nnU-Net configuration")
    ap.add_argument("--out", type=Path, required=True, help="Output file (must not exist)")
    a = ap.parse_args(argv)

    if a.out.exists():
        raise FileExistsError(f"--out {a.out} already exists")

    raw_root = Path(os.environ["nnUNet_raw"])
    preprocessed_root = Path(os.environ["nnUNet_preprocessed"])
    results_root = Path(os.environ["nnUNet_results"])

    info = json.loads((raw_root / DATASET_NAME / "cases.json").read_text())
    splits = json.loads((preprocessed_root / DATASET_NAME / "splits_final.json").read_text())
    fold_of_case = {c: fold for fold, split in enumerate(splits) for c in split["val"]}

    registry_of_file = {}
    for r in load_registry():
        registry_of_file.setdefault(r["file"], []).append(r)

    lesion_boxes_of_case = {}
    for case, case_info in info.items():
        if case_info["kind"] != "lesion":
            continue
        lesion_boxes_of_case[case] = [b["box"] for b in gt_boxes(registry_of_file.get(case, []))]

    label_map_of_case = {
        case: load_label_map(validation_path(results_root, a.config, fold_of_case[case], case))
        for case in lesion_boxes_of_case
    }

    lines = []
    for name, min_voxels in POPULATIONS:
        s = summarize_population(lesion_boxes_of_case, label_map_of_case, min_voxels)
        frac = s["touched"] / EXPECTED_N_GT
        label = name if name == "all" else f"{name} (>= {min_voxels} voxels)"
        lines.append(f"GT lesions {EXPECTED_N_GT}; touched by any predicted voxel (population={label}): "
                     f"{s['touched']} ({frac:.3f})")
        lines.append(f"GT lesions whose overlapping component(s) include one with box IoU < {IOU_THRESHOLD} "
                     f"(population={label}): {s['iou_below_0.1']}")
        lines.append(f"GT lesions covered per predicted component (5 = 5+) (population={label}): "
                     f"{s['lesions_per_component']}")
    lines.append("Command: " + " ".join(sys.argv))

    a.out.write_text("\n".join(lines) + "\n")
    print(f"Diagnostic written to {a.out}")


if __name__ == "__main__":
    main()
