"""Out-of-fold evaluation of the brain multi-disease detectors (spec 2026-09-29 §5, §6): ground-truth and predicted
components with their binding, the D1-style gate per disease, size strata, nnU-Net's Dice, and the agreement of the
binding between matched pairs (NOT_EVIDENCE)."""
import json
from pathlib import Path

import nibabel as nib
import numpy as np

from anatobind.bind.brain_lookup import BrainBinder
from anatobind.eval.detection_metrics import gate, scan_matches, sweep
from anatobind.eval.lesion_boxes import load_label_map, load_nnunet_probabilities
from anatobind.eval.lesion_components import component_rows, components
from anatobind.infer.brain_disease import bind_rows, detections
from anatobind.nnunet.brain_disease import DISEASES, fold_dir

EARLY_STOP = 0.3
N_FOLDS = 5


def case_scan(job):
    """(case, disease, label, prediction nii, prediction npz, anatomy, voxel_mm3) -> {"case", "gt", "dets"}; every
    row carries its binding."""
    case, disease, label_p, nii, npz, anatomy, voxel_mm3 = job
    family = DISEASES[disease]["type"]
    label, pred = load_label_map(label_p), load_label_map(nii)
    seg_img = nib.load(str(anatomy))
    if label.shape != pred.shape or label.shape != seg_img.shape:
        raise ValueError(f"{case}: label {label.shape}, prediction {pred.shape} and anatomy {seg_img.shape} differ")
    binder = BrainBinder(np.asarray(seg_img.dataobj), seg_img.header.get_zooms()[:3])
    comp, n = components(label)
    gt = bind_rows(component_rows(comp, n, voxel_mm3, family), comp, binder)
    dets, dcomp = detections(pred, load_nnunet_probabilities(npz, pred), voxel_mm3, family)
    return {"case": case, "gt": gt, "dets": bind_rows(dets, dcomp, binder)}


def jobs(results_root, raw_root, disease, splits, folds, info, anatomy_of):
    out = []
    for f in folds:
        d = fold_dir(results_root, disease, f) / "validation"
        for case in splits[f]["val"]:
            nii, npz = d / f"{case}.nii.gz", d / f"{case}.npz"
            for p in (nii, npz):
                if not p.exists():
                    raise FileNotFoundError(f"fold {f} case {case}: missing {p}")
            out.append((case, disease, Path(raw_root) / DISEASES[disease]["name"] / "labelsTr" / f"{case}.nii.gz", nii, npz,
                        anatomy_of(case), info[case]["voxel_mm3"]))
    return out


def evaluate(scans):
    rows = sweep(scans)
    return {"rows": rows, "gate": gate(rows), "n_scans": len(scans), "n_gt": rows[0]["n_gt"],
            "n_ignored": sum(1 for s in scans for r in s["gt"] if r.get("ignore"))}


def verdict(result, folds):
    """The gate when all five folds were scored; otherwise an early reading with the early-stop flag (spec M4)."""
    full = sorted(folds) == list(range(N_FOLDS))
    sens = result["gate"]["sensitivity_family"]
    return {"kind": "gate" if full else "early_reading", "folds": sorted(folds), "pass": result["gate"]["pass"] if full else None,
            "sensitivity": sens, "thr": result["gate"]["thr"], "fp_per_scan": result["gate"]["fp_per_scan"],
            "stop_remaining_folds": bool(not full and sens < EARLY_STOP)}


def strata(scans, thr, key):
    """Sensitivity per stratum over the ground truth that is not ignored; key(scan, row) names the stratum."""
    out = {}
    for s in scans:
        hits, _, _ = scan_matches(s, thr)
        for g, r in enumerate(s["gt"]):
            if r.get("ignore"):
                continue
            e = out.setdefault(key(s, r), {"n_gt": 0, "n_hit": 0})
            e["n_gt"] += 1
            e["n_hit"] += int(g in hits)
    for e in out.values():
        e["sensitivity"] = e["n_hit"] / e["n_gt"]
    return out


def binding_agreement(scans, thr):
    """NOT_EVIDENCE. Over matched (ground truth, detection) pairs: the share with the same main structure and the
    share with the same side; over all detections at thr: the share without any host."""
    n = same_host = same_side = n_det = no_host = 0
    for s in scans:
        hits, _, dets = scan_matches(s, thr)
        n_det += len(dets)
        no_host += sum(1 for d in dets if d["host"] is None)
        for g, p in hits.items():
            n += 1
            same_host += int(s["gt"][g]["host"] == dets[p]["host"])
            same_side += int(s["gt"][g]["side"] == dets[p]["side"])
    return {"n_pairs": n, "host_agreement": same_host / n if n else None, "side_agreement": same_side / n if n else None,
            "n_detections": n_det, "no_host_rate": no_host / n_det if n_det else None}


def dice_summary(results_root, disease, folds):
    """Foreground Dice of the validation cases that have ground truth, from nnU-Net's own summary.json."""
    vals = []
    for f in folds:
        s = json.loads((fold_dir(results_root, disease, f) / "validation" / "summary.json").read_text())
        vals += [c["metrics"]["1"]["Dice"] for c in s["metric_per_case"] if c["metrics"]["1"]["n_ref"] > 0]
    return {"n_cases": len(vals), "mean": float(np.mean(vals)) if vals else None,
            "median": float(np.median(vals)) if vals else None}
