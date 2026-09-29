"""Out-of-fold evaluation of the brain multi-disease detectors (spec 2026-09-29 §5, §6): ground-truth and predicted
components with their binding, the D1-style gate per disease, size strata, nnU-Net's Dice, and the agreement of the
binding between matched pairs (NOT_EVIDENCE)."""
import json
from pathlib import Path

import nibabel as nib
import numpy as np

from anatobind.bind.brain_lookup import BrainBinder
from anatobind.eval.detection_metrics import FP_MAX, gate, scan_matches, sweep
from anatobind.eval.lesion_boxes import load_label_map, load_nnunet_probabilities
from anatobind.eval.lesion_components import component_rows, components
from anatobind.infer.brain_disease import NEAR_MM, bind_rows, detections
from anatobind.nnunet.brain_disease import DISEASES, fold_dir

EARLY_STOP = 0.3
REACH_LESIONS = 2
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
    if len(set(folds)) != len(folds) or not set(folds) <= set(range(N_FOLDS)):
        raise ValueError(f"folds must be distinct and within 0..{N_FOLDS - 1}: {list(folds)}")
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


def beyond_budget(rows, fp_max=FP_MAX):
    """The row with the highest threshold whose false positives exceed the budget, or None. The grid is coarse: the
    threshold that just meets the budget lies between this row and the operating point."""
    over = [r for r in rows if r["fp_per_scan"] > fp_max]
    return max(over, key=lambda r: r["thr"]) if over else None


def within_reach(row):
    """Could a threshold near this row reach EARLY_STOP? Yes when the row reaches it or misses it by at most
    REACH_LESIONS lesions: the matching is redone at every threshold, so the hits need not fall as the threshold
    rises, and a threshold between two rows of the grid can find a lesion or two more than the lower row."""
    return row["n_hit_family"] + REACH_LESIONS >= EARLY_STOP * row["n_gt"]


def verdict(result, folds):
    """The gate when all five folds were scored; otherwise an early reading with the early-stop flag (spec M4).

    Without an operating point (no threshold keeps the false positives within the budget) no sensitivity was measured:
    it is None and the gate fails. An early reading stops the remaining folds only when it is clear: the sensitivity
    at the operating point is under EARLY_STOP, and the row just beyond the budget is out of reach of it too (see
    within_reach). Every other low reading is undecided: the remaining folds go on and the user decides. A reading
    without an operating point is always undecided; out_of_reach then says whether it is decisive in substance."""
    full = sorted(folds) == list(range(N_FOLDS))
    g = result["gate"]
    found = g["thr"] is not None
    sens = g["sensitivity_family"] if found else None
    b = beyond_budget(result.get("rows", []))
    low = found and sens < EARLY_STOP
    reach = b is not None and within_reach(b)
    return {"kind": "gate" if full else "early_reading", "folds": sorted(folds), "pass": g["pass"] if full else None,
            "operating_point": found, "sensitivity": sens, "thr": g["thr"], "fp_per_scan": g["fp_per_scan"],
            "beyond_budget": None if b is None else {"thr": b["thr"], "sensitivity": b["sensitivity_family"],
                                                     "fp_per_scan": b["fp_per_scan"], "n_hit": b["n_hit_family"],
                                                     "n_gt": b["n_gt"], "out_of_reach": not reach},
            "stop_remaining_folds": bool(not full and low and not reach),
            "early_stop_undecided": bool(not full and (not found or (low and reach)))}


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


def false_positive_spread(scans, thr):
    """How the false positives at thr spread over the scans: the budget is a mean, a few scans can carry it."""
    fps = [scan_matches(s, thr)[1] for s in scans]
    return {"n_scans": len(fps), "median": float(np.median(fps)) if fps else None, "max": int(max(fps)) if fps else None,
            "n_scans_over_budget": sum(1 for f in fps if f > FP_MAX)}


def binding_agreement(scans, thr):
    """NOT_EVIDENCE. Over matched (ground truth, detection) pairs: the share with the same main structure, with the
    same side of the whole lesion and with the same side of the main structure (the one the sentence writes); over all
    detections at thr: the share without any host, the share that overlaps no structure and was bound to the nearest
    one, and the share that the sentence calls not located (no host, or the nearest one beyond NEAR_MM)."""
    n = same_host = same_side = same_host_side = n_det = no_host = nearest = unlocated = 0
    for s in scans:
        hits, _, dets = scan_matches(s, thr)
        n_det += len(dets)
        no_host += sum(1 for d in dets if d["host"] is None)
        nearest += sum(1 for d in dets if d["host_rule"] == "nearest")
        unlocated += sum(1 for d in dets if d["host"] is None
                         or (d["host_rule"] == "nearest" and d["host_distance_mm"] > NEAR_MM))
        for g, p in hits.items():
            n += 1
            same_host += int(s["gt"][g]["host"] == dets[p]["host"])
            same_side += int(s["gt"][g]["side"] == dets[p]["side"])
            same_host_side += int(s["gt"][g]["host_side"] == dets[p]["host_side"])
    return {"n_pairs": n, "host_agreement": same_host / n if n else None, "side_agreement": same_side / n if n else None,
            "host_side_agreement": same_host_side / n if n else None,
            "n_detections": n_det, "no_host_rate": no_host / n_det if n_det else None,
            "nearest_rate": nearest / n_det if n_det else None, "unlocated_rate": unlocated / n_det if n_det else None}


def dice_summary(results_root, disease, folds):
    """Foreground Dice of the validation cases that have ground truth, from nnU-Net's own summary.json."""
    vals = []
    for f in folds:
        s = json.loads((fold_dir(results_root, disease, f) / "validation" / "summary.json").read_text())
        vals += [c["metrics"]["1"]["Dice"] for c in s["metric_per_case"] if c["metrics"]["1"]["n_ref"] > 0]
    return {"n_cases": len(vals), "mean": float(np.mean(vals)) if vals else None,
            "median": float(np.median(vals)) if vals else None}
