"""Out-of-fold evaluation of the brain small-lesion detector (spec 2026-09-28 §4): decode nnU-Net validation outputs
into scored 3D boxes, sweep thresholds, and read the D1 gate; normal-volume false positives and strata are reported,
not gated."""
from anatobind.eval.detection_metrics import gate, match_scan, sweep
from anatobind.eval.lesion_boxes import BRAIN_MIN_VOXELS, decode_boxes, load_label_map, load_nnunet_probabilities
from anatobind.nnunet.brain_lesion import FAMILIES, validation_npz_path, validation_path


def scan_record(case, gt, label_map, probs):
    return {"case": case, "gt": gt, "dets": decode_boxes(label_map, probs, BRAIN_MIN_VOXELS, FAMILIES)}


def collect(results_root, config, splits, gt_of_case):
    scans = []
    for fold, split in enumerate(splits):
        for case in split["val"]:
            nii, npz = validation_path(results_root, config, fold, case), validation_npz_path(results_root, config, fold, case)
            if not nii.exists() or not npz.exists():
                raise FileNotFoundError(f"fold {fold} case {case}: missing {nii if not nii.exists() else npz}")
            lab = load_label_map(nii)
            scans.append(scan_record(case, gt_of_case[case], lab, load_nnunet_probabilities(npz, lab)))
    return scans


def normal_fp_per_scan(scans, normal_cases, thr):
    normal = [s for s in scans if s["case"] in normal_cases]
    return sum(sum(1 for d in s["dets"] if d["score"] >= thr) for s in normal) / len(normal) if normal else 0.0


def strata_sensitivity(scans, thr, stratum_of):
    out = {}
    for s in scans:
        dets = [d for d in s["dets"] if d["score"] >= thr]
        pairs = match_scan(s["gt"], dets)
        for g, r in enumerate(s["gt"]):
            k = stratum_of[r["lesion_id"]]
            e = out.setdefault(k, {"n_gt": 0, "n_hit": 0})
            e["n_gt"] += 1
            e["n_hit"] += int(g in pairs)
    for e in out.values():
        e["sensitivity"] = e["n_hit"] / e["n_gt"]
    return out


def evaluate(scans, normal_cases):
    rows = sweep(scans)
    g = gate(rows)
    thr = g["thr"]
    return {"rows": rows, "gate": g, "normal_fp_per_scan": normal_fp_per_scan(scans, normal_cases, thr) if thr is not None else None,
            "n_scans": len(scans), "n_gt": sum(len(s["gt"]) for s in scans)}
