"""nnDetection second arm evaluation (spec 2026-09-28 brain-nndet §5, §7): runner JSON -> scans in S2's format, rule A
against nnU-Net on the same folds, the paired table, and the ground-truth-as-prediction coordinate check."""
import math

from anatobind.eval.brain_detector import scan_record
from anatobind.eval.detection_metrics import IOU, match_scan, operating_point
from anatobind.eval.lesion_boxes import load_label_map, load_nnunet_probabilities
from anatobind.eval.matching import iou3d
from anatobind.nndet.boxes import detections, to_corner_box
from anatobind.nnunet.brain_lesion import gt_boxes, validation_npz_path, validation_path

RULE_A_MARGIN = 0.05


def gt_of_cases(case_kinds, registry):
    by_file = {}
    for r in registry:
        by_file.setdefault(r["file"], []).append(r)
    out = {}
    for case, kind in case_kinds.items():
        if kind == "lesion":
            if case not in by_file:
                raise ValueError(f"lesion case {case} has no registry rows")
            out[case] = gt_boxes(by_file[case])
        else:
            out[case] = []
    return out


def nndet_scans(cases_json, case_ids, gt_of_case):
    missing = [c for c in case_ids if c not in cases_json]
    if missing:
        raise KeyError(f"runner JSON lacks {len(missing)} cases: {missing[:5]}")
    return [{"case": c, "gt": gt_of_case[c], "dets": detections(cases_json[c])} for c in case_ids]


def nnunet_scans(results_root, config, splits, folds, gt_of_case):
    scans = []
    for f in folds:
        for case in splits[f]["val"]:
            nii, npz = validation_path(results_root, config, f, case), validation_npz_path(results_root, config, f, case)
            if not nii.exists() or not npz.exists():
                raise FileNotFoundError(f"fold {f} case {case}: missing {nii if not nii.exists() else npz}")
            lab = load_label_map(nii)
            scans.append(scan_record(case, gt_of_case[case], lab, load_nnunet_probabilities(npz, lab)))
    return scans


def rule_a(rows_nndet, rows_base, n_gt, margin=RULE_A_MARGIN):
    """Spec N3: continue iff nnDetection's hits at its <= 2 FP/scan operating point reach the baseline's hits at its
    own operating point plus ceil(margin * n_gt)."""
    op_n, op_b = operating_point(rows_nndet), operating_point(rows_base)
    base_hits = op_b["n_hit"] if op_b else 0
    need = base_hits + math.ceil(margin * n_gt - 1e-9)
    hits = op_n["n_hit"] if op_n else 0
    return {"pass": bool(hits >= need), "nndet_hits": hits, "baseline_hits": base_hits, "required": need, "n_gt": n_gt,
            "margin": margin, "nndet_thr": op_n["thr"] if op_n else None, "baseline_thr": op_b["thr"] if op_b else None}


def found_lesions(scans, thr):
    out = set()
    for s in scans:
        dets = [d for d in s["dets"] if d["score"] >= thr]
        out |= {s["gt"][g]["lesion_id"] for g in match_scan(s["gt"], dets)}
    return out


def paired_table(scans_a, thr_a, scans_b, thr_b):
    ids = {r["lesion_id"] for s in scans_a for r in s["gt"]}
    if ids != {r["lesion_id"] for s in scans_b for r in s["gt"]}:
        raise ValueError("the two scan sets do not hold the same lesions")
    a, b = found_lesions(scans_a, thr_a), found_lesions(scans_b, thr_b)
    return {"both": len(a & b), "only_a": len(a - b), "only_b": len(b - a), "neither": len(ids - a - b),
            "n_gt": len(ids)}


def gt_check(cases_json, instance_map, gt_of_case):
    """Spec §5 check two: every instance present after preprocessing must overlap its own registry box with
    IoU >= IOU. Returns {"iou": {lesion_id: IoU}, "lost": [...], "below_iou": [...]}."""
    ious, lost, below = {}, [], []
    for case, gts in gt_of_case.items():
        rec = cases_json.get(case, {"boxes": [], "scores": [], "instances": []})
        present = {int(k): b for k, b in zip(rec["instances"], rec["boxes"])}
        lid_of = {int(k): int(v) for k, v in instance_map.get(case, {}).items()}
        extra = sorted(set(present) - set(lid_of))
        if extra:
            raise ValueError(f"{case}: instances {extra} are not in the instance map")
        reg = {r["lesion_id"]: r["box"] for r in gts}
        if sorted(reg) != sorted(lid_of.values()):
            raise ValueError(f"{case}: registry lesions {sorted(reg)} != instance map lesions {sorted(lid_of.values())}")
        for k, lid in lid_of.items():
            if k not in present:
                lost.append(lid)
                continue
            ious[lid] = float(iou3d([to_corner_box(present[k])], [reg[lid]])[0, 0])
            if ious[lid] < IOU:
                below.append(lid)
    return {"iou": ious, "lost": sorted(lost), "below_iou": sorted(below)}
