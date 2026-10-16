#!/usr/bin/env python3
"""Audit saved validation boxes, without loading images, checkpoints, or models.

Only cases.json and aggregate.json are read from each supplied evaluation directory.
The output contains source/sequence aggregates, never patient or case identifiers.
The existing selected score thresholds are reused, never searched or changed.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path
import shlex
import sys

import numpy as np

from anatobind.eval.detection_metrics import scan_matches
from anatobind.eval.matching import iou3d


def require(condition, message):
    if not condition:
        raise ValueError(message)


def finite_number(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def clean(value):
    """Convert nonfinite values to JSON null and NumPy scalars to Python values."""
    if isinstance(value, dict):
        return {str(k): clean(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [clean(v) for v in value]
    if isinstance(value, np.generic):
        return clean(value.item())
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def distribution(values):
    values = [float(v) for v in values if finite_number(v)]
    if not values:
        return {"n": 0, "min": None, "p10": None, "p25": None,
                "median": None, "p75": None, "p90": None, "max": None}
    qs = np.quantile(values, [0, .1, .25, .5, .75, .9, 1])
    return dict(n=len(values), **dict(zip(
        ["min", "p10", "p25", "median", "p75", "p90", "max"], qs.tolist())))


def ratio(numerator, denominator):
    return numerator / denominator if denominator else None


def key(row):
    # Identifiers remain in memory solely to enforce paired row membership.
    return tuple(row[k] for k in ("source", "sequence", "patient", "case"))


def load_eval(root):
    root = root.resolve()
    require(root.is_dir(), "Evaluation directory does not exist")
    paths = {name: root / name for name in ("cases.json", "aggregate.json")}
    cases = json.loads(paths["cases.json"].read_text())
    aggregate = json.loads(paths["aggregate.json"].read_text())
    require(isinstance(cases, list) and bool(cases), "cases.json must be a nonempty list")
    require(isinstance(aggregate, dict), "aggregate.json must be an object")
    require(aggregate.get("run", {}).get("split") == "val", "Only saved validation evaluations are accepted")
    require(aggregate.get("n_results") == len(cases), "Aggregate row count differs from cases.json")
    seen = set()
    for row in cases:
        require(isinstance(row, dict), "Invalid case row")
        require(all(isinstance(row.get(k), str) and row[k] for k in
                    ("source", "sequence", "patient", "case")), "Missing row identity fields")
        require(key(row) not in seen, "Duplicate case identity; no identifiers are printed")
        seen.add(key(row))
        u = row.get("u")
        if u is None:
            continue
        require(isinstance(u, dict), "Invalid U object")
        for kind in ("gt", "dets"):
            require(isinstance(u.get(kind), list), "Invalid U instance list")
            for item in u[kind]:
                box = item.get("box")
                require(isinstance(box, list) and len(box) == 6 and all(finite_number(v) for v in box),
                        "Box must have six finite coordinates")
                require(all(box[d + 3] > box[d] for d in range(3)), "Box has nonpositive extent")
                require(finite_number(item.get("volume_mm3")) and item["volume_mm3"] >= 0,
                        "Missing or invalid component volume")
                if kind == "dets":
                    require(finite_number(item.get("score")), "Detection score must be finite")
        r = row.get("r")
        if r is not None:
            require(isinstance(r, dict) and isinstance(r.get("instances"), list), "Invalid R object")
    groups = aggregate.get("u", {}).get("per_source_sequence", {})
    require(isinstance(groups, dict), "Missing aggregate U source/sequence table")
    gated = {name: entry for name, entry in groups.items() if entry.get("gated") is True}
    require(bool(gated), "No gated source/sequence entries")
    for entry in gated.values():
        require(finite_number(entry.get("thr")) and 0 <= entry["thr"] <= 1, "Missing selected U threshold")
    provenance = {name: {"path": str(path), "sha256": sha256(path)} for name, path in paths.items()}
    return cases, aggregate, gated, provenance


def check_pair(new, old):
    new_map, old_map = {key(r): r for r in new}, {key(r): r for r in old}
    require(new_map.keys() == old_map.keys(),
            "New/old case membership differs; comparison refused (identifiers withheld)")
    for identity, row in new_map.items():
        other = old_map[identity]
        require(bool(row.get("u")) == bool(other.get("u")), "Paired U availability differs")
        if row.get("u"):
            require(row["u"]["gt"] == other["u"]["gt"], "Paired saved GT definitions differ; comparison refused")
    return {"all_rows": len(new), "same_identity_set": True, "same_saved_gt": True}


def geometry(gt, dets):
    """Per-GT maximum box IoU and coverage by the largest >100 mL component."""
    if not gt:
        return [], 0
    if not dets:
        return [0.] * len(gt), 0
    gboxes = np.asarray([g["box"] for g in gt], float)
    dboxes = np.asarray([d["box"] for d in dets], float)
    ious = iou3d(gboxes, dboxes)
    largest = max(range(len(dets)), key=lambda j: dets[j]["volume_mm3"])
    count = 0
    if dets[largest]["volume_mm3"] > 100000:
        box = dboxes[largest]
        inter = np.clip(np.minimum(gboxes[:, 3:], box[3:]) -
                        np.maximum(gboxes[:, :3], box[:3]), 0, None).prod(-1)
        coverage = inter / (gboxes[:, 3:] - gboxes[:, :3]).prod(-1)
        count = int(((coverage >= .9) & (ious[:, largest] < .1)).sum())
    return ious.max(axis=1).tolist(), count


def summarize(rows, entry):
    thr = entry["thr"]
    maxima = {"raw": [], "kept": []}
    overlaps = {"raw": [], "kept": []}
    bbox_containment = {"raw": 0, "kept": 0}
    n_det = {"raw": 0, "kept": 0}
    gt_count = ignored_count = hit_count = fp_count = 0
    oracle = {"all": [], "missed": [], "hit": []}
    oracle_crop_share = []
    oracle_missing = 0
    for row in rows:
        u = row["u"]
        real_indices = [i for i, g in enumerate(u["gt"]) if not g.get("ignore")]
        gt = [u["gt"][i] for i in real_indices]
        hits, fp, kept = scan_matches(u, thr)
        gt_count += len(gt)
        ignored_count += len(u["gt"]) - len(gt)
        hit_count += len(hits)
        fp_count += fp
        for kind, dets in (("raw", u["dets"]), ("kept", kept)):
            n_det[kind] += len(dets)
            # Empty prediction volumes count as zero in per-volume summaries.
            maxima[kind].append(max((d["volume_mm3"] / 1000 for d in dets), default=0.))
            values, count = geometry(gt, dets)
            overlaps[kind].extend(values)
            bbox_containment[kind] += count
        r_by_id = {i["instance"]: i for i in (row.get("r") or {}).get("instances", [])}
        for gi in real_indices:
            instance = r_by_id.get(u["gt"][gi].get("instance"))
            if instance is None or not finite_number(instance.get("query_iou")):
                oracle_missing += 1
                continue
            oracle["all"].append(instance["query_iou"])
            oracle["hit" if gi in hits else "missed"].append(instance["query_iou"])
            if gi not in hits and finite_number(instance.get("in_crop_share")):
                oracle_crop_share.append(instance["in_crop_share"])
    recomputed = {"n_scans": len(rows), "n_gt": gt_count, "n_ignored": ignored_count,
                  "n_hit": hit_count, "n_fp": fp_count,
                  "sensitivity": ratio(hit_count, gt_count), "fp_per_scan": ratio(fp_count, len(rows))}
    for name, value in recomputed.items():
        original = entry.get(name)
        require(finite_number(original) and math.isclose(original, value, rel_tol=1e-12, abs_tol=1e-12),
                "Recomputed U counts/rates disagree with saved aggregate: " + name)
    return {"selected_score_threshold": thr,
            "threshold_selection_note": "Reused saved validation-selected score threshold; no threshold search performed",
            "saved_instance_probability_threshold": None,
            "recomputed": recomputed, "aggregate_values_verified": True,
            "detections": n_det,
            "per_volume_largest_component_ml": {kind: distribution(values) for kind, values in maxima.items()},
            "volumes_largest_component_gt_100ml": {
                kind: {"n": sum(v > 100 for v in values), "denominator": len(values),
                       "fraction": ratio(sum(v > 100 for v in values), len(values))}
                for kind, values in maxima.items()},
            "per_gt_max_box_iou": {kind: distribution(values) for kind, values in overlaps.items()},
            "per_gt_any_box_iou_ge_0_1": {
                kind: {"n": sum(v >= .1 for v in values), "denominator": len(values),
                       "fraction": ratio(sum(v >= .1 for v in values), len(values))}
                for kind, values in overlaps.items()},
            "largest_gt_100ml_component_bbox_gt_coverage_ge_0_9_and_iou_lt_0_1": bbox_containment,
            "oracle_given_gt_crop_query_iou": {kind: distribution(values) for kind, values in oracle.items()},
            "oracle_missed_given_gt_crop_in_crop_share": distribution(oracle_crop_share),
            "oracle_query_iou_missing_gt": oracle_missing}


def f(value):
    return "null" if value is None else f"{value:.4g}"


def report(result):
    lines = ["# Saved validation U diagnostic", "", "**NOT_EVIDENCE — exploratory saved-validation diagnostic.**", "",
             "The new and old C2 evaluations have identical row identities and saved GT definitions. "
             "Only cases.json and aggregate.json were read from the supplied evaluation directories. "
             "No model, checkpoint, GPU, raw image, sealed test inference, or new threshold search was used.", "",
             "Each arm uses its own already selected validation score threshold. Differences are descriptive; "
             "they do not isolate the training-loss effect from the changed operating point. The saved evaluator "
             "labels validation threshold selection as 'test-selected'; run.split=val is checked explicitly here.", "",
             "| arm | source/sequence | scans | GT | score thr | raw / kept dets | hits / FP | raw ≥.1 box coverage | kept ≥.1 box coverage |",
             "|---|---|---:|---:|---:|---:|---:|---:|---:|"]
    for arm in ("old", "new"):
        for group, s in result[arm].items():
            c, d, cov = s["recomputed"], s["detections"], s["per_gt_any_box_iou_ge_0_1"]
            lines.append(f"| {arm} | {group} | {c['n_scans']} | {c['n_gt']} | {f(s['selected_score_threshold'])} | "
                         f"{d['raw']} / {d['kept']} | {c['n_hit']} / {c['n_fp']} | {cov['raw']['n']} | {cov['kept']['n']} |")
    lines += ["", "Every recomputed scan count, GT count, ignored count, hit count, FP count, sensitivity and "
              "FP/scan equals its original aggregate value. Matching reuses the repository's one-to-one Hungarian "
              "box IoU ≥0.1 rule, with ignored GT matched after counted GT.", "",
              "| arm | source/sequence | raw largest component mL p10 / median / p90 | raw >100 mL volumes | raw / kept bbox containment with IoU <.1 | missed GT oracle query IoU median / p90 |",
              "|---|---|---:|---:|---:|---:|"]
    for arm in ("old", "new"):
        for group, s in result[arm].items():
            v = s["per_volume_largest_component_ml"]["raw"]
            b = s["volumes_largest_component_gt_100ml"]["raw"]
            c = s["largest_gt_100ml_component_bbox_gt_coverage_ge_0_9_and_iou_lt_0_1"]
            q = s["oracle_given_gt_crop_query_iou"]["missed"]
            lines.append(f"| {arm} | {group} | {f(v['p10'])} / {f(v['median'])} / {f(v['p90'])} | "
                         f"{b['n']}/{b['denominator']} | {c['raw']} / {c['kept']} | {f(q['median'])} / {f(q['p90'])} |")
    lines += ["", "## Interpretation and evidence limits", "",
              "Raw here means all saved detections, which are already connected components after the saved "
              "instance probability threshold and component processing. It is not the unthresholded probability map. "
              "The >100 mL statistic is descriptive and is not a new acceptance gate. Empty prediction volumes "
              "contribute zero to the per-volume maximum distribution.", "",
              "Raw ≥0.1 box coverage bounds what score filtering could localize from these saved components. "
              "Low raw coverage cannot be repaired merely by retaining more of those same components. The "
              "raw-to-kept difference describes additional score filtering losses. Coverage allows many GT boxes "
              "to share one predicted box; the actual hit count additionally applies one-to-one matching.", "",
              "The bbox-containment count uses each volume's largest component only when its saved mask volume "
              "exceeds 100 mL: its box covers ≥90% of a GT box while their box IoU is <0.1. This is bbox geometry, "
              "not voxelwise evidence that a predicted mask swallowed a lesion. Saved boxes cannot establish "
              "mask contamination, probability calibration, voxelwise lesion overlap, or the causal role of the U loss.", "",
              "The saved query IoU is a distinct oracle metric (anatobind/aur/infer.py, bind_instances): "
              "(1) each crop is centered on the GT instance's mean voxel position; "
              "(2) all event queries are considered without the presence gate; "
              "(3) each query mask is binarized with probability >0.5; "
              "(4) query_iou is the maximum mask IoU against the GT instance inside that crop "
              "(when every hard IoU is zero, selection uses maximum soft overlap and query_iou stays zero). "
              "This differs from whole-volume saved components produced at the saved instance probability "
              "threshold (0.3 here), then scored with one-to-one bbox IoU ≥0.1 and the saved score threshold. "
              "These different thresholds, geometry metrics and conditioning do not identify the same fault "
              "or a causal root cause. This is not end-to-end detection or binding ability. Missing R values are counted "
              "explicitly in summary.json. All distributions (min, p10, p25, median, p75, p90, max), kept-component "
              "volume statistics and both raw/kept containment counts are in the anonymous JSON.", "",
              "Further model/image diagnostics would be necessary to distinguish voxelwise oversegmentation from "
              "box extent artifacts and to attribute failures to training, sliding-window probability merging, or "
              "component decoding. This audit neither performs nor authorizes such work.", "",
              "## Reproduce", "", "```sh", result["reproduce_command"], "```", "",
              "The output directory must not already exist. Source file SHA256 values are recorded below and in summary.json.", ""]
    findings = ["## Findings supported by the saved validation data", ""]
    for group in result["new"]:
        new, old = result["new"][group], result["old"][group]
        c = new["recomputed"]
        raw = new["per_gt_any_box_iou_ge_0_1"]["raw"]["n"]
        findings += [f"- {group}: the new saved components provide any box IoU ≥0.1 for {raw}/{c['n_gt']} "
                     f"GT lesions ({f(100 * raw / c['n_gt'])}%), before score filtering; "
                     f"only {c['n_hit']} are one-to-one hits at the existing threshold. "
                     f"Old raw coverage was {old['per_gt_any_box_iou_ge_0_1']['raw']['n']}/{c['n_gt']}. "
                     "Score filtering alone cannot supply boxes absent from the saved component set."]
        v, ov = (s["per_volume_largest_component_ml"]["raw"]["median"] for s in (new, old))
        large = new["volumes_largest_component_gt_100ml"]["raw"]
        findings += [f"- {group}: the raw per-volume largest-component median is {f(v)} mL "
                     f"(old {f(ov)} mL); {large['n']}/{large['denominator']} volumes exceed 100 mL. "
                     "These component-size and box-geometry observations support a coarse component/localization "
                     "failure pattern, while leaving voxelwise lesion overlap unresolved."]
    findings += ["", "For BMSR and ISLES, local query masks also have poor IoU under this distinct saved "
                 "oracle metric. These values cannot locate the same fault as the whole-volume box metric, "
                 "identify a causal root cause, isolate window merging or score filtering, or measure end-to-end R.", ""]
    insertion = lines.index("## Interpretation and evidence limits")
    lines[insertion:insertion] = findings
    for arm, files in result["provenance"].items():
        for name, item in files.items():
            lines += [f"- {arm} {name}: `{item['sha256']}` (`{item['path']}`)"]
    return "\n".join(lines) + "\n"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--new-eval", type=Path, required=True)
    parser.add_argument("--old-eval", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    require(not args.out.exists(), "Output directory must be new")
    new, new_agg, new_groups, new_prov = load_eval(args.new_eval)
    old, old_agg, old_groups, old_prov = load_eval(args.old_eval)
    pairing = check_pair(new, old)
    require(new_groups.keys() == old_groups.keys(), "Gated source/sequence membership differs")
    result = {"evidence": "NOT_EVIDENCE", "pairing": pairing,
              "provenance": {"new": new_prov, "old": old_prov},
              "definitions": {"match_iou": .1, "large_component_description_ml": 100,
                              "bbox_gt_coverage": .9, "output": "anonymous source/sequence aggregates only",
                              "oracle_query_iou": {
                                  "crop": "centered on the GT instance mean voxel position",
                                  "queries": "all event queries, without presence gating",
                                  "mask_threshold": "query probability >0.5",
                                  "metric": "maximum query-mask IoU against GT inside crop; zero-hard-IoU fallback uses soft overlap and returns IoU zero"},
                              "oracle_limit": "Distinct from whole-volume instance probability threshold 0.3 plus bbox IoU >=0.1; cannot locate the same fault or causal root cause; not end-to-end ability"}}
    for arm, rows, aggregate, groups in (("new", new, new_agg, new_groups), ("old", old, old_agg, old_groups)):
        result[arm] = {}
        for group, entry in sorted(groups.items()):
            selected = [row for row in rows if f"{row['source']}/{row['sequence']}" == group and row.get("u")]
            require(bool(selected), "Gated group has no saved U rows")
            result[arm][group] = summarize(selected, entry)
            result[arm][group]["saved_instance_probability_threshold"] = aggregate["run"].get("instance_threshold")
    script = Path(__file__).resolve()
    result["provenance"]["audit_code"] = {
        path.name: {"path": str(path), "sha256": sha256(path)} for path in
        (script, script.parents[1] / "anatobind/eval/detection_metrics.py", script.parents[1] / "anatobind/eval/matching.py",
         script.parents[1] / "anatobind/aur/infer.py")}
    cmd = [sys.executable, str(script), "--new-eval", str(args.new_eval.resolve()),
           "--old-eval", str(args.old_eval.resolve()), "--out",
           str(args.out.resolve().with_name(args.out.name + "_reproduction"))]
    result["reproduce_command"] = ("CUDA_VISIBLE_DEVICES='' PYTHONNOUSERSITE=1 PYTHONPATH=. OMP_NUM_THREADS=1 "
                                    "nice -n19 " + shlex.join(cmd))
    result = clean(result)
    rendered = report(result)
    args.out.mkdir(parents=True, exist_ok=False)
    (args.out / "summary.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    (args.out / "REPORT.md").write_text(rendered)
    print("Saved anonymous validation diagnostic; all existing aggregate U counts and rates match.")


if __name__ == "__main__":
    try:
        main()
    except (ValueError, KeyError, TypeError, OSError, json.JSONDecodeError) as exc:
        # Do not include exception data that might contain case/patient identifiers.
        print(f"Saved-evaluation audit failed ({type(exc).__name__}); no case identifiers emitted.", file=sys.stderr)
        raise SystemExit(1)
