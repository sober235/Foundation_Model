#!/usr/bin/env python3
"""Post-hoc descriptive FP morphology; aggregate-only, completed 405-row val only."""
import argparse
from collections import Counter, defaultdict
import json
import math
import os
from pathlib import Path
import resource
import sys
import tempfile
import time

import paired_analysis as P

CLASSES = ("TP", "FP", "ignored_excused")
FRACTIONS = ("<=0.1", "(0.1,0.5]", ">0.5")


def fraction_bin(value):
    return FRACTIONS[0] if value <= .1 else FRACTIONS[1] if value <= .5 else FRACTIONS[2]


def describe(values):
    result = P.distribution(values)
    if values:
        result.update(min=min(values), mean=sum(values) / len(values))
    return result


def preflight(runs):
    """Read completion/contract metadata only; do not touch private row data."""
    n = 0
    for run in runs:
        marker = P.read(run / "COMPLETE.json")
        contract = P.read(run / "contract.json")
        if (marker.get("status") != "complete" or not marker.get("all_reproduction_checks_passed")
                or contract["mode"] != "val" or marker["rows"] != contract["n_rows"]):
            raise ValueError("only completed full-validation shards accepted")
        n += contract["n_rows"]
    if n != 405:
        raise ValueError("completion metadata must cover exactly 405 rows before private data is read")


def parent_key(det):
    origin = det["detector_origin"]
    # Private in-memory grouping only. Identical local query numbers across windows/rows
    # do not imply identical parents. The row index is appended by summarize().
    return (tuple(tuple(axis) for axis in origin["window"]), origin["query"], origin["component"])


def classified(row, E, match_scan):
    threshold = P.SCORES[row["source"]]
    scan = E.scan("anonymous", row["u"]["gt"], row["u"]["dets"])
    hits, fp, kept = E.scan_matches(scan, threshold)
    used = set(hits.values())
    rest_indices = [i for i in range(len(kept)) if i not in used]
    ignored = [g for g in scan["gt"] if g.get("ignore")]
    # This is exactly detection_metrics.scan_matches' second matching stage.
    pairs = match_scan(ignored, [kept[i] for i in rest_indices]) if ignored and rest_indices else {}
    excused = {rest_indices[i] for i in pairs.values()}
    labels = ["TP" if i in used else "ignored_excused" if i in excused else "FP" for i in range(len(kept))]
    if Counter(labels)["FP"] != fp or Counter(labels)["TP"] != len(hits):
        raise ValueError("classification differs from official matching")
    if len({d["instance"] for d in kept}) != len(kept):
        raise ValueError("duplicate detection instance id")
    return list(zip(kept, labels))


def summarize(rows, E, match_scan):
    children = {label: [] for label in CLASSES}
    parents = {}
    all_multiplicities = Counter()
    for row_index, row in enumerate(rows):
        # Multiplicity counts final post-ownership, post-volume-floor children, not
        # unobserved rejected fragments. Count all scores before applying fixed point.
        local = Counter(parent_key(d) for d in row["u"]["dets"])
        all_multiplicities.update({(row_index, p): n for p, n in local.items()})
        for det, label in classified(row, E, match_scan):
            origin = det["detector_origin"]
            n, parent_n = det["n_voxels"], origin["parent_voxels"]
            fraction = origin["retained_fraction"]
            if (n <= 0 or parent_n < n or det["volume_mm3"] <= 0
                    or not 0 < fraction <= 1 or not math.isclose(fraction, n / parent_n, abs_tol=1e-12, rel_tol=0)
                    or det["score"] != origin["parent_score"]):
                raise ValueError("invalid inherited parent-score or retained-fraction metadata")
            p = (row_index, parent_key(det))
            parent = parents.setdefault(p, {"n_voxels": parent_n, "score": det["score"], "labels": Counter()})
            if parent["n_voxels"] != parent_n or parent["score"] != det["score"]:
                raise ValueError("inconsistent metadata within parent")
            parent["labels"][label] += 1
            children[label].append({"n_voxels": n, "volume_mm3": det["volume_mm3"],
                                    "parent_n_voxels": parent_n, "fraction": fraction,
                                    "score": det["score"], "size": E.size_stratum(det["volume_mm3"]),
                                    "parent": p})
    # All siblings inherit exactly one parent score, so fixed filtering must keep
    # every final sibling of an active parent. Fail rather than infer missing siblings.
    if any(sum(v["labels"].values()) != all_multiplicities[p] for p, v in parents.items()):
        raise ValueError("fixed score filter split a parent sibling set")
    counts = {label: len(children[label]) for label in CLASSES}
    output = {"n_scans": len(rows), "retained_detections": sum(counts.values()), "classes": {}}
    for label, items in children.items():
        parent_ids = {c["parent"] for c in items}
        parent_rows = [parents[p] for p in parent_ids]
        cross = {f: {s: 0 for s in E.STRATA} for f in FRACTIONS}
        for c in items:
            cross[fraction_bin(c["fraction"])][c["size"]] += 1
        output["classes"][label] = {
            "n_children": len(items), "share_of_retained_detections": P.ratio(len(items), sum(counts.values())),
            "child_voxels": describe([c["n_voxels"] for c in items]),
            "child_volume_mm3": describe([c["volume_mm3"] for c in items]),
            "parent_voxels_child_weighted": describe([c["parent_n_voxels"] for c in items]),
            "retained_fraction_child_weighted": describe([c["fraction"] for c in items]),
            "inherited_score_child_weighted": describe([c["score"] for c in items]),
            "fraction_by_physical_size_children": cross,
            "fraction_by_physical_size_share_within_class": {
                f: {s: P.ratio(n, len(items)) for s, n in bins.items()} for f, bins in cross.items()},
            "n_distinct_parents_nonexclusive": len(parent_ids),
            "parent_voxels_parent_weighted": describe([p["n_voxels"] for p in parent_rows]),
            "inherited_score_parent_weighted": describe([p["score"] for p in parent_rows]),
            "all_final_children_per_parent": describe([all_multiplicities[p] for p in parent_ids]),
            "same_class_children_per_parent": describe([parents[p]["labels"][label] for p in parent_ids]),
            "n_children_from_parent_with_multiple_final_children": sum(all_multiplicities[c["parent"]] > 1 for c in items),
            "n_distinct_parents_with_multiple_final_children": sum(all_multiplicities[p] > 1 for p in parent_ids)}
    output["parent_summary"] = {
        "n_distinct_active_parents": len(parents),
        "final_child_multiplicity_per_active_parent": describe([all_multiplicities[p] for p in parents]),
        "n_parents_by_class_membership": dict(Counter("+".join(label for label in CLASSES if p["labels"][label])
                                                      for p in parents.values()))}
    return output


def self_test(E, match_scan):
    import copy
    truth = [{"instance": 1, "box": [0, 0, 0, 2, 2, 2]},
             {"instance": None, "box": [20, 0, 0, 22, 2, 2], "ignore": True}]
    def det(i, box, n, parent=200, component=1, score=.7):
        return {"instance": i, "box": box, "n_voxels": n, "volume_mm3": n, "score": score,
                "detector_origin": {"window": [[0, 30], [0, 30], [0, 30]], "query": 1,
                                    "component": component, "parent_voxels": parent,
                                    "retained_fraction": n / parent, "parent_score": score}}
    row = {"case": "PRIVATE_CASE_SENTINEL", "patient": "PRIVATE_PATIENT_SENTINEL", "source": "bmsr",
           "sequence": "T1c", "u": {"gt": truth, "dets": [
               det(1, truth[0]["box"], 20), det(2, [10, 0, 0, 12, 2, 2], 100),
               det(3, truth[1]["box"], 120, component=2),
               det(4, [30, 0, 0, 32, 2, 2], 15, component=3, score=.699999)]}}
    labels = [x[1] for x in classified(row, E, match_scan)]
    assert labels == ["TP", "FP", "ignored_excused"]
    result = summarize([row, copy.deepcopy(row)], E, match_scan)
    assert result["classes"]["TP"]["n_children"] == 2
    assert result["classes"]["FP"]["n_children"] == 2
    assert result["classes"]["ignored_excused"]["n_children"] == 2
    assert result["parent_summary"]["n_distinct_active_parents"] == 4  # same origin across rows is distinct
    assert result["parent_summary"]["n_parents_by_class_membership"]["TP+FP"] == 2
    assert result["classes"]["FP"]["all_final_children_per_parent"]["max"] == 2
    assert result["classes"]["TP"]["fraction_by_physical_size_children"]["<=0.1"]["<5"] == 2
    assert result["classes"]["FP"]["fraction_by_physical_size_children"]["(0.1,0.5]"]["5-10"] == 2
    assert result["classes"]["ignored_excused"]["fraction_by_physical_size_children"][">0.5"]["5-10"] == 2
    assert "PRIVATE_" not in json.dumps(result)
    assert fraction_bin(.1) == "<=0.1" and fraction_bin(.5) == "(0.1,0.5]"
    assert fraction_bin(.50000001) == ">0.5"
    bad = copy.deepcopy(row); bad["u"]["dets"][1]["score"] = .69999
    try:
        summarize([bad], E, match_scan)
    except ValueError:
        pass
    else:
        raise AssertionError("inconsistent sibling score accepted")
    empty = copy.deepcopy(row); empty["u"]["dets"] = []
    assert summarize([empty], E, match_scan)["retained_detections"] == 0
    all_rows = [dict(copy.deepcopy(row), case="PRIVATE_CASE_" + str(i)) for i in range(405)]
    full = summarize(all_rows, E, match_scan)
    assert full["n_scans"] == 405 and full["classes"]["FP"]["n_children"] == 405
    assert full["parent_summary"]["n_distinct_active_parents"] == 810
    assert "PRIVATE_" not in json.dumps(full)
    with tempfile.TemporaryDirectory(dir=Path(__file__).parent) as tmp:
        runs = [Path(tmp) / "s0", Path(tmp) / "s1"]
        for run, n in zip(runs, (203, 202)):
            run.mkdir()
            (run / "contract.json").write_text(json.dumps({"mode": "val", "n_rows": n}))
            (run / "COMPLETE.json").write_text(json.dumps({"status": "complete", "rows": n,
                                                          "all_reproduction_checks_passed": True}))
        preflight(runs)  # succeeds using only metadata; no private row files exist
        try:
            preflight(runs[:1])
        except ValueError:
            pass
        else:
            raise AssertionError("partial membership accepted")
        (runs[1] / "COMPLETE.json").write_text(json.dumps({"status": "partial"}))
        try:
            preflight(runs)
        except ValueError:
            pass
        else:
            raise AssertionError("partial completion marker accepted")
    print("Synthetic FP mechanism checks passed.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--run", type=Path, action="append", default=[])
    parser.add_argument("--baseline-cases", type=Path)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ[name] = "4"
    sys.path.insert(0, str(args.repo))
    from anatobind.aur import eval as E
    from anatobind.eval.detection_metrics import match_scan
    import torch
    torch.set_num_threads(4)
    if args.self_test:
        self_test(E, match_scan)
        return 0
    if not args.run or args.baseline_cases is None or args.out is None:
        parser.error("--run, --baseline-cases and --out required")
    # Check every completion marker BEFORE opening any private row data.
    preflight(args.run)
    start = time.monotonic()
    old, rows, refs, records, contracts, done, provenance = P.load_complete(args.run, args.baseline_cases)
    if len(rows) != 405 or any(c["mode"] != "val" for c in contracts):
        raise ValueError("exact complete 405-row validation membership required")
    from anatobind.aur.query_decoder import decoder_rules
    if any(c["decoder_rules"] != decoder_rules("query_cc_v1") for c in contracts):
        raise ValueError("analysis rules differ from frozen decoder rules")
    groups = defaultdict(list)
    for row in rows:
        if row.get("u"):
            if row["source"] not in P.SCORES:
                raise ValueError("unregistered supervised U source")
            groups[(row["source"], row["sequence"])].append(row)
    output = {"schema": "aur_query_cc_fp_morphology_aggregate_v1", "evidence": "NOT_EVIDENCE",
              "interpretation": "Post-hoc exploratory error analysis; descriptive associations, not causal proof or threshold selection.",
              "rows": len(rows), "fixed_scores": P.SCORES, "fraction_bins": list(FRACTIONS),
              "physical_child_strata_equivalent_diameter_mm": list(E.STRATA),
              "source_sequence": {source + "/" + sequence: {
                  "score": P.SCORES[source], "registered_S7_sequence": sequence == P.SEQUENCES[source],
                  **summarize(group, E, match_scan)} for (source, sequence), group in sorted(groups.items())},
              "provenance": provenance, "frozen_code_sha": contracts[0]["code_sha"],
              "analysis_script_sha256": P.sha(__file__), "paired_loader_sha256": P.sha(P.__file__),
              "matching_source_sha256": {p: P.sha(args.repo / p) for p in
                  ("anatobind/aur/eval.py", "anatobind/eval/detection_metrics.py",
                   "anatobind/eval/matching.py", "anatobind/eval/lesion_components.py")},
              "notes": ["Final surviving children only: rejected/tiny fragments and pre-ownership parent shapes are not observed.",
                        "Parent counts within TP/FP/excused classes are nonexclusive; a parent may produce mixed outcomes.",
                        "All siblings inherit parent score; metadata and complete sibling retention are checked.",
                        "Small or low-retained-fraction FP does not establish ownership as its cause; candidate scores differ in calibration from legacy fused means.",
                        "No post-ownership child probability mean is saved; no claim that residual probabilities are low.",
                        "No thresholds are swept or reselected, and no loss change follows from this diagnostic alone."]}
    output["analysis_resources"] = {"seconds_excluding_imports": time.monotonic() - start,
                                    "max_rss_gib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2**20,
                                    "cpu_threads": 4, "gpu_hidden": True}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(output, indent=2, allow_nan=False)
    with args.out.open("x") as f:
        f.write(payload + "\n")
    print("Anonymous completed-405 FP morphology aggregates written.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
