#!/usr/bin/env python3
"""Aggregate-only CPU analysis of COMPLETE frozen query_cc_v1 replay runs.

Private input identities and per-GT states exist only in memory. No raw image reads,
model construction, inference, threshold sweep, or per-case/per-GT output occurs.
"""
import argparse
from collections import Counter
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import resource
import statistics
import sys
import time

SCORES = {"bmsr": .70, "isles": .60, "pdgm": .75}
SEQUENCES = {"bmsr": "T1c", "isles": "DWI", "pdgm": "FLAIR"}
STATS = ("raw_components", "eligible_candidates", "suppressed_candidates",
         "fully_occluded_candidates", "tiny_fragments", "final_instances")
EXCLUDE_R = {"matched_det", "matched_score", "binding_origin"}


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def key(row):
    return tuple(str(row[k]) for k in ("source", "patient", "case", "sequence"))


def exact(a, b):
    """JSON-value equality, treating same-position NaNs as equal; no tolerance."""
    if isinstance(a, dict) and isinstance(b, dict):
        a, b = {str(k): v for k, v in a.items()}, {str(k): v for k, v in b.items()}
        return a.keys() == b.keys() and all(exact(a[k], b[k]) for k in a)
    if isinstance(a, (list, tuple)) and isinstance(b, (list, tuple)):
        return len(a) == len(b) and all(exact(x, y) for x, y in zip(a, b))
    if isinstance(a, float) and isinstance(b, float) and math.isnan(a) and math.isnan(b):
        return True
    return a == b


def ratio(n, d):
    return n / d if d else None


def transition(a, b):
    c = Counter(zip(a, b))
    return {"n": len(a), "both_success": c[True, True],
            "rescued_by_query": c[False, True], "harmed_by_query": c[True, False],
            "both_fail": c[False, False]}


def distribution(values):
    values = sorted(values)
    if not values:
        return {"n": 0}
    return {"n": len(values), "sum": sum(values), "median": statistics.median(values),
            "p95_nearest_rank": values[max(0, math.ceil(.95 * len(values)) - 1)],
            "max": max(values)}


def normalized_bindings(row, threshold):
    floor = row.get("e2e_score_floor")
    if floor is not None and floor != threshold:
        raise ValueError("binding score floor differs from frozen source threshold")
    records = row.get("e2e") or []
    if len({p["det"] for p in records}) != len(records):
        raise ValueError("duplicate predicted binding detection id")
    return {p["det"]: p for p in records if p["score"] >= threshold}


def paired_group(old, new, E):
    source = new[0]["source"]
    threshold = SCORES[source]
    u = {name: Counter() for name in ("legacy_max", "query_cc_v1")}
    strata = {}
    u_old, u_new, e_old, e_new, common_old, common_new = [], [], [], [], [], []
    r_counts = {name: Counter() for name in u}
    r_b0_common = {name: Counter() for name in u}
    state_transitions = Counter()
    for a, b in zip(old, new):
        if not exact(a["u"]["gt"], b["u"]["gt"]):
            raise ValueError("paired U GT differs")
        per_gt = []
        for name, row in (("legacy_max", a), ("query_cc_v1", b)):
            hits, fp, kept = E.scan_matches(E.scan("anonymous", row["u"]["gt"], row["u"]["dets"]), threshold)
            bindings = normalized_bindings(row, threshold)
            retained_ids = {p["instance"] for p in kept}
            if not set(bindings).issubset(retained_ids):
                raise ValueError("binding has no retained detection")
            u[name].update(n_scans=1, n_gt=sum(not g.get("ignore") for g in row["u"]["gt"]),
                           n_hit=len(hits), n_fp=fp, n_kept=len(kept),
                           n_ignored_gt=sum(bool(g.get("ignore")) for g in row["u"]["gt"]),
                           n_excused_detections=len(kept)-len(hits)-fp)
            controlled = {p["instance"]: p for p in (row.get("r") or {}).get("instances", [])}
            # A predicted R record must carry the truth from exactly this normalized match.
            expected = {kept[d]["instance"]: controlled[row["u"]["gt"][g]["instance"]]["truth"]
                        for g, d in hits.items() if row["u"]["gt"][g]["instance"] in controlled}
            for did, binding in bindings.items():
                if binding.get("truth") != expected.get(did):
                    raise ValueError("predicted R truth not normalized after fixed score filtering")
            states = {}
            for g, gt in enumerate(row["u"]["gt"]):
                if gt.get("ignore"):
                    continue
                st = strata.setdefault(gt["stratum"], {"n_gt": 0, "legacy_hits": 0, "query_hits": 0,
                                                       "both_hit": 0, "rescued_by_query": 0,
                                                       "harmed_by_query": 0, "both_missed": 0})
                st["legacy_hits" if name == "legacy_max" else "query_hits"] += int(g in hits)
                if gt["instance"] not in controlled:
                    continue
                truth = controlled[gt["instance"]]["truth"]
                binding = bindings.get(kept[hits[g]]["instance"]) if g in hits else None
                correct = binding is not None and binding["r"] == truth
                state = ("missed" if g not in hits else "matched_unbound" if binding is None
                         else "matched_correct" if correct else "matched_wrong")
                states[g] = (state, binding, truth, correct)
                r_counts[name].update(eligible_gt=1, matched_gt=int(g in hits),
                                      bound_matched_gt=int(binding is not None), correct_e2e_gt=int(correct))
            per_gt.append((hits, states))
        ah, ar = per_gt[0]
        bh, br = per_gt[1]
        if ar.keys() != br.keys():
            raise ValueError("controlled eligible GT membership differs")
        for g, gt in enumerate(a["u"]["gt"]):
            if gt.get("ignore"):
                continue
            x, y = g in ah, g in bh
            u_old.append(x); u_new.append(y)
            st = strata[gt["stratum"]]
            st["n_gt"] += 1
            st["both_hit" if x and y else "rescued_by_query" if y else "harmed_by_query" if x else "both_missed"] += 1
        for g in ar:
            x, y = ar[g], br[g]
            if x[2] != y[2]:
                raise ValueError("controlled pseudo truth differs")
            e_old.append(x[3]); e_new.append(y[3])
            state_transitions[x[0] + "->" + y[0]] += 1
            if x[1] is not None and y[1] is not None:
                common_old.append(x[3]); common_new.append(y[3])
                for name, state in (("legacy_max", x), ("query_cc_v1", y)):
                    rr, bb = state[3], state[1]["b0"] == state[2]
                    r_b0_common[name].update({"both_right" if rr and bb else "R_rescues_B0" if rr
                                               else "R_harms_B0" if bb else "both_wrong": 1})
    for counts in u.values():
        counts["sensitivity"] = ratio(counts["n_hit"], counts["n_gt"])
        counts["fp_per_scan"] = ratio(counts["n_fp"], counts["n_scans"])
    for st in strata.values():
        st["legacy_sensitivity"] = ratio(st["legacy_hits"], st["n_gt"])
        st["query_sensitivity"] = ratio(st["query_hits"], st["n_gt"])
    for counts in r_counts.values():
        for label, n in (("detection_coverage", "matched_gt"), ("binding_coverage", "bound_matched_gt"),
                         ("e2e_success_over_all_eligible_gt", "correct_e2e_gt")):
            counts[label] = ratio(counts[n], counts["eligible_gt"])
    delta = {k: (u["query_cc_v1"][k] - u["legacy_max"][k]
                 if u["query_cc_v1"][k] is not None and u["legacy_max"][k] is not None else None)
             for k in ("n_hit", "n_fp", "n_kept", "sensitivity", "fp_per_scan")}
    return {"score": threshold, "U": u, "U_paired_gt": transition(u_old, u_new), "U_size_strata": strata,
            "U_query_minus_legacy": delta,
            "R": r_counts, "R_e2e_paired_all_eligible_gt": transition(e_old, e_new),
            "R_common_matched_and_bound_gt": transition(common_old, common_new),
            "R_vs_B0_on_common_matched_and_bound_gt": r_b0_common,
            "R_state_transitions_all_eligible_gt": dict(sorted(state_transitions.items()))}


def invariance(old, new):
    out = Counter(rows=len(new), A_exact_rows=0, controlled_instances=0,
                  controlled_exact_instances=0, controlled_membership_mismatch_rows=0,
                  origin_query_window_unavailable_in_legacy_instances=0)
    for a, b in zip(old, new):
        out["A_exact_rows"] += int(exact(a["a"], b["a"]))
        aa = {i["instance"]: i for i in (a.get("r") or {}).get("instances", [])}
        bb = {i["instance"]: i for i in (b.get("r") or {}).get("instances", [])}
        out["controlled_membership_mismatch_rows"] += int(aa.keys() != bb.keys())
        for k, x in aa.items():
            out["controlled_instances"] += 1
            shared = {f: v for f, v in x.items() if f not in EXCLUDE_R}
            y = bb.get(k, {})
            out["controlled_exact_instances"] += int(exact(shared, {f: y.get(f) for f in shared}))
            out["origin_query_window_unavailable_in_legacy_instances"] += int("binding_origin" not in x)
    out["A_saved_scores_exact"] = out["A_exact_rows"] == out["rows"]
    out["controlled_shared_fields_exact"] = (out["controlled_exact_instances"] == out["controlled_instances"]
                                               and not out["controlled_membership_mismatch_rows"])
    return dict(out)


def analyze(old, new, references, E):
    old = sorted(old, key=key); new = sorted(new, key=key)
    if [key(r) for r in old] != [key(r) for r in new]:
        raise ValueError("paired row membership differs")
    out = {"invariance": invariance(old, new), "source_sequence": {}}
    groups = sorted({(r["source"], r["sequence"]) for r in new if r.get("u")})
    for source, sequence in groups:
        if source not in SCORES:
            raise ValueError("unregistered supervised U source")
        select = lambda rows: [r for r in rows if (r["source"], r["sequence"]) == (source, sequence) and r.get("u")]
        group = paired_group(select(old), select(new), E)
        group["S7_comparable_sequence"] = sequence == SEQUENCES[source]
        if group["S7_comparable_sequence"]:
            ref = references[source]
            if ref["n_scans"] != group["U"]["query_cc_v1"]["n_scans"]:
                raise ValueError("S7 budget not on identical complete scan membership")
            group["S7"] = {k: ref[k] for k in ("thr", "n_scans", "n_gt", "n_hit", "n_fp", "sensitivity", "fp_per_scan")}
            for metric in group["U"].values():
                metric["within_S7_FP_budget"] = metric["fp_per_scan"] <= ref["fp_per_scan"] + 1e-9
        out["source_sequence"][source + "/" + sequence] = group
    return out


def load_complete(runs, baseline):
    records, contracts, completions, references = [], [], [], []
    provenance = []
    for run in runs:
        contract = read(run / "contract.json")
        done = read(run / "COMPLETE.json")  # No partial-run reporting mode.
        if done.get("status") != "complete" or not done.get("all_reproduction_checks_passed"):
            raise ValueError("run incomplete or reproduction failed")
        if contract["decoder_profile"] != "query_cc_v1" or contract["score_thresholds"] != SCORES:
            raise ValueError("profile or frozen scores differ")
        if contract["saved_cases_sha256"] != sha(baseline):
            raise ValueError("baseline cases hash differs from frozen runner")
        selected = read(run / "selection_private.json")
        expected = {key(r) for r in selected}
        selection_sha = hashlib.sha256(json.dumps([key(r) for r in selected], ensure_ascii=False,
                                                separators=(",", ":")).encode()).hexdigest()
        if selection_sha != contract["selection_sha256"]:
            raise ValueError("frozen private selection hash differs")
        local = [read(p) for p in sorted((run / "row_records").glob("*.json"))]
        if (len(local) != contract["n_rows"] or len(local) != done["rows"]
                or {key(r["result"]) for r in local} != expected):
            raise ValueError("completed record membership mismatch")
        if any(r["code_sha"] != contract["code_sha"] for r in local):
            raise ValueError("row code differs from frozen run")
        if any(r["result"].get("decoder_profile") != "query_cc_v1" for r in local):
            raise ValueError("row decoder profile differs from frozen run")
        manifest = read(run / "input_manifest_private.json")
        manifest_sha = hashlib.sha256(json.dumps(manifest, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
        if manifest_sha != contract["raw_inputs_sha256"]:
            raise ValueError("frozen input manifest differs")
        for r in local:
            token = hashlib.sha256(json.dumps(key(r["result"]), separators=(",", ":")).encode()).hexdigest()
            if r["input_sha256"] != manifest[token]:
                raise ValueError("row input provenance differs")
        if sha(run / "reference_frozen.json") != contract["reference_sha256"]:
            raise ValueError("reference hash differs")
        records.extend(local); contracts.append(contract); completions.append(done)
        references.append(read(run / "reference_frozen.json"))
        provenance.append({"contract_sha256": sha(run / "contract.json"),
                           "complete_sha256": sha(run / "COMPLETE.json"),
                           "row_records_content_sha256": hashlib.sha256("".join(sorted(
                               hashlib.sha256(json.dumps(r, sort_keys=True).encode()).hexdigest() for r in local)).encode()).hexdigest()})
    invariant_fields = ("code_sha", "checkpoint_sha256", "samples_sha256", "val_patients_sha256", "mode",
                        "saved_cases_sha256", "decoder_rules", "score_thresholds", "reference_sha256")
    if any(any(c[k] != contracts[0][k] for k in invariant_fields) for c in contracts):
        raise ValueError("mixed frozen run contracts")
    new = [r["result"] for r in records]
    if len({key(r) for r in new}) != len(new):
        raise ValueError("duplicate paired rows")
    all_old = read(baseline)
    if len({key(r) for r in all_old}) != len(all_old):
        raise ValueError("duplicate baseline rows")
    old_index = {key(r): r for r in all_old}
    if not {key(r) for r in new}.issubset(old_index):
        raise ValueError("new rows absent in baseline")
    if contracts[0]["mode"] == "val" and {key(r) for r in new} != set(old_index):
        raise ValueError("full val analysis requires the COMPLETE baseline row membership")
    if contracts[0]["mode"] == "six" and (len(runs) != 1 or len(new) != 6):
        raise ValueError("six probe requires exactly its six complete rows")
    return [old_index[key(r)] for r in new], new, references[0], records, contracts, completions, provenance


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--repo", type=Path, required=True)
    p.add_argument("--run", type=Path, action="append", default=[])
    p.add_argument("--baseline-cases", type=Path)
    p.add_argument("--out", type=Path)
    p.add_argument("--self-test", action="store_true")
    args = p.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ[name] = "4"
    sys.path.insert(0, str(args.repo))
    spec = importlib.util.spec_from_file_location("frozen_decoder_runner", args.repo / "scripts/aur_compare_decoders.py")
    runner = importlib.util.module_from_spec(spec); spec.loader.exec_module(runner)
    runner.torch.set_num_threads(4)
    if args.self_test:
        self_test(runner)
        print("Synthetic aggregate-only checks passed.")
        return 0
    if not args.run or args.baseline_cases is None or args.out is None:
        p.error("--run, --baseline-cases and --out are required")
    start = time.monotonic()
    old, new, refs, records, contracts, done, provenance = load_complete(args.run, args.baseline_cases)
    if any(c["decoder_rules"] != runner.decoder_rules("query_cc_v1") for c in contracts):
        raise ValueError("analysis decoder rules differ from frozen replay")
    normalized_old = [runner.fixed_legacy_e2e(r) for r in old]
    result = analyze(normalized_old, new, refs, runner.E)
    result.update(schema="aur_query_cc_paired_aggregate_v1", evidence="NOT_EVIDENCE", mode=contracts[0]["mode"],
                  scope="six-case mechanism probe only" if contracts[0]["mode"] == "six" else "complete frozen validation",
                  rows=len(new), source_score_thresholds=SCORES, provenance=provenance,
                  comparison="Same pre-fixed numeric source score; observed FP is NOT equalized. Parent component and fused-map scores have different calibration.",
                  legacy_matching_normalization={"changed_cached_binding_truths": sum(
                      x.get("truth") != y.get("truth") for a, b in zip(old, normalized_old)
                      for x, y in zip(a.get("e2e") or [], b.get("e2e") or [])),
                      "method": "runner.fixed_legacy_e2e; saved predictions unchanged"},
                  frozen_code_sha=contracts[0]["code_sha"], checkpoint_sha256=contracts[0]["checkpoint_sha256"],
                  analysis_script_sha256=sha(__file__),
                  analysis_dependency_sha256={str(path): sha(args.repo / path) for path in
                      ("scripts/aur_compare_decoders.py", "anatobind/aur/eval.py",
                       "anatobind/eval/detection_metrics.py", "anatobind/eval/matching.py")},
                  inference_resources={"recorded_row_seconds": distribution([r["seconds"] for r in records]),
                                       "peak_cuda_allocated_gib": max(r["peak_cuda_gib"] for r in records),
                                       "windows": sum(r["result"]["n_windows"] for r in records),
                                       "completed_process_elapsed_seconds": [r["seconds"] for r in done],
                                       "timing_note": "Query replay only; saved legacy timing is not a paired baseline. Sum of row times is not multi-shard wall time."},
                  decoder_counts={k: distribution([r["decoder_stats"][k] for r in new
                                                    if k in r.get("decoder_stats", {})]) for k in STATS},
                  retained_fraction=distribution([d["detector_origin"]["retained_fraction"] for r in new
                                                  for d in (r.get("u") or {}).get("dets", [])]),
                  limitations=["All A/R labels are pseudo references; no clinical or human Level R evidence.",
                               "Six probe is not representative; no source-general conclusion before full val.",
                               "A invariance covers saved numerical scores, not voxelwise maps absent from records.",
                               "Controlled exact invariance covers saved shared fields; legacy has no saved query/window origin.",
                               "R common-match comparisons condition on intersection; all-GT E2E is reported separately.",
                               "No threshold optimization; source budgets compared only on registered S7 sequence.",
                               "Budget failure at these fixed scores does not reject query-preserving decoding at other calibration; no such calibration is evaluated here.",
                               "BMSR and ISLES use AUR-resampled versus native S7 grids (approximate comparison).",
                               "CUDA peak is allocated memory, not reserved memory or total device footprint."])
    result["analysis_resources"] = {"seconds": time.monotonic()-start,
                                    "max_rss_gib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2**20,
                                    "cpu_threads": 4, "gpu_hidden": True}
    safe = json.loads(json.dumps(result, allow_nan=False))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("x") as f:
        json.dump(safe, f, indent=2, allow_nan=False); f.write("\n")
    print("Anonymous aggregate written; " + str(len(new)) + " complete paired rows analyzed.")
    inv = result["invariance"]
    return 0 if inv["A_saved_scores_exact"] and inv["controlled_shared_fields_exact"] else 2


def self_test(runner):
    """Meaningful checks: matching normalization, ignored GT, inclusive score, R selection, invariance."""
    import copy
    gt = [{"instance": 1, "box": [0, 0, 0, 2, 2, 2], "stratum": "small"},
          {"instance": 2, "box": [10, 0, 0, 12, 2, 2], "stratum": "small"},
          {"instance": None, "box": [20, 0, 0, 22, 2, 2], "stratum": "small", "ignore": True}]
    def det(i, box, score):
        return {"instance": i, "box": box, "score": score}
    old = {"source": "bmsr", "patient": "PRIVATE_PATIENT_SENTINEL", "case": "PRIVATE_CASE_SENTINEL",
           "sequence": "T1c", "a": {"x": float("nan"), "score": .8},
           "u": {"gt": gt, "dets": [det(1, gt[0]["box"], .5), det(2, [0, 0, 0, 2, 2, 3], .7),
                                      det(3, gt[2]["box"], .7)]},
           "r": {"instances": [{"instance": 1, "truth": 2, "r": 2, "r_probs": [.1, .9]},
                                {"instance": 2, "truth": 3, "r": 3, "r_probs": [.2, .8]}]},
           "e2e": [{"det": 2, "score": .7, "r": 2, "b0": 1, "truth": None},
                   {"det": 3, "score": .7, "r": 2, "b0": 2, "truth": None}]}
    old = runner.fixed_legacy_e2e(old)
    assert old["e2e"][0]["truth"] == 2  # low-score perfect box no longer steals GT
    new = copy.deepcopy(old)
    new["u"]["dets"] = [det(2, [0, 0, 0, 2, 2, 3], .7), det(4, gt[1]["box"], .7),
                              det(3, gt[2]["box"], .7), det(5, [40, 0, 0, 42, 2, 2], .7)]
    new["e2e"] = [{"det": 2, "score": .7, "r": 1, "b0": 2, "truth": 2},
                  {"det": 4, "score": .7, "r": 3, "b0": 2, "truth": 3},
                  {"det": 3, "score": .7, "r": 2, "b0": 2, "truth": None},
                  {"det": 5, "score": .7, "r": 2, "b0": 2, "truth": None}]
    refs = {"bmsr": {"thr": .65, "n_scans": 1, "n_gt": 2, "n_hit": 1,
                     "n_fp": 0, "sensitivity": .5, "fp_per_scan": 0.}}
    result = analyze([old], [new], refs, runner.E)
    g = result["source_sequence"]["bmsr/T1c"]
    assert g["U"]["query_cc_v1"]["n_fp"] == 1
    assert g["U"]["legacy_max"]["n_excused_detections"] == 1
    assert g["U_paired_gt"]["rescued_by_query"] == 1
    assert g["R_e2e_paired_all_eligible_gt"]["rescued_by_query"] == 1
    assert g["R_e2e_paired_all_eligible_gt"]["harmed_by_query"] == 1
    assert g["R_common_matched_and_bound_gt"]["n"] == 1
    assert result["invariance"]["A_saved_scores_exact"]
    assert "PRIVATE_" not in json.dumps(result)
    new["r"]["instances"][0]["r_probs"][0] += 1e-12
    assert not invariance([old], [new])["controlled_shared_fields_exact"]
    new["a"]["score"] += 1e-12
    assert not invariance([old], [new])["A_saved_scores_exact"]
    broken = copy.deepcopy(old); broken["e2e_score_floor"] = .6
    try:
        paired_group([old], [broken], runner.E)
    except ValueError:
        pass
    else:
        raise AssertionError("mismatched binding score floor accepted")


if __name__ == "__main__":
    sys.exit(main())
