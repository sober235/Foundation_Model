"""Evaluation of AnatoBind-Brain on the test patients (spec §7; SSL-first plan T12; gates G2, G3).

A: the whole-volume entity map (anatobind.aur.infer.predict_volume) against the SynthSeg pseudo-label, Dice per sided
host class (13) and per entity (32), macro means over the classes present. U: the lesion instances of the lesion
probability map against the instances of the lesion pseudo-label, matched one-to-one on box IoU >= 0.1 like S2 / S7
(anatobind.eval.detection_metrics); the operating point is the score threshold with the best sensitivity within the
reference detector's false positives per scan on the same patients (ties: the highest threshold, S7's rule). R,
controlled track: every pseudo-label instance is bound from one crop centred on it (anatobind.aur.infer.bind_instances)
and compared with the host truth of anatobind.aur.targets.host_targets; B0 is the same lookup on the model's own
predicted anatomy, B0* the lookup on the SynthSeg map (the truth's own source, a ceiling). Every number here rests on
pseudo-labels: NOT_EVIDENCE."""
import json

import numpy as np
from scipy import ndimage

from anatobind.aur.dataset import load_volume
from anatobind.aur.infer import bind_instances, instances_from_probability, predict_volume
from anatobind.aur.labels import ENTITY_LABELS, HOST_NAMES, HOST_SIDE, HOST_TISSUE, NO_HOST, N_HOSTS, SEQ_TYPES, entity_to_synthseg, host_map
from anatobind.aur.targets import host_targets
from anatobind.eval.detection_metrics import scan_matches as _scan_matches
from anatobind.eval.detection_metrics import sweep
from anatobind.eval.lesion_components import STRATA, component_rows, components, size_stratum

INSTANCE_THRESHOLD = 0.3
A_GATE = 0.80
U_MARGIN = 0.05
LOCAL_DICE_MIN = 0.8                   # the recognition–binding gap counts instances whose local A Dice reaches this
# the reference detector per source: S7's disease, its pre-registered threshold, the sequence whose lesion labels
# equal S7's (the gate is judged on that sequence only) and the grid relation (S7 predicted on the native grid)
REFERENCE = {"pdgm": {"disease": "glioma", "thr": 0.60, "sequence": "FLAIR", "grid": "same 1 mm grid"},
             "bmsr": {"disease": "metastasis", "thr": 0.65, "sequence": "T1c", "grid": "AUR 1 mm resampled vs native (approximate)"},
             "isles": {"disease": "infarct", "thr": 0.50, "sequence": "DWI", "grid": "AUR 1 mm resampled vs native 2 mm (approximate)"}}
NOT_IMPLEMENTED = ("SibBMS 10-case annotated subset (U, report only)", "fastMRI 433-volume reliable-slice A Dice and 1297-box host agreement (report only)",
                   "end-to-end R on a human-labelled set (Level R: the sheet is exported, the statistics wait for the readers)")


def _dice(a, b):
    sa, sb = int(a.sum()), int(b.sum())
    if sa + sb == 0:
        return float("nan")
    return 2.0 * int((a & b).sum()) / (sa + sb)


def _nanmean(values):
    arr = np.asarray([v for v in values if v == v], float)
    return float(arr.mean()) if arr.size else float("nan")


def _dice_from_ids(pred_ids, true_ids, n_classes):
    """Dice of the classes 1..n_classes from two integer maps, by one joint bincount."""
    pred_ids, true_ids = np.asarray(pred_ids).astype(np.int64).ravel(), np.asarray(true_ids).astype(np.int64).ravel()
    joint = np.bincount(pred_ids * (n_classes + 1) + true_ids, minlength=(n_classes + 1) ** 2).reshape(n_classes + 1, n_classes + 1)
    inter, sp, st = np.diag(joint), joint.sum(1), joint.sum(0)
    out = []
    for c in range(1, n_classes + 1):
        out.append(float("nan") if sp[c] + st[c] == 0 else 2.0 * inter[c] / (sp[c] + st[c]))
    return out, [bool(st[c] > 0) for c in range(1, n_classes + 1)]


def host_dice(pred_seg, true_seg):
    """Dice per sided host class and per entity between two SynthSeg-valued maps; macro = mean over the classes
    that exist in either map (nan where both are empty)."""
    hp, ht = host_map(np.asarray(pred_seg)), host_map(np.asarray(true_seg))
    per_host, in_truth = _dice_from_ids(hp, ht, N_HOSTS)
    hosts = {HOST_NAMES[i]: per_host[i] for i in range(N_HOSTS)}
    lut = np.zeros(256, np.int64)
    for i, l in enumerate(ENTITY_LABELS):
        lut[l] = i + 1
    per_entity, _ = _dice_from_ids(lut[np.asarray(pred_seg).astype(np.int64)], lut[np.asarray(true_seg).astype(np.int64)], len(ENTITY_LABELS))
    entities = {int(l): per_entity[i] for i, l in enumerate(ENTITY_LABELS)}
    return {"hosts": hosts, "host_macro": _nanmean(hosts.values()), "entities": entities, "entity_macro": _nanmean(entities.values()),
            "n_hosts_in_truth": int(sum(in_truth))}


def instance_rows(inst, voxel_mm3):
    """One row per instance id of an instance map: half-open box in array order, voxel count, volume, size stratum."""
    rows = []
    for k, sl in enumerate(ndimage.find_objects(np.asarray(inst)), start=1):
        if sl is None:
            continue
        nv = int((inst[sl] == k).sum())
        mm3 = nv * float(voxel_mm3)
        rows.append({"instance": k, "box": [sl[0].start, sl[1].start, sl[2].start, sl[0].stop, sl[1].stop, sl[2].stop],
                     "n_voxels": nv, "volume_mm3": mm3, "stratum": size_stratum(mm3)})
    return rows


def ignored_rows(small, voxel_mm3):
    """The components of the sub-floor mask as ground-truth rows flagged ignore (S7 M7: a detection on them is
    neither a hit nor a false positive)."""
    comp, n = components(np.asarray(small))
    rows = []
    for r in component_rows(comp, n, voxel_mm3, "lesion", min_mm3=0.0):
        rows.append({"instance": None, "box": [int(v) for v in r["box"]], "n_voxels": r["n_voxels"], "volume_mm3": float(r["mm3"]),
                     "stratum": size_stratum(r["mm3"]), "ignore": True})
    return rows


def scan(case, gt_rows, det_rows):
    """A scan in the detection_metrics convention (every row a 'lesion')."""
    return {"case": case, "gt": [{**r, "family": "lesion"} for r in gt_rows], "dets": [{**d, "family": "lesion"} for d in det_rows]}


def scan_matches(s, thr):
    return _scan_matches(s, thr)


def _count(scans, thr):
    n_gt = n_hit = n_fp = 0
    for s in scans:
        hits, fp, _ = _scan_matches(s, thr)
        n_gt += sum(1 for r in s["gt"] if not r.get("ignore"))
        n_hit += len(hits)
        n_fp += fp
    return n_gt, n_hit, n_fp


def sensitivity_at_threshold(scans, thr):
    n_gt, n_hit, n_fp = _count(scans, thr)
    return {"thr": float(thr), "sensitivity": n_hit / n_gt if n_gt else float("nan"), "fp_per_scan": n_fp / len(scans) if scans else float("nan"),
            "n_gt": n_gt, "n_hit": n_hit, "n_fp": n_fp, "n_scans": len(scans)}


def sensitivity_at_budget(scans, fp_per_scan):
    """The score threshold (grid 0.05 … 0.95) with the best sensitivity among those with at most fp_per_scan false
    positives per scan; ties go to the highest threshold. Without any threshold within the budget the strictest one
    is returned and `within_budget` is False."""
    rows = sweep(scans)
    within = [r for r in rows if r["fp_per_scan"] <= fp_per_scan + 1e-9]
    pool = within if within else [rows[-1]]
    best = max(pool, key=lambda r: (r["sensitivity"], r["thr"]))
    return {"thr": best["thr"], "sensitivity": best["sensitivity"], "fp_per_scan": best["fp_per_scan"], "n_gt": best["n_gt"],
            "n_hit": best["n_hit"], "n_fp": best["n_fp"], "n_scans": best["n_scans"], "budget": float(fp_per_scan), "within_budget": bool(within)}


def strata_sensitivity(scans, thr):
    """Hits over truths per size stratum at a threshold."""
    out = {s: {"n_gt": 0, "n_hit": 0} for s in STRATA}
    for s in scans:
        hits, _, _ = _scan_matches(s, thr)
        for g, r in enumerate(s["gt"]):
            if r.get("ignore"):
                continue
            out[r["stratum"]]["n_gt"] += 1
            out[r["stratum"]]["n_hit"] += int(g in hits)
    for v in out.values():
        v["sensitivity"] = v["n_hit"] / v["n_gt"] if v["n_gt"] else float("nan")
    return out


def _side(h):
    """left / right, 'midline' for the brainstem, 'none' for no host (so that a brainstem answer to a no-host truth
    is not side-correct)."""
    if h >= N_HOSTS:
        return "none"
    return HOST_SIDE[h] or "midline"


def _tissue(h):
    return HOST_TISSUE[h] if h < N_HOSTS else None


def binding_accuracy(pred, truth):
    """Host accuracy (ABA), side accuracy and tissue-family accuracy of predicted host indices against the truth."""
    pred, truth = np.asarray(pred, int), np.asarray(truth, int)
    n = int(pred.size)
    if n == 0:
        return {"n": 0, "aba": float("nan"), "side": float("nan"), "tissue": float("nan")}
    return {"n": n, "aba": float((pred == truth).mean()),
            "side": float(np.mean([_side(p) == _side(t) for p, t in zip(pred, truth)])),
            "tissue": float(np.mean([_tissue(p) == _tissue(t) for p, t in zip(pred, truth)]))}


def rescue_harm(r, b0, truth):
    """How the relation head and the B0 lookup agree with the truth instance by instance."""
    r, b0, truth = (np.asarray(x, int) for x in (r, b0, truth))
    rr, bb = r == truth, b0 == truth
    n = int(truth.size)
    return {"n": n, "both_right": int((rr & bb).sum()), "both_wrong": int((~rr & ~bb).sum()), "rescue": int((rr & ~bb).sum()),
            "harm": int((~rr & bb).sum()), "b0_wrong_share": float((~bb).mean()) if n else float("nan")}


def local_host_dice(pred_seg, true_seg, window):
    """The 13-host macro Dice inside a window (the bind crop of an instance): the local recognition quality."""
    sl = tuple(slice(max(a, 0), min(b, n)) for (a, b), n in zip(window, pred_seg.shape))
    return host_dice(pred_seg[sl], true_seg[sl])["host_macro"]


def evaluate_case(model, row, crop, device, batch_size=1, instance_threshold=INSTANCE_THRESHOLD, bind_predicted=True):
    """The three tracks on one sample row (a volume of one sequence). R, controlled: every pseudo-label instance bound
    from its own crop, against the host truth, with B0 (the lookup on the predicted anatomy), the local A Dice of the
    bind window and the best detection on it (for the gap and the end-to-end track at aggregation). R, end to end: the
    predicted instances bound the same way, with B0 on the predicted anatomy, matched to the truth at aggregation."""
    vol = load_volume(row)
    pred = predict_volume(model, vol["image"], vol["affine"], crop, device, batch_size=batch_size)
    pred_seg, true_seg = entity_to_synthseg(pred["entity"]), entity_to_synthseg(vol["entity"])
    res = {"case": row["case"], "source": row["source"], "sequence": row["sequence"], "n_windows": pred["n_windows"],
           "a_supervised": bool(vol["a_supervised"]), "u_supervised": bool(vol["u_supervised"]), "r_supervised": bool(vol["r_supervised"]),
           "seq_pred": int(np.argmax(pred["seq_probs"])), "seq_truth": int(vol["seq"]), "seq_probs": [float(p) for p in pred["seq_probs"]],
           "a": host_dice(pred_seg, true_seg), "u": None, "r": None, "e2e": None}
    if not vol["u_supervised"]:
        return res
    gt = instance_rows(vol["instance"], vol["voxel_mm3"]) + ignored_rows(vol["small"], vol["voxel_mm3"])
    det_inst, dets = instances_from_probability(pred["lesion_prob"], vol["voxel_mm3"], instance_threshold)
    res["u"] = {"gt": gt, "dets": dets}
    counted = [g for g in gt if not g.get("ignore")]
    if not counted or not vol["r_supervised"]:
        return res
    hits, _, kept = scan_matches(scan(row["case"], gt, dets), 0.0)                   # every detection: who covers which truth
    best_det = {g: kept[d] for g, d in hits.items()}
    bound = bind_instances(model, vol["image"], vol["affine"], vol["instance"], crop, device)
    truth = vol["hosts"]["host"]
    b0 = host_targets(vol["instance"], pred_seg, vol["spacing"])["host"]
    by_id = {r["instance"]: r for r in counted}
    instances = []
    for b in bound:
        k = b["instance"]
        g = next(i for i, r in enumerate(gt) if r.get("instance") == k)
        d = best_det.get(g)
        instances.append({"instance": k, "truth": int(truth[k - 1]), "r": int(b["host"]), "b0": int(b0[k - 1]),
                          "r_probs": [float(p) for p in b["host_probs"]], "query_iou": b["query_iou"], "zero_overlap": b["zero_overlap"],
                          "in_crop_share": b["in_crop_share"], "volume_mm3": by_id[k]["volume_mm3"], "stratum": by_id[k]["stratum"],
                          "box": by_id[k]["box"], "local_dice": local_host_dice(pred_seg, true_seg, b["window"]),
                          "matched_score": float(d["score"]) if d else None, "matched_det": int(d["instance"]) if d else None})
    res["r"] = {"instances": instances}
    if bind_predicted and dets:
        pb = bind_instances(model, vol["image"], vol["affine"], det_inst, crop, device)
        b0_pred = host_targets(det_inst, pred_seg, vol["spacing"])["host"]
        det_truth = {d["instance"]: int(truth[gt[g]["instance"] - 1]) for g, d in best_det.items()}
        res["e2e"] = [{"det": p["instance"], "r": int(p["host"]), "b0": int(b0_pred[p["instance"] - 1]), "score": next(d["score"] for d in dets if d["instance"] == p["instance"]),
                       "truth": det_truth.get(p["instance"]), "query_iou": p["query_iou"], "zero_overlap": p["zero_overlap"]} for p in pb]
    return res


def _r_block(instances, r_key="r", b0_key="b0", truth_key="truth"):
    truth = [i[truth_key] for i in instances]
    r, b0 = [i[r_key] for i in instances], [i[b0_key] for i in instances]
    ar, ab = binding_accuracy(r, truth), binding_accuracy(b0, truth)
    return {"n": len(instances), "aba_r": ar["aba"], "aba_b0": ab["aba"], "side_r": ar["side"], "side_b0": ab["side"],
            "tissue_r": ar["tissue"], "tissue_b0": ab["tissue"], "rescue_harm": rescue_harm(r, b0, truth),
            "n_zero_overlap": int(sum(bool(i.get("zero_overlap")) for i in instances))}


def _u_entry(scans, ref, fixed_thr, u_margin):
    entry = {"n_scans": len(scans), "n_gt": sum(1 for x in scans for g in x["gt"] if not g.get("ignore")),
             "n_ignored": sum(1 for x in scans for g in x["gt"] if g.get("ignore")), "saturated_cases": [x["case"] for x in scans if len(x["gt"]) > 64]}
    if ref is None:
        entry.update({"sweep": sweep(scans), "pass": None, "threshold_selection": None})
        return entry
    if fixed_thr is not None:
        at = sensitivity_at_threshold(scans, fixed_thr)
        at["within_budget"] = at["fp_per_scan"] <= ref["fp_per_scan"] + 1e-9
        at["budget"] = float(ref["fp_per_scan"])
        entry["threshold_selection"] = "fixed (chosen on the validation split)"
    else:
        at = sensitivity_at_budget(scans, ref["fp_per_scan"])
        entry["threshold_selection"] = "test-selected (optimistic: the best of 19 thresholds on this very set)"
    entry.update(at)
    entry["strata"] = strata_sensitivity(scans, at["thr"])
    entry["pass"] = bool(at["within_budget"] and at["sensitivity"] >= ref["sensitivity"] - u_margin)
    return entry


def aggregate(results, reference, a_gate=A_GATE, u_margin=U_MARGIN, fixed_thresholds=None):
    """Per-source and overall tables of the tracks and the gates G2 (A, U) and G3 (R). `reference` maps a source to
    the reference detector's {"sensitivity", "fp_per_scan", "thr", "n_scans", "n_gt", ...} on the same patients;
    `fixed_thresholds` maps a source to a score threshold chosen on the validation split (else the threshold is
    selected on this set and labelled so). U is pooled per (source, sequence); the gate reads the sequence whose
    lesion labels are the reference's. A's gate reads the A-supervised rows only."""
    fixed_thresholds = fixed_thresholds or {}
    sources = sorted({r["source"] for r in results})
    # A
    a_rows = [r for r in results if r.get("a") and r.get("a_supervised", True)]
    a_unsup = [r for r in results if r.get("a") and not r.get("a_supervised", True)]
    per_host = {h: _nanmean(r["a"]["hosts"].get(h, float("nan")) for r in a_rows) for h in HOST_NAMES[:N_HOSTS]}
    a = {"n_cases": len(a_rows), "host_macro": _nanmean(r["a"]["host_macro"] for r in a_rows),
         "entity_macro": _nanmean(r["a"]["entity_macro"] for r in a_rows), "per_host": per_host,
         "per_source": {s: {"n_cases": sum(1 for r in a_rows if r["source"] == s),
                            "host_macro": _nanmean(r["a"]["host_macro"] for r in a_rows if r["source"] == s)} for s in sources},
         "unsupervised": {"n_cases": len(a_unsup), "host_macro": _nanmean(r["a"]["host_macro"] for r in a_unsup),
                          "note": "rows whose SynthSeg labels failed QC (thick slices): reported, not gated"}}
    a["gate"] = {"threshold": a_gate, "pass": bool(a["host_macro"] == a["host_macro"] and a["host_macro"] >= a_gate)}
    # U per (source, sequence); the gate on the reference sequence of each source
    u = {"per_source_sequence": {}, "per_source": {}, "margin": u_margin}
    for s in sources:
        seqs = sorted({r["sequence"] for r in results if r["source"] == s and r.get("u")})
        ref = reference.get(s)
        for q in seqs:
            scans = [scan(r["case"], r["u"]["gt"], r["u"]["dets"]) for r in results if r["source"] == s and r["sequence"] == q and r.get("u")]
            judged = ref is not None and q == REFERENCE.get(s, {}).get("sequence")
            entry = _u_entry(scans, ref if judged else None, fixed_thresholds.get(s), u_margin)
            entry.update({"sequence": q, "reference": ref if judged else None, "gated": judged,
                          "grid": REFERENCE.get(s, {}).get("grid") if judged else None})
            u["per_source_sequence"][f"{s}/{q}"] = entry
            if judged:
                u["per_source"][s] = entry
    judged = [e["pass"] for e in u["per_source"].values() if e["pass"] is not None]
    u["gate"] = {"pass": bool(judged) and all(judged), "sources_judged": len(judged)}
    # R, controlled
    instances = [dict(i, source=r["source"], sequence=r["sequence"]) for r in results if r.get("r") for i in r["r"]["instances"]]
    controlled = _r_block(instances) if instances else {"n": 0}
    controlled["per_source"] = {s: _r_block([i for i in instances if i["source"] == s]) for s in sources if any(i["source"] == s for i in instances)}
    controlled["per_stratum"] = {st: _r_block([i for i in instances if i["stratum"] == st]) for st in STRATA if any(i["stratum"] == st for i in instances)}
    # the recognition–binding gap (spec 方案实验 1): local A Dice >= LOCAL_DICE_MIN and a detection matched at the chosen threshold
    def thr_of(i):
        e = u["per_source"].get(i["source"])
        return e.get("thr") if e else None
    gap_pool = [i for i in instances if i.get("local_dice") is not None and i["local_dice"] >= LOCAL_DICE_MIN and i.get("matched_score") is not None
                and thr_of(i) is not None and i["matched_score"] >= thr_of(i)]
    gap = _r_block(gap_pool) if gap_pool else {"n": 0}
    gap["condition"] = f"local 13-host Dice >= {LOCAL_DICE_MIN} in the bind window and a detection matched at the source's U threshold"
    # R, end to end: the predicted instances matched to a truth at the chosen threshold
    e2e_pool = []
    for r in results:
        if not r.get("e2e"):
            continue
        e = u["per_source"].get(r["source"])
        thr = e.get("thr") if e else None
        for d in r["e2e"]:
            if d["truth"] is not None and thr is not None and d["score"] >= thr:
                e2e_pool.append(dict(d, source=r["source"]))
    e2e = _r_block(e2e_pool) if e2e_pool else {"n": 0}
    r = {"controlled": controlled, "gap": gap, "end_to_end": e2e,
         "gate": {"pass": bool(instances) and controlled["aba_r"] >= controlled["aba_b0"], "n_instances": len(instances)},
         "truth_note": "the controlled-track truth is the SynthSeg lookup itself (host_targets on the SynthSeg map); no B0* column, it would be 1.0 by construction"}
    seq = [x for x in results if "seq_pred" in x]
    s_acc = float(np.mean([x["seq_pred"] == x["seq_truth"] for x in seq])) if seq else float("nan")
    return {"evidence": "NOT_EVIDENCE", "n_results": len(results), "sources": sources, "a": a, "u": u, "r": r,
            "s": {"accuracy": s_acc, "n": len(seq)}, "not_implemented": list(NOT_IMPLEMENTED)}


def _f(v, d=4):
    return "nan" if v is None or v != v else f"{v:.{d}f}"


def markdown(agg):
    lines = [f"# AnatoBind-Brain test-set evaluation ({agg['evidence']}: every number rests on SynthSeg / lesion pseudo-labels)", ""]
    a = agg["a"]
    lines += ["## A (gate G2, part 1)", "", f"13-host macro Dice {_f(a['host_macro'])} over {a['n_cases']} A-supervised volumes (entity macro {_f(a['entity_macro'])}); "
              f"gate ≥ {a['gate']['threshold']}: **{'pass' if a['gate']['pass'] else 'fail'}**. Unsupervised (QC-failed) rows: {a['unsupervised']['n_cases']}, "
              f"host macro {_f(a['unsupervised']['host_macro'])} (not gated).", "", "| source | volumes | host macro Dice |", "|---|---|---|"]
    lines += [f"| {s} | {v['n_cases']} | {_f(v['host_macro'])} |" for s, v in a["per_source"].items()]
    lines += ["", "| host | Dice |", "|---|---|"] + [f"| {h} | {_f(d)} |" for h, d in a["per_host"].items()]
    u = agg["u"]
    lines += ["", "## U (gate G2, part 2)", "", f"Sensitivity at the reference detector's false positives per scan on the same patients; pass = ≥ reference − {u['margin']}; "
              f"the gate reads one sequence per source (the one with the reference's lesion labels); gate: **{'pass' if u['gate']['pass'] else 'fail'}** "
              f"({u['gate']['sources_judged']} sources judged)", "",
              "| source/sequence | gated | scans | GT (ignored) | thr | selection | sensitivity | FP/scan | budget | ref sens | ref thr | ref scans | ref GT | grid | <5 mm | 5–10 mm | ≥10 mm | pass |",
              "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for key, e in u["per_source_sequence"].items():
        ref = e.get("reference") or {}
        st = e.get("strata", {})
        lines.append(f"| {key} | {e['gated']} | {e['n_scans']} | {e['n_gt']} ({e['n_ignored']}) | {_f(e.get('thr'), 2)} | {e.get('threshold_selection') or '—'} | "
                     f"{_f(e.get('sensitivity'))} | {_f(e.get('fp_per_scan'), 3)} | {_f(e.get('budget'), 3)} | {_f(ref.get('sensitivity'))} | {_f(ref.get('thr'), 2)} | "
                     f"{ref.get('n_scans', '—')} | {ref.get('n_gt', '—')} | {e.get('grid') or '—'} | {_f(st.get('<5', {}).get('sensitivity'))} | "
                     f"{_f(st.get('5-10', {}).get('sensitivity'))} | {_f(st.get('>=10', {}).get('sensitivity'))} | {e['pass']} |")
    c = agg["r"]["controlled"]
    lines += ["", "## R, controlled track (gate G3)", "", f"{c.get('n', 0)} instances (zero-overlap bindings {c.get('n_zero_overlap', 0)}); "
              f"ABA(R) {_f(c.get('aba_r'))} vs ABA(B0) {_f(c.get('aba_b0'))}; side R {_f(c.get('side_r'))} / B0 {_f(c.get('side_b0'))}; "
              f"tissue R {_f(c.get('tissue_r'))} / B0 {_f(c.get('tissue_b0'))}; gate ABA(R) ≥ ABA(B0): **{'pass' if agg['r']['gate']['pass'] else 'fail'}**. "
              f"{agg['r']['truth_note']}.", ""]
    if c.get("rescue_harm"):
        rh = c["rescue_harm"]
        lines += [f"All controlled instances: B0 wrong on {rh['b0_wrong_share']:.4f}; R rescues {rh['rescue']}, harms {rh['harm']}, both right {rh['both_right']}, both wrong {rh['both_wrong']}.", ""]
    g = agg["r"]["gap"]
    if g.get("n"):
        rh = g["rescue_harm"]
        lines += [f"Recognition–binding gap ({g['condition']}): {g['n']} instances, B0 wrong on {rh['b0_wrong_share']:.4f}; R rescues {rh['rescue']}, harms {rh['harm']}.", ""]
    else:
        lines += [f"Recognition–binding gap: no instance meets the condition ({g.get('condition', '')}).", ""]
    e = agg["r"]["end_to_end"]
    lines += [f"End to end (predicted lesions matched to a truth at the U threshold): {e.get('n', 0)} lesions, ABA(R) {_f(e.get('aba_r'))} vs ABA(B0) {_f(e.get('aba_b0'))}.", ""]
    lines += ["| split | n | ABA(R) | ABA(B0) | rescue | harm |", "|---|---|---|---|---|---|"]
    for name, block in list(c.get("per_source", {}).items()) + list(c.get("per_stratum", {}).items()):
        lines.append(f"| {name} | {block['n']} | {_f(block['aba_r'])} | {_f(block['aba_b0'])} | {block['rescue_harm']['rescue']} | {block['rescue_harm']['harm']} |")
    lines += ["", f"Sequence-type accuracy {_f(agg['s']['accuracy'])} over {agg['s']['n']} volumes.", "",
              "Not in this report: " + "; ".join(agg.get("not_implemented", [])) + ".", ""]
    return "\n".join(lines)


def level_r_rows(results):
    """One row per controlled-track instance for the Level R readers: what to read (case, sequence, box, volume) in
    `sheet`, the model's and the lookup's answers in `key` (kept apart so that the readers stay blind)."""
    sheet, key = [], []
    for r in results:
        if not r.get("r"):
            continue
        for i in r["r"]["instances"]:
            ident = {"source": r["source"], "case": r["case"], "sequence": r["sequence"], "lesion_id": i["instance"]}
            sheet.append({**ident, "box_zyx": i.get("box"), "volume_mm3": i["volume_mm3"], "stratum": i["stratum"],
                          "reader1_host": "", "reader1_side": "", "reader2_host": "", "reader2_side": "", "arbitration_host": "", "arbitration_side": ""})
            key.append({**ident, "r_host": HOST_NAMES[i["r"]], "b0_host": HOST_NAMES[i["b0"]], "pseudo_truth_host": HOST_NAMES[i["truth"]],
                        "r_probs": i["r_probs"], "query_iou": i["query_iou"], "zero_overlap": i.get("zero_overlap")})
    return sheet, key


def write_report(out_dir, agg, results):
    import csv
    out_dir.mkdir(parents=True, exist_ok=False)
    (out_dir / "aggregate.json").write_text(json.dumps(agg, indent=1, default=str))
    (out_dir / "cases.json").write_text(json.dumps(results, indent=1, default=str))
    (out_dir / "REPORT.md").write_text(markdown(agg))
    sheet, key = level_r_rows(results)
    if sheet:
        with open(out_dir / "level_r_sheet.csv", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(sheet[0].keys()))
            w.writeheader()
            w.writerows(sheet)
        (out_dir / "level_r_key.json").write_text(json.dumps(key, indent=1, default=str))


# ---- the reference detector on the same patients: S7's out-of-fold nnU-Net predictions on their native grids ----
from pathlib import Path  # noqa: E402

import nibabel as nib  # noqa: E402

from anatobind.eval.brain_disease import case_scan  # noqa: E402  (module attribute: tests replace it)
from anatobind.nnunet.brain_disease import DISEASES, anatomy_path, fold_dir  # noqa: E402

NNUNET_ROOT = Path("/data2/congcong/data/FM_data/derived/nnunet")


def reference_scans(source, cases, nnunet_root=NNUNET_ROOT, anatomy_of=anatomy_path):
    """S7's out-of-fold predictions of the cases of a source as detection scans (native grid, S7's own ground-truth
    components and binding). Raises KeyError for a source without a reference detector."""
    disease = REFERENCE[source]["disease"]
    name = DISEASES[disease]["name"]
    root = Path(nnunet_root)
    splits = json.loads((root / "preprocessed" / name / "splits_final.json").read_text())
    fold_of = {c: f for f, s in enumerate(splits) for c in s["val"]}
    jobs = []
    for case in sorted(set(cases)):
        if case not in fold_of:
            raise ValueError(f"{case} is in no validation fold of {name}: no out-of-fold prediction")
        d = fold_dir(root / "results", disease, fold_of[case]) / "validation"
        label = root / "raw" / name / "labelsTr" / f"{case}.nii.gz"
        vox = abs(float(np.linalg.det(nib.load(str(label)).affine[:3, :3])))
        jobs.append((case, disease, label, d / f"{case}.nii.gz", d / f"{case}.npz", anatomy_of(disease, case), vox))
    return [case_scan(j) for j in jobs]


def reference_summary(source, scans):
    """The reference detector at its S7 threshold on these scans."""
    at = sensitivity_at_threshold(scans, REFERENCE[source]["thr"])
    return {**at, "disease": REFERENCE[source]["disease"], "source": source}
