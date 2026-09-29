#!/usr/bin/env python
# scripts/eval_brain_disease.py
"""Out-of-fold evaluation of one brain disease detector (spec 2026-09-29 §5, §6).

  PYTHONNOUSERSITE=1 PYTHONPATH=. nice -n 19 ~/anaconda3/envs/nvgen/bin/python scripts/eval_brain_disease.py \
      --disease glioma --folds 0 --out docs/verification/2026-09-30/brain_multidisease/glioma_fold0
  ... --disease glioma --folds 0 1 2 3 4 --out DIR --records /data2/.../derived/brain_disease/glioma/records
"""
import argparse
import json
import sys
from multiprocessing import Pool
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from anatobind.eval.brain_disease import (  # noqa: E402
    binding_agreement, case_scan, dice_summary, evaluate, false_positive_spread, jobs, strata, verdict,
)
from anatobind.eval.lesion_components import MIN_MM3, STRATA, size_stratum  # noqa: E402
from anatobind.infer.brain_disease import study_record  # noqa: E402
from anatobind.nnunet.brain_disease import DISEASES, FM, anatomy_path  # noqa: E402

NNUNET = FM / "derived/nnunet"


def table(title, st, order):
    out = [f"### {title}\n\n| stratum | N (GT) | N (hit) | sensitivity |\n|---|---|---|---|\n"]
    out += [f"| {k} | {st[k]['n_gt']} | {st[k]['n_hit']} | {st[k]['sensitivity']:.4f} |\n" for k in order if k in st]
    return out + ["\n"]


def main(argv=None):
    ap = argparse.ArgumentParser(description="Out-of-fold evaluation of one brain disease detector")
    ap.add_argument("--disease", choices=sorted(DISEASES), required=True)
    ap.add_argument("--folds", type=int, nargs="+", required=True)
    ap.add_argument("--out", type=Path, required=True, help="report directory (must not exist)")
    ap.add_argument("--records", type=Path, help="directory for one structured record per case (must not exist)")
    ap.add_argument("--workers", type=int, default=8)
    a = ap.parse_args(argv)
    for p in (a.out, a.records):
        if p is not None and p.exists():
            raise FileExistsError(f"{p} already exists")
    name = DISEASES[a.disease]["name"]
    info = json.loads((NNUNET / "raw" / name / "cases.json").read_text())
    splits = json.loads((NNUNET / "preprocessed" / name / "splits_final.json").read_text())
    val = sorted(c for s in splits for c in s["val"])
    if val != sorted(info):
        raise AssertionError(f"{name}: the folds do not cover the {len(info)} cases exactly once")
    todo = jobs(NNUNET / "results", NNUNET / "raw", a.disease, splits, a.folds, info, lambda c: anatomy_path(a.disease, c, FM))
    if a.workers > 1:
        with Pool(a.workers) as pool:
            scans = pool.map(case_scan, todo, chunksize=2)
    else:
        scans = [case_scan(j) for j in todo]
    result = evaluate(scans)
    if result["n_scans"] != sum(len(splits[f]["val"]) for f in a.folds):
        raise AssertionError(f"scored {result['n_scans']} scans, the folds hold {sum(len(splits[f]['val']) for f in a.folds)}")
    v = verdict(result, a.folds)
    thr = v["thr"]
    dice = dice_summary(NNUNET / "results", a.disease, a.folds)

    L = [f"# Brain multi-disease detector: {a.disease} ({name}), folds {sorted(a.folds)}\n\n",
         ("All five folds: the line below is the gate (spec M2).\n\n" if v["kind"] == "gate" else
          "Fold subset: an early reading, NOT the gate (spec M4).\n\n"),
         "## Verdict\n\n```json\n", json.dumps(v, indent=1), "\n```\n\n",
         ("" if thr is not None else "No threshold of the grid keeps the false positives at or below 2 per scan: there is no "
          "operating point, so no sensitivity was measured and strata, binding agreement and records are not produced. "
          "With all five folds this fails the gate; an early reading without an operating point does not stop the "
          "remaining folds by itself (the rule of spec M4 needs a measured sensitivity): that decision is the user's.\n\n"),
         ("" if not (thr is None and v["beyond_budget"] and v["beyond_budget"]["out_of_reach"]) else
          f"In substance this reading is decisive all the same: the row at threshold {v['beyond_budget']['thr']:.2f} still "
          f"exceeds the budget ({v['beyond_budget']['fp_per_scan']:.4f} false positives per scan) and finds "
          f"{v['beyond_budget']['n_hit']} of {v['beyond_budget']['n_gt']} lesions; a threshold that meets the budget lies "
          "above it and is not expected to find more.\n\n"),
         ("" if not (v["early_stop_undecided"] and thr is not None) else
          f"The sensitivity at the operating point is under 0.3, but the row just beyond the budget (threshold "
          f"{v['beyond_budget']['thr']:.2f}: {v['beyond_budget']['n_hit']} of {v['beyond_budget']['n_gt']} lesions at "
          f"{v['beyond_budget']['fp_per_scan']:.4f} false positives per scan) reaches 0.3 or misses it by at most two "
          "lesions, and the matching is redone at every threshold: a threshold between the two rows of the grid may reach "
          "0.3. The early stop is undecided, the remaining folds go on, the decision is the user's.\n\n"),
         f"Scans {result['n_scans']}; ground-truth lesions counted {result['n_gt']}; ignored (< {MIN_MM3:g} mm3) "
         f"{result['n_ignored']}.\n\n",
         "Scores are mean foreground probabilities over components of the argmax map, so they exceed 0.5 by construction: "
         "the rows up to threshold 0.50 are identical and the budget of 2 false positives per scan may not be reached.\n\n",
         "## FROC\n\n| thr | n_hit | sensitivity | FP per scan |\n|---|---|---|---|\n"]
    L += [f"| {r['thr']:.2f} | {r['n_hit']} | {r['sensitivity']:.4f} | {r['fp_per_scan']:.4f} |\n" for r in result["rows"]]
    L += ["\n## Dice (report only, nnU-Net summary.json, cases with ground truth)\n\n```json\n", json.dumps(dice, indent=1), "\n```\n\n"]
    if thr is not None:
        L += ["## False positives per scan at the operating threshold (report only)\n\n```json\n",
              json.dumps(false_positive_spread(scans, thr), indent=1), "\n```\n\n"]
        L += ["## Strata at the operating threshold (report only)\n\n"]
        L += table("Equivalent diameter (mm)", strata(scans, thr, lambda s, r: size_stratum(r["mm3"])), STRATA)
        if a.disease == "metastasis":
            L += table("Prior craniotomy, biopsy or resection",
                       strata(scans, thr, lambda s, r: "yes" if info[s["case"]].get("prior_surgery") else "no"), ("no", "yes"))
        L += ["## Binding agreement (NOT_EVIDENCE: the anatomy is a SynthSeg pseudo-label)\n\n```json\n",
              json.dumps(binding_agreement(scans, thr), indent=1), "\n```\n\n"]
    L += ["## Command\n\n```\n", " ".join(sys.argv), "\n```\n"]
    fold_of = {c: f for f in a.folds for c in splits[f]["val"]}
    records = {}
    if a.records is not None and thr is not None:
        records = {s["case"]: json.dumps(study_record(s["case"], a.disease, thr, s["dets"], [fold_of[s["case"]]]),
                                         ensure_ascii=False, indent=1) for s in scans}
    a.out.mkdir(parents=True)                                   # everything is computed: only now is anything written
    (a.out / "REPORT.md").write_text("".join(L))
    (a.out / "verdict.json").write_text(json.dumps(v, indent=1))
    (a.out / "froc.csv").write_text("thr,n_hit,sensitivity,fp_per_scan\n" + "".join(
        f"{r['thr']:.2f},{r['n_hit']},{r['sensitivity']:.6f},{r['fp_per_scan']:.6f}\n" for r in result["rows"]))
    out = [f"Disease: {a.disease}; folds {sorted(a.folds)}; kind {v['kind']}\n",
           f"Scans: {result['n_scans']}; lesions counted: {result['n_gt']}; ignored: {result['n_ignored']}\n",
           (f"Sensitivity {v['sensitivity']:.4f} at threshold {thr} with {v['fp_per_scan']} FP per scan\n" if thr is not None
            else "No threshold keeps the false positives at or below 2 per scan: no operating point\n"),
           f"Pass: {v['pass']}; stop remaining folds: {v['stop_remaining_folds']}; "
           f"early stop undecided: {v['early_stop_undecided']}\n",
           f"Dice mean {dice['mean']} over {dice['n_cases']} cases\n"]
    (a.out / "output.txt").write_text("".join(out))
    if records:
        a.records.mkdir(parents=True)
        for case, text in records.items():
            (a.records / f"{case}.json").write_text(text)
    print("".join(out), end="")


if __name__ == "__main__":
    main()
