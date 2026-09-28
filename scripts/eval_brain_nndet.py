#!/usr/bin/env python
# scripts/eval_brain_nndet.py
"""nnDetection second arm: coordinate check or fold evaluation (spec 2026-09-28 brain-nndet §5, §7).

  PYTHONNOUSERSITE=1 PYTHONPATH=. ~/anaconda3/envs/nvgen/bin/python scripts/eval_brain_nndet.py --gt-check GT.json --out DIR
  PYTHONNOUSERSITE=1 PYTHONPATH=. ~/anaconda3/envs/nvgen/bin/python scripts/eval_brain_nndet.py \
      --dets fold0_default.json --swept fold0_swept.json --folds 0 --out DIR
"""
import argparse
import hashlib
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from anatobind.eval.brain_detector import evaluate, normal_fp_per_scan, strata_maps, strata_sensitivity  # noqa: E402
from anatobind.eval.brain_nndet import (  # noqa: E402
    gt_check, gt_of_cases, nndet_scans, nnunet_scans, paired_table, rule_a,
)
from anatobind.eval.detection_metrics import operating_point  # noqa: E402
from anatobind.level_r.registry import BANDS, load_registry  # noqa: E402
from anatobind.nndet.boxes import load_runner_json  # noqa: E402
from anatobind.nndet.brain_task import TASK_NAME  # noqa: E402
from anatobind.nnunet.brain_lesion import DATASET_NAME  # noqa: E402

FM = Path("/data2/congcong/data/FM_data")
CASES_JSON = FM / "derived/nnunet/raw" / DATASET_NAME / "cases.json"
SPLITS_JSON = FM / "derived/nnunet/preprocessed" / DATASET_NAME / "splits_final.json"
NNUNET_RESULTS = FM / "derived/nnunet/results"
INSTANCE_MAP = FM / "derived/nndet" / TASK_NAME / "instances.json"
EXPECTED_N_GT = 1297
EXPECTED_N_SCANS = 253
BASELINE = "2d"
REPORT_ONLY = ("3d_fullres",)
STRATA_ORDER = {"band": list(BANDS), "n_slices": ["1", ">1"], "inplane_tertile": ["tertile_1", "tertile_2", "tertile_3"]}


def sha256(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def run_gt_check(a, gt_of_case, info):
    cases = load_runner_json(a.gt_check)["cases"]
    res = gt_check(cases, json.loads(Path(INSTANCE_MAP).read_text()), gt_of_case)
    scans = nndet_scans(cases, sorted(info), gt_of_case)
    row = evaluate(scans, {c for c, v in info.items() if v["kind"] == "normal"})["rows"][0]
    ious = sorted(res["iou"].values())
    summary = {"pass": not res["below_iou"] and row["n_fp"] == 0, "n_gt": row["n_gt"], "n_present": len(ious),
               "lost": res["lost"], "below_iou": res["below_iou"], "min_iou": ious[0] if ious else None,
               "median_iou": ious[len(ious) // 2] if ious else None, "n_iou_ge_0_99": sum(i >= 0.99 for i in ious),
               "n_hit_at_0_05": row["n_hit"], "n_fp": row["n_fp"], "input": str(a.gt_check),
               "input_sha256": sha256(a.gt_check)}
    a.out.mkdir(parents=True)
    (a.out / "gt_check.json").write_text(json.dumps(summary, indent=1))
    lines = ["# nnDetection coordinate check: ground truth as prediction (spec §5 check two)\n\n",
             f"Pass: {summary['pass']}\n\n", "```json\n", json.dumps(summary, indent=1), "\n```\n\n",
             "## Command\n\n```\n", " ".join(sys.argv), "\n```\n"]
    (a.out / "gt_check.md").write_text("".join(lines))
    print(json.dumps(summary))
    if not summary["pass"]:
        raise SystemExit(1)


def _gate_block(title, ev, normal):
    g, thr = ev["gate"], ev["gate"]["thr"]
    nfp = normal_fp_per_scan(ev["scans"], normal, thr) if thr is not None else None
    return [f"## {title}\n\n```json\n", json.dumps(g, indent=1), "\n```\n\n",
            f"Normal-volume FP per volume at the operating threshold: {'n/a' if nfp is None else f'{nfp:.4f}'}\n\n"]


def run_folds(a, gt_of_case, info, registry):
    splits = json.loads(Path(SPLITS_JSON).read_text())
    cases = [c for f in a.folds for c in splits[f]["val"]]
    normal = {c for c, v in info.items() if v["kind"] == "normal"}

    def ev_of(scans):
        ev = evaluate(scans, normal)
        ev["scans"] = scans
        return ev

    nd = ev_of(nndet_scans(load_runner_json(a.dets)["cases"], cases, gt_of_case))
    base = ev_of(nnunet_scans(NNUNET_RESULTS, BASELINE, splits, a.folds, gt_of_case))
    others = {cfg: ev_of(nnunet_scans(NNUNET_RESULTS, cfg, splits, a.folds, gt_of_case)) for cfg in REPORT_ONLY}
    swept = ev_of(nndet_scans(load_runner_json(a.swept)["cases"], cases, gt_of_case)) if a.swept else None
    ra = rule_a(nd["rows"], base["rows"], nd["n_gt"])
    op_n, op_b = operating_point(nd["rows"]), operating_point(base["rows"])
    paired = paired_table(nd["scans"], op_n["thr"], base["scans"], op_b["thr"]) if op_n and op_b else None
    five = sorted(a.folds) == [0, 1, 2, 3, 4]

    a.out.mkdir(parents=True)
    L = [f"# nnDetection second arm, folds {a.folds}\n\n",
         ("Five folds: the nnDetection gate line below is the D1 gate.\n\n" if five else
          "Fold subset: these numbers decide rule A only; they are NOT the D1 gate.\n\n"),
         f"Inputs: `{a.dets}` sha256 {sha256(a.dets)}" + (f"; swept `{a.swept}` sha256 {sha256(a.swept)}" if a.swept else "") + "\n\n",
         "## Rule A (spec N3)\n\n```json\n", json.dumps(ra, indent=1), "\n```\n\n"]
    L += _gate_block("nnDetection, default postprocessing", nd, normal)
    L += ["### FROC\n\n| thr | n_hit | sensitivity | FP per volume |\n|---|---|---|---|\n"]
    L += [f"| {r['thr']:.2f} | {r['n_hit']} | {r['sensitivity']:.4f} | {r['fp_per_scan']:.4f} |\n" for r in nd["rows"]]
    L += ["\n"]
    L += _gate_block(f"nnU-Net {BASELINE}, same folds (rule A baseline)", base, normal)
    for cfg, ev in others.items():
        L += _gate_block(f"nnU-Net {cfg}, same folds (report only)", ev, normal)
    L += ["## Paired table (each model at its own operating point, not gated)\n\n"]
    if paired:
        L += ["| group | lesions |\n|---|---|\n", f"| both | {paired['both']} |\n",
              f"| only nnDetection | {paired['only_a']} |\n", f"| only nnU-Net {BASELINE} | {paired['only_b']} |\n",
              f"| neither | {paired['neither']} |\n\n"]
    else:
        L += ["No operating point for one of the two models.\n\n"]
    if op_n:
        maps = strata_maps(registry)
        L += ["## Strata (nnDetection at its operating point, not gated)\n\n"]
        for name, m in maps.items():
            st = strata_sensitivity(nd["scans"], op_n["thr"], m)
            keys = [k for k in STRATA_ORDER.get(name, sorted(st)) if k in st]
            L += [f"### {name}\n\n| stratum | N (GT) | N (hit) | sensitivity |\n|---|---|---|---|\n"]
            L += [f"| {k} | {st[k]['n_gt']} | {st[k]['n_hit']} | {st[k]['sensitivity']:.4f} |\n" for k in keys]
            L += ["\n"]
    if swept:
        L += _gate_block("Swept postprocessing (NOT_GATE: parameters tuned on these validation cases)", swept, normal)
    L += ["## Command\n\n```\n", " ".join(sys.argv), "\n```\n"]
    (a.out / "REPORT.md").write_text("".join(L))
    (a.out / "rule_a.json").write_text(json.dumps(ra, indent=1))
    (a.out / "froc.csv").write_text("thr,n_hit,sensitivity,fp_per_scan\n" + "".join(
        f"{r['thr']:.2f},{r['n_hit']},{r['sensitivity']:.6f},{r['fp_per_scan']:.6f}\n" for r in nd["rows"]))
    out = [f"Folds: {a.folds}; scans {nd['n_scans']}; lesions {nd['n_gt']}\n", f"Rule A pass: {ra['pass']}\n"]
    if op_n:
        out.append(f"nnDetection operating threshold {op_n['thr']:.2f}: hits {op_n['n_hit']}, "
                   f"sensitivity {op_n['sensitivity']:.4f}, FP per volume {op_n['fp_per_scan']:.4f}\n")
        out.append(f"Normal FP per volume at {op_n['thr']:.2f}: {normal_fp_per_scan(nd['scans'], normal, op_n['thr']):.4f}\n")
    if op_b:
        out.append(f"nnU-Net {BASELINE} operating threshold {op_b['thr']:.2f}: hits {op_b['n_hit']}, "
                   f"sensitivity {op_b['sensitivity']:.4f}, FP per volume {op_b['fp_per_scan']:.4f}\n")
    (a.out / "output.txt").write_text("".join(out))
    print("".join(out), end="")


def main(argv=None):
    ap = argparse.ArgumentParser(description="nnDetection second arm: coordinate check or fold evaluation")
    ap.add_argument("--out", type=Path, required=True, help="output directory (must not exist)")
    mode = ap.add_mutually_exclusive_group(required=True)
    mode.add_argument("--gt-check", type=Path, help="runner gt JSON")
    mode.add_argument("--dets", type=Path, help="runner extract JSON, default parameters (the only input rule A reads)")
    ap.add_argument("--swept", type=Path, help="runner extract JSON, swept parameters (NOT_GATE)")
    ap.add_argument("--folds", type=int, nargs="+", default=[0])
    a = ap.parse_args(argv)
    if a.out.exists():
        raise FileExistsError(f"--out {a.out} already exists")
    info = json.loads(Path(CASES_JSON).read_text())
    registry = load_registry()
    gt_of_case = gt_of_cases({c: v["kind"] for c, v in info.items()}, registry)
    n_gt = sum(len(v) for v in gt_of_case.values())
    if n_gt != EXPECTED_N_GT or len(info) != EXPECTED_N_SCANS:
        raise AssertionError(f"expected {EXPECTED_N_GT} lesions in {EXPECTED_N_SCANS} scans, got {n_gt} in {len(info)}")
    if a.gt_check:
        run_gt_check(a, gt_of_case, info)
    else:
        run_folds(a, gt_of_case, info, registry)


if __name__ == "__main__":
    main()
