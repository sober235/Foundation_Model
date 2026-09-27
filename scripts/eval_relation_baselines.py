#!/usr/bin/env python
# scripts/eval_relation_baselines.py
"""Score a run's out-of-fold predictions against a label source (spec 2026-09-27 §6.2–6.8).

  PYTHONNOUSERSITE=1 PYTHONPATH=. ~/anaconda3/envs/nvgen/bin/python scripts/eval_relation_baselines.py \
      --run runs/relation/c1_stage1 --table /data2/congcong/data/FM_data/derived/relation/v1 --labels C1 \
      --out docs/verification/<date>/relation_baselines

--labels R is the one-shot final evaluation: it needs --unblind, --sealed-dir and --manifest, and every access is logged
by the sealed loader.
"""
import argparse
import csv
import io
import json
import sys
from contextlib import redirect_stdout
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from anatobind.eval.relation_metrics import (  # noqa: E402
    by_stratum, delta_d_curve, gate_r1, is_correct, mcnemar, patient_bootstrap, rescue_harm, summary,
)
from anatobind.relation.baselines import PRIOR_VARIANTS  # noqa: E402
from anatobind.relation.cv import read_preds  # noqa: E402
from anatobind.relation.labels import acceptable_matrix, c1_labels, r_test_labels  # noqa: E402
from anatobind.relation.table import load_table  # noqa: E402

STRATA = ("band", "stratum_geometry", "stratum_series", "lesion_type", "is_3mm")


def md_table(headers, rows):
    lines = ["| " + " | ".join(headers) + " |", "|" + "---|" * len(headers)]
    lines += ["| " + " | ".join(str(x) for x in r) + " |" for r in rows]
    return "\n".join(lines)


def fmt(x):
    return "" if x is None else f"{x:.4f}" if isinstance(x, float) else str(x)


def evaluate(run_dir, table, labels, excluded, evidence, n_boot):
    run = json.loads((Path(run_dir) / "run.json").read_text(encoding="utf-8"))
    has, acc = acceptable_matrix(labels, table.lesion_id)
    keep = np.nonzero(has)[0]
    arms = {}
    for p in sorted((Path(run_dir) / "preds").glob("*.csv")):
        pr = read_preds(p)
        order = [pr["lesion_id"].tolist().index(int(l)) for l in table.lesion_id]
        arms[p.stem] = pr["probs"][order][keep]
    missing = [name for name, present in (("b0", "b0" in arms), ("bprior_*", any(a.startswith("bprior_") for a in arms)),
                                          ("bgeo_*", any(a.startswith("bgeo_") for a in arms))) if not present]
    if missing:
        print(f"{run_dir}: evaluation needs b0, at least one bprior_* and at least one bgeo_* arm; missing {missing} "
              f"(found {sorted(arms)}). Run scripts/run_relation_baselines.py with those arms included.", file=sys.stderr)
        sys.exit(2)
    acc, patients = acc[keep], table.patients()[keep]
    c1 = table.c1_slot()[keep]
    bgeo_best = run.get("bgeo_best") or next(a for a in arms if a.startswith("bgeo_"))
    # spec §6.3: the Bprior comparator is the best of the four variants on these labels (conservative for any model
    # compared with it); plain max breaks only exact ties, keeping the first maximal variant in PRIOR_VARIANTS order
    # (there is no tolerance here, unlike select_config's P11 rule)
    prior_arms = [f"bprior_{v}" for v in PRIOR_VARIANTS if f"bprior_{v}" in arms]
    prior_best = max(prior_arms, key=lambda name: is_correct(arms[name], acc).mean())
    comparators = {"b0": arms["b0"], "bprior": arms[prior_best], "bgeo": arms[bgeo_best], "b2": arms.get("b2")}
    comparators = {k: v for k, v in comparators.items() if v is not None}
    correct = {a: is_correct(p, acc) for a, p in arms.items()}
    rows_long, report = [], []
    report.append(f"# Relation baselines report ({evidence})\n")
    if evidence == "NOT_EVIDENCE":
        report.append("**NOT_EVIDENCE**: every number below is scored against the pseudo label C1, which is a function of the "
                      "geometry columns the arms read; it proves the pipeline runs and is fair, not that any arm binds better.\n")
    report.append(f"run: {run_dir} (git {run.get('git_commit', '')}, table manifest {run.get('table_manifest_sha256', '')[:12]}, labels {run.get('labels')})\n")
    report.append(f"lesions scored: {len(keep)} of {len(table)}; excluded (not_a_lesion): {len(excluded)}\n")
    report.append("## Arms and metrics\n")
    rows = []
    for a, p in arms.items():
        s = summary(p, acc)
        agree = float((p.argmax(1) == c1).mean())
        rows.append([a, s["n"], fmt(s["accuracy"]), fmt(s["singleton_rate"]), fmt(s["top2_accuracy"]), fmt(s["singleton"]["accuracy"]),
                     fmt(s["singleton"]["macro_f1"]), fmt(s["singleton"]["balanced_accuracy"]), fmt(agree)])
        for m, v in (("accuracy", s["accuracy"]), ("singleton_rate", s["singleton_rate"]), ("top2_accuracy", s["top2_accuracy"]),
                     ("singleton_accuracy", s["singleton"]["accuracy"]), ("macro_f1", s["singleton"]["macro_f1"]),
                     ("balanced_accuracy", s["singleton"]["balanced_accuracy"]), ("agreement_with_c1", agree)):
            rows_long.append({"arm": a, "metric": m, "stratum": "all", "value": v, "lo": "", "hi": ""})
    report.append(md_table(["arm", "n", "accuracy", "singleton rate", "top-2", "singleton acc", "macro-F1", "balanced acc", "agreement with C1"], rows) + "\n")
    report.append(f"Bgeo+ comparator: {bgeo_best} (selection {json.dumps(run.get('bgeo_selection', {}))}); "
                  f"Bprior comparator: {prior_best} (best of the four variants on these labels)\n")
    report.append("## Rescue / harm against the comparators (patient bootstrap 95% CI, exact McNemar)\n")
    rows = []
    for a in arms:
        if a in ("b0",) or a.startswith("bprior"):
            continue
        for cname, cp in comparators.items():
            if cname == "b2" and a == "b2":
                continue
            rh = rescue_harm(correct[a], is_correct(cp, acc))
            diff = correct[a].astype(float) - is_correct(cp, acc).astype(float)
            mean, lo, hi = patient_bootstrap(diff, patients, n_boot)
            pv = mcnemar(rh["rescue"], rh["harm"])
            rows.append([a, cname, rh["rescue"], rh["harm"], rh["net"], fmt(mean), fmt(lo), fmt(hi), fmt(pv)])
            rows_long.append({"arm": a, "metric": f"net_rescue_vs_{cname}", "stratum": "all", "value": mean, "lo": lo, "hi": hi})
    report.append(md_table(["arm", "comparator", "rescue", "harm", "net", "net rate", "CI lo", "CI hi", "McNemar p"], rows) + "\n")
    report.append("## Strata (set-valued accuracy)\n")
    for key in STRATA:
        keys = table.column(key)[keep].astype(str)
        rows = []
        for a in arms:
            st = by_stratum(correct[a].astype(float), keys)
            rows.append([a] + [f"{st[s]['mean']:.4f} (n={st[s]['n']})" for s in sorted(st)])
            for s in st:
                rows_long.append({"arm": a, "metric": "accuracy", "stratum": f"{key}={s}", "value": st[s]["mean"], "lo": "", "hi": ""})
        report.append(f"### {key}\n" + md_table(["arm"] + sorted(set(keys.tolist())), rows) + "\n")
    report.append("## Net rescue vs Bgeo+ along delta_d (patient bootstrap)\n")
    dd = table.column("delta_d_mm")[keep].astype(float)
    for a in ("b1", "b2"):
        if a not in arms:
            continue
        diff = correct[a].astype(float) - is_correct(comparators["bgeo"], acc).astype(float)
        curve = delta_d_curve(diff, dd, patients, n_boot=n_boot)
        report.append(f"### {a}\n" + md_table(["bin", "n", "mean", "lo", "hi"], [[c["bin"], c["n"], fmt(c["mean"]), fmt(c["lo"]), fmt(c["hi"])] for c in curve]) + "\n")
        for c in curve:
            rows_long.append({"arm": a, "metric": "net_rescue_vs_bgeo", "stratum": f"delta_d={c['bin']}", "value": c["mean"], "lo": c["lo"], "hi": c["hi"]})
    report.append("## Gate R1 rehearsal (v2.6 §12.4)\n")
    for a in ("b1", "b2"):
        if a in arms and "b2" in comparators:
            g = gate_r1(arms[a], {"bprior": comparators["bprior"], "bgeo": comparators["bgeo"], "b2": comparators["b2"]}, acc, patients, n_boot)
            report.append(f"- {a}: go={g['go']} (net rescue vs Bgeo+ {g['net_rescue_vs_bgeo']['mean']:.4f} [{g['net_rescue_vs_bgeo']['lo']:.4f}, "
                          f"{g['net_rescue_vs_bgeo']['hi']:.4f}]; > Bprior {g['gt_bprior']}, > Bgeo+ {g['gt_bgeo']}, > B2 {g['gt_b2']})")
    report.append("\n## Sanity checks\n")
    report.append("- agreement with C1 per arm is in the metrics table: Bgeo+ close to 1 and B2 clearly lower is the written evidence that C1 is a function of the geometry columns.")
    ties = []
    for name, rec in run.get("records", {}).items():
        for k, f in rec.get("folds", {}).items():
            if f.get("tie"):
                ties.append(f"{name} fold {k}: chose {f['chosen']} among {f['scores']} (tie rule P11)")
    report.append("- inner-selection ties: " + (f"{len(ties)}\n  - " + "\n  - ".join(ties) if ties else "none"))
    report.append("\n## Commands\n")
    report.append("```\nscripts/run_relation_baselines.py (see run.json)\nscripts/eval_relation_baselines.py " + " ".join(sys.argv[1:]) + "\n```\n")
    return "\n".join(report), rows_long


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", type=Path, required=True)
    ap.add_argument("--table", type=Path, required=True)
    ap.add_argument("--labels", choices=("C1", "R"), required=True)
    ap.add_argument("--unblind", action="store_true")
    ap.add_argument("--sealed-dir", type=Path)
    ap.add_argument("--manifest", type=Path)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--n-boot", type=int, default=10_000)
    a = ap.parse_args(argv)
    table, _ = load_table(a.table, with_patches=False)
    if a.labels == "C1":
        labels, excluded, evidence = c1_labels(table), [], "NOT_EVIDENCE"
    else:
        if not a.unblind or not (a.sealed_dir and a.manifest):
            print("--labels R is the one-shot final evaluation: it needs --unblind, --sealed-dir and --manifest (v2.6 §12.7)",
                  file=sys.stderr)
            sys.exit(2)
        labels, excluded = r_test_labels(a.sealed_dir, a.manifest, unblind=True)
        evidence = "R"
    buf = io.StringIO()
    with redirect_stdout(buf):
        report, rows_long = evaluate(a.run, table, labels, excluded, evidence, a.n_boot)
        print(report)
    a.out.mkdir(parents=True, exist_ok=True)
    (a.out / "REPORT.md").write_text(report, encoding="utf-8")
    with open(a.out / "tables.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=["arm", "metric", "stratum", "value", "lo", "hi"])
        w.writeheader()
        w.writerows(rows_long)
    (a.out / "output.txt").write_text(buf.getvalue(), encoding="utf-8")
    print(report)


if __name__ == "__main__":
    main()
