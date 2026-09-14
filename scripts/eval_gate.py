#!/usr/bin/env python
# scripts/eval_gate.py
"""leg 2 的闸门报告与 H2、H3。

  PYTHONNOUSERSITE=1 PYTHONPATH=. ~/anaconda3/envs/nvgen/bin/python scripts/eval_gate.py \
      --out docs/verification/2026-09-14/gate --figs ~/figs/anatobind_leg2
"""
import argparse
import csv
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from anatobind.data_engine.fastmri_knee import EXPORT_ROOT  # noqa: E402
from anatobind.eval.gate import VIEW_ORDER, h2, h3, risk_coverage  # noqa: E402

VIEW_LABELS = {"clean": "clean", "noise_q1": "noise\nq1", "noise_q2": "noise\nq2", "noise_q3": "noise\nq3",
               "us4": "us\n4x", "us8": "us\n8x", "us16": "us\n16x"}
COLORS = {"head": "#2a78d6", "peak": "#eb6834", "rule": "#1baf7a", "min": "#eda100"}


def read_scores(path, export_root):
    rows = []
    with open(path, newline="") as fh:
        for r in csv.DictReader(fh):
            meta = json.loads((Path(export_root) / r["file"] / "meta.json").read_text())
            rows.append({**r, "fold": int(r["fold"]), "correct": float(r["correct"]),
                         "head_score": float(r["head_score"]), "peak_score": float(r["peak_score"]),
                         "min_lesion": float(r["min_lesion"]) if r["min_lesion"] else 0.0,
                         "patient": meta["patient_id"]})
    return rows


def fig_risk_coverage(scan_rows, path):
    fig, axes = plt.subplots(1, len(VIEW_ORDER), figsize=(2.0 * len(VIEW_ORDER), 2.6), sharey=True)
    for ax, v in zip(axes, VIEW_ORDER):
        sel = [r for r in scan_rows if r["view"] == v]
        for key, label in (("head_score", "scan head"), ("peak_score", "peak score")):
            if not sel:
                continue
            cov, risk = risk_coverage([r[key] for r in sel], [r["correct"] for r in sel])
            ax.plot(cov, risk, lw=2, color=COLORS["head" if key == "head_score" else "peak"], label=label)
        ax.axhline(0.05, ls="--", lw=1, color="0.4")
        ax.set_title(VIEW_LABELS[v], fontsize=9)
        ax.set_xlabel("coverage")
    axes[0].set_ylabel("risk among accepted scans")
    axes[0].legend(loc="upper left", fontsize=8, frameon=False)
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def fig_coverage_bars(res, path):
    names = [("coverage_head", "learned scan gate", "head"), ("coverage_rule", "acceleration rule", "rule"),
             ("coverage_min_baseline", "min of lesion scores", "min")]
    fig, ax = plt.subplots(figsize=(5.2, 3.0))
    ax.bar([n[1] for n in names], [res[n[0]] for n in names], color=[COLORS[n[2]] for n in names], width=0.6)
    for i, n in enumerate(names):
        ax.text(i, res[n[0]] + 0.01, f"{res[n[0]]:.2f}", ha="center", fontsize=9)
    ax.set_ylabel(f"coverage at risk <= {res['risk_target']:.2f}")
    ax.set_ylim(0, 1.05)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--scores", type=Path, default=Path("runs/reliability_scores.csv"))
    ap.add_argument("--export-root", type=Path, default=EXPORT_ROOT)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--figs", type=Path, required=True)
    ap.add_argument("--reps", type=int, default=10000)
    a = ap.parse_args(argv)
    a.out.mkdir(parents=True, exist_ok=True)
    a.figs.mkdir(parents=True, exist_ok=True)

    rows = read_scores(a.scores, a.export_root)
    lesion_rows = [r for r in rows if r["kind"] == "lesion"]
    scan_rows = [r for r in rows if r["kind"] == "scan"]
    summary = {"n_lesion_rows": len(lesion_rows), "n_scan_rows": len(scan_rows),
               "n_patients": len({r["patient"] for r in rows}),
               "h2": h2(lesion_rows, reps=a.reps), "h3": h3(scan_rows, reps=a.reps)}
    (a.out / "summary.json").write_text(json.dumps(summary, indent=1))
    with open(a.out / "records.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)

    print("H2 (per-lesion reliability against the peak score, within each view):")
    for v in VIEW_ORDER:
        d = summary["h2"][v]
        print(f"  {v:9} n={d['n']:5d}  delta AURC {d['delta_aurc']:+.4f}  "
              f"95% CI [{d['ci95'][0]:+.4f}, {d['ci95'][1]:+.4f}]  {'pass' if d['pass'] else ''}")
    print(f"H2 VERDICT: {'PASS' if summary['h2']['pass'] else 'FAIL'}")
    r = summary["h3"]
    print(f"\nH3 at risk <= {r['risk_target']}: learned gate {r['coverage_head']:.3f}, "
          f"acceleration rule {r['coverage_rule']:.3f}, min baseline {r['coverage_min_baseline']:.3f}")
    print(f"  vs rule: {r['vs_rule']['delta_coverage']:+.3f} CI {r['vs_rule']['ci95']}")
    print(f"  vs min : {r['vs_min']['delta_coverage']:+.3f} CI {r['vs_min']['ci95']}")
    print(f"H3 VERDICT: {'PASS' if r['pass'] else 'FAIL'}")

    fig_risk_coverage(scan_rows, a.figs / "risk_coverage.png")
    fig_coverage_bars(r, a.figs / "coverage_at_risk.png")
    print(f"\nfigures: {a.figs}/risk_coverage.png, {a.figs}/coverage_at_risk.png")


if __name__ == "__main__":
    main()
