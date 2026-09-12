#!/usr/bin/env python
"""Gates G1 and G2 on the m1r export (RESEARCH_PLAN v2.2 13.6, amended 2026-09-12).

Gate path: each annotated lesion enters the lookup with its annotated box and class, and only the
anatomy is predicted (nnU-Net masks decide G2; our masks are reported). Auxiliary path: our own
detections, matched to the truth, so detection recall and its buckets are still reported.
Predictions come only from the cached files (m1r_pred); the export supplies the truth.

  D=docs/verification/$(date +%F); mkdir -p $D
  PYTHONNOUSERSITE=1 PYTHONPATH=. ~/anaconda3/envs/nvgen/bin/python scripts/eval_g1_g2.py --out $D | tee $D/g1_g2_output.txt
"""
import argparse
import csv
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from anatobind.eval.g2 import GATE_VIEWS, bootstrap_ci, delta, g2, paired_counts  # noqa: E402
from anatobind.eval.lookup import LabelIndex, b0_host  # noqa: E402
from anatobind.eval.matching import BUCKETS, bucket, detected, iou3d, match  # noqa: E402
from anatobind.eval.predict import load_prediction  # noqa: E402
from anatobind.train.cache import VIEWS, load_array, load_meta  # noqa: E402
from anatobind.train.dataset_v2 import read_all_boxes  # noqa: E402

FM = Path("/data2/congcong/data/FM_data/derived/skmtea")
STYLE = Path(__file__).resolve().parents[1] / "docs/figures/anatobind_academic.mplstyle"
THRESHOLDS = (0.05, 0.1, 0.25)
PRIMARY = 0.1
VARIANTS = ("nnunet", "ours")
TITLES = {"nnunet": "B0 on nnU-Net masks (decides G2)", "ours": "B0 on our masks (report only)"}
BUCKET_COLORS = {"correct": "#2a78d6", "wrong_host": "#eb6834", "wrong_class": "#1baf7a", "miss": "#eda100"}
INK, MUTED, GRID, BASELINE = "#0b0b0b", "#898781", "#e1e0d9", "#c3c2b7"
VIEW_LABELS = {"clean": "clean", "noise_q1": "noise\nq1", "noise_q2": "noise\nq2", "noise_q3": "noise\nq3",
               "us4": "us\n4×", "us8": "us\n8×", "us16": "us\n16×"}
FIELDS = ["fold", "scan", "view", "thr", "ann_id", "layer", "gt_cls", "tissue_id", "host_label", "pred_cls", "iou",
          "b0_nnunet", "b0_ours", "bucket_nnunet", "bucket_ours",
          "given_b0_nnunet", "given_b0_ours", "given_bucket_nnunet", "given_bucket_ours"]
GATE_KEY = "given_bucket"          # annotated box and class, predicted anatomy (user decision Q19, 2026-09-12)
AUX_KEY = "bucket"                 # our detections


def parse(argv):
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--export-root", type=Path, default=FM / "m1r")
    ap.add_argument("--cache-root", type=Path, default=FM / "m1r_cache")
    ap.add_argument("--pred-root", type=Path, default=FM / "m1r_pred")
    ap.add_argument("--figs", type=Path, default=Path.home() / "figs" / "anatobind_m1r")
    ap.add_argument("--skip-gap", action="store_true")
    return ap.parse_args(argv)


def dice(a, b):
    s = a.sum() + b.sum()
    return float(2.0 * np.logical_and(a, b).sum() / s) if s else float("nan")


def evaluate(a, folds):
    records = []
    det_counts = defaultdict(Counter)
    for scan, fold in sorted(folds.items()):
        rows = read_all_boxes(a.export_root / scan / "boxes.csv")
        gt = np.array([r["box"] for r in rows], dtype=float).reshape(-1, 6)
        spacing = load_meta(a.cache_root / scan)["spacing_mm"]
        for view in VIEWS:
            ours = load_prediction(a.pred_root / "ours" / f"fold{fold}" / f"{scan}__{view}.npz")
            nn = load_prediction(a.pred_root / "nnunet" / f"{scan}__{view}.npz")
            det = detected(ours["cls_prob"])
            pboxes, pcls = ours["boxes_vox"][det], ours["cls_prob"][det].argmax(-1)
            index = {"ours": LabelIndex(ours["label_map"], spacing), "nnunet": LabelIndex(nn["label_map"], spacing)}
            iou = iou3d(gt, pboxes)
            for thr in THRESHOLDS:
                m = match(gt, pboxes, thr)
                if thr == PRIMARY:
                    det_counts[view]["detections"] += len(det)
                    det_counts[view]["matched"] += len(m)
                    det_counts[view]["truth"] += len(rows)
                for j, r in enumerate(rows):
                    p = m.get(j)
                    pc = None if p is None else int(pcls[p])
                    rec = {"fold": fold, "scan": scan, "view": view, "thr": thr, "ann_id": r["ann_id"],
                           "layer": r["layer"], "gt_cls": r["cls"], "tissue_id": r["tissue_id"],
                           "host_label": r["host_label"], "pred_cls": "" if pc is None else pc,
                           "iou": "" if p is None else round(float(iou[j, p]), 4)}
                    for v in VARIANTS:
                        label = None if p is None else b0_host(index[v], pboxes[p], pc)
                        rec[f"b0_{v}"] = "" if label is None else label
                        rec[f"bucket_{v}"] = bucket(r["cls"], r["tissue_id"], pc, label) if r["layer"] == "in_seg" else ""
                        given = b0_host(index[v], gt[j], r["cls"])
                        rec[f"given_b0_{v}"] = "" if given is None else given
                        rec[f"given_bucket_{v}"] = (bucket(r["cls"], r["tissue_id"], r["cls"], given)
                                                    if r["layer"] == "in_seg" else "")
                    records.append(rec)
    return records, det_counts


def outcomes_of(records, variant, thr=PRIMARY, key=GATE_KEY):
    out = defaultdict(dict)
    for r in records:
        if r["thr"] == thr and r["layer"] == "in_seg":
            out[(r["scan"], r["ann_id"])][r["view"]] = r[f"{key}_{variant}"]
    return out


def bucket_table(records, variant, thr, key=GATE_KEY):
    t = {v: Counter() for v in VIEWS}
    for r in records:
        if r["thr"] == thr and r["layer"] == "in_seg":
            t[r["view"]][r[f"{key}_{variant}"]] += 1
    return t


def paired_by_view(outcomes):
    res = {}
    for v in VIEWS[1:]:
        _, arr = paired_counts(outcomes, v)
        d, rate_ref, rate_view = delta(arr)
        res[v] = {"n_lesions": int(arr[:, 0].sum()), "rate_clean": rate_ref, "rate_view": rate_view,
                  "delta": d, "ci95": list(bootstrap_ci(arr))}
    return res


def geometry_gap(a, folds):
    """Mean Dice over structures of our label maps, held-out versus training scans of each fold model."""
    acc = defaultdict(list)
    for fold in range(5):
        for scan, scan_fold in sorted(folds.items()):
            seg = np.asarray(load_array(a.cache_root / scan, "seg"))
            for view in VIEWS:
                lm = load_prediction(a.pred_root / "ours" / f"fold{fold}" / f"{scan}__{view}.npz")["label_map"]
                d = [dice(lm == k, seg == k) for k in range(1, 7) if (seg == k).any()]
                acc[(view, "held_out" if scan_fold == fold else "training")].append(float(np.mean(d)))
    return {f"{view}|{group}": float(np.mean(v)) for (view, group), v in acc.items()}


def fig_buckets(tables, path):
    plt.style.use(str(STYLE))
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.4), sharey=True)
    x = np.arange(len(VIEWS))
    for ax, variant in zip(axes, VARIANTS):
        t, bottom = tables[variant], np.zeros(len(VIEWS))
        for b in BUCKETS:
            frac = np.array([t[v][b] / max(sum(t[v].values()), 1) for v in VIEWS])
            ax.bar(x, frac, bottom=bottom, width=0.72, color=BUCKET_COLORS[b], edgecolor="white", linewidth=2,
                   label=b.replace("_", " "))
            for xi in range(len(VIEWS)):
                if frac[xi] >= 0.08:
                    ax.text(xi, bottom[xi] + frac[xi] / 2, f"{frac[xi]:.0%}", ha="center", va="center", fontsize=9, color=INK)
            bottom += frac
        ax.set_xticks(x, [VIEW_LABELS[v] for v in VIEWS], fontsize=10)
        ax.set_title(TITLES[variant], fontsize=12, loc="left")
        ax.set_ylim(0, 1)
        ax.yaxis.grid(True, color=GRID, linewidth=0.6)
        ax.set_axisbelow(True)
    axes[0].set_ylabel("share of in-segmentation lesions")
    axes[1].legend(loc="upper left", bbox_to_anchor=(1.0, 1.0))
    fig.tight_layout()
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def fig_paired(pairs, path):
    plt.style.use(str(STYLE))
    views = VIEWS[1:]
    x = np.arange(len(views))
    fig, ax = plt.subplots(figsize=(8.8, 4.4))
    for i, v in enumerate(views):
        if v in GATE_VIEWS:
            ax.axvspan(i - 0.42, i + 0.42, color=GRID, alpha=0.7, linewidth=0)
    ax.axhline(0.0, color=BASELINE, linewidth=1.0)
    ax.axhline(0.05, color=MUTED, linewidth=1.2, linestyle=(0, (4, 3)))
    ax.text(len(views) - 0.55, 0.052, "G2 threshold +5 points", ha="right", va="bottom", fontsize=9, color=MUTED)
    for off, variant, face in ((-0.13, "nnunet", BUCKET_COLORS["wrong_host"]), (0.13, "ours", "white")):
        d = np.array([pairs[variant][v]["delta"] for v in views])
        lo = np.array([pairs[variant][v]["ci95"][0] for v in views])
        hi = np.array([pairs[variant][v]["ci95"][1] for v in views])
        ax.errorbar(x + off, d, yerr=[d - lo, hi - d], fmt="o", markersize=8, color=BUCKET_COLORS["wrong_host"],
                    markerfacecolor=face, markeredgewidth=1.6, elinewidth=2, capsize=0, label=TITLES[variant])
    ax.set_xticks(x, [VIEW_LABELS[v] for v in views], fontsize=10)
    for lab, v in zip(ax.get_xticklabels(), views):
        if v in GATE_VIEWS:
            lab.set_fontweight("bold")
    ax.set_ylabel("wrong-host rate, view minus clean")
    ax.yaxis.grid(True, color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=2)
    fig.tight_layout()
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def main(argv=None):
    a = parse(argv)
    a.out.mkdir(parents=True, exist_ok=True)
    a.figs.mkdir(parents=True, exist_ok=True)
    folds = json.loads((a.export_root / "splits.json").read_text())["folds"]
    missing = [str(p) for scan, f in folds.items() for v in VIEWS
               for p in (a.pred_root / "ours" / f"fold{f}" / f"{scan}__{v}.npz", a.pred_root / "nnunet" / f"{scan}__{v}.npz")
               if not p.exists()]
    print(f"G1: {len(folds) * len(VIEWS)} held-out (scan, view) pairs; {len(missing)} prediction files missing")
    if missing:
        print("\n".join(missing[:10]))
        sys.exit(1)

    records, det_counts = evaluate(a, folds)
    with open(a.out / "records.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=FIELDS)
        w.writeheader()
        w.writerows(records)
    n_in_seg = len({(r["scan"], r["ann_id"]) for r in records if r["layer"] == "in_seg"})
    print(f"in-segmentation lesions scored at family level: {n_in_seg}")

    summary = {"n_scans": len(folds), "n_in_seg": n_in_seg, "detection": {}, "buckets": {}, "g2": {}, "paired": {}}
    print("\ndetection at IoU 0.1 (our U_B): recall = matched / truth over all layers, precision = matched / detections")
    for v in VIEWS:
        c = det_counts[v]
        rec, prec = c["matched"] / max(c["truth"], 1), c["matched"] / max(c["detections"], 1)
        summary["detection"][v] = {"recall": rec, "precision": prec, **c}
        print(f"  {v:9s} recall {rec:.3f}  precision {prec:.3f}  ({c['matched']}/{c['truth']} truth, {c['detections']} detections)")

    print("\nGATE PATH: annotated box and class, predicted anatomy (decision 2026-09-12); "
          "AUXILIARY PATH: our detections at IoU 0.1")
    tables = {}
    for variant in VARIANTS:
        t = bucket_table(records, variant, PRIMARY, GATE_KEY)
        tables[variant] = t
        summary["buckets"][f"given:{variant}"] = {v: dict(t[v]) for v in VIEWS}
        print(f"\nbuckets, gate path, {TITLES[variant]}:")
        print("  " + " " * 9 + "".join(f"{b:>13s}" for b in BUCKETS))
        for v in VIEWS:
            n = max(sum(t[v].values()), 1)
            print(f"  {v:9s}" + "".join(f"{t[v][b] / n:13.3f}" for b in BUCKETS) + f"   n={sum(t[v].values())}")
    for thr in THRESHOLDS:
        for variant in VARIANTS:
            t = bucket_table(records, variant, thr, AUX_KEY)
            summary["buckets"][f"predicted:{variant}@{thr}"] = {v: dict(t[v]) for v in VIEWS}
            print(f"\nbuckets, auxiliary path (our detections), {TITLES[variant]}, IoU {thr}:")
            print("  " + " " * 9 + "".join(f"{b:>13s}" for b in BUCKETS))
            for v in VIEWS:
                n = max(sum(t[v].values()), 1)
                print(f"  {v:9s}" + "".join(f"{t[v][b] / n:13.3f}" for b in BUCKETS) + f"   n={sum(t[v].values())}")

    summary["g2_predicted"] = {variant: g2(outcomes_of(records, variant, key=AUX_KEY)) for variant in VARIANTS}
    for variant in VARIANTS:
        outcomes = outcomes_of(records, variant, key=GATE_KEY)
        summary["g2"][variant] = g2(outcomes)
        summary["paired"][variant] = paired_by_view(outcomes)
        print(f"\nG2 (gate path), {TITLES[variant]}:")
        for v in VIEWS[1:]:
            p = summary["paired"][variant][v]
            gate = " <- gate" if v in GATE_VIEWS else ""
            print(f"  {v:9s} paired n={p['n_lesions']:4d}  clean {p['rate_clean']:.3f}  view {p['rate_view']:.3f}  "
                  f"delta {p['delta']:+.3f}  95% CI [{p['ci95'][0]:+.3f}, {p['ci95'][1]:+.3f}]{gate}")

    for layer, cls in (("effusion", 2), ("ligament", 3)):
        rows = [r for r in records if r["thr"] == PRIMARY and r["layer"] == layer]
        det = [r for r in rows if r["pred_cls"] != ""]
        right = [r for r in det if r["pred_cls"] == cls]
        summary[layer] = {"n": len(rows), "detected": len(det), "right_class": len(right)}
        print(f"\n{layer}: {len(rows)} (scan, view) rows, detected {len(det)}, right class {len(right)}")

    if not a.skip_gap:
        summary["geometry_gap"] = geometry_gap(a, folds)
        print("\ngeometry gap, our masks, mean Dice over present structures:")
        for v in VIEWS:
            g = summary["geometry_gap"]
            print(f"  {v:9s} training {g[f'{v}|training']:.3f}   held-out {g[f'{v}|held_out']:.3f}")

    fig_buckets(tables, a.figs / "buckets_by_view.png")
    fig_paired(summary["paired"], a.figs / "paired_wrong_host.png")
    (a.out / "summary.json").write_text(json.dumps(summary, indent=1, default=float))
    verdict = "PASS" if summary["g2"]["nnunet"]["pass"] else "FAIL"
    print(f"\nG2 VERDICT (gate path: annotated box and class on nnU-Net masks; +5 points at noise_q3 or us16, "
          f"scan bootstrap CI excluding 0): {verdict}")
    aux = summary["g2_predicted"]["nnunet"]
    print(f"auxiliary (our detections, IoU 0.1): noise_q3 delta {aux['noise_q3']['delta']:+.3f} n={aux['noise_q3']['n_lesions']}, "
          f"us16 delta {aux['us16']['delta']:+.3f} n={aux['us16']['n_lesions']}")


if __name__ == "__main__":
    main()
