#!/usr/bin/env python
"""Where do the misses come from? Join records.csv (one row per annotated box) with the export boxes.csv
to get each box's size, then report localisation hits by family x size tercile, the score of hits versus
the operating threshold, and per-scan hit counts.

  PYTHONNOUSERSITE=1 PYTHONPATH=. ~/anaconda3/envs/nvgen/bin/python docs/verification/2026-09-23/knee_eval/miss_analysis.py <eval_dir> [view]
"""
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

EXPORT = Path("/data2/congcong/data/FM_data/derived/skmtea/m1r")
VOX_ML = 0.625 * 0.625 * 0.8 / 1000.0


def main(eval_dir, view="clean"):
    eval_dir = Path(eval_dir)
    gate = json.loads((eval_dir / "gate.json").read_text())
    thr = gate["thr"] if gate["thr"] is not None else 1.0
    rows = [r for r in csv.DictReader(open(eval_dir / "records.csv", newline="")) if r["view"] == view]
    size = {}
    for scan in {r["scan"] for r in rows}:
        for b in csv.DictReader(open(EXPORT / scan / "boxes.csv", newline="")):
            v = (int(b["x1"]) - int(b["x0"])) * (int(b["y1"]) - int(b["y0"])) * (int(b["z1"]) - int(b["z0"]))
            size[(scan, int(b["ann_id"]))] = v * VOX_ML
    for r in rows:
        r["ml"] = size[(r["scan"], int(r["ann_id"]))]
    print(f"{view}: {len(rows)} annotated boxes in {len({r['scan'] for r in rows})} scans, operating threshold {thr}")
    print("\nlocalisation hits (IoU >= 0.1 at the operating point) by family x size tercile [mL range]:")
    for fam in sorted({r["gt_family"] for r in rows}):
        fr = sorted((r for r in rows if r["gt_family"] == fam), key=lambda r: r["ml"])
        n = len(fr)
        parts = [fr[: n // 3], fr[n // 3: 2 * n // 3], fr[2 * n // 3:]]
        cells = []
        for p in parts:
            if not p:
                continue
            hits = sum(r["matched"] == "1" for r in p)
            cells.append(f"{hits}/{len(p)} [{p[0]['ml']:.1f}-{p[-1]['ml']:.1f}]")
        total = sum(r["matched"] == "1" for r in fr)
        print(f"  {fam:17s} {total}/{n} ({total / n:.2f})  terciles: " + " | ".join(cells))
    print("\nhits: predicted family = truth?  ", end="")
    hits = [r for r in rows if r["matched"] == "1"]
    print(f"{sum(r['det_family'] == r['gt_family'] for r in hits)}/{len(hits)}")
    scores = np.array([float(r["det_score"]) for r in hits]) if hits else np.array([])
    if len(scores):
        print(f"hit scores: min {scores.min():.2f}, median {np.median(scores):.2f}, max {scores.max():.2f}")
    per_scan = defaultdict(lambda: [0, 0])
    for r in rows:
        per_scan[r["scan"]][1] += 1
        per_scan[r["scan"]][0] += r["matched"] == "1"
    zero = sum(1 for h, n in per_scan.values() if h == 0)
    print(f"scans with no hit at all: {zero}/{len(per_scan)}")
    print("\nbinding on hits (in-seg only): " + ", ".join(
        f"{k}: {v}" for k, v in sorted(
            defaultdict(int, {b: sum(1 for r in hits if r['bucket'] == b) for b in {r['bucket'] for r in hits} - {''}}).items())))


if __name__ == "__main__":
    main(*sys.argv[1:])
