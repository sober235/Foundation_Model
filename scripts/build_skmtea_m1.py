#!/usr/bin/env python
"""Export the M1 SKM-TEA dataset: one folder per scan, resumable, CPU process pool (nice 19).

  PYTHONNOUSERSITE=1 PYTHONPATH=. ~/anaconda3/envs/nvgen/bin/python scripts/build_skmtea_m1.py --workers 4
Outputs: /data2/congcong/data/FM_data/derived/skmtea/m1/<scan>/..., annotations_screened.csv, splits.json, manifest.csv
"""
import argparse
import csv
import json
import os
import sys
import time
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from anatobind.data_engine.skmtea import export_scan, load_orientations, screen_split  # noqa: E402
from anatobind.data_engine.splits import five_fold_by_scan, stratum_of  # noqa: E402

FM = Path("/data2/congcong/data/FM_data")
RAW, SEG, ANN = FM / "SKM-TEA/files_recon_calib-24", FM / "SKM-TEA_ltr/segmentation_masks/dicom-track", FM / "SKM-TEA_ltr/annotations/v1.0.0"
CONDITIONS = {"noise": [0.25, 0.5, 1.0], "us": [4, 8, 16]}
MANIFEST_FIELDS = ["scan_id", "out_dir", "n_boxes_kept", "n_ambiguous", "n_no_overlap", "n_unresolved",
                   "spacing", "nrmse", "files", "status", "seconds"]


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out-root", type=Path, default=FM / "derived/skmtea/m1")
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--dry-run", action="store_true")
    return p.parse_args()


def job(args):
    scan, rows, out_dir, seed, orientation = args
    os.nice(19)
    t0 = time.time()
    try:
        row = export_scan(RAW / f"{scan}.h5", SEG / f"{scan}.nii.gz", rows, out_dir, CONDITIONS, rng_seed=seed,
                          orientation=orientation)
        row["status"] = "ok"
    except Exception as exc:
        row = {"scan_id": scan, "out_dir": str(out_dir), "n_boxes_kept": "", "n_ambiguous": "", "files": [],
               "status": f"error: {type(exc).__name__}: {exc}"[:200]}
    row["seconds"] = round(time.time() - t0, 1)
    return row


def main():
    a = parse_args()
    a.out_root.mkdir(parents=True, exist_ok=True)
    rows = [r for s in ("train", "val", "test") for r in screen_split(ANN / f"{s}.json")]
    with open(a.out_root / "annotations_screened.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    orient = load_orientations(ANN)
    scans = sorted(p.name[:-3] for p in RAW.glob("MTR_*.h5"))
    n_in_seg = Counter(r["scan_id"] for r in rows if r["keep"] and r["layer"] == "in_seg")
    folds = five_fold_by_scan(scans, {s: stratum_of(n_in_seg[s]) for s in scans}, seed=a.seed)
    (a.out_root / "splits.json").write_text(json.dumps({"seed": a.seed, "folds": folds}, indent=1))
    by_scan = {s: [r for r in rows if r["scan_id"] == s] for s in scans}
    pending = [s for s in scans if not (a.out_root / s / "boxes.csv").exists()]
    if a.limit:
        pending = pending[: a.limit]
    print(f"scans {len(scans)} | annotations {len(rows)} kept {sum(r['keep'] for r in rows)} | pending {len(pending)}", flush=True)
    if a.dry_run or not pending:
        return
    jobs = [(s, by_scan[s], a.out_root / s, a.seed + i, orient[s]) for i, s in enumerate(scans) if s in pending]
    manifest = a.out_root / "manifest.csv"
    existing = {}
    if manifest.exists():
        with open(manifest, newline="") as f:
            existing = {r["scan_id"]: r for r in csv.DictReader(f)}
    with ProcessPoolExecutor(max_workers=a.workers) as ex:
        for row in ex.map(job, jobs):
            row["files"] = ";".join(row["files"])
            existing[row["scan_id"]] = row
            with open(manifest, "w", newline="") as f:
                w = csv.DictWriter(f, fieldnames=MANIFEST_FIELDS)
                w.writeheader()
                w.writerows(existing[k] for k in sorted(existing))
            print(f"{row['scan_id']}: {row['status']} in {row['seconds']} s", flush=True)


if __name__ == "__main__":
    main()
