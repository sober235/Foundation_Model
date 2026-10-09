#!/usr/bin/env python
"""Build the Stage I manifests (anatobind.aur.ssl.samples; SSL-first plan T01, decision Q6 of 2026-10-09).

    PYTHONNOUSERSITE=1 PYTHONPATH=. nice -n 19 python scripts/aur_ssl_prepare.py \
        --aur-samples /data2/congcong/data/FM_data/derived/aur/samples_1mm_v1.json \
        --resample-root /data2/congcong/data/FM_data/derived/aur/resampled_1mm_v1 \
        --out /data2/congcong/data/FM_data/derived/aur/ssl_manifest_v1 --workers 4

Steps: list HCP (T1w + T2w + SynthSeg map per subject), resample the HCP volumes to 1 mm under
<resample-root>/hcp/<case>/ (resume: files that exist are kept), draw the validation patients (a tenth of the train
patients per source, seed 0), build the Stage I rows, write the manifest directory (must not exist), print the leakage
report and the per-source counts. --no-hcp leaves HCP out; --limit-hcp N takes the first N subjects (smoke runs)."""
import argparse
import json
import sys
import time
from multiprocessing import Pool
from pathlib import Path

from anatobind.anatomy.sources import FM
from anatobind.aur.resample import resample_case
from anatobind.aur.samples import read_samples
from anatobind.aur.ssl.samples import VAL_SHARE, hcp_samples, inventory, leakage_report, ssl_rows, validation_patients, write_manifest


def _work(args):
    case, rows, out_root = args
    t = time.time()
    try:
        return case, resample_case(rows, out_root, resume=True), None, time.time() - t
    except Exception as e:
        return case, None, f"{type(e).__name__}: {e}", time.time() - t


def resample_hcp(rows, out_root, workers):
    """HCP rows -> the same rows on the 1 mm files (resampled once; existing files kept). Failed cases are left out."""
    by_case = {}
    for r in rows:
        by_case.setdefault(r["case"], []).append(r)
    out, failed, t0 = [], [], time.time()
    with Pool(workers) as pool:
        for i, (case, new, err, dt) in enumerate(pool.imap_unordered(_work, [(c, g, out_root) for c, g in by_case.items()]), 1):
            if err:
                failed.append((case, err))
                print(f"[{i}/{len(by_case)}] {case}: FAILED {err}", flush=True)
            else:
                out += new
                if i % 25 == 0 or i == len(by_case):
                    print(f"[{i}/{len(by_case)}] {case}: {dt:.1f} s, elapsed {time.time() - t0:.0f} s", flush=True)
    return out, failed


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--aur-samples", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--resample-root", type=Path, default=None, help="where the 1 mm HCP files go (<root>/hcp/<case>/)")
    ap.add_argument("--fm-root", type=Path, default=FM)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--val-share", type=float, default=VAL_SHARE)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--no-hcp", action="store_true")
    ap.add_argument("--limit-hcp", type=int, default=None)
    a = ap.parse_args(argv)
    if a.workers > 4:
        ap.error("--workers above 4 is not allowed on this host")
    if a.out.exists():
        ap.error(f"{a.out} exists; use a new name")
    aur = read_samples(a.aur_samples)
    hcp, skipped, failed = [], [], []
    if not a.no_hcp:
        if a.resample_root is None:
            ap.error("--resample-root is needed for HCP (0.7 mm volumes are resampled to 1 mm)")
        hcp, skipped = hcp_samples(a.fm_root)
        if a.limit_hcp is not None:
            keep = sorted({r["patient"] for r in hcp})[:a.limit_hcp]
            hcp = [r for r in hcp if r["patient"] in keep]
        print(f"HCP: {len(hcp)} rows / {len({r['patient'] for r in hcp})} subjects; skipped {len(skipped)}: {skipped[:10]}", flush=True)
        hcp, failed = resample_hcp(hcp, a.resample_root, a.workers)
    val = validation_patients(aur + hcp, share=a.val_share, seed=a.seed)
    rows = ssl_rows(aur, hcp, val)
    report = leakage_report(rows, aur)
    report["hcp_skipped"] = skipped
    report["hcp_failed"] = failed
    inv = inventory(rows)
    paths = write_manifest(a.out, rows, val, report, inv)
    print(json.dumps({k: v for k, v in report.items() if k != "per_source"}, indent=1))
    for source, d in report["per_source"].items():
        print(f"{source}: train {d['train']['rows']} rows / {d['train']['patients']} patients; val {d['val']['rows']} rows / {d['val']['patients']} patients")
    print("wrote", paths)
    return 0 if report["ok"] and not failed else 1


if __name__ == "__main__":
    sys.exit(main())
