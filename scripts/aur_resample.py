#!/usr/bin/env python
"""Resample the AUR sample table to 1 mm isotropic RAS grids (anatobind.aur.resample; decision Q4 (b), 2026-10-09).

    PYTHONNOUSERSITE=1 PYTHONPATH=. nice -n 19 python scripts/aur_resample.py \
        --samples /data2/congcong/data/FM_data/derived/aur/samples.json \
        --out-root /data2/congcong/data/FM_data/derived/aur/resampled_1mm_v1 \
        --out-samples /data2/congcong/data/FM_data/derived/aur/samples_1mm_v1.json --workers 4

Cases already at 1 mm keep their files. Every other case is written once under out-root/<source>/<case>/; a second run
refuses unless --resume (files that exist are kept, the rest are made). The table written at the end carries the new
paths, the native spacing, the resampled flag and the thick-slice rule (native slice > 3 mm: A and R supervision off).
--dry-run only prints the plan. Workers stay at 4 at most (the host's memory watchdog)."""
import argparse
import sys
import time
from multiprocessing import Pool
from pathlib import Path

from anatobind.aur.resample import apply_thickness_rule, counts, plan, resample_case, write_table
from anatobind.aur.samples import read_samples


def _work(args):
    case, rows, out_root, resume = args
    t = time.time()
    try:
        return case, resample_case(rows, out_root, resume=resume), None, time.time() - t
    except Exception as e:                      # one failed case must not stop the others; it is listed at the end
        return case, None, f"{type(e).__name__}: {e}", time.time() - t


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--samples", type=Path, required=True)
    ap.add_argument("--out-root", type=Path, required=True)
    ap.add_argument("--out-samples", type=Path, required=True)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--limit", type=int, default=None, help="only the first N cases to resample (smoke runs)")
    a = ap.parse_args(argv)
    if a.workers > 4:
        ap.error("--workers above 4 is not allowed on this host")
    if a.out_samples.exists():
        ap.error(f"{a.out_samples} exists; use a new name")
    if a.out_root.exists() and not a.resume:
        ap.error(f"{a.out_root} exists; use --resume to keep its files or a new name")
    rows = read_samples(a.samples)
    todo, kept = plan(rows)
    if a.limit is not None:
        todo = todo[:a.limit]
    by_source = {}
    for case, group in todo:
        by_source[group[0]["source"]] = by_source.get(group[0]["source"], 0) + 1
    print(f"{len(rows)} rows; {len(kept)} rows kept at 1 mm; {len(todo)} cases to resample: {by_source}", flush=True)
    if a.dry_run:
        return 0
    a.out_root.mkdir(parents=True, exist_ok=True)
    new_rows, failed = [], []
    t0 = time.time()
    with Pool(a.workers) as pool:
        for i, (case, out, err, dt) in enumerate(pool.imap_unordered(_work, [(c, g, a.out_root, a.resume) for c, g in todo]), 1):
            if err:
                failed.append((case, err))
                print(f"[{i}/{len(todo)}] {case}: FAILED {err}", flush=True)
            else:
                new_rows += out
                print(f"[{i}/{len(todo)}] {case}: {len(out)} rows, {dt:.1f} s, elapsed {time.time() - t0:.0f} s", flush=True)
    table = apply_thickness_rule(kept + new_rows)
    table.sort(key=lambda r: (r["source"], r["case"], r["source_sequence"]))
    c = write_table(a.out_samples, table)
    for source, d in c.items():
        print(f"{source}: {d['cases']} cases / {d['rows']} rows / {d['resampled']} resampled rows / {d['thick']} thick-slice rows")
    print(f"wrote {a.out_samples}: {len(table)} rows; {len(failed)} cases failed")
    for case, err in failed:
        print(f"  FAILED {case}: {err}")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
