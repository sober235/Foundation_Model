#!/usr/bin/env python
"""Build the uncompressed training cache for m1r (one folder per scan, 4 workers, nice 19).

  PYTHONNOUSERSITE=1 PYTHONPATH=. ~/anaconda3/envs/nvgen/bin/python scripts/build_m1r_cache.py --workers 4
"""
import argparse
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from anatobind.train.cache import cache_scan  # noqa: E402
from anatobind.train.dataset import list_ready_scans  # noqa: E402

FM = Path("/data2/congcong/data/FM_data/derived/skmtea")
M1R, CACHE = FM / "m1r", FM / "m1r_cache"


def job(scan):
    os.nice(19)
    t0 = time.time()
    done = cache_scan(M1R / scan, CACHE / scan)
    return scan, done, round(time.time() - t0, 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=4)
    a = ap.parse_args()
    scans = list_ready_scans(M1R)
    with ProcessPoolExecutor(max_workers=a.workers) as ex:
        for scan, done, sec in ex.map(job, scans):
            print(f"{scan}: {'cached' if done else 'already cached'} in {sec} s", flush=True)
    print(f"{len(scans)} scans in {CACHE}")


if __name__ == "__main__":
    main()
