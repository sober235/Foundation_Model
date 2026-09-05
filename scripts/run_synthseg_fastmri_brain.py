#!/usr/bin/env python
"""Brain pseudo-labels for fastMRI volumes: SynthSeg-robust 2.0 on a CPU process pool.

Each worker converts a chunk of h5 files to NIfTI, runs SynthSeg once over that
folder (nice 19, CPU only) and resamples the 1 mm labels back onto the native
grid. Outputs land under --work-root:
  chunks/NNNN/{stage,seg_1mm,volumes.csv,manifest.csv,synthseg.log}
  seg_native/<stem>_seg.nii.gz           (shared, used to resume)
  manifest.csv                           (consolidated, one row per stem)

Example (annotated fastMRI+ brain volumes only):
  PYTHONNOUSERSITE=1 PYTHONPATH=. ~/anaconda3/envs/nvgen/bin/python \
      scripts/run_synthseg_fastmri_brain.py --annotated-only --workers 4 --threads 12
"""
import argparse
import csv
import os
import subprocess
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from anatobind.data_engine.synthseg_pipeline import (  # noqa: E402
    MANIFEST_FIELDS,
    annotated_files,
    chunked,
    pending_stems,
    resolve_h5,
    run_batch,
    write_manifest,
)

FM = Path("/data2/congcong/data/FM_data")


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--annotations", type=Path, default=FM / "fastMRI_lh_brain_knee/Annotations/brain.csv")
    p.add_argument("--kspace-root", type=Path, default=FM / "fastMRI_lh_brain_knee/kspace/brain")
    p.add_argument("--work-root", type=Path, default=FM / "derived/synthseg/fastmri_brain")
    p.add_argument("--synthseg-home", type=Path, default=Path.home() / "src/SynthSeg")
    p.add_argument("--python", type=Path, default=Path.home() / "anaconda3/envs/synthseg/bin/python")
    p.add_argument("--annotated-only", action="store_true", help="only stems present in the fastMRI+ csv")
    p.add_argument("--limit", type=int, default=None, help="process at most N pending volumes")
    p.add_argument("--chunk-size", type=int, default=25)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--threads", type=int, default=12, help="TensorFlow threads per worker")
    p.add_argument("--dry-run", action="store_true")
    return p.parse_args()


def all_stems(kspace_root):
    return sorted(p.name[:-3] for split in ("multicoil_train", "multicoil_val") for p in (kspace_root / split).glob("*.h5"))


def make_runner(log_path):
    def runner(argv):
        env = {**os.environ, "PYTHONNOUSERSITE": "1", "CUDA_VISIBLE_DEVICES": ""}
        with open(log_path, "a") as log:
            subprocess.run(["nice", "-n", "19"] + argv, check=True, stdout=log, stderr=subprocess.STDOUT, env=env)
    return runner


def process_chunk(job):
    i, h5s, work_root, synthseg_home, python, threads, native_dir = job
    chunk_dir = Path(work_root) / "chunks" / f"{i:04d}"
    chunk_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    try:
        rows = run_batch(h5s, chunk_dir, synthseg_home, python, threads,
                         runner=make_runner(chunk_dir / "synthseg.log"), native_dir=native_dir)
    except Exception as exc:  # keep the pool alive; record the failure per stem
        rows = [{"stem": Path(h).name[:-3], "h5": str(h), "nii": "", "seg_1mm": "", "seg_native": "",
                 "status": f"error: {type(exc).__name__}: {exc}"[:200]} for h in h5s]
    return i, rows, time.time() - t0


def merge_manifest(work_root, new_rows):
    path = Path(work_root) / "manifest.csv"
    rows = {}
    if path.exists():
        with open(path, newline="") as f:
            rows.update({r["stem"]: r for r in csv.DictReader(f)})
    rows.update({r["stem"]: r for r in new_rows})
    write_manifest([rows[k] for k in sorted(rows)], path)
    return path


def main():
    a = parse_args()
    native_dir = a.work_root / "seg_native"
    stems = annotated_files(a.annotations) if a.annotated_only else all_stems(a.kspace_root)
    todo = pending_stems(stems, native_dir)
    if a.limit:
        todo = todo[: a.limit]
    h5s, missing = [], []
    for s in todo:
        try:
            h5s.append(resolve_h5(s, a.kspace_root))
        except FileNotFoundError:
            missing.append(s)
    print(f"stems total {len(stems)} | pending {len(todo)} | h5 found {len(h5s)} | missing h5 {len(missing)}", flush=True)
    if missing:
        print("missing:", ", ".join(missing[:10]), "..." if len(missing) > 10 else "", flush=True)
    chunks = chunked(h5s, a.chunk_size)
    start = int(time.time()) % 100000  # chunk ids unique across resumed runs
    jobs = [(start + i, c, a.work_root, a.synthseg_home, a.python, a.threads, native_dir) for i, c in enumerate(chunks)]
    print(f"{len(chunks)} chunks x {a.chunk_size} | {a.workers} workers x {a.threads} threads", flush=True)
    if a.dry_run or not jobs:
        return
    done_rows = []
    t0 = time.time()
    with ProcessPoolExecutor(max_workers=a.workers) as ex:
        for i, rows, dt in ex.map(process_chunk, jobs):
            done_rows.extend(rows)
            ok = sum(r["status"] == "ok" for r in rows)
            merge_manifest(a.work_root, rows)
            print(f"chunk {i:04d}: {ok}/{len(rows)} ok in {dt / 60:.1f} min | total {len(done_rows)}/{len(h5s)} "
                  f"| elapsed {(time.time() - t0) / 60:.1f} min", flush=True)
    n_ok = sum(r["status"] == "ok" for r in done_rows)
    print(f"finished: {n_ok}/{len(done_rows)} ok; manifest {a.work_root / 'manifest.csv'}", flush=True)


if __name__ == "__main__":
    main()
