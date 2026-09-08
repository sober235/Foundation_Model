#!/usr/bin/env python
"""Brain pseudo-labels with SynthSeg-robust 2.0 on a CPU process pool.

Inputs are either fastMRI brain h5 files (default; --annotated-only restricts to
fastMRI+ annotated stems) or NIfTI files selected with --glob. Each worker stages
a chunk, runs SynthSeg once over that folder (nice 19, CPU only) and resamples
the 1 mm labels back onto the native grid. Outputs land under --work-root:
  chunks/NNNNN/{stage,seg_1mm,volumes.csv,manifest.csv,synthseg.log}
  seg_native/<stem>_seg.nii.gz           (shared, used to resume)
  manifest.csv                           (consolidated, one row per stem)

Examples:
  PYTHONNOUSERSITE=1 PYTHONPATH=. ~/anaconda3/envs/nvgen/bin/python \
      scripts/run_synthseg_fastmri_brain.py --annotated-only --workers 4 --threads 12
  ... --glob '/data2/.../HCP_lh_T1T2/*/T1w/T1w_acpc_dc_restore_brain.nii' \
      --stem-prefix-parent 2 --work-root /data2/.../derived/synthseg/hcp
"""
import argparse
import csv
import glob
import os
import subprocess
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from anatobind.data_engine.synthseg_pipeline import (  # noqa: E402
    annotated_files,
    chunked,
    pending_stems,
    resolve_h5,
    run_batch,
    stem_of,
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
    p.add_argument("--stems-file", type=Path, default=None,
                   help="fastMRI h5 mode: restrict to the stems listed in this file (one per line, a 'file' header is ignored)")
    p.add_argument("--glob", nargs="+", default=None, help="NIfTI inputs (glob patterns) instead of fastMRI h5")
    p.add_argument("--stem-prefix-parent", type=int, default=0,
                   help="prefix the stem with N parent folder names (e.g. HCP: 100206_T1w_acpc_dc_restore_brain)")
    p.add_argument("--limit", type=int, default=None, help="process at most N pending volumes")
    p.add_argument("--chunk-size", type=int, default=25)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--threads", type=int, default=12, help="TensorFlow threads per worker")
    p.add_argument("--dry-run", action="store_true")
    return p.parse_args()


def all_stems(kspace_root):
    return sorted(p.name[:-3] for split in ("multicoil_train", "multicoil_val") for p in (kspace_root / split).glob("*.h5"))


def nifti_inputs(patterns, prefix_parents, alias_dir):
    """(stem, path) pairs for NIfTI inputs; aliases via symlink when parent folders are prefixed."""
    paths = sorted({Path(p) for pat in patterns for p in glob.glob(pat)})
    out = []
    for p in paths:
        stem = stem_of(p)
        if prefix_parents:
            parents = [q.name for q in list(p.parents)[:prefix_parents]][::-1]
            stem = "_".join(parents + [stem])
            alias_dir.mkdir(parents=True, exist_ok=True)
            alias = alias_dir / (stem + (".nii.gz" if p.name.endswith(".nii.gz") else ".nii"))
            if not alias.is_symlink():
                os.symlink(p.resolve(), alias)
            p = alias
        out.append((stem, p))
    return out


def make_runner(log_path):
    def runner(argv):
        env = {**os.environ, "PYTHONNOUSERSITE": "1", "CUDA_VISIBLE_DEVICES": ""}
        with open(log_path, "a") as log:
            subprocess.run(["nice", "-n", "19"] + argv, check=True, stdout=log, stderr=subprocess.STDOUT, env=env)
    return runner


def process_chunk(job):
    i, inputs, work_root, synthseg_home, python, threads, native_dir = job
    chunk_dir = Path(work_root) / "chunks" / f"{i:05d}"
    chunk_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    try:
        rows = run_batch(inputs, chunk_dir, synthseg_home, python, threads,
                         runner=make_runner(chunk_dir / "synthseg.log"), native_dir=native_dir)
    except Exception as exc:  # keep the pool alive; record the failure per input
        rows = [{"stem": stem_of(h), "h5": str(h), "nii": "", "seg_1mm": "", "seg_native": "",
                 "status": f"error: {type(exc).__name__}: {exc}"[:200]} for h in inputs]
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
    missing = []
    if a.glob:
        pairs = nifti_inputs(a.glob, a.stem_prefix_parent, a.work_root / "inputs")
        by_stem = dict(pairs)
        stems = sorted(by_stem)
        todo = pending_stems(stems, native_dir)
        if a.limit:
            todo = todo[: a.limit]
        inputs = [by_stem[s] for s in todo]
    else:
        if a.stems_file:
            stems = [s.strip() for s in a.stems_file.read_text().splitlines() if s.strip() and s.strip() != "file"]
        else:
            stems = annotated_files(a.annotations) if a.annotated_only else all_stems(a.kspace_root)
        todo = pending_stems(stems, native_dir)
        if a.limit:
            todo = todo[: a.limit]
        inputs = []
        for s in todo:
            try:
                inputs.append(resolve_h5(s, a.kspace_root))
            except FileNotFoundError:
                missing.append(s)
    print(f"stems total {len(stems)} | pending {len(todo)} | inputs {len(inputs)} | missing {len(missing)}", flush=True)
    if missing:
        print("missing:", ", ".join(missing[:10]), "..." if len(missing) > 10 else "", flush=True)
    chunks = chunked(inputs, a.chunk_size)
    start = int(time.time()) % 100000  # chunk ids unique across resumed runs
    jobs = [(start + i, c, a.work_root, a.synthseg_home, a.python, a.threads, native_dir) for i, c in enumerate(chunks)]
    print(f"{len(chunks)} chunks x {a.chunk_size} | {a.workers} workers x {a.threads} threads", flush=True)
    if inputs:
        print("first input:", inputs[0], "-> stem", stem_of(inputs[0]), flush=True)
    if a.dry_run or not jobs:
        return
    done_rows = []
    t0 = time.time()
    with ProcessPoolExecutor(max_workers=a.workers) as ex:
        for i, rows, dt in ex.map(process_chunk, jobs):
            done_rows.extend(rows)
            ok = sum(r["status"] == "ok" for r in rows)
            merge_manifest(a.work_root, rows)
            print(f"chunk {i:05d}: {ok}/{len(rows)} ok in {dt / 60:.1f} min | total {len(done_rows)}/{len(inputs)} "
                  f"| elapsed {(time.time() - t0) / 60:.1f} min", flush=True)
    n_ok = sum(r["status"] == "ok" for r in done_rows)
    print(f"finished: {n_ok}/{len(done_rows)} ok; manifest {a.work_root / 'manifest.csv'}", flush=True)


if __name__ == "__main__":
    main()
