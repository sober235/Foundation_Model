#!/usr/bin/env python
# scripts/build_relation_table.py
"""Build the relation feature table (spec 2026-09-27 §4) from the Gate 0.5 registry, the Level R fold table, the
SynthSeg parcellations and the h5 RSS volumes. Refuses an existing output directory.

  PYTHONNOUSERSITE=1 PYTHONPATH=. nice -n 19 ~/anaconda3/envs/nvgen/bin/python scripts/build_relation_table.py \
      --out /data2/congcong/data/FM_data/derived/relation/v1 | tee docs/verification/<date>/relation_table_output.txt
"""
import argparse
import json
import subprocess
import sys
from pathlib import Path

import h5py
import nibabel as nib
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from anatobind.data_engine.fastmri import read_fastmri_plus_rows  # noqa: E402
from anatobind.data_engine.fastmri_knee import volume_geometry  # noqa: E402
from anatobind.level_r.admin import sha256_file  # noqa: E402
from anatobind.level_r.export import match_registry, merged_lesions, small_lesion_rows  # noqa: E402
from anatobind.level_r.registry import REGISTRY, load_registry  # noqa: E402
from anatobind.relation.build import build_table  # noqa: E402

FM = Path("/data2/congcong/data/FM_data")
CSV = FM / "fastMRI_lh_brain_knee/Annotations/brain.csv"
KROOT = FM / "fastMRI_lh_brain_knee/kspace/brain"
SEG_ROOT = FM / "derived/synthseg/fastmri_brain/seg_native"
FOLDS = Path("data/level_r/folds.json")


def h5_of(stem):
    for split in ("multicoil_train", "multicoil_val"):
        p = KROOT / split / f"{stem}.h5"
        if p.exists():
            return p
    raise FileNotFoundError(stem)


def make_loader(registry_by_file, csv_by_file):
    def load_volume(f):
        path = h5_of(f)
        g = volume_geometry(path)
        with h5py.File(path) as h:
            rss = h["reconstruction_rss"][()]
        rss = np.ascontiguousarray(rss.transpose(2, 1, 0)).astype(np.float32)          # (slice, row, col) -> (col, row, slice)
        img = nib.load(str(SEG_ROOT / f"{f}_seg.nii.gz"))
        seg = np.asarray(img.dataobj).astype(np.int16)
        zooms = tuple(float(z) for z in img.header.get_zooms()[:3])
        if seg.shape != rss.shape or seg.shape != (g["n_cols"], g["n_rows"], g["slices"]):
            raise ValueError(f"{f}: seg {seg.shape}, rss {rss.shape}, geometry {(g['n_cols'], g['n_rows'], g['slices'])}")
        matched = match_registry(registry_by_file[f], merged_lesions(csv_by_file.get(f, []), g["n_rows"]))
        return seg, zooms, rss, {lid: L["members"] for lid, L in matched.items()}
    return load_volume


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--registry", type=Path, default=REGISTRY)
    ap.add_argument("--folds", type=Path, default=FOLDS)
    ap.add_argument("--limit", type=int, default=0, help="first N files only (smoke runs)")
    ap.add_argument("--min-c1-agreement", type=float, default=0.99,
                    help="spec §4.7 threshold; the real build keeps 0.99, a smoke run on a few files may pass 0")
    a = ap.parse_args()
    registry = load_registry(a.registry)
    folds = json.loads(a.folds.read_text())["patient_fold"]
    registry_by_file, csv_by_file = {}, {}
    for r in registry:
        registry_by_file.setdefault(r["file"], []).append(r)
    for r in small_lesion_rows(read_fastmri_plus_rows(CSV)):
        csv_by_file.setdefault(r["file"], []).append(r)
    if a.limit:
        keep = sorted(registry_by_file)[:a.limit]
        registry = [r for r in registry if r["file"] in keep]
    commit = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
    sources = {"git_commit": commit, "registry": str(a.registry), "registry_sha256": sha256_file(a.registry),
               "folds": str(a.folds), "folds_sha256": sha256_file(a.folds), "seg_root": str(SEG_ROOT), "kspace_root": str(KROOT)}
    rows, patches, manifest = build_table(registry, folds, make_loader(registry_by_file, csv_by_file), a.out, sources,
                                          log=lambda m: print(m, flush=True), min_agreement=a.min_c1_agreement)
    print(json.dumps({k: v for k, v in manifest.items() if k != "c1_agreement"}, indent=1))
    print("c1_agreement", manifest["c1_agreement"]["fraction"], "mismatches", manifest["c1_agreement"]["mismatches"])
    print(f"wrote {a.out}: {len(rows)} lesions, patches {patches['image'].shape}")


if __name__ == "__main__":
    main()
