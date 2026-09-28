#!/usr/bin/env python
# scripts/brain_detector_prepare.py
"""Build Dataset903 (spec 2026-09-28 §2) or write its splits.

  source scripts/nnunet_env.sh
  PYTHONPATH=. nice -n 19 python scripts/brain_detector_prepare.py --stage raw
  nice -n 19 nnUNetv2_plan_and_preprocess -d 903 -c 2d 3d_fullres --verify_dataset_integrity -np 4
  PYTHONPATH=. python scripts/brain_detector_prepare.py --stage splits
"""
import argparse
import csv
import json
import os
import sys
from pathlib import Path

import nibabel as nib
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from anatobind.data_engine.fastmri import read_fastmri_plus_rows, rss_h5_to_nifti  # noqa: E402
from anatobind.data_engine.fastmri_knee import volume_geometry  # noqa: E402
from anatobind.level_r.export import match_registry, merged_lesions, small_lesion_rows  # noqa: E402
from anatobind.level_r.registry import load_registry  # noqa: E402
from anatobind.nnunet.brain_lesion import (  # noqa: E402
    DATASET_NAME, assign_normal_folds, build_raw, case_folds, check_members_inside, normal_files, paint_members,
    write_label, write_splits,
)

FM = Path("/data2/congcong/data/FM_data")
CSV = FM / "fastMRI_lh_brain_knee/Annotations/brain.csv"
KROOT = FM / "fastMRI_lh_brain_knee/kspace/brain"
FOLDS = Path("data/level_r/folds.json")


def h5_of(stem):
    for split in ("multicoil_train", "multicoil_val"):
        p = KROOT / split / f"{stem}.h5"
        if p.exists():
            return p
    raise FileNotFoundError(stem)


def stage_raw(raw_root):
    registry = load_registry()
    reg_by_file, csv_by_file = {}, {}
    for r in registry:
        reg_by_file.setdefault(r["file"], []).append(r)
    for r in small_lesion_rows(read_fastmri_plus_rows(CSV)):
        csv_by_file.setdefault(r["file"], []).append(r)
    with open(CSV, newline="") as fh:
        normals = normal_files(list(csv.DictReader(fh)))
    lesion_files = sorted(reg_by_file)
    if set(normals) & set(lesion_files):
        raise ValueError("a normal volume is also a registry volume")
    info = {}

    def write_case(case, img_path, lab_path):
        path = h5_of(case)
        g = volume_geometry(path)
        rss_h5_to_nifti(path, img_path, pad_to_slices=0)
        shape = nib.load(str(img_path)).shape
        if shape != (g["n_cols"], g["n_rows"], g["slices"]):
            raise ValueError(f"{case}: image {shape} != RSS {(g['n_cols'], g['n_rows'], g['slices'])}")
        if case in reg_by_file:
            matched = match_registry(reg_by_file[case], merged_lesions(csv_by_file.get(case, []), g["n_rows"]))
            members_of = {lid: L["members"] for lid, L in matched.items()}
            check_members_inside(members_of, shape)
            lab = paint_members(members_of, shape)
            kind = "lesion"
        else:
            lab = np.zeros(shape, np.uint8)
            kind = "normal"
        write_label(lab, img_path, lab_path)
        info[case] = {"patient_id": g["patient_id"], "kind": kind, "n_label_voxels": int(lab.sum())}
        print(f"{len(info)}/{len(lesion_files) + len(normals)} {case} {kind} {int(lab.sum())} label voxels", flush=True)

    base = build_raw(raw_root, lesion_files + normals, write_case)
    (base / "cases.json").write_text(json.dumps(info, indent=1))
    n_lesion = sum(1 for v in info.values() if v["kind"] == "lesion")
    print(f"wrote {base}: {len(info)} cases ({n_lesion} lesion, {len(info) - n_lesion} normal), "
          f"{sum(1 for v in info.values() if v['kind'] == 'normal' and v['n_label_voxels'])} normal cases with label voxels")


def stage_splits(raw_root, preprocessed_root):
    info = json.loads((Path(raw_root) / DATASET_NAME / "cases.json").read_text())
    lesion = {c: v["patient_id"] for c, v in info.items() if v["kind"] == "lesion"}
    normal = {c: v["patient_id"] for c, v in info.items() if v["kind"] == "normal"}
    fold_of = case_folds(lesion, normal, json.loads(FOLDS.read_text())["patient_fold"], assign_normal_folds(normal.values()))
    p = write_splits(preprocessed_root, fold_of)
    per = {f: sum(1 for v in fold_of.values() if v == f) for f in range(5)}
    print(f"wrote {p}: cases per fold {per}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=("raw", "splits"), required=True)
    a = ap.parse_args()
    if a.stage == "raw":
        stage_raw(os.environ["nnUNet_raw"])
    else:
        stage_splits(os.environ["nnUNet_raw"], os.environ["nnUNet_preprocessed"])


if __name__ == "__main__":
    main()
