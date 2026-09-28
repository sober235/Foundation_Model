#!/usr/bin/env python
# scripts/nndet_prepare.py
"""Build nnDetection's Task903 from Dataset903 (spec 2026-09-28 brain-nndet §4), or write its splits.

  PYTHONNOUSERSITE=1 PYTHONPATH=. nice -n 19 ~/anaconda3/envs/nvgen/bin/python scripts/nndet_prepare.py --stage task
  bash -c 'source scripts/nndet_env.sh && nice -n 19 nndet_prep 903 -np 4 -npp 4 --full_check'
  PYTHONNOUSERSITE=1 PYTHONPATH=. ~/anaconda3/envs/nvgen/bin/python scripts/nndet_prepare.py --stage splits
"""
import argparse
import hashlib
import json
import shutil
import sys
from pathlib import Path

import nibabel as nib
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from anatobind.data_engine.fastmri import read_fastmri_plus_rows  # noqa: E402
from anatobind.level_r.export import match_registry, merged_lesions, small_lesion_rows  # noqa: E402
from anatobind.level_r.registry import load_registry  # noqa: E402
from anatobind.nndet.brain_task import (  # noqa: E402
    TASK_NAME, build_task, changed_boxes, check_same_support, instances_json, paint_instances, write_instance_label,
    write_splits_pkl,
)
from anatobind.nnunet.brain_lesion import DATASET_NAME  # noqa: E402

FM = Path("/data2/congcong/data/FM_data")
CSV = FM / "fastMRI_lh_brain_knee/Annotations/brain.csv"
NNUNET_RAW = FM / "derived/nnunet/raw" / DATASET_NAME
NNUNET_SPLITS = FM / "derived/nnunet/preprocessed" / DATASET_NAME / "splits_final.json"
DET_DATA = FM / "derived/nndet"
N_LESIONS = 1297


def sha256(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def stage_task():
    info = json.loads((NNUNET_RAW / "cases.json").read_text())
    reg_by_file, csv_by_file = {}, {}
    for r in load_registry():
        reg_by_file.setdefault(r["file"], []).append(r)
    for r in small_lesion_rows(read_fastmri_plus_rows(CSV)):
        csv_by_file.setdefault(r["file"], []).append(r)
    instance_map, changed = {}, {}

    def write_case(case, img_out, labels_dir):
        src = NNUNET_RAW / "imagesTr" / f"{case}_0000.nii.gz"
        shutil.copyfile(src, img_out)
        if sha256(src) != sha256(img_out):
            raise ValueError(f"{case}: the copied image differs from {src}")
        binary = np.asarray(nib.load(str(NNUNET_RAW / "labelsTr" / f"{case}.nii.gz")).dataobj)
        shape = binary.shape                                   # (col, row, slice); shape[1] = RSS rows
        if info[case]["kind"] == "lesion":
            matched = match_registry(reg_by_file[case], merged_lesions(csv_by_file.get(case, []), shape[1]))
            members_of = {lid: L["members"] for lid, L in matched.items()}
            inst, instance_of = paint_instances(members_of, shape)
            ch = changed_boxes(members_of, inst, instance_of)
            if ch:
                changed[case] = ch
        else:
            inst, instance_of = np.zeros(shape, np.uint16), {}
        check_same_support(inst, binary, case)
        write_instance_label(inst, img_out, labels_dir / f"{case}.nii.gz")
        (labels_dir / f"{case}.json").write_text(json.dumps(instances_json(instance_of)))
        instance_map[case] = {str(k): int(lid) for lid, k in instance_of.items()}
        print(f"{len(instance_map)}/{len(info)} {case} {info[case]['kind']} {len(instance_of)} instances", flush=True)

    base = build_task(DET_DATA, sorted(info), write_case)
    n = sum(len(v) for v in instance_map.values())
    (base / "instances.json").write_text(json.dumps(instance_map, indent=1))
    (base / "build_report.json").write_text(json.dumps({"n_cases": len(info), "n_instances": n,
                                                        "changed_boxes": changed}, indent=1))
    if n != N_LESIONS:
        raise AssertionError(f"expected {N_LESIONS} instances, got {n}")
    print(f"wrote {base}: {len(info)} cases, {n} instances; lesions whose box changed by overwrite: {changed}")


def stage_splits():
    prep = DET_DATA / TASK_NAME / "preprocessed"
    if not prep.is_dir():
        raise FileNotFoundError(f"{prep} missing: run nndet_prep 903 first")
    splits = json.loads(NNUNET_SPLITS.read_text())
    info = json.loads((NNUNET_RAW / "cases.json").read_text())
    val = sorted(c for s in splits for c in s["val"])
    if len(splits) != 5 or val != sorted(info) or len(val) != 253:
        raise AssertionError("Dataset903 splits do not cover the 253 cases exactly once")
    p = write_splits_pkl(prep, splits)
    print(f"wrote {p}: validation cases per fold {[len(s['val']) for s in splits]}")


def main():
    ap = argparse.ArgumentParser(description="Build nnDetection Task903 or write its splits")
    ap.add_argument("--stage", choices=("task", "splits"), required=True)
    a = ap.parse_args()
    stage_task() if a.stage == "task" else stage_splits()


if __name__ == "__main__":
    main()
