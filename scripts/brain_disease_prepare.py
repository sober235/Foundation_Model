#!/usr/bin/env python
# scripts/brain_disease_prepare.py
"""Build one brain disease dataset for nnU-Net (spec 2026-09-29 §3), or write its folds.

  source scripts/nnunet_env.sh
  PYTHONPATH=. nice -n 19 python scripts/brain_disease_prepare.py --disease glioma --stage raw
  nice -n 19 nnUNetv2_plan_and_preprocess -d 904 -c 3d_fullres --verify_dataset_integrity -np 4 -npfp 4
  PYTHONPATH=. python scripts/brain_disease_prepare.py --disease glioma --stage splits
"""
import argparse
import collections
import csv
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from anatobind.nnunet.brain_disease import DISEASES, FM, build_raw, case_folds, sources, write_splits  # noqa: E402

BMSR_SUBJECTS = FM / "derived/brain_disease/bmsr_subjects.csv"
EXPECTED = {"glioma": (501, 495), "metastasis": (461, 314), "infarct": (250, 250)}     # (scans, patients)


def surgery_flags(path):
    """{case: {"prior_surgery": bool}} from the subject table exported to CSV (columns SubjectID and
    'Prior Craniotomy/Biopsy/Resection')."""
    with open(path, newline="") as fh:
        rows = list(csv.DictReader(fh))
    out = {}
    for r in rows:
        v = r["Prior Craniotomy/Biopsy/Resection"].strip()
        if v not in ("Yes", "No"):
            raise ValueError(f"{r['SubjectID']}: unexpected surgery value {v!r}")
        out[r["SubjectID"].strip()] = {"prior_surgery": v == "Yes"}
    return out


def stage_raw(disease, raw_root):
    srcs = sources(disease, FM)
    extra = surgery_flags(BMSR_SUBJECTS) if disease == "metastasis" else None
    if extra is not None and set(extra) != set(srcs):
        raise ValueError(f"the subject table and the scans differ: {sorted(set(extra) ^ set(srcs))[:5]}")
    base = build_raw(raw_root, disease, srcs, extra)
    info = json.loads((base / "cases.json").read_text())
    n, p = len(info), len({v["patient"] for v in info.values()})
    print(f"wrote {base}: {n} cases, {p} patients, {sum(1 for v in info.values() if not v['n_label_voxels'])} cases without label voxels")
    if (n, p) != EXPECTED[disease]:
        raise AssertionError(f"expected {EXPECTED[disease]} (scans, patients), got {(n, p)}")


def stage_splits(disease, raw_root, preprocessed_root):
    info = json.loads((Path(raw_root) / DISEASES[disease]["name"] / "cases.json").read_text())
    cf = case_folds(disease, sorted(info))
    p = write_splits(preprocessed_root, disease, cf)
    print(f"wrote {p}: cases per fold {sorted(collections.Counter(cf.values()).items())}")


def main():
    ap = argparse.ArgumentParser(description="Build a brain disease dataset for nnU-Net or write its folds")
    ap.add_argument("--disease", choices=sorted(DISEASES), required=True)
    ap.add_argument("--stage", choices=("raw", "splits"), required=True)
    a = ap.parse_args()
    if a.stage == "raw":
        stage_raw(a.disease, os.environ["nnUNet_raw"])
    else:
        stage_splits(a.disease, os.environ["nnUNet_raw"], os.environ["nnUNet_preprocessed"])


if __name__ == "__main__":
    main()
