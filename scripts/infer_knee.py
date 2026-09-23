#!/usr/bin/env python
"""Knee: one NIfTI volume -> anatomy.nii.gz, lesions.nii.gz, lesions.csv, overlay.png (spec 2026-09-23 §3.4).

  source scripts/nnunet_env.sh
  PYTHONPATH=. python scripts/infer_knee.py --image <vol.nii.gz> --out <dir> [--frame h5|world] [--folds 0 1 2 3 4] [--gpu 0]

--frame h5: the volume is already on the SKM-TEA export grid (export files, caches).
--frame world: trust the NIfTI affine and reorder the axes to (I, P, R) first (DICOM-derived volumes).
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from anatobind.infer.knee import CSV_FIELDS, run  # noqa: E402


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--frame", choices=("h5", "world"), default="h5")
    ap.add_argument("--folds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    ap.add_argument("--gpu", type=int, default=0)
    a = ap.parse_args(argv)
    rows = run(a.image, a.out, a.frame, a.folds, a.gpu)
    print("\t".join(CSV_FIELDS))
    for r in rows:
        print("\t".join(str(r[k]) for k in CSV_FIELDS))
    print(f"{len(rows)} lesions -> {a.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
