#!/usr/bin/env python
# scripts/infer_brain_anatomy.py
"""S4 inference entry (spec 2026-10-02 §8): a fastMRI FLAIR stack -> skull-stripped stack -> anatomy map in SynthSeg
label values (+ an optional binding demonstration of one box with BrainBinder).

  PYTHONNOUSERSITE=1 PYTHONPATH=. ~/anaconda3/envs/nvgen/bin/python scripts/infer_brain_anatomy.py \
      --h5 <fastMRI h5> --out <new directory> --gpu <idle gpu> [--box x0 y0 z0 x1 y1 z1]
  ... --nifti <RSS stack> ...
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from anatobind.infer.brain_anatomy import run  # noqa: E402


def parse(argv=None):
    ap = argparse.ArgumentParser(description="S4 brain anatomy on a fastMRI FLAIR stack")
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--h5", type=Path, help="fastMRI h5 (RSS reconstruction is used)")
    src.add_argument("--nifti", type=Path, help="a stack already in the RSS NIfTI frame")
    ap.add_argument("--out", type=Path, required=True, help="output directory (must not exist)")
    ap.add_argument("--gpu", type=int, required=True)
    ap.add_argument("--box", type=int, nargs=6, metavar=("X0", "Y0", "Z0", "X1", "Y1", "Z1"), help="half-open box on the stack grid to bind")
    return ap.parse_args(argv)


def main(argv=None, run_fn=run):
    a = parse(argv)
    rec = run_fn(a.out, a.gpu, h5=a.h5, nifti=a.nifti, box=tuple(a.box) if a.box else None)
    print(f"brain {rec['brain_ml']} mL; reliable slices {rec['reliable_slices']}; anatomy -> {rec['anatomy']}")
    if "binding" in rec:
        b = rec["binding"]
        print(f"box {rec['box']}: host {b['host']} ({b['host_rule']}), side {b['host_side']}, fractions {b['host_fractions']}")
    return rec


if __name__ == "__main__":
    main()
