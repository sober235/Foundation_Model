#!/usr/bin/env python
# scripts/infer_brain_disease.py
"""Brain disease inference: a study's sequences and its SynthSeg label map -> the structured record (spec 2026-09-29 §7).

  PYTHONNOUSERSITE=1 PYTHONPATH=. ~/anaconda3/envs/nvgen/bin/python scripts/infer_brain_disease.py --disease infarct \
      --images DWI.nii.gz ADC.nii.gz --anatomy DWI_seg.nii.gz --threshold 0.55 --out DIR --folds 0 1 2 3 4 --gpu 4
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from anatobind.infer.brain_disease import run  # noqa: E402
from anatobind.nnunet.brain_disease import DISEASES  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description="Brain disease inference with binding and a structured record")
    ap.add_argument("--disease", choices=sorted(DISEASES), required=True)
    ap.add_argument("--images", type=Path, nargs="+", required=True, help="one NIfTI per channel, in the model's channel order")
    ap.add_argument("--anatomy", type=Path, required=True, help="SynthSeg label map on the images' grid")
    ap.add_argument("--threshold", type=float, required=True, help="the disease's operating threshold (verdict.json)")
    ap.add_argument("--out", type=Path, required=True, help="output directory (must not exist)")
    ap.add_argument("--folds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    ap.add_argument("--gpu", type=int, required=True)
    ap.add_argument("--study", help="study name written into the record (default: the first image's file name)")
    a = ap.parse_args()
    print(f"channels expected: {DISEASES[a.disease]['channels']}")
    rec = run(a.disease, a.images, a.anatomy, a.out, a.folds, a.gpu, a.threshold, study=a.study)
    print(f"{len(rec['lesions'])} lesions -> {a.out / 'record.json'}")
    print(rec["sentence"])


if __name__ == "__main__":
    main()
