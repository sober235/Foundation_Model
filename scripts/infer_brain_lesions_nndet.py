#!/usr/bin/env python
# scripts/infer_brain_lesions_nndet.py
"""Brain small-lesion inference with the nnDetection second arm (spec 2026-09-28 brain-nndet §8).

  PYTHONNOUSERSITE=1 PYTHONPATH=. ~/anaconda3/envs/nvgen/bin/python scripts/infer_brain_lesions_nndet.py \
      --h5 FILE.h5 --out DIR --fold 0 --gpu 3
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from anatobind.infer.brain_nndet import run  # noqa: E402
from anatobind.nndet.brain_task import MODEL_ID, TASK_NAME  # noqa: E402

DET_MODELS = Path("/data2/congcong/data/FM_data/derived/nndet_models")


def main():
    ap = argparse.ArgumentParser(description="Brain small-lesion inference, nnDetection arm")
    ap.add_argument("--h5", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True, help="output directory (must not exist)")
    ap.add_argument("--fold", type=int, default=0)
    ap.add_argument("--gpu", type=int, required=True)
    a = ap.parse_args()
    train_dir = DET_MODELS / TASK_NAME / MODEL_ID / f"fold{a.fold}"
    if not (train_dir / "model_last.ckpt").exists():
        raise FileNotFoundError(f"{train_dir / 'model_last.ckpt'} missing")
    rows = run(a.h5, a.out, train_dir, a.gpu)
    print(f"{len(rows)} lesions -> {a.out / 'lesions.json'}")


if __name__ == "__main__":
    main()
