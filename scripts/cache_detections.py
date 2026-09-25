#!/usr/bin/env python
# scripts/cache_detections.py
"""把某折检测器在其留出患者上的折外检出缓存成 pickle。

  CUDA_VISIBLE_DEVICES=0 PYTHONNOUSERSITE=1 PYTHONPATH=. ~/anaconda3/envs/nvgen/bin/python \
      scripts/cache_detections.py --fold 0
"""
import argparse
import pickle
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from anatobind.data_engine.fastmri_knee import EXPORT_ROOT, VIEWS  # noqa: E402
from anatobind.eval.detect3d import aggregate_to_3d, detect_volume  # noqa: E402
from anatobind.model.detector2d import Detector2D  # noqa: E402
from anatobind.train.train_detector import load_fold  # noqa: E402


def default_run_dir(fold):
    return Path(f"runs/detector_gate0_fold{fold}")


def refuse_existing_cache(out_dir, overwrite):
    """Refuse to overwrite an existing fold cache unless --overwrite is passed."""
    out_dir = Path(out_dir)
    if not overwrite and out_dir.exists() and any(out_dir.glob("*.pkl")):
        raise SystemExit(f"{out_dir} already has cached detections; pass --overwrite to replace them.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fold", type=int, required=True)
    ap.add_argument("--run", type=Path, default=None)
    ap.add_argument("--export-root", type=Path, default=EXPORT_ROOT)
    ap.add_argument("--out", type=Path, default=EXPORT_ROOT / "detections")
    ap.add_argument("--overwrite", action="store_true", help="allow replacing an existing fold cache")
    ap.add_argument("--score-min", type=float, default=0.05,
                    help="lowest per-slice score kept in the cache (0.01 for the Gate 0 rerun so the FROC sweep has room)")
    a = ap.parse_args()
    run = a.run or default_run_dir(a.fold)
    out_dir = a.out / f"fold{a.fold}"
    refuse_existing_cache(out_dir, a.overwrite)
    state = torch.load(run / "last.pt", map_location="cpu", weights_only=False)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = Detector2D(**state["config"]["model"]).to(device)
    model.load_state_dict(state["model"])
    _, held, _ = load_fold(a.export_root, a.fold)
    out_dir.mkdir(parents=True, exist_ok=True)
    n = 0
    for f in held:
        for view in VIEWS:
            vol = np.load(a.export_root / f / f"{view}.npy").astype(np.float32)
            dets, gfeat = detect_volume(model, vol, device, score_min=a.score_min)
            with open(out_dir / f"{f}__{view}.pkl", "wb") as fh:
                pickle.dump({"dets": aggregate_to_3d(dets), "global_feat": gfeat}, fh)
            n += 1
    print(f"fold {a.fold}: {n} (volume, view) detection files in {out_dir}")


if __name__ == "__main__":
    main()
