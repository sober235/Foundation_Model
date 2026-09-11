#!/usr/bin/env python
"""Cache one fold model's predictions for every scan and view.

  CUDA_VISIBLE_DEVICES=6 PYTHONNOUSERSITE=1 PYTHONPATH=. ~/anaconda3/envs/nvgen/bin/python scripts/cache_predictions.py --fold 0
Held-out and training scans alike: the training-scan predictions measure the train/test geometry
gap (13.6) and feed the stage-III binders of the next batch.
"""
import argparse
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from anatobind.eval.predict import predict_volume, save_prediction  # noqa: E402
from anatobind.model.upstream import Upstream  # noqa: E402
from anatobind.train.dataset import list_ready_scans  # noqa: E402
from anatobind.train.dataset_v2 import WholeVolumeDataset  # noqa: E402

FM = Path("/data2/congcong/data/FM_data/derived/skmtea")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fold", type=int, required=True)
    ap.add_argument("--runs", type=Path, default=Path("runs"))
    a = ap.parse_args()
    device = torch.device("cuda")
    state = torch.load(a.runs / f"upstream_fold{a.fold}" / "last.pt", map_location="cpu", weights_only=False)
    model = Upstream(**state["config"]["model"]).to(device)
    model.load_state_dict(state["model"])
    out = FM / "m1r_pred" / "ours" / f"fold{a.fold}"
    out.mkdir(parents=True, exist_ok=True)
    ds = WholeVolumeDataset(list_ready_scans(FM / "m1r"), FM / "m1r_cache", FM / "m1r", train=False)
    t0 = time.time()
    for i, (scan, view) in enumerate(ds.items):
        path = out / f"{scan}__{view}.npz"
        if path.exists():
            continue
        s = ds[i]
        save_prediction(path, predict_volume(model, torch.from_numpy(s["image"])[None], s["valid_depth"], device))
        if i % 70 == 0:
            print(f"{i}/{len(ds)} after {time.time() - t0:.0f} s", flush=True)
    print(f"fold {a.fold}: {len(ds)} predictions in {out}")


if __name__ == "__main__":
    main()
