"""leg 2 检测器的单折训练。固定步数、最后一个检查点、不做验证：留出患者上不选任何东西。

  CUDA_VISIBLE_DEVICES=0 PYTHONNOUSERSITE=1 PYTHONPATH=. ~/anaconda3/envs/nvgen/bin/python \
      scripts/train_detector.py --fold 0 --out runs/detector_fold0 --resume
"""
import argparse
import csv
import json
import math
import random
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from anatobind.data_engine.fastmri_knee import EXPORT_ROOT
from anatobind.model.dense_head_2d import centre_loss_2d
from anatobind.model.detector2d import Detector2D
from anatobind.train.dataset_knee import SlabDataset, collate_slabs, seed_worker

FULL = dict(num_classes=5, embed_dim=48, stem_ch=32, d_model=256, slab=5)
TINY = dict(num_classes=5, embed_dim=12, stem_ch=8, d_model=16, slab=5)


def parse(argv):
    ap = argparse.ArgumentParser()
    ap.add_argument("--fold", type=int, required=True)
    ap.add_argument("--steps", type=int, default=20000)
    ap.add_argument("--warmup", type=int, default=500)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--wd", type=float, default=0.05)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--ckpt-every", type=int, default=1000)
    ap.add_argument("--log-every", type=int, default=50)
    ap.add_argument("--export-root", type=Path, default=EXPORT_ROOT)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--grad-ckpt", action="store_true")
    ap.add_argument("--tiny", action="store_true")
    ap.add_argument("--cpu", action="store_true")
    return ap.parse_args(argv)


def lr_lambda(warmup, total):
    def f(step):
        if step < warmup:
            return (step + 1) / warmup
        return 0.5 * (1.0 + math.cos(math.pi * (step - warmup) / max(1, total - warmup)))
    return f


def load_fold(export_root, fold):
    folds = json.loads((Path(export_root) / "folds.json").read_text())["folds"]
    train = sorted(f for f, k in folds.items() if k != fold)
    held = sorted(f for f, k in folds.items() if k == fold)
    with open(Path(export_root) / "lesions.csv", newline="") as fh:
        lesions = [{**r, **{k: int(r[k]) for k in ("z0", "z1", "x0", "y0", "x1", "y1", "n_boxes")}}
                   for r in csv.DictReader(fh)]
    return train, held, lesions


def save_atomic(obj, path):
    tmp = path.with_suffix(".tmp")
    torch.save(obj, tmp)
    tmp.replace(path)


def main(argv=None):
    a = parse(argv)
    random.seed(a.seed)
    np.random.seed(a.seed)
    torch.manual_seed(a.seed)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() and not a.cpu else "cpu")
    train_files, _, lesions = load_fold(a.export_root, a.fold)
    ds = SlabDataset(train_files, a.export_root, lesions, train=True, seed=a.seed)
    loader = DataLoader(ds, batch_size=a.batch, shuffle=True, drop_last=True, num_workers=a.workers,
                        collate_fn=collate_slabs, worker_init_fn=seed_worker,
                        persistent_workers=a.workers > 0)
    cfg = TINY if a.tiny else FULL
    model = Detector2D(**cfg, use_checkpoint=a.grad_ckpt).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=a.wd)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda(a.warmup, a.steps))
    step, ckpt = 0, out / "ckpt.pt"
    if a.resume and ckpt.exists():
        state = torch.load(ckpt, map_location="cpu", weights_only=False)
        model.load_state_dict(state["model"])
        opt.load_state_dict(state["opt"])
        sched.load_state_dict(state["sched"])
        step = state["step"]
    config = {**{k: str(v) if isinstance(v, Path) else v for k, v in vars(a).items()},
              "model": cfg, "n_train": len(train_files), "device": str(device),
              "parameters": sum(p.numel() for p in model.parameters())}
    (out / "config.json").write_text(json.dumps(config, indent=1))
    amp = device.type == "cuda"
    log = open(out / "metrics.jsonl", "a")
    t0 = time.time()
    model.train()
    while step < a.steps:
        for batch in loader:
            if step >= a.steps:
                break
            image = batch["image"].to(device, non_blocking=True)
            batch = {**batch, "boxes": [b.to(device) for b in batch["boxes"]],
                     "box_classes": [c.to(device) for c in batch["box_classes"]]}
            with torch.autocast(device.type, dtype=torch.bfloat16, enabled=amp):
                pred = model(image)
            parts = centre_loss_2d(pred, batch)
            loss = parts["heat"] + parts["offset"] + parts["size"]
            opt.zero_grad(set_to_none=True)
            loss.backward()
            gnorm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
            step += 1
            row = {"step": step, "loss": float(loss), **{k: float(v) for k, v in parts.items()},
                   "grad_norm": float(gnorm), "lr": sched.get_last_lr()[0],
                   "seconds": round(time.time() - t0, 1)}
            log.write(json.dumps(row) + "\n")
            log.flush()
            if step == 1 or step % a.log_every == 0:
                print(json.dumps(row), flush=True)
            if step % a.ckpt_every == 0:
                save_atomic({"model": model.state_dict(), "opt": opt.state_dict(), "sched": sched.state_dict(),
                             "step": step, "config": config}, ckpt)
    save_atomic({"model": model.state_dict(), "config": config, "step": step}, out / "last.pt")
    log.close()
    print(f"done: {out} at step {step}", flush=True)
