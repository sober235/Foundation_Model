"""Stage-II training of the whole-volume upstream for one fold (RESEARCH_PLAN v2.2 13.6).

Fixed step count, last checkpoint, no validation: nothing is selected on held-out scans.
Four whole volumes per step (user decision 2026-09-11): on the shared GPUs every kernel launch
stalls about 1.2 ms, so a step costs about the same whatever it carries; 7500 steps = 30,000 volumes.
Resumable: ckpt.pt is rewritten every --ckpt-every steps; last.pt is written at the end.

  CUDA_VISIBLE_DEVICES=6 PYTHONNOUSERSITE=1 PYTHONPATH=. ~/anaconda3/envs/nvgen/bin/python \
      scripts/train_upstream.py --fold 0 --steps 7500 --out runs/upstream_fold0 --resume
"""
import argparse
import json
import math
import random
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from anatobind.model.upstream import Upstream
from anatobind.model.upstream_losses import upstream_loss
from anatobind.train.dataset import load_fold
from anatobind.train.dataset_v2 import WholeVolumeDataset, collate_batch, seed_worker

FM = Path("/data2/congcong/data/FM_data/derived/skmtea")
# pixel_dim=0: the fold-0 pilot configuration. The top-down pixel decoder (pixel_dim=64) was not better
# after 500 steps on 2026-09-12 (mask loss 0.83 vs 0.70), so the batch-1 folds keep the tested head.
FULL = dict(K=6, M=20, num_classes=4, d_model=256, embed_dim=64, layers=6, heads=8, mask_dim=32, pixel_dim=0)
TINY = dict(K=6, M=4, num_classes=4, d_model=32, embed_dim=16, layers=2, heads=4, mask_dim=8)


def parse(argv):
    ap = argparse.ArgumentParser()
    ap.add_argument("--fold", type=int, required=True)
    ap.add_argument("--steps", type=int, default=7500)
    ap.add_argument("--warmup", type=int, default=125)
    ap.add_argument("--batch", type=int, default=4, help="whole volumes per step")
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--wd", type=float, default=0.05)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--ckpt-every", type=int, default=1000)
    ap.add_argument("--log-every", type=int, default=50)
    ap.add_argument("--export-root", type=Path, default=FM / "m1r")
    ap.add_argument("--cache-root", type=Path, default=FM / "m1r_cache")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--grad-ckpt", action="store_true")
    ap.add_argument("--tiny", action="store_true", help="test-sized model")
    ap.add_argument("--cpu", action="store_true")
    return ap.parse_args(argv)


def lr_lambda(warmup, total):
    def f(step):
        if step < warmup:
            return (step + 1) / warmup
        return 0.5 * (1.0 + math.cos(math.pi * (step - warmup) / max(1, total - warmup)))
    return f


def to_device(batch, device):
    out = {}
    for k, v in batch.items():
        if torch.is_tensor(v):
            out[k] = v.to(device, non_blocking=True)
        elif isinstance(v, list) and v and torch.is_tensor(v[0]):
            out[k] = [t.to(device) for t in v]
        else:
            out[k] = v
    return out


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
    train_ids, _ = load_fold(a.export_root, a.fold)
    ds = WholeVolumeDataset(train_ids, a.cache_root, a.export_root, train=True, seed=a.seed)
    loader = DataLoader(ds, batch_size=a.batch, shuffle=True, drop_last=True, num_workers=a.workers,
                        collate_fn=collate_batch, worker_init_fn=seed_worker, persistent_workers=a.workers > 0)
    cfg = TINY if a.tiny else FULL
    model = Upstream(**cfg, use_checkpoint=a.grad_ckpt).to(device)
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
              "model": cfg, "n_train": len(train_ids), "device": str(device),
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
            batch = to_device(batch, device)
            with torch.autocast(device.type, dtype=torch.bfloat16, enabled=amp):
                pred = model(batch["image"])
            loss, parts = upstream_loss(pred, batch)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            gnorm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
            step += 1
            row = {"step": step, "loss": float(loss), **parts, "grad_norm": float(gnorm),
                   "lr": sched.get_last_lr()[0], "seconds": round(time.time() - t0, 1),
                   "scans": batch["scan_id"], "views": batch["view"]}
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
