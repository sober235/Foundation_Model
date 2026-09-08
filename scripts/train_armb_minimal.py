"""Fold-0 smoke trainer for the arm-B minimal path.

This is NOT an M1 experiment.  The model is fed ground-truth masks, presence and
matched boxes for its geometry features, so the reported host accuracy is not
evidence for or against the M1 proposition -- it is logged as NOT-EVIDENCE.  The
purpose of this script is to prove the data engine, the frame convention and the
loss plumbing agree with each other.  See
docs/superpowers/plans/2026-09-07-m1-arm-b-minimal-path.md.

    PYTHONNOUSERSITE=1 PYTHONPATH=. CUDA_VISIBLE_DEVICES=0 \
        ~/anaconda3/envs/nvgen/bin/python scripts/train_armb_minimal.py --steps 300
"""

import argparse
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from anatobind.model.armb import ArmBMinimal
from anatobind.train.dataset import SkmteaArmBDataset, collate, load_fold

ROOT = Path("/data2/congcong/data/FM_data/derived/skmtea/m1")


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


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


@torch.no_grad()
def evaluate(model, loader, device, amp):
    model.eval()
    hit = total = 0
    losses = []
    for batch in loader:
        batch = to_device(batch, device)
        with torch.autocast(device.type, dtype=torch.bfloat16, enabled=amp):
            out = model(batch)
            loss, _ = model.compute_loss(out, batch)
        losses.append(float(loss))
        h, t = model.host_accuracy(out, batch)
        hit, total = hit + h, total + t
    model.train()
    return {
        "val_loss": float(np.mean(losses)) if losses else float("nan"),
        "val_host_acc_NOT_EVIDENCE": hit / total if total else float("nan"),
        "val_matched": total,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, default=ROOT)
    ap.add_argument("--fold", type=int, default=0)
    ap.add_argument("--steps", type=int, default=300)
    ap.add_argument("--batch", type=int, default=2)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--patch", type=int, nargs=3, default=[64, 128, 128])
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--eval-every", type=int, default=50)
    ap.add_argument("--out", type=Path, default=None)
    a = ap.parse_args()

    seed_all(a.seed)
    out_dir = a.out or Path("runs") / f"armb_minimal_{time.strftime('%Y%m%d_%H%M%S')}"
    out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    amp = device.type == "cuda"

    train_ids, val_ids = load_fold(a.root, a.fold)
    if not train_ids:
        raise SystemExit(f"no exported scans under {a.root}")
    patch = tuple(a.patch)
    train_ds = SkmteaArmBDataset(train_ids, a.root, patch=patch, train=True, seed=a.seed)
    val_ds = SkmteaArmBDataset(val_ids, a.root, patch=patch, train=False, seed=a.seed)
    train_loader = DataLoader(train_ds, batch_size=a.batch, shuffle=True, drop_last=True,
                              num_workers=a.workers, collate_fn=collate)
    val_loader = DataLoader(val_ds, batch_size=a.batch, shuffle=False,
                            num_workers=a.workers, collate_fn=collate)

    model = ArmBMinimal().to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=0.05)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=a.steps)

    config = {
        **vars(a), "root": str(a.root), "out": str(out_dir),
        "device": str(device), "n_train": len(train_ids), "n_val": len(val_ids),
        "parameters": model.backbone.num_parameters(),
        "total_parameters": sum(p.numel() for p in model.parameters()),
        "note": "MINIMAL PATH -- ground-truth geometry; accuracy is NOT evidence",
    }
    (out_dir / "config.json").write_text(json.dumps(config, indent=1))
    print(json.dumps(config, indent=1), flush=True)

    log = open(out_dir / "metrics.jsonl", "a")
    step, t0 = 0, time.time()
    model.train()
    while step < a.steps:
        for batch in train_loader:
            if step >= a.steps:
                break
            batch = to_device(batch, device)
            with torch.autocast(device.type, dtype=torch.bfloat16, enabled=amp):
                loss, parts = model.compute_loss(model(batch), batch)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            gnorm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
            step += 1

            row = {"step": step, "loss": float(loss), **parts,
                   "grad_norm": float(gnorm), "lr": sched.get_last_lr()[0],
                   "seconds": round(time.time() - t0, 1)}
            if step % 10 == 0 or step == 1:
                print(json.dumps(row), flush=True)
            if step % a.eval_every == 0 or step == a.steps:
                row.update(evaluate(model, val_loader, device, amp))
                print(json.dumps(row), flush=True)
            log.write(json.dumps(row) + "\n")
            log.flush()

    torch.save({"model": model.state_dict(), "config": config}, out_dir / "last.pt")
    log.close()
    print(f"done: {out_dir}", flush=True)


if __name__ == "__main__":
    main()
