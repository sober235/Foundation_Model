#!/usr/bin/env python
# scripts/aur_probe.py
"""Memory and speed probe of the AnatoBind brain model (spec §12 P3): synthetic crops of the training size, the full
loss (A + S + U + R), AMP bfloat16, gradient checkpointing; reports peak memory and seconds per step as one JSON line.

Single card:  PYTHONNOUSERSITE=1 PYTHONPATH=. nice -n 19 ~/anaconda3/envs/nvgen/bin/python scripts/aur_probe.py --gpu 7 --batch 2 --steps 20
Four cards:   PYTHONNOUSERSITE=1 PYTHONPATH=. CUDA_VISIBLE_DEVICES=4,5,6,7 nice -n 19 ~/anaconda3/envs/nvgen/bin/torchrun --nproc_per_node 4 scripts/aur_probe.py --ddp --batch 2 --steps 200

Writes nothing; the caller redirects stdout into the records folder."""
import argparse
import json
import os
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from anatobind.aur import losses as L  # noqa: E402
from anatobind.aur.labels import N_ENTITIES, N_HOST_CLASSES  # noqa: E402
from anatobind.aur.model import AnatoBindBrain  # noqa: E402


def synthetic_batch(batch, crop, device, seed=0):
    g = torch.Generator().manual_seed(seed)
    D, H, W = crop
    image = torch.randn(batch, 1, D, H, W, generator=g)
    valid = torch.ones(batch, D, H, W)
    valid[:, :, :, W - W // 8:] = 0.0
    axes = [torch.arange(n, dtype=torch.float32) + 0.5 for n in crop]
    coords = torch.stack(torch.meshgrid(*axes, indexing="ij"), 0)[None].expand(batch, -1, -1, -1, -1).clone()
    local = torch.stack(torch.meshgrid(*[(a / n) * 2 - 1 for a, n in zip(axes, crop)], indexing="ij"), 0)[None].expand(batch, -1, -1, -1, -1).clone()
    entity = torch.randint(0, N_ENTITIES + 1, (batch, D, H, W), generator=g)
    instance = torch.zeros(batch, D, H, W, dtype=torch.long)
    instance[:, D // 4:D // 2, H // 4:H // 2, W // 4:W // 2] = 1
    instance[:, D // 2:3 * D // 4, H // 2:3 * H // 4, W // 2:3 * W // 4] = 2
    return {k: v.to(device) for k, v in dict(image=image, valid=valid, coords=coords, local=local, entity=entity, instance=instance).items()}


def step(model, batch, points, opt, scaler_dtype):
    out_model = model.module if hasattr(model, "module") else model
    with torch.autocast("cuda", dtype=scaler_dtype):
        out = model(batch["image"], batch["valid"], batch["coords"], batch["local"])
        em, um = out_model.entity_masks(out, points), out_model.event_masks(out, points)
        entity_pts, inst_pts = L.gather(batch["entity"], points), L.gather(batch["instance"], points)
        B = em.shape[0]
        parts = L.entity_loss(em.float(), out["entity_presence"].float(), entity_pts, torch.zeros_like(entity_pts, dtype=torch.bool),
                              torch.ones(B, N_ENTITIES, dtype=torch.bool, device=em.device))
        targets = [(inst_pts[b][None] == torch.tensor([[1], [2]], device=em.device)).float() for b in range(B)]
        u, matches = L.event_loss(out["event_presence"].float(), um.float(), targets, torch.ones_like(inst_pts, dtype=torch.float32),
                                  torch.ones(B, dtype=torch.bool, device=em.device))
        parts.update(u)
        parts.update(L.seq_loss(out["seq_logits"].float(), torch.zeros(B, dtype=torch.long, device=em.device)))
        r = {"r_host": 0.0, "r_hard": 0.0}
        for b in range(B):
            qi, ti = matches[b]
            logits = out_model.bind(out, b, qi).float()
            part = L.relation_loss(logits, torch.tensor([0, 1], device=em.device)[ti], torch.tensor([[1, -1], [0, -1]], device=em.device)[ti])
            r = {k: r[k] + part[k] / B for k in r}
        parts.update(r)
        total, logged = L.total(parts)
    opt.zero_grad(set_to_none=True)
    total.backward()
    opt.step()
    return float(total)


def main(argv=None):
    ap = argparse.ArgumentParser(description="AnatoBind brain memory / speed probe")
    ap.add_argument("--gpu", type=int, default=None, help="card for the single-card probe (ignored under --ddp)")
    ap.add_argument("--ddp", action="store_true", help="run under torchrun, one process per card")
    ap.add_argument("--batch", type=int, default=2)
    ap.add_argument("--crop", type=int, nargs=3, default=(128, 160, 160))
    ap.add_argument("--steps", type=int, default=20)
    ap.add_argument("--points", type=int, default=L.N_POINTS)
    ap.add_argument("--no-checkpoint", action="store_true")
    a = ap.parse_args(argv)
    if a.ddp:
        torch.distributed.init_process_group("nccl")
        rank, world = torch.distributed.get_rank(), torch.distributed.get_world_size()
        device = torch.device("cuda", int(os.environ["LOCAL_RANK"]))
    else:
        rank, world = 0, 1
        device = torch.device("cuda", a.gpu if a.gpu is not None else 0)
    torch.cuda.set_device(device)
    model = AnatoBindBrain(use_checkpoint=not a.no_checkpoint).to(device)
    if a.ddp:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[device.index])
    opt = torch.optim.AdamW(model.parameters(), lr=1e-4)
    batch = synthetic_batch(a.batch, tuple(a.crop), device, seed=rank)
    points = L.sample_points(batch["valid"].cpu(), a.points, torch.Generator().manual_seed(rank)).to(device)
    torch.cuda.reset_peak_memory_stats(device)
    losses = []
    for i in range(a.steps):
        if i == 3:
            torch.cuda.synchronize(device)
            t0 = time.time()
        losses.append(step(model, batch, points, opt, torch.bfloat16))
    torch.cuda.synchronize(device)
    per_step = (time.time() - t0) / max(a.steps - 3, 1)
    peak = torch.cuda.max_memory_allocated(device) / 2 ** 30
    result = {"rank": rank, "world": world, "gpu": device.index, "batch": a.batch, "crop": list(a.crop), "points": a.points,
              "checkpoint": not a.no_checkpoint, "steps": a.steps, "s_per_step": round(per_step, 3), "peak_gib": round(peak, 2),
              "first_loss": round(losses[0], 4), "last_loss": round(losses[-1], 4),
              "params": (model.module if a.ddp else model).num_parameters()}
    print(json.dumps(result))
    if a.ddp:
        torch.distributed.destroy_process_group()
    return 0


if __name__ == "__main__":
    sys.exit(main())
