"""The Stage I trainer (SSL-first plan §3.3, T07/T08; decisions Q11, Q19 of 2026-10-09).

One code path serves the probe (a few timed steps, no validation) and the training (pilot and main run): torchrun
starts one process per GPU; every process draws its own volumes with the patient-balanced weights (its own seed per
epoch and rank), cuts crops_per_volume crops of each, and feeds micro-batches of `microbatch` crops; `grad_accum`
micro-batches make one optimizer step, so one step sees microbatch x world x grad_accum source crops and the budget is
counted in those crops (seen_crops), not in steps. AdamW, bf16 autocast, linear warmup then cosine to zero over the
steps the budget needs. Every val_every steps (and at the end) the validation crops (fixed, from the validation
patients, no gradient) give the masked Huber, its no-learning baseline, the two-view cosine, the effective rank of the
projections and the hidden share; the lowest validation Huber selects the exported backbone. Resume checkpoints and
exports go to new file names; the run directory must not exist when the run starts. Logs hold no patient detail."""
import json
import math
import os
import platform
import time
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, WeightedRandomSampler

from anatobind.aur.crops import CROP
from anatobind.aur.ssl import checkpoint as CK
from anatobind.aur.ssl.dataset import SSLDataset, collate_ssl, source_weights
from anatobind.aur.ssl.model import StageOne, effective_rank
from anatobind.aur.ssl.samples import HCP_EXPOSURE_CAP

DEFAULTS = {"seen_crops": 320_000, "microbatch": 2, "grad_accum": 2, "crops_per_volume": 2, "lr": 3e-4, "weight_decay": 0.05,
            "warmup_steps": 1000, "mask_ratio": 0.6, "block_mm": (16.0, 32.0), "contrast_weight": 0.1, "temperature": 0.2,
            "crop": CROP, "workers": 4, "seed": 0, "val_every": 5000, "val_volumes": 32, "save_every": 5000, "log_every": 20,
            "hcp_cap": HCP_EXPOSURE_CAP, "use_checkpoint": False, "max_steps": None, "probe": False, "no_rotate": False,
            "embed": 64, "depths": (2, 2, 6, 2), "heads": (2, 4, 8, 16)}


def distributed():
    """(rank, world, local_rank); initialises the process group when torchrun launched us."""
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        if not dist.is_initialized():
            dist.init_process_group("nccl" if torch.cuda.is_available() else "gloo")
        return dist.get_rank(), dist.get_world_size(), int(os.environ.get("LOCAL_RANK", 0))
    return 0, 1, 0


def lr_lambda(step, warmup, total):
    if step < warmup:
        return (step + 1) / max(warmup, 1)
    t = (step - warmup) / max(total - warmup, 1)
    return 0.5 * (1.0 + math.cos(math.pi * min(t, 1.0)))


def make_loader(rows, cfg, rank, world, epoch, device_count):
    """A loader over the train rows: weighted draws with replacement, this rank's and epoch's seed; one volume per
    micro-batch when crops_per_volume == microbatch (the common case), else the nearest number of volumes."""
    weights, share = source_weights(rows, cfg["hcp_cap"])
    volumes_per_batch = max(1, cfg["microbatch"] // cfg["crops_per_volume"])
    draws = max(volumes_per_batch, (cfg["seen_crops"] // max(world, 1) // cfg["crops_per_volume"]) + volumes_per_batch)
    g = torch.Generator().manual_seed(int(np.random.SeedSequence([cfg["seed"], epoch, rank]).generate_state(1)[0]))
    sampler = WeightedRandomSampler(torch.as_tensor(weights, dtype=torch.double), num_samples=draws, replacement=True, generator=g)
    ds = SSLDataset(rows, cfg["crop"], cfg["crops_per_volume"], do_rotate=not cfg["no_rotate"], seed=cfg["seed"] + 1000 * rank)
    ds.set_epoch(epoch)
    loader = DataLoader(ds, batch_size=volumes_per_batch, sampler=sampler, num_workers=cfg["workers"], collate_fn=collate_ssl,
                        pin_memory=device_count > 0, drop_last=True, persistent_workers=cfg["workers"] > 0, prefetch_factor=2 if cfg["workers"] > 0 else None)
    return loader, share


def validation_batches(rows, cfg, device):
    """Fixed validation crops: val_volumes volumes (seeded), crops_per_volume crops each, no rotation, as micro-batches."""
    if not rows:
        return []
    rng = np.random.default_rng(cfg["seed"] + 7)
    pick = [rows[i] for i in rng.choice(len(rows), size=min(cfg["val_volumes"], len(rows)), replace=False)]
    ds = SSLDataset(pick, cfg["crop"], cfg["crops_per_volume"], do_rotate=False, seed=cfg["seed"] + 7)
    items = [ds[i] for i in range(len(ds))]
    crops = [c for item in items for c in item]
    step = max(1, cfg["microbatch"])
    return [collate_ssl([crops[i:i + step]]) for i in range(0, len(crops), step)]


def to_device(batch, device):
    return {k: (v.to(device, non_blocking=True) if torch.is_tensor(v) else v) for k, v in batch.items()}


@torch.no_grad()
def validate(model, batches, device, seed):
    """Mean masked Huber, its baseline, the two-view cosine, the effective rank of z1 and the hidden share."""
    model.eval()
    sums, n, zs = {"mim": 0.0, "mim_baseline": 0.0, "hidden_share": 0.0, "contrast_acc": 0.0}, 0, []
    cos = 0.0
    for i, b in enumerate(batches):
        b = to_device(b, device)
        g = torch.Generator().manual_seed(seed + i)
        with torch.autocast(device_type="cuda" if device.type == "cuda" else "cpu", dtype=torch.bfloat16, enabled=device.type == "cuda"):
            out = model(b, g)
        for k in sums:
            sums[k] += float(out[k])
        cos += float((out["z1"] * out["z2"]).sum(-1).mean())
        zs.append(out["z1"].float().cpu())
        n += 1
    model.train()
    if n == 0:
        return {}
    res = {k: v / n for k, v in sums.items()}
    res["view_cosine"] = cos / n
    z = torch.cat(zs, 0)
    res["effective_rank_z"] = float(effective_rank(z)) if z.shape[0] > 1 else float("nan")
    res["val_crops"] = int(z.shape[0])
    return res


def run(cfg, samples, out_dir, resume=None):
    cfg = {**DEFAULTS, **cfg}
    rank, world, local_rank = distributed()
    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.cuda.set_device(device)
    out_dir = Path(out_dir)
    if rank == 0:
        if out_dir.exists() and resume is None:
            raise FileExistsError(f"{out_dir} exists; a run never writes into an existing directory")
        out_dir.mkdir(parents=True, exist_ok=True)
    if world > 1:
        dist.barrier()
    rows = json.loads(Path(samples).read_text())
    train_rows = [r for r in rows if r.get("ssl_split", "train") == "train"]
    val_rows = [r for r in rows if r.get("ssl_split") == "val"]
    model = StageOne(embed=cfg["embed"], depths=tuple(cfg["depths"]), heads=tuple(cfg["heads"]), use_checkpoint=cfg["use_checkpoint"],
                     mask_ratio=cfg["mask_ratio"], block_mm=tuple(cfg["block_mm"]), contrast_weight=cfg["contrast_weight"],
                     temperature=cfg["temperature"]).to(device)
    global_batch = cfg["microbatch"] * world * cfg["grad_accum"]
    total_steps = math.ceil(cfg["seen_crops"] / global_batch)
    if cfg["max_steps"] is not None:
        total_steps = min(total_steps, int(cfg["max_steps"]))
    opt = torch.optim.AdamW(model.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"], betas=(0.9, 0.95))
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: lr_lambda(s, cfg["warmup_steps"], total_steps))
    step, seen = 0, 0
    if resume is not None:
        meta, _ = CK.load_resume(resume, model, opt, sched, map_location=device)
        step, seen = int(meta["step"]), int(meta["seen_crops"])
    ddp = DistributedDataParallel(model, device_ids=[local_rank] if device.type == "cuda" else None) if world > 1 else model
    val_batches = [] if cfg["probe"] else validation_batches(val_rows, cfg, device)
    log_path = out_dir / f"log_rank{rank}.jsonl"
    best = {"mim": float("inf"), "step": None, "path": None}
    times, waits = [], []
    if rank == 0:
        info = {"config": cfg, "samples": str(samples), "manifest_sha256": CK.file_sha256(samples), "code_sha": CK.code_sha(), "world": world,
                "global_batch": global_batch, "total_steps": total_steps, "train_rows": len(train_rows), "val_rows": len(val_rows),
                "train_patients": len({(r["source"], r["patient"]) for r in train_rows}), "params": model.num_parameters(),
                "backbone_params": model.backbone.num_parameters(), "host": platform.node(),
                "gpus": [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())] if device.type == "cuda" else [],
                "started": time.strftime("%Y-%m-%d %H:%M:%S")}
        (out_dir / ("probe_config.json" if cfg["probe"] else "run_config.json")).write_text(json.dumps(info, indent=1, default=str))
    model.train()
    epoch, done = 0, step >= total_steps
    t_step = time.time()
    while not done:
        loader, share = make_loader(train_rows, cfg, rank, world, epoch, torch.cuda.device_count())
        it = iter(loader)
        while not done:
            t0 = time.time()
            micro = []
            try:
                for _ in range(cfg["grad_accum"]):
                    micro.append(next(it))
            except StopIteration:
                break
            waits.append(time.time() - t0)
            opt.zero_grad(set_to_none=True)
            stats = {}
            for j, b in enumerate(micro):
                b = to_device(b, device)
                g = torch.Generator().manual_seed(int(np.random.SeedSequence([cfg["seed"], step, rank, j]).generate_state(1)[0]))
                with torch.autocast(device_type="cuda" if device.type == "cuda" else "cpu", dtype=torch.bfloat16, enabled=device.type == "cuda"):
                    out = ddp(b, g)
                    (out["loss"] / cfg["grad_accum"]).backward()
                for k in ("loss", "mim", "contrast", "contrast_acc", "mim_baseline", "hidden_share"):
                    stats[k] = stats.get(k, 0.0) + float(out[k]) / cfg["grad_accum"]
            if not all(math.isfinite(v) for v in stats.values()):
                raise FloatingPointError(f"non-finite loss at step {step}: {stats}")
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
            step += 1
            seen += global_batch
            if device.type == "cuda":
                torch.cuda.synchronize()
            times.append(time.time() - t_step)
            t_step = time.time()
            if rank == 0 and (step % cfg["log_every"] == 0 or step == 1):
                rec = {"step": step, "seen_crops": seen, "lr": sched.get_last_lr()[0], "step_s": times[-1], "data_wait_s": waits[-1], **stats}
                if device.type == "cuda":
                    rec["peak_alloc_gib"] = torch.cuda.max_memory_allocated(device) / 2 ** 30
                    rec["peak_reserved_gib"] = torch.cuda.max_memory_reserved(device) / 2 ** 30
                with open(log_path, "a") as f:
                    f.write(json.dumps(rec) + "\n")
                print(json.dumps({k: (round(v, 4) if isinstance(v, float) else v) for k, v in rec.items()}), flush=True)
            done = step >= total_steps or seen >= cfg["seen_crops"]
            if not cfg["probe"] and (step % cfg["val_every"] == 0 or done) and val_batches:
                res = validate(model, val_batches, device, cfg["seed"] + 99)
                if world > 1:
                    t = torch.tensor([res["mim"]], device=device)
                    dist.all_reduce(t)
                    res["mim_all_ranks_mean"] = float(t) / world
                if rank == 0:
                    res.update({"step": step, "seen_crops": seen})
                    with open(out_dir / "val.jsonl", "a") as f:
                        f.write(json.dumps(res) + "\n")
                    print("VAL", json.dumps({k: (round(v, 5) if isinstance(v, float) else v) for k, v in res.items()}), flush=True)
                    if res["mim"] < best["mim"]:
                        meta = CK.metadata(cfg, samples, step, seen, cfg["seed"], extra={"validation": res, "stage": "I"})
                        best = {"mim": res["mim"], "step": step, "path": CK.export_backbone(out_dir / f"ssl_stage1_step{step}.pt", model.backbone, meta)}
                        (out_dir / "best.json").write_text(json.dumps(best, indent=1))
            if rank == 0 and not cfg["probe"] and (step % cfg["save_every"] == 0 or done):
                meta = CK.metadata(cfg, samples, step, seen, cfg["seed"], extra={"stage": "I"})
                CK.save_resume(out_dir / f"resume_step{step}.pt", model, opt, sched, meta)
        epoch += 1
    summary = None
    if rank == 0:
        arr = np.array(times[1:]) if len(times) > 1 else np.array(times)
        summary = {"steps": step, "seen_crops": seen, "world": world, "global_batch": global_batch,
                   "step_s_mean": float(arr.mean()) if arr.size else None, "step_s_p95": float(np.percentile(arr, 95)) if arr.size else None,
                   "data_wait_s_mean": float(np.mean(waits[1:])) if len(waits) > 1 else None,
                   "crops_per_s": float(global_batch / arr.mean()) if arr.size else None, "best": best,
                   "finished": time.strftime("%Y-%m-%d %H:%M:%S")}
        if device.type == "cuda":
            summary["peak_alloc_gib"] = torch.cuda.max_memory_allocated(device) / 2 ** 30
            summary["peak_reserved_gib"] = torch.cuda.max_memory_reserved(device) / 2 ** 30
            summary["gpu"] = torch.cuda.get_device_name(device)
        if not cfg["probe"] and best["path"]:
            final = out_dir / "ssl_stage1_best.pt"                     # a copy of the best validation export under the fixed name
            if not final.exists():
                torch.save(torch.load(best["path"], map_location="cpu", weights_only=False), str(final))
                final.with_suffix(".json").write_text(json.dumps(best, indent=1))
            summary["final_export"] = str(final)
        (out_dir / ("probe.json" if cfg["probe"] else "summary.json")).write_text(json.dumps(summary, indent=1, default=str))
        print("SUMMARY", json.dumps(summary, default=str), flush=True)
    if world > 1:
        dist.barrier()
        dist.destroy_process_group()
    return summary
