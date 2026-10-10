"""The Stage II / III trainer (spec §6; SSL-first plan 2026-10-09 T10, T11; decisions Q12, Q16; readiness review P0-4).

Stage II learns A + U + S from the Stage I backbone (`--init-backbone`, strict load, refused unless the G1 report
passed or the run is a pilot; `--init random` is the C0 control and says so in its records); the relation head is
frozen and takes no part. Stage III inherits the whole Stage II export (`--resume-stage2`, strict) and adds the
relation losses with its own learning rate per parameter group. One optimizer step sees microbatch x world x
grad_accum crops; the budget is counted in crops (seen_crops). AdamW, bf16 autocast, linear warmup then cosine; in
Stage II the backbone runs at a tenth of its rate for the first backbone_warm_steps. Every rank draws its own share of
the training rows per epoch (a seeded permutation, sharded), cuts crops_per_volume crops per volume (half of them
lesion-centred when the volume has instances) and samples the loss points with the instance map so that every lesion
of a crop carries positive points. The A / U / R losses are gated per sample by a_supervised / u_supervised /
r_supervised; the parts a rank's batch does not exercise still touch every trainable parameter with a zero so that DDP
never waits for a gradient. Every val_every steps (and at the end) the validation crops (fixed, from the validation
patients, no augmentation, no gradient) give the same losses plus the entity point Dice (A-supervised crops only), the
share of lesion instances matched to a present query and (Stage III) the host accuracy, all as counts so that no
undefined ratio enters the logs; the lowest validation loss selects the export. A resume checkpoint carries the step,
the exposure count, the epoch and the position in it, so that a resumed run continues the data stream and the best
record instead of starting the epoch again. Resume checkpoints and exports go to new file names; the run directory
must not exist when the run starts. Logs hold no patient detail."""
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
from torch.utils.data import DataLoader

from anatobind.aur import losses as L
from anatobind.aur.crops import CROP
from anatobind.aur.dataset import AURDataset, collate, event_targets_at_points
from anatobind.aur.model import AnatoBindBrain
from anatobind.aur.ssl import checkpoint as CK
from anatobind.aur.ssl.train import distributed, lr_lambda

STAGES = {
    "II": {"seen_crops": 240_000, "lrs": {"backbone": 5e-4, "heads": 5e-4}, "warmup_steps": 1000, "backbone_warm_steps": 1000,
           "backbone_warm_scale": 0.1, "export": "aur_stage2_best.pt"},
    "III": {"seen_crops": 80_000, "lrs": {"backbone": 5e-5, "heads": 2.5e-4, "relation": 5e-4}, "warmup_steps": 200,      # warmup PROPOSED
            "backbone_warm_steps": 0, "backbone_warm_scale": 1.0, "export": "aur_stage3_best.pt"},
}
DEFAULTS = {"stage": "II", "model": {}, "microbatch": 4, "grad_accum": 1, "crops_per_volume": 2, "crop": CROP, "points": L.N_POINTS,
            "workers": 4, "seed": 0, "weight_decay": 0.05, "val_every": 500, "val_volumes": 32, "save_every": 500, "log_every": 20,
            "lambda_r": L.LAMBDA_R, "lambda_h": L.LAMBDA_H, "use_checkpoint": False, "max_steps": None, "lesion_share": 0.5,
            "init": "backbone", "init_backbone": None, "resume_stage2": None, "g1_report": None, "pilot": False, "stop_after": None}
HEAD_MODULES = ("entities", "events", "sequence", "masks")
COUNT_KEYS = ("r_samples", "r_correct", "r_instances")
LOSS_KEYS = ("a_mask", "a_presence", "u_presence", "u_mask", "s", "r_host", "r_hard")


def stage_defaults(stage):
    if stage not in STAGES:
        raise ValueError(f"stage must be one of {sorted(STAGES)}, got {stage!r}")
    return {k: (dict(v) if isinstance(v, dict) else v) for k, v in STAGES[stage].items()}


def split_rows(rows, val_patients):
    """Training rows = the train split minus the validation patients; validation rows = the train split's rows of the
    validation patients ({source: [patient, ...]}, the Stage I file, decision Q6)."""
    val = {(s, str(p)) for s, ps in val_patients.items() for p in ps}
    train_split = [r for r in rows if r.get("split", "train") == "train"]
    train = [r for r in train_split if (r["source"], str(r["patient"])) not in val]
    held = [r for r in train_split if (r["source"], str(r["patient"])) in val]
    if val and not held:
        raise ValueError("no training row belongs to a validation patient; the validation file does not match the samples")
    return train, held


def stage_two_frozen(model):
    """What Stage II does not train: the relation head, the Stage I mask token (unused after Stage I) and the coarse
    mask embedding (read only by the relation head's geometry in Stage III); frozen, they are neither touched by the
    zero gradient nor shrunk by the weight decay (review of 2026-10-09, item 8)."""
    return list(model.relation.parameters()) + [model.backbone.mask_token] + list(model.masks.embed_coarse.parameters())


def param_groups(model, stage, lrs):
    """Parameter groups with their rates; Stage II freezes stage_two_frozen (no group, no gradient)."""
    if stage == "II":
        frozen = {id(p) for p in stage_two_frozen(model)}
        for p in model.parameters():
            p.requires_grad_(id(p) not in frozen)
        heads = [p for name in HEAD_MODULES for p in getattr(model, name).parameters() if id(p) not in frozen]
        return [{"name": "backbone", "params": [p for p in model.backbone.parameters() if id(p) not in frozen], "lr": lrs["backbone"]},
                {"name": "heads", "params": heads, "lr": lrs["heads"]}]
    for p in model.parameters():
        p.requires_grad_(True)
    heads = [p for name in HEAD_MODULES for p in getattr(model, name).parameters()]
    return [{"name": "backbone", "params": list(model.backbone.parameters()), "lr": lrs["backbone"]},
            {"name": "heads", "params": heads, "lr": lrs["heads"]},
            {"name": "relation", "params": list(model.relation.parameters()), "lr": lrs["relation"]}]


def group_lr_lambda(name, stage, warmup, total, backbone_warm_steps, backbone_warm_scale):
    """The schedule factor of one group: warmup then cosine, and in Stage II the backbone scaled down for its first
    backbone_warm_steps steps (decision Q12)."""
    def f(step):
        base = lr_lambda(step, warmup, total)
        if stage == "II" and name == "backbone" and step < backbone_warm_steps:
            return base * backbone_warm_scale
        return base
    return f


def resolve_init(cfg):
    """The Stage I export to initialise from, or None for the random-initialisation control. The main line needs a
    passing G1 report (`g1_report.json` of scripts/aur_ssl_eval.py); a pilot run may skip it."""
    init = cfg.get("init", "backbone")
    if init == "random":
        if cfg.get("init_backbone"):
            raise ValueError("init 'random' (the C0 control) conflicts with --init-backbone")
        return None
    if init != "backbone":
        raise ValueError(f"init must be 'backbone' or 'random', got {init!r}")
    path = cfg.get("init_backbone")
    if not path:
        raise ValueError("init 'backbone' needs --init-backbone <ssl_stage1_best.pt>")
    if not cfg.get("pilot", False):
        report = cfg.get("g1_report")
        if not report:
            raise ValueError("the main line needs --g1-report <g1_report.json> with a passing G1 verdict (or --pilot for a smoke run)")
        verdict = json.loads(Path(report).read_text())
        if not verdict.get("g1", {}).get("pass", False):
            raise ValueError(f"G1 did not pass in {report}; Stage II of the main line is refused")
    return path


def init_backbone(model, path):
    """Strict load of the Stage I export into the model's backbone; returns its metadata."""
    return CK.load_backbone(path, model.backbone)


def export_model(path, model, meta):
    """The whole model (backbone, heads, relation) under a fresh name, with a JSON sidecar."""
    path = CK._fresh(path)
    torch.save({"model_state_dict": {k: v.detach().cpu() for k, v in model.state_dict().items()}, "meta": meta}, str(path))
    path.with_suffix(".json").write_text(json.dumps(meta, indent=1, default=str))
    return str(path)


def load_model(path, model, map_location="cpu"):
    """Strict load of a Stage II / III export into `model`; returns its metadata."""
    ck = torch.load(str(path), map_location=map_location, weights_only=False)
    model.load_state_dict(ck["model_state_dict"], strict=True)
    return ck["meta"]


def init_from_stage2(model, path):
    meta = load_model(path, model)
    if meta.get("stage") != "II":
        raise ValueError(f"{path} is not a Stage II export (stage {meta.get('stage')!r})")
    return meta


def to_device(batch, device):
    out = {}
    for k, v in batch.items():
        if torch.is_tensor(v):
            out[k] = v.to(device, non_blocking=True)
        elif isinstance(v, list) and v and torch.is_tensor(v[0]):
            out[k] = [t.to(device, non_blocking=True) for t in v]
        else:
            out[k] = v
    return out


def _forward_losses(module, model, batch, points, stage, lambda_r, lambda_h):
    """One micro-batch: (total loss, logged values (floats and the R counts), matches, extras for validation)."""
    device = points.device
    out = model(batch["image"], batch["valid"], batch["coords"], batch["local"])
    em, um = module.entity_masks(out, points).float(), module.event_masks(out, points).float()
    entity_pts = L.gather(batch["entity"], points)
    a_ignore_pts = L.gather(batch["a_ignore"].long(), points).bool()
    inst_pts = L.gather(batch["instance"], points)
    weight_pts = L.gather(batch["point_weight"], points)
    parts = L.entity_loss(em, out["entity_presence"].float(), entity_pts, a_ignore_pts, batch["entity_present"], batch["a_supervised"])
    targets = event_targets_at_points(inst_pts, batch["n_instances"])
    with torch.autocast(device_type=device.type, enabled=False):                        # the Hungarian cost in full precision
        u, matches = L.event_loss(out["event_presence"].float(), um, targets, weight_pts, batch["u_supervised"])
    parts.update(u)
    parts.update(L.seq_loss(out["seq_logits"].float(), batch["seq"]))
    r_samples, r_correct, r_total = 0, 0, 0
    if stage == "III":
        acc = {"r_host": em.new_zeros(()), "r_hard": em.new_zeros(())}
        for b, m in enumerate(matches):
            if m is None or not bool(batch["r_supervised"][b]) or m[0].numel() == 0:
                continue
            qi, ti = m
            logits = module.bind(out, b, qi).float()
            host, neg = batch["host"][b].to(device)[ti], batch["negatives"][b].to(device)[ti]
            part = L.relation_loss(logits, host, neg)
            acc = {k: acc[k] + part[k] for k in acc}
            r_samples += 1
            r_correct += int((logits.argmax(1) == host).sum())
            r_total += int(host.numel())
        parts.update({k: v / max(r_samples, 1) for k, v in acc.items()})
    total, logged = L.total(parts, lambda_r, lambda_h)
    touch = sum(((p * 0.0).sum() for p in module.parameters() if p.requires_grad), em.new_zeros(()))
    total = total + touch
    logged.setdefault("r_host", 0.0)
    logged.setdefault("r_hard", 0.0)
    logged.update({"r_samples": r_samples, "r_correct": r_correct, "r_instances": r_total})
    extras = {"out": out, "em": em, "entity_pts": entity_pts, "a_ignore_pts": a_ignore_pts, "matches": matches}
    return total, logged, matches, extras


def compute_losses(module, model, batch, points, stage, lambda_r=L.LAMBDA_R, lambda_h=L.LAMBDA_H):
    """module: the AnatoBindBrain (for masks and bind); model: the callable (the same module or its DDP wrapper);
    batch: a collated batch on the points' device; points (B, P) flat indices. Returns (total, logged, matches)."""
    total, logged, matches, _ = _forward_losses(module, model, batch, points, stage, lambda_r, lambda_h)
    return total, logged, matches


def entity_point_dice(em, entity_pts, a_ignore_pts, present, a_supervised=None):
    """Mean hard Dice at the points over the (sample, entity) pairs present in the crop, lesion points left out and
    samples without A supervision left out."""
    K = em.shape[1]
    keep = ~a_ignore_pts
    pred = (em > 0) & keep[:, None, :]
    target = (entity_pts[:, None, :] == torch.arange(1, K + 1, device=em.device)[None, :, None]) & keep[:, None, :]
    inter = (pred & target).sum(-1).float()
    dice = (2 * inter) / (pred.sum(-1) + target.sum(-1)).float().clamp(min=1.0)
    w = present.float()
    if a_supervised is not None:
        w = w * a_supervised.to(w.device).float()[:, None]
    return float((dice * w).sum() / w.sum().clamp(min=1.0))


@torch.no_grad()
def validate(module, model, batches, device, stage, cfg):
    """Validation losses (mean over the batches), entity point Dice, the counts behind the share of lesion instances
    matched to a present query and the host accuracy (Stage III), and the ratios where defined (None otherwise)."""
    model.eval()
    sums, n, dice = {}, 0, []
    counts = {"u_matched_n": 0, "u_instances": 0, "r_samples": 0, "r_correct": 0, "r_instances": 0}
    for i, b in enumerate(batches):
        b = to_device(b, device)
        points = L.sample_points(b["valid"], cfg["points"], torch.Generator(device=device).manual_seed(cfg["seed"] + 99 + i), instance=b["instance"])
        with torch.autocast(device_type="cuda" if device.type == "cuda" else "cpu", dtype=torch.bfloat16, enabled=device.type == "cuda"):
            total, logged, matches, ex = _forward_losses(module, model, b, points, stage, cfg["lambda_r"], cfg["lambda_h"])
        sums["loss"] = sums.get("loss", 0.0) + float(total)
        for k in LOSS_KEYS:
            sums[k] = sums.get(k, 0.0) + float(logged.get(k, 0.0))
        dice.append(entity_point_dice(ex["em"], ex["entity_pts"], ex["a_ignore_pts"], b["entity_present"], b["a_supervised"]))
        presence = ex["out"]["event_presence"].float().sigmoid()
        for s, m in enumerate(matches):
            if m is None:
                continue
            qi, _ = m
            counts["u_matched_n"] += int((presence[s, qi] > 0.5).sum())
            counts["u_instances"] += int(b["n_instances"][s])
        for k in COUNT_KEYS:
            counts[k] += int(logged[k])
        n += 1
    model.train()
    if n == 0:
        return {}
    res = {k: v / n for k, v in sums.items()}
    res["a_dice"] = float(np.mean(dice))
    res.update(counts)
    res["u_matched"] = counts["u_matched_n"] / counts["u_instances"] if counts["u_instances"] else None
    res["r_acc"] = counts["r_correct"] / counts["r_instances"] if counts["r_instances"] else None
    res["val_crops"] = sum(int(b["image"].shape[0]) for b in batches)
    return res


def validation_batches(rows, cfg):
    """Fixed validation crops: val_volumes volumes (seeded), crops_per_volume crops each, no augmentation."""
    if not rows:
        return []
    rng = np.random.default_rng(cfg["seed"] + 7)
    pick = [rows[i] for i in rng.choice(len(rows), size=min(cfg["val_volumes"], len(rows)), replace=False)]
    ds = AURDataset(pick, cfg["crop"], cfg["crops_per_volume"], do_augment=False, lesion_share=cfg["lesion_share"], seed=cfg["seed"] + 7)
    crops = [c for i in range(len(ds)) for c in ds[i]]
    step = max(1, cfg["microbatch"])
    return [collate([crops[i:i + step]]) for i in range(0, len(crops), step)]


def rank_rows(rows, seed, epoch, rank, world):
    """This rank's share of a seeded permutation of the rows for the epoch (disjoint across ranks)."""
    order = np.random.default_rng([seed, epoch]).permutation(len(rows))
    return [rows[i] for i in order[rank::world]]


def make_loader(rows, cfg, rank, world, epoch, device_count, skip_batches=0):
    """This rank's rows for the epoch, crops_per_volume crops each, micro-batches of microbatch crops. skip_batches
    (a resumed run) drops the rows of the first micro-batches instead of loading them; None when nothing is left."""
    if cfg["microbatch"] % cfg["crops_per_volume"]:
        raise ValueError(f"microbatch {cfg['microbatch']} must be a multiple of crops_per_volume {cfg['crops_per_volume']}")
    volumes_per_batch = cfg["microbatch"] // cfg["crops_per_volume"]
    mine = rank_rows(rows, cfg["seed"], epoch, rank, world)
    start = int(skip_batches) * volumes_per_batch
    if skip_batches and len(mine) - start < volumes_per_batch:
        return None
    ds = AURDataset(mine[start:], cfg["crop"], cfg["crops_per_volume"], do_augment=True, lesion_share=cfg["lesion_share"],
                    seed=cfg["seed"] + 1000 * rank, index_offset=start)
    ds.set_epoch(epoch)
    loader = DataLoader(ds, batch_size=volumes_per_batch, shuffle=False, num_workers=cfg["workers"], collate_fn=collate, pin_memory=device_count > 0,
                        drop_last=True, persistent_workers=cfg["workers"] > 0, prefetch_factor=2 if cfg["workers"] > 0 else None)
    if len(loader) == 0:
        raise ValueError(f"rank {rank} holds {len(mine)} rows, fewer than the {volumes_per_batch} volumes of one micro-batch")
    return loader


def _clean(rec):
    """JSON-safe: an undefined float becomes None."""
    return {k: (None if isinstance(v, float) and not math.isfinite(v) else v) for k, v in rec.items()}


def run(cfg, samples, val_patients, out_dir, resume=None):
    stage = cfg.get("stage", DEFAULTS["stage"])
    cfg = {**DEFAULTS, **stage_defaults(stage), **cfg}
    cfg["stage"] = stage
    if cfg["microbatch"] % cfg["crops_per_volume"]:
        raise ValueError(f"microbatch {cfg['microbatch']} must be a multiple of crops_per_volume {cfg['crops_per_volume']}")
    if stage == "III":
        if not cfg.get("resume_stage2"):
            raise ValueError("stage III needs resume_stage2 (the Stage II export aur_stage2_best.pt)")
        if cfg.get("init_backbone") or cfg.get("g1_report"):
            raise ValueError("stage III takes resume_stage2 only; init_backbone and g1_report are Stage II options")
    init_path = resolve_init(cfg) if stage == "II" else None
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
    val = json.loads(Path(val_patients).read_text())
    train_rows, val_rows = split_rows(rows, val)
    torch.manual_seed(cfg["seed"])
    model = AnatoBindBrain(**cfg["model"], use_checkpoint=cfg["use_checkpoint"]).to(device)
    init_meta = None
    if stage == "II" and init_path is not None:
        init_meta = init_backbone(model, init_path)
    if stage == "III":
        init_meta = init_from_stage2(model, cfg["resume_stage2"])
    groups = param_groups(model, stage, cfg["lrs"])
    global_batch = cfg["microbatch"] * world * cfg["grad_accum"]
    total_steps = math.ceil(cfg["seen_crops"] / global_batch)
    if cfg["max_steps"] is not None:
        total_steps = min(total_steps, int(cfg["max_steps"]))
    opt = torch.optim.AdamW(groups, weight_decay=cfg["weight_decay"], betas=(0.9, 0.95))
    sched = torch.optim.lr_scheduler.LambdaLR(opt, [group_lr_lambda(g["name"], stage, cfg["warmup_steps"], total_steps, cfg["backbone_warm_steps"],
                                                                     cfg["backbone_warm_scale"]) for g in groups])
    step, seen, epoch, epoch_step = 0, 0, 0, 0
    best = {"loss": float("inf"), "step": None, "path": None}
    schedule = {"total_steps": total_steps, "global_batch": global_batch, "lrs": cfg["lrs"], "warmup_steps": cfg["warmup_steps"],
                "backbone_warm_steps": cfg["backbone_warm_steps"], "seen_crops": cfg["seen_crops"]}
    if resume is not None:
        meta, _ = CK.load_resume(resume, model, opt, sched, map_location=device)
        step, seen = int(meta["step"]), int(meta["seen_crops"])
        epoch, epoch_step = int(meta.get("epoch", 0)), int(meta.get("epoch_step", 0))
        saved = meta.get("schedule")
        if saved is not None and saved != schedule:
            raise ValueError(f"resume schedule mismatch: the checkpoint was made with {saved}, this run has {schedule}")
        if (out_dir / "best.json").exists():
            best = json.loads((out_dir / "best.json").read_text())
    ddp = DistributedDataParallel(model, device_ids=[local_rank] if device.type == "cuda" else None) if world > 1 else model
    val_batches = validation_batches(val_rows, cfg)
    log_path = out_dir / f"log_rank{rank}.jsonl"
    times, waits = [], []
    extra_meta = {"stage": stage, "init": cfg["init"] if stage == "II" else "stage2", "init_backbone": str(init_path) if init_path else None,
                  "resume_stage2": str(cfg["resume_stage2"]) if cfg["resume_stage2"] else None, "val_patients": str(val_patients),
                  "val_patients_sha256": CK.file_sha256(val_patients),
                  "init_meta": {k: init_meta.get(k) for k in ("stage", "step", "seen_crops", "code_sha")} if init_meta else None}
    if rank == 0:
        info = {"config": cfg, "samples": str(samples), "manifest_sha256": CK.file_sha256(samples), "code_sha": CK.code_sha(), "world": world,
                "global_batch": global_batch, "total_steps": total_steps, "train_rows": len(train_rows), "val_rows": len(val_rows),
                "train_patients": len({(r["source"], r["patient"]) for r in train_rows}), "params": model.num_parameters(),
                "trainable_params": sum(p.numel() for p in model.parameters() if p.requires_grad), "lrs": cfg["lrs"],
                "init_backbone": str(init_path) if init_path else None, "resume_stage2": str(cfg["resume_stage2"]) if cfg["resume_stage2"] else None,
                "resumed_from": str(resume) if resume else None, "host": platform.node(),
                "gpus": [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())] if device.type == "cuda" else [],
                "started": time.strftime("%Y-%m-%d %H:%M:%S"), **extra_meta}
        name = "run_config.json" if resume is None else f"run_config_resume_step{step}.json"       # a resumed invocation never rewrites the first
        (out_dir / name).write_text(json.dumps(info, indent=1, default=str))
    model.train()
    trainable = [p for p in model.parameters() if p.requires_grad]
    done, stopped, start_step = step >= total_steps, False, step
    skip = epoch_step * cfg["grad_accum"] if resume is not None else 0
    t_step = time.time()
    while not done:
        loader = make_loader(train_rows, cfg, rank, world, epoch, torch.cuda.device_count(), skip_batches=skip)   # a resumed run continues its epoch
        if loader is None:                                        # it stopped at the epoch's end
            epoch, epoch_step, skip = epoch + 1, 0, 0
            continue
        it = iter(loader)
        skip = 0
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
                b = to_device(b, device)                                   # the loss points are drawn on the device (2026-10-10:
                g = torch.Generator(device=device).manual_seed(int(np.random.SeedSequence([cfg["seed"], step, rank, j]).generate_state(1)[0]))
                points = L.sample_points(b["valid"], cfg["points"], g, instance=b["instance"])   # on the CPU they bound a 12-crop step)
                with torch.autocast(device_type="cuda" if device.type == "cuda" else "cpu", dtype=torch.bfloat16, enabled=device.type == "cuda"):
                    total, logged, _ = compute_losses(model, ddp, b, points, stage, cfg["lambda_r"], cfg["lambda_h"])
                    (total / cfg["grad_accum"]).backward()
                stats["loss"] = stats.get("loss", 0.0) + float(total) / cfg["grad_accum"]
                for k, v in logged.items():
                    if k in COUNT_KEYS:
                        stats[k] = stats.get(k, 0) + int(v)
                    else:
                        stats[k] = stats.get(k, 0.0) + float(v) / cfg["grad_accum"]
            if not all(math.isfinite(v) for k, v in stats.items() if k not in COUNT_KEYS):
                raise FloatingPointError(f"non-finite loss at step {step}: {stats}")
            torch.nn.utils.clip_grad_norm_(trainable, 1.0)
            opt.step()
            sched.step()
            step += 1
            seen += global_batch
            epoch_step += 1
            if device.type == "cuda":
                torch.cuda.synchronize()
            times.append(time.time() - t_step)
            t_step = time.time()
            if rank == 0 and (step % cfg["log_every"] == 0 or step == 1):
                rec = {"step": step, "seen_crops": seen, "epoch": epoch, "lr": {g["name"]: lr for g, lr in zip(groups, sched.get_last_lr())},
                       "step_s": times[-1], "data_wait_s": waits[-1], **stats}
                if stage == "II":
                    rec = {k: v for k, v in rec.items() if not k.startswith("r_")}
                if device.type == "cuda":
                    rec["peak_alloc_gib"] = torch.cuda.max_memory_allocated(device) / 2 ** 30
                    rec["peak_reserved_gib"] = torch.cuda.max_memory_reserved(device) / 2 ** 30
                rec = _clean(rec)
                with open(log_path, "a") as f:
                    f.write(json.dumps(rec) + "\n")
                print(json.dumps({k: (round(v, 4) if isinstance(v, float) else v) for k, v in rec.items()}), flush=True)
            done = step >= total_steps or seen >= cfg["seen_crops"]
            stopped = cfg["stop_after"] is not None and step - start_step >= int(cfg["stop_after"]) and not done
            if (step % cfg["val_every"] == 0 or done or stopped) and val_batches:
                res = validate(model, ddp, val_batches, device, stage, cfg)
                if world > 1:
                    t = torch.tensor([res["loss"]], device=device)
                    dist.all_reduce(t)
                    res["loss_all_ranks_mean"] = float(t) / world
                score = res.get("loss_all_ranks_mean", res["loss"])
                if rank == 0:
                    res.update({"step": step, "seen_crops": seen})
                    res = _clean(res)
                    with open(out_dir / "val.jsonl", "a") as f:
                        f.write(json.dumps(res) + "\n")
                    print("VAL", json.dumps({k: (round(v, 5) if isinstance(v, float) else v) for k, v in res.items()}), flush=True)
                    if score < best["loss"]:
                        path = out_dir / cfg["export"].replace("_best.pt", f"_step{step}.pt")
                        if not path.exists():
                            meta = CK.metadata(cfg, samples, step, seen, cfg["seed"], extra={**extra_meta, "validation": res})
                            export_model(path, model, meta)
                        best = {"loss": score, "step": step, "path": str(path)}
                        (out_dir / "best.json").write_text(json.dumps(best, indent=1))
            if rank == 0 and (step % cfg["save_every"] == 0 or done or stopped):
                meta = CK.metadata(cfg, samples, step, seen, cfg["seed"], extra={**extra_meta, "epoch": epoch, "epoch_step": epoch_step, "schedule": schedule})
                CK.save_resume(out_dir / f"resume_step{step}.pt", model, opt, sched, meta)
            if stopped:
                break
        if stopped:
            break
        if not done:
            epoch, epoch_step = epoch + 1, 0
    summary = None
    if rank == 0:
        arr = np.array(times[1:]) if len(times) > 1 else np.array(times)
        summary = {"stage": stage, "steps": step, "seen_crops": seen, "world": world, "global_batch": global_batch,
                   "step_s_mean": float(arr.mean()) if arr.size else None, "step_s_p95": float(np.percentile(arr, 95)) if arr.size else None,
                   "data_wait_s_mean": float(np.mean(waits[1:])) if len(waits) > 1 else None,
                   "crops_per_s": float(global_batch / arr.mean()) if arr.size else None, "best": best, "stopped": stopped,
                   "total_steps": total_steps, "finished": time.strftime("%Y-%m-%d %H:%M:%S")}
        if device.type == "cuda":
            summary["peak_alloc_gib"] = torch.cuda.max_memory_allocated(device) / 2 ** 30
            summary["peak_reserved_gib"] = torch.cuda.max_memory_reserved(device) / 2 ** 30
            summary["gpu"] = torch.cuda.get_device_name(device)
        if best["path"] and not stopped:
            final = out_dir / cfg["export"]                            # the best validation export under the fixed name
            sidecar = final.with_suffix(".json")
            if not final.exists():
                torch.save(torch.load(best["path"], map_location="cpu", weights_only=False), str(final))
                summary["final_export"] = str(final)
            else:                                                     # a resumed run: the fixed file is never overwritten
                prior = json.loads(sidecar.read_text()) if sidecar.exists() else {}
                summary["final_export"] = str(final) if prior.get("step") == best["step"] else best["path"]
            sidecar.write_text(json.dumps({**best, "final_export": summary["final_export"]}, indent=1))
        name = f"summary_stop_step{step}.json" if stopped else "summary.json"                       # summary.json only when the run is done
        (out_dir / name).write_text(json.dumps(summary, indent=1, default=str))
        print("SUMMARY", json.dumps(summary, default=str), flush=True)
    if world > 1:
        dist.barrier()
        dist.destroy_process_group()
    return summary
