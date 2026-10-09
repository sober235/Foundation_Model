#!/usr/bin/env python
"""Stage I training (anatobind.aur.ssl.train; SSL-first plan T08; decisions Q11, Q19 of 2026-10-09).

    export PYTHONNOUSERSITE=1 PYTHONPATH=. NCCL_P2P_DISABLE=1 TORCH_NCCL_ASYNC_ERROR_HANDLING=1 OMP_NUM_THREADS=8
    CUDA_VISIBLE_DEVICES=<four idle GPUs> torchrun --standalone --nproc_per_node=4 scripts/aur_ssl_train.py \
        --samples <ssl_manifest>/samples_ssl.json --seen-crops 320000 --microbatch 2 --grad-accum 2 \
        --mask-ratio 0.60 --contrast-weight 0.10 --out <NEW_STAGE_I_DIR>

The budget is in source crops seen (every optimizer step sees microbatch x GPUs x grad-accum of them). The run
directory must not exist; --resume <resume_step*.pt> continues a run in its directory. Outputs: run_config.json,
log_rank0.jsonl, val.jsonl, best.json, ssl_stage1_step*.pt (backbone exports at the validation improvements),
resume_step*.pt, summary.json and ssl_stage1_best.pt (a copy of the best export)."""
import argparse
import sys
from pathlib import Path

from anatobind.aur.ssl.train import DEFAULTS, run


def parser():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--samples", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--resume", type=Path, default=None)
    ap.add_argument("--seen-crops", type=int, default=DEFAULTS["seen_crops"])
    ap.add_argument("--microbatch", type=int, default=DEFAULTS["microbatch"])
    ap.add_argument("--grad-accum", type=int, default=DEFAULTS["grad_accum"])
    ap.add_argument("--crops-per-volume", type=int, default=DEFAULTS["crops_per_volume"])
    ap.add_argument("--lr", type=float, default=DEFAULTS["lr"])
    ap.add_argument("--weight-decay", type=float, default=DEFAULTS["weight_decay"])
    ap.add_argument("--warmup-steps", type=int, default=DEFAULTS["warmup_steps"])
    ap.add_argument("--mask-ratio", type=float, default=DEFAULTS["mask_ratio"])
    ap.add_argument("--block-mm", type=float, nargs=2, default=list(DEFAULTS["block_mm"]))
    ap.add_argument("--contrast-weight", type=float, default=DEFAULTS["contrast_weight"])
    ap.add_argument("--temperature", type=float, default=DEFAULTS["temperature"])
    ap.add_argument("--crop", type=int, nargs=3, default=list(DEFAULTS["crop"]))
    ap.add_argument("--workers", type=int, default=DEFAULTS["workers"])
    ap.add_argument("--seed", type=int, default=DEFAULTS["seed"])
    ap.add_argument("--val-every", type=int, default=DEFAULTS["val_every"])
    ap.add_argument("--val-volumes", type=int, default=DEFAULTS["val_volumes"])
    ap.add_argument("--save-every", type=int, default=DEFAULTS["save_every"])
    ap.add_argument("--log-every", type=int, default=DEFAULTS["log_every"])
    ap.add_argument("--hcp-cap", type=float, default=DEFAULTS["hcp_cap"])
    ap.add_argument("--use-checkpoint", action="store_true")
    ap.add_argument("--no-rotate", action="store_true")
    ap.add_argument("--max-steps", type=int, default=None)
    ap.add_argument("--embed", type=int, default=DEFAULTS["embed"])
    return ap


def config(a, probe=False):
    return {"seen_crops": a.seen_crops, "microbatch": a.microbatch, "grad_accum": a.grad_accum, "crops_per_volume": a.crops_per_volume,
            "lr": a.lr, "weight_decay": a.weight_decay, "warmup_steps": a.warmup_steps, "mask_ratio": a.mask_ratio, "block_mm": tuple(a.block_mm),
            "contrast_weight": a.contrast_weight, "temperature": a.temperature, "crop": tuple(a.crop), "workers": a.workers, "seed": a.seed,
            "val_every": a.val_every, "val_volumes": a.val_volumes, "save_every": a.save_every, "log_every": a.log_every, "hcp_cap": a.hcp_cap,
            "use_checkpoint": a.use_checkpoint, "no_rotate": a.no_rotate, "max_steps": a.max_steps, "probe": probe, "embed": a.embed}


def main(argv=None):
    a = parser().parse_args(argv)
    if a.workers > 12:
        raise SystemExit("--workers above 12 per process is not allowed on this host (48 threads in all)")
    run(config(a), a.samples, a.out, resume=a.resume)
    return 0


if __name__ == "__main__":
    sys.exit(main())
