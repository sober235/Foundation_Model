#!/usr/bin/env python
"""Stage I memory / speed probe (SSL-first plan T07): the trainer for a few timed steps, no validation, no export.

    export PYTHONNOUSERSITE=1 PYTHONPATH=. NCCL_P2P_DISABLE=1 TORCH_NCCL_ASYNC_ERROR_HANDLING=1 OMP_NUM_THREADS=8
    # single card
    CUDA_VISIBLE_DEVICES=<idle GPU> python scripts/aur_ssl_probe.py --samples <samples_ssl.json> --microbatch 2 --steps 20 --out <NEW_P0_DIR>/single_b2
    # four cards
    CUDA_VISIBLE_DEVICES=<four idle GPUs> timeout 900 torchrun --standalone --nproc_per_node=4 scripts/aur_ssl_probe.py \
        --samples <samples_ssl.json> --microbatch 2 --grad-accum 2 --steps 100 --out <NEW_P0_DIR>/ddp_b2x2

Writes probe_config.json, log_rank0.jsonl and probe.json (mean and p95 step time, data wait, crops per second, peak
allocated and reserved memory, GPU name) into the new directory. Real volumes from the manifest, the real loader."""
import argparse
import sys
from pathlib import Path

from anatobind.aur.ssl.train import DEFAULTS, run


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--samples", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--steps", type=int, default=20)
    ap.add_argument("--microbatch", type=int, default=DEFAULTS["microbatch"])
    ap.add_argument("--grad-accum", type=int, default=1)
    ap.add_argument("--crops-per-volume", type=int, default=DEFAULTS["crops_per_volume"])
    ap.add_argument("--crop", type=int, nargs=3, default=list(DEFAULTS["crop"]))
    ap.add_argument("--workers", type=int, default=DEFAULTS["workers"])
    ap.add_argument("--mask-ratio", type=float, default=DEFAULTS["mask_ratio"])
    ap.add_argument("--use-checkpoint", action="store_true")
    ap.add_argument("--log-every", type=int, default=5)
    ap.add_argument("--embed", type=int, default=DEFAULTS["embed"])
    a = ap.parse_args(argv)
    cfg = {"seen_crops": 10 ** 9, "max_steps": a.steps, "probe": True, "microbatch": a.microbatch, "grad_accum": a.grad_accum,
           "crops_per_volume": a.crops_per_volume, "crop": tuple(a.crop), "workers": a.workers, "mask_ratio": a.mask_ratio,
           "use_checkpoint": a.use_checkpoint, "log_every": a.log_every, "warmup_steps": 5, "embed": a.embed}
    run(cfg, a.samples, a.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
