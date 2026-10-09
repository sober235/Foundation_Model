#!/usr/bin/env python
"""Stage II / III training of AnatoBind-Brain (SSL-first plan T10, T11; spec §6).

    export PYTHONNOUSERSITE=1 PYTHONPATH=. NCCL_P2P_DISABLE=1 TORCH_NCCL_ASYNC_ERROR_HANDLING=1 OMP_NUM_THREADS=8
    # Stage II from the Stage I backbone (the main line; needs the passing G1 report)
    CUDA_VISIBLE_DEVICES=<idle GPUs> torchrun --standalone --nproc_per_node=<n> scripts/aur_train.py --stage II \
        --init-backbone <STAGE_I_DIR>/ssl_stage1_best.pt --g1-report <G1_DIR>/g1_report.json \
        --samples <samples_1mm_v1.json> --val-patients <ssl_manifest_v1/val_patients.json> --out <NEW_STAGE_II_DIR>
    # the C0 control: the same stage from a random backbone
    ... scripts/aur_train.py --stage II --init random --samples ... --val-patients ... --out <NEW_C0_DIR>
    # Stage III from the whole Stage II export
    ... scripts/aur_train.py --stage III --resume-stage2 <STAGE_II_DIR>/aur_stage2_best.pt --samples ... --val-patients ... --out <NEW_STAGE_III_DIR>

Budgets, rates and warm-ups default to the stage's (anatobind/aur/train.py STAGES); --seen-crops, --lr-* override
them. --microbatch is the number of crops per rank per optimizer step (a multiple of --crops-per-volume)."""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from anatobind.aur.train import DEFAULTS, run, stage_defaults  # noqa: E402


def parser():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--stage", choices=("II", "III"), required=True)
    ap.add_argument("--samples", type=Path, required=True, help="samples_1mm_v1.json")
    ap.add_argument("--val-patients", type=Path, required=True, help="val_patients.json of the Stage I manifest (decision Q6)")
    ap.add_argument("--out", type=Path, required=True, help="a new directory")
    ap.add_argument("--init", choices=("backbone", "random"), default="backbone", help="Stage II: Stage I backbone (main line) or random (C0)")
    ap.add_argument("--init-backbone", type=str, default=None, help="Stage II: ssl_stage1_best.pt")
    ap.add_argument("--g1-report", type=str, default=None, help="Stage II main line: g1_report.json with a passing verdict")
    ap.add_argument("--pilot", action="store_true", help="smoke run: no G1 report needed")
    ap.add_argument("--resume-stage2", type=str, default=None, help="Stage III: aur_stage2_best.pt")
    ap.add_argument("--resume", type=Path, default=None, help="a resume_step*.pt of this very run")
    ap.add_argument("--seen-crops", type=int, default=None)
    ap.add_argument("--microbatch", type=int, default=DEFAULTS["microbatch"])
    ap.add_argument("--grad-accum", type=int, default=DEFAULTS["grad_accum"])
    ap.add_argument("--crops-per-volume", type=int, default=DEFAULTS["crops_per_volume"])
    ap.add_argument("--crop", type=int, nargs=3, default=list(DEFAULTS["crop"]))
    ap.add_argument("--points", type=int, default=DEFAULTS["points"])
    ap.add_argument("--workers", type=int, default=DEFAULTS["workers"])
    ap.add_argument("--seed", type=int, default=DEFAULTS["seed"])
    ap.add_argument("--lr-backbone", type=float, default=None)
    ap.add_argument("--lr-heads", type=float, default=None)
    ap.add_argument("--lr-relation", type=float, default=None)
    ap.add_argument("--warmup-steps", type=int, default=None)
    ap.add_argument("--weight-decay", type=float, default=DEFAULTS["weight_decay"])
    ap.add_argument("--val-every", type=int, default=DEFAULTS["val_every"])
    ap.add_argument("--val-volumes", type=int, default=DEFAULTS["val_volumes"])
    ap.add_argument("--save-every", type=int, default=DEFAULTS["save_every"])
    ap.add_argument("--log-every", type=int, default=DEFAULTS["log_every"])
    ap.add_argument("--max-steps", type=int, default=None)
    ap.add_argument("--use-checkpoint", action="store_true")
    return ap


def config_from_args(a):
    cfg = {"stage": a.stage, "init": a.init, "init_backbone": a.init_backbone, "g1_report": a.g1_report, "pilot": a.pilot,
           "resume_stage2": a.resume_stage2, "microbatch": a.microbatch, "grad_accum": a.grad_accum, "crops_per_volume": a.crops_per_volume,
           "crop": tuple(a.crop), "points": a.points, "workers": a.workers, "seed": a.seed, "weight_decay": a.weight_decay,
           "val_every": a.val_every, "val_volumes": a.val_volumes, "save_every": a.save_every, "log_every": a.log_every,
           "max_steps": a.max_steps, "use_checkpoint": a.use_checkpoint}
    stage = stage_defaults(a.stage)
    cfg["seen_crops"] = a.seen_crops if a.seen_crops is not None else stage["seen_crops"]
    cfg["warmup_steps"] = a.warmup_steps if a.warmup_steps is not None else stage["warmup_steps"]
    lrs = dict(stage["lrs"])
    for name in lrs:
        value = getattr(a, f"lr_{name}")
        if value is not None:
            lrs[name] = value
    cfg["lrs"] = lrs
    return cfg


def main(argv=None):
    a = parser().parse_args(argv)
    run(config_from_args(a), a.samples, a.val_patients, a.out, resume=a.resume)
    return 0


if __name__ == "__main__":
    sys.exit(main())
