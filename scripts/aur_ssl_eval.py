#!/usr/bin/env python
"""Gate G1: the frozen-representation probes, Stage I backbone against a random initialisation (anatobind.aur.ssl.eval).

    export PYTHONNOUSERSITE=1 PYTHONPATH=.
    CUDA_VISIBLE_DEVICES=<idle GPU> python scripts/aur_ssl_eval.py \
        --checkpoint <STAGE_I_DIR>/ssl_stage1_best.pt \
        --samples /data2/congcong/data/FM_data/derived/aur/samples_1mm_v1.json \
        --val-patients /data2/congcong/data/FM_data/derived/aur/ssl_manifest_v1/val_patients.json \
        --out <NEW_G1_DIR>

Writes g1_report.json: the host readout (13-host macro Dice, per host, 3 probe seeds, patient bootstrap of the
difference SSL - random), the lesion separability (AUC, by size bin) and the G1 verdict against the PROPOSED thresholds
(decision Q8 of 2026-10-09). NOT_EVIDENCE: pseudo-labels."""
import argparse
import json
import sys
from pathlib import Path

import torch

from anatobind.aur.crops import CROP
from anatobind.aur.ssl.eval import run


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--checkpoint", type=Path, required=True)
    ap.add_argument("--samples", type=Path, required=True)
    ap.add_argument("--val-patients", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--crop", type=int, nargs=3, default=list(CROP))
    ap.add_argument("--calibration", type=int, default=96)
    ap.add_argument("--validation", type=int, default=48)
    ap.add_argument("--lesion", type=int, default=48)
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--host-only", action="store_true", help="the early probe of a running Stage I: host readout only, no lesion arm, no verdict")
    a = ap.parse_args(argv)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    rows = json.loads(a.samples.read_text())
    val = json.loads(a.val_patients.read_text())
    report = run(rows, val, a.checkpoint, a.out, device, tuple(a.crop), a.calibration, a.validation, a.lesion, tuple(a.seeds), lesion=not a.host_only)
    print(json.dumps(report["g1"], indent=1, default=str))
    print("host macro Dice: ssl %.4f random %.4f" % (report["arms"]["ssl"]["host"]["macro_dice_mean"], report["arms"]["random"]["host"]["macro_dice_mean"]))
    if a.host_only:
        return 0
    return 0 if report["g1"]["pass"] else 1


if __name__ == "__main__":
    sys.exit(main())
