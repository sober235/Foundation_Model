#!/usr/bin/env python
"""AnatoBind-Brain on the fastMRI FLAIR stacks, the S4 measure (anatobind.aur.fastmri; spec §7, report only).

    PYTHONNOUSERSITE=1 PYTHONPATH=. ~/anaconda3/envs/nvgen/bin/python scripts/aur_eval_fastmri.py \
        --checkpoint <STAGE_III_DIR>/aur_stage3_best.pt --out <NEW_RECORD_DIR> --gpu <idle card>

Reads the skull-stripped stacks of the S4 evaluation and SynthSeg's maps, the fastMRI+ lesion boxes of the Level R
registry; writes summary.json, per_stem.json and REPORT.md (AnatoBind next to the S4 student, from S4's verdict.json)
into a new directory. NOT_EVIDENCE."""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from anatobind.anatomy.labels import NAMES  # noqa: E402
from anatobind.aur import fastmri as FX  # noqa: E402
from anatobind.aur.crops import CROP  # noqa: E402
from anatobind.aur.infer import model_from_export  # noqa: E402
from anatobind.aur.ssl.checkpoint import code_sha, file_sha256  # noqa: E402
from anatobind.level_r.registry import load_registry  # noqa: E402

FM = Path("/data2/congcong/data/FM_data")
STACKS = FM / "derived/brain_anatomy/eval_20261004_0233/stripped"
SEGS = FM / "derived/synthseg/fastmri_brain/seg_native"
S4 = Path(__file__).resolve().parents[1] / "docs/verification/2026-10-02/brain_anatomy_flair/eval/verdict.json"


def _f(v):
    return "n/a" if v is None else f"{v:.4f}"


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--checkpoint", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True, help="a new record directory")
    ap.add_argument("--stacks", type=Path, default=STACKS)
    ap.add_argument("--segs", type=Path, default=SEGS)
    ap.add_argument("--s4-verdict", type=Path, default=S4)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--crop", type=int, nargs=3, default=None)
    ap.add_argument("--batch-size", type=int, default=2)
    dev = ap.add_mutually_exclusive_group(required=True)
    dev.add_argument("--gpu", type=int)
    dev.add_argument("--cpu", action="store_true")
    a = ap.parse_args(argv)
    if a.out.exists():
        raise FileExistsError(f"{a.out} exists")
    device = torch.device("cpu") if a.cpu else torch.device("cuda", a.gpu)
    model, meta = model_from_export(a.checkpoint, device)
    crop = tuple(a.crop) if a.crop else tuple(meta.get("config", {}).get("crop", CROP))
    rows = {}
    for r in load_registry():
        rows.setdefault(r["file"], []).append(r)
    stems = sorted(p.name[:-len("_0000.nii.gz")] for p in a.stacks.glob("*_0000.nii.gz"))[: a.limit]
    per_stem, t0 = {}, time.time()
    for i, stem in enumerate(stems):
        res = FX.evaluate_stack(model, a.stacks / f"{stem}_0000.nii.gz", a.segs / f"{stem}_seg.nii.gz", rows.get(stem, []), crop, device, a.batch_size)
        res.pop("pred_native")
        per_stem[stem] = res
        if (i + 1) % 25 == 0 or i + 1 == len(stems):
            print(f"[{i + 1}/{len(stems)}] {time.time() - t0:.0f} s", flush=True)
    summary = FX.summarize(per_stem)
    summary["run"] = {"checkpoint": str(a.checkpoint), "checkpoint_sha256": file_sha256(a.checkpoint), "code_sha": code_sha(), "crop": list(crop),
                      "stacks": str(a.stacks), "n_stacks": len(stems), "seconds": round(time.time() - t0, 1)}
    a.out.mkdir(parents=True)
    (a.out / "summary.json").write_text(json.dumps(summary, indent=1, default=str))
    (a.out / "per_stem.json").write_text(json.dumps(per_stem, indent=1, default=str))
    s4 = json.loads(a.s4_verdict.read_text()) if a.s4_verdict.exists() else None
    d, h = summary["fastmri_dice"], summary["host_agreement"]
    L = [f"# AnatoBind-Brain on the fastMRI FLAIR stacks ({summary['evidence']})", "",
         f"{summary['n_stacks']} stacks, the S4 measure (reliable slices; boxes whose slices are all reliable). The stacks are resampled to 1 mm, "
         "predicted, and the prediction is brought back by nearest neighbour.", "",
         "| measure | AnatoBind | S4 student |", "|---|---|---|",
         f"| mean host Dice (13 classes, reliable slices) | {_f(d['mean_host_dice'])} | {_f(s4['fastmri_dice']['mean_host_dice']) if s4 else 'n/a'} |",
         f"| box host agreement ({h['n_evaluated']} boxes) | {_f(h['rate'])} | {_f(s4['host_agreement']['rate']) if s4 else 'n/a'} |", "",
         "| class | AnatoBind | S4 student |", "|---|---|---|"]
    for name, v in d["per_class"].items():
        L.append(f"| {name} | {_f(v)} | {_f(s4['fastmri_dice']['per_class'].get(name)) if s4 else 'n/a'} |")
    L += ["", "The SynthSeg reference itself barely holds the deep structures on these slices (S4 record): the Dice of thalamus, basal ganglia, "
          "brainstem and cerebellum measures the reference as much as the model.", ""]
    (a.out / "REPORT.md").write_text("\n".join(L))
    print("\n".join(L))
    return 0


if __name__ == "__main__":
    sys.exit(main())
