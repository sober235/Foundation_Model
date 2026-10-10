#!/usr/bin/env python
"""Attribution diagnostics of a Stage I checkpoint (anatobind.aur.ssl.attrib; review of 2026-10-10): per backbone part,
the gradient RMS of the reconstruction vs the weighted contrastive term, and the masked reconstruction loss with the
decoder reading all levels / levels 2-4 / F1 only. Runs on the CPU by default (it never needs a card).

    CUDA_VISIBLE_DEVICES="" PYTHONNOUSERSITE=1 PYTHONPATH=. nice -n 19 ~/anaconda3/envs/nvgen/bin/python scripts/aur_ssl_attrib.py \
        --checkpoint <RUN>/resume_step2000.pt --samples <ssl_manifest_v1/samples_ssl.json> --out <RUN>/attrib_step2000.json

--checkpoint init uses a fresh model of the default configuration. The batch: --patients validation patients of the
Stage I manifest (seeded), --crops-per-volume crops each, the Stage I views (intensity augmentation, rotation)."""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from anatobind.aur.crops import CROP  # noqa: E402
from anatobind.aur.ssl.attrib import gradient_balance, reconstruction_ablation  # noqa: E402
from anatobind.aur.ssl.dataset import collate_ssl, load_image, make_ssl_crop  # noqa: E402
from anatobind.aur.ssl.model import DEFAULTS, StageOne  # noqa: E402


def model_from(checkpoint):
    if str(checkpoint) == "init":
        torch.manual_seed(0)
        return StageOne(), {"step": 0}
    ck = torch.load(str(checkpoint), map_location="cpu", weights_only=False)
    cfg = ck["meta"].get("config", {})
    model = StageOne(**{k: (tuple(v) if isinstance(v, list) else v) for k, v in cfg.items() if k in DEFAULTS})
    model.load_state_dict(ck["model"], strict=True)
    return model, ck["meta"]


def batch_from(samples, n_patients, crops_per_volume, crop, seed):
    rows = [r for r in json.loads(Path(samples).read_text()) if r.get("ssl_split") == "val"]
    order = np.random.default_rng(seed).permutation(len(rows))
    seen, items = set(), []
    for i in order:
        r = rows[i]
        key = (r["source"], r["patient"])
        if key in seen:
            continue
        seen.add(key)
        vol = load_image(r)
        items.append([make_ssl_crop(vol, np.random.default_rng([seed, len(items), j]), crop=crop) for j in range(crops_per_volume)])
        if len(items) == n_patients:
            break
    return collate_ssl(items)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--checkpoint", required=True, help="a Stage I resume checkpoint, or 'init'")
    ap.add_argument("--samples", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True, help="a new JSON file")
    ap.add_argument("--patients", type=int, default=6)
    ap.add_argument("--crops-per-volume", type=int, default=2)
    ap.add_argument("--crop", type=int, nargs=3, default=list(CROP))
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--threads", type=int, default=32)
    a = ap.parse_args(argv)
    if a.out.exists():
        raise FileExistsError(f"{a.out} exists")
    torch.set_num_threads(a.threads)
    model, meta = model_from(a.checkpoint)
    batch = batch_from(a.samples, a.patients, a.crops_per_volume, tuple(a.crop), a.seed)
    res = {"checkpoint": str(a.checkpoint), "step": meta.get("step"), "n_crops": int(batch["view1"].shape[0]),
           "gradient_balance": gradient_balance(model, batch, seed=a.seed), "reconstruction": reconstruction_ablation(model.eval(), batch, seed=a.seed),
           "evidence": "diagnostic only (attribution of a Stage I round), not a gate"}
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(res, indent=1))
    print(json.dumps(res["reconstruction"]), flush=True)
    for name, v in res["gradient_balance"]["parts"].items():
        print(f"{name:15s} MIM {v['mim']:.2e}  contrast {v['contrast']:.2e}  MIM/contrast {v['ratio_mim_over_contrast']:.2f}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
