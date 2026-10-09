#!/usr/bin/env python
"""One brain MRI volume through AnatoBind-Brain (spec §8; SSL-first plan T12).

    PYTHONNOUSERSITE=1 PYTHONPATH=. ~/anaconda3/envs/nvgen/bin/python scripts/infer_anatobind_brain.py \
        --nifti <volume.nii.gz> --checkpoint <STAGE_III_DIR>/aur_stage3_best.pt --out <NEW_DIR> --gpu <idle card> [--seq FLAIR]

Writes anatomy.nii.gz (SynthSeg values from the A head), lesions.nii.gz (instance ids from the U head) in the input's
own orientation, and record.json (per lesion: host, side, 14 host probabilities, volume, box; the sentences). Without
--seq the S head's prediction is used and the record says so. Every output rests on pseudo-label-trained heads:
NOT_EVIDENCE."""
import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from anatobind.aur import infer as I  # noqa: E402
from anatobind.aur.crops import CROP  # noqa: E402
from anatobind.aur.labels import SEQ_TYPES, entity_to_synthseg  # noqa: E402
from anatobind.aur.ssl.checkpoint import code_sha, file_sha256  # noqa: E402


def parser():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--nifti", type=Path, required=True)
    ap.add_argument("--checkpoint", type=Path, required=True, help="aur_stage2_best.pt or aur_stage3_best.pt")
    ap.add_argument("--out", type=Path, required=True, help="a new directory")
    dev = ap.add_mutually_exclusive_group(required=True)
    dev.add_argument("--gpu", type=int, default=None, help="CUDA device index")
    dev.add_argument("--cpu", action="store_true")
    ap.add_argument("--seq", choices=SEQ_TYPES, default=None, help="the sequence type; predicted by the S head when absent")
    ap.add_argument("--crop", type=int, nargs=3, default=None, help="window size (z, y, x); default: the export's training crop")
    ap.add_argument("--batch-size", type=int, default=1)
    ap.add_argument("--instance-threshold", type=float, default=I.LESION_THRESHOLD, help="lesion-map threshold of the evaluated operating point")
    ap.add_argument("--score-threshold", type=float, default=None, help="drop lesions whose mean probability is below this (the evaluated score threshold)")
    return ap


def main(argv=None):
    a = parser().parse_args(argv)
    if a.out.exists():
        raise FileExistsError(f"{a.out} exists; the output directory must be new")
    if a.cpu:
        device = torch.device("cpu")
    else:
        if not torch.cuda.is_available():
            raise RuntimeError("--gpu given but CUDA is not available")
        device = torch.device("cuda", a.gpu)
    t0 = time.time()
    model, meta = I.model_from_export(a.checkpoint, device)
    crop = tuple(a.crop) if a.crop else tuple(meta.get("config", {}).get("crop", CROP))
    image, affine, orig = I.load_input(a.nifti)
    pred = I.predict_volume(model, image, affine, crop, device, batch_size=a.batch_size)
    voxel_mm3 = abs(float(np.linalg.det(affine[:3, :3])))
    inst, rows = I.instances_from_probability(pred["lesion_prob"], voxel_mm3, a.instance_threshold, score_threshold=a.score_threshold)
    bound = I.bind_instances(model, image, affine, inst, crop, device)
    lesions = I.lesion_records(rows, bound)
    seq = a.seq if a.seq else SEQ_TYPES[int(np.argmax(pred["seq_probs"]))]
    record = {"input": str(a.nifti), "checkpoint": str(a.checkpoint), "checkpoint_sha256": file_sha256(a.checkpoint), "code_sha": code_sha(),
              "stage": meta.get("stage"), "crop": list(crop), "sequence": seq, "sequence_source": "given" if a.seq else "predicted",
              "seq_probs": {s: float(p) for s, p in zip(SEQ_TYPES, pred["seq_probs"])}, "n_windows": pred["n_windows"],
              "anatomy_source": I.ANATOMY_SOURCE, "evidence": "NOT_EVIDENCE",
              "operating_point": {"instance_threshold": a.instance_threshold, "score_threshold": a.score_threshold,
                                  "note": "pass the thresholds the evaluation report chose; without --score-threshold every component above the map threshold is kept"},
              "entity_presence_mean_over_windows": [float(p) for p in pred["entity_presence"]], "n_lesions": len(lesions), "lesions": lesions,
              "sentences": I.sentences(lesions), "seconds": None}
    record["seconds"] = round(time.time() - t0, 1)
    I.write_outputs(a.out, orig, entity_to_synthseg(pred["entity"]), inst, record)
    print("\n".join(record["sentences"]))
    print(f"wrote {a.out} ({record['seconds']} s, {pred['n_windows']} windows, {len(lesions)} lesions)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
