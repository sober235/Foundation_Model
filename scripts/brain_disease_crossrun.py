#!/usr/bin/env python
# scripts/brain_disease_crossrun.py
"""Cross false-alarm check (spec 2026-09-29 M11, report only): run one disease's detector on another disease's data.

Only three pairs have the sequences: infarct model on glioma data (PDGM has DWI and ADC), metastasis model on glioma
data (T1, T1c, FLAIR), glioma model on metastasis data (BMSR's T2 is synthetic). The cases are the fold 0 validation
cases of the data's own split; the model never saw that dataset.

  PYTHONNOUSERSITE=1 PYTHONPATH=. nice -n 19 ~/anaconda3/envs/nvgen/bin/python scripts/brain_disease_crossrun.py \
      --model infarct --data glioma --threshold 0.55 --gpu 4 \
      --work /data2/congcong/data/FM_data/derived/brain_disease/crossrun/infarct_on_glioma \
      --out docs/verification/2026-09-30/brain_multidisease/crossrun/infarct_on_glioma
"""
import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from anatobind.eval.lesion_boxes import load_label_map, load_nnunet_probabilities  # noqa: E402
from anatobind.eval.lesion_components import component_mask, component_rows, components  # noqa: E402
from anatobind.infer.brain_disease import detections, run_nnunet  # noqa: E402
from anatobind.nnunet.brain_disease import DISEASES, FM, channel_path, fold_dir  # noqa: E402

NNUNET = FM / "derived/nnunet"
CROSS = {("infarct", "glioma"): ("DWI", "ADC"),
         ("metastasis", "glioma"): ("T1", "T1c", "FLAIR"),
         ("glioma", "metastasis"): ("T1pre", "T1post", "T2Synth", "FLAIR")}
NOTES = {("glioma", "metastasis"): "channel 2 (T2) is BMSR's synthetic T2"}


def overlap_counts(dets, det_comp, gt_rows, gt_comp, thr):
    """Detections at thr against another disease's ground truth: how many detections touch any ground-truth voxel, and
    how many counted ground-truth lesions are touched by any detection.

    A labelled fragment under the volume floor is lesion tissue: a detection on it is on ground truth. It is no counted
    lesion, so it is never in n_gt and cannot be claimed."""
    kept = [d for d in dets if d["score"] >= thr]
    gt_any = gt_comp > 0
    det_any = np.zeros(gt_comp.shape, bool)
    touching = 0
    for d in kept:
        sl, m = component_mask(det_comp, d)
        det_any[sl] |= m
        touching += int((gt_any[sl] & m).any())
    real = [r for r in gt_rows if not r["ignore"]]
    claimed = 0
    for r in real:
        sl, m = component_mask(gt_comp, r)
        claimed += int((det_any[sl] & m).any())
    return {"n_det": len(kept), "n_det_on_gt": touching, "n_gt": len(real), "n_gt_claimed": claimed}


def summarise(per_case):
    tot = {k: sum(c[k] for c in per_case.values()) for k in ("n_det", "n_det_on_gt", "n_gt", "n_gt_claimed")}
    n = len(per_case)
    return {"n_scans": n, **tot, "det_per_scan": tot["n_det"] / n if n else None,
            "scans_with_any_detection": sum(1 for c in per_case.values() if c["n_det"]),
            "share_of_detections_on_gt": tot["n_det_on_gt"] / tot["n_det"] if tot["n_det"] else None,
            "share_of_gt_claimed": tot["n_gt_claimed"] / tot["n_gt"] if tot["n_gt"] else None}


def main(argv=None):
    ap = argparse.ArgumentParser(description="Run one disease's detector on another disease's data (report only)")
    ap.add_argument("--model", choices=sorted(DISEASES), required=True)
    ap.add_argument("--data", choices=sorted(DISEASES), required=True)
    ap.add_argument("--threshold", type=float, required=True, help="the model's operating threshold")
    ap.add_argument("--folds", type=int, nargs="+", default=[0, 1, 2, 3, 4], help="model folds to ensemble")
    ap.add_argument("--gpu", type=int, required=True)
    ap.add_argument("--work", type=Path, required=True, help="inputs and predictions (must not exist)")
    ap.add_argument("--out", type=Path, required=True, help="report directory (must not exist)")
    a = ap.parse_args(argv)
    if (a.model, a.data) not in CROSS:
        raise SystemExit(f"no cross run for the {a.model} model on {a.data} data; possible: {sorted(CROSS)}")
    for p in (a.work, a.out):
        if p.exists():
            raise FileExistsError(f"{p} already exists")
    for f in a.folds:
        ckpt = fold_dir(NNUNET / "results", a.model, f) / "checkpoint_final.pth"
        if not ckpt.is_file():
            raise FileNotFoundError(f"the {a.model} model has no trained fold {f}: missing {ckpt}")
    host = DISEASES[a.data]["name"]
    info = json.loads((NNUNET / "raw" / host / "cases.json").read_text())
    cases = json.loads((NNUNET / "preprocessed" / host / "splits_final.json").read_text())[0]["val"]
    srcs = {case: [channel_path(a.data, case, ch, FM) for ch in CROSS[(a.model, a.data)]] for case in cases}
    for case, paths in srcs.items():
        for src in paths:
            if not src.is_file():
                raise FileNotFoundError(f"{case}: missing {src}")
    (a.work / "input").mkdir(parents=True)
    for case, paths in srcs.items():
        for k, src in enumerate(paths):
            os.symlink(src.resolve(), a.work / "input" / f"{case}_{k:04d}.nii.gz")
    run_nnunet(DISEASES[a.model]["id"], a.work / "input", a.work / "pred", a.folds, a.gpu)
    per_case = {}
    for case in cases:
        pred = load_label_map(a.work / "pred" / f"{case}.nii.gz")
        gt = load_label_map(NNUNET / "raw" / host / "labelsTr" / f"{case}.nii.gz")
        if pred.shape != gt.shape:
            raise ValueError(f"{case}: prediction {pred.shape} and ground truth {gt.shape} differ")
        vox = info[case]["voxel_mm3"]
        dets, det_comp = detections(pred, load_nnunet_probabilities(a.work / "pred" / f"{case}.npz", pred), vox, DISEASES[a.model]["type"])
        gt_comp, n = components(gt)
        per_case[case] = overlap_counts(dets, det_comp, component_rows(gt_comp, n, vox, DISEASES[a.data]["type"]), gt_comp, a.threshold)
    s = {"model": a.model, "data": a.data, "cases": f"fold 0 validation cases of {host}",
         "channels": list(CROSS[(a.model, a.data)]), "note": NOTES.get((a.model, a.data)),
         "threshold": a.threshold, "model_folds": sorted(a.folds), **summarise(per_case)}
    a.out.mkdir(parents=True)
    (a.out / "crossrun.json").write_text(json.dumps({"summary": s, "per_case": per_case}, indent=1))
    (a.out / "REPORT.md").write_text("".join([
        f"# Cross false-alarm check: {a.model} model on {a.data} data (report only, spec M11)\n\n",
        "The model never saw this dataset. Resolution, preprocessing and scanners differ from its training data, so these "
        "numbers describe this pair of datasets, not the diseases in general.\n\n",
        "Counting: a detection is on ground truth when it shares a voxel with any labelled voxel, fragments under "
        "10 mm3 included; a ground-truth lesion is claimed when a detection shares a voxel with it, and only lesions of "
        "at least 10 mm3 are counted. These are counts of overlap: not a sensitivity, not a precision and not a "
        "false-positive rate.\n\n"
        "Threshold: it is the model's operating threshold, measured on single-fold models (every case predicted by the "
        "fold that held it out). Here the model's folds are averaged; the behaviour of the averaged model at this "
        "threshold was not measured on its own data.\n\n"
        "```json\n", json.dumps(s, indent=1),
        "\n```\n\n## Command\n\n```\n", " ".join(sys.argv), "\n```\n"]))
    print(json.dumps(s))


if __name__ == "__main__":
    main()
