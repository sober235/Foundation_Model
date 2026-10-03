#!/usr/bin/env python
# scripts/eval_brain_anatomy.py
"""S4 evaluation (spec 2026-10-02 §7, A11, A13): agreement of the student with the SynthSeg pseudo-labels on the
fastMRI stacks (reliable slices), host agreement over the fastMRI+ lesion boxes, the outline model's Dice on its test
stacks, and Dice on the simulated test set. All numbers are NOT_EVIDENCE (pseudo-labels).

  PYTHONNOUSERSITE=1 PYTHONPATH=. nice -n 19 ~/anaconda3/envs/nvgen/bin/python scripts/eval_brain_anatomy.py \
      --out docs/verification/2026-10-02/brain_anatomy_flair/eval --work /data2/congcong/data/FM_data/derived/brain_anatomy/eval_<date> --gpu <idle>

Both directories must not exist. The two nnU-Net models run once each over all stacks (batch prediction)."""
import argparse
import csv
import json
import os
import sys
from pathlib import Path

import nibabel as nib
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from anatobind.anatomy.labels import HOST_IDS, to_student  # noqa: E402
from anatobind.anatomy.outline import postprocess  # noqa: E402
from anatobind.eval.brain_anatomy import (  # noqa: E402
    GATES, class_dice, host_agreement, outline_dice, pool_agreement, reliable_slices, summarize_dice, verdict,
)
from anatobind.eval.brain_disease import code_version  # noqa: E402
from anatobind.eval.lesion_boxes import load_label_map  # noqa: E402
from anatobind.infer.brain_anatomy import OUTLINE, STUDENT, run_nnunet  # noqa: E402
from anatobind.infer.knee import NNUNET_ROOT  # noqa: E402
from anatobind.level_r.registry import load_registry  # noqa: E402
from anatobind.nnunet.brain_disease import FM  # noqa: E402

WORK = FM / "derived/brain_anatomy"
FASTMRI_SEG = FM / "derived/synthseg/fastmri_brain/seg_native"
DATASET907 = "Dataset907_BrainAnatomyFLAIR"
DATASET908 = "Dataset908_FastMRIBrainOutline"
LOW_SLICES = (0, 1)


def link_stacks(dataset_dir, staging):
    """Every fastMRI stack of Dataset908 (training and test images) -> <staging>/<stem>_0000.nii.gz symlinks."""
    staging.mkdir(parents=True)
    stems = {}
    for folder in ("imagesTr", "imagesTs"):
        for p in sorted((dataset_dir / folder).glob("*_0000.nii.gz")):
            stem = p.name[:-len("_0000.nii.gz")]
            os.symlink(p.resolve(), staging / p.name)
            stems[stem] = folder
    return stems


def mask_stacks(staging, masks_dir, out_dir):
    """Skull-stripped copies of the stacks: image x postprocessed outline mask."""
    out_dir.mkdir(parents=True)
    for p in sorted(staging.glob("*_0000.nii.gz")):
        stem = p.name[:-len("_0000.nii.gz")]
        img = nib.load(str(p))
        mask = postprocess(load_label_map(masks_dir / f"{stem}.nii.gz"))
        if mask.shape != img.shape:
            raise ValueError(f"{stem}: outline {mask.shape} and stack {img.shape} differ")
        nib.save(nib.Nifti1Image(np.asarray(img.dataobj).astype(np.float32) * mask, img.affine), str(out_dir / p.name))
        nib.save(nib.Nifti1Image(mask, img.affine), str(out_dir / f"{stem}_mask.nii.gz"))


def stem_metrics(student, synthseg, mask, spacing, rows, is_test):
    """One fastMRI stack: Dice of the host classes on the reliable slices, host agreement of its lesion boxes, outline
    Dice when the stack is an outline-model test stack, and the student's brain share in the lowest two slices."""
    area = spacing[0] * spacing[1]
    reliable = reliable_slices(synthseg, area)
    d = class_dice(student, student_of(synthseg), slices=reliable) if len(reliable) else {c: None for c in HOST_IDS}
    agree = host_agreement(student, synthseg, spacing, rows, reliable)
    low = {int(k): round(float((student[:, :, k] > 0).sum() * area / 100.0), 1) for k in LOW_SLICES if k < student.shape[2]}   # cm2 of labelled tissue
    return {"reliable": [reliable.start, reliable.stop - 1] if len(reliable) else None, "dice": d, "agreement": agree,
            "outline_dice": outline_dice(mask, synthseg, area) if is_test else None, "low_slice_area_cm2": low}


def student_of(synthseg):
    return to_student(synthseg)


def evaluate_fastmri(stems, pred_student_dir, masked_dir, seg_dir, registry_rows):
    per_stem = {}
    for stem, folder in sorted(stems.items()):
        seg_img = nib.load(str(Path(seg_dir) / f"{stem}_seg.nii.gz"))
        synthseg = np.asarray(seg_img.dataobj).astype(np.int16)
        spacing = tuple(float(z) for z in seg_img.header.get_zooms()[:3])
        student = load_label_map(pred_student_dir / f"{stem}.nii.gz")
        mask = load_label_map(masked_dir / f"{stem}_mask.nii.gz")
        if student.shape != synthseg.shape:
            raise ValueError(f"{stem}: student {student.shape} and SynthSeg {synthseg.shape} differ")
        per_stem[stem] = {"split": "test" if folder == "imagesTs" else "train",
                          **stem_metrics(student, synthseg, mask, spacing, registry_rows.get(stem, []), folder == "imagesTs")}
    return per_stem


def evaluate_simulated(pred_dir, label_dir, samples):
    per_sample = {}
    for s in sorted(samples):
        pred, ref = load_label_map(pred_dir / f"{s}.nii.gz"), load_label_map(label_dir / f"{s}.nii.gz")
        per_sample[s] = class_dice(pred, ref, classes=tuple(HOST_IDS) + (14,))
    return per_sample


def summarize(per_stem, per_sample):
    dice = summarize_dice([v["dice"] for v in per_stem.values()])
    agreement = pool_agreement([v["agreement"] for v in per_stem.values()])
    outl = [v["outline_dice"] for v in per_stem.values() if v["outline_dice"] is not None]
    outline = {"n_test_stacks": len(outl), "mean": float(np.mean(outl)) if outl else None, "min": float(np.min(outl)) if outl else None}
    sim = summarize_dice(list(per_sample.values())) if per_sample else None
    vent = [d.get(14) for d in per_sample.values() if d.get(14) is not None]
    if sim is not None:
        sim["ventricles"] = float(np.mean(vent)) if vent else None
    low = {k: float(np.median([v["low_slice_area_cm2"].get(k, v["low_slice_area_cm2"].get(str(k), 0.0)) for v in per_stem.values()])) for k in LOW_SLICES}
    v = verdict(agreement["rate"], dice["mean_host_dice"], outline["mean"])
    return {"verdict": v, "fastmri_dice": dice, "host_agreement": {k: val for k, val in agreement.items()}, "outline": outline,
            "simulated_test": sim, "low_slice_median_area_cm2": low, "n_stacks": len(per_stem)}


def report(summary, per_stem, out, command, code):
    v, d, a, o, s = summary["verdict"], summary["fastmri_dice"], summary["host_agreement"], summary["outline"], summary["simulated_test"]
    L = ["# S4 brain anatomy on fastMRI FLAIR: evaluation (NOT_EVIDENCE: agreement with SynthSeg pseudo-labels)\n\n",
         f"Verdict (spec A11): **{'pass' if v['pass'] else 'fail'}** — host agreement {v['values']['host_agreement']}, mean host Dice {v['values']['mean_host_dice']}, "
         f"outline Dice {v['values']['outline_dice']} (gates {GATES['host_agreement']} / {GATES['mean_host_dice']} / {GATES['outline_dice']}).\n\n",
         "The final judgement of S4 waits for the Level R reader labels (A12); the lowest two slices are reported, not judged.\n\n",
         "## fastMRI stacks (reliable slices only)\n\n", "```json\n", json.dumps({"n_stacks": summary["n_stacks"], "host_agreement": a, "mean_host_dice": d["mean_host_dice"]}, indent=1), "\n```\n\n",
         "| class | mean Dice over stacks |\n|---|---|\n"]
    L += [f"| {name} | {'n/a' if val is None else f'{val:.4f}'} |\n" for name, val in d["per_class"].items()]
    L += ["\n## Outline model (its test stacks)\n\n", "```json\n", json.dumps(o, indent=1), "\n```\n\n",
          "## Lowest two slices (student output, median labelled area in cm2; no reliable reference there)\n\n", "```json\n",
          json.dumps(summary["low_slice_median_area_cm2"], indent=1), "\n```\n\n"]
    if s is not None:
        L += ["## Simulated test set (A13, report only)\n\n", "```json\n", json.dumps(s, indent=1), "\n```\n\n"]
    L += ["## Command\n\n```\n", command, "\n```\n\n", f"Code: commit {code}\n"]
    (out / "REPORT.md").write_text("".join(L))
    (out / "verdict.json").write_text(json.dumps(summary, indent=1))
    with open(out / "per_stem.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["stem", "split", "reliable_first", "reliable_last", "mean_host_dice", "n_lesions", "n_evaluated", "n_agree", "outline_dice"])
        for stem, v in sorted(per_stem.items()):
            vals = [x for x in v["dice"].values() if x is not None]
            w.writerow([stem, v["split"], *(v["reliable"] or ["", ""]), f"{np.mean(vals):.4f}" if vals else "", v["agreement"]["n_lesions"],
                        v["agreement"]["n_evaluated"], v["agreement"]["n_agree"], "" if v["outline_dice"] is None else f"{v['outline_dice']:.4f}"])
    (out / "per_stem.json").write_text(json.dumps(per_stem, indent=1))


def main(argv=None, predict=run_nnunet):
    ap = argparse.ArgumentParser(description="Evaluate the S4 anatomy model")
    ap.add_argument("--out", type=Path, required=True, help="record directory (must not exist)")
    ap.add_argument("--work", type=Path, required=True, help="prediction directory under /data2 (must not exist)")
    ap.add_argument("--gpu", type=int, required=True)
    ap.add_argument("--sim-work", type=Path, default=WORK, help="where sim/test and the manifest live")
    ap.add_argument("--seg-dir", type=Path, default=FASTMRI_SEG)
    a = ap.parse_args(argv)
    for p in (a.out, a.work):
        if p.exists():
            raise FileExistsError(f"{p} exists")
    code = code_version(Path(__file__).resolve().parents[1])
    raw = NNUNET_ROOT / "raw"
    stems = link_stacks(raw / DATASET908, a.work / "stacks")
    predict(OUTLINE["id"], OUTLINE["config"], a.work / "stacks", a.work / "pred_outline", [0], a.gpu)
    mask_stacks(a.work / "stacks", a.work / "pred_outline", a.work / "stripped")
    predict(STUDENT["id"], STUDENT["config"], a.work / "stripped", a.work / "pred_student", [0], a.gpu)
    rows = {}
    for r in load_registry():
        rows.setdefault(r["file"], []).append(r)
    per_stem = evaluate_fastmri(stems, a.work / "pred_student", a.work / "stripped", a.seg_dir, rows)
    samples = [p.name[:-len("_0000.nii.gz")] for p in sorted((raw / DATASET907 / "imagesTs").glob("*_0000.nii.gz"))]
    per_sample = {}
    if samples:
        predict(STUDENT["id"], STUDENT["config"], raw / DATASET907 / "imagesTs", a.work / "pred_sim_test", [0], a.gpu)
        per_sample = evaluate_simulated(a.work / "pred_sim_test", a.sim_work / "sim" / "test", samples)
    summary = summarize(per_stem, per_sample)
    a.out.mkdir(parents=True)
    report(summary, per_stem, a.out, " ".join(sys.argv), code)
    v = summary["verdict"]
    print(f"stacks {summary['n_stacks']}; host agreement {v['values']['host_agreement']}; mean host Dice {v['values']['mean_host_dice']}; "
          f"outline Dice {v['values']['outline_dice']}; pass {v['pass']}")
    return summary


if __name__ == "__main__":
    main()
