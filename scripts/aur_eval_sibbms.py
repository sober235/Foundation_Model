#!/usr/bin/env python
"""U on the 10 SibBMS subjects with MS plaque annotations (spec §7, report only; anatobind.aur.eval.evaluate_u_external).

    PYTHONNOUSERSITE=1 PYTHONPATH=. ~/anaconda3/envs/nvgen/bin/python scripts/aur_eval_sibbms.py \
        --checkpoint <STAGE_III_DIR>/aur_stage3_best.pt --out <NEW_RECORD_DIR> --gpu <idle card> [--sequence flair]

MS plaques are a pathology U never saw in training (gliomas, metastases, infarcts). The annotations live on the native
1 mm grid of each subject's annotation folder, with the four sequences; every label value > 0 counts as lesion (the
values 1-3 come without a legend). Of the 10 subjects, 3 are in the AnatoBind test split; the other 7 are training
patients whose images (never their plaques) supervised A and S and Stage I: both groups are reported, the test three
separately. Writes per_subject.json, sweep.json and REPORT.md into a new directory. NOT_EVIDENCE as an MS claim: one
annotation set, ten subjects."""
import argparse
import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from anatobind.aur import eval as E  # noqa: E402
from anatobind.aur.crops import CROP  # noqa: E402
from anatobind.aur.infer import model_from_export  # noqa: E402
from anatobind.aur.ssl.checkpoint import code_sha, file_sha256  # noqa: E402

ANNOTATION = Path("/data2/congcong/data/FM_data/SibBMS_ms/sibbms/Output/Annotation")
SAMPLES = Path("/data2/congcong/data/FM_data/derived/aur/samples_1mm_v1.json")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--checkpoint", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--annotation", type=Path, default=ANNOTATION)
    ap.add_argument("--samples", type=Path, default=SAMPLES)
    ap.add_argument("--sequence", default="flair", choices=("flair", "t1", "t1c", "t2"))
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
    split = {}
    for r in json.loads(a.samples.read_text()):
        if r["source"] == "sibbms" and str(r["patient"]).startswith("MS_"):
            split[str(r["patient"])[3:]] = r["split"]
    per, subjects = {}, sorted(p.name for p in a.annotation.iterdir() if p.is_dir())
    for sub in subjects:
        d = a.annotation / sub / "ses-001"
        image = next(iter(sorted(d.glob(f"{sub}_*{a.sequence}.nii.gz"))), None) if a.sequence != "t1" else d / f"{sub}_t1.nii.gz"
        label = d / f"{sub}_Segmentation-label.nii.gz"
        if image is None or not image.exists() or not label.exists():
            raise FileNotFoundError(f"{sub}: no {a.sequence} image or label in {d}")
        res = E.evaluate_u_external(model, image, label, crop, device, batch_size=a.batch_size)
        res["split"] = split.get(sub, "unknown")
        per[sub] = res
        print(f"{sub} ({res['split']}): {sum(1 for g in res['gt'] if not g.get('ignore'))} plaques, {len(res['dets'])} detections", flush=True)
    sweeps = {"all": E.u_sweep(per), "test": E.u_sweep({k: v for k, v in per.items() if v["split"] == "test"}) if any(v["split"] == "test" for v in per.values()) else []}
    a.out.mkdir(parents=True)
    (a.out / "per_subject.json").write_text(json.dumps(per, indent=1, default=str))
    (a.out / "sweep.json").write_text(json.dumps({"sweeps": sweeps, "run": {"checkpoint": str(a.checkpoint), "checkpoint_sha256": file_sha256(a.checkpoint),
                                                                           "code_sha": code_sha(), "sequence": a.sequence, "crop": list(crop)}}, indent=1))
    L = [f"# U on the 10 SibBMS MS subjects ({a.sequence}; report only, NOT_EVIDENCE as an MS claim)", "",
         "MS plaques were never a U training target. 3 subjects are in the test split; 7 are training patients whose images (not their plaques) were seen.", "",
         "| subject | split | plaques (≥ 10 mm³) | ignored (< 10 mm³) | label values (mm³) |", "|---|---|---|---|---|"]
    for sub, r in per.items():
        L.append(f"| {sub} | {r['split']} | {sum(1 for g in r['gt'] if not g.get('ignore'))} | {sum(1 for g in r['gt'] if g.get('ignore'))} | {r['label_values']} |")
    for name, rows in sweeps.items():
        if not rows:
            continue
        L += ["", f"## Sweep ({name} subjects)", "", "| threshold | sensitivity | FP / scan | lesions |", "|---|---|---|---|"]
        L += [f"| {r['thr']:.2f} | {r['sensitivity']:.4f} | {r['fp_per_scan']:.2f} | {r['n_gt']} |" for r in rows]
    (a.out / "REPORT.md").write_text("\n".join(L) + "\n")
    print("\n".join(L))
    return 0


if __name__ == "__main__":
    sys.exit(main())
