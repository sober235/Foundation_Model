"""Brain multi-disease inference and the per-study structured record (spec 2026-09-29 §6, §7).

A study's sequences (one disease model's native channels, one grid) and a SynthSeg label map on that grid go in; scored
lesions with their host structure and side, a disease impression and one sentence come out. The impression is the
model's disease: this entry detects one kind of lesion, it does not tell diseases apart (spec M13)."""
import json
import os
import subprocess
from pathlib import Path

import nibabel as nib
import numpy as np

from anatobind.bind.brain_lookup import BrainBinder
from anatobind.eval.lesion_boxes import load_label_map, load_nnunet_probabilities
from anatobind.eval.lesion_components import component_mask, component_rows, components
from anatobind.infer.knee import nnunet_env
from anatobind.nnunet.brain_disease import CONFIG, DISEASES, TRAINER

SIDE_ZH = {"left": "左侧", "right": "右侧", "bilateral": "双侧", "midline": ""}
HOST_ZH = {"white_matter": "大脑白质", "cortex": "大脑皮层", "thalamus": "丘脑", "basal_ganglia": "基底节",
           "brainstem": "脑干", "cerebellum": "小脑", "other_deep_grey": "深部灰质（海马、杏仁核等）"}
TYPE_ZH = {"tumor": "肿瘤样异常", "metastasis": "转移瘤样异常", "infarct": "梗死样异常"}
NOWHERE_ZH = "未能定位的区域"
MAX_SENTENCE_LESIONS = 5
INVOLVED_MIN = 0.10


def detections(pred, probs, voxel_mm3, family):
    """Scored components of the predicted label map, highest score first, and the component map they index.
    Components under the volume floor are dropped; the score is the mean foreground probability in the component."""
    comp, n = components(np.asarray(pred) == 1)
    rows = []
    for r in component_rows(comp, n, voxel_mm3, family):
        if r["ignore"]:
            continue
        sl, m = component_mask(comp, r)
        rows.append({**{k: v for k, v in r.items() if k != "ignore"}, "score": float(probs[1][sl][m].mean())})
    return sorted(rows, key=lambda r: -r["score"]), comp


def bind_rows(rows, comp, binder):
    """Attach host, host_rule, host_fractions and side to every row (in place); returns rows."""
    for r in rows:
        r.update(binder.bind(*component_mask(comp, r)))
    return rows


def volume_text(mm3):
    if mm3 >= 10000:
        return f"约 {mm3 / 1000:.0f} mL"
    if mm3 >= 1000:
        return f"约 {mm3 / 1000:.1f} mL"
    return f"约 {mm3:.0f} mm³"


def lesion_clause(lesion):
    where = SIDE_ZH[lesion["side"]] + HOST_ZH[lesion["host"]] if lesion["host"] else NOWHERE_ZH
    involved = [HOST_ZH[h] for h, f in sorted(lesion["host_fractions"].items(), key=lambda kv: -kv[1])
                if h != lesion["host"] and f >= INVOLVED_MIN]
    text = f"{where}存在{TYPE_ZH[lesion['type']]}，体积{volume_text(lesion['volume_mm3'])}"
    return text + (f"，累及{'、'.join(involved)}" if involved else "")


def study_record(study, disease, threshold, rows):
    """rows: bound detections (any score); only those at or above the threshold enter the record."""
    spec = DISEASES[disease]
    lesions = [{"type": spec["type"], "score": round(float(r["score"]), 4), "box": [int(v) for v in r["box"]],
                "volume_mm3": round(float(r["mm3"]), 1), "host": r["host"], "host_rule": r["host_rule"],
                "host_fractions": r["host_fractions"], "side": r["side"]}
               for r in sorted(rows, key=lambda r: -r["score"]) if r["score"] >= threshold]
    if lesions:
        rest = len(lesions) - MAX_SENTENCE_LESIONS
        sentence = "；".join(lesion_clause(l) for l in lesions[:MAX_SENTENCE_LESIONS])
        sentence += (f"；另有 {rest} 处同类异常" if rest > 0 else "") + f"。{spec['impression']}。"
        impression = spec["impression"]
    else:
        sentence, impression = f"未见{TYPE_ZH[spec['type']]}。", "未见相关异常"
    return {"study": study, "disease_model": disease, "threshold": float(threshold), "impression": impression,
            "lesions": lesions, "sentence": sentence}


def run_nnunet(dataset_id, in_dir, out_dir, folds, gpu):
    cmd = ["nnUNetv2_predict", "-i", str(in_dir), "-o", str(out_dir), "-d", str(dataset_id), "-c", CONFIG, "-tr", TRAINER,
           "-f", *[str(f) for f in folds], "-npp", "2", "-nps", "2", "--disable_progress_bar", "--save_probabilities"]
    subprocess.run(["nice", "-n", "19", *cmd], check=True, env=nnunet_env(gpu))


def link_inputs(in_dir, case, images, n_channels):
    if len(images) != n_channels:
        raise ValueError(f"{n_channels} channels are needed, {len(images)} given")
    Path(in_dir).mkdir(parents=True)
    for k, p in enumerate(images):
        if not Path(p).is_file():
            raise FileNotFoundError(p)
        os.symlink(Path(p).resolve(), Path(in_dir) / f"{case}_{k:04d}.nii.gz")


def run(disease, images, anatomy, out_dir, folds, gpu, threshold, predict=run_nnunet):
    """images: one NIfTI per channel, in the order of DISEASES[disease]["channels"], all on the anatomy's grid."""
    out = Path(out_dir)
    if out.exists():
        raise FileExistsError(f"{out} exists")
    spec = DISEASES[disease]
    seg_img = nib.load(str(anatomy))
    link_inputs(out / "input", "case", images, len(spec["channels"]))
    predict(spec["id"], out / "input", out / "pred", folds, gpu)
    pred = load_label_map(out / "pred" / "case.nii.gz")
    if pred.shape != seg_img.shape:
        raise ValueError(f"prediction {pred.shape} and anatomy {seg_img.shape} are on different grids")
    zooms = tuple(float(z) for z in seg_img.header.get_zooms()[:3])
    rows, comp = detections(pred, load_nnunet_probabilities(out / "pred" / "case.npz", pred), float(np.prod(zooms)), spec["type"])
    bind_rows(rows, comp, BrainBinder(np.asarray(seg_img.dataobj), zooms))
    record = study_record(Path(images[0]).name, disease, threshold, rows)
    (out / "record.json").write_text(json.dumps(record, ensure_ascii=False, indent=1))
    return record
