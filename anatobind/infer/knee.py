"""One volume in, three outputs out: anatomy label map, lesion boxes with family, host structure (spec §3.4).

Both nnU-Net models are called through nnUNetv2_predict as subprocesses (the predictor is injectable
so the pipeline is testable without a GPU); the rest is lesion_boxes decoding and lookup.
"""
import csv
import json
import os
import shutil
import subprocess
from pathlib import Path

import matplotlib
import nibabel as nib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from anatobind.eval.lesion_boxes import decode_boxes, load_label_map, load_nnunet_probabilities  # noqa: E402
from anatobind.eval.lookup import LabelIndex, describe_host  # noqa: E402
from anatobind.infer.canonical import to_export_frame  # noqa: E402
from anatobind.nnunet.lesion_labels import LOOKUP_CLASS_OF_FAMILY  # noqa: E402
from anatobind.nnunet.prepare import DATASET_ID as ANATOMY_ID  # noqa: E402
from anatobind.nnunet.prepare import TRAINER  # noqa: E402
from anatobind.nnunet.prepare_lesion import DATASET_ID as LESION_ID  # noqa: E402

CSV_FIELDS = ["lesion_id", "family", "score", "x0", "y0", "z0", "x1", "y1", "z1", "n_voxels",
              "host_label", "host_name", "side", "host_fractions"]
NNUNET_ROOT = Path("/data2/congcong/data/FM_data/derived/nnunet")
CASE = "case"


def nnunet_env(gpu):
    env = dict(os.environ)
    env.setdefault("nnUNet_raw", str(NNUNET_ROOT / "raw"))
    env.setdefault("nnUNet_preprocessed", str(NNUNET_ROOT / "preprocessed"))
    env.setdefault("nnUNet_results", str(NNUNET_ROOT / "results"))
    env["PYTHONNOUSERSITE"] = "1"
    env["CUDA_VISIBLE_DEVICES"] = str(gpu)
    env["PATH"] = str(Path.home() / "anaconda3/envs/nvgen/bin") + os.pathsep + env.get("PATH", "")
    return env


def run_nnunet(dataset_id, in_dir, out_dir, folds, gpu, save_probabilities):
    cmd = ["nnUNetv2_predict", "-i", str(in_dir), "-o", str(out_dir), "-d", str(dataset_id), "-c", "3d_fullres",
           "-tr", TRAINER, "-f", *[str(f) for f in folds], "-npp", "2", "-nps", "2", "--disable_progress_bar"]
    if save_probabilities:
        cmd.append("--save_probabilities")
    subprocess.run(cmd, check=True, env=nnunet_env(gpu))


def lesion_table(dets, anatomy, spacing):
    index = LabelIndex(anatomy, spacing)
    rows = []
    for i, d in enumerate(dets, start=1):
        h = describe_host(index, d["box"], LOOKUP_CLASS_OF_FAMILY[d["family"]])
        x0, y0, z0, x1, y1, z1 = (int(v) for v in d["box"])
        rows.append({"lesion_id": i, "family": d["family"], "score": round(float(d["score"]), 4),
                     "x0": x0, "y0": y0, "z0": z0, "x1": x1, "y1": y1, "z1": z1, "n_voxels": int(d["n_voxels"]),
                     "host_label": "" if h["host_label"] is None else h["host_label"], "host_name": h["host_name"],
                     "side": h["side"],
                     "host_fractions": json.dumps({str(k): round(v, 4) for k, v in h["host_fractions"].items()})})
    return rows


def write_table(rows, path):
    with open(path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=CSV_FIELDS)
        w.writeheader()
        w.writerows(rows)


def render_overlay(image, anatomy, box, path):
    """Three orthogonal slices through the box centre, anatomy in colour, the box as a rectangle."""
    c = [min((box[i] + box[i + 3]) // 2, image.shape[i] - 1) for i in range(3)]
    planes = [(image[c[0]], anatomy[c[0]], (box[1], box[2], box[4], box[5]), "x"),
              (image[:, c[1]], anatomy[:, c[1]], (box[0], box[2], box[3], box[5]), "y"),
              (image[:, :, c[2]], anatomy[:, :, c[2]], (box[0], box[1], box[3], box[4]), "z")]
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    for ax, (im, lab, (a0, b0, a1, b1), name) in zip(axes, planes):
        ax.imshow(im.T, cmap="gray", origin="lower", vmin=np.percentile(im, 1), vmax=max(np.percentile(im, 99), 1e-6))
        ax.imshow(np.ma.masked_where(lab.T == 0, lab.T), cmap="tab10", origin="lower", vmin=0, vmax=9, alpha=0.4,
                  interpolation="nearest")
        ax.add_patch(plt.Rectangle((a0, b0), a1 - a0, b1 - b0, fill=False, edgecolor="yellow", linewidth=1.5))
        ax.set_title(f"slice through box centre, axis {name}", fontsize=9)
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def run(image_path, out_dir, frame, folds, gpu, predict=run_nnunet):
    out_dir = Path(out_dir)
    img = nib.load(str(image_path))
    if frame == "world":
        img = to_export_frame(img)
    in_dir = out_dir / "input"
    in_dir.mkdir(parents=True, exist_ok=True)
    nib.save(img, str(in_dir / f"{CASE}_0000.nii.gz"))
    predict(ANATOMY_ID, in_dir, out_dir / "anatomy_raw", folds, gpu, False)
    predict(LESION_ID, in_dir, out_dir / "lesion_raw", folds, gpu, True)
    anatomy = load_label_map(out_dir / "anatomy_raw" / f"{CASE}.nii.gz")
    lesion = load_label_map(out_dir / "lesion_raw" / f"{CASE}.nii.gz")
    dets = decode_boxes(lesion, load_nnunet_probabilities(out_dir / "lesion_raw" / f"{CASE}.npz", lesion))
    spacing = tuple(float(v) for v in img.header.get_zooms()[:3])
    rows = lesion_table(dets, anatomy, spacing)
    shutil.copyfile(out_dir / "anatomy_raw" / f"{CASE}.nii.gz", out_dir / "anatomy.nii.gz")
    shutil.copyfile(out_dir / "lesion_raw" / f"{CASE}.nii.gz", out_dir / "lesions.nii.gz")
    write_table(rows, out_dir / "lesions.csv")
    image = np.asanyarray(img.dataobj).astype(np.float32)
    box = dets[0]["box"] if dets else tuple(s // 2 for s in image.shape) + tuple(s // 2 + 1 for s in image.shape)
    render_overlay(image, anatomy, box, out_dir / "overlay.png")
    return rows
