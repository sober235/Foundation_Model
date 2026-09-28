"""Brain small-lesion inference for S5 (spec 2026-09-28 §5): a fastMRI FLAIR h5 → nnU-Net (Dataset903) → scored
lesions with per-slice [row0, row1, col0, col1] boxes in the RSS frame, the Level R export's box format."""
import json
import subprocess
from pathlib import Path

import numpy as np
from scipy import ndimage

from anatobind.data_engine.fastmri import rss_h5_to_nifti
from anatobind.eval.lesion_boxes import BRAIN_MIN_VOXELS, STRUCTURE, decode_boxes, load_label_map, load_nnunet_probabilities
from anatobind.infer.knee import nnunet_env
from anatobind.nnunet.brain_lesion import DATASET_ID, FAMILIES, TRAINER


def run_nnunet(dataset_id, in_dir, out_dir, folds, gpu, config):
    cmd = ["nnUNetv2_predict", "-i", str(in_dir), "-o", str(out_dir), "-d", str(dataset_id), "-c", config, "-tr", TRAINER,
           "-f", *[str(f) for f in folds], "-npp", "2", "-nps", "2", "--disable_progress_bar", "--save_probabilities"]
    subprocess.run(cmd, check=True, env=nnunet_env(gpu))


def lesion_rows(dets, label_map):
    out = []
    # Label all connected components once to isolate each detection
    comp, n_comp = ndimage.label(label_map == 1, structure=STRUCTURE)

    for d in dets:
        c0, r0, s0, c1, r1, s1 = d["box"]
        # Find which component has the most voxels inside this detection's box
        box_region = comp[c0:c1, r0:r1, s0:s1]
        comp_ids, counts = np.unique(box_region[box_region > 0], return_counts=True)
        if len(comp_ids) == 0:
            continue
        comp_id = comp_ids[np.argmax(counts)]

        boxes = {}
        for s in range(s0, s1):
            # Extract only voxels of this component
            mask = comp[c0:c1, r0:r1, s] == comp_id
            cols, rows = np.nonzero(mask)
            if cols.size:
                boxes[str(s)] = [[int(r0 + rows.min()), int(r0 + rows.max() + 1), int(c0 + cols.min()), int(c0 + cols.max() + 1)]]
        out.append({"z0": int(s0), "z1": int(s1 - 1), "score": float(d["score"]), "boxes": boxes})
    return out


def run(h5_path, out_dir, folds, gpu, config="2d", predict=run_nnunet):
    out = Path(out_dir)
    if out.exists():
        raise FileExistsError(f"{out} exists")
    (out / "input").mkdir(parents=True)
    rss_h5_to_nifti(h5_path, out / "input" / "case_0000.nii.gz", pad_to_slices=0)
    predict(DATASET_ID, out / "input", out / "pred", folds, gpu, config)
    lab = load_label_map(out / "pred" / "case.nii.gz")
    dets = decode_boxes(lab, load_nnunet_probabilities(out / "pred" / "case.npz", lab), BRAIN_MIN_VOXELS, FAMILIES)
    rows = lesion_rows(dets, lab)
    (out / "lesions.json").write_text(json.dumps(rows, indent=1))
    return rows
