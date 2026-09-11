"""Uncompressed, pre-normalised training arrays for the m1r export.

Reading and normalising one 256x256x160 gzip NIfTI costs 6-11 s on the shared machine (measured
2026-09-10) while a whole-volume GPU step is ~0.1 s. The cache stores each view already
percentile-clipped, z-scored and permuted to the model frame (Z, Y, X) as float16 .npy, so a data
worker only memory-maps it. meta.json is written last; its presence marks a finished scan.
"""
import json
from pathlib import Path

import nibabel as nib
import numpy as np

from anatobind.train.dataset import normalise_volume, to_model_frame

VIEW_FILES = {
    "clean": "image_clean_e1.nii.gz",
    "noise_q1": "image_noise_q1_e1.nii.gz",
    "noise_q2": "image_noise_q2_e1.nii.gz",
    "noise_q3": "image_noise_q3_e1.nii.gz",
    "us4": "image_us4_e1.nii.gz",
    "us8": "image_us8_e1.nii.gz",
    "us16": "image_us16_e1.nii.gz",
}
VIEWS = tuple(VIEW_FILES)


def cache_scan(export_dir, cache_dir):
    export_dir, cache_dir = Path(export_dir), Path(cache_dir)
    if (cache_dir / "meta.json").exists():
        return False
    cache_dir.mkdir(parents=True, exist_ok=True)
    shape = zooms = None
    for view, name in VIEW_FILES.items():
        img = nib.load(str(export_dir / name))
        arr = to_model_frame(normalise_volume(img.get_fdata(dtype=np.float32))).astype(np.float16)
        np.save(cache_dir / f"{view}.npy", arr)
        if view == "clean":
            shape, zooms = arr.shape, img.header.get_zooms()[:3]
    seg = to_model_frame(np.asanyarray(nib.load(str(export_dir / "seg.nii.gz")).dataobj).astype(np.uint8))
    assert seg.shape == shape, (seg.shape, shape)
    np.save(cache_dir / "seg.npy", seg)
    meta = {"shape": list(shape), "spacing_mm": [float(v) for v in zooms[::-1]], "views": list(VIEWS)}
    (cache_dir / "meta.json").write_text(json.dumps(meta))
    return True


def load_meta(cache_dir):
    return json.loads((Path(cache_dir) / "meta.json").read_text())


def load_array(cache_dir, name):
    return np.load(Path(cache_dir) / f"{name}.npy", mmap_mode="r")
