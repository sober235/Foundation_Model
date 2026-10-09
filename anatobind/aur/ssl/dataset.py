"""Stage I crops (SSL-first plan §2.2, §3.2; decisions Q3, Q6, Q7 of 2026-10-09).

One item is one sample row (a volume of one sequence) and yields `crops_per_volume` source crops of it, each with two
intensity views. Stage I reads no label: the volume is reoriented to RAS, normalised over the whole volume (the same
normalisation as Stage II; the hidden voxels of a patch therefore enter two global percentiles and nothing else), cut
into a crop that is centred on a random foreground voxel most of the time (so that a block mask always finds
foreground), rotated in plane by up to ±10° like Stage II (the image and its validity; the coordinate grids stay),
and turned into two views by a bias field, a gamma and Gaussian noise. No blur (a blur is a convolution and would carry
hidden voxels into visible ones), no mirroring (sided anatomy). Invalid voxels stay at -1 in both views. The patient id
is an integer hash of source:patient for the contrastive loss; the sampler weights (source_weights) balance the
patients within a source and cap HCP's share of the exposures."""
import zlib
from collections import Counter

import nibabel as nib
import numpy as np
import torch
from torch.utils.data import Dataset

from anatobind.anatomy.simulate import bias_field
from anatobind.aur.crops import CROP, PATCH, coordinates_mm, crop_window, extract, local_coordinates, normalise, rotate_inplane, spacing_zyx, to_zyx
from anatobind.aur.dataset import ROTATION_DEG, canonical, spacing_of
from anatobind.aur.labels import SEQ_INDEX
from anatobind.aur.ssl.samples import HCP_EXPOSURE_CAP

FOREGROUND_SHARE = 0.8          # share of the crops centred on a random foreground voxel (the rest uniform)
SOURCES = ("pdgm", "bmsr", "isles", "sibbms", "hcp")
SOURCE_INDEX = {s: i for i, s in enumerate(SOURCES)}


def patient_id(source, patient):
    """A stable 32-bit id of source:patient (the same patient under two sources stays two ids on purpose)."""
    return int(zlib.crc32(f"{source}:{patient}".encode()))


def load_image(row):
    """One sample row -> the normalised (z, y, x) image with its geometry; no label is read."""
    img = canonical(nib.load(row["image"]))
    affine = np.asarray(img.affine, dtype=np.float64)
    spacing = spacing_zyx(spacing_of(img))
    image = normalise(to_zyx(np.asarray(img.dataobj).astype(np.float32)))
    return {"image": image, "affine": affine, "spacing": spacing, "case": row["case"], "source": row["source"],
            "patient": patient_id(row["source"], row["patient"]), "seq": SEQ_INDEX[row["sequence"]]}


def intensity_view(image, rng, p=0.8):
    """A bias field, a gamma and Gaussian noise, each with probability p, on an image in [-1, 1]; invalid voxels
    (exactly -1) keep their value. Pointwise operations only: no blur."""
    img = np.asarray(image, np.float32).copy()
    fg = img > -1.0
    if rng.random() < p:
        field = bias_field(img.shape, rng.uniform(-1, 1, 6), amp=0.2).astype(np.float32)
        img = np.where(fg, (img + 1.0) * field - 1.0, img)
    if rng.random() < p:
        g = float(rng.uniform(0.7, 1.4))
        img = np.where(fg, np.power(np.clip((img + 1.0) / 2.0, 0.0, 1.0), g) * 2.0 - 1.0, img)
    if rng.random() < p:
        img = np.where(fg, img + rng.normal(0.0, float(rng.uniform(0.0, 0.05)), img.shape).astype(np.float32), img)
    return img.astype(np.float32)


def make_ssl_crop(vol, rng, crop=CROP, do_rotate=True, foreground_share=FOREGROUND_SHARE):
    """One source crop of a loaded volume with two intensity views, as tensors."""
    if any(c % p for c, p in zip(crop, PATCH)):
        raise ValueError(f"crop {crop} is not a multiple of the patch {PATCH}")
    image = vol["image"]
    centre = None
    if rng.random() < foreground_share:
        fg = np.argwhere(image > -1.0)
        if len(fg):
            centre = fg[int(rng.integers(0, len(fg)))]
    window = crop_window(image.shape, crop, rng, centre)
    img, valid = extract(image, window, fill=-1.0)
    if not valid.any():
        raise ValueError(f"{vol['case']}: the crop window {window} holds no voxel of the volume {image.shape}")
    coords = coordinates_mm(window, affine=vol["affine"])
    local = local_coordinates(window, image.shape)
    if do_rotate:
        angle = float(rng.uniform(-ROTATION_DEG, ROTATION_DEG))
        img, valid = rotate_inplane([img, valid.astype(np.uint8)], angle, [1, 0])
        valid = valid.astype(bool)
        img = np.where(valid, img, -1.0).astype(np.float32)
    view1, view2 = intensity_view(img, rng), intensity_view(img, rng)
    return {"view1": torch.from_numpy(np.ascontiguousarray(view1))[None], "view2": torch.from_numpy(np.ascontiguousarray(view2))[None],
            "valid": torch.from_numpy(valid.astype(np.float32)),
            "coords": torch.from_numpy(np.ascontiguousarray(coords, dtype=np.float32)),
            "local": torch.from_numpy(np.ascontiguousarray(local, dtype=np.float32)),
            "spacing": torch.tensor(vol["spacing"], dtype=torch.float32), "patient": torch.tensor(vol["patient"], dtype=torch.int64),
            "seq": torch.tensor(vol["seq"]), "source": torch.tensor(SOURCE_INDEX[vol["source"]]), "case": vol["case"]}


class SSLDataset(Dataset):
    def __init__(self, rows, crop=CROP, crops_per_volume=2, do_rotate=True, seed=0):
        self.rows, self.crop, self.k, self.do_rotate, self.seed = list(rows), tuple(crop), crops_per_volume, do_rotate, seed
        self.epoch = 0

    def set_epoch(self, epoch):
        self.epoch = int(epoch)

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        rng = np.random.default_rng([self.seed, self.epoch, i])
        vol = load_image(self.rows[i])
        return [make_ssl_crop(vol, rng, self.crop, self.do_rotate) for _ in range(self.k)]


STACKED = ("view1", "view2", "valid", "coords", "local", "spacing", "patient", "seq", "source")


def collate_ssl(items):
    crops = [c for item in items for c in item]
    out = {k: torch.stack([c[k] for c in crops]) for k in STACKED}
    out["case"] = [c["case"] for c in crops]
    return out


def source_weights(rows, hcp_cap=HCP_EXPOSURE_CAP):
    """Per-row sampling weights (sum 1): every source gets its natural share of the rows, except HCP, whose share is
    capped at hcp_cap (the excess goes to the others in proportion); within a source every patient weighs the same
    and a patient's rows share its weight. Returns (weights (N,) float64, {source: share})."""
    n = len(rows)
    by_source = Counter(r["source"] for r in rows)
    share = {s: c / n for s, c in by_source.items()}
    if share.get("hcp", 0.0) > hcp_cap:
        rest = 1.0 - share["hcp"]
        share = {s: (hcp_cap if s == "hcp" else v * (1.0 - hcp_cap) / rest) for s, v in share.items()}
    patients = {s: len({r["patient"] for r in rows if r["source"] == s}) for s in by_source}
    rows_of_patient = Counter((r["source"], r["patient"]) for r in rows)
    w = np.array([share[r["source"]] / patients[r["source"]] / rows_of_patient[(r["source"], r["patient"])] for r in rows], np.float64)
    return w / w.sum(), share
