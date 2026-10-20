"""Image-only SSL observations for homogeneous 2D or 3D batches.

No supervision or lesion-centred selection enters this module. Independent
planes are normalized from the complete raw plane before any crop is cut.
The 3D crop delegates to the established numerical path without changing RNG.
"""
import nibabel as nib
import numpy as np
import torch
from torch.utils.data import Dataset

from anatobind.aur.crops import (PATCH, coordinates_mm, crop_window, extract, local_coordinates, normalise,
                                 rotate_inplane, spacing_zyx, to_zyx)
from anatobind.aur.dataset import ROTATION_DEG, canonical, spacing_of
from anatobind.aur.labels import SEQ_INDEX
from anatobind.aur.ssl import dataset as legacy


def load_unified_ssl_image(row):
    """Read only image and source/patient/sequence metadata, never label paths."""
    image = canonical(nib.load(row["image"]))
    raw = to_zyx(np.asarray(image.dataobj).astype(np.float32))
    affine = np.asarray(image.affine, dtype=np.float64)
    spacing = spacing_zyx(spacing_of(image))
    if (raw.ndim != 3 or not all(raw.shape) or not np.isfinite(raw).all()
            or affine.shape != (4, 4) or not np.isfinite(affine).all()
            or not np.isfinite(np.linalg.det(affine[:3, :3])) or abs(np.linalg.det(affine[:3, :3])) <= 0
            or not np.isfinite(spacing).all() or min(spacing) <= 0):
        raise ValueError("SSL image requires finite 3D intensities and nonsingular physical geometry")
    if image.header.get_xyzt_units()[0] not in ("mm", "unknown"):
        raise ValueError("SSL image spatial units must be millimetres")
    return {"image": normalise(raw), "raw_image": raw, "affine": affine, "spacing": spacing,
            "case": row["case"], "source": row["source"],
            "patient": legacy.patient_id(row["source"], row["patient"]), "seq": SEQ_INDEX[row["sequence"]]}


def _validate_crop(input_mode, crop, foreground_share):
    if input_mode not in ("2d", "3d"):
        raise ValueError("SSL input mode must be 2d or 3d")
    patch = (1, *PATCH[1:]) if input_mode == "2d" else PATCH
    if (len(crop) != 3 or any(not isinstance(c, (int, np.integer)) or c <= 0 for c in crop)
            or any(c % p for c, p in zip(crop, patch)) or (input_mode == "2d" and crop[0] != 1)):
        raise ValueError("SSL crop must use positive mode-specific patch multiples and 2d depth 1")
    if not np.isfinite(foreground_share) or not 0 <= foreground_share <= 1:
        raise ValueError("SSL foreground share must be between zero and one")


def make_unified_ssl_crop(vol, rng, input_mode, crop, do_rotate=True,
                          foreground_share=legacy.FOREGROUND_SHARE, plane_index=None):
    """Return two intensity views, validity, physical coordinates and SSL metadata.

    A 2D plane has three patient-space coordinate channels and local normal
    coordinate zero. Foreground sampling uses image intensity only. A fixed
    plane index is available for frozen representation probes.
    """
    crop = tuple(crop)
    _validate_crop(input_mode, crop, foreground_share)
    if input_mode == "3d":
        if plane_index is not None:
            raise ValueError("a fixed plane index requires 2d input mode")
        return {**legacy.make_ssl_crop(vol, rng, crop, do_rotate, foreground_share), "input_mode": input_mode}
    raw = vol.get("raw_image")
    if raw is None or np.ndim(raw) != 3 or not np.isfinite(raw).all():
        raise ValueError("2d SSL requires the finite unclipped raw image")
    if plane_index is not None and (not isinstance(plane_index, (int, np.integer)) or not 0 <= plane_index < raw.shape[0]):
        raise ValueError("SSL plane index is outside source image")
    centre = None
    foreground = rng.random() < foreground_share
    if plane_index is None:
        if foreground:
            indices = np.flatnonzero(raw != 0)
            if indices.size:
                centre = np.unravel_index(indices[int(rng.integers(indices.size))], raw.shape)
                plane_index = int(centre[0])
        if plane_index is None:
            plane_index = int(rng.integers(raw.shape[0]))
    plane_index = int(plane_index)
    plane = normalise(raw[plane_index:plane_index + 1])
    if centre is None and foreground:
        indices = np.flatnonzero(plane > -1.)
        if indices.size:
            centre = np.unravel_index(indices[int(rng.integers(indices.size))], plane.shape)
    plane_centre = (0, *centre[1:]) if centre is not None else None
    window = crop_window(plane.shape, crop, rng, plane_centre)
    source_window = [(plane_index, plane_index + 1), *window[1:]]
    image, valid = extract(plane, window, fill=-1.)
    coords = coordinates_mm(source_window, affine=vol["affine"])
    local = local_coordinates(window, plane.shape)
    if do_rotate:
        angle = float(rng.uniform(-ROTATION_DEG, ROTATION_DEG))
        image, valid = rotate_inplane([image, valid.astype(np.uint8)], angle, [1, 0])
        valid = valid.astype(bool)
    image = np.where(valid, image, -1.).astype(np.float32)
    view1, view2 = legacy.intensity_view(image, rng), legacy.intensity_view(image, rng)
    return {"view1": torch.from_numpy(np.ascontiguousarray(view1))[None],
            "view2": torch.from_numpy(np.ascontiguousarray(view2))[None],
            "valid": torch.from_numpy(valid.astype(np.float32)),
            "coords": torch.from_numpy(np.ascontiguousarray(coords, dtype=np.float32)),
            "local": torch.from_numpy(np.ascontiguousarray(local, dtype=np.float32)),
            "spacing": torch.tensor(vol["spacing"], dtype=torch.float32),
            "patient": torch.tensor(vol["patient"], dtype=torch.int64), "seq": torch.tensor(vol["seq"]),
            "source": torch.tensor(legacy.SOURCE_INDEX[vol["source"]]), "case": vol["case"], "input_mode": input_mode}


class UnifiedSSLDataset(Dataset):
    """Deterministic source crops; held-out validation requires an explicit split."""
    def __init__(self, rows, input_mode, crop, crops_per_volume=2, do_rotate=True, seed=0, ssl_split="train"):
        self.rows, self.input_mode, self.crop = list(rows), input_mode, tuple(crop)
        _validate_crop(input_mode, self.crop, legacy.FOREGROUND_SHARE)
        if ssl_split not in ("train", "val"):
            raise ValueError("SSL dataset requires an explicit train/validation partition")
        for row in self.rows:
            if "role" in row:
                expected_role = "train_ssl" if ssl_split == "train" else "ssl_validation"
                valid = (row["role"] == expected_role and row.get("split") == ssl_split
                         and row.get("ssl_split", ssl_split) == ssl_split)
            else:
                # Historical SSL validation was held out from original TRAIN
                # patients and expressed only by ssl_split, not original split.
                valid = row.get("split", "train") == "train" and row.get("ssl_split", "train") == ssl_split
            if not valid:
                raise ValueError("SSL dataset rows conflict with the requested train/validation partition")
        if not isinstance(crops_per_volume, int) or crops_per_volume <= 0:
            raise ValueError("SSL crops per volume must be a positive integer")
        self.k, self.do_rotate, self.seed, self.epoch = crops_per_volume, do_rotate, seed, 0

    def set_epoch(self, epoch):
        self.epoch = int(epoch)

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        rng = np.random.default_rng([self.seed, self.epoch, index])
        vol = load_unified_ssl_image(self.rows[index])
        return [make_unified_ssl_crop(vol, rng, self.input_mode, self.crop, self.do_rotate) for _ in range(self.k)]


def collate_unified_ssl(items):
    """Stack a homogeneous mode batch; mixing observational dimensions is invalid."""
    crops = [crop for item in items for crop in item]
    modes = {crop["input_mode"] for crop in crops}
    if len(modes) != 1:
        raise ValueError("SSL collation requires exactly one input mode")
    return {**legacy.collate_ssl(items), "input_mode": modes.pop()}
