"""Volumes -> training crops with targets (spec §3.2, §6).

One dataset item is one sample row (a volume of one sequence) and yields `crops_per_volume` crops of it: the volume is
read once, normalised, its entity map, lesion instances and host targets built once, then each crop is cut, rotated
in-plane, intensity-augmented and labelled. Half of the crops of a volume with instances are centred on a random
instance voxel. Targets per crop: the entity map, the instance map (ids renumbered 1..n within the crop), the point
weight (0 on components under the volume floor), the entity-ignore mask (every lesion voxel of the case), the host /
host probabilities / hard negatives of the crop's instances, the sequence type and the U supervision flag."""
import nibabel as nib
import numpy as np
import torch
from torch.utils.data import Dataset

from anatobind.aur.crops import (CROP, PATCH, augment, coordinates_mm, crop_window, extract, local_coordinates, normalise,
                                 rotate_inplane, spacing_zyx, to_zyx)
from anatobind.aur.labels import SEQ_INDEX, entity_map, entity_to_synthseg
from anatobind.aur.targets import host_targets, lesion_instances
from anatobind.eval.lesion_components import min_voxels_for

ROTATION_DEG = 10.0


def _verify_grid(image, labels, name):
    if image.shape[:3] != labels.shape[:3]:
        raise ValueError(f"{name}: image/label shape mismatch")
    if not np.allclose(image.affine, labels.affine, atol=1e-3, rtol=1e-5):
        raise ValueError(f"{name}: NIfTI affine mismatch")
    units = (image.header.get_xyzt_units()[0], labels.header.get_xyzt_units()[0])
    if any(unit not in ("mm", "unknown") for unit in units):
        raise ValueError(f"{name}: non-mm spatial units")


def load_volume(row):
    image_nii, anatomy_nii = nib.load(row["image"]), nib.load(row["anatomy"])
    _verify_grid(image_nii, anatomy_nii, f"{row['case']} anatomy")
    affine = np.asarray(image_nii.affine, np.float64)
    spacing = spacing_zyx(image_nii.header.get_zooms()[:3])
    voxel_mm3 = abs(float(np.linalg.det(affine[:3, :3])))
    if not np.isfinite(voxel_mm3) or voxel_mm3 <= 0:
        raise ValueError(f"{row['case']}: invalid spatial affine")
    image = normalise(to_zyx(np.asarray(image_nii.dataobj).astype(np.float32)))
    seg = to_zyx(np.asarray(anatomy_nii.dataobj).astype(np.int16))
    entity = entity_map(seg)
    if row["lesion"]:
        lesion_nii = nib.load(row["lesion"])
        _verify_grid(image_nii, lesion_nii, f"{row['case']} lesion")
        lesion = to_zyx(np.asarray(lesion_nii.dataobj).astype(np.int16))
        inst, small = lesion_instances(lesion, row["u_values"], voxel_mm3)
        a_ignore = np.isin(lesion, list(row["a_ignore_values"]))
    else:
        inst = np.zeros(seg.shape, np.int32)
        small = np.zeros(seg.shape, bool)
        a_ignore = np.zeros(seg.shape, bool)
    return {"image": image, "entity": entity, "instance": inst, "small": small,
            "a_ignore": a_ignore, "spacing": spacing, "affine": affine, "voxel_mm3": voxel_mm3,
            "seq": SEQ_INDEX[row["sequence"]], "a_supervised": bool(row.get("a_supervised", True)),
            "r_supervised": bool(row.get("r_supervised", True)),
            "u_supervised": bool(row["u_supervised"]), "case": row["case"]}


def crop_instances(inst, small, voxel_mm3):
    floor = min_voxels_for(voxel_mm3)
    ids = [int(k) for k in np.unique(inst) if k > 0]
    kept = [k for k in ids if int((inst == k).sum()) >= floor]
    out = np.zeros(inst.shape, np.int64)
    for j, k in enumerate(kept, 1):
        out[inst == k] = j
    small = small | np.isin(inst, [k for k in ids if k not in kept])
    return out, small, kept


def make_crop(vol, rng, crop=CROP, do_augment=True, lesion_centred=False):
    """Generate final augmented crop and compute local relation targets on it."""
    if any(c % p for c, p in zip(crop, PATCH)):
        raise ValueError(f"crop {crop} is not a multiple of the patch {PATCH}")
    inst = vol["instance"]
    centre = None
    if lesion_centred and inst.max() > 0:
        k = int(rng.integers(1, inst.max() + 1))
        vox = np.argwhere(inst == k)
        centre = vox[int(rng.integers(0, len(vox)))]
    window = crop_window(inst.shape, crop, rng, centre)
    image, valid = extract(vol["image"], window, fill=-1.0)
    if not valid.any():
        raise ValueError(f"{vol['case']}: empty crop")
    entity, _ = extract(vol["entity"], window, fill=0)
    instance, _ = extract(inst, window, fill=0)
    small, _ = extract(vol["small"], window, fill=False)
    a_ignore, _ = extract(vol["a_ignore"], window, fill=False)
    coords = coordinates_mm(window, affine=vol["affine"]) if "affine" in vol else coordinates_mm(window, vol["spacing"])
    local = local_coordinates(window, inst.shape)
    if do_augment:
        angle = float(rng.uniform(-ROTATION_DEG, ROTATION_DEG))
        rotated = rotate_inplane([image, valid.astype(np.uint8), entity, instance,
                                  small.astype(np.uint8), a_ignore.astype(np.uint8),
                                  *coords, *local], angle, [1, 0, 0, 0, 0, 0] + [1] * 6)
        image, valid, entity, instance, small, a_ignore = rotated[:6]
        coords = np.stack(rotated[6:9], 0)
        local = np.stack(rotated[9:12], 0)
        valid, small, a_ignore = valid.astype(bool), small.astype(bool), a_ignore.astype(bool)
        image = augment(np.where(valid, image, -1.0).astype(np.float32), rng)
    renumbered, small, ids = crop_instances(instance, small, vol.get("voxel_mm3", float(np.prod(vol["spacing"]))))
    relation_targets = host_targets(renumbered.astype(np.int32), entity_to_synthseg(entity), vol["spacing"])
    return {"image": torch.from_numpy(np.ascontiguousarray(image))[None], "valid": torch.from_numpy(valid.astype(np.float32)),
            "coords": torch.from_numpy(np.ascontiguousarray(coords, np.float32)),
            "local": torch.from_numpy(np.ascontiguousarray(local, np.float32)),
            "entity": torch.from_numpy(entity.astype(np.int64)), "instance": torch.from_numpy(renumbered),
            "point_weight": torch.from_numpy((~small & valid).astype(np.float32)),
            "a_ignore": torch.from_numpy(a_ignore),
            "host": torch.from_numpy(relation_targets["host"]),
            "host_probs": torch.from_numpy(relation_targets["probs"]),
            "negatives": torch.from_numpy(relation_targets["negatives"]),
            "seq": torch.tensor(vol["seq"]),
            "a_supervised": torch.tensor(vol.get("a_supervised", True)),
            "r_supervised": torch.tensor(vol.get("r_supervised", True)),
            "u_supervised": torch.tensor(vol["u_supervised"]),
            "n_instances": len(ids), "case": vol["case"]}


class AURDataset(Dataset):
    def __init__(self, rows, crop=CROP, crops_per_volume=2, do_augment=True, lesion_share=0.5, seed=0):
        self.rows, self.crop, self.k, self.do_augment, self.lesion_share, self.seed = list(rows), tuple(crop), crops_per_volume, do_augment, lesion_share, seed
        self.epoch = 0

    def set_epoch(self, epoch):
        self.epoch = int(epoch)

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        rng = np.random.default_rng([self.seed, self.epoch, i])
        vol = load_volume(self.rows[i])
        return [make_crop(vol, rng, self.crop, self.do_augment, lesion_centred=rng.random() < self.lesion_share) for _ in range(self.k)]


def collate(items):
    """A list of crop lists -> one batch dict: fixed-size tensors stacked, per-crop targets kept as lists."""
    crops = [c for item in items for c in item]
    out = {k: torch.stack([c[k] for c in crops]) for k in ("image", "valid", "coords", "local", "entity", "instance", "point_weight", "a_ignore", "seq", "a_supervised", "r_supervised", "u_supervised")}
    for k in ("host", "host_probs", "negatives", "n_instances", "case"):
        out[k] = [c[k] for c in crops]
    return out


def event_targets_at_points(instance_pts, n_instances):
    """instance_pts (B, P) crop-local instance ids at the sampled points -> list of (N_b, P) float targets."""
    out = []
    for b, n in enumerate(n_instances):
        ids = torch.arange(1, n + 1, device=instance_pts.device)
        out.append((instance_pts[b][None, :] == ids[:, None]).float())
    return out
