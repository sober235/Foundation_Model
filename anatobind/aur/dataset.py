"""Volumes -> training crops with targets (spec §3.2, §6; SSL-first plan 2026-10-09 §2.2, §4).

Every volume is first brought to RAS by axis flips / permutations (nibabel's closest canonical): the sources store LPS
(PDGM), LAS (ISLES, 9 BMSR cases) and RAS (SibBMS, BMSR) volumes, and the sided labels must lie on one side of the
array, or the sources would mirror each other (an implicit mirroring; S4 lesson A17). The image, the anatomy map and
the lesion map must share one grid: the same shape, the same affine (within 1e-3) and spatial units in mm (or the
unknown units the derived maps ship); anything else is refused, nothing is resampled or flipped silently. One dataset
item is one sample row (a volume of one sequence) and yields `crops_per_volume` crops of it: the volume is read once,
normalised, its entity map and lesion instances built once, then each crop is cut, rotated in-plane,
intensity-augmented and labelled. Half of the crops of a volume with instances are centred on a random instance voxel.
Targets per crop: the entity map, the entity presence, the instance map (ids renumbered 1..n within the crop), the
point weight (0 on components under the volume floor and outside the volume), the entity-ignore mask (every lesion
voxel of the case), the host / host probabilities / hard negatives of the crop's instances, the sequence type and the
A / U / R supervision flags. An instance whose part inside the crop is under the volume floor is not an instance of the
crop: its voxels join the small mask (no loss). The host targets are computed on the crop after its rotation (the
overlap of the in-crop part of each instance with the in-crop anatomy), not copied from the whole volume: the relation
head sees the crop, so its target is what the crop shows. The coordinate grids (physical mm and local) are not rotated
with the image: the rotation changes the anatomy's pose in the scanner frame, which the model must tolerate (plan §2.2
as amended on 2026-10-09)."""
import nibabel as nib
import numpy as np
import torch
from torch.utils.data import Dataset

from anatobind.aur.crops import (CROP, PATCH, augment, coordinates_mm, crop_window, extract, local_coordinates, normalise,
                                 rotate_inplane, spacing_zyx, to_zyx)
from anatobind.aur.labels import N_ENTITIES, SEQ_INDEX, entity_map, entity_to_synthseg
from anatobind.aur.targets import host_targets, lesion_instances
from anatobind.eval.lesion_components import min_voxels_for

ROTATION_DEG = 10.0
AFFINE_ATOL = 1e-3


def canonical(img):
    """The image reoriented to RAS by axis flips / permutations only (no resampling): the same anatomy lies on the same
    side of the array whatever the source stored."""
    return nib.as_closest_canonical(img)


def spacing_of(img):
    """Voxel spacing (x, y, z) in mm from the affine's column norms (right after a reorientation too)."""
    a = np.asarray(img.affine, dtype=np.float64)[:3, :3]
    return tuple(float(v) for v in np.sqrt((a ** 2).sum(0)))


def verify_grid(image, labels, name):
    """Refuse a label map that is not on the image's grid: shape, affine (within AFFINE_ATOL) and spatial units."""
    if image.shape[:3] != labels.shape[:3]:
        raise ValueError(f"{name}: image {image.shape[:3]} and label map {labels.shape[:3]} differ in shape")
    if not np.allclose(image.affine, labels.affine, atol=AFFINE_ATOL, rtol=1e-5):
        raise ValueError(f"{name}: NIfTI affine mismatch between the image and the label map")
    units = (image.header.get_xyzt_units()[0], labels.header.get_xyzt_units()[0])
    if any(unit not in ("mm", "unknown") for unit in units):
        raise ValueError(f"{name}: spatial units {units} are not mm")


def load_volume(row):
    """Read one sample row into (z, y, x) arrays with its targets; every file is reoriented to RAS first and checked
    against the image's grid."""
    img_i, seg_i = canonical(nib.load(row["image"])), canonical(nib.load(row["anatomy"]))
    verify_grid(img_i, seg_i, f"{row['case']} {row['sequence']} anatomy")
    affine = np.asarray(img_i.affine, dtype=np.float64)
    spacing = spacing_zyx(spacing_of(img_i))
    voxel_mm3 = abs(float(np.linalg.det(affine[:3, :3])))
    if not np.isfinite(voxel_mm3) or voxel_mm3 <= 0:
        raise ValueError(f"{row['case']} {row['sequence']}: the affine has no spatial volume")
    image = normalise(to_zyx(np.asarray(img_i.dataobj).astype(np.float32)))
    seg = to_zyx(np.asarray(seg_i.dataobj).astype(np.int16))
    entity = entity_map(seg)
    if row["lesion"]:
        les_i = canonical(nib.load(row["lesion"]))
        verify_grid(img_i, les_i, f"{row['case']} {row['sequence']} lesion")
        les = to_zyx(np.asarray(les_i.dataobj).astype(np.int16))
        inst, small = lesion_instances(les, row["u_values"], voxel_mm3)
        a_ignore = np.isin(les, list(row["a_ignore_values"]))
    else:
        inst, small, a_ignore = np.zeros(seg.shape, np.int32), np.zeros(seg.shape, bool), np.zeros(seg.shape, bool)
    return {"image": image, "entity": entity, "instance": inst, "small": small, "a_ignore": a_ignore,
            "hosts": host_targets(inst, seg, spacing), "spacing": spacing, "affine": affine, "voxel_mm3": voxel_mm3,
            "seq": SEQ_INDEX[row["sequence"]], "a_supervised": bool(row.get("a_supervised", True)),
            "r_supervised": bool(row.get("r_supervised", True)), "u_supervised": bool(row["u_supervised"]),
            "case": row["case"]}


def make_crop(vol, rng, crop=CROP, do_augment=True, lesion_centred=False):
    """One crop of a loaded volume as tensors."""
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
        raise ValueError(f"{vol['case']}: the crop window {window} holds no voxel of the volume {inst.shape}")
    entity, _ = extract(vol["entity"], window, fill=0)
    instance, _ = extract(inst, window, fill=0)
    small, _ = extract(vol["small"], window, fill=False)
    a_ignore, _ = extract(vol["a_ignore"], window, fill=False)
    affine = vol.get("affine")
    coords = coordinates_mm(window, affine=affine) if affine is not None else coordinates_mm(window, vol["spacing"])
    local = local_coordinates(window, inst.shape)
    if do_augment:
        angle = float(rng.uniform(-ROTATION_DEG, ROTATION_DEG))
        image, valid, entity, instance, small, a_ignore = rotate_inplane(
            [image, valid.astype(np.uint8), entity, instance, small.astype(np.uint8), a_ignore.astype(np.uint8)], angle, [1, 0, 0, 0, 0, 0])
        valid, small, a_ignore = valid.astype(bool), small.astype(bool), a_ignore.astype(bool)
        image = np.where(valid, image, -1.0).astype(np.float32)
        image = augment(image, rng)
    renumbered, small, ids = crop_instances(instance, small, vol.get("voxel_mm3", float(np.prod(vol["spacing"]))))
    hosts = host_targets(renumbered.astype(np.int32), entity_to_synthseg(entity), vol["spacing"])
    entity_present = np.isin(np.arange(1, N_ENTITIES + 1), np.unique(entity))          # the entity has voxels in the crop
    return {"image": torch.from_numpy(np.ascontiguousarray(image))[None], "valid": torch.from_numpy(valid.astype(np.float32)),
            "coords": torch.from_numpy(np.ascontiguousarray(coords, dtype=np.float32)),
            "local": torch.from_numpy(np.ascontiguousarray(local, dtype=np.float32)),
            "entity": torch.from_numpy(entity.astype(np.int64)), "entity_present": torch.from_numpy(entity_present), "instance": torch.from_numpy(renumbered),
            "point_weight": torch.from_numpy((~small & valid).astype(np.float32)), "a_ignore": torch.from_numpy(a_ignore),
            "host": torch.from_numpy(hosts["host"]), "host_probs": torch.from_numpy(hosts["probs"]),
            "negatives": torch.from_numpy(hosts["negatives"]), "seq": torch.tensor(vol["seq"]),
            "a_supervised": torch.tensor(bool(vol.get("a_supervised", True))), "r_supervised": torch.tensor(bool(vol.get("r_supervised", True))),
            "u_supervised": torch.tensor(vol["u_supervised"]), "n_instances": len(ids), "case": vol["case"]}


def crop_instances(instance, small, voxel_mm3):
    """(instance map renumbered 1..n over the instances whose in-crop part reaches the volume floor, the small mask with
    the slivers added, the kept original ids in order)."""
    floor = min_voxels_for(voxel_mm3)
    ids = [int(k) for k in np.unique(instance) if k > 0]
    kept = [k for k in ids if int((instance == k).sum()) >= floor]
    renumbered = np.zeros(instance.shape, np.int64)
    for new, old in enumerate(kept, start=1):
        renumbered[instance == old] = new
    small = small | np.isin(instance, [k for k in ids if k not in kept])
    return renumbered, small, kept


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


STACKED = ("image", "valid", "coords", "local", "entity", "entity_present", "instance", "point_weight", "a_ignore", "seq",
           "a_supervised", "r_supervised", "u_supervised")
LISTED = ("host", "host_probs", "negatives", "n_instances", "case")


def collate(items):
    """A list of crop lists -> one batch dict: fixed-size tensors stacked, per-crop targets kept as lists."""
    crops = [c for item in items for c in item]
    out = {k: torch.stack([c[k] for c in crops]) for k in STACKED}
    for k in LISTED:
        out[k] = [c[k] for c in crops]
    return out


def event_targets_at_points(instance_pts, n_instances):
    """instance_pts (B, P) crop-local instance ids at the sampled points -> list of (N_b, P) float targets."""
    out = []
    for b, n in enumerate(n_instances):
        ids = torch.arange(1, n + 1, device=instance_pts.device)
        out.append((instance_pts[b][None, :] == ids[:, None]).float())
    return out
