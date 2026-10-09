"""Volumes -> training crops with targets (spec §3.2, §6).

Every volume is first brought to RAS by axis flips / permutations (nibabel's closest canonical): the sources store LPS
(PDGM), LAS (ISLES, 9 BMSR cases) and RAS (SibBMS, BMSR) volumes, and the sided labels must lie on one side of the
array, or the sources would mirror each other (an implicit mirroring; S4 lesson A17). One dataset item is one sample row (a volume of one sequence) and yields `crops_per_volume` crops of it: the volume is
read once, normalised, its entity map, lesion instances and host targets built once, then each crop is cut, rotated
in-plane, intensity-augmented and labelled. Half of the crops of a volume with instances are centred on a random
instance voxel. Targets per crop: the entity map, the instance map (ids renumbered 1..n within the crop), the point
weight (0 on components under the volume floor), the entity-ignore mask (every lesion voxel of the case), the host /
host probabilities / hard negatives of the crop's instances, the sequence type and the U supervision flag. An
instance whose part inside the crop is under the volume floor is not an instance of the crop: its voxels join the
small mask (no loss). The host targets of an instance cut by the crop are those of the whole instance."""
import nibabel as nib
import numpy as np
import torch
from torch.utils.data import Dataset

from anatobind.aur.crops import (CROP, PATCH, augment, coordinates_mm, crop_window, extract, local_coordinates, normalise,
                                 rotate_inplane, spacing_zyx, to_zyx)
from anatobind.aur.labels import N_ENTITIES, SEQ_INDEX, entity_map
from anatobind.aur.targets import host_targets, lesion_instances
from anatobind.eval.lesion_components import min_voxels_for

ROTATION_DEG = 10.0


def canonical(img):
    """The image reoriented to RAS by axis flips / permutations only (no resampling): the same anatomy lies on the same
    side of the array whatever the source stored."""
    return nib.as_closest_canonical(img)


def spacing_of(img):
    """Voxel spacing (x, y, z) in mm from the affine's column norms (right after a reorientation too)."""
    a = np.asarray(img.affine, dtype=np.float64)[:3, :3]
    return tuple(float(v) for v in np.sqrt((a ** 2).sum(0)))


def load_volume(row):
    """Read one sample row into (z, y, x) arrays with its targets; every file is reoriented to RAS first."""
    img_i, seg_i = canonical(nib.load(row["image"])), canonical(nib.load(row["anatomy"]))
    if img_i.shape[:3] != seg_i.shape[:3]:
        raise ValueError(f"{row['case']} {row['sequence']}: image {img_i.shape} and anatomy {seg_i.shape} differ")
    spacing = spacing_zyx(spacing_of(img_i))
    image = normalise(to_zyx(np.asarray(img_i.dataobj).astype(np.float32)))
    seg = to_zyx(np.asarray(seg_i.dataobj).astype(np.int16))
    entity = entity_map(seg)
    if row["lesion"]:
        les = to_zyx(np.asarray(canonical(nib.load(row["lesion"])).dataobj).astype(np.int16))
        if les.shape != seg.shape:
            raise ValueError(f"{row['case']}: lesion map {les.shape} and anatomy {seg.shape} differ")
        inst, small = lesion_instances(les, row["u_values"], float(np.prod(spacing)))
        a_ignore = np.isin(les, list(row["a_ignore_values"]))
    else:
        inst, small, a_ignore = np.zeros(seg.shape, np.int32), np.zeros(seg.shape, bool), np.zeros(seg.shape, bool)
    return {"image": image, "entity": entity, "instance": inst, "small": small, "a_ignore": a_ignore,
            "hosts": host_targets(inst, seg, spacing), "spacing": spacing, "seq": SEQ_INDEX[row["sequence"]],
            "u_supervised": bool(row["u_supervised"]), "case": row["case"]}


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
    coords, local = coordinates_mm(window, vol["spacing"]), local_coordinates(window, inst.shape)
    if do_augment:
        angle = float(rng.uniform(-ROTATION_DEG, ROTATION_DEG))
        image, valid, entity, instance, small, a_ignore = rotate_inplane(
            [image, valid.astype(np.uint8), entity, instance, small.astype(np.uint8), a_ignore.astype(np.uint8)], angle, [1, 0, 0, 0, 0, 0])
        valid, small, a_ignore = valid.astype(bool), small.astype(bool), a_ignore.astype(bool)
        image = np.where(valid, image, -1.0).astype(np.float32)
        image = augment(image, rng)
    renumbered, small, ids = crop_instances(instance, small, float(np.prod(vol["spacing"])))
    sel = np.array(ids, np.int64) - 1
    hosts = vol["hosts"]
    entity_present = np.isin(np.arange(1, N_ENTITIES + 1), np.unique(entity))          # the entity has voxels in the crop
    return {"image": torch.from_numpy(np.ascontiguousarray(image))[None], "valid": torch.from_numpy(valid.astype(np.float32)),
            "coords": torch.from_numpy(coords), "local": torch.from_numpy(local),
            "entity": torch.from_numpy(entity.astype(np.int64)), "entity_present": torch.from_numpy(entity_present), "instance": torch.from_numpy(renumbered),
            "point_weight": torch.from_numpy((~small).astype(np.float32)), "a_ignore": torch.from_numpy(a_ignore),
            "host": torch.from_numpy(hosts["host"][sel]), "host_probs": torch.from_numpy(hosts["probs"][sel]),
            "negatives": torch.from_numpy(hosts["negatives"][sel]), "seq": torch.tensor(vol["seq"]),
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


def collate(items):
    """A list of crop lists -> one batch dict: fixed-size tensors stacked, per-crop targets kept as lists."""
    crops = [c for item in items for c in item]
    out = {k: torch.stack([c[k] for c in crops]) for k in ("image", "valid", "coords", "local", "entity", "entity_present", "instance", "point_weight", "a_ignore", "seq", "u_supervised")}
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
