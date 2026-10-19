"""Private training views for one AUR model with real 2D and 3D observations.

The 3D path delegates numerical work to the existing dataset. A 2D view uses
the raw, complete source plane's intensity scale before cropping. Source 3D
lesion qualification precedes planar components/area qualification. Relation
targets retain their source identity; planar eligibility is opt-in, per source
instance, and means the caller verified that target for this observation.
"""
from collections.abc import Mapping
from copy import deepcopy

import nibabel as nib
import numpy as np
import torch
from scipy import ndimage

from anatobind.aur import dataset as legacy
from anatobind.aur.crops import (augment, coordinates_mm, crop_window, extract,
                                 local_coordinates, normalise, rotate_inplane, to_zyx)
from anatobind.aur.labels import N_ENTITIES, N_HOST_CLASSES


def load_unified_volume(row):
    """Load legacy targets plus unclipped intensities and private split provenance.

    The extra image read is deliberate: a globally clipped/normalised volume
    cannot recover the intensity distribution of an independent source plane.
    This wrapper does not alter the legacy loader or its 3D behaviour.
    """
    vol = legacy.load_volume(row)
    image = legacy.canonical(nib.load(row["image"]))
    raw = to_zyx(np.asarray(image.dataobj).astype(np.float32))
    if raw.shape != vol["image"].shape or not np.allclose(image.affine, vol["affine"],
                                                        atol=legacy.AFFINE_ATOL, rtol=1e-5):
        raise ValueError("unified raw image grid changed during loading")
    if not np.isfinite(raw).all():
        raise ValueError("unified raw image contains nonfinite intensities")
    vol["raw_image"] = raw
    for key in ("source", "patient", "split", "source_r_eligible", "source_r_targets"):
        if key in row:
            vol[key] = row[key]
    return vol


def _window(vol, rng, crop, lesion_centred):
    centre = None
    inst = vol["instance"]
    if lesion_centred and inst.max() > 0:
        k = int(rng.integers(1, inst.max() + 1))
        vox = np.argwhere(inst == k)
        centre = vox[int(rng.integers(0, len(vox)))]
    return crop_window(inst.shape, crop, rng, centre)


def _eligibility(vol, source_ids, default):
    source = vol.get("source_r_eligible")
    if source is None:
        values = [default] * len(source_ids)
    elif isinstance(source, Mapping):
        values = [bool(source.get(int(k), False)) for k in source_ids]
    else:
        source = np.asarray(source)
        if source.ndim != 1 or len(source) != int(vol["instance"].max()):
            raise ValueError("source relation eligibility must align with source instances")
        values = [bool(source[int(k) - 1]) for k in source_ids]
    return torch.tensor(values, dtype=torch.bool) & bool(vol.get("r_supervised", True))


def _source_targets(vol, source_ids, eligible):
    targets = vol.get("source_r_targets", vol["hosts"])
    n = int(vol["instance"].max())
    arrays = {"host": np.asarray(targets["host"]), "probs": np.asarray(targets["probs"]),
              "negatives": np.asarray(targets["negatives"])}
    if (arrays["host"].shape != (n,) or arrays["probs"].shape != (n, N_HOST_CLASSES)
            or arrays["negatives"].shape != (n, 2)):
        raise ValueError("source relation targets must align with source instances")
    indices = np.asarray(source_ids, dtype=np.int64) - 1
    selected = indices[eligible.numpy()]
    if (not all(np.isfinite(a).all() for a in arrays.values())
            or (arrays["probs"] < 0).any()
            or (selected.size and (
                (arrays["host"][selected] < 0).any()
                or (arrays["host"][selected] >= N_HOST_CLASSES).any()
                or (arrays["host"][selected] != np.floor(arrays["host"][selected])).any()
                or not np.allclose(arrays["probs"][selected].sum(-1), 1.0, atol=1e-5)))):
        raise ValueError("source relation targets are invalid for eligible instances")
    return {"host": torch.from_numpy(np.ascontiguousarray(arrays["host"][indices], dtype=np.int64)),
            "host_probs": torch.from_numpy(np.ascontiguousarray(arrays["probs"][indices], dtype=np.float32)),
            "negatives": torch.from_numpy(np.ascontiguousarray(arrays["negatives"][indices], dtype=np.int64))}


def _metadata(crop, vol, mode, source_ids, window):
    affine = np.asarray(vol["affine"], np.float64)
    area = float(np.linalg.norm(np.cross(affine[:3, 0], affine[:3, 1])))
    crop.update(input_mode=mode, spacing=torch.tensor(vol["spacing"], dtype=torch.float32),
                source_instance_ids=torch.tensor(source_ids, dtype=torch.int64),
                source_window=tuple(tuple(int(v) for v in w) for w in window),
                source_affine=affine.copy(), measurement_unit="mm2" if mode == "2d" else "mm3",
                pixel_area_mm2=area,
                voxel_volume_mm3=None if mode == "2d" else float(vol["voxel_mm3"]))
    return crop


def _three_dimensional(vol, rng, crop, do_augment, lesion_centred):
    # The shadow generator recovers provenance only; it cannot change the real
    # generator's state or the legacy numerical path.
    shadow = np.random.Generator(type(rng.bit_generator)())
    shadow.bit_generator.state = deepcopy(rng.bit_generator.state)
    out = legacy.make_crop(vol, rng, crop, do_augment, lesion_centred)
    window = _window(vol, shadow, crop, lesion_centred)
    source, _ = extract(vol["instance"], window, fill=0)
    if do_augment:
        angle = float(shadow.uniform(-legacy.ROTATION_DEG, legacy.ROTATION_DEG))
        source = rotate_inplane([source], angle, [0])[0]
    _, _, ids = legacy.crop_instances(source, np.zeros(source.shape, bool), vol["voxel_mm3"])
    if len(ids) != out["n_instances"]:
        raise ValueError("legacy instance provenance does not align with crop")
    out["r_eligible"] = _eligibility(vol, ids, default=True)
    return _metadata(out, vol, "3d", ids, window)


def make_unified_crop(vol, rng, input_mode, crop, do_augment=True, lesion_centred=False,
                      min_area_mm2=1.0, n_events=64, plane_index=None):
    """Return legacy batch fields plus mode, physical units and per-instance R eligibility.

    ``source_r_eligible`` is either a mapping keyed by one-based source instance
    ID or a sequence in source-target order. It is false by default in 2D. The
    caller may supply ``source_r_targets`` in the same array schema as ``hosts``.
    No plane-specific overlap heuristic can turn an unobserved host into no-host.
    Source fields and case identifiers remain private training metadata.
    """
    if input_mode not in ("2d", "3d"):
        raise ValueError("input mode must be 2d or 3d")
    crop = tuple(crop)
    if len(crop) != 3 or any(not isinstance(v, (int, np.integer)) or v <= 0 for v in crop):
        raise ValueError("crop must contain three positive integer dimensions")
    if not isinstance(n_events, (int, np.integer)) or n_events <= 0:
        raise ValueError("query capacity must be a positive integer")
    if input_mode == "3d":
        out = _three_dimensional(vol, rng, crop, do_augment, lesion_centred)
        if out["n_instances"] > n_events:
            raise ValueError("crop instances exceed query capacity")
        return out
    if crop[0] != 1 or any(c % p for c, p in zip(crop, (1, 4, 4))):
        raise ValueError("2d crop must have depth 1 and in-plane patch multiples")
    if not np.isfinite(min_area_mm2) or min_area_mm2 <= 0:
        raise ValueError("minimum area must be positive and finite")
    raw = vol.get("raw_image")
    if raw is None or np.shape(raw) != np.shape(vol["instance"]) or not np.isfinite(raw).all():
        raise ValueError("2d views require the finite unclipped raw image")
    affine = np.asarray(vol["affine"], np.float64)
    if affine.shape != (4, 4) or not np.isfinite(affine).all():
        raise ValueError("2d views require a finite source affine")
    pixel_area = float(np.linalg.norm(np.cross(affine[:3, 0], affine[:3, 1])))
    if not np.isfinite(pixel_area) or pixel_area <= 0:
        raise ValueError("2d source pixel area must be positive and finite")

    inst = vol["instance"]
    centre = None
    if plane_index is None and lesion_centred and inst.max() > 0:
        k = int(rng.integers(1, inst.max() + 1))
        positions = np.argwhere(inst == k)
        centre = positions[int(rng.integers(0, len(positions)))]
        plane_index = int(centre[0])
    elif plane_index is None:
        plane_index = int(rng.integers(0, inst.shape[0]))
    if (not isinstance(plane_index, (int, np.integer))
            or not 0 <= plane_index < inst.shape[0]):
        raise ValueError("plane index is outside source volume")
    plane_index = int(plane_index)
    plane_slice = slice(plane_index, plane_index + 1)
    plane_centre = None if centre is None else (0, *centre[1:])
    plane_window = crop_window((1, *inst.shape[1:]), crop, rng, plane_centre)
    source_window = [(plane_index, plane_index + 1), *plane_window[1:]]
    image, valid = extract(normalise(raw[plane_slice]), plane_window, fill=-1.0)
    entity, _ = extract(vol["entity"][plane_slice], plane_window, fill=0)
    instance, _ = extract(inst[plane_slice], plane_window, fill=0)
    small, _ = extract(vol["small"][plane_slice], plane_window, fill=False)
    a_ignore, _ = extract(vol["a_ignore"][plane_slice], plane_window, fill=False)
    coords = coordinates_mm(source_window, affine=affine)
    # Independent 2D observations have no observed through-plane extent.
    local = local_coordinates(plane_window, (1, *inst.shape[1:]))
    if do_augment:
        angle = float(rng.uniform(-legacy.ROTATION_DEG, legacy.ROTATION_DEG))
        image, valid, entity, instance, small, a_ignore = rotate_inplane(
            [image, valid.astype(np.uint8), entity, instance, small.astype(np.uint8), a_ignore.astype(np.uint8)],
            angle, [1, 0, 0, 0, 0, 0])
        valid, small, a_ignore = valid.astype(bool), small.astype(bool), a_ignore.astype(bool)
        image = augment(np.where(valid, image, -1.0).astype(np.float32), rng)
    image = np.where(valid, image, -1.0).astype(np.float32)
    a_ignore = a_ignore | ~valid
    renumbered = np.zeros(instance.shape, np.int64)
    ids = []
    for source_id in np.unique(instance):
        if source_id <= 0:
            continue
        components, count = ndimage.label(instance[0] == source_id, structure=np.ones((3, 3), bool))
        for component in range(1, count + 1):
            mask = components == component
            if int(mask.sum()) * pixel_area + 1e-9 < min_area_mm2:
                small[0] |= mask
                continue
            if len(ids) >= n_events:
                raise ValueError("crop instances exceed query capacity")
            ids.append(int(source_id))
            renumbered[0, mask] = len(ids)
    eligible = _eligibility(vol, ids, default=False)
    out = {"image": torch.from_numpy(np.ascontiguousarray(image))[None],
           "valid": torch.from_numpy(valid.astype(np.float32)),
           "coords": torch.from_numpy(coords), "local": torch.from_numpy(local),
           "entity": torch.from_numpy(entity.astype(np.int64)),
           "entity_present": torch.from_numpy(np.isin(np.arange(1, N_ENTITIES + 1), np.unique(entity))),
           "instance": torch.from_numpy(renumbered),
           "point_weight": torch.from_numpy((~small & valid).astype(np.float32)),
           "a_ignore": torch.from_numpy(a_ignore.astype(bool)), "seq": torch.tensor(vol["seq"]),
           "a_supervised": torch.tensor(bool(vol.get("a_supervised", True))),
           "r_supervised": torch.tensor(bool(vol.get("r_supervised", True))),
           "u_supervised": torch.tensor(bool(vol["u_supervised"])),
           "n_instances": len(ids), "case": vol["case"],
           "r_eligible": eligible, **_source_targets(vol, ids, eligible)}
    return _metadata(out, vol, "2d", ids, source_window)


def collate_unified(items):
    """Collate homogeneous-mode crop lists without losing per-instance masks."""
    crops = [crop for item in items for crop in item]
    if not crops or len({c["input_mode"] for c in crops}) != 1:
        raise ValueError("unified batches must be nonempty and homogeneous in input mode")
    out = legacy.collate(items)
    out["input_mode"] = crops[0]["input_mode"]
    out["spacing"] = torch.stack([c["spacing"] for c in crops])
    for key in ("r_eligible", "source_instance_ids", "source_window", "source_affine",
                "pixel_area_mm2", "voxel_volume_mm3"):
        out[key] = [c[key] for c in crops]
    out["measurement_unit"] = crops[0]["measurement_unit"]
    return out
