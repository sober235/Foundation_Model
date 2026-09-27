"""Build the relation feature table (spec 2026-09-27 §4): per-lesion rows from SynthSeg geometry, image patches from
the RSS, the class-level pseudo label C1, the §4.7 checks and the manifest. Volumes come through an injected loader so
tests run on synthetic arrays."""
import numpy as np
from scipy import ndimage

from anatobind.relation.table import PATCH_PX, PATCH_SLICES, PIXEL_MM
from anatobind.eval.geometry import (
    CANDIDATE_MM, CLASS_NAMES, DIST_CAP_MM, HOST_CLASSES, SLOT_FIELDS, SOFT_SIGMA_MM, class_maps, family_sides,
    host_class_map, interface_margin, landmark_map, lesion_centroid, lesion_class_distances, member_rects,
    min_in_lesion, side_of, slot_features, third,
)
import json

from anatobind.eval.lookup import class_level_host
from anatobind.level_r.registry import is_3mm
from anatobind.level_r.store import now_iso
from anatobind.relation.table import (
    LESION_TYPE_OF_LABEL, PATCH_MM, PATCH_SLICES, ROW_THIRDS, SLICE_THIRDS, SLOTS, TABLE_VERSION, slot_col, write_table,
)


def zscore_volume(rss, seg):
    """FLAIR z-scored over the brain voxels (seg > 0) of its own volume (spec §4.4)."""
    v = np.asarray(rss, np.float32)
    brain = np.asarray(seg) > 0
    mu, sd = float(v[brain].mean()), float(v[brain].std())
    if not sd > 0:
        raise ValueError("brain intensities are constant; refusing to z-score")
    return ((v - mu) / sd).astype(np.float32)


def lesion_patch(zvol, classmap, rects, spacing, centroid):
    """(image, mask, classmap) patches of PATCH_PX x PATCH_PX at PIXEL_MM centred on the lesion centroid, PATCH_SLICES
    slices around the centre slice with edge slices replicated. Axes: (slice, col, row). Image is bilinear,
    mask and classmap nearest-neighbour; outside the volume is zero."""
    nc, nr, ns = zvol.shape
    zc = int(np.clip(np.rint(centroid[2]), 0, ns - 1))
    u = (np.arange(PATCH_PX) - (PATCH_PX - 1) / 2) * PIXEL_MM
    cc, rr = np.meshgrid(centroid[0] + u / spacing[0], centroid[1] + u / spacing[1], indexing="ij")
    coords = np.stack([cc, rr])
    img = np.zeros((PATCH_SLICES, PATCH_PX, PATCH_PX), np.float32)
    msk = np.zeros((PATCH_SLICES, PATCH_PX, PATCH_PX), bool)
    cmp = np.zeros((PATCH_SLICES, PATCH_PX, PATCH_PX), np.int8)
    half = PATCH_SLICES // 2
    for k, dz in enumerate(range(-half, half + 1)):
        z = int(np.clip(zc + dz, 0, ns - 1))
        img[k] = ndimage.map_coordinates(zvol[:, :, z], coords, order=1, mode="constant", cval=0.0)
        plane = np.zeros((nc, nr), np.uint8)
        for c0, c1, r0, r1, s in rects:
            if s == z:
                plane[c0:c1, r0:r1] = 1
        msk[k] = ndimage.map_coordinates(plane, coords, order=0, mode="constant", cval=0) > 0
        cmp[k] = ndimage.map_coordinates(classmap[:, :, z].astype(np.int8), coords, order=0, mode="constant", cval=0)
    return img.astype(np.float16), msk, cmp


LABEL_TO_CLASS = {label: name for name, labels in HOST_CLASSES.items() for label in labels}
MIN_C1_AGREEMENT = 0.99
TOL_MM = 1e-6


class BuildError(ValueError):
    pass


def volume_context(seg, spacing):
    cm = host_class_map(seg)
    slices = np.nonzero((seg > 0).any((0, 1)))[0]
    return {"seg": seg, "cm": cm, "maps": class_maps(cm, spacing), "spacing": tuple(float(s) for s in spacing),
            "families": family_sides(seg), "vent": landmark_map(seg, spacing),
            "brain_slices": (int(slices.min()), int(slices.max())) if slices.size else (0, seg.shape[2] - 1)}


def lesion_row(reg, members, ctx, fold):
    seg, cm, maps, sp = ctx["seg"], ctx["cm"], ctx["maps"], ctx["spacing"]
    rects = member_rects(members, seg.shape)
    if not rects:
        raise BuildError(f"lesion {reg['lesion_id']}: no voxel on the grid")
    slots = slot_features(cm, maps, rects, sp)
    c1, source, frac = class_level_host(slots)
    _, d1, d2 = interface_margin(lesion_class_distances({c: maps[c][0] for c in maps}, rects))
    centroid = lesion_centroid(rects)
    zc = int(np.clip(np.rint(centroid[2]), 0, seg.shape[2] - 1))
    brain_rows = np.nonzero((seg[:, :, zc] > 0).any(0))[0]
    row_third = third(int(np.rint(centroid[1])), int(brain_rows.min()), int(brain_rows.max()), ROW_THIRDS) if brain_rows.size else ROW_THIRDS[1]
    slice_third = third(zc, ctx["brain_slices"][0], ctx["brain_slices"][1], SLICE_THIRDS)
    side = side_of(seg, rects, ctx["families"])
    volume = sum((b - a) * (d - c) for a, b, c, d, _ in rects) * sp[0] * sp[1] * sp[2]
    row = {"lesion_id": reg["lesion_id"], "file": reg["file"], "patient_id": reg["patient_id"], "fold": fold,
           "stratum_geometry": reg["stratum_geometry"], "stratum_series": reg["stratum_series"], "band": reg["band"],
           "is_3mm": is_3mm(reg["stratum_geometry"]), "lesion_type": LESION_TYPE_OF_LABEL[reg["label"]],
           "n_slices": reg["n_slices"], "inplane_mm": reg["inplane_mm"], "x0": reg["x0"], "y0": reg["y0"], "x1": reg["x1"],
           "y1": reg["y1"], "z0": reg["z0"], "z1": reg["z1"], "spacing_col_mm": sp[0], "spacing_row_mm": sp[1],
           "spacing_slice_mm": sp[2], "slice_thickness_mm": sp[2], "extent_x_mm": (reg["x1"] - reg["x0"]) * sp[0],
           "extent_y_mm": (reg["y1"] - reg["y0"]) * sp[1], "extent_z_mm": (reg["z1"] - reg["z0"] + 1) * sp[2],
           "volume_mm3": float(volume), "centroid_col": float(centroid[0]), "centroid_row": float(centroid[1]),
           "centroid_slice": float(centroid[2]), "side": side, "row_third": row_third, "slice_third": slice_third,
           "coarse_location": f"{side}|{row_third}|{slice_third}", "d1_mm": float(d1), "d_interface_mm": float(d2),
           "delta_d_mm": float(d2 - d1), "dist_cortex_mm": slots[CLASS_NAMES.index("cortex") + 1]["min_surface_mm"],
           "dist_ventricle_mm": min_in_lesion(ctx["vent"], rects), "c1_class": CLASS_NAMES[c1 - 1], "c1_slot": c1 - 1,
           "c1_source": source, "c1_overlap": float(frac),
           "registry_lookup_class": LABEL_TO_CLASS.get(int(reg["host_lookup_parenchyma"]), "")}
    for i, s in enumerate(SLOTS, start=1):
        for f in SLOT_FIELDS:
            row[slot_col(s, f)] = slots[i][f]
    return row


def check_rows(rows, registry_rows, patient_fold, patches, min_agreement=MIN_C1_AGREEMENT):
    """The spec §4.7 checks; returns the manifest counts or raises BuildError."""
    ids = [r["lesion_id"] for r in rows]
    if len(ids) != len(set(ids)):
        raise BuildError("duplicate lesion_id")
    if registry_rows and set(ids) != {r["lesion_id"] for r in registry_rows}:
        raise BuildError(f"{len(ids)} rows for {len(registry_rows)} registry lesions: the lesion_id sets differ")
    reg_by_id = {r["lesion_id"]: r for r in registry_rows}
    folds_of = {}
    for r in rows:
        folds_of.setdefault(r["patient_id"], set()).add(r["fold"])
        if patient_fold is not None and patient_fold.get(r["patient_id"]) != r["fold"]:
            raise BuildError(f"patient {r['patient_id']} is not in the fold table with fold {r['fold']}")
        reg = reg_by_id.get(r["lesion_id"])
        if reg is not None:
            for k in ("d1_mm", "d_interface_mm", "delta_d_mm"):
                if abs(float(reg[k]) - r[k]) > TOL_MM:
                    raise BuildError(f"lesion {r['lesion_id']}: {k} {r[k]} differs from the registry {reg[k]}")
        if not any(r[slot_col(s, "candidate")] for s in SLOTS):
            raise BuildError(f"lesion {r['lesion_id']}: no candidate")
    if any(len(v) != 1 for v in folds_of.values()):
        raise BuildError("a patient appears in more than one fold (folds are by patient)")
    mismatches = [r["lesion_id"] for r in rows if r["c1_class"] != r["registry_lookup_class"]]
    fraction = 1.0 - len(mismatches) / len(rows)
    if fraction < min_agreement:
        raise BuildError(f"C1 agrees with the registry lookup on {fraction:.4f} < {min_agreement}; mismatches {mismatches[:20]}")
    img = np.asarray(patches["image"], np.float32).reshape(len(rows), -1)
    if (img.std(1) == 0).any():
        raise BuildError("a constant image patch")
    per_fold = {}
    for r in rows:
        d = per_fold.setdefault(str(r["fold"]), {"patients": set(), "lesions": 0})
        d["patients"].add(r["patient_id"])
        d["lesions"] += 1
    counts = {}
    for r in rows:
        counts[r["c1_class"]] = counts.get(r["c1_class"], 0) + 1
    sources = {}
    for r in rows:
        sources[r["c1_source"]] = sources.get(r["c1_source"], 0) + 1
    forced = sum(1 for r in rows if all(r[slot_col(s, "min_surface_mm")] > CANDIDATE_MM for s in SLOTS if r[slot_col(s, "candidate")]))
    return {"n_lesions": len(rows), "n_patients": len(folds_of),
            "per_fold": {k: {"patients": len(v["patients"]), "lesions": v["lesions"]} for k, v in sorted(per_fold.items())},
            "c1_distribution": counts, "c1_source_counts": sources, "n_candidate_fallback": forced,
            "c1_agreement": {"fraction": fraction, "mismatches": mismatches}}


def build_table(registry_rows, patient_fold, load_volume, out_dir, sources, log=print, min_agreement=MIN_C1_AGREEMENT):
    by_file = {}
    for r in registry_rows:
        by_file.setdefault(r["file"], []).append(r)
    rows, images, masks, cmaps, ids = [], [], [], [], []
    for i, f in enumerate(sorted(by_file)):
        seg, spacing, rss, members_of = load_volume(f)
        ctx = volume_context(seg, spacing)
        z = zscore_volume(rss, seg)
        for reg in sorted(by_file[f], key=lambda r: r["lesion_id"]):
            if reg["patient_id"] not in patient_fold:
                raise BuildError(f"patient {reg['patient_id']} is not in the fold table")
            rects = member_rects(members_of[reg["lesion_id"]], seg.shape)
            rows.append(lesion_row(reg, members_of[reg["lesion_id"]], ctx, patient_fold[reg["patient_id"]]))
            img, msk, cmp = lesion_patch(z, ctx["cm"], rects, spacing, lesion_centroid(rects))
            images.append(img)
            masks.append(msk)
            cmaps.append(cmp)
            ids.append(reg["lesion_id"])
        log(f"{i + 1}/{len(by_file)} {f}: {len(by_file[f])} lesions")
    order = np.argsort(ids)
    rows = [rows[k] for k in order]
    patches = {"lesion_id": np.array(ids)[order], "image": np.stack(images)[order], "mask": np.stack(masks)[order],
               "classmap": np.stack(cmaps)[order],
               "meta": np.array(json.dumps({"window_mm": PATCH_MM, "pixel_mm": PIXEL_MM, "slices": "centre-1, centre, centre+1",
                                            "axes": "lesion, slice, col, row", "image": "per-volume z-score over seg > 0"}))}
    report = check_rows(rows, registry_rows, patient_fold, patches, min_agreement)
    manifest = {"version": TABLE_VERSION, "built_at": now_iso(), **sources,
                "parameters": {"candidate_mm": CANDIDATE_MM, "distance_cap_mm": DIST_CAP_MM, "soft_sigma_mm": SOFT_SIGMA_MM,
                               "patch_mm": PATCH_MM, "pixel_mm": PIXEL_MM, "patch_slices": PATCH_SLICES,
                               "min_c1_agreement": min_agreement}, **report}
    write_table(out_dir, rows, patches, manifest)
    return rows, patches, manifest
