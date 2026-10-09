"""Offline resampling of the sample table to 1 mm isotropic RAS grids (decision of 2026-10-09, Q4 (b)).

The four sources store PDGM and SibBMS at 1 mm, BMSR at 0.43-1.17 mm in plane with 1-5 mm slices and ISLES at 2 mm
(54 cases at 4.8 mm slices). A fixed 128 x 160 x 160 voxel crop therefore spans 69-188 mm in plane across the sources,
and the anatomy pseudo-labels are SynthSeg's, computed at 1 mm. Every case whose voxels are not 1 mm is resampled once,
offline, onto a 1 mm isotropic RAS-aligned grid that covers the case's world extent: the image with trilinear
interpolation, the anatomy and lesion maps with nearest neighbour onto the same grid. Nothing is resampled a second
time at training; 1 mm cases are used as stored (they are reoriented, not resampled, when loaded). A case whose native
slice thickness is above THICK_MM keeps its image, U and S supervision but loses the A and R supervision: a SynthSeg map
on 4.8 mm slices is not a trustworthy anatomy target, and interpolation does not create the anatomy the acquisition
did not record. Outputs are never overwritten: a second run refuses, or skips the files that exist with resume."""
import json
from collections import defaultdict
from pathlib import Path

import nibabel as nib
import numpy as np
from nibabel.processing import resample_from_to, resample_to_output

TARGET_MM = 1.0
THICK_MM = 3.0
SPACING_TOL = 1e-3


def native_spacing(path):
    """(x, y, z) voxel spacing in mm from the header."""
    return tuple(float(v) for v in nib.load(str(path)).header.get_zooms()[:3])


def needs_resampling(spacing, target=TARGET_MM, tol=SPACING_TOL):
    return any(abs(float(s) - target) > tol for s in spacing)


def thick_slice(spacing, limit=THICK_MM):
    return max(float(s) for s in spacing) > limit


def _target_grid(img, target=TARGET_MM):
    """The 1 mm isotropic RAS-aligned grid covering the image's world extent (nibabel's output grid)."""
    out = resample_to_output(img, voxel_sizes=(target, target, target), order=0, mode="constant", cval=0.0)
    return out.shape, out.affine


def _resample(path, grid, order, dtype, out_path):
    img = nib.load(str(path))
    out = resample_from_to(img, grid, order=order, mode="constant", cval=0.0)
    data = np.asarray(out.dataobj).astype(dtype)
    res = nib.Nifti1Image(data, out.affine)
    res.header.set_xyzt_units("mm")
    nib.save(res, str(out_path))


def resample_case(rows, out_root, resume=False):
    """All rows of one case (they share the anatomy and lesion maps) -> the same rows with the files resampled to 1 mm
    under out_root/<source>/<case>/<basename>. Refuses to overwrite an existing file unless `resume`, in which case
    existing files are kept as they are."""
    out_root = Path(out_root)
    case, source = rows[0]["case"], rows[0]["source"]
    folder = out_root / source / case
    folder.mkdir(parents=True, exist_ok=True)
    anatomy = nib.load(rows[0]["anatomy"])
    grid = _target_grid(anatomy)
    spacing = [float(v) for v in anatomy.header.get_zooms()[:3]]
    done = {}

    def emit(src, order, dtype):
        src = str(src)
        if src in done:
            return done[src]
        dst = folder / Path(src).name
        if dst.exists():
            if not resume:
                raise FileExistsError(f"{dst} exists; this run does not overwrite (use resume to keep it)")
        else:
            _resample(src, grid, order, dtype, dst)
        done[src] = str(dst)
        return done[src]

    out = []
    for r in rows:
        new = dict(r)
        new["anatomy"] = emit(r["anatomy"], 0, np.int16)
        new["image"] = emit(r["image"], 1, np.float32)
        new["lesion"] = emit(r["lesion"], 0, np.int16) if r["lesion"] else None
        new["native_spacing"] = spacing
        new["resampled"] = True
        out.append(new)
    return out


def plan(rows):
    """(cases to resample as [(case, rows)], rows kept as stored with native_spacing / resampled=False)."""
    by_case = defaultdict(list)
    for r in rows:
        by_case[r["case"]].append(r)
    todo, kept = [], []
    for case, group in by_case.items():
        spacing = native_spacing(group[0]["anatomy"])
        if needs_resampling(spacing):
            todo.append((case, group))
        else:
            kept += [{**r, "native_spacing": [float(s) for s in spacing], "resampled": False} for r in group]
    return todo, kept


def apply_thickness_rule(rows, limit=THICK_MM):
    """Rows whose native slice thickness is above `limit` lose the A and R supervision (U and S stay)."""
    out = []
    for r in rows:
        spacing = r.get("native_spacing") or list(native_spacing(r["anatomy"]))
        thick = thick_slice(spacing, limit)
        new = {**r, "native_spacing": [float(s) for s in spacing], "thick_slice": bool(thick)}
        if thick:
            new["a_supervised"], new["r_supervised"] = False, False
        out.append(new)
    return out


def counts(rows):
    """{source: {"cases", "rows", "resampled", "thick"}} for the records."""
    out = {}
    for r in rows:
        c = out.setdefault(r["source"], {"cases": set(), "rows": 0, "resampled": 0, "thick": 0})
        c["cases"].add(r["case"])
        c["rows"] += 1
        c["resampled"] += int(bool(r.get("resampled")))
        c["thick"] += int(bool(r.get("thick_slice")))
    return {s: {**c, "cases": len(c["cases"])} for s, c in out.items()}


def write_table(path, rows):
    path = Path(path)
    if path.exists():
        raise FileExistsError(f"{path} exists; nothing is overwritten here, use a new name")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(rows, indent=1))
    return counts(rows)
