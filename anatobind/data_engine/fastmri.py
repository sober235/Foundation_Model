"""fastMRI raw-file helpers: RSS stack -> NIfTI with a physical affine.

Conventions (see tests/test_fastmri_nifti.py for the evidence):
- ``reconstruction_rss`` is ordered (slice, row, col); rows are the readout
  axis (ISMRMRD x), columns the phase axis (y).
- The stored RSS lives at the *acquired* resolution, so pixel spacing is
  encodedSpace FOV / encodedSpace matrix per axis (readout oversampling is
  already folded into both numbers). reconSpace can be the vendor's
  interpolated grid (e.g. AXFLAIR_203: 512x512 at 0.43 mm while the RSS is
  276x276 at 0.86 mm), so it is only used for the slice spacing (fov_z).
- Rows run anterior -> posterior, slices inferior -> superior. Columns are
  assumed radiological (image left = patient right); this left/right
  handedness is an assumption, not a measured fact.
- Output array is reordered to (col, row, slice) so that NIfTI axis codes are
  (L, P, S).
- SynthSeg treats a last dimension <= 10 as channels and refuses <= 3 slices,
  so stacks with 4..11 slices are zero-padded symmetrically to at least 12
  slices (world coordinates of the real slices are unchanged) and stacks with
  fewer than 4 slices raise ``TooFewSlices``.
"""
import csv
from collections import defaultdict
from pathlib import Path
import xml.etree.ElementTree as ET

import h5py
import nibabel as nib
import numpy as np

MIN_SLICES = 4
PAD_TO_SLICES = 12

# --- fastMRI+ box convention (Gate 0; v2.6 §4.1) ---------------------------------------------------------------
# The fastMRI+ README: "In the process of converting the images to DICOM, the pixel arrays were flipped (up/down)
# to provide a view that was closer to DICOM orientation and assist with labeling." So the CSV y counts rows from
# the BOTTOM of the reconstruction_rss array. Every export from Gate 0 on stores rows counted from the top, and
# says so in its manifest (transform_version 2). Verified on 24 brain and 30 knee volumes on 2026-09-15
# (docs/verification/2026-09-16-brain-probe/REPORT.md §2) and on every annotated volume by
# scripts/audit_fastmri_plus_boxes.py.
BOX_CONVENTION_CSV = "fastmri_plus_csv_rows_from_bottom"
BOX_CONVENTION_RSS = "rss_rows_from_top"
TRANSFORM_VERSION = 2        # 1 = CSV boxes used as-is (leg 2 before 2026-09-24), 2 = convert_box_csv_to_rss applied
MIN_BOX_SIDE = 3


def convert_box_csv_to_rss(x, y, width, height, n_rows):
    """fastMRI+ CSV box -> half-open box on the RSS array, (row0, row1, col0, col1), rows counted from the top."""
    return n_rows - y - height, n_rows - y, x, x + width


def convert_box_rss_to_csv(row0, row1, col0, col1, n_rows):
    """Inverse of convert_box_csv_to_rss: (x, y, width, height) as fastMRI+ would have written it."""
    return col0, n_rows - row1, col1 - col0, row1 - row0


def rss_spacing_mm(header_xml):
    """(slice, row, col) voxel size in mm of the stored RSS: acquired resolution in plane (encodedSpace
    FOV / matrix, readout oversampling already folded in), reconSpace fov z through plane. Same rule as
    rss_h5_to_nifti."""
    g = parse_recon_geometry(header_xml)
    return (g["fov_z_mm"], g["enc_fov_x_mm"] / g["enc_nx"], g["enc_fov_y_mm"] / g["enc_ny"])


def voxel_to_world(index, spacing):
    """(slice, row, col) index -> mm. fastMRI h5 files carry no patient position, so the array origin is 0 mm and
    this is the only physical frame the data has."""
    return tuple(float(i) * float(s) for i, s in zip(index, spacing))


def world_to_voxel(point_mm, spacing):
    return tuple(float(p) / float(s) for p, s in zip(point_mm, spacing))


def read_fastmri_plus_rows(csv_path):
    """fastMRI+ CSV -> per-slice boxes in the CSV frame, one dict per row: file, slice, x, y, width, height, label.
    Study-level rows carry no box and are skipped, as are rows whose geometry is not integer; labels are stripped
    (the file has "Joint Effusion " with a trailing space)."""
    out = []
    with open(csv_path, newline="") as fh:
        for r in csv.DictReader(fh):
            if r["study_level"].strip() == "Yes":
                continue
            try:
                out.append({"file": r["file"], "slice": int(r["slice"]), "x": int(r["x"]), "y": int(r["y"]),
                            "width": int(r["width"]), "height": int(r["height"]), "label": r["label"].strip()})
            except ValueError:
                continue
    return out


def rows_to_rss_frame(rows, n_rows_of):
    """CSV-frame rows -> RSS-frame rows (x = col0, y = row0 counted from the top; width/height unchanged).
    n_rows_of: the row count of each file's reconstruction_rss, an int (same for every file) or {file: int}."""
    out = []
    for r in rows:
        n = n_rows_of[r["file"]] if isinstance(n_rows_of, dict) else int(n_rows_of)
        row0, row1, col0, col1 = convert_box_csv_to_rss(r["x"], r["y"], r["width"], r["height"], n)
        out.append({**r, "x": col0, "y": row0, "width": col1 - col0, "height": row1 - row0})
    return out


def box_iou_2d(a, b):
    """IoU of two boxes given as dicts with x, y, width, height (any frame, as long as both share it)."""
    ax1, ay1 = a["x"] + a["width"], a["y"] + a["height"]
    bx1, by1 = b["x"] + b["width"], b["y"] + b["height"]
    iw = max(0, min(ax1, bx1) - max(a["x"], b["x"]))
    ih = max(0, min(ay1, by1) - max(a["y"], b["y"]))
    inter = iw * ih
    union = a["width"] * a["height"] + b["width"] * b["height"] - inter
    return inter / union if union else 0.0


def merge_boxes_3d(rows, group_key, iou_min=0.3):
    """Boxes of one (file, group_key) on adjacent slices with in-plane IoU >= iou_min form one 3D lesion (the leg 2
    rule). Each lesion keeps its member rows and the enclosing box: x0/y0 inclusive, x1/y1 exclusive, z0..z1
    inclusive slice indices."""
    lesions = []
    by_group = defaultdict(list)
    for r in rows:
        by_group[(r["file"], r[group_key])].append(r)
    for (file, group), boxes in sorted(by_group.items()):
        parent = list(range(len(boxes)))

        def find(i):
            while parent[i] != i:
                parent[i] = parent[parent[i]]
                i = parent[i]
            return i

        for i, a in enumerate(boxes):
            for j, b in enumerate(boxes):
                if j <= i or abs(a["slice"] - b["slice"]) != 1:
                    continue
                if box_iou_2d(a, b) >= iou_min:
                    parent[find(i)] = find(j)
        comps = defaultdict(list)
        for i in range(len(boxes)):
            comps[find(i)].append(boxes[i])
        for members in comps.values():
            lesions.append({
                "file": file, group_key: group,
                "z0": min(m["slice"] for m in members), "z1": max(m["slice"] for m in members),
                "x0": min(m["x"] for m in members), "y0": min(m["y"] for m in members),
                "x1": max(m["x"] + m["width"] for m in members), "y1": max(m["y"] + m["height"] for m in members),
                "n_boxes": len(members), "members": members,
            })
    return lesions


class TooFewSlices(ValueError):
    """The stack has too few slices to be segmented as a 3D volume."""


def rss_affine(row_spacing_mm, col_spacing_mm, slice_spacing_mm, shape=None):
    """4x4 affine for an array ordered (col, row, slice); axis codes (L, P, S).

    If ``shape`` (col, row, slice) is given, the volume centre is placed at the
    world origin.
    """
    aff = np.diag([-float(col_spacing_mm), -float(row_spacing_mm), float(slice_spacing_mm), 1.0])
    if shape is not None:
        centre_vox = (np.asarray(shape, dtype=float) - 1.0) / 2.0
        aff[:3, 3] = -aff[:3, :3] @ centre_vox
    return aff


def _space(root, name):
    def local(tag):
        return tag.rsplit("}", 1)[-1]

    node = next(e for e in root.iter() if local(e.tag) == name)
    vals = {}
    for group in node:
        for item in group:
            vals[(local(group.tag), local(item.tag))] = float(item.text)
    return vals


def parse_recon_geometry(header_xml):
    """reconSpace and encodedSpace matrix sizes and fields of view from an ISMRMRD header."""
    root = ET.fromstring(header_xml)
    rec, enc = _space(root, "reconSpace"), _space(root, "encodedSpace")
    return {
        "nx": int(rec[("matrixSize", "x")]),
        "ny": int(rec[("matrixSize", "y")]),
        "fov_x_mm": rec[("fieldOfView_mm", "x")],
        "fov_y_mm": rec[("fieldOfView_mm", "y")],
        "fov_z_mm": rec[("fieldOfView_mm", "z")],
        "enc_nx": int(enc[("matrixSize", "x")]),
        "enc_ny": int(enc[("matrixSize", "y")]),
        "enc_fov_x_mm": enc[("fieldOfView_mm", "x")],
        "enc_fov_y_mm": enc[("fieldOfView_mm", "y")],
        "enc_fov_z_mm": enc[("fieldOfView_mm", "z")],
    }


def _pad_slices(vol, min_slices):
    """Zero-pad axis 2 of a (col, row, slice) array symmetrically up to ``min_slices``."""
    n = vol.shape[2]
    if n >= min_slices:
        return vol
    total = min_slices - n
    total += total % 2  # keep the padding symmetric so real slices keep their world z
    each = total // 2
    return np.pad(vol, ((0, 0), (0, 0), (each, each)))


def rss_h5_to_nifti(h5_path, out_path, pad_to_slices=PAD_TO_SLICES):
    """Write the RSS reconstruction of a fastMRI h5 file as a NIfTI volume."""
    with h5py.File(h5_path, "r") as f:
        rss = f["reconstruction_rss"][()]
        header = f["ismrmrd_header"][()]
    if isinstance(header, bytes):
        header = header.decode()
    geo = parse_recon_geometry(header)

    n_slice, n_row, n_col = rss.shape
    if n_slice < MIN_SLICES:
        raise TooFewSlices(f"{h5_path}: {n_slice} slices, need at least {MIN_SLICES}")
    if n_row > geo["enc_nx"] or n_col > geo["enc_ny"]:
        raise ValueError(
            f"RSS slice shape {(n_row, n_col)} exceeds the encoded matrix "
            f"{(geo['enc_nx'], geo['enc_ny'])} in {h5_path}; not a crop of the acquired grid"
        )
    row_spacing = geo["enc_fov_x_mm"] / geo["enc_nx"]
    col_spacing = geo["enc_fov_y_mm"] / geo["enc_ny"]
    slice_spacing = geo["fov_z_mm"]

    vol = np.ascontiguousarray(rss.transpose(2, 1, 0)).astype(np.float32)
    vol = _pad_slices(vol, pad_to_slices)
    aff = rss_affine(row_spacing, col_spacing, slice_spacing, shape=vol.shape)
    img = nib.Nifti1Image(vol, aff)
    img.header.set_xyzt_units("mm")
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    nib.save(img, str(out_path))
    return out_path
