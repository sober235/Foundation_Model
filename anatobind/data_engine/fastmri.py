"""fastMRI raw-file helpers: RSS stack -> NIfTI with a physical affine.

Conventions (see tests/test_fastmri_nifti.py for the evidence):
- ``reconstruction_rss`` is ordered (slice, row, col); rows are the readout
  axis (ISMRMRD reconSpace x), columns the phase axis (reconSpace y).
- Rows run anterior -> posterior, slices inferior -> superior. Columns are
  assumed radiological (image left = patient right); this left/right
  handedness is an assumption, not a measured fact.
- Output array is reordered to (col, row, slice) so that NIfTI axis codes are
  (L, P, S).
"""
from pathlib import Path
import xml.etree.ElementTree as ET

import h5py
import nibabel as nib
import numpy as np


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


def parse_recon_geometry(header_xml):
    """Return reconSpace matrix size and field of view from an ISMRMRD header."""
    root = ET.fromstring(header_xml)

    def local(tag):
        return tag.rsplit("}", 1)[-1]

    recon = next(e for e in root.iter() if local(e.tag) == "reconSpace")
    vals = {}
    for group in recon:
        for item in group:
            vals[(local(group.tag), local(item.tag))] = float(item.text)
    return {
        "nx": int(vals[("matrixSize", "x")]),
        "ny": int(vals[("matrixSize", "y")]),
        "fov_x_mm": vals[("fieldOfView_mm", "x")],
        "fov_y_mm": vals[("fieldOfView_mm", "y")],
        "fov_z_mm": vals[("fieldOfView_mm", "z")],
    }


def rss_h5_to_nifti(h5_path, out_path):
    """Write the RSS reconstruction of a fastMRI h5 file as a NIfTI volume."""
    with h5py.File(h5_path, "r") as f:
        rss = f["reconstruction_rss"][()]
        header = f["ismrmrd_header"][()]
    if isinstance(header, bytes):
        header = header.decode()
    geo = parse_recon_geometry(header)

    n_slice, n_row, n_col = rss.shape
    if (n_row, n_col) != (geo["nx"], geo["ny"]):
        raise ValueError(
            f"RSS slice shape {(n_row, n_col)} does not match reconSpace matrix "
            f"{(geo['nx'], geo['ny'])} in {h5_path}"
        )
    row_spacing = geo["fov_x_mm"] / geo["nx"]
    col_spacing = geo["fov_y_mm"] / geo["ny"]
    slice_spacing = geo["fov_z_mm"]

    vol = np.ascontiguousarray(rss.transpose(2, 1, 0)).astype(np.float32)
    aff = rss_affine(row_spacing, col_spacing, slice_spacing, shape=vol.shape)
    img = nib.Nifti1Image(vol, aff)
    img.header.set_xyzt_units("mm")
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    nib.save(img, str(out_path))
    return out_path
