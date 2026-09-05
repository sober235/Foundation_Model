"""fastMRI RSS stack -> NIfTI with a physically meaningful affine.

Geometry facts these tests pin down (established 2026-09-05):
- fastMRI h5 ``reconstruction_rss`` is ordered (slice, row, col).
- Rows run anterior -> posterior, slices run inferior -> superior (checked on
  rendered brain volumes). Columns are assumed radiological (image left =
  patient right), so increasing column index points to patient Left.
  Resulting NIfTI axis codes for an array ordered (col, row, slice): (L, P, S).
- In-plane spacing comes from the ISMRMRD reconSpace: rows (readout, x) get
  fov_x / nx, columns (phase, y) get fov_y / ny. Slice spacing is reconSpace
  fov_z; official fastMRI brain DICOMs confirm SpacingBetweenSlices ==
  SliceThickness == 5 mm (no gap).
"""
import h5py
import nibabel as nib
import numpy as np
import pytest

from anatobind.data_engine.fastmri import parse_recon_geometry, rss_affine, rss_h5_to_nifti

HEADER_XML = (
    '<?xml version="1.0" encoding="utf-8"?>'
    '<ismrmrdHeader xmlns="http://www.ismrm.org/ISMRMRD"><encoding>'
    "<encodedSpace><matrixSize><x>16</x><y>6</y><z>1</z></matrixSize>"
    "<fieldOfView_mm><x>11</x><y>3</y><z>7.5</z></fieldOfView_mm></encodedSpace>"
    "<reconSpace><matrixSize><x>8</x><y>6</y><z>1</z></matrixSize>"
    "<fieldOfView_mm><x>5.5</x><y>3</y><z>5</z></fieldOfView_mm></reconSpace>"
    "</encoding></ismrmrdHeader>"
)


def test_rss_affine_axis_codes_are_L_P_S():
    aff = rss_affine(row_spacing_mm=0.6875, col_spacing_mm=0.5, slice_spacing_mm=5.0)
    assert nib.aff2axcodes(aff) == ("L", "P", "S")


def test_rss_affine_zooms_follow_array_order_col_row_slice():
    aff = rss_affine(row_spacing_mm=0.6875, col_spacing_mm=0.5, slice_spacing_mm=5.0)
    zooms = np.sqrt((aff[:3, :3] ** 2).sum(axis=0))
    assert np.allclose(zooms, [0.5, 0.6875, 5.0])


def test_parse_recon_geometry_reads_recon_space_not_encoded_space():
    geo = parse_recon_geometry(HEADER_XML)
    assert geo["nx"] == 8 and geo["ny"] == 6
    assert geo["fov_x_mm"] == pytest.approx(5.5)
    assert geo["fov_y_mm"] == pytest.approx(3.0)
    assert geo["fov_z_mm"] == pytest.approx(5.0)


def _write_fake_fastmri(path, rss):
    with h5py.File(path, "w") as f:
        f.create_dataset("reconstruction_rss", data=rss)
        f.create_dataset("ismrmrd_header", data=np.bytes_(HEADER_XML.encode()))
        f.attrs["acquisition"] = "AXFLAIR"


def test_rss_h5_to_nifti_has_expected_shape_zooms_and_values(tmp_path):
    rss = np.random.RandomState(0).rand(3, 8, 6).astype(np.float32)  # (slice, row, col)
    _write_fake_fastmri(tmp_path / "f.h5", rss)

    out = rss_h5_to_nifti(tmp_path / "f.h5", tmp_path / "f.nii.gz")

    img = nib.load(str(out))
    assert img.shape == (6, 8, 3)  # (col, row, slice)
    assert np.allclose(img.header.get_zooms(), (0.5, 0.6875, 5.0))
    assert nib.aff2axcodes(img.affine) == ("L", "P", "S")
    data = np.asarray(img.dataobj)
    assert np.allclose(data[:, :, 1], rss[1].T)


def test_rss_h5_to_nifti_rejects_shape_header_mismatch(tmp_path):
    rss = np.zeros((3, 7, 6), dtype=np.float32)  # rows != recon nx (8)
    _write_fake_fastmri(tmp_path / "bad.h5", rss)
    with pytest.raises(ValueError):
        rss_h5_to_nifti(tmp_path / "bad.h5", tmp_path / "bad.nii.gz")
