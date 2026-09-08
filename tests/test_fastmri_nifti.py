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

from anatobind.data_engine.fastmri import TooFewSlices, parse_recon_geometry, rss_affine, rss_h5_to_nifti

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
    rss = np.random.RandomState(0).rand(12, 8, 6).astype(np.float32)  # (slice, row, col)
    _write_fake_fastmri(tmp_path / "f.h5", rss)

    out = rss_h5_to_nifti(tmp_path / "f.h5", tmp_path / "f.nii.gz")

    img = nib.load(str(out))
    assert img.shape == (6, 8, 12)  # (col, row, slice)
    assert np.allclose(img.header.get_zooms(), (0.5, 0.6875, 5.0))
    assert nib.aff2axcodes(img.affine) == ("L", "P", "S")
    data = np.asarray(img.dataobj)
    assert np.allclose(data[:, :, 1], rss[1].T)


def test_rss_h5_to_nifti_rejects_rss_larger_than_encoded_matrix(tmp_path):
    rss = np.zeros((12, 20, 6), dtype=np.float32)  # rows 20 > encoded nx 16: cannot be a crop
    _write_fake_fastmri(tmp_path / "bad.h5", rss)
    with pytest.raises(ValueError):
        rss_h5_to_nifti(tmp_path / "bad.h5", tmp_path / "bad.nii.gz")


# Low-resolution fastMRI FLAIR series (e.g. AXFLAIR_203): the stored RSS is at the
# acquired resolution while reconSpace describes the vendor's 2x interpolated grid.
HEADER_INTERP_XML = HEADER_XML.replace(
    "<reconSpace><matrixSize><x>8</x><y>6</y><z>1</z></matrixSize>"
    "<fieldOfView_mm><x>5.5</x><y>3</y><z>5</z></fieldOfView_mm></reconSpace>",
    "<reconSpace><matrixSize><x>32</x><y>32</y><z>1</z></matrixSize>"
    "<fieldOfView_mm><x>5.5</x><y>5.5</y><z>5</z></fieldOfView_mm></reconSpace>",
)


def test_parse_recon_geometry_also_returns_encoded_space():
    geo = parse_recon_geometry(HEADER_INTERP_XML)
    assert geo["enc_nx"] == 16 and geo["enc_ny"] == 6
    assert geo["enc_fov_x_mm"] == pytest.approx(11.0)
    assert geo["enc_fov_y_mm"] == pytest.approx(3.0)


def test_spacing_comes_from_encoded_space_not_interpolated_recon_space(tmp_path):
    rss = np.zeros((12, 6, 6), dtype=np.float32)  # crop of the 16x6 acquired grid, smaller than recon 32x32
    with h5py.File(tmp_path / "lowres.h5", "w") as f:
        f.create_dataset("reconstruction_rss", data=rss)
        f.create_dataset("ismrmrd_header", data=np.bytes_(HEADER_INTERP_XML.encode()))
    img = nib.load(str(rss_h5_to_nifti(tmp_path / "lowres.h5", tmp_path / "lowres.nii.gz")))
    assert np.allclose(img.header.get_zooms(), (0.5, 0.6875, 5.0))  # 3/6 and 11/16, not 5.5/32


def test_thin_stacks_are_padded_symmetrically_to_twelve_slices(tmp_path):
    # SynthSeg treats a last dimension <= 10 as channels, so 10-slice stacks must be padded.
    rss = np.random.RandomState(1).rand(10, 8, 6).astype(np.float32)
    _write_fake_fastmri(tmp_path / "thin.h5", rss)
    img = nib.load(str(rss_h5_to_nifti(tmp_path / "thin.h5", tmp_path / "thin.nii.gz")))
    assert img.shape == (6, 8, 12)
    data = np.asarray(img.dataobj)
    assert np.all(data[:, :, 0] == 0) and np.all(data[:, :, 11] == 0)
    assert np.allclose(data[:, :, 1], rss[0].T)
    # world z of original slice 0 is unchanged by the padding: (0 - 4.5) * 5 mm
    assert img.affine[2, 3] + 1 * 5.0 == pytest.approx(-22.5)


def test_fewer_than_four_slices_is_rejected(tmp_path):
    rss = np.zeros((2, 8, 6), dtype=np.float32)
    _write_fake_fastmri(tmp_path / "two.h5", rss)
    with pytest.raises(TooFewSlices):
        rss_h5_to_nifti(tmp_path / "two.h5", tmp_path / "two.nii.gz")
