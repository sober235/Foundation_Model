"""Synthetic-only tests for read-only, privacy-safe classic MR intake."""

import hashlib
import json
import subprocess
import sys
import warnings
from pathlib import Path

import numpy as np
import pytest

pydicom = pytest.importorskip("pydicom")
from pydicom.dataset import Dataset, FileDataset, FileMetaDataset
from pydicom.uid import ExplicitVRLittleEndian, MRImageStorage

from anatobind.aur.dicom import (
    DicomInputError,
    inspect_dicom_files,
    read_dicom_input,
    verify_sources,
)


def make_mr(tmp_path, index=1, *, pixels=None, **changes):
    """Write fake identifiers and hand-specified geometry, never real data."""
    meta = FileMetaDataset()
    meta.TransferSyntaxUID = ExplicitVRLittleEndian
    meta.MediaStorageSOPClassUID = MRImageStorage
    meta.MediaStorageSOPInstanceUID = f"1.2.826.0.1.3680043.10.987.1.{index}"
    path = tmp_path / f"synthetic_private_{index}.dcm"
    ds = FileDataset(str(path), {}, file_meta=meta, preamble=b"\0" * 128)
    ds.SOPClassUID = MRImageStorage
    ds.SOPInstanceUID = meta.MediaStorageSOPInstanceUID
    ds.StudyInstanceUID = "1.2.826.0.1.3680043.10.987.2"
    ds.SeriesInstanceUID = "1.2.826.0.1.3680043.10.987.3"
    ds.FrameOfReferenceUID = "1.2.826.0.1.3680043.10.987.4"
    ds.PatientID = "SYNTHETIC_PRIVATE_PATIENT"
    ds.Modality = "MR"
    ds.Rows = 2
    ds.Columns = 3
    ds.ImageOrientationPatient = [1, 0, 0, 0, 1, 0]
    ds.ImagePositionPatient = [10, 20, (index - 1) * 4]
    ds.PixelSpacing = [2, 3]
    ds.SliceThickness = 7  # Never a substitute for slice-position spacing.
    ds.SpacingBetweenSlices = 9
    ds.SamplesPerPixel = 1
    ds.PhotometricInterpretation = "MONOCHROME2"
    ds.BitsAllocated = 16
    ds.BitsStored = 16
    ds.HighBit = 15
    ds.PixelRepresentation = 0
    ds.EchoTime = 10
    ds.EchoNumbers = 1
    ds.TemporalPositionIdentifier = 1
    ds.AcquisitionNumber = 1
    ds.ImageType = ["ORIGINAL", "PRIMARY", "M"]
    for key, value in changes.items():
        if value is None:
            del ds[key]
        else:
            setattr(ds, key, value)
    values = np.arange(6, dtype=np.uint16).reshape(2, 3) if pixels is None else pixels
    ds.PixelData = np.asarray(values, dtype=np.uint16).tobytes()
    ds.save_as(path, enforce_file_format=True)
    return path


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_single_plane_uses_pixel_centers_and_has_area_only(tmp_path):
    # Catches row/column spacing swaps, double LPS conversion, and fake volume.
    path = make_mr(tmp_path)
    before = digest(path)
    image = read_dicom_input([path])
    assert image.input_mode == "2d"
    assert image.image_dhw.dtype == np.float32
    np.testing.assert_array_equal(image.image_dhw, [[[0, 1, 2], [3, 4, 5]]])
    # Affine multiplies (column, row, depth, 1), mapping pixel centers to RAS.
    np.testing.assert_allclose(image.affine_ras, [
        [-3, 0, 0, -10], [0, -2, 0, -20], [0, 0, 1, 0], [0, 0, 0, 1]
    ])
    assert image.measurement_unit == "mm2"
    assert image.pixel_area_mm2 == 6
    assert image.voxel_volume_mm3 is None
    assert image.source_refs[0].sha256 == before
    assert image.verify_sources() is True
    assert verify_sources(image.source_refs) is True
    assert digest(path) == before
    public = repr(image) + repr(image.source_refs)
    assert str(tmp_path) not in public
    assert "SYNTHETIC_PRIVATE" not in public
    assert "1.2.826" not in public


def test_oblique_stack_sorts_real_positions_and_maps_back_to_original_frames(tmp_path):
    # x=(0,1,0), y=(.6,0,.8), normal=(.8,0,-.6), positions separated by 5mm.
    iop = [0, 1, 0, .6, 0, .8]
    p0 = make_mr(tmp_path, 1, ImageOrientationPatient=iop,
                 ImagePositionPatient=[10, 20, 30], RescaleSlope=2, RescaleIntercept=-1)
    p1 = make_mr(tmp_path, 2, ImageOrientationPatient=iop,
                 ImagePositionPatient=[14, 20, 27], RescaleSlope=2, RescaleIntercept=-1,
                 pixels=np.arange(10, 16).reshape(2, 3))
    before = [digest(p0), digest(p1)]
    image = read_dicom_input([p1, p0], input_mode="3d")
    np.testing.assert_allclose(image.affine_ras, [
        [0, -1.2, -4, -10], [-3, 0, 0, -20], [0, 1.6, -3, 30], [0, 0, 0, 1]
    ], atol=1e-6)
    np.testing.assert_array_equal(image.image_dhw[:, 0, 0], [-1, 19])
    assert image.measurement_unit == "mm3"
    assert image.voxel_volume_mm3 == pytest.approx(30)
    assert [ref.path for ref in image.source_refs] == [p1, p0]
    assert [(item.output_depth_index, item.source_ref_index, item.source_frame_index)
            for item in image.frame_map] == [(0, 1, 0), (1, 0, 0)]
    assert [digest(p0), digest(p1)] == before


@pytest.mark.parametrize("changes,code", [
    ({"ImagePositionPatient": None}, "geometry_unsupported"),
    ({"ImageOrientationPatient": None}, "geometry_unsupported"),
    ({"PixelSpacing": None}, "geometry_unsupported"),
    ({"PixelSpacing": [0, 1]}, "geometry_unsupported"),
    ({"PixelSpacing": [float("nan"), 1]}, "geometry_unsupported"),
    ({"PixelSpacing": [1e308, 1e308]}, "geometry_unsupported"),
    ({"ImageOrientationPatient": [1, 0, 0, 1, 0, 0]}, "geometry_unsupported"),
    ({"ImageOrientationPatient": [2, 0, 0, 0, 1, 0]}, "geometry_unsupported"),
    ({"ImagePositionPatient": [1, 2, float("inf")]}, "geometry_unsupported"),
    ({"SOPClassUID": "1.2.840.10008.5.1.4.1.1.4.1"}, "enhanced_unsupported"),
    ({"PhotometricInterpretation": "MONOCHROME1"}, "pixel_format_unsupported"),
    ({"NumberOfFrames": 2}, "multiframe_unsupported"),
    ({"RescaleSlope": float("nan")}, "intensity_transform_unsupported"),
    ({"RescaleSlope": 0}, "intensity_transform_unsupported"),
])
def test_rejects_unsupported_single_objects_with_safe_messages(tmp_path, changes, code):
    path = make_mr(tmp_path, **changes)
    with pytest.raises(DicomInputError) as exc:
        read_dicom_input([path])
    assert exc.value.code == code
    assert str(exc.value) == code
    assert str(tmp_path) not in repr(exc.value)


@pytest.mark.parametrize("changes,code", [
    ({"SeriesInstanceUID": "1.2.826.0.1.3680043.10.987.8"}, "mixed_series"),
    ({"FrameOfReferenceUID": "1.2.826.0.1.3680043.10.987.9"}, "mixed_frame_of_reference"),
    ({"SOPInstanceUID": "1.2.826.0.1.3680043.10.987.1.1"}, "duplicate_source"),
    ({"ImagePositionPatient": [10, 20, 0]}, "duplicate_position"),
    ({"ImagePositionPatient": [11, 20, 4]}, "geometry_unsupported"),
    ({"ImageOrientationPatient": [0, 1, 0, 1, 0, 0]}, "geometry_unsupported"),
    ({"PixelSpacing": [2, 4]}, "geometry_unsupported"),
    ({"EchoTime": 20}, "multidimensional_unsupported"),
    ({"EchoNumbers": 2}, "multidimensional_unsupported"),
    ({"TemporalPositionIdentifier": 2}, "multidimensional_unsupported"),
    ({"AcquisitionNumber": 2}, "multidimensional_unsupported"),
    ({"DiffusionBValue": 1000}, "multidimensional_unsupported"),
])
def test_rejects_incompatible_stack_members(tmp_path, changes, code):
    p0 = make_mr(tmp_path, 1)
    p1 = make_mr(tmp_path, 2, **changes)
    with pytest.raises(DicomInputError) as exc:
        read_dicom_input([p0, p1])
    assert exc.value.code == code


@pytest.mark.parametrize("b_values", [(0,), (1000,), (0, 1000)])
def test_nested_diffusion_sequences_are_explicitly_unsupported(tmp_path, b_values):
    # Otherwise identical series/positions must not hide acquisition dimensions
    # inside a sequence that this classic-only reader does not yet interpret.
    paths = []
    for index, b_value in enumerate(b_values, start=1):
        diffusion = Dataset()
        diffusion.DiffusionBValue = b_value
        paths.append(make_mr(tmp_path, index, MRDiffusionSequence=[diffusion]))
    before = [digest(path) for path in paths]
    with pytest.raises(DicomInputError, match="^multidimensional_unsupported$"):
        read_dicom_input(paths)
    summary = inspect_dicom_files(paths)
    assert summary["supported_object_count"] == 0
    assert summary["rejection_counts"] == {"multidimensional_unsupported": len(paths)}
    assert [digest(path) for path in paths] == before


@pytest.mark.parametrize("count", [1, 2])
def test_declared_multiple_times_require_explicit_temporal_position(tmp_path, count):
    paths = [make_mr(tmp_path, index, NumberOfTemporalPositions=2,
                     TemporalPositionIdentifier=None) for index in range(1, count + 1)]
    with pytest.raises(DicomInputError, match="^multidimensional_unsupported$"):
        read_dicom_input(paths)
    summary = inspect_dicom_files(paths)
    assert summary["input_support"] == "unsupported"
    assert summary["rejection_counts"] == {"multidimensional_unsupported": count}


@pytest.mark.parametrize("identifier", [0, -1, 3])
def test_temporal_identifier_must_be_within_declared_position_range(tmp_path, identifier):
    path = make_mr(tmp_path, NumberOfTemporalPositions=2, TemporalPositionIdentifier=identifier)
    with pytest.raises(DicomInputError, match="^multidimensional_unsupported$"):
        read_dicom_input([path])


@pytest.mark.parametrize("identifier", [1, 2])
def test_common_valid_temporal_subset_remains_a_supported_spatial_stack(tmp_path, identifier):
    paths = [make_mr(tmp_path, index, NumberOfTemporalPositions=2,
                     TemporalPositionIdentifier=identifier) for index in (1, 2)]
    volume = read_dicom_input(paths)
    assert volume.input_mode == "3d" and volume.image_dhw.shape == (2, 2, 3)
    assert inspect_dicom_files(paths)["input_support"] == "supported_3d"


def test_irregular_stack_and_mode_mismatch_are_rejected(tmp_path):
    paths = [make_mr(tmp_path, i) for i in (1, 2)]
    paths.append(make_mr(tmp_path, 3, ImagePositionPatient=[10, 20, 10]))
    with pytest.raises(DicomInputError, match="geometry_unsupported"):
        read_dicom_input(paths)
    with pytest.raises(DicomInputError, match="input_mode_mismatch"):
        read_dicom_input(paths[:1], input_mode="3d")
    with pytest.raises(DicomInputError, match="input_mode_mismatch"):
        read_dicom_input(paths[:2], input_mode="2d")


def test_source_integrity_check_detects_modification(tmp_path):
    path = make_mr(tmp_path)
    image = read_dicom_input([path])
    path.write_bytes(path.read_bytes() + b"changed")
    with pytest.raises(DicomInputError, match="source_integrity_failed"):
        verify_sources(image.source_refs)


def test_relative_source_remains_verifiable_after_working_directory_changes(tmp_path, monkeypatch):
    path = make_mr(tmp_path)
    monkeypatch.chdir(tmp_path)
    image = read_dicom_input([path.name])
    monkeypatch.chdir(tmp_path.parent)
    assert image.verify_sources() is True
    assert image.source_refs[0].path == path


def test_area_matches_the_accepted_patient_space_geometry(tmp_path):
    # Rounded IOP values may be within tolerance without having exactly unit norm.
    path = make_mr(tmp_path, ImageOrientationPatient=[0.99999, 0, 0, 0, 1, 0])
    image = read_dicom_input([path])
    assert image.pixel_area_mm2 == pytest.approx(5.99994, rel=1e-9)


def test_modality_lut_is_explicitly_unsupported(tmp_path):
    from pydicom.dataset import Dataset
    from pydicom.sequence import Sequence
    lut = Dataset()
    lut.LUTDescriptor = [2, 0, 16]
    lut.LUTData = b"\0\0\1\0"
    path = make_mr(tmp_path, ModalityLUTSequence=Sequence([lut]))
    with pytest.raises(DicomInputError, match="intensity_transform_unsupported"):
        read_dicom_input([path])


def test_inspection_is_metadata_only_and_anonymous_even_without_pixel_data(tmp_path):
    good = make_mr(tmp_path, 1)
    bad = make_mr(tmp_path, 2, ImagePositionPatient=None)
    for path in (good, bad):
        ds = pydicom.dcmread(path)
        del ds.PixelData
        ds.save_as(path, enforce_file_format=True)
    report = inspect_dicom_files([good, bad])
    assert report["metadata_only"] is True
    assert report["object_count"] == 2
    assert report["classic_mr_count"] == 2
    assert report["supported_object_count"] == 1
    assert report["rejection_counts"] == {"geometry_unsupported": 1}
    output = json.dumps(report)
    assert str(tmp_path) not in output
    assert "SYNTHETIC_PRIVATE" not in output
    assert "1.2.826" not in output


def test_cli_outputs_anonymous_metadata_summary_and_safe_read_errors(tmp_path):
    path = make_mr(tmp_path)
    command = [sys.executable, "scripts/inspect_aur_dicom.py", "--input-files", str(path), "--limit", "32"]
    result = subprocess.run(command, capture_output=True, text=True, check=False)
    assert result.returncode == 0
    assert json.loads(result.stdout)["object_count"] == 1
    assert str(tmp_path) not in result.stdout + result.stderr
    missing = str(tmp_path / "SYNTHETIC_PRIVATE_MISSING.dcm")
    result = subprocess.run(command[:3] + [missing], capture_output=True, text=True, check=False)
    assert result.returncode == 0
    assert json.loads(result.stdout)["rejection_counts"] == {"read_failed": 1}
    assert missing not in result.stdout + result.stderr


def test_empty_input_and_invalid_mode_have_safe_errors():
    with pytest.raises(DicomInputError, match="empty_input"):
        read_dicom_input([])
    with pytest.raises(DicomInputError, match="input_mode_unsupported"):
        read_dicom_input([], input_mode="4d")


def test_partial_rescale_cannot_silently_invent_the_other_coefficient(tmp_path):
    path = make_mr(tmp_path, RescaleSlope=2)
    with pytest.raises(DicomInputError, match="intensity_transform_unsupported"):
        read_dicom_input([path])


def test_malformed_identifier_is_rejected_without_emitting_private_warnings(tmp_path, caplog):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        path = make_mr(tmp_path, SOPInstanceUID="SYNTHETIC_PRIVATE_INVALID_UID")
    caplog.clear()
    with warnings.catch_warnings(record=True) as emitted:
        warnings.simplefilter("always")
        with pytest.raises(DicomInputError, match="identifiers_unsupported"):
            read_dicom_input([path])
        report = inspect_dicom_files([path])
    assert emitted == []
    assert caplog.records == []
    assert report["rejection_counts"] == {"identifiers_unsupported": 1}


def test_cli_limit_is_explicit_and_invalid_arguments_do_not_echo_paths(tmp_path):
    paths = [make_mr(tmp_path, index) for index in (1, 2)]
    command = [sys.executable, "scripts/inspect_aur_dicom.py", "--input-files", *map(str, paths)]
    result = subprocess.run(command + ["--limit", "1"], capture_output=True, text=True)
    report = json.loads(result.stdout)
    assert report["object_count"] == 1
    assert report["provided_object_count"] == 2
    assert report["truncated"] is True
    result = subprocess.run(command + ["--limit", "SYNTHETIC_PRIVATE_INVALID"],
                            capture_output=True, text=True)
    assert result.returncode == 2
    assert "SYNTHETIC_PRIVATE" not in result.stdout + result.stderr
