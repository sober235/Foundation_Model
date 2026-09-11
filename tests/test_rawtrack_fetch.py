import io
import tarfile
import zipfile

import pytest

from anatobind.data_engine.rawtrack_fetch import check_complete, extract_raw_track, select_members

NAMES = [
    "skm-tea/segmentation_masks/raw-data-track/MTR_001.nii.gz",
    "skm-tea/segmentation_masks/dicom-track/MTR_001.nii.gz",
    "raw-data-track/MTR_005.nii.gz",
    "raw-data-track/README.txt",
    "segmentation_masks/raw-data-track-old/MTR_006.nii.gz",
]


def _payload(name):
    return f"payload of {name}".encode()


def test_only_raw_track_niftis_are_selected():
    assert select_members(NAMES) == {NAMES[0]: "MTR_001.nii.gz", NAMES[2]: "MTR_005.nii.gz"}


def test_extract_from_a_tar_keeps_the_raw_track_copy(tmp_path):
    arc = tmp_path / "a.tar.gz"
    with tarfile.open(arc, "w:gz") as tar:
        for n in NAMES:
            data = _payload(n)
            info = tarfile.TarInfo(n)
            info.size = len(data)
            tar.addfile(info, io.BytesIO(data))
    assert extract_raw_track(arc, tmp_path / "out") == ["MTR_001.nii.gz", "MTR_005.nii.gz"]
    # the raw-data-track member, not the dicom-track one with the same file name
    assert (tmp_path / "out/MTR_001.nii.gz").read_bytes() == _payload(NAMES[0])


def test_extract_from_a_zip(tmp_path):
    arc = tmp_path / "a.zip"
    with zipfile.ZipFile(arc, "w") as zf:
        for n in NAMES:
            zf.writestr(n, _payload(n))
    assert extract_raw_track(arc, tmp_path / "out") == ["MTR_001.nii.gz", "MTR_005.nii.gz"]


def test_a_file_that_is_not_an_archive_is_refused(tmp_path):
    p = tmp_path / "x.bin"
    p.write_bytes(b"not an archive")
    with pytest.raises(ValueError):
        extract_raw_track(p, tmp_path / "out")


def test_completeness_reports_missing_and_unexpected_scans():
    assert check_complete(["MTR_001.nii.gz", "MTR_999.nii.gz"], ["MTR_001", "MTR_005"]) == (["MTR_005"], ["MTR_999"])
