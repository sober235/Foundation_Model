import numpy as np
import pytest

from anatobind.data_engine.fastmri import (
    BOX_CONVENTION_RSS, TRANSFORM_VERSION, convert_box_csv_to_rss, convert_box_rss_to_csv, rss_spacing_mm,
    voxel_to_world, world_to_voxel,
)

# a knee header: 640 x 368 acquired over 280 x 161.42 mm, reconSpace 320 x 320 over 140 mm, 3 mm slices
HEADER = (
    '<?xml version="1.0" encoding="utf-8"?>'
    '<ismrmrdHeader xmlns="http://www.ismrm.org/ISMRMRD"><encoding>'
    "<encodedSpace><matrixSize><x>640</x><y>368</y><z>1</z></matrixSize>"
    "<fieldOfView_mm><x>280</x><y>161.42</y><z>4.5</z></fieldOfView_mm></encodedSpace>"
    "<reconSpace><matrixSize><x>320</x><y>320</y><z>1</z></matrixSize>"
    "<fieldOfView_mm><x>140</x><y>140</y><z>3</z></fieldOfView_mm></reconSpace>"
    "</encoding></ismrmrdHeader>"
)


def _label_as_fastmri_plus_did(rss_slice):
    """What the annotator saw was the RSS slice flipped up/down; the CSV box is drawn around the bright block
    in that flipped image: (x, y, width, height) with y counted from the top of the FLIPPED image."""
    flipped = rss_slice[::-1]
    rows, cols = np.nonzero(flipped > 0)
    return int(cols.min()), int(rows.min()), int(cols.max() - cols.min() + 1), int(rows.max() - rows.min() + 1)


def test_known_bright_block_is_covered_exactly_after_conversion():
    n_rows = 320
    img = np.zeros((n_rows, 320), np.float32)
    row0, row1, col0, col1 = 40, 61, 200, 233               # the block lives in rows 40..60 of the RSS array
    img[row0:row1, col0:col1] = 1.0
    x, y, w, h = _label_as_fastmri_plus_did(img)
    assert (y, h) == (n_rows - row1, row1 - row0)           # i.e. the CSV y is measured from the bottom
    assert convert_box_csv_to_rss(x, y, w, h, n_rows) == (row0, row1, col0, col1)
    r0, r1, c0, c1 = convert_box_csv_to_rss(x, y, w, h, n_rows)
    assert img[r0:r1, c0:c1].all() and img.sum() == (r1 - r0) * (c1 - c0)


def test_using_the_csv_box_as_is_misses_the_block():
    img = np.zeros((320, 320), np.float32)
    img[40:61, 200:233] = 1.0
    x, y, w, h = _label_as_fastmri_plus_did(img)
    assert img[y:y + h, x:x + w].sum() == 0                 # what leg 2 trained on before Gate 0


@pytest.mark.parametrize("n_rows", [320, 276, 260, 213])
def test_csv_to_rss_to_csv_round_trip_is_exact(n_rows):
    rng = np.random.default_rng(n_rows)
    for _ in range(200):
        y = int(rng.integers(0, n_rows - 3))
        x = int(rng.integers(0, 300))
        h = int(rng.integers(3, n_rows - y + 1))
        w = int(rng.integers(3, 40))
        assert convert_box_rss_to_csv(*convert_box_csv_to_rss(x, y, w, h, n_rows), n_rows) == (x, y, w, h)


def test_rss_spacing_is_acquired_resolution_in_plane_and_recon_fov_z_through_plane():
    sp = rss_spacing_mm(HEADER)
    assert sp[0] == pytest.approx(3.0)
    assert sp[1] == pytest.approx(280 / 640)
    assert sp[2] == pytest.approx(161.42 / 368)


def test_voxel_and_world_are_index_times_spacing_and_invert_each_other():
    sp = (3.0, 0.4375, 0.4386)
    assert voxel_to_world((2, 10, 100), sp) == pytest.approx((6.0, 4.375, 43.86))
    assert world_to_voxel(voxel_to_world((2, 10, 100), sp), sp) == pytest.approx((2.0, 10.0, 100.0))


def test_the_convention_constants_are_what_the_manifest_will_record():
    assert TRANSFORM_VERSION == 2
    assert BOX_CONVENTION_RSS == "rss_rows_from_top"
