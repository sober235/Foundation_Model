import numpy as np

from anatobind.data_engine.resample import downsample2_inplane_image, downsample2_inplane_labels, scale_box_inplane


def test_image_downsample_averages_2x2_blocks_and_keeps_z():
    v = np.arange(4 * 4 * 2, dtype=np.float32).reshape(4, 4, 2)
    d = downsample2_inplane_image(v)
    assert d.shape == (2, 2, 2) and d.dtype == np.float32
    assert d[0, 0, 0] == v[0:2, 0:2, 0].mean()


def test_label_downsample_takes_the_majority_with_ties_to_smallest():
    lab = np.zeros((2, 2, 1), dtype=np.uint8)
    lab[0, 0, 0], lab[0, 1, 0], lab[1, 0, 0], lab[1, 1, 0] = 5, 5, 6, 0
    assert downsample2_inplane_labels(lab)[0, 0, 0] == 5
    lab[0, 1, 0] = 6  # now 5,6,6,0 -> 6
    assert downsample2_inplane_labels(lab)[0, 0, 0] == 6
    lab[:] = 0
    lab[0, 0, 0], lab[1, 1, 0] = 3, 4  # two zeros beat one 3 and one 4 -> 0
    assert downsample2_inplane_labels(lab)[0, 0, 0] == 0
    lab[:, :, 0] = [[3, 4], [4, 3]]  # exact 2/2 tie between 3 and 4 -> smallest label id 3
    assert downsample2_inplane_labels(lab)[0, 0, 0] == 3


def test_scale_box_inplane_floors_mins_and_ceils_maxes():
    assert scale_box_inplane((330, 232, 54, 335, 251, 64)) == (165, 116, 54, 168, 126, 64)
