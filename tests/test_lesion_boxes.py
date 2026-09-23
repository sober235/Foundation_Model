import nibabel as nib
import numpy as np
import pytest

from anatobind.eval.lesion_boxes import MIN_VOXELS, decode_boxes, load_label_map, load_nnunet_probabilities


def _map():
    m = np.zeros((20, 20, 20), np.uint8)
    m[2:6, 2:6, 2:6] = 1        # cartilage blob A, 64 voxels
    m[12:15, 12:15, 12:15] = 1  # cartilage blob B, 27 voxels
    m[0:10, 14:20, 0:5] = 4     # effusion, 300 voxels
    m[18, 18, 18] = 2           # a 1-voxel meniscal speck: below MIN_VOXELS
    return m


def _onehot(m, n_classes=5):
    p = np.zeros((n_classes,) + m.shape, np.float32)
    for c in range(n_classes):
        p[c] = m == c
    return p


def test_one_box_per_component_with_family_label_and_corners():
    dets = decode_boxes(_map(), _onehot(_map()))
    assert len(dets) == 3 and sorted(d["n_voxels"] for d in dets) == [27, 64, 300]
    fams = sorted((d["family"], d["box"]) for d in dets)
    assert fams == [("Cartilage Lesion", (2, 2, 2, 6, 6, 6)), ("Cartilage Lesion", (12, 12, 12, 15, 15, 15)),
                    ("Effusion", (0, 14, 0, 10, 20, 5))]
    assert all(d["score"] == 1.0 for d in dets) and all(d["label"] in (1, 4) for d in dets)


def test_the_score_is_the_mean_family_probability_inside_the_component():
    m = np.zeros((8, 8, 8), np.uint8)
    m[0:4, 0:4, 0:4] = 2
    p = np.zeros((5, 8, 8, 8), np.float32)
    p[2, 0:4, 0:4, 0:4] = 0.6
    p[2, 0:2, 0:4, 0:4] = 0.8   # half the voxels at 0.8, half at 0.6
    assert decode_boxes(m, p)[0]["score"] == pytest.approx(0.7)


def test_components_below_min_voxels_are_dropped_and_the_fallback_score_grows_with_size():
    dets = decode_boxes(_map())
    assert all(d["n_voxels"] >= MIN_VOXELS for d in dets) and len(dets) == 3
    small = next(d for d in dets if d["n_voxels"] == 27)
    big = next(d for d in dets if d["n_voxels"] == 300)
    assert small["score"] == pytest.approx(0.5) and big["score"] > small["score"]
    assert dets[0]["score"] >= dets[-1]["score"]


def test_decoding_an_empty_map_gives_no_boxes():
    assert decode_boxes(np.zeros((5, 5, 5), np.uint8)) == []


def test_nnunet_probabilities_come_back_in_the_export_frame(tmp_path):
    m = _map()
    p_zyx = np.ascontiguousarray(_onehot(m).transpose(0, 3, 2, 1))          # what nnU-Net writes
    np.savez_compressed(tmp_path / "case.npz", probabilities=p_zyx.astype(np.float16))
    p = load_nnunet_probabilities(tmp_path / "case.npz", m)
    assert p.shape == (5, 20, 20, 20) and p.dtype == np.float32 and np.array_equal(p.argmax(0), m)
    with pytest.raises(ValueError):
        load_nnunet_probabilities(tmp_path / "case.npz", np.roll(m, 7, axis=0))


def test_load_label_map_keeps_the_nifti_array_order(tmp_path):
    m = _map()
    nib.save(nib.Nifti1Image(m, np.diag([0.625, 0.625, 0.8, 1.0])), str(tmp_path / "seg.nii.gz"))
    assert np.array_equal(load_label_map(tmp_path / "seg.nii.gz"), m)
