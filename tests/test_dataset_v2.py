import numpy as np
import pytest

torch = pytest.importorskip("torch")

from synth import write_boxes

from anatobind.train.dataset import UNKNOWN_HOST
from anatobind.train.dataset_v2 import (
    DEGRADED, NONE_HOST, NUM_CLASSES, WholeVolumeDataset, collate_one, flip_depth, pad_depth, read_all_boxes,
)


def test_every_layer_becomes_a_target_with_its_host_encoding(synthetic_m1r):
    export, _, scans = synthetic_m1r
    rows = read_all_boxes(export / scans[0] / "boxes.csv")
    assert [r["cls"] for r in rows] == [1, 0, 2, 3]
    assert [r["host_label"] for r in rows] == [2, 5, NONE_HOST, UNKNOWN_HOST]
    assert [r["tissue_id"] for r in rows] == [4, 1, -1, 2]
    assert rows[0]["box"] == (3, 10, 10, 8, 20, 16)          # (x, y, z) export box in the (z, y, x) frame
    assert NUM_CLASSES == 4 and NONE_HOST == 7


def test_an_unresolved_side_is_unknown_not_none(tmp_path):
    write_boxes(tmp_path / "boxes.csv", [{"ann_id": 9, "layer": "in_seg", "supercategory": "Meniscal Tear",
                                          "tissue_id": 1, "host_label": "", "host_side": "unresolved",
                                          "x0": 1, "y0": 1, "z0": 1, "x1": 3, "y1": 3, "z1": 3}])
    assert read_all_boxes(tmp_path / "boxes.csv")[0]["host_label"] == UNKNOWN_HOST


def test_depth_is_padded_to_a_multiple_of_32(synthetic_m1r):
    export, cache, scans = synthetic_m1r
    s = WholeVolumeDataset(scans[:1], cache, export, train=False, views=("clean",))[0]
    assert s["image"].shape == (1, 32, 64, 64) and s["valid_depth"] == 30
    assert (s["seg"][30:] == 0).all()
    assert np.allclose(s["image"][0, 30:], s["image"][0, :30].min())


def test_an_aligned_depth_is_left_alone():
    out, d = pad_depth(np.zeros((64, 2, 2)))
    assert out.shape == (64, 2, 2) and d == 64


def test_the_flip_keeps_boxes_on_the_same_voxels():
    rng = np.random.default_rng(0)
    image = rng.random((30, 8, 8)).astype(np.float32)
    seg = np.zeros((30, 8, 8), np.int64)
    seg[3:7, 2:4, 2:4] = 5
    fi, fs, fb = flip_depth(image, seg, np.array([[3, 2, 2, 7, 4, 4]], np.float32), 30)
    z0, y0, x0, z1, y1, x1 = fb[0].astype(int)
    assert (fs[z0:z1, y0:y1, x0:x1] == 5).all() and fs.sum() == seg.sum()
    assert np.array_equal(fi[::-1], image)


def test_training_draws_the_clean_view_half_of_the_time(synthetic_m1r):
    export, cache, scans = synthetic_m1r
    ds = WholeVolumeDataset(scans, cache, export, train=True, seed=0)
    draws = [ds._view() for _ in range(6000)]
    assert 0.47 < draws.count("clean") / 6000 < 0.53
    for v in DEGRADED:
        assert 0.065 < draws.count(v) / 6000 < 0.10


def test_evaluation_enumerates_every_scan_and_view_in_order(synthetic_m1r):
    export, cache, scans = synthetic_m1r
    ds = WholeVolumeDataset(scans[:2], cache, export, train=False)
    assert len(ds) == 14
    assert (ds[3]["scan_id"], ds[3]["view"]) == (scans[0], "noise_q3")
    assert (ds[7]["scan_id"], ds[7]["view"]) == (scans[1], "clean")


def test_collate_one_adds_the_batch_axis(synthetic_m1r):
    export, cache, scans = synthetic_m1r
    b = collate_one([WholeVolumeDataset(scans[:1], cache, export, train=False, views=("us16",))[0]])
    assert b["image"].shape == (1, 1, 32, 64, 64) and b["seg"].shape == (1, 32, 64, 64)
    assert b["boxes"][0].shape == (4, 6) and b["view"] == ["us16"] and b["valid_depth"].tolist() == [30]


def test_collate_batch_pads_every_volume_to_the_deepest_member(tmp_path):
    from synth import DEFAULT_BOXES, write_synthetic_scan

    from anatobind.train.cache import cache_scan
    from anatobind.train.dataset_v2 import collate_batch

    for scan, depth in (("MTR_001", 30), ("MTR_002", 40)):
        d = write_synthetic_scan(tmp_path / "exp", scan, DEFAULT_BOXES, shape=(64, 64, depth))
        cache_scan(d, tmp_path / "cache" / scan)
    ds = WholeVolumeDataset(["MTR_001", "MTR_002"], tmp_path / "cache", tmp_path / "exp", train=False, views=("clean",))
    b = collate_batch([ds[0], ds[1]])
    assert b["image"].shape == (2, 1, 64, 64, 64) and b["seg"].shape == (2, 64, 64, 64)
    assert b["valid_depth"].tolist() == [30, 40]
    assert (b["seg"][0, 30:] == 0).all()
    assert (b["image"][0, 0, 30:] == b["image"][0, 0, :30].min()).all()
    assert len(b["boxes"]) == 2 and b["view"] == ["clean", "clean"]
