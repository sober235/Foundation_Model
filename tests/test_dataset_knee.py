import json

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from anatobind.data_engine.fastmri_knee import VIEWS
from anatobind.train.dataset_knee import SlabDataset, collate_slabs


def _export(tmp_path, name="file1", slices=9, size=32):
    d = tmp_path / name
    d.mkdir(parents=True)
    rng = np.random.default_rng(0)
    for v in VIEWS:
        np.save(d / f"{v}.npy", rng.normal(size=(slices, size, size)).astype(np.float16))
    (d / "meta.json").write_text(json.dumps({"slices": slices, "size": size, "patient_id": "p1"}))
    lesions = [{"lesion_id": 0, "file": name, "family": "meniscus", "z0": 3, "z1": 5,
                "x0": 4, "y0": 6, "x1": 14, "y1": 18, "n_boxes": 3}]
    return d.parent, lesions


def test_a_sample_is_a_slab_centred_on_its_slice(tmp_path):
    root, lesions = _export(tmp_path)
    ds = SlabDataset(["file1"], root, lesions=lesions, train=False, views=("clean",), slab=5)
    s = ds[0]
    assert s["image"].shape == (1, 5, 32, 32)
    assert s["view"] == "clean" and 0 <= s["slice"] < 9


def test_slabs_at_the_volume_edge_replicate_rather_than_wrap(tmp_path):
    _export(tmp_path, name="file1")
    root, _ = _export(tmp_path, name="file2")
    ds = SlabDataset(["file2"], root, lesions=[], train=False, views=("clean",), slab=5)
    first = [s for s in (ds[i] for i in range(len(ds))) if s["slice"] == 0][0]["image"][0]
    assert first.shape == (5, 32, 32)
    # slice 0 draws indices clip([-2,-1,0,1,2]) = [0,0,0,1,2]: the first three planes are the same
    torch.testing.assert_close(first[0], first[1])
    torch.testing.assert_close(first[1], first[2])
    assert not torch.allclose(first[2], first[3])
    last = [s for s in (ds[i] for i in range(len(ds))) if s["slice"] == 8][0]["image"][0]
    torch.testing.assert_close(last[3], last[4])           # the far edge replicates too, it does not wrap


def test_only_boxed_slices_come_from_an_annotated_volume(tmp_path):
    root, lesions = _export(tmp_path)                       # file1: 9 slices, one lesion on slices 3-5
    ds = SlabDataset(["file1"], root, lesions=lesions, train=False, views=("clean",))
    assert sorted({ds[i]["slice"] for i in range(len(ds))}) == [3, 4, 5]


def test_every_slice_comes_from_a_volume_with_no_annotation(tmp_path):
    _export(tmp_path, name="file1")
    root, _ = _export(tmp_path, name="file2")
    ds = SlabDataset(["file2"], root, lesions=[], train=False, views=("clean",))
    assert sorted({ds[i]["slice"] for i in range(len(ds))}) == list(range(9))
    assert all(len(ds[i]["boxes"]) == 0 for i in range(len(ds)))


def test_a_slice_inside_a_lesion_carries_its_box(tmp_path):
    root, lesions = _export(tmp_path)
    ds = SlabDataset(["file1"], root, lesions=lesions, train=False, views=("clean",), slab=5)
    boxed = [s for s in (ds[i] for i in range(len(ds))) if len(s["boxes"])]
    assert boxed, "slices 3-5 must carry the lesion box"
    assert {int(s["slice"]) for s in boxed} == {3, 4, 5}
    b = boxed[0]["boxes"][0]
    assert b.tolist() == [6.0, 4.0, 18.0, 14.0]                 # y0, x0, y1, x1
    assert int(boxed[0]["box_classes"][0]) == 0                 # meniscus is family index 0


def test_collate_stacks_slabs_and_keeps_boxes_per_sample(tmp_path):
    root, lesions = _export(tmp_path)
    ds = SlabDataset(["file1"], root, lesions=lesions, train=False, views=("clean",), slab=5)
    b = collate_slabs([ds[0], ds[1]])
    assert b["image"].shape == (2, 1, 5, 32, 32)
    assert len(b["boxes"]) == 2 and len(b["file"]) == 2


def test_training_mode_samples_views_and_can_flip(tmp_path):
    root, lesions = _export(tmp_path)
    ds = SlabDataset(["file1"], root, lesions=lesions, train=True, p_clean=0.0, seed=0)
    views = {ds[i]["view"] for i in range(30)}
    assert views and "clean" not in views                       # p_clean=0 never draws the clean view


def test_translation_fills_rather_than_wrapping():
    from anatobind.train.dataset_knee import _shift
    img = np.zeros((1, 4, 4), np.float32)
    img[0, 0, :] = 1.0
    out = _shift(img, 2, 0)
    assert out[0, 2].tolist() == [1.0] * 4          # content moved down by two
    assert out[0, 0].tolist() == [0.0] * 4          # filled
    assert out[0, 3].tolist() == [0.0] * 4          # the top row did NOT reappear at the bottom
    assert _shift(img, 9, 9).sum() == 0.0           # a shift past the image empties it


def test_a_box_whose_centre_leaves_the_image_is_dropped(tmp_path):
    root, lesions = _export(tmp_path)
    ds = SlabDataset(["file1"], root, lesions=lesions, train=True, seed=0)
    for i in range(40):
        s = ds[i % len(ds)]
        h, w = s["image"].shape[-2:]
        for b in s["boxes"]:
            cy, cx = (b[0] + b[2]) / 2, (b[1] + b[3]) / 2
            assert 0 <= cy < h and 0 <= cx < w
        assert len(s["boxes"]) == len(s["box_classes"])


def test_training_mode_only_draws_views_it_was_given(tmp_path):
    root, lesions = _export(tmp_path)
    ds = SlabDataset(["file1"], root, lesions=lesions, train=True, views=("clean", "us16"), p_clean=0.5, seed=0)
    assert {ds[i % len(ds)]["view"] for i in range(60)} <= {"clean", "us16"}
