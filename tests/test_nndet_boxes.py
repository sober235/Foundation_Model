# tests/test_nndet_boxes.py
import importlib.util
import json
from pathlib import Path

import pytest

from anatobind.nndet.boxes import LAYOUT, detections, lesion_rows, load_runner_json, to_corner_box


def test_corner_box_reorders_the_runner_layout():
    assert to_corner_box([1, 2, 3, 4, 5, 6]) == (5.0, 2.0, 1.0, 6.0, 4.0, 3.0)


def test_runner_and_reader_share_the_layout_string():
    path = Path(__file__).resolve().parents[1] / "scripts/nndet_runner.py"
    spec = importlib.util.spec_from_file_location("nndet_runner", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    assert mod.LAYOUT == LAYOUT


def test_reader_refuses_foreign_layout_and_other_classes(tmp_path):
    p = tmp_path / "r.json"
    p.write_text(json.dumps({"layout": "other", "cases": {}}))
    with pytest.raises(ValueError, match="layout"):
        load_runner_json(p)
    with pytest.raises(ValueError, match="class"):
        detections({"boxes": [[0, 0, 1, 1, 0, 1]], "scores": [0.5], "labels": [1]})
    with pytest.raises(ValueError, match="length"):
        detections({"boxes": [[0, 0, 1, 1, 0, 1]], "scores": []})
    assert detections({"boxes": [[0, 1, 2, 3, 4, 5]], "scores": [0.25]}) == [
        {"box": (4.0, 1.0, 0.0, 5.0, 3.0, 2.0), "score": 0.25, "family": "small_lesion"}]


def test_lesion_rows_round_half_up_keep_one_voxel_clip_drop_and_sort():
    dets = [{"box": (2.5, 1.4, 0.6, 4.4, 3.5, 1.2), "score": 0.4, "family": "small_lesion"},    # sub-voxel in z
            {"box": (-3.0, 10.0, 1.0, 2.0, 14.0, 3.0), "score": 0.9, "family": "small_lesion"},   # left part outside
            {"box": (30.0, 0.0, 0.0, 34.0, 2.0, 1.0), "score": 0.7, "family": "small_lesion"}]    # fully outside
    assert lesion_rows(dets, (20, 16, 4)) == [
        {"z0": 1, "z1": 2, "score": 0.9, "boxes": {"1": [[10, 14, 0, 2]], "2": [[10, 14, 0, 2]]}},
        {"z0": 1, "z1": 1, "score": 0.4, "boxes": {"1": [[1, 4, 3, 4]]}},
    ]
    assert lesion_rows([], (20, 16, 4)) == []
