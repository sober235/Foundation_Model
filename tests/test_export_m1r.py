import csv
import os

import nibabel as nib
import numpy as np

from anatobind.data_engine.skmtea import link_files, write_seg_and_boxes

ROW = {"ann_id": 1, "split": "train", "layer": "in_seg", "supercategory": "Meniscal Tear", "category_id": 3,
       "tissue_id": 1, "keep": True, "x0": 0, "y0": 0, "z0": 0, "x1": 2, "y1": 4, "z1": 2}


def _seg(medial_first=True):
    s = np.zeros((8, 8, 4), np.uint8)
    s[0:3], s[5:8] = (5, 6) if medial_first else (6, 5)
    return s


def _rows(path):
    with open(path, newline="") as fh:
        return list(csv.DictReader(fh))


def test_hosts_are_resolved_on_the_segmentation_it_is_given(tmp_path):
    counts = write_seg_and_boxes(_seg(True), (0.625, 0.625, 0.8), [ROW], tmp_path / "a")
    r = _rows(tmp_path / "a/boxes.csv")[0]
    assert (r["host_label"], r["host_side"]) == ("5", "medial") and counts["n_boxes_kept"] == 1
    write_seg_and_boxes(_seg(False), (0.625, 0.625, 0.8), [ROW], tmp_path / "b")
    r = _rows(tmp_path / "b/boxes.csv")[0]
    assert (r["host_label"], r["host_side"]) == ("6", "lateral")


def test_seg_is_written_at_half_in_plane_resolution(tmp_path):
    write_seg_and_boxes(_seg(True), (0.625, 0.625, 0.8), [ROW], tmp_path)
    img = nib.load(str(tmp_path / "seg.nii.gz"))
    assert img.shape == (4, 4, 4) and np.allclose(img.header.get_zooms(), (0.625, 0.625, 0.8))


def test_rows_dropped_by_rule_d5_are_not_written(tmp_path):
    write_seg_and_boxes(_seg(True), (0.625, 0.625, 0.8), [{**ROW, "keep": False}], tmp_path)
    assert _rows(tmp_path / "boxes.csv") == []


def test_link_files_hard_links_and_is_idempotent(tmp_path):
    (tmp_path / "src").mkdir()
    (tmp_path / "src/a.nii.gz").write_bytes(b"x")
    link_files(tmp_path / "src", tmp_path / "dst", ["a.nii.gz"])
    link_files(tmp_path / "src", tmp_path / "dst", ["a.nii.gz"])
    assert os.stat(tmp_path / "dst/a.nii.gz").st_ino == os.stat(tmp_path / "src/a.nii.gz").st_ino
