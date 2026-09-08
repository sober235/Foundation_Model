import json

import pytest

from anatobind.data_engine.skmtea import (
    SUPER_TO_TISSUES, TISSUE_TO_SEG, layer_of, normalise_box, screen_annotation, screen_split,
)

IMAGE = {"id": 7, "scan_id": "MTR_999", "matrix_shape": [512, 512, 160]}


def test_label_tables():
    assert TISSUE_TO_SEG[1] == (5, 6) and TISSUE_TO_SEG[6] == (3, 4)
    assert TISSUE_TO_SEG[4] == (2,) and TISSUE_TO_SEG[5] == (1,)
    assert SUPER_TO_TISSUES["Cartilage Lesion"] == {4, 5, 6}
    assert SUPER_TO_TISSUES["Effusion"] == {-1}


def test_layer_of():
    assert [layer_of(t) for t in (1, 4, 5, 6)] == ["in_seg"] * 4
    assert layer_of(-1) == "effusion" and layer_of(2) == "ligament" and layer_of(3) == "ligament"


def test_normalise_box_rounds_and_orders_endpoints():
    assert normalise_box([330.0, 232.0, 54.0, 5.0, 19.0, 10.0]) == (330, 232, 54, 335, 251, 64)
    assert normalise_box([10.0, 10.0, 10.0, -4.0, 3.0, 2.0]) == (6, 10, 10, 10, 13, 12)


def test_screen_keeps_a_valid_box():
    r = screen_annotation({"id": 1, "tissue_id": 4, "bbox": [10, 20, 30, 5, 6, 7]}, IMAGE, "Cartilage Lesion")
    assert r["keep"] and r["reason"] == "" and r["box"] == (10, 20, 30, 15, 26, 37)


def test_screen_drops_non_positive_extent():
    r = screen_annotation({"id": 1, "tissue_id": 4, "bbox": [10, 20, 30, -5, 6, 7]}, IMAGE, "Cartilage Lesion")
    assert not r["keep"] and r["reason"] == "non-positive extent"


def test_screen_drops_out_of_bounds_after_sorting():
    r = screen_annotation({"id": 1, "tissue_id": 4, "bbox": [500, 20, 30, 20, 6, 7]}, IMAGE, "Cartilage Lesion")
    assert not r["keep"] and r["reason"] == "out of bounds"


def test_screen_drops_category_tissue_mismatch():
    r = screen_annotation({"id": 1, "tissue_id": 2, "bbox": [10, 20, 30, 5, 6, 7]}, IMAGE, "Cartilage Lesion")
    assert not r["keep"] and r["reason"] == "category-tissue mismatch"


def test_screen_split_reads_coco_like_json(tmp_path):
    doc = {
        "categories": [{"id": 12, "name": "Cartilage Lesion (1)", "supercategory": "Cartilage Lesion"},
                       {"id": 16, "name": "Effusion", "supercategory": "Effusion"}],
        "images": [{"id": 7, "scan_id": "MTR_999", "matrix_shape": [512, 512, 160]}],
        "annotations": [
            {"id": 1, "image_id": 7, "category_id": 12, "tissue_id": 4, "bbox": [10, 20, 30, 5, 6, 7]},
            {"id": 2, "image_id": 7, "category_id": 16, "tissue_id": -1, "bbox": [10, 20, 30, 5, 6, -7]},
        ],
    }
    p = tmp_path / "train.json"
    p.write_text(json.dumps(doc))
    rows = screen_split(p)
    assert [r["keep"] for r in rows] == [True, False]
    assert rows[0]["layer"] == "in_seg" and rows[1]["layer"] == "effusion"
    assert rows[0]["split"] == "train" and rows[0]["scan_id"] == "MTR_999" and rows[0]["depth"] == 160
    assert (rows[0]["x0"], rows[0]["x1"]) == (10, 15)
