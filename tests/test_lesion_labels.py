import numpy as np
from synth import DEFAULT_BOXES, write_synthetic_export

from anatobind.nnunet.lesion_labels import (
    FAMILY_LABELS, FAMILY_OF_LABEL, LABELS, NAME_OF_LABEL, box_volume, boxes_to_label_map, paint_order,
    read_boxes_xyz,
)


def _row(family, box, ann_id=1):
    return {"ann_id": ann_id, "supercategory": family, "layer": "in_seg", "tissue_id": 4, "host_label": 2,
            "host_side": "single", "box": box}


def test_label_tables_agree_with_each_other():
    assert LABELS == {"background": 0, "cartilage_lesion": 1, "meniscal_tear": 2, "ligament_tear": 3, "effusion": 4}
    assert FAMILY_OF_LABEL[FAMILY_LABELS["Effusion"]] == "Effusion"
    assert NAME_OF_LABEL == {1: "cartilage_lesion", 2: "meniscal_tear", 3: "ligament_tear", 4: "effusion"}


def test_disjoint_boxes_paint_their_own_voxels_and_nothing_else():
    rows = [_row("Cartilage Lesion", (0, 0, 0, 2, 3, 4)), _row("Meniscal Tear", (5, 5, 5, 7, 7, 7), 2)]
    lab = boxes_to_label_map(rows, (10, 10, 10))
    assert lab.dtype == np.uint8 and lab.shape == (10, 10, 10)
    assert (lab == 1).sum() == box_volume((0, 0, 0, 2, 3, 4)) == 24
    assert (lab == 2).sum() == 8 and (lab == 0).sum() == 1000 - 24 - 8


def test_a_small_box_inside_an_effusion_box_wins_the_overlap():
    big = _row("Effusion", (0, 0, 0, 10, 10, 10))
    small = _row("Cartilage Lesion", (2, 2, 2, 4, 4, 4), 2)
    lab = boxes_to_label_map([big, small], (10, 10, 10))
    assert (lab == 1).sum() == 8 and (lab == 4).sum() == 1000 - 8
    # painting order is independent of the input order
    assert np.array_equal(lab, boxes_to_label_map([small, big], (10, 10, 10)))


def test_between_two_non_effusion_boxes_the_smaller_one_wins():
    large = _row("Meniscal Tear", (0, 0, 0, 6, 6, 6))
    small = _row("Ligament Tear", (4, 4, 4, 8, 8, 8), 2)
    lab = boxes_to_label_map([large, small], (10, 10, 10))
    assert lab[5, 5, 5] == 3 and lab[1, 1, 1] == 2
    assert paint_order([small, large]) == [1, 0]


def test_effusion_is_painted_first_even_when_it_is_smaller():
    eff = _row("Effusion", (0, 0, 0, 2, 2, 2))
    tear = _row("Meniscal Tear", (0, 0, 0, 5, 5, 5), 2)
    assert paint_order([tear, eff]) == [1, 0]
    assert boxes_to_label_map([tear, eff], (6, 6, 6))[0, 0, 0] == 2


def test_boxes_are_clipped_to_the_volume_and_degenerate_boxes_paint_nothing():
    lab = boxes_to_label_map([_row("Cartilage Lesion", (8, 8, 8, 20, 20, 20)), _row("Effusion", (1, 1, 1, 1, 5, 5), 2)],
                             (10, 10, 10))
    assert (lab == 1).sum() == 8 and (lab == 4).sum() == 0


def test_read_boxes_xyz_returns_every_row_in_the_export_frame(tmp_path):
    write_synthetic_export(tmp_path, ["MTR_001"], {"MTR_001": 0})
    rows = read_boxes_xyz(tmp_path / "MTR_001" / "boxes.csv")
    assert [r["supercategory"] for r in rows] == [b["supercategory"] for b in DEFAULT_BOXES]
    assert rows[0]["box"] == (10, 10, 3, 16, 20, 8) and all(isinstance(v, int) for v in rows[0]["box"])
    assert rows[0]["host_label"] == 2 and rows[2]["host_label"] is None and rows[3]["layer"] == "ligament"
    lab = boxes_to_label_map(rows, (64, 64, 30))
    assert lab[12, 15, 5] == 1 and lab[35, 30, 7] == 2 and lab[22, 48, 15] == 3 and lab[50, 45, 15] == 4
