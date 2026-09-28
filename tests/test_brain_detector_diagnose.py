import importlib.util
from pathlib import Path

import numpy as np

spec = importlib.util.spec_from_file_location("dbd", Path(__file__).resolve().parents[1] / "scripts/diagnose_brain_detector.py")
dbd = importlib.util.module_from_spec(spec)
spec.loader.exec_module(dbd)


def _synthetic_case():
    """One case, 4 registry lesions on a (42, 42, 8) grid:
    - A is touched only by a 4-voxel blob that exactly matches A's box (IoU 1.0, not below 0.1).
    - B and C are touched by one 30-voxel diagonal component (26-connectivity) that starts inside B's box
      and ends inside C's box; its bounding box (10,10,0,40,40,1) dwarfs either box, so IoU << 0.1 for both.
    - D is untouched by anything.
    Hand-computed (brief final-fix-brief.md group 3): touched all 3 / decoded 2 (the 4-voxel blob is below
    BRAIN_MIN_VOXELS=9), histograms all {1: 1, 2: 1} / decoded {2: 1}.
    """
    shape = (42, 42, 8)
    label_map = np.zeros(shape, np.uint8)
    label_map[0:2, 0:2, 0] = 1              # 4-voxel blob, exactly A's box
    for i in range(30):
        label_map[10 + i, 10 + i, 0] = 1    # 30-voxel diagonal component, spans B's box to C's box

    boxes = [
        (0, 0, 0, 2, 2, 1),      # A: exactly the blob
        (10, 10, 0, 12, 12, 1),  # B: overlaps the diagonal's start
        (38, 38, 0, 40, 40, 1),  # C: overlaps the diagonal's end
        (0, 0, 5, 2, 2, 6),      # D: untouched
    ]
    return label_map, boxes


def test_diagnose_case_touched_and_iou_below_hand_computed_population_all():
    label_map, boxes = _synthetic_case()
    per_lesion, component_hits = dbd.diagnose_case(label_map, boxes, min_voxels=1)
    # A: touched by the 4-voxel blob; the blob's bbox exactly equals A's box -> IoU 1.0, not below 0.1
    assert per_lesion[0] == {"touched": True, "iou_below": False}
    # B, C: touched by the 30-voxel diagonal whose bbox (10,10,0,40,40,1) (vol 900) dwarfs either 4-voxel
    # box -> IoU = 4/900 = 0.0044, below 0.1
    assert per_lesion[1] == {"touched": True, "iou_below": True}
    assert per_lesion[2] == {"touched": True, "iou_below": True}
    # D: untouched by any component
    assert per_lesion[3] == {"touched": False, "iou_below": False}
    # the diagonal component touches 2 of this case's lesion boxes (B and C); the blob touches 1 (A)
    assert sorted(component_hits.values()) == [1, 2]


def test_diagnose_case_decoded_population_drops_the_undersized_blob():
    label_map, boxes = _synthetic_case()
    per_lesion, component_hits = dbd.diagnose_case(label_map, boxes, min_voxels=9)
    # A's only touching component (4 voxels) is below the 9-voxel floor: no longer touched
    assert per_lesion[0] == {"touched": False, "iou_below": False}
    assert per_lesion[1] == {"touched": True, "iou_below": True}
    assert per_lesion[2] == {"touched": True, "iou_below": True}
    assert per_lesion[3] == {"touched": False, "iou_below": False}
    assert sorted(component_hits.values()) == [2]


def test_summarize_population_all_vs_decoded_matches_hand_computed_brief_numbers():
    label_map, boxes = _synthetic_case()
    cases = {"case1": boxes}
    maps = {"case1": label_map}

    all_pop = dbd.summarize_population(cases, maps, min_voxels=1)
    assert all_pop["touched"] == 3
    assert all_pop["iou_below_0.1"] == 2
    assert all_pop["lesions_per_component"] == {1: 1, 2: 1}

    decoded_pop = dbd.summarize_population(cases, maps, min_voxels=9)
    assert decoded_pop["touched"] == 2
    assert decoded_pop["iou_below_0.1"] == 2
    assert decoded_pop["lesions_per_component"] == {2: 1}


def test_box_iou_matches_hand_computed_values():
    # A's box exactly equals the blob's bbox -> IoU 1.0
    assert dbd.box_iou((0, 0, 0, 2, 2, 1), (0, 0, 0, 2, 2, 1)) == 1.0
    # B's box (vol 4) against the diagonal's bbox (vol 900), intersection vol 4 -> 4 / (4 + 900 - 4) = 4/900
    assert abs(dbd.box_iou((10, 10, 0, 12, 12, 1), (10, 10, 0, 40, 40, 1)) - 4 / 900) < 1e-9
    # disjoint boxes -> 0.0
    assert dbd.box_iou((0, 0, 0, 1, 1, 1), (5, 5, 5, 6, 6, 6)) == 0.0
