import numpy as np

from anatobind.data_engine.skmtea import host_seg_label


def _seg():
    s = np.zeros((40, 40, 10), dtype=np.uint8)
    s[5:15, :, :] = 5   # medial meniscus block
    s[25:35, :, :] = 6  # lateral meniscus block
    s[:, :, 8:] = 2     # femoral cartilage slab at the top slices
    return s


def test_meniscus_box_on_medial_side():
    r = host_seg_label(_seg(), (6, 10, 2, 12, 20, 5), tissue_id=1)
    assert r["label"] == 5 and r["side"] == "medial" and r["ratio"] > 0.6


def test_meniscus_box_straddling_both_sides_is_ambiguous():
    s = _seg()
    s[15:25, :, :] = 0
    r = host_seg_label(s, (12, 10, 2, 28, 20, 5), tissue_id=1, pad=0)
    assert r["label"] is None and r["side"] == "ambiguous"


def test_single_label_tissue_reports_single():
    r = host_seg_label(_seg(), (0, 0, 8, 5, 5, 10), tissue_id=4)
    assert r["label"] == 2 and r["side"] == "single"


def test_single_label_tissue_keeps_its_host_when_the_box_misses_the_mask():
    """tissue_id is the primary host truth (5.1); overlap must not veto it."""
    r = host_seg_label(_seg(), (18, 18, 0, 22, 22, 2), tissue_id=4, pad=0)
    assert r["label"] == 2 and r["side"] == "single_no_overlap" and r["n_voxels"] == 0


def test_two_label_tissue_with_no_overlap_is_unresolved_not_none():
    """'none' is the effusion/ligament value; an in_seg row must stay distinguishable from those."""
    r = host_seg_label(_seg(), (18, 18, 0, 22, 22, 2), tissue_id=1, pad=0)
    assert r["label"] is None and r["side"] == "unresolved" and r["n_voxels"] == 0


def test_effusion_and_ligament_have_no_host():
    assert host_seg_label(_seg(), (0, 0, 0, 5, 5, 5), tissue_id=-1)["side"] == "none"
    assert host_seg_label(_seg(), (0, 0, 0, 5, 5, 5), tissue_id=2)["side"] == "none"
