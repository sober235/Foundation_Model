import numpy as np

from anatobind.eval.lookup import BRAIN_ALL, BRAIN_PARENCHYMA, BrainLookup


def _seg():
    seg = np.zeros((40, 40, 2), np.int16)
    seg[:20, :, :] = 2            # WM
    seg[20:30, :, :] = 3          # cortex
    seg[30:, :, :] = 24           # CSF; cols 30..40 (the outermost)
    return seg


def test_argmax_overlap_wins_and_reports_the_fraction():
    lk = BrainLookup(_seg(), (0.5, 0.5, 5.0), BRAIN_ALL)
    label, frac = lk.host([(16, 24, 10, 14, 0)])         # cols 16..23: 4 WM columns + 4 cortex columns
    assert label in (2, 3) and frac == 0.5
    label, frac = lk.host([(16, 22, 10, 14, 0)])         # 4 WM + 2 cortex
    assert label == 2 and frac == 4 / 6


def test_zero_overlap_falls_back_to_the_nearest_candidate_in_mm():
    seg = _seg()
    seg[30:, :, :] = 0                                    # background beyond the cortex
    lk = BrainLookup(seg, (0.5, 0.5, 5.0), BRAIN_PARENCHYMA)
    label, frac = lk.host([(35, 38, 10, 14, 0)])
    assert label == 3 and frac == 0.0


def test_parenchyma_candidates_exclude_ventricles_and_csf():
    assert 24 not in BRAIN_PARENCHYMA and 4 not in BRAIN_PARENCHYMA and 24 in BRAIN_ALL
    lk = BrainLookup(_seg(), (0.5, 0.5, 5.0), BRAIN_PARENCHYMA)
    assert lk.host([(32, 36, 10, 14, 0)]) == (3, 0.0)     # the CSF voxels count as no overlap; nearest parenchyma is cortex


def test_no_voxels_on_the_grid_gives_none():
    assert BrainLookup(_seg(), (0.5, 0.5, 5.0), BRAIN_ALL).host([]) == (None, 0.0)
