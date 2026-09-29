# tests/test_brain_disease_components.py
import numpy as np
import pytest

from anatobind.eval.lesion_components import (
    MIN_MM3, component_mask, component_rows, components, equivalent_diameter_mm, min_voxels_for, size_stratum,
)


def test_volume_floor_in_voxels_for_the_three_voxel_sizes():
    assert MIN_MM3 == 10.0
    assert min_voxels_for(1.0) == 10            # PDGM, 1 mm isotropic: 10 voxels = 10 mm3
    assert min_voxels_for(0.859375 * 0.859375 * 1.5) == 10   # BMSR, 1.108 mm3: 9 voxels = 9.97 mm3 is under the floor
    assert min_voxels_for(8.0) == 2             # ISLES, 2 mm isotropic: one voxel = 8 mm3 is under the floor
    assert min_voxels_for(20.0) == 1            # a voxel larger than the floor always counts
    assert min_voxels_for(2.5) == 4             # exactly 10 mm3


def test_size_strata_use_the_equivalent_diameter():
    assert equivalent_diameter_mm(523.5988) == pytest.approx(10.0, abs=1e-3)
    assert size_stratum(65.0) == "<5" and size_stratum(66.0) == "5-10"      # 5 mm sphere = 65.45 mm3
    assert size_stratum(523.0) == "5-10" and size_stratum(524.0) == ">=10"  # 10 mm sphere = 523.6 mm3


def _mask():
    m = np.zeros((12, 12, 6), np.uint8)
    m[1:3, 1:3, 1:3] = 1          # 8 voxels
    m[3, 3, 3] = 1                # touches the block only by a corner: same component with 26-connectivity
    m[8:11, 8:11, 2:5] = 1        # 27 voxels
    m[0, 11, 5] = 1               # a single voxel
    return m


def test_rows_use_26_connectivity_and_flag_small_components():
    comp, n = components(_mask())
    rows = sorted(component_rows(comp, n, 1.0, "tumor"), key=lambda r: r["n_voxels"])
    assert n == 3 and [r["n_voxels"] for r in rows] == [1, 9, 27]
    assert [r["ignore"] for r in rows] == [True, True, False]       # 1 mm3 and 9 mm3 are under 10 mm3
    assert rows[1]["box"] == (1, 1, 1, 4, 4, 4) and rows[2]["box"] == (8, 8, 2, 11, 11, 5)
    assert rows[2]["mm3"] == 27.0 and rows[2]["family"] == "tumor"
    assert sorted(r["component"] for r in rows) == [1, 2, 3]
    big = sorted(component_rows(comp, n, 8.0, "infarct"), key=lambda r: r["n_voxels"])   # 2 mm isotropic voxels
    assert [r["ignore"] for r in big] == [True, False, False] and big[1]["mm3"] == 72.0


def test_component_mask_selects_only_its_own_component():
    m = np.zeros((6, 6, 1), np.uint8)
    m[0, 0:5, 0] = 1
    m[0:5, 0, 0] = 1              # an L
    m[3, 3, 0] = 1                # a separate voxel inside the L's box
    comp, n = components(m)
    rows = sorted(component_rows(comp, n, 20.0, "x"), key=lambda r: -r["n_voxels"])
    sl, mask = component_mask(comp, rows[0])
    assert n == 2 and mask.shape == (5, 5, 1) and int(mask.sum()) == 9 and not mask[3, 3, 0]
    assert sl == (slice(0, 5), slice(0, 5), slice(0, 1))


def test_an_empty_mask_has_no_rows():
    comp, n = components(np.zeros((4, 4, 4), np.uint8))
    assert n == 0 and component_rows(comp, n, 1.0, "tumor") == []
