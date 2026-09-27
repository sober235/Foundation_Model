import json

import numpy as np
import pytest

from anatobind.eval.geometry import SLOT_FIELDS
from anatobind.relation.table import (
    BRAIN_EXTRA, COLUMN_TYPES, LESION_COLUMNS, LESION_GEOMETRY, N_OUT, PAIR_FIELDS, SLOTS, SLOT_COLUMNS, Table,
    features_flat, features_per_slot, load_table, mask_to_candidates, slot_col, write_table,
)


def _row(lid, patient, fold, c1="white_matter", side="image_left", ltype="nonspecific_wm_lesion", cand=("white_matter", "cortex")):
    r = {"lesion_id": lid, "file": f"file_{patient}", "patient_id": patient, "fold": fold, "stratum_geometry": "inplane_0.69_slice_5",
         "stratum_series": "200_201", "band": "0-2", "is_3mm": False, "lesion_type": ltype, "n_slices": 1, "inplane_mm": 4.8,
         "x0": 10, "y0": 10, "x1": 14, "y1": 14, "z0": 1, "z1": 1, "spacing_col_mm": 0.6875, "spacing_row_mm": 0.6875,
         "spacing_slice_mm": 5.0, "slice_thickness_mm": 5.0, "extent_x_mm": 2.75, "extent_y_mm": 2.75, "extent_z_mm": 5.0,
         "volume_mm3": 37.8, "centroid_col": 11.5, "centroid_row": 11.5, "centroid_slice": 1.0, "side": side,
         "row_third": "middle", "slice_third": "middle", "coarse_location": f"{side}|middle|middle", "d1_mm": 0.0,
         "d_interface_mm": 1.5, "delta_d_mm": 1.5, "dist_cortex_mm": 1.5, "dist_ventricle_mm": 12.0, "c1_class": c1,
         "c1_slot": SLOTS.index(c1), "c1_source": "overlap", "c1_overlap": 1.0, "registry_lookup_class": c1}
    for s in SLOTS:
        for f in SLOT_FIELDS:
            r[slot_col(s, f)] = {"in_volume": s in cand, "candidate": s in cand, "ioa": 1.0 if s == c1 else 0.0,
                                 "soft_overlap": 1.0 if s == c1 else 0.2}.get(f, 3.0 if s in cand else 30.0)
    return r


def test_column_lists_cover_the_spec_and_are_typed():
    assert len(SLOT_COLUMNS) == 70 and slot_col("basal_ganglia", "ioa") == "bg_ioa"
    assert set(LESION_COLUMNS) | set(SLOT_COLUMNS) == set(COLUMN_TYPES)
    assert "c1_slot" in LESION_COLUMNS and COLUMN_TYPES["is_3mm"] is bool and COLUMN_TYPES["d_interface_mm"] is float


def test_blocks_and_parity_between_flat_and_per_slot():
    t = Table([_row(0, "p0", 0), _row(1, "p1", 1, c1="cortex", side="midline", ltype="lacunar_infarct")])
    sb, lb = t.slot_block(), t.lesion_block()
    assert sb.shape == (2, 7, 9) and lb.shape == (2, 17) and sb.dtype == np.float32
    assert np.allclose(sb[0, 0], [1.0] + [t.rows[0][slot_col("white_matter", f)] for f in PAIR_FIELDS])
    assert np.allclose(lb[1, :8], [t.rows[1][c] for c in LESION_GEOMETRY])          # float32 block: 37.8 is not exact
    assert np.allclose(lb[1, 8:12], [t.rows[1][c] for c in BRAIN_EXTRA])
    assert list(lb[1, 12:15]) == [0.0, 0.0, 1.0] and list(lb[1, 15:]) == [0.0, 1.0]      # midline, lacunar
    flat, per = features_flat(t), features_per_slot(t)
    assert flat.shape == (2, 80) and per.shape == (2, 7, 26)
    assert np.array_equal(per[:, :, :9].reshape(2, 63), flat[:, :63]) and np.array_equal(per[:, 3, 9:], flat[:, 63:])
    assert t.c1_slot().tolist() == [0, 1] and t.candidates().sum(1).tolist() == [2, 2] and t.folds().tolist() == [0, 1]


def test_mask_to_candidates_zeroes_non_candidates_and_renormalises():
    probs = np.full((1, N_OUT), 1 / N_OUT)
    cand = np.zeros((1, 7), bool)
    cand[0, [0, 1]] = True
    out = mask_to_candidates(probs, cand)
    assert np.allclose(out[0, [0, 1, 7]], 1 / 3) and out[0, 2:7].sum() == 0 and np.isclose(out.sum(), 1)


def test_mask_to_candidates_turns_a_row_with_no_candidate_mass_into_uniform_over_its_candidates():
    probs = np.zeros((2, N_OUT))
    probs[0, 4] = 1.0                                                  # all mass on brainstem, which is not a candidate
    probs[1, [0, 7]] = 0.5
    cand = np.zeros((2, 7), bool)
    cand[:, [0, 2, 3]] = True
    out = mask_to_candidates(probs, cand)
    assert np.allclose(out[0, [0, 2, 3]], 1 / 3) and out[0, [1, 4, 5, 6, 7]].sum() == 0      # P7: never an all-zero row
    assert np.allclose(out[1, [0, 7]], 0.5) and np.allclose(out.sum(1), 1)


def test_write_refuses_an_existing_directory_and_load_round_trips(tmp_path):
    rows = [_row(0, "p0", 0), _row(1, "p1", 1)]
    patches = {"lesion_id": np.array([0, 1]), "image": np.zeros((2, 3, 48, 48), np.float16),
               "mask": np.zeros((2, 3, 48, 48), bool), "classmap": np.zeros((2, 3, 48, 48), np.int8)}
    out = write_table(tmp_path / "v1", rows, patches, {"version": "v1"})
    assert (out / "table.csv").exists() and (out / "patches.npz").exists()
    assert json.loads((out / "manifest.json").read_text())["version"] == "v1"
    with pytest.raises(FileExistsError):
        write_table(tmp_path / "v1", rows, patches, {"version": "v1"})
    t, p = load_table(out)
    assert t.rows[1]["lesion_type"] == "nonspecific_wm_lesion" and t.rows[1]["is_3mm"] is False and t.rows[0]["fold"] == 0
    assert isinstance(t.rows[0]["d_interface_mm"], float) and p["image"].shape == (2, 3, 48, 48)
    assert t.subset(np.array([1])).lesion_id.tolist() == [1]
