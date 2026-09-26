from collections import Counter

import pytest

from anatobind.level_r.pilot import allocate, cell_of, sample_pilot

SIZES = {"0|A": 500, "0-2|A": 200, "2-4|A": 200, ">4|A": 300, "0|B": 20, "0-2|B": 3, "2-4|B": 6, ">4|B": 3}


def _registry(sizes, patients_per_cell=None):
    rows, lid = [], 0
    for cell, n in sizes.items():
        band, stratum = cell.split("|")
        k = (patients_per_cell or {}).get(cell, n)                 # default: every lesion its own patient
        for i in range(n):
            rows.append({"lesion_id": lid, "band": band, "stratum_geometry": stratum, "patient_id": f"{cell}-p{i % k}"})
            lid += 1
    return rows


def test_allocate_takes_small_cells_whole_floors_at_min_and_splits_the_rest_by_largest_remainder():
    a = allocate(SIZES, n=150, min_per_cell=8)
    assert a == {"0|A": 54, "0-2|A": 22, "2-4|A": 22, ">4|A": 32, "0|B": 8, "0-2|B": 3, "2-4|B": 6, ">4|B": 3}
    assert sum(a.values()) == 150


def test_allocate_refuses_a_budget_that_cannot_honour_the_floor():
    with pytest.raises(ValueError):
        allocate(SIZES, n=20, min_per_cell=8)


def test_sample_pilot_hits_150_honours_cells_and_is_deterministic():
    reg = _registry(SIZES)
    p = sample_pilot(reg, n=150, min_per_cell=8, max_per_patient=3, seed=0)
    ids = p["lesion_ids"]
    assert len(ids) == 150 and ids == sorted(set(ids))
    by_id = {r["lesion_id"]: r for r in reg}
    counts = Counter(cell_of(by_id[i]) for i in ids)
    assert counts == {"0|A": 54, "0-2|A": 22, "2-4|A": 22, ">4|A": 32, "0|B": 8, "0-2|B": 3, "2-4|B": 6, ">4|B": 3}
    assert p["cells"]["0|B"] == {"n": 20, "alloc": 8, "picked": 8} and p["cells"]["0-2|B"] == {"n": 3, "alloc": 3, "picked": 3}
    assert p == sample_pilot(reg, 150, 8, 3, 0) and p["lesion_ids"] != sample_pilot(reg, 150, 8, 3, 1)["lesion_ids"]


def test_patient_cap_moves_the_shortfall_to_the_largest_cells():
    reg = _registry(SIZES, patients_per_cell={"0|A": 10})          # 500 lesions from 10 patients: at most 30 pickable
    p = sample_pilot(reg, n=150, min_per_cell=8, max_per_patient=3, seed=0)
    by_id = {r["lesion_id"]: r for r in reg}
    per_patient = Counter(by_id[i]["patient_id"] for i in p["lesion_ids"])
    assert len(p["lesion_ids"]) == 150 and max(per_patient.values()) <= 3
    assert p["cells"]["0|A"]["picked"] == 30 and p["cells"]["0|A"]["alloc"] == 54
    assert sum(c["picked"] for c in p["cells"].values()) == 150


def test_sample_pilot_fails_loudly_when_the_cap_makes_150_impossible():
    reg = _registry({"0|A": 300}, patients_per_cell={"0|A": 20})   # 20 patients x 3 = 60 < 150
    with pytest.raises(ValueError):
        sample_pilot(reg, n=150, min_per_cell=8, max_per_patient=3, seed=0)
