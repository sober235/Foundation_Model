import math

import pytest

from anatobind.eval.level_r_stats import (
    bootstrap_ci, cohen_kappa, confusion, gate_r7, gwet_ac1, pairs, positive_agreement, raw_agreement, set_agreement,
    strata_report, summarise, time_summary,
)

X = ["A", "A", "B", "B", "C"]
Y = ["A", "B", "B", "B", "C"]


def _lab(lid, host, acc=None, nal=False, t=None):
    return {"lesion_id": lid, "primary_host": None if nal else host, "acceptable_hosts": [] if nal else (acc or [host]),
            "not_a_lesion": nal, "time_seconds": t}


def test_hand_computed_three_class_table():
    assert raw_agreement(X, Y) == pytest.approx(0.8)
    assert cohen_kappa(X, Y) == pytest.approx((0.8 - 0.36) / 0.64)                      # p_e = (2*1 + 2*3 + 1*1)/25
    assert gwet_ac1(X, Y) == pytest.approx((0.8 - 0.31) / 0.69)                         # pi = .3/.5/.2 -> p_e = .62/2
    pa = positive_agreement(X, Y)
    assert pa["A"]["positive_agreement"] == pytest.approx(2 / 3) and pa["B"]["positive_agreement"] == pytest.approx(0.8)
    assert pa["C"] == {"n_x": 1, "n_y": 1, "positive_agreement": 1.0}
    assert confusion(X, Y) == {"categories": ["A", "B", "C"], "counts": [[1, 1, 0], [0, 2, 0], [0, 0, 1]]}
    assert gwet_ac1(X, Y, categories=["A", "B", "C", "D"]) == pytest.approx((0.8 - 0.62 / 3) / (1 - 0.62 / 3))


def test_pairs_use_latest_answers_of_both_readers_and_map_not_a_lesion_to_its_own_class():
    a = [_lab(0, "white_matter", ["white_matter", "cortex"]), _lab(1, None, nal=True), _lab(2, "cortex")]
    b = [_lab(0, "cortex", ["cortex"]), _lab(1, "white_matter"), _lab(3, "cortex")]
    P = pairs(a, b)
    assert [p["lesion_id"] for p in P] == [0, 1]
    assert (P[0]["x"], P[0]["y"], P[0]["sx"], P[0]["sy"]) == ("white_matter", "cortex", {"white_matter", "cortex"}, {"cortex"})
    assert (P[1]["x"], P[1]["sx"]) == ("not_a_lesion", {"not_a_lesion"})
    assert set_agreement(P) == pytest.approx(0.5)                                        # lesion 0 sets intersect, lesion 1 do not
    assert [p["lesion_id"] for p in pairs(a, b, lesion_ids=[1, 3])] == [1]


def test_bootstrap_ci_contains_the_point_estimate_and_narrows_with_more_patients():
    small = {f"p{i}": ([1.0, 1.0, 0.0] if i % 2 else [1.0, 0.0, 0.0]) for i in range(10)}      # pooled mean 0.5
    big = {f"p{i}": ([1.0, 1.0, 0.0] if i % 2 else [1.0, 0.0, 0.0]) for i in range(100)}
    lo, hi = bootstrap_ci(small, n_boot=500, seed=0)
    lo2, hi2 = bootstrap_ci(big, n_boot=500, seed=0)
    assert lo <= 0.5 <= hi and lo2 <= 0.5 <= hi2 and (hi2 - lo2) < (hi - lo)
    assert bootstrap_ci(small, n_boot=500, seed=0) == (lo, hi)
    assert bootstrap_ci({"p0": [1.0, 1.0]}, n_boot=50) == (1.0, 1.0)


def _reg(lid, patient, band, stratum):
    return {"lesion_id": lid, "patient_id": patient, "band": band, "stratum_geometry": stratum}


REG = {0: _reg(0, "p0", "0", "inplane_0.69_slice_5"), 1: _reg(1, "p0", "0", "inplane_0.69_slice_5"), 2: _reg(2, "p1", "0-2", "inplane_0.69_slice_5"),
       3: _reg(3, "p2", ">4", "inplane_0.62_slice_3"), 4: _reg(4, "p3", "2-4", "inplane_0.86_slice_5")}
A = [_lab(0, "white_matter", t=30), _lab(1, "cortex", ["cortex", "white_matter"], t=60), _lab(2, "white_matter", t=45), _lab(3, "cerebellum", t=20), _lab(4, "thalamus", t=90)]
B = [_lab(0, "white_matter"), _lab(1, "white_matter"), _lab(2, "cortex"), _lab(3, "cerebellum"), _lab(4, "basal_ganglia")]


def test_strata_report_sums_to_the_total_and_separates_3mm_volumes():
    P = pairs(A, B)
    s = strata_report(P, REG)
    assert sum(v["n"] for v in s["band"].values()) == 5 == sum(v["n"] for v in s["stratum_geometry"].values())
    assert s["band"]["0"] == {"n": 2, "raw": 0.5, "set_agreement": 1.0}                 # lesion 1: sets {cortex, wm} & {wm}
    assert s["band"][">4"]["raw"] == 1.0 and s["slice_3mm"] == {"n": 1, "raw": 1.0, "set_agreement": 1.0}
    assert s["stratum_geometry"]["inplane_0.86_slice_5"]["raw"] == 0.0


def test_time_summary_and_gate():
    t = time_summary(A)
    assert t["n"] == 5 and t["median_s"] == 45 and t["q1_s"] == 30 and t["q3_s"] == 60 and t["hours_for_1297"] == pytest.approx(45 * 1297 / 3600)
    assert time_summary([_lab(0, "cortex")]) == {"n": 0, "median_s": None, "q1_s": None, "q3_s": None, "hours_for_1297": None}
    assert gate_r7(0.81, 0.72) == {"all_ci_low": 0.81, "pass_all": True, "band0_raw": 0.72, "pass_band0": True, "single_host_endpoint_allowed": True}
    assert gate_r7(0.79, 0.72)["pass_all"] is False and gate_r7(0.85, 0.69)["single_host_endpoint_allowed"] is False


def test_summarise_assembles_everything_and_restricts_to_a_lesion_subset():
    s = summarise(A, B, list(REG.values()), n_boot=200, seed=0)
    assert s["n_pairs"] == 5 and s["n_patients"] == 4 and s["raw"] == pytest.approx(0.4)
    assert s["raw_ci95"][0] <= 0.4 <= s["raw_ci95"][1] and not math.isnan(s["kappa"]) and not math.isnan(s["ac1"])
    assert s["set_agreement"] == pytest.approx(0.6) and s["gate_r7"]["band0_raw"] == 0.5 and s["gate_r7"]["pass_all"] is False
    assert set(s["strata"]) == {"band", "stratum_geometry", "slice_3mm"} and s["time"]["reader_a"]["n"] == 5 and s["time"]["reader_b"]["n"] == 0
    sub = summarise(A, B, list(REG.values()), lesion_ids=[0, 3], n_boot=50)
    assert sub["n_pairs"] == 2 and sub["raw"] == 1.0
