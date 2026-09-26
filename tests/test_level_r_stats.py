import pytest

from anatobind.eval.level_r_stats import (
    bootstrap_ci, cohen_kappa, confusion, field_agreement, gate_r7, gwet_ac1, pairs, positive_agreement, raw_agreement,
    set_agreement, strata_report, summarise, time_summary,
)

X = ["A", "A", "B", "B", "C"]
Y = ["A", "B", "B", "B", "C"]


def _lab(lid, host, acc=None, nal=False, t=None, lesion_type="nonspecific_wm_lesion", side="image_left"):
    return {"lesion_id": lid, "primary_host": None if nal else host, "acceptable_hosts": [] if nal else (acc or [host]),
            "not_a_lesion": nal, "time_seconds": t, "lesion_type": None if nal else lesion_type, "side": None if nal else side}


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
    s = strata_report(P, REG, n_boot=200, seed=0)
    assert sum(v["n"] for v in s["band"].values()) == 5 == sum(v["n"] for v in s["stratum_geometry"].values())
    # lesion 1: sets {cortex, wm} & {wm}; both band-0 lesions belong to p0, so every bootstrap draw is p0's mean 0.5
    assert s["band"]["0"] == {"n": 2, "raw": 0.5, "set_agreement": 1.0, "raw_ci95": [0.5, 0.5]}
    assert s["band"][">4"] == {"n": 1, "raw": 1.0, "set_agreement": 1.0} and s["slice_3mm"] == {"n": 1, "raw": 1.0, "set_agreement": 1.0}
    assert s["stratum_geometry"]["inplane_0.86_slice_5"]["raw"] == 0.0 and "raw_ci95" not in s["stratum_geometry"]["inplane_0.86_slice_5"]


def test_band0_interval_resamples_patients():
    reg = {0: _reg(0, "p0", "0", "s_slice_5"), 1: _reg(1, "p0", "0", "s_slice_5"), 2: _reg(2, "p1", "0", "s_slice_5")}
    P = pairs([_lab(0, "cortex"), _lab(1, "cortex"), _lab(2, "cortex")], [_lab(0, "cortex"), _lab(1, "thalamus"), _lab(2, "cortex")])
    lo, hi = strata_report(P, reg, n_boot=2000, seed=0)["band"]["0"]["raw_ci95"]
    assert (lo, hi) == (0.5, 1.0)          # draws {p0,p0} -> 0.5, {p0,p1} -> 2/3, {p1,p1} -> 1.0, each end with probability 1/4


def _row(lid, t, reader="r1"):
    return {**_lab(lid, "cortex", t=t), "reader_id": reader}


def test_time_summary_sums_revisits_and_restricts_to_the_requested_lesions():
    t = time_summary(A)
    assert t["n"] == 5 and t["median_s"] == 45 and t["q1_s"] == 30 and t["q3_s"] == 60 and t["hours_for_1297"] == pytest.approx(45 * 1297 / 3600)
    assert time_summary([_lab(0, "cortex")]) == {"n": 0, "median_s": None, "q1_s": None, "q3_s": None, "hours_for_1297": None}
    rows = [_row(0, 30), _row(0, 15), _row(1, 60), _row(2, None), _row(3, 100)]     # lesion 0 read twice: 30 + 15 s
    t = time_summary(rows, lesion_ids=[0, 1, 2])
    assert t == {"n": 2, "median_s": 52.5, "q1_s": 48.75, "q3_s": 56.25, "hours_for_1297": pytest.approx(52.5 * 1297 / 3600)}
    assert time_summary(rows)["n"] == 3 and time_summary(rows)["median_s"] == 60


def test_gate_reports_the_band0_interval_without_changing_the_rule():
    assert gate_r7(0.81, 0.72, 0.55) == {"all_ci_low": 0.81, "pass_all": True, "band0_raw": 0.72, "band0_ci_low": 0.55, "pass_band0": True,
                                          "single_host_endpoint_allowed": True}
    assert gate_r7(0.79, 0.72, 0.9)["pass_all"] is False and gate_r7(0.85, 0.69, 0.9)["single_host_endpoint_allowed"] is False


NWML, LAC, PVS = "nonspecific_wm_lesion", "lacunar_infarct", "perivascular_space"
TA = [_lab(0, "white_matter"), _lab(1, "white_matter", side="image_right"), _lab(2, "cortex", lesion_type=LAC),
      _lab(3, None, nal=True), _lab(4, "white_matter", lesion_type=PVS, side="midline")]
TB = [_lab(0, "white_matter"), _lab(1, "white_matter", lesion_type=LAC, side="image_right"),
      _lab(2, "cortex", lesion_type=LAC, side="image_right"), _lab(3, "white_matter"), _lab(4, None, nal=True)]


def test_lesion_type_and_side_agreement_by_hand_with_not_a_lesion_as_its_own_class():
    # lesion type  A: nwml nwml lac  NAL  pvs      side  A: L R L NAL M
    #              B: nwml lac  lac  nwml NAL            B: L R R L   NAL
    t = field_agreement(TA, TB, "lesion_type")
    assert t["n"] == 5 and t["raw"] == pytest.approx(0.4)                                     # lesions 0 and 2
    assert t["confusion"] == {"categories": [LAC, NWML, "not_a_lesion", PVS],
                              "counts": [[1, 0, 0, 0], [1, 1, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0]]}
    pa = t["positive_agreement"]
    assert pa[LAC] == {"n_x": 1, "n_y": 2, "positive_agreement": pytest.approx(2 / 3)}
    assert pa[NWML]["positive_agreement"] == 0.5 and pa["not_a_lesion"]["positive_agreement"] == 0.0 and pa[PVS]["n_y"] == 0
    s = field_agreement(TA, TB, "side")
    assert s["n"] == 5 and s["raw"] == pytest.approx(0.4)                                     # lesions 0 and 1
    assert s["confusion"]["categories"] == ["image_left", "image_right", "midline", "not_a_lesion"]
    assert s["positive_agreement"]["image_right"]["positive_agreement"] == pytest.approx(2 / 3)
    assert s["positive_agreement"]["image_left"]["positive_agreement"] == 0.5
    sub = field_agreement(TA, TB, "side", lesion_ids=[0, 2, 7])
    assert sub["n"] == 2 and sub["raw"] == 0.5


def test_summarise_reports_lesion_type_and_side_agreement_over_the_same_pairs():
    s = summarise(TA, TB, list(REG.values()), n_boot=50, seed=0)
    assert s["lesion_type_agreement"] == field_agreement(TA, TB, "lesion_type")
    assert s["side_agreement"] == field_agreement(TA, TB, "side")
    sub = summarise(TA, TB, list(REG.values()), lesion_ids=[0, 2], n_boot=50)
    assert sub["lesion_type_agreement"]["n"] == 2 and sub["lesion_type_agreement"]["raw"] == 1.0 and sub["side_agreement"]["raw"] == 0.5
    assert set(s["gate_r7"]) == {"all_ci_low", "pass_all", "band0_raw", "band0_ci_low", "pass_band0", "single_host_endpoint_allowed"}


def test_summarise_assembles_everything_and_restricts_to_a_lesion_subset():
    s = summarise(A, B, list(REG.values()), n_boot=200, seed=0)
    assert s["n_pairs"] == 5 and s["n_patients"] == 4 and s["raw"] == pytest.approx(0.4)
    assert s["raw_ci95"][0] <= 0.4 <= s["raw_ci95"][1] and s["kappa"] == pytest.approx((0.4 - 0.24) / (1 - 0.24))
    # AC1 over the fixed K = 8 classes (7 hosts + not_a_lesion): pi = .4/.2/.2/.1/.1 -> sum pi(1 - pi) = .74, p_e = .74 / 7
    assert s["ac1_categories"] == 8 and s["ac1"] == pytest.approx((0.4 - 0.74 / 7) / (1 - 0.74 / 7))
    assert s["set_agreement"] == pytest.approx(0.6) and s["gate_r7"]["band0_raw"] == 0.5 and s["gate_r7"]["pass_all"] is False
    assert s["gate_r7"]["band0_ci_low"] == 0.5 == s["strata"]["band"]["0"]["raw_ci95"][0]
    assert set(s["strata"]) == {"band", "stratum_geometry", "slice_3mm"} and s["time"]["reader_a"]["n"] == 5 and s["time"]["reader_b"]["n"] == 0
    sub = summarise(A, B, list(REG.values()), lesion_ids=[0, 3], n_boot=50)
    assert sub["n_pairs"] == 2 and sub["raw"] == 1.0
    assert sub["time"]["reader_a"]["n"] == 2 and sub["time"]["reader_a"]["median_s"] == 25            # lesions 0 and 3 only: 30, 20 s
    rows = A + [{**_lab(3, "cerebellum", t=50)}]                                                       # lesion 3 revisited: 20 + 50 s
    assert summarise(A, B, list(REG.values()), n_boot=50, a_rows=rows)["time"]["reader_a"]["median_s"] == 60   # 30 45 60 70 90
