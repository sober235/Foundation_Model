# tests/test_aur_eval.py
"""Evaluation of AnatoBind-Brain on the test patients (SSL-first plan T12; spec §7): entity / host Dice, lesion
detection at the reference false-positive budget, the controlled relation track against the B0 lookup, the gates."""
import json

import nibabel as nib
import numpy as np
import pytest
import torch

import anatobind.aur.eval as E
from anatobind.aur.labels import HOST_NAMES, HOST_SIDE, HOST_TISSUE, NO_HOST, N_HOSTS
from anatobind.aur.model import AnatoBindBrain

TINY = dict(embed=32, depths=(1, 1, 1, 1), heads=(1, 1, 1, 1), window=(2, 4, 4), patch=(2, 4, 4), d_model=16, n_events=4,
            mask_dim=4, pixel_dim=8, dec_layers=1, dec_heads=2, rel_layers=1, rel_heads=2, use_checkpoint=False)
WM_L, WM_R, VENT = 2, 41, 4


def _row(tmp_path, name="c0", with_lesion=True, source="pdgm"):
    shape, spacing = (40, 36, 12), (1.0, 1.0, 2.0)
    affine = np.diag(list(spacing) + [1.0])
    seg = np.zeros(shape, np.int16)
    seg[:20] = WM_L
    seg[20:] = WM_R
    seg[18:22, 16:20, :] = VENT
    img = np.where(seg > 0, 300.0, 0.0).astype(np.float32) + np.random.default_rng(0).normal(0, 5, shape).astype(np.float32)
    les = np.zeros(shape, np.uint8)
    if with_lesion:
        les[4:7, 4:7, 3:5] = 1                    # 36 mm3 in the left white matter
        les[30:34, 20:24, 6:9] = 1                # 96 mm3 in the right white matter
    paths = {}
    for key, arr in (("image", img), ("anatomy", seg), ("lesion", les)):
        p = tmp_path / f"{name}_{key}.nii.gz"
        nib.save(nib.Nifti1Image(arr, affine), str(p))
        paths[key] = str(p)
    return {"case": name, "source": source, "patient": name, "sequence": "FLAIR", "source_sequence": "FLAIR", "image": paths["image"],
            "anatomy": paths["anatomy"], "lesion": paths["lesion"] if with_lesion else None, "u_supervised": with_lesion,
            "u_values": [1] if with_lesion else [], "a_ignore_values": [1] if with_lesion else [], "split": "test",
            "a_supervised": True, "r_supervised": True}


def test_host_and_entity_dice_per_class_and_macro():
    truth = np.zeros((4, 6, 6), np.int16)
    truth[:, :, :3] = WM_L
    truth[:, :, 3:] = WM_R
    pred = truth.copy()
    pred[:, :, 2] = WM_R                                      # one column of the left white matter called right
    d = E.host_dice(pred, truth)
    left, right = HOST_NAMES.index("white_matter_left"), HOST_NAMES.index("white_matter_right")
    assert d["hosts"][HOST_NAMES[left]] == pytest.approx(2 * 48 / (48 + 72)) and d["hosts"][HOST_NAMES[right]] == pytest.approx(2 * 72 / (96 + 72))
    assert np.isnan(d["hosts"]["thalamus_left"]) and d["host_macro"] == pytest.approx(np.nanmean(list(d["hosts"].values())))
    assert d["entities"][WM_L] == d["hosts"][HOST_NAMES[left]] and len(d["entities"]) == 32 and len(d["hosts"]) == N_HOSTS
    perfect = E.host_dice(truth, truth)
    assert perfect["host_macro"] == pytest.approx(1.0) and perfect["n_hosts_in_truth"] == 2


def test_instance_rows_and_scan_follow_the_detection_conventions():
    inst = np.zeros((10, 20, 20), np.int32)
    inst[2:6, 2:6, 2:6] = 1
    inst[8:10, 15:18, 15:19] = 2
    rows = E.instance_rows(inst, voxel_mm3=2.0)
    assert [r["instance"] for r in rows] == [1, 2] and rows[0]["box"] == [2, 2, 2, 6, 6, 6] and rows[0]["volume_mm3"] == 128.0
    assert rows[1]["n_voxels"] == 24 and rows[1]["stratum"] == "<5" and rows[0]["stratum"] == "5-10"
    dets = [{"instance": 1, "box": [2, 2, 2, 6, 6, 6], "score": 0.9, "volume_mm3": 128.0}, {"instance": 2, "box": [0, 0, 0, 2, 2, 2], "score": 0.3, "volume_mm3": 16.0}]
    scan = E.scan("c0", rows, dets)
    assert scan["case"] == "c0" and all(r["family"] == "lesion" for r in scan["gt"] + scan["dets"]) and len(scan["dets"]) == 2
    hits, fp, kept = E.scan_matches(scan, thr=0.5)
    assert hits == {0: 0} and fp == 0 and len(kept) == 1
    hits, fp, _ = E.scan_matches(scan, thr=0.2)
    assert hits == {0: 0} and fp == 1


def test_sensitivity_at_the_reference_budget_picks_the_best_threshold_within_it():
    def s(case, boxes_gt, dets):
        return E.scan(case, [{"instance": i + 1, "box": b, "n_voxels": 8, "volume_mm3": 8.0, "stratum": "<5"} for i, b in enumerate(boxes_gt)], dets)
    scans = [s("a", [[0, 0, 0, 2, 2, 2]], [{"box": [0, 0, 0, 2, 2, 2], "score": 0.9}, {"box": [5, 5, 5, 7, 7, 7], "score": 0.4}]),
             s("b", [[0, 0, 0, 2, 2, 2]], [{"box": [0, 0, 0, 2, 2, 2], "score": 0.3}])]
    at = E.sensitivity_at_budget(scans, fp_per_scan=0.0)
    assert at["thr"] == 0.9 and at["sensitivity"] == pytest.approx(0.5) and at["fp_per_scan"] == 0.0 and at["n_gt"] == 2
    loose = E.sensitivity_at_budget(scans, fp_per_scan=0.5)
    assert loose["sensitivity"] == pytest.approx(1.0) and loose["fp_per_scan"] == pytest.approx(0.5) and loose["thr"] == 0.3
    fixed = E.sensitivity_at_threshold(scans, thr=0.3)
    assert fixed["sensitivity"] == pytest.approx(1.0) and fixed["fp_per_scan"] == pytest.approx(0.5)


def test_binding_accuracy_side_tissue_and_rescue_harm():
    left_wm, right_wm = HOST_NAMES.index("white_matter_left"), HOST_NAMES.index("white_matter_right")
    left_th = HOST_NAMES.index("thalamus_left")
    truth = np.array([left_wm, left_wm, right_wm, NO_HOST])
    r = np.array([left_wm, right_wm, right_wm, NO_HOST])              # one side error
    b0 = np.array([left_wm, left_wm, left_th, left_wm])                 # one tissue error, one host where there is none
    acc = E.binding_accuracy(r, truth)
    assert acc["aba"] == pytest.approx(0.75) and acc["side"] == pytest.approx(0.75) and acc["tissue"] == pytest.approx(1.0) and acc["n"] == 4
    assert E.binding_accuracy(b0, truth)["aba"] == pytest.approx(0.5) and E.binding_accuracy(b0, truth)["side"] == pytest.approx(0.5)
    rh = E.rescue_harm(r, b0, truth)
    assert rh == {"n": 4, "both_right": 1, "both_wrong": 0, "rescue": 2, "harm": 1, "b0_wrong_share": 0.5}
    assert E.binding_accuracy(np.zeros(0, int), np.zeros(0, int))["n"] == 0 and np.isnan(E.binding_accuracy(np.zeros(0, int), np.zeros(0, int))["aba"])
    assert HOST_SIDE[left_wm] == "left" and HOST_TISSUE[left_wm] == HOST_TISSUE[right_wm]


def test_evaluate_case_runs_the_three_tracks_on_a_tiny_model(tmp_path):
    torch.manual_seed(0)
    model = AnatoBindBrain(**TINY).eval()
    row = _row(tmp_path)
    res = E.evaluate_case(model, row, crop=(8, 16, 16), device=torch.device("cpu"), batch_size=4)
    assert res["case"] == "c0" and res["patient"] == "c0" and res["source"] == "pdgm" and res["sequence"] == "FLAIR" and res["n_windows"] > 1
    assert set(res["a"]) >= {"hosts", "host_macro", "entities", "entity_macro"} and 0.0 <= res["a"]["host_macro"] <= 1.0
    assert len(res["u"]["gt"]) == 2 and all({"box", "score", "volume_mm3"} <= set(d) for d in res["u"]["dets"])
    assert len(res["r"]["instances"]) == 2
    for inst in res["r"]["instances"]:
        assert {"instance", "truth", "r", "b0", "r_probs", "query_iou", "zero_overlap", "volume_mm3", "stratum", "box", "local_dice", "matched_score"} <= set(inst)
        assert 0 <= inst["r"] < N_HOSTS + 1 and 0.0 <= inst["local_dice"] <= 1.0
    assert res["e2e"] is not None and all({"det", "r", "b0", "score", "truth"} <= set(d) for d in res["e2e"])
    assert res["r"]["instances"][0]["truth"] == HOST_NAMES.index("white_matter_left")
    assert res["r"]["instances"][1]["truth"] == HOST_NAMES.index("white_matter_right")
    assert res["seq_pred"] in range(6) and res["seq_truth"] == 3
    bare = E.evaluate_case(model, _row(tmp_path, "c1", with_lesion=False), crop=(8, 16, 16), device=torch.device("cpu"))
    assert bare["u"] is None and bare["r"] is None and bare["e2e"] is None and bare["a"]["host_macro"] >= 0.0
    no_e2e = E.evaluate_case(model, _row(tmp_path, "c2"), crop=(8, 16, 16), device=torch.device("cpu"), bind_predicted=False)
    assert no_e2e["e2e"] is None and len(no_e2e["r"]["instances"]) == 2


def test_ignored_rows_excuse_detections_on_sub_floor_components():
    small = np.zeros((6, 6, 6), bool)
    small[1, 1, 1] = True
    rows = E.ignored_rows(small, voxel_mm3=1.0)
    assert len(rows) == 1 and rows[0]["ignore"] and rows[0]["box"] == [1, 1, 1, 2, 2, 2] and rows[0]["instance"] is None
    gt = [{"instance": 1, "box": [3, 3, 3, 5, 5, 5], "n_voxels": 8, "volume_mm3": 8.0, "stratum": "<5"}] + rows
    dets = [{"box": [3, 3, 3, 5, 5, 5], "score": 0.9}, {"box": [1, 1, 1, 2, 2, 2], "score": 0.8}]
    hits, fp, _ = E.scan_matches(E.scan("c", gt, dets), 0.5)
    assert hits == {0: 0} and fp == 0
    assert E.sensitivity_at_threshold([E.scan("c", gt, dets)], 0.5)["n_gt"] == 1


def _case(name, source, dice, inst, sequence="FLAIR", a_supervised=True, e2e=None):
    gt = [{"instance": i + 1, "box": [0, 0, 0, 2, 2, 2], "n_voxels": 8, "volume_mm3": 8.0, "stratum": "<5"} for i in range(len(inst))]
    dets = [{"instance": 1, "box": [0, 0, 0, 2, 2, 2], "score": 0.9, "volume_mm3": 8.0}] if inst else []
    return {"case": name, "source": source, "sequence": sequence, "a_supervised": a_supervised,
            "a": {"hosts": {HOST_NAMES[0]: dice, HOST_NAMES[1]: dice}, "host_macro": dice, "entities": {}, "entity_macro": dice},
            "u": {"gt": gt, "dets": dets} if inst else None,
            "r": {"instances": [{"instance": i + 1, "truth": t, "r": r, "b0": b, "stratum": "<5", "volume_mm3": 8.0, "local_dice": 0.9,
                                 "matched_score": 0.9, "zero_overlap": False, "box": [0, 0, 0, 2, 2, 2], "r_probs": [], "query_iou": 0.5}
                                for i, (t, r, b) in enumerate(inst)]} if inst else None,
            "e2e": e2e}


def test_aggregate_builds_per_source_tables_and_the_gates():
    left_wm, right_wm = HOST_NAMES.index("white_matter_left"), HOST_NAMES.index("white_matter_right")
    results = [_case("a", "pdgm", 0.9, [(left_wm, left_wm, right_wm)], e2e=[{"det": 1, "r": left_wm, "b0": right_wm, "score": 0.9, "truth": left_wm}]),
               _case("b", "pdgm", 0.7, [(right_wm, left_wm, right_wm)]),
               _case("b2", "pdgm", 0.1, [(right_wm, right_wm, right_wm)], sequence="T1c"),
               _case("c", "bmsr", 0.85, []), _case("d", "bmsr", 0.95, [(left_wm, left_wm, left_wm)], sequence="T1c"),
               _case("e", "isles", 0.2, [], a_supervised=False)]
    reference = {"pdgm": {"sensitivity": 0.8, "fp_per_scan": 0.5, "thr": 0.6, "n_scans": 2, "n_gt": 2},
                 "bmsr": {"sensitivity": 0.7, "fp_per_scan": 0.5, "thr": 0.65, "n_scans": 2, "n_gt": 1}}
    agg = E.aggregate(results, reference, a_gate=0.80, u_margin=0.05)
    assert agg["a"]["n_cases"] == 5 and agg["a"]["unsupervised"]["n_cases"] == 1 and agg["a"]["unsupervised"]["host_macro"] == pytest.approx(0.2)
    assert agg["a"]["host_macro"] == pytest.approx(np.mean([0.9, 0.7, 0.1, 0.85, 0.95])) and agg["a"]["gate"]["pass"] is False
    assert set(agg["u"]["per_source_sequence"]) == {"pdgm/FLAIR", "pdgm/T1c", "bmsr/T1c"}
    assert agg["u"]["per_source_sequence"]["pdgm/T1c"]["gated"] is False and agg["u"]["per_source_sequence"]["pdgm/FLAIR"]["gated"] is True
    assert agg["u"]["per_source"]["pdgm"]["sensitivity"] == pytest.approx(1.0) and agg["u"]["per_source"]["pdgm"]["reference"]["sensitivity"] == 0.8
    assert "test-selected" in agg["u"]["per_source"]["pdgm"]["threshold_selection"] and agg["u"]["per_source"]["pdgm"]["grid"] == "same 1 mm grid"
    assert agg["u"]["per_source"]["pdgm"]["pass"] is True and agg["u"]["per_source"]["bmsr"]["pass"] is True and agg["u"]["gate"]["pass"] is True
    fixed = E.aggregate(results, reference, fixed_thresholds={"pdgm": 0.95})
    assert fixed["u"]["per_source"]["pdgm"]["sensitivity"] == pytest.approx(0.0) and "fixed" in fixed["u"]["per_source"]["pdgm"]["threshold_selection"]
    c = agg["r"]["controlled"]
    assert c["aba_r"] == pytest.approx(3 / 4) and c["aba_b0"] == pytest.approx(3 / 4) and "b0_star" not in c
    assert c["rescue_harm"]["rescue"] == 1 and c["rescue_harm"]["harm"] == 1 and c["n_zero_overlap"] == 0
    assert agg["r"]["gate"]["pass"] is True and agg["evidence"] == "NOT_EVIDENCE"
    assert c["per_stratum"]["<5"]["n"] == 4 and c["per_source"]["bmsr"]["aba_r"] == pytest.approx(1.0)
    assert agg["r"]["gap"]["n"] == 4 and agg["r"]["end_to_end"]["n"] == 1 and agg["r"]["end_to_end"]["aba_r"] == pytest.approx(1.0)
    assert agg["not_implemented"]
    text = E.markdown(agg)
    assert "NOT_EVIDENCE" in text and "pdgm/FLAIR" in text and "G2" in text and "G3" in text and "test-selected" in text and "Not in this report" in text
    json.dumps(agg)                                                  # serialisable


def test_write_report_exports_the_level_r_sheet_and_key(tmp_path):
    left_wm = HOST_NAMES.index("white_matter_left")
    results = [_case("a", "pdgm", 0.9, [(left_wm, left_wm, left_wm)])]
    agg = E.aggregate(results, {})
    out = tmp_path / "rep"
    E.write_report(out, agg, results)
    assert (out / "REPORT.md").is_file() and (out / "aggregate.json").is_file() and (out / "cases.json").is_file()
    sheet = (out / "level_r_sheet.csv").read_text().splitlines()
    assert sheet[0].startswith("source,case,sequence,lesion_id,box_zyx,volume_mm3,stratum,reader1_host") and len(sheet) == 2 and "white_matter" not in sheet[1]
    key = json.loads((out / "level_r_key.json").read_text())
    assert key[0]["r_host"] == "white_matter_left" and key[0]["pseudo_truth_host"] == "white_matter_left"
