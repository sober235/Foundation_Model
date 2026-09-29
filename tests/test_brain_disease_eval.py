# tests/test_brain_disease_eval.py
import json

import nibabel as nib
import numpy as np
import pytest

from anatobind.eval.brain_disease import (
    beyond_budget, binding_agreement, case_scan, code_version, dice_summary, evaluate, false_positive_spread, jobs, strata,
    verdict,
)
from anatobind.eval.lesion_components import size_stratum
from anatobind.nnunet.brain_disease import fold_dir

AFF = np.diag([1.0, 1.0, 1.0, 1.0])
SHAPE = (24, 24, 8)


def _save(path, data, affine=AFF):
    path.parent.mkdir(parents=True, exist_ok=True)
    nib.save(nib.Nifti1Image(data, affine), str(path))
    return path


def _case(tmp_path, name, gt_blocks, pred_blocks):
    """blocks: [(slices, probability)]; the anatomy is left white matter for x < 12 and right for x >= 12.
    Probabilities are binary fractions (0.875, 0.75, 0.625) so that float32 means are exact."""
    lab, pred = np.zeros(SHAPE, np.uint8), np.zeros(SHAPE, np.uint8)
    probs = np.zeros((2,) + SHAPE, np.float32)
    probs[0] = 1.0
    for sl, _ in gt_blocks:
        lab[sl] = 1
    for sl, p in pred_blocks:
        pred[sl] = 1
        probs[1][sl], probs[0][sl] = p, 1.0 - p
    seg = np.zeros(SHAPE, np.int16)
    seg[:12], seg[12:] = 2, 41
    d = tmp_path / name
    d.mkdir(parents=True)
    np.savez(d / "p.npz", probabilities=np.ascontiguousarray(probs.transpose(0, 3, 2, 1)))   # nnU-Net order (C, Z, Y, X)
    return (name, "glioma", _save(d / "lab.nii.gz", lab), _save(d / "pred.nii.gz", pred), d / "p.npz", _save(d / "seg.nii.gz", seg), 1.0)


BIG = (slice(2, 6), slice(2, 6), slice(2, 5))          # 48 voxels, left
SHIFTED = (slice(3, 7), slice(2, 6), slice(2, 5))      # box IoU with BIG: 36 / 60 = 0.6
SMALL = (slice(16, 18), slice(16, 18), slice(2, 4))    # 8 voxels = 8 mm3, right: under the floor
OTHER = (slice(16, 20), slice(4, 8), slice(2, 5))      # 48 voxels, right
STRAY = (slice(8, 11), slice(16, 19), slice(5, 8))     # 27 voxels, left


def test_case_scan_gives_bound_ground_truth_and_detections(tmp_path):
    s = case_scan(_case(tmp_path, "c", [(BIG, 1), (SMALL, 1)], [(SHIFTED, 0.875), (STRAY, 0.625)]))
    assert [(r["n_voxels"], r["ignore"], r["side"], r["host"]) for r in s["gt"]] == [
        (48, False, "left", "white_matter"), (8, True, "right", "white_matter")]
    assert [(r["n_voxels"], r["score"], r["side"]) for r in s["dets"]] == [(48, 0.875, "left"), (27, 0.625, "left")]


def test_case_scan_refuses_grids_that_differ(tmp_path):
    job = list(_case(tmp_path, "c", [(BIG, 1)], [(BIG, 0.875)]))
    job[5] = _save(tmp_path / "other_seg.nii.gz", np.zeros((24, 24, 9), np.int16))
    with pytest.raises(ValueError, match="c: label"):
        case_scan(tuple(job))


def _scans(tmp_path):
    return [case_scan(_case(tmp_path, "a", [(BIG, 1), (SMALL, 1)], [(SHIFTED, 0.875), (SMALL, 0.875), (STRAY, 0.625)])),
            case_scan(_case(tmp_path, "b", [(OTHER, 1)], [])),
            case_scan(_case(tmp_path, "n", [], [(STRAY, 0.75)]))]


def test_evaluate_counts_by_hand(tmp_path):
    r = evaluate(_scans(tmp_path))
    # Ground truth that counts: BIG (a) and OTHER (b) = 2; SMALL (8 mm3) is ignored. The prediction on SMALL is itself
    # under the volume floor and is dropped before matching. BIG is found by SHIFTED (score 0.875, box IoU 0.6) up to
    # threshold 0.85; OTHER is never found. False positives: the strays at 0.625 (a) and 0.75 (n).
    assert (r["n_scans"], r["n_gt"], r["n_ignored"]) == (3, 2, 1)
    by = {round(x["thr"], 2): x for x in r["rows"]}
    assert (by[0.05]["n_hit"], by[0.05]["n_fp"]) == (1, 2) and by[0.6]["n_fp"] == 2
    assert (by[0.65]["n_fp"], by[0.75]["n_fp"], by[0.8]["n_fp"]) == (1, 1, 0)
    assert (by[0.85]["n_hit"], by[0.9]["n_hit"]) == (1, 0)
    assert r["gate"] == {"pass": True, "thr": 0.85, "sensitivity_family": 0.5, "fp_per_scan": 0.0}


def test_verdict_is_a_gate_only_with_five_folds(tmp_path):
    r = evaluate(_scans(tmp_path))
    # the row just beyond the budget does not exist here: the strays never exceed 2 per scan
    assert verdict(r, [0]) == {"kind": "early_reading", "folds": [0], "pass": None, "operating_point": True,
                               "sensitivity": 0.5, "thr": 0.85, "fp_per_scan": 0.0, "beyond_budget": None,
                               "stop_remaining_folds": False, "early_stop_undecided": False}
    assert verdict(r, [4, 3, 2, 1, 0])["kind"] == "gate" and verdict(r, [0, 1, 2, 3, 4])["pass"] is True
    low = {"gate": {"pass": False, "thr": 0.55, "sensitivity_family": 0.29, "fp_per_scan": 1.0}}
    assert verdict(low, [0])["stop_remaining_folds"] is True and verdict(low, [0, 1, 2, 3, 4])["stop_remaining_folds"] is False


def test_strata_and_binding_agreement(tmp_path):
    scans = _scans(tmp_path)
    st = strata(scans, 0.5, lambda s, r: size_stratum(r["mm3"]))
    assert st == {"<5": {"n_gt": 2, "n_hit": 1, "sensitivity": 0.5}}            # 48 mm3 is a 4.5 mm sphere
    by_case = strata(scans, 0.5, lambda s, r: s["case"])
    assert by_case == {"a": {"n_gt": 1, "n_hit": 1, "sensitivity": 1.0}, "b": {"n_gt": 1, "n_hit": 0, "sensitivity": 0.0}}
    assert binding_agreement(scans, 0.5) == {"n_pairs": 1, "host_agreement": 1.0, "side_agreement": 1.0,
                                             "host_side_agreement": 1.0, "n_detections": 3, "no_host_rate": 0.0,
                                             "nearest_rate": 0.0, "unlocated_rate": 0.0}
    assert false_positive_spread(scans, 0.5) == {"n_scans": 3, "median": 1.0, "max": 1, "n_scans_over_budget": 0}
    assert false_positive_spread(scans, 0.7) == {"n_scans": 3, "median": 0.0, "max": 1, "n_scans_over_budget": 0}
    assert binding_agreement(scans, 0.95)["host_agreement"] is None


def test_jobs_name_a_missing_validation_file(tmp_path):
    splits = [{"train": ["b"], "val": ["a"]}, {"train": ["a"], "val": ["b"]}]
    info = {"a": {"voxel_mm3": 1.0}, "b": {"voxel_mm3": 2.0}}
    v = fold_dir(tmp_path / "res", "glioma", 1) / "validation"
    v.mkdir(parents=True)
    (v / "b.nii.gz").write_text("")
    with pytest.raises(FileNotFoundError, match="fold 1 case b: missing .*b.npz"):
        jobs(tmp_path / "res", tmp_path / "raw", "glioma", splits, [1], info, lambda c: tmp_path / f"{c}_seg.nii.gz")
    (v / "b.npz").write_text("")
    out = jobs(tmp_path / "res", tmp_path / "raw", "glioma", splits, [1], info, lambda c: tmp_path / f"{c}_seg.nii.gz")
    assert out == [("b", "glioma", tmp_path / "raw" / "Dataset904_PDGMGlioma" / "labelsTr" / "b.nii.gz", v / "b.nii.gz",
                    v / "b.npz", tmp_path / "b_seg.nii.gz", 2.0)]


def test_dice_summary_skips_cases_without_ground_truth(tmp_path):
    for f, cases in ((0, [(0.8, 10), (float("nan"), 0)]), (1, [(0.6, 5)])):
        v = fold_dir(tmp_path, "infarct", f) / "validation"
        v.mkdir(parents=True)
        (v / "summary.json").write_text(json.dumps({"metric_per_case": [{"metrics": {"1": {"Dice": d, "n_ref": n}}} for d, n in cases]}))
    assert dice_summary(tmp_path, "infarct", [0, 1]) == {"n_cases": 2, "mean": pytest.approx(0.7), "median": pytest.approx(0.7)}
    assert dice_summary(tmp_path, "infarct", [1])["n_cases"] == 1


def test_verdict_without_an_operating_point_measures_nothing_and_stops_nothing():
    none = {"gate": {"pass": False, "thr": None, "sensitivity_family": 0.0, "fp_per_scan": None}}   # what gate() returns
    assert verdict(none, [0]) == {"kind": "early_reading", "folds": [0], "pass": None, "operating_point": False,
                                  "sensitivity": None, "thr": None, "fp_per_scan": None, "beyond_budget": None,
                                  "stop_remaining_folds": False, "early_stop_undecided": True}
    full = verdict(none, [0, 1, 2, 3, 4])
    assert full["kind"] == "gate" and full["pass"] is False and full["sensitivity"] is None
    edge = {"gate": {"pass": False, "thr": 0.9, "sensitivity_family": 0.3, "fp_per_scan": 2.0}}
    assert verdict(edge, [0])["stop_remaining_folds"] is False      # the rule is "below 0.3"


def test_jobs_refuse_folds_given_twice_or_out_of_range(tmp_path):
    splits = [{"train": [], "val": ["a"]}] * 5
    for folds in ([0, 0], [5], [-1, 0]):
        with pytest.raises(ValueError, match="folds must be distinct and within 0..4"):
            jobs(tmp_path / "res", tmp_path / "raw", "glioma", splits, folds, {}, lambda c: tmp_path / c)


def _rows(*triples, n_gt=100):
    """(threshold, lesions found, false positives per scan) of a fold with n_gt counted lesions."""
    return [{"thr": t, "n_gt": n_gt, "n_hit_family": h, "sensitivity_family": h / n_gt, "fp_per_scan": f} for t, h, f in triples]


def test_an_early_stop_needs_the_row_beyond_the_budget_to_be_out_of_reach_too():
    low = {"pass": False, "thr": 0.95, "sensitivity_family": 0.27, "fp_per_scan": 1.6}
    # 0.90 exceeds the budget and finds 34 of 100: a threshold between the two rows may reach 0.3
    near = verdict({"gate": low, "rows": _rows((0.85, 40, 3.1), (0.90, 34, 2.4), (0.95, 27, 1.6))}, [0])
    assert near["beyond_budget"] == {"thr": 0.90, "sensitivity": 0.34, "fp_per_scan": 2.4, "n_hit": 34, "n_gt": 100,
                                     "out_of_reach": False}
    assert near["stop_remaining_folds"] is False and near["early_stop_undecided"] is True
    # 28 of 100 is two lesions short of 0.3: the matching is redone per threshold, so this is still within reach
    close = verdict({"gate": low, "rows": _rows((0.90, 28, 2.4), (0.95, 27, 1.6))}, [0])
    assert close["stop_remaining_folds"] is False and close["early_stop_undecided"] is True
    # 27 of 100 is three lesions short: out of reach, the reading is clear
    far = verdict({"gate": low, "rows": _rows((0.90, 27, 2.4), (0.95, 27, 1.6))}, [0])
    assert far["beyond_budget"]["out_of_reach"] is True
    assert far["stop_remaining_folds"] is True and far["early_stop_undecided"] is False
    # the budget is never exceeded: the operating point is the lowest threshold, nothing lies beyond it
    alone = verdict({"gate": dict(low, thr=0.05), "rows": _rows((0.05, 27, 1.6), (0.95, 20, 0.4))}, [0])
    assert alone["beyond_budget"] is None and alone["stop_remaining_folds"] is True
    full = verdict({"gate": low, "rows": _rows((0.90, 34, 2.4), (0.95, 27, 1.6))}, [0, 1, 2, 3, 4])
    assert full["stop_remaining_folds"] is False and full["early_stop_undecided"] is False and full["pass"] is False
    assert beyond_budget(_rows((0.5, 90, 2.0))) is None         # exactly the budget is within the budget
    # no operating point: always undecided; out_of_reach tells whether the reading is decisive in substance
    none = {"pass": False, "thr": None, "sensitivity_family": 0.0, "fp_per_scan": None}
    flood = verdict({"gate": none, "rows": _rows((0.90, 30, 4.0), (0.95, 20, 2.5))}, [0])
    assert flood["early_stop_undecided"] is True and flood["stop_remaining_folds"] is False
    assert flood["beyond_budget"]["thr"] == 0.95 and flood["beyond_budget"]["out_of_reach"] is True


def test_case_scan_counts_in_millimetres_on_a_grid_that_is_not_isotropic(tmp_path):
    aff = np.diag([1.0, 1.0, 2.5, 1.0])                          # 2.5 mm3 per voxel: the floor is 4 voxels
    lab, pred = np.zeros(SHAPE, np.uint8), np.zeros(SHAPE, np.uint8)
    lab[2:7, 2, 2] = 1                                           # 5 voxels = 12.5 mm3: counted
    lab[2:5, 10, 2] = 1                                          # 3 voxels = 7.5 mm3: ignored
    pred[2:7, 2, 2] = 1                                          # found
    pred[2:5, 20, 6] = 1                                         # 3 voxels: dropped
    pred[10:14, 12, 4] = 1                                       # 4 voxels = 10 mm3, on no structure
    probs = np.zeros((2,) + SHAPE, np.float32)
    probs[0] = 1.0
    probs[1][pred == 1], probs[0][pred == 1] = 0.875, 0.125
    seg = np.zeros(SHAPE, np.int16)
    seg[:8, :8] = 2                                              # left white matter around the lesion
    seg[10:14, 12, 1] = 3                                        # left cortex 3 voxels below the last detection: 7.5 mm
    seg[10:14, 17, 4] = 41                                       # right white matter 5 voxels beside it: 5 mm
    d = tmp_path / "c"
    np.savez(_save(d / "lab.nii.gz", lab, aff).parent / "p.npz", probabilities=np.ascontiguousarray(probs.transpose(0, 3, 2, 1)))
    s = case_scan(("c", "infarct", d / "lab.nii.gz", _save(d / "pred.nii.gz", pred, aff), d / "p.npz",
                   _save(d / "seg.nii.gz", seg, aff), 2.5))
    assert [(r["n_voxels"], r["mm3"], r["ignore"]) for r in s["gt"]] == [(5, 12.5, False), (3, 7.5, True)]
    assert sorted((r["n_voxels"], r["mm3"], r["host"], r["host_rule"], r["side"]) for r in s["dets"]) == [
        (4, 10.0, "white_matter", "nearest", "right"), (5, 12.5, "white_matter", "overlap", "left")]
    far = next(r for r in s["dets"] if r["host_rule"] == "nearest")
    assert far["host_distance_mm"] == 5.0 and far["host_sides"] == {"white_matter": "right"}
    # one of the two detections is bound by the nearest rule, 5 mm away: near enough to be located
    assert binding_agreement([s], 0.5)["nearest_rate"] == 0.5 and binding_agreement([s], 0.5)["unlocated_rate"] == 0.0


def test_the_two_side_agreements_are_counted_apart():
    box = (2, 2, 2, 6, 6, 5)
    gt = {"box": box, "family": "tumor", "host": "thalamus", "side": "right", "host_side": "left"}
    det = {"box": box, "family": "tumor", "score": 0.875, "host": "thalamus", "host_rule": "overlap", "side": "right",
           "host_side": "right", "host_distance_mm": 0.0}
    out = binding_agreement([{"case": "c", "gt": [gt], "dets": [det]}], 0.5)
    assert (out["n_pairs"], out["side_agreement"], out["host_side_agreement"]) == (1, 1.0, 0.0)


def test_the_code_version_names_the_commit_and_marks_changed_files(tmp_path):
    import subprocess
    assert code_version(tmp_path / "no_such_folder") == "unknown"
    repo = tmp_path / "repo"
    repo.mkdir()
    git = ["git", "-c", "user.name=t", "-c", "user.email=t@example.org", "-c", "commit.gpgsign=false"]
    subprocess.run(git + ["init", "-q"], cwd=repo, check=True)
    (repo / "a.txt").write_text("one\n")
    subprocess.run(git + ["add", "a.txt"], cwd=repo, check=True)
    subprocess.run(git + ["commit", "-q", "-m", "one"], cwd=repo, check=True)
    clean = code_version(repo)
    assert len(clean) >= 7 and not clean.endswith("+")
    (repo / "untracked.txt").write_text("x\n")
    assert code_version(repo) == clean                         # files git does not track do not count
    (repo / "a.txt").write_text("two\n")
    assert code_version(repo) == clean + "+"
