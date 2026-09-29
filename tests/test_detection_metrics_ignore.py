# tests/test_detection_metrics_ignore.py
import copy

from anatobind.eval.detection_metrics import gate, per_family, scan_matches, sweep


def _gt(box, ignore=None):
    r = {"family": "tumor", "box": box}
    if ignore is not None:
        r["ignore"] = ignore
    return r


def _det(box, score):
    return {"family": "tumor", "box": box, "score": score}


A, B, C, D = (0, 0, 0, 4, 4, 4), (10, 10, 10, 14, 14, 14), (20, 20, 20, 22, 22, 22), (30, 30, 30, 34, 34, 34)


def _scan():
    # ground truth: A and B real, C ignored (a fragment); detections: A found, C found, one stray at D; B missed
    return {"case": "s", "gt": [_gt(A, False), _gt(B, False), _gt(C, True)],
            "dets": [_det(A, 0.9), _det(C, 0.8), _det(D, 0.7)]}


def test_ignored_rows_leave_the_denominator_and_excuse_their_detections():
    hits, fp, dets = scan_matches(_scan(), 0.5)
    assert hits == {0: 0} and fp == 1 and len(dets) == 3          # only the stray at D is a false positive
    row = sweep([_scan()], thresholds=(0.5,))[0]
    assert (row["n_gt"], row["n_hit"], row["n_fp"]) == (2, 1, 1)
    assert row["sensitivity"] == 0.5 and row["fp_per_scan"] == 1.0
    high = sweep([_scan()], thresholds=(0.85,))[0]                # only the detection on A survives
    assert (high["n_gt"], high["n_hit"], high["n_fp"]) == (2, 1, 0)


def test_a_real_lesion_wins_a_detection_over_an_ignored_neighbour():
    # one detection overlaps both a real lesion and an ignored fragment more strongly
    real, frag, det = (0, 0, 0, 4, 4, 4), (1, 1, 1, 4, 4, 4), (1, 1, 1, 4, 4, 4)
    s = {"case": "s", "gt": [_gt(frag, True), _gt(real, False)], "dets": [_det(det, 0.9)]}
    hits, fp, _ = scan_matches(s, 0.5)
    assert hits == {1: 0} and fp == 0


def test_without_flags_the_counts_are_the_plain_ones():
    s = _scan()
    plain = copy.deepcopy(s)
    for r in plain["gt"]:
        r.pop("ignore")
    row = sweep([plain], thresholds=(0.5,))[0]
    assert (row["n_gt"], row["n_hit"], row["n_fp"]) == (3, 2, 1)   # C counts and is found; the stray stays
    false_flags = copy.deepcopy(s)
    for r in false_flags["gt"]:
        r["ignore"] = False
    assert sweep([false_flags], thresholds=(0.5,))[0] == row
    assert gate(sweep([plain]))["pass"] is True                    # 2 of 3 at 1 false positive per scan


def test_per_family_leaves_ignored_rows_out():
    assert per_family([_scan()], 0.5) == {"tumor": {"n_gt": 2, "n_hit": 1, "n_hit_family": 1, "sensitivity": 0.5,
                                                    "sensitivity_family": 0.5}}
    only_a_fragment = {"case": "s", "gt": [_gt(C, True)], "dets": [_det(C, 0.9)]}
    assert per_family([only_a_fragment], 0.5) == {}                # a detection on the fragment is no hit
    plain = copy.deepcopy(_scan())
    for r in plain["gt"]:
        r.pop("ignore")
    assert per_family([plain], 0.5)["tumor"] == {"n_gt": 3, "n_hit": 2, "n_hit_family": 2, "sensitivity": 2 / 3,
                                                 "sensitivity_family": 2 / 3}


def test_a_fragment_excuses_one_detection_and_empty_scans_count_every_detection():
    twice = {"case": "s", "gt": [_gt(C, True)], "dets": [_det(C, 0.9), _det(C, 0.8), _det(D, 0.7)]}
    hits, fp, _ = scan_matches(twice, 0.5)
    assert hits == {} and fp == 2                                  # the second detection on C and the stray at D
    row = sweep([twice, _scan()], thresholds=(0.5,))[0]            # a scan with fragments only adds nothing to n_gt
    assert (row["n_gt"], row["n_hit"], row["n_fp"], row["n_scans"]) == (2, 1, 3, 2)
    empty = {"case": "n", "gt": [], "dets": [_det(A, 0.9), _det(B, 0.6)]}
    assert scan_matches(empty, 0.5)[:2] == ({}, 2)
