# tests/test_aur_ssl_eval.py
import json

import nibabel as nib
import numpy as np
import pytest
import torch

import anatobind.aur.ssl.eval as E
from anatobind.aur.ssl import checkpoint as CK
from anatobind.aur.swin import SwinBackbone

SMALL = {"embed": 32, "depths": (1, 1, 1, 1), "heads": (1, 2, 4, 8)}


def _rows(tmp_path, n=6):
    """n volumes (24, 20, 12) at 1 mm: left white matter x < 12, right x >= 12, a lesion in the left white matter in the
    even ones; patients p0..p{n-1}; the last two are validation patients."""
    rows = []
    rng = np.random.default_rng(0)
    for i in range(n):
        seg = np.zeros((24, 20, 12), np.int16)
        seg[:12] = 2
        seg[12:] = 41
        img = np.where(seg == 2, 200.0, 100.0).astype(np.float32) + rng.normal(0, 5, seg.shape).astype(np.float32)
        les = np.zeros(seg.shape, np.uint8)
        if i % 2 == 0:
            les[3:7, 3:7, 3:7] = 1
            img[3:7, 3:7, 3:7] += 150.0
        paths = {}
        for name, arr in (("image", img), ("anatomy", seg), ("lesion", les)):
            p = tmp_path / f"v{i}_{name}.nii.gz"
            nib.save(nib.Nifti1Image(arr, np.eye(4)), str(p))
            paths[name] = str(p)
        rows.append({"case": f"c{i}", "source": "pdgm", "patient": f"p{i}", "sequence": "FLAIR", "source_sequence": "FLAIR", "image": paths["image"],
                     "anatomy": paths["anatomy"], "lesion": paths["lesion"], "u_supervised": True, "u_values": [1], "a_ignore_values": [1],
                     "a_supervised": True, "r_supervised": True, "split": "train"})
    return rows, {"pdgm": [f"p{n - 2}", f"p{n - 1}"]}


def test_row_selection_respects_the_validation_patients(tmp_path):
    rows, val = _rows(tmp_path)
    cal, vl = E.select_rows(rows, val, 10, 10, seed=0)
    assert {r["patient"] for r in cal} == {"p0", "p1", "p2", "p3"} and {r["patient"] for r in vl} == {"p4", "p5"}
    lc, lv = E.select_rows(rows, val, 10, 10, seed=0, lesion_only=True)
    assert all(r["u_supervised"] for r in lc + lv)


def test_auc_and_bootstrap_helpers():
    assert E.auc([0.9, 0.8, 0.1, 0.2], [1, 1, 0, 0]) == 1.0 and E.auc([0.1, 0.2, 0.9, 0.8], [1, 1, 0, 0]) == 0.0
    assert E.auc([0.5, 0.5, 0.5, 0.5], [1, 1, 0, 0]) == 0.5 and np.isnan(E.auc([0.5], [1]))
    a = [(torch.tensor([1.0] * 13), torch.tensor([1.0] * 13), torch.tensor([1.0] * 13))] * 4
    b = [(torch.tensor([0.5] * 13), torch.tensor([1.0] * 13), torch.tensor([1.0] * 13))] * 4
    d = E.bootstrap_difference(a, b, ["p0", "p0", "p1", "p1"], n_boot=50)
    assert d["mean"] == pytest.approx(0.5) and d["ci95"][0] == pytest.approx(0.5)
    dice, macro = E.dice_from_counts(b)
    assert macro == pytest.approx(0.5) and len(dice) == 13


def test_the_probe_run_writes_a_g1_report(tmp_path):
    rows, val = _rows(tmp_path)
    bb = SwinBackbone(**SMALL)
    ck = CK.export_backbone(tmp_path / "ssl.pt", bb, {"stage": "I", "step": 1})
    report = E.run(rows, val, ck, tmp_path / "g1", torch.device("cpu"), crop=(8, 16, 16), n_calibration=4, n_validation=2, n_lesion=3, seeds=(0, 1), backbone_kwargs=SMALL)
    assert set(report["arms"]) == {"ssl", "random"} and len(report["arms"]["ssl"]["host"]["seeds"]) == 2
    g1 = report["g1"]
    assert set(g1) >= {"macro_dice_gain", "host_pass", "lesion_pass", "pass", "gain_ci_low_min"} and "NOT_EVIDENCE" in g1["evidence"]
    assert np.isfinite(report["arms"]["ssl"]["host"]["macro_dice_mean"]) and report["n_validation_crops"] == 2
    saved = json.loads((tmp_path / "g1" / "g1_report.json").read_text())
    assert saved["thresholds"]["status"] == "PROPOSED" and saved["checkpoint_meta"]["step"] == 1
    with pytest.raises(FileExistsError):
        E.run(rows, val, ck, tmp_path / "g1", torch.device("cpu"), crop=(8, 16, 16), backbone_kwargs=SMALL)


def test_a_host_only_probe_skips_the_lesion_arm_and_gives_no_verdict(tmp_path):
    """The early probe of a running Stage I (2026-10-10): host readout only, one seed, no G1 verdict."""
    rows, val = _rows(tmp_path)
    ck = CK.export_backbone(tmp_path / "ssl.pt", SwinBackbone(**SMALL), {"stage": "I", "step": 1000})
    report = E.run(rows, val, ck, tmp_path / "probe", torch.device("cpu"), crop=(8, 16, 16), n_calibration=4, n_validation=2, seeds=(0,),
                   backbone_kwargs=SMALL, lesion=False)
    assert report["mode"] == "host-only probe" and set(report["arms"]) == {"ssl", "random"}
    assert report["arms"]["ssl"]["lesion"] is None and report["arms"]["random"]["lesion"] is None
    g1 = report["g1"]
    assert g1["pass"] is None and g1["lesion_pass"] is None and "not the G1 verdict" in g1["note"]
    assert np.isfinite(g1["macro_dice_gain"]) and g1["host_macro_ssl"] == report["arms"]["ssl"]["host"]["macro_dice_mean"]
    full = E.run(rows, val, ck, tmp_path / "full", torch.device("cpu"), crop=(8, 16, 16), n_calibration=4, n_validation=2, n_lesion=3, seeds=(0,), backbone_kwargs=SMALL)
    assert full["mode"] == "gate" and full["g1"]["pass"] in (True, False)


def test_shared_parallel_crops_give_the_same_report(tmp_path):
    """The probe crops are built once (in loader workers) and shared by both arms (2026-10-10: serially and per arm a
    probe took 40-70 minutes); the report must not change by a single bit."""
    rows, val = _rows(tmp_path)
    ck = CK.export_backbone(tmp_path / "ssl.pt", SwinBackbone(**SMALL), {"stage": "I", "step": 1})
    kw = dict(crop=(8, 16, 16), n_calibration=4, n_validation=2, n_lesion=3, seeds=(0, 1), backbone_kwargs=SMALL)
    old = E.run(rows, val, ck, tmp_path / "old", torch.device("cpu"), share_crops=False, **kw)
    new = E.run(rows, val, ck, tmp_path / "new", torch.device("cpu"), share_crops=True, workers=2, **kw)
    assert json.dumps(old["arms"], sort_keys=True, default=str) == json.dumps(new["arms"], sort_keys=True, default=str)
    assert json.dumps(old["g1"], sort_keys=True, default=str) == json.dumps(new["g1"], sort_keys=True, default=str)
    probe = E.run(rows, val, ck, tmp_path / "probe", torch.device("cpu"), share_crops=True, workers=2, lesion=False, **kw)
    assert probe["g1"]["host_macro_ssl"] == old["arms"]["ssl"]["host"]["macro_dice_mean"]
