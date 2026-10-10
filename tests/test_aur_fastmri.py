# tests/test_aur_fastmri.py
"""AnatoBind-Brain on the fastMRI FLAIR stacks (spec §7, report only; the S4 measure): the stack is resampled to 1 mm,
predicted, and the prediction is brought back to the stack's grid by nearest neighbour; the reliable-slice host Dice
and the box host agreement are S4's functions."""
import nibabel as nib
import numpy as np
import pytest
import torch

import anatobind.aur.fastmri as FX
from anatobind.aur.labels import entity_map
from anatobind.aur.model import AnatoBindBrain

TINY = dict(embed=32, depths=(1, 1, 1, 1), heads=(1, 1, 1, 1), window=(2, 4, 4), patch=(2, 4, 4), d_model=16, n_events=4,
            mask_dim=4, pixel_dim=8, dec_layers=1, dec_heads=2, rel_layers=1, rel_heads=2, use_checkpoint=False)


def _stack(tmp_path, stem="file_brain_AXFLAIR_test"):
    """A (48, 44, 8) LPS stack at (1.0, 1.0, 4.0) mm: left / right white matter, a left thalamus block, a brainstem
    block; the image is brighter in the white matter; SynthSeg map on the same grid."""
    shape = (48, 44, 8)
    seg = np.zeros(shape, np.int16)
    seg[6:24, 6:38, 1:7] = 2            # array x < 24 is image-left in LPS: a left structure in RAS terms is the other side
    seg[24:42, 6:38, 1:7] = 41
    seg[16:24, 18:26, 2:5] = 10
    seg[22:26, 28:34, 1:4] = 16
    img = np.where(seg > 0, 300.0, 0.0).astype(np.float32) + np.random.default_rng(0).normal(0, 3, shape).astype(np.float32)
    aff = np.diag([-1.0, -1.0, 4.0, 1.0])
    aff[:3, 3] = [47.0, 43.0, -10.0]
    ip, sp = tmp_path / f"{stem}_0000.nii.gz", tmp_path / f"{stem}_seg.nii.gz"
    nib.save(nib.Nifti1Image(img, aff), str(ip))
    nib.save(nib.Nifti1Image(seg, aff), str(sp))
    assert "".join(nib.aff2axcodes(aff)) == "LPS"
    return ip, sp, seg


def test_the_geometry_round_trip_keeps_a_perfect_prediction_perfect(tmp_path, monkeypatch):
    """Feed SynthSeg's own map (resampled to the 1 mm grid) as the model's prediction: back on the stack's grid the
    reliable-slice host Dice must be ~1 (this guards the resampling and the axis order of the whole chain)."""
    ip, sp, seg = _stack(tmp_path)

    def oracle(model, image, affine, crop, device, batch_size=1):
        ref = nib.load(str(sp))
        grid = FX.grid_1mm(nib.load(str(ip)))
        on_grid = np.asarray(nib.processing.resample_from_to(ref, grid, order=0).dataobj).astype(np.int16)
        return {"entity": entity_map(on_grid.transpose(2, 1, 0))}                    # (z, y, x), entity values

    monkeypatch.setattr(FX, "predict_volume", oracle)
    res = FX.evaluate_stack(None, ip, sp, rows=[], crop=(8, 16, 16), device=torch.device("cpu"))
    vals = [v for v in res["dice"].values() if v is not None]
    assert res["reliable"] is not None and vals and min(vals) > 0.95
    assert res["pred_native"].shape == seg.shape


def test_a_tiny_model_runs_the_stack_and_the_boxes(tmp_path):
    ip, sp, _ = _stack(tmp_path)
    rows = [{"x0": 16, "x1": 24, "y0": 18, "y1": 26, "z0": 2, "z1": 4, "lesion_id": 1}]
    torch.manual_seed(0)
    model = AnatoBindBrain(**TINY).eval()
    res = FX.evaluate_stack(model, ip, sp, rows=rows, crop=(8, 16, 16), device=torch.device("cpu"))
    assert set(res) >= {"reliable", "dice", "agreement", "pred_native"} and res["agreement"]["n_lesions"] == 1
    summary = FX.summarize({"s1": res, "s2": res})
    assert summary["n_stacks"] == 2 and set(summary) >= {"fastmri_dice", "host_agreement"} and summary["evidence"].startswith("NOT_EVIDENCE")
    with pytest.raises(ValueError, match="differ"):
        bad = tmp_path / "bad_seg.nii.gz"
        nib.save(nib.Nifti1Image(np.zeros((10, 10, 4), np.int16), np.eye(4)), str(bad))
        FX.evaluate_stack(model, ip, bad, rows=[], crop=(8, 16, 16), device=torch.device("cpu"))


def test_the_script_runs_end_to_end_on_a_tiny_export(tmp_path, monkeypatch):
    import importlib.util
    import json
    from pathlib import Path

    import anatobind.aur.train as T

    stacks, segs = tmp_path / "stacks", tmp_path / "segs"
    stacks.mkdir()
    segs.mkdir()
    ip, sp, _ = _stack(stacks)
    sp.rename(segs / sp.name)
    torch.manual_seed(0)
    export = T.export_model(tmp_path / "aur.pt", AnatoBindBrain(**TINY), {"stage": "III", "config": {"model": TINY, "crop": [8, 16, 16]}})
    path = Path(__file__).resolve().parents[1] / "scripts" / "aur_eval_fastmri.py"
    spec = importlib.util.spec_from_file_location("aur_eval_fastmri", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    monkeypatch.setattr(mod, "load_registry", lambda: [{"file": "file_brain_AXFLAIR_test", "x0": 16, "x1": 24, "y0": 18, "y1": 26, "z0": 2, "z1": 4, "lesion_id": 1}])
    out = tmp_path / "rec"
    assert mod.main(["--checkpoint", str(export), "--out", str(out), "--stacks", str(stacks), "--segs", str(segs), "--cpu", "--s4-verdict", str(tmp_path / "none.json")]) == 0
    summary = json.loads((out / "summary.json").read_text())
    assert summary["n_stacks"] == 1 and summary["host_agreement"]["n_lesions"] == 1 and (out / "REPORT.md").is_file()
