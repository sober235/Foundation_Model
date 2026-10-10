# tests/test_aur_ssl_attrib.py
"""Attribution diagnostics of a Stage I checkpoint (review of 2026-10-10): the gradient of each term per backbone part,
and the reconstruction with levels dropped from the decoder."""
import torch

import anatobind.aur.ssl.attrib as A
import anatobind.aur.ssl.model as M

SMALL = {"embed": 32, "depths": (1, 1, 1, 1), "heads": (1, 2, 4, 8)}


def _batch(b=4, shape=(16, 32, 32)):
    torch.manual_seed(0)
    d, h, w = shape
    view = torch.full((b, 1) + shape, -1.0)
    view[:, :, 2:d - 2, 4:h - 4, 4:w - 8] = torch.rand(b, 1, d - 4, h - 8, w - 12) * 1.6 - 0.8
    valid = torch.ones((b,) + shape)
    coords = torch.stack(torch.meshgrid(torch.arange(d) * 2.0, torch.arange(h) * 1.0, torch.arange(w) * 1.0, indexing="ij"))[None].expand(b, 3, d, h, w).contiguous()
    return {"view1": view, "view2": (view + 0.05 * torch.randn_like(view)).where(view > -1.0, view), "valid": valid, "coords": coords,
            "local": coords / 32 - 0.5, "spacing": torch.tensor([[2.0, 1.0, 1.0]] * b), "patient": torch.tensor([0, 0, 1, 1]), "seq": torch.zeros(b, dtype=torch.long)}


def test_gradient_balance_reports_every_part_for_both_terms():
    model = M.StageOne(**SMALL, mask_ratio=0.5, block_mm=(8.0, 16.0))
    res = A.gradient_balance(model, _batch(), seed=0)
    parts = res["parts"]
    assert set(parts) == {"stem", "stage1.blocks", "stage1.merge", "stage2.blocks", "stage2.merge", "stage3.blocks", "stage3.merge", "stage4.blocks"}
    for name, v in parts.items():
        assert v["mim"] > 0 and v["contrast"] >= 0 and v["ratio_mim_over_contrast"] > 0, name
    assert res["contrast_weight"] == model.cfg["contrast_weight"] and res["mim"] > 0
    assert all(p.grad is None for p in model.parameters())                  # the model is left without gradients


def test_reconstruction_ablation_drops_levels_from_the_decoder():
    model = M.StageOne(**SMALL, mask_ratio=0.5, block_mm=(8.0, 16.0)).eval()
    res = A.reconstruction_ablation(model, _batch(), seed=0)
    assert set(res) >= {"full", "no_f1", "f1_only", "baseline", "hidden_voxels"}
    assert res["hidden_voxels"] > 0 and res["full"] != res["no_f1"] and res["full"] != res["f1_only"]
    again = A.reconstruction_ablation(model, _batch(), seed=0)
    assert again == res                                                      # same seed, same masks, same numbers


def test_the_script_runs_on_a_tiny_manifest(tmp_path):
    import importlib.util
    import json
    from pathlib import Path

    import nibabel as nib
    import numpy as np

    rows = []
    rng = np.random.default_rng(0)
    for i in range(3):
        img = np.zeros((24, 20, 12), np.float32)
        img[4:20, 4:16, 2:10] = rng.uniform(100, 300, (16, 12, 8))
        p = tmp_path / f"v{i}.nii.gz"
        nib.save(nib.Nifti1Image(img, np.eye(4)), str(p))
        rows.append({"case": f"c{i}", "source": "pdgm", "patient": f"p{i}", "sequence": "T1", "source_sequence": "T1", "image": str(p),
                     "anatomy": None, "lesion": None, "split": "train", "ssl_split": "val"})
    m = tmp_path / "samples_ssl.json"
    m.write_text(json.dumps(rows))
    path = Path(__file__).resolve().parents[1] / "scripts" / "aur_ssl_attrib.py"
    spec = importlib.util.spec_from_file_location("aur_ssl_attrib", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    out = tmp_path / "attrib.json"
    assert mod.main(["--checkpoint", "init", "--samples", str(m), "--out", str(out), "--patients", "2", "--crop", "8", "16", "16", "--threads", "4"]) == 0
    res = json.loads(out.read_text())
    assert res["n_crops"] == 4 and "stage4.blocks" in res["gradient_balance"]["parts"] and res["reconstruction"]["hidden_voxels"] > 0
