# tests/test_aur_ssl_model.py
import pytest
import torch

import anatobind.aur.ssl.checkpoint as CK
import anatobind.aur.ssl.model as M
from anatobind.aur.swin import SwinBackbone

SMALL = {"embed": 32, "depths": (1, 1, 1, 1), "heads": (1, 2, 4, 8)}


def _batch(b=2, shape=(16, 32, 32)):
    torch.manual_seed(0)
    d, h, w = shape
    view = torch.full((b, 1) + shape, -1.0)
    view[:, :, 2:d - 2, 4:h - 4, 4:w - 8] = torch.rand(b, 1, d - 4, h - 8, w - 12) * 1.6 - 0.8
    valid = torch.ones((b,) + shape)
    valid[..., w - 4:] = 0.0
    coords = torch.stack(torch.meshgrid(torch.arange(d) * 2.0, torch.arange(h) * 1.0, torch.arange(w) * 1.0, indexing="ij"))[None].expand(b, 3, d, h, w).contiguous()
    return {"view1": view, "view2": (view + 0.05 * torch.randn_like(view)).where(view > -1.0, view), "valid": valid, "coords": coords,
            "local": coords / 32 - 0.5, "spacing": torch.tensor([[2.0, 1.0, 1.0]] * b), "patient": torch.arange(b, dtype=torch.int64), "seq": torch.zeros(b, dtype=torch.long)}


def test_stage_one_forward_backward_and_statistics():
    model = M.StageOne(**SMALL, mask_ratio=0.5, block_mm=(8.0, 16.0))
    out = model(_batch(), torch.Generator().manual_seed(0))
    for k in ("loss", "mim", "contrast", "contrast_acc", "mim_baseline", "hidden_share"):
        assert torch.isfinite(out[k]), k
    assert out["hidden_voxels"] > 0 and abs(float(out["hidden_share"]) - 0.5) < 0.15 and out["z1"].shape == (2, 128)
    out["loss"].backward()
    for name, part in (("backbone", model.backbone), ("decoder", model.decoder), ("projector", model.projector)):
        assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in part.parameters()), name
    assert model.backbone.mask_token.grad is not None
    with pytest.raises(KeyError):
        M.StageOne(mask_ration=0.5)
    same = M.StageOne(**SMALL, mask_ratio=0.5, block_mm=(8.0, 16.0))
    same.load_state_dict(model.state_dict())
    a = same(_batch(), torch.Generator().manual_seed(0))
    b = same(_batch(), torch.Generator().manual_seed(0))
    assert torch.equal(a["mim"], b["mim"])                                        # the same generator seed gives the same masks


def test_the_reconstruction_alone_trains_every_stage_of_the_backbone():
    """The reconstruction must reach all four stages (the 2026-10-09 G1 failure: only stage 1 learnt from it)."""
    torch.manual_seed(0)
    model = M.StageOne(**SMALL, mask_ratio=0.5, block_mm=(8.0, 16.0), contrast_weight=0.0)
    out = model(_batch(), torch.Generator().manual_seed(0))
    out["mim"].backward()
    bb = model.backbone
    assert bb.patch_embed.weight.grad.abs().sum() > 0 and bb.local_embed[0].weight.grad.abs().sum() > 0
    for i, stage in enumerate(bb.stages):
        grads = [p.grad for p in stage.parameters()]
        assert all(g is not None for g in grads) and sum(float(g.abs().sum()) for g in grads) > 0, f"stage {i + 1} gets no reconstruction gradient"
        if stage.merge is not None:
            assert any(p.grad.abs().sum() > 0 for p in stage.merge.parameters()), f"merge after stage {i + 1}"


def test_the_reconstruction_never_sees_a_hidden_voxel():
    torch.manual_seed(0)
    model = M.StageOne(**SMALL).eval()
    batch = _batch(b=1)
    from anatobind.aur.ssl.masking import batch_masks, foreground_patches, hidden_voxels
    fg = foreground_patches(batch["view1"], batch["valid"])
    hidden = batch_masks(fg, torch.Generator().manual_seed(3), ratio=0.5, block_mm=(8.0, 16.0), spacing_mm=(2.0, 1.0, 1.0))
    assert hidden.any()
    with torch.no_grad():
        ref, _, _ = model.view(batch["view1"], batch["valid"], batch["coords"], batch["local"], hidden)
        tampered = batch["view1"].clone()
        hv = hidden_voxels(hidden)
        tampered[:, 0][hv] = torch.rand(int(hv.sum())) * 4 - 2
        same, _, _ = model.view(tampered, batch["valid"], batch["coords"], batch["local"], hidden)
    assert torch.equal(ref, same)


def test_effective_rank_separates_spread_from_collapse():
    spread = torch.randn(64, 16)
    collapsed = torch.randn(64, 1) @ torch.randn(1, 16)
    assert M.effective_rank(spread) > 8 and M.effective_rank(collapsed) < 1.5


def test_backbone_export_loads_strictly_into_stage_two_backbone(tmp_path):
    model = M.StageOne(**SMALL)
    meta = CK.metadata(model.cfg, tmp_path / "missing.json", step=3, seen_crops=48, seed=0, extra={"note": "test"})
    assert meta["manifest_sha256"] is None and meta["step"] == 3 and meta["note"] == "test" and isinstance(meta["code_sha"], str)
    path = CK.export_backbone(tmp_path / "ssl_stage1_best.pt", model.backbone, meta)
    assert (tmp_path / "ssl_stage1_best.json").is_file()
    target = SwinBackbone(**SMALL)
    loaded = CK.load_backbone(path, target)
    assert loaded["seen_crops"] == 48
    for k, v in model.backbone.state_dict().items():
        assert torch.equal(target.state_dict()[k], v)
    with pytest.raises(FileExistsError):
        CK.export_backbone(path, model.backbone, meta)
    other = SwinBackbone(embed=16, depths=(1, 1, 1, 1), heads=(1, 2, 4, 8))
    with pytest.raises(RuntimeError, match="shape mismatch"):
        CK.load_backbone(path, other)
    saved = torch.load(path, weights_only=False)
    assert set(saved) == {"backbone_state_dict", "meta"} and not any(k.startswith("decoder") for k in saved["backbone_state_dict"])


def test_resume_checkpoint_round_trip(tmp_path):
    model = M.StageOne(**SMALL)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: 0.5)
    out = model(_batch(), torch.Generator().manual_seed(1))
    out["loss"].backward()
    opt.step()
    sched.step()
    meta = CK.metadata(model.cfg, None, step=1, seen_crops=16, seed=0)
    p = CK.save_resume(tmp_path / "resume_step1.pt", model, opt, sched, meta)
    fresh = M.StageOne(**SMALL)
    opt2 = torch.optim.AdamW(fresh.parameters(), lr=1e-3)
    sched2 = torch.optim.lr_scheduler.LambdaLR(opt2, lambda s: 0.5)
    meta2, rng = CK.load_resume(p, fresh, opt2, sched2)
    assert meta2["step"] == 1 and sched2.last_epoch == 1 and rng is not None
    assert all(torch.equal(a, b) for a, b in zip(model.state_dict().values(), fresh.state_dict().values()))
    assert opt2.state_dict()["state"].keys() == opt.state_dict()["state"].keys()
    with pytest.raises(FileExistsError):
        CK.save_resume(p, model, opt, sched, meta)
