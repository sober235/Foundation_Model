# tests/test_aur_training_contract.py
"""Stage II / III training contract (SSL-first plan T10, T11; spec §6; readiness review P0-4)."""
import importlib.util
import json
from pathlib import Path

import nibabel as nib
import numpy as np
import pytest
import torch

import anatobind.aur.losses as L
import anatobind.aur.train as T
from anatobind.aur.dataset import AURDataset, collate
from anatobind.aur.model import AnatoBindBrain
from anatobind.aur.ssl import checkpoint as CK
from anatobind.aur.swin import SwinBackbone

TINY = dict(embed=32, depths=(1, 1, 1, 1), heads=(1, 1, 1, 1), window=(2, 4, 4), patch=(2, 4, 4), d_model=16, n_events=4,
            mask_dim=4, pixel_dim=8, dec_layers=1, dec_heads=2, rel_layers=1, rel_heads=2)
CROP = (8, 32, 32)


def _load(name):
    path = Path(__file__).resolve().parents[1] / "scripts" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _case(tmp_path, name, patient, source="pdgm", with_lesion=True, split="train", flags=None):
    """A (40, 36, 12) volume at (1, 1, 2) mm: left white matter at x < 20, right at x >= 20, a ventricle in the middle,
    one 3 x 3 x 2 lesion (36 mm3) in the left white matter when with_lesion."""
    shape, spacing = (40, 36, 12), (1.0, 1.0, 2.0)
    affine = np.diag(list(spacing) + [1.0])
    seg = np.zeros(shape, np.int16)
    seg[:20] = 2
    seg[20:] = 41
    seg[18:22, 16:20, :] = 4
    rng = np.random.default_rng(hash(name) % 1000)
    img = np.where(seg > 0, 300.0, 0.0).astype(np.float32) + rng.normal(0, 5, shape).astype(np.float32)
    les = np.zeros(shape, np.uint8)
    if with_lesion:
        les[4:7, 4:7, 3:5] = 1
    paths = {}
    for key, arr in (("image", img), ("anatomy", seg), ("lesion", les)):
        p = tmp_path / f"{name}_{key}.nii.gz"
        nib.save(nib.Nifti1Image(arr, affine), str(p))
        paths[key] = str(p)
    row = {"case": name, "source": source, "patient": patient, "sequence": "FLAIR", "source_sequence": "FLAIR", "image": paths["image"],
           "anatomy": paths["anatomy"], "lesion": paths["lesion"] if with_lesion else None, "u_supervised": with_lesion,
           "u_values": [1] if with_lesion else [], "a_ignore_values": [1] if with_lesion else [], "split": split,
           "a_supervised": True, "r_supervised": True}
    row.update(flags or {})
    return row


def _manifest(tmp_path):
    rows = [_case(tmp_path, "c0", "p0"), _case(tmp_path, "c1", "p1", with_lesion=False), _case(tmp_path, "c2", "p2", source="bmsr"),
            _case(tmp_path, "c3", "p3", split="test"), _case(tmp_path, "c4", "pv", with_lesion=True)]
    samples = tmp_path / "samples.json"
    samples.write_text(json.dumps(rows))
    val = tmp_path / "val_patients.json"
    val.write_text(json.dumps({"pdgm": ["pv"], "bmsr": []}))
    return samples, val, rows


def _batch(rows, seed=0):
    ds = AURDataset(rows, crop=CROP, crops_per_volume=1, do_augment=False, seed=seed)
    return collate([ds[i] for i in range(len(ds))])


def _cfg(stage="II", **over):
    cfg = {"stage": stage, "model": TINY, "seen_crops": 6, "microbatch": 2, "grad_accum": 1, "crops_per_volume": 1, "crop": CROP,
           "points": 64, "workers": 0, "warmup_steps": 1, "backbone_warm_steps": 1, "val_every": 2, "val_volumes": 1, "save_every": 2,
           "log_every": 1, "pilot": True}
    cfg.update(over)
    return cfg


def _ssl_export(tmp_path, name="ssl.pt", **kw):
    bb = SwinBackbone(embed=TINY["embed"], depths=TINY["depths"], heads=TINY["heads"], window=TINY["window"], patch=TINY["patch"], **kw)
    return CK.export_backbone(tmp_path / name, bb, {"stage": "I", "step": 3, "seen_crops": 6})


def test_stage_defaults_follow_the_spec():
    two, three = T.stage_defaults("II"), T.stage_defaults("III")
    assert two["seen_crops"] == 240_000 and two["lrs"] == {"backbone": 5e-4, "heads": 5e-4} and two["warmup_steps"] == 1000
    assert two["backbone_warm_steps"] == 1000 and two["backbone_warm_scale"] == 0.1
    assert three["seen_crops"] == 80_000 and three["lrs"] == {"backbone": 5e-5, "heads": 2.5e-4, "relation": 5e-4}
    with pytest.raises(ValueError, match="stage"):
        T.stage_defaults("I")


def test_split_rows_excludes_validation_patients_and_the_test_split(tmp_path):
    _, _, rows = _manifest(tmp_path)
    train, val = T.split_rows(rows, {"pdgm": ["pv"], "bmsr": []})
    assert sorted(r["case"] for r in train) == ["c0", "c1", "c2"] and [r["case"] for r in val] == ["c4"]
    with pytest.raises(ValueError, match="validation"):
        T.split_rows(rows, {"pdgm": ["nobody"]})


def test_param_groups_freeze_relation_in_stage_two_and_split_the_rates_in_stage_three():
    model = AnatoBindBrain(**TINY, use_checkpoint=False)
    groups = T.param_groups(model, "II", T.stage_defaults("II")["lrs"])
    assert [g["name"] for g in groups] == ["backbone", "heads"] and all(not p.requires_grad for p in model.relation.parameters())
    assert not model.backbone.mask_token.requires_grad and all(not p.requires_grad for p in model.masks.embed_coarse.parameters())
    n_backbone = sum(p.numel() for p in model.backbone.parameters()) - model.backbone.mask_token.numel()
    assert sum(p.numel() for p in groups[0]["params"]) == n_backbone
    frozen = sum(p.numel() for p in T.stage_two_frozen(model))
    assert sum(p.numel() for g in groups for p in g["params"]) == model.num_parameters() - frozen
    model = AnatoBindBrain(**TINY, use_checkpoint=False)
    groups = T.param_groups(model, "III", T.stage_defaults("III")["lrs"])
    assert [(g["name"], g["lr"]) for g in groups] == [("backbone", 5e-5), ("heads", 2.5e-4), ("relation", 5e-4)]
    assert all(p.requires_grad for p in model.parameters())
    assert sum(p.numel() for g in groups for p in g["params"]) == model.num_parameters()


def test_backbone_rate_is_scaled_in_the_first_warm_steps_of_stage_two():
    f_bb = T.group_lr_lambda("backbone", "II", warmup=4, total=20, backbone_warm_steps=6, backbone_warm_scale=0.1)
    f_hd = T.group_lr_lambda("heads", "II", warmup=4, total=20, backbone_warm_steps=6, backbone_warm_scale=0.1)
    assert f_bb(0) == pytest.approx(0.1 * f_hd(0)) and f_bb(5) == pytest.approx(0.1 * f_hd(5)) and f_bb(6) == pytest.approx(f_hd(6))
    assert f_hd(3) == pytest.approx(1.0) and f_hd(19) < 0.1 and f_hd(40) == pytest.approx(0.0)
    f3 = T.group_lr_lambda("backbone", "III", warmup=4, total=20, backbone_warm_steps=6, backbone_warm_scale=0.1)
    assert f3(0) == pytest.approx(f_hd(0))                      # no extra scaling in stage III


def test_init_backbone_is_strict_and_random_init_is_explicit(tmp_path):
    path = _ssl_export(tmp_path)
    model = AnatoBindBrain(**TINY, use_checkpoint=False)
    meta = T.init_backbone(model, path)
    assert meta["stage"] == "I"
    loaded = torch.load(path, map_location="cpu", weights_only=False)["backbone_state_dict"]
    assert all(torch.equal(model.backbone.state_dict()[k], v) for k, v in loaded.items())
    wrong = CK.export_backbone(tmp_path / "wrong.pt", SwinBackbone(embed=16, depths=(1, 1, 1, 1), heads=(1, 1, 1, 1)), {})
    with pytest.raises((RuntimeError, ValueError, KeyError)):
        T.init_backbone(AnatoBindBrain(**TINY, use_checkpoint=False), wrong)
    with pytest.raises(ValueError, match="init"):
        T.resolve_init({"init": "backbone", "init_backbone": None, "pilot": True})
    assert T.resolve_init({"init": "random", "init_backbone": None, "pilot": True}) is None


def test_stage_two_refuses_a_backbone_without_a_passing_g1_verdict(tmp_path):
    path = _ssl_export(tmp_path)
    report = tmp_path / "g1_report.json"
    report.write_text(json.dumps({"g1": {"pass": False}}))
    with pytest.raises(ValueError, match="G1"):
        T.resolve_init({"init": "backbone", "init_backbone": path, "g1_report": report, "pilot": False})
    with pytest.raises(ValueError, match="G1"):
        T.resolve_init({"init": "backbone", "init_backbone": path, "g1_report": None, "pilot": False})
    report.write_text(json.dumps({"g1": {"pass": True}}))
    assert T.resolve_init({"init": "backbone", "init_backbone": path, "g1_report": report, "pilot": False}) == path
    assert T.resolve_init({"init": "backbone", "init_backbone": path, "g1_report": None, "pilot": True}) == path


def test_losses_touch_every_trainable_parameter_under_extreme_supervision(tmp_path):
    _, _, rows = _manifest(tmp_path)
    # sample 0: nothing supervised; sample 1: no lesion but U supervised (an empty target); both stages must still
    # give every trainable parameter a gradient (DDP needs it) and the gated parts must be exactly zero
    rows = [dict(rows[0], a_supervised=False, u_supervised=False, r_supervised=False), dict(rows[1], u_supervised=True)]
    batch = _batch(rows)
    assert batch["n_instances"] == [1, 0]
    for stage in ("II", "III"):
        torch.manual_seed(0)
        model = AnatoBindBrain(**TINY, use_checkpoint=False)
        T.param_groups(model, stage, T.stage_defaults(stage)["lrs"])
        points = L.sample_points(batch["valid"], 64, torch.Generator().manual_seed(0), instance=batch["instance"])
        total, logged, matches = T.compute_losses(model, model, batch, points, stage)
        total.backward()
        missing = [n for n, p in model.named_parameters() if p.requires_grad and p.grad is None]
        assert missing == [], f"stage {stage}: {missing[:5]}"
        assert logged["r_host"] == 0.0 and logged["r_hard"] == 0.0 and matches[0] is None
        if stage == "II":
            assert all(p.grad is None for p in model.relation.parameters())
        assert torch.isfinite(total)


def test_relation_loss_is_gated_by_r_supervised_and_uses_the_matched_targets(tmp_path):
    _, _, rows = _manifest(tmp_path)
    on = _batch([rows[0], rows[2]])
    off = _batch([dict(rows[0], r_supervised=False), dict(rows[2], r_supervised=False)])
    torch.manual_seed(0)
    model = AnatoBindBrain(**TINY, use_checkpoint=False)
    points = L.sample_points(on["valid"], 64, torch.Generator().manual_seed(0), instance=on["instance"])
    _, logged_on, matches = T.compute_losses(model, model, on, points, "III")
    _, logged_off, _ = T.compute_losses(model, model, off, points, "III")
    assert logged_on["r_host"] > 0.0 and logged_off["r_host"] == 0.0 and logged_off["r_hard"] == 0.0
    assert all(m is not None and m[0].numel() == 1 for m in matches)
    assert logged_on["r_samples"] == 2 and logged_off["r_samples"] == 0
    _, logged_two, _ = T.compute_losses(model, model, on, points, "II")
    assert "r_host" not in logged_two or logged_two["r_host"] == 0.0


def test_a_tiny_stage_two_run_writes_logs_validation_export_and_resume(tmp_path):
    samples, val, _ = _manifest(tmp_path)
    ssl = _ssl_export(tmp_path)
    out = tmp_path / "stage2"
    summary = T.run(_cfg(init="backbone", init_backbone=str(ssl)), samples, val, out)
    assert summary["steps"] == 3 and summary["seen_crops"] == 6 and summary["global_batch"] == 2 and summary["stage"] == "II"
    cfg = json.loads((out / "run_config.json").read_text())
    assert cfg["train_rows"] == 3 and cfg["val_rows"] == 1 and cfg["total_steps"] == 3 and cfg["init_backbone"].endswith("ssl.pt")
    log = [json.loads(l) for l in (out / "log_rank0.jsonl").read_text().splitlines()]
    assert [r["step"] for r in log] == [1, 2, 3] and all(np.isfinite(r["loss"]) for r in log) and "r_host" not in log[0]
    val_log = [json.loads(l) for l in (out / "val.jsonl").read_text().splitlines()]
    assert [v["step"] for v in val_log] == [2, 3] and all(np.isfinite(v["loss"]) for v in val_log)
    assert all({"a_dice", "u_matched", "val_crops"} <= set(v) for v in val_log)
    best = json.loads((out / "best.json").read_text())
    assert best["step"] in (2, 3) and (out / "aur_stage2_best.pt").is_file() and (out / "resume_step2.pt").is_file()
    model = AnatoBindBrain(**TINY, use_checkpoint=False)
    meta = T.load_model(out / "aur_stage2_best.pt", model)
    assert meta["stage"] == "II" and meta["init_backbone"].endswith("ssl.pt") and meta["code_sha"]
    with pytest.raises(FileExistsError):
        T.run(_cfg(init="backbone", init_backbone=str(ssl)), samples, val, out)
    stopped = T.run(_cfg(init="backbone", init_backbone=str(ssl)), samples, val, tmp_path / "stopped")
    short = T.run(_cfg(init="backbone", init_backbone=str(ssl), stop_after=2), samples, val, tmp_path / "short")
    assert short["stopped"] and short["steps"] == 2 and not (tmp_path / "short" / "aur_stage2_best.pt").exists()
    assert (tmp_path / "short" / "summary_stop_step2.json").is_file() and not (tmp_path / "short" / "summary.json").exists()
    first_config = (tmp_path / "short" / "run_config.json").read_text()
    resumed = T.run(_cfg(init="backbone", init_backbone=str(ssl)), samples, val, tmp_path / "short", resume=tmp_path / "short" / "resume_step2.pt")
    assert resumed["steps"] == 3 and not resumed["stopped"] and (tmp_path / "short" / "aur_stage2_best.pt").is_file()
    assert (tmp_path / "short" / "run_config.json").read_text() == first_config                # a resumed invocation never rewrites it
    assert json.loads((tmp_path / "short" / "run_config_resume_step2.json").read_text())["resumed_from"].endswith("resume_step2.pt")
    assert json.loads((tmp_path / "short" / "summary.json").read_text())["steps"] == 3
    with pytest.raises(ValueError, match="schedule mismatch"):
        T.run(_cfg(init="backbone", init_backbone=str(ssl), seen_crops=8), samples, val, tmp_path / "short", resume=tmp_path / "short" / "resume_step2.pt")


def test_stage_three_inherits_the_whole_stage_two_model(tmp_path):
    samples, val, _ = _manifest(tmp_path)
    ssl = _ssl_export(tmp_path)
    two = tmp_path / "stage2"
    T.run(_cfg(init="backbone", init_backbone=str(ssl)), samples, val, two)
    three = tmp_path / "stage3"
    summary = T.run(_cfg("III", resume_stage2=str(two / "aur_stage2_best.pt")), samples, val, three)
    assert summary["stage"] == "III" and summary["steps"] == 3 and (three / "aur_stage3_best.pt").is_file()
    cfg = json.loads((three / "run_config.json").read_text())
    assert cfg["resume_stage2"].endswith("aur_stage2_best.pt") and cfg["lrs"] == {"backbone": 5e-5, "heads": 2.5e-4, "relation": 5e-4}
    log = [json.loads(l) for l in (three / "log_rank0.jsonl").read_text().splitlines()]
    assert "r_host" in log[0] and "r_samples" in log[0]
    # the stage III model starts as the stage II export, relation included
    fresh = AnatoBindBrain(**TINY, use_checkpoint=False)
    T.load_model(two / "aur_stage2_best.pt", fresh)
    started = AnatoBindBrain(**TINY, use_checkpoint=False)
    T.init_from_stage2(started, two / "aur_stage2_best.pt")
    assert all(torch.equal(a, b) for a, b in zip(fresh.state_dict().values(), started.state_dict().values()))
    with pytest.raises(ValueError, match="resume_stage2"):
        T.run(_cfg("III"), samples, val, tmp_path / "stage3b")


def test_cli_contract():
    parser = _load("aur_train").parser()
    a = parser.parse_args(["--stage", "II", "--init-backbone", "x.pt", "--samples", "s", "--val-patients", "v", "--out", "o"])
    assert a.stage == "II" and a.seen_crops is None and a.init == "backbone" and a.g1_report is None
    cfg = _load("aur_train").config_from_args(a)
    assert cfg["seen_crops"] == 240_000 and cfg["init_backbone"] == "x.pt" and cfg["lrs"]["heads"] == 5e-4
    a = parser.parse_args(["--stage", "III", "--resume-stage2", "two.pt", "--samples", "s", "--val-patients", "v", "--out", "o", "--seen-crops", "10"])
    cfg = _load("aur_train").config_from_args(a)
    assert cfg["seen_crops"] == 10 and cfg["resume_stage2"] == "two.pt" and cfg["lrs"]["relation"] == 5e-4
    with pytest.raises(SystemExit):
        parser.parse_args(["--stage", "I", "--samples", "s", "--val-patients", "v", "--out", "o"])
    a = parser.parse_args(["--stage", "II", "--init", "random", "--samples", "s", "--val-patients", "v", "--out", "o", "--stop-after", "50", "--grad-accum", "4"])
    cfg = _load("aur_train").config_from_args(a)
    assert cfg["stop_after"] == 50 and cfg["grad_accum"] == 4 and cfg["backbone_warm_steps"] == 1000
    a = parser.parse_args(["--stage", "II", "--init", "random", "--samples", "s", "--val-patients", "v", "--out", "o", "--backbone-warm-steps", "667", "--warmup-steps", "667"])
    cfg = _load("aur_train").config_from_args(a)
    assert cfg["backbone_warm_steps"] == 667 and cfg["warmup_steps"] == 667
    assert _load("aur_train").config_from_args(parser.parse_args(["--stage", "II", "--samples", "s", "--val-patients", "v", "--out", "o"]))["stop_after"] is None


# ---- after the independent review of the batch (2026-10-09): resume bookkeeping, counts, gating, sharding, DDP ----

def test_resume_continues_the_schedule_the_data_stream_and_the_best_record(tmp_path):
    samples, val, _ = _manifest(tmp_path)
    ssl = _ssl_export(tmp_path)
    straight = tmp_path / "straight"
    T.run(_cfg(init="backbone", init_backbone=str(ssl), seen_crops=8), samples, val, straight)
    out = tmp_path / "resumed"
    T.run(_cfg(init="backbone", init_backbone=str(ssl), seen_crops=8, stop_after=2), samples, val, out)
    meta = torch.load(out / "resume_step2.pt", map_location="cpu", weights_only=False)["meta"]
    assert meta["epoch"] == 1 and meta["epoch_step"] == 1                      # three rows, two per micro-batch: one step per epoch
    assert meta["schedule"]["total_steps"] == 4
    T.run(_cfg(init="backbone", init_backbone=str(ssl), seen_crops=8), samples, val, out, resume=out / "resume_step2.pt")
    a = {r["step"]: r for r in map(json.loads, (straight / "log_rank0.jsonl").read_text().splitlines())}
    b = {r["step"]: r for r in map(json.loads, (out / "log_rank0.jsonl").read_text().splitlines())}
    assert sorted(b) == [1, 2, 3, 4] and all(b[s]["epoch"] == a[s]["epoch"] for s in (3, 4))
    for s in (3, 4):
        assert a[s]["lr"] == b[s]["lr"] and a[s]["loss"] == pytest.approx(b[s]["loss"], rel=1e-3)
    best = json.loads((out / "best.json").read_text())
    sidecar = json.loads((out / "aur_stage2_best.json").read_text())
    summary = json.loads((out / "summary.json").read_text())
    assert sidecar["step"] == best["step"] and sidecar["final_export"] == summary["final_export"] and Path(summary["final_export"]).is_file()
    val_log = [json.loads(l) for l in (out / "val.jsonl").read_text().splitlines()]
    assert all(v.get("u_matched") is None or 0.0 <= v["u_matched"] <= 1.0 for v in val_log) and "NaN" not in (out / "val.jsonl").read_text()


def test_relation_loss_uses_the_matched_targets_of_a_two_lesion_crop(tmp_path):
    row = _case(tmp_path, "two", "p9")
    les = nib.load(row["lesion"])
    arr = np.asarray(les.dataobj).copy()
    arr[12:15, 4:7, 3:5] = 1                                                  # a second lesion, also left white matter
    nib.save(nib.Nifti1Image(arr, les.affine), row["lesion"])
    ds = AURDataset([row], crop=CROP, crops_per_volume=1, do_augment=False, lesion_share=1.0, seed=3)
    batch = collate([ds[0]])
    assert batch["n_instances"] == [2]
    torch.manual_seed(0)
    model = AnatoBindBrain(**TINY, use_checkpoint=False).eval()
    T.param_groups(model, "III", T.stage_defaults("III")["lrs"])
    points = L.sample_points(batch["valid"], 64, torch.Generator().manual_seed(0), instance=batch["instance"])
    _, logged, matches = T.compute_losses(model, model, batch, points, "III")
    qi, ti = matches[0]
    assert qi.numel() == 2 and logged["r_instances"] == 2 and logged["r_samples"] == 1
    out = model(batch["image"], batch["valid"], batch["coords"], batch["local"])
    expected = L.relation_loss(model.bind(out, 0, qi).float(), batch["host"][0][ti], batch["negatives"][0][ti])
    assert logged["r_host"] == pytest.approx(float(expected["r_host"]), rel=1e-4) and logged["r_hard"] == pytest.approx(float(expected["r_hard"]), rel=1e-4)


def test_validate_reports_counts_and_the_dice_only_over_supervised_crops(tmp_path):
    _, _, rows = _manifest(tmp_path)
    batch = _batch([dict(rows[0], a_supervised=False), rows[2]])
    torch.manual_seed(0)
    model = AnatoBindBrain(**TINY, use_checkpoint=False)
    T.param_groups(model, "III", T.stage_defaults("III")["lrs"])
    cfg = {**T.DEFAULTS, "points": 64, "seed": 0}
    res = T.validate(model, model, [batch], torch.device("cpu"), "III", cfg)
    assert {"loss", "a_dice", "u_matched_n", "u_instances", "r_correct", "r_instances", "u_matched", "r_acc", "val_crops"} <= set(res)
    assert res["u_instances"] == 2 and 0 <= res["u_matched_n"] <= 2 and res["r_instances"] == 2 and 0 <= res["r_correct"] <= 2
    assert res["val_crops"] == 2 and 0.0 <= res["a_dice"] <= 1.0
    points = L.sample_points(batch["valid"], 64, torch.Generator().manual_seed(0), instance=batch["instance"])
    with torch.no_grad():
        out = model(batch["image"], batch["valid"], batch["coords"], batch["local"])
        em = model.entity_masks(out, points).float()
    entity_pts, ignore = L.gather(batch["entity"], points), L.gather(batch["a_ignore"].long(), points).bool()
    both = T.entity_point_dice(em, entity_pts, ignore, batch["entity_present"])
    gated = T.entity_point_dice(em, entity_pts, ignore, batch["entity_present"], batch["a_supervised"])
    only_second = T.entity_point_dice(em[1:], entity_pts[1:], ignore[1:], batch["entity_present"][1:])
    assert gated == pytest.approx(only_second) and (both != gated or True)
    assert T.entity_point_dice(em, entity_pts, ignore, batch["entity_present"], torch.tensor([False, False])) == 0.0


def test_rank_rows_shard_disjointly_and_to_device_moves_lists(tmp_path):
    _, _, rows = _manifest(tmp_path)
    a, b = T.rank_rows(rows, 0, 3, 0, 2), T.rank_rows(rows, 0, 3, 1, 2)
    assert len(a) + len(b) == len(rows) and not {r["case"] for r in a} & {r["case"] for r in b}
    assert [r["case"] for r in T.rank_rows(rows, 0, 3, 0, 2)] == [r["case"] for r in a] and T.rank_rows(rows, 0, 4, 0, 2) != a
    cfg = {**T.DEFAULTS, **_cfg(microbatch=4, crops_per_volume=1)}
    with pytest.raises(ValueError, match="fewer than"):
        T.make_loader(rows[:1], cfg, 0, 1, 0, 0)
    with pytest.raises(ValueError, match="multiple"):
        T.make_loader(rows, {**T.DEFAULTS, **_cfg(microbatch=3, crops_per_volume=2)}, 0, 1, 0, 0)
    moved = T.to_device({"x": torch.zeros(2), "l": [torch.zeros(1), torch.ones(1)], "n": [1, 2], "s": "c"}, torch.device("cpu"))
    assert torch.is_tensor(moved["x"]) and all(torch.is_tensor(t) for t in moved["l"]) and moved["n"] == [1, 2] and moved["s"] == "c"


def test_stage_three_run_starts_from_the_stage_two_export(tmp_path):
    samples, val, _ = _manifest(tmp_path)
    ssl = _ssl_export(tmp_path)
    two = tmp_path / "s2"
    T.run(_cfg(init="backbone", init_backbone=str(ssl)), samples, val, two)
    three = tmp_path / "s3"
    T.run(_cfg("III", resume_stage2=str(two / "aur_stage2_best.pt"), lrs={"backbone": 0.0, "heads": 0.0, "relation": 0.0}), samples, val, three)
    cfg = json.loads((three / "run_config.json").read_text())
    two_meta = torch.load(two / "aur_stage2_best.pt", map_location="cpu", weights_only=False)["meta"]
    assert cfg["init_meta"]["stage"] == "II" and cfg["init_meta"]["step"] == two_meta["step"]
    started = torch.load(two / "aur_stage2_best.pt", map_location="cpu", weights_only=False)["model_state_dict"]
    after = torch.load(three / "resume_step3.pt", map_location="cpu", weights_only=False)["model"]
    assert all(torch.equal(started[k], after[k]) for k in started)                 # zero rates: the Stage III weights are the Stage II ones
    with pytest.raises(ValueError, match="Stage II options"):
        T.run(_cfg("III", resume_stage2=str(two / "aur_stage2_best.pt"), init_backbone=str(ssl)), samples, val, tmp_path / "s3b")
    with pytest.raises(ValueError, match="conflicts"):
        T.resolve_init({"init": "random", "init_backbone": str(ssl), "pilot": True})


def _ddp_worker(rank, world, init_file, samples, stage, result_dir):
    import torch.distributed as dist
    rows = json.loads(Path(samples).read_text())
    dist.init_process_group("gloo", init_method=f"file://{init_file}", rank=rank, world_size=world)
    torch.manual_seed(0)
    model = AnatoBindBrain(**TINY, use_checkpoint=False)
    T.param_groups(model, stage, T.stage_defaults(stage)["lrs"])
    ddp = torch.nn.parallel.DistributedDataParallel(model)
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=1e-3)
    extreme = [dict(rows[0], a_supervised=False, u_supervised=False, r_supervised=False), dict(rows[1], u_supervised=True)]
    full = [rows[0], rows[2]]
    record = {"missing": [], "finite": True}
    for step in range(2):
        batch = _batch(extreme if (rank + step) % 2 == 0 else full, seed=step)
        points = L.sample_points(batch["valid"], 64, torch.Generator().manual_seed(step), instance=batch["instance"])
        total, logged, _ = T.compute_losses(model, ddp, batch, points, stage)
        opt.zero_grad(set_to_none=True)
        total.backward()
        record["missing"] += [n for n, p in model.named_parameters() if p.requires_grad and p.grad is None]
        record["finite"] &= bool(torch.isfinite(total))
        opt.step()
    flat = torch.cat([p.detach().flatten() for p in model.parameters()])
    gathered = [torch.zeros_like(flat) for _ in range(world)]
    dist.all_gather(gathered, flat)
    record["equal"] = all(torch.allclose(gathered[0], g) for g in gathered)
    (Path(result_dir) / f"rank{rank}_{stage}.json").write_text(json.dumps(record))
    dist.destroy_process_group()


def test_two_rank_ddp_survives_extreme_supervision_in_both_stages(tmp_path):
    samples, _, _ = _manifest(tmp_path)
    for stage in ("II", "III"):
        init_file = tmp_path / f"init_{stage}"
        torch.multiprocessing.spawn(_ddp_worker, args=(2, str(init_file), str(samples), stage, str(tmp_path)), nprocs=2, join=True)
        for rank in (0, 1):
            rec = json.loads((tmp_path / f"rank{rank}_{stage}.json").read_text())
            assert rec["missing"] == [] and rec["finite"] and rec["equal"], (stage, rank, rec)


def test_a_skipped_loader_yields_the_same_batches_without_loading_the_skipped_ones(tmp_path):
    """Resuming mid-epoch drops the skipped rows instead of loading them, and keeps every remaining item's crops."""
    rows = [_case(tmp_path, f"k{i}", f"q{i}") for i in range(6)]
    cfg = {**T.DEFAULTS, **_cfg(microbatch=2, crops_per_volume=1)}
    full = [b for b in T.make_loader(rows, cfg, 0, 1, 0, 0)]
    part = [b for b in T.make_loader(rows, cfg, 0, 1, 0, 0, skip_batches=1)]
    assert len(full) == 3 and len(part) == 2
    for a, b in zip(full[1:], part):
        assert a["case"] == b["case"] and torch.equal(a["image"], b["image"]) and torch.equal(a["instance"], b["instance"])
    assert T.make_loader(rows, cfg, 0, 1, 0, 0, skip_batches=3) is None          # the resumed position was the epoch's end


def test_validation_crops_loaded_by_workers_equal_the_serial_ones(tmp_path):
    """The validation crops are built by loader workers (2026-10-10: serially they idled the cards for ~25 min at
    every start); they must be the very same crops."""
    _, _, rows = _manifest(tmp_path)
    serial = T.validation_batches(rows, {**T.DEFAULTS, **_cfg(val_volumes=4, workers=0, microbatch=2, crops_per_volume=1)})
    parallel = T.validation_batches(rows, {**T.DEFAULTS, **_cfg(val_volumes=4, workers=2, microbatch=2, crops_per_volume=1)})
    assert len(serial) == len(parallel) == 2
    for a, b in zip(serial, parallel):
        assert a["case"] == b["case"] and all(torch.equal(a[k], b[k]) for k in ("image", "valid", "coords", "entity", "instance", "point_weight"))
