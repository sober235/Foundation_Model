"""New TRAIN-only A/R measurements; all fixtures are synthetic and run on CPU."""
import importlib
import importlib.util
import hashlib
import json
import stat
from pathlib import Path

import numpy as np
import pytest
import torch


def api():
    assert importlib.util.find_spec("anatobind.aur.ar_diagnostics") is not None, "A/R diagnostics are not implemented"
    return importlib.import_module("anatobind.aur.ar_diagnostics")


def test_entity_statistics_ignore_margin_ties_absence_and_sample_duplicates():
    d = api()
    probs = torch.full((32, 1, 1, 5), .1)
    probs[0, 0, 0] = torch.tensor([.8, .4, .7, .9, .2])
    probs[1, 0, 0] = torch.tensor([.2, .9, .7, .1, .8])
    gt = torch.tensor([[[1, 1, 1, 1, 2]]])
    valid = torch.tensor([[[1., 1., 1., 0., 1.]]])
    ignore = torch.tensor([[[False, True, False, False, False]]])
    points = torch.tensor([0, 0, 1, 2, 3, 4])
    summary = d.entity_query_statistics([(0, probs.logit())], gt, valid, ignore, points)
    a, b, absent = summary[:3]
    assert a["positive_voxels"] == 2 and a["sampled_positive_points"] == 3
    assert a["sampled_unique_positive_voxels"] == 2
    assert a["winner_voxels"] == 2  # the lowest-index query wins an exact tie
    assert a["positive_probability_sum"] == pytest.approx(1.5)
    assert a["winner_margin_sum"] == pytest.approx(.6)
    assert a["sampled_positive_voxel_fraction"] == 1.0
    assert a["sampled_positive_probability_mean"] == pytest.approx(2.3 / 3)
    assert b["positive_voxels"] == 1 and b["winner_voxels"] == 1
    assert absent["positive_probability_mean"] is None and absent["winner_margin_mean"] is None


def test_streaming_blocks_equal_the_complete_matrix():
    d = api()
    g = torch.Generator().manual_seed(4)
    logits = torch.randn(32, 2, 4, 4, generator=g)
    gt = torch.randint(0, 33, (2, 4, 4), generator=g)
    v, ig = torch.ones_like(gt).float(), torch.zeros_like(gt).bool()
    full = d.entity_query_statistics([(0, logits)], gt, v, ig, torch.tensor([0, 4, 4, 15]))
    chunks = [(i, logits[i:i + 3]) for i in range(0, 32, 3)]
    assert d.entity_query_statistics(chunks, gt, v, ig, torch.tensor([0, 4, 4, 15])) == full
    with pytest.raises(ValueError):
        d.entity_query_statistics([(1, logits)], gt, v, ig, torch.tensor([0]))


def test_pooling_averages_probabilities_before_geometry():
    d = api()
    logits = torch.tensor([[[[-8., 2.], [1., 1.]]]])
    actual = d.pool_probabilities(logits, (1, 2, 2))
    assert actual.item() == pytest.approx(logits.sigmoid().mean().item())
    assert not torch.allclose(actual, logits.mean().sigmoid())


SMALL = dict(embed=32, depths=(1, 1, 1, 1), heads=(1, 2, 4, 8), window=(2, 2, 2),
             d_model=16, n_events=3, mask_dim=4, pixel_dim=8, dec_layers=1,
             dec_heads=2, rel_layers=1, rel_heads=2, use_checkpoint=False)


def synthetic_crop():
    shape = (8, 16, 16)
    z, y, x = torch.meshgrid(*(torch.arange(n) for n in shape), indexing="ij")
    image = torch.randn((1,) + shape, generator=torch.Generator().manual_seed(4))
    entity = torch.ones(shape, dtype=torch.long)
    entity[:, :, 8:] = 2
    instance = torch.zeros(shape, dtype=torch.long)
    instance[2:6, 6:10, 6:10] = 1
    return dict(image=image, valid=torch.ones(shape), coords=torch.stack((z, y, x)).float(),
                local=torch.stack((z / 4 - 1, y / 8 - 1, x / 8 - 1)).float(),
                entity=entity, a_ignore=torch.zeros(shape, dtype=torch.bool), instance=instance,
                entity_present=torch.arange(32) < 2, a_supervised=torch.tensor(True),
                point_weight=torch.ones(shape))


def test_one_capture_freezes_tokens_and_matches_original_bind_without_u_decoder(monkeypatch):
    d = api()
    from anatobind.aur.model import AnatoBindBrain
    from anatobind.aur.query_decoder import QueryCCDecoder
    monkeypatch.setattr(QueryCCDecoder, "__init__", lambda *a, **k: pytest.fail("U decoding repeated"))
    torch.manual_seed(3)
    model = AnatoBindBrain(**SMALL).eval().requires_grad_(False)
    calls = []
    hook = model.backbone.register_forward_hook(lambda *args: calls.append(1))
    before = d.state_digest(model.state_dict())
    meta, arrays = d.collect_crop(model, synthetic_crop(), points=32, seed=5, query_chunk=2)
    hook.remove()
    assert len(calls) == 1 and d.state_digest(model.state_dict()) == before
    assert meta["coarse_control_max_abs"] <= 1e-6
    assert meta["tokens_before"] == meta["tokens_after"]
    assert arrays["coarse_geo"].shape == (3, 13, 15)
    assert arrays["dense_logits"].shape == (3, 14)
    assert len(meta["a_queries"]) == 32 and all(p.grad is None for p in model.parameters())
    replay = d.relation_from_arrays(model.relation, arrays)
    np.testing.assert_allclose(replay["coarse_logits"], arrays["coarse_logits"], rtol=1e-5, atol=1e-6)
    same = dict(arrays, dense_a=arrays["coarse_a"], dense_u=arrays["coarse_u"])
    identity = d.relation_from_arrays(model.relation, same)
    np.testing.assert_array_equal(identity["coarse_geo"], identity["dense_geo"])
    np.testing.assert_array_equal(identity["coarse_logits"], identity["dense_logits"])


def test_relation_swap_is_gt_free_and_capture_chunk_size_does_not_select_queries():
    d = api()
    from anatobind.aur.model import AnatoBindBrain
    model = AnatoBindBrain(**SMALL).eval().requires_grad_(False)
    crop = synthetic_crop()
    first_meta, first = d.collect_crop(model, crop, points=16, seed=5, query_chunk=1)
    changed = dict(crop, entity=torch.zeros_like(crop["entity"]), instance=torch.zeros_like(crop["instance"]),
                   a_ignore=torch.ones_like(crop["a_ignore"]))
    second_meta, second = d.collect_crop(model, changed, points=16, seed=5, query_chunk=32)
    for name in d.ARRAY_KEYS:
        np.testing.assert_allclose(first[name], second[name], rtol=1e-5, atol=1e-6)
    assert first_meta["a_queries"][0]["positive_voxels"] > 0
    assert all(r["positive_voxels"] == 0 for r in second_meta["a_queries"])


def test_cli_contract_and_anonymous_failure(tmp_path, capsys):
    d = api()
    a = d.parser().parse_args(["--samples", "fixture-samples", "--val-patients", "fixture-split",
                               "--checkpoint", "fixture-c2", "--out", str(tmp_path / "fresh")])
    assert tuple(a.crop) == (8, 32, 32) and a.per_source == 1 and a.query_chunk == 4
    assert d.main(["--out", str(tmp_path / "private-sentinel-missing")], replay=True) == 1
    assert "private-sentinel" not in capsys.readouterr().out


@pytest.mark.parametrize("replay,extra", [(False, ["--points", "SYNTHETIC_PRIVATE_MARKER"]),
                                         (True, ["--SYNTHETIC_PRIVATE_MARKER"])])
def test_cli_argument_errors_do_not_echo_user_values(tmp_path, capsys, replay, extra):
    d = api()
    argv = ["--out", str(tmp_path / "out")]
    if not replay:
        argv += ["--samples", "fixture", "--val-patients", "fixture", "--checkpoint", "fixture"]
    with pytest.raises(SystemExit) as stopped:
        d.main(argv + extra, replay=replay)
    assert stopped.value.code == 2
    output = capsys.readouterr()
    assert "SYNTHETIC_PRIVATE_MARKER" not in output.out + output.err


@pytest.mark.parametrize("path", ["repo", "outside"])
def test_collection_refuses_output_in_git_or_outside_home_before_loading_inputs(tmp_path, monkeypatch, path):
    d = api()
    from pathlib import Path
    repo = Path(d.__file__).resolve().parents[2]
    out = repo / "synthetic-private-output-must-not-exist" if path == "repo" else tmp_path / "fresh"
    args = d.parser().parse_args(["--samples", "SYNTHETIC_PRIVATE_MARKER", "--val-patients", "fixture",
                                 "--checkpoint", "fixture", "--out", str(out)])
    monkeypatch.setattr(d, "file_digest", lambda *a: pytest.fail("inputs opened before output policy"))
    with pytest.raises(ValueError, match="Private output"):
        d.run_collection(args)
    assert not out.exists()


def test_selection_is_patient_disjoint_deterministic_and_requires_eligible_train_rows():
    d = api()
    rows = []
    for source, sequence in d.PAIRS:
        for p in range(3):
            rows.append(dict(source=source, sequence=sequence, patient=str(p), case=f"fixture-{p}",
                             split="train", u_supervised=True, a_supervised=True, r_supervised=True))
        rows.append(dict(rows[-1], split="test", patient="test"))
    train, selection = d.select_train_rows(rows, {s: ["0"] for s, _ in d.PAIRS}, per_source=1, seed=7)
    assert len(selection) == 3 and all(train[i]["patient"] != "0" for i in selection)
    assert d.select_train_rows(rows, {s: ["0"] for s, _ in d.PAIRS}, 1, 7)[1] == selection
    with pytest.raises(ValueError):
        d.select_train_rows(rows, {}, 10, 7)


def test_archive_replays_without_images_checkpoint_or_full_model_and_detects_tampering(tmp_path, monkeypatch):
    d = api()
    from anatobind.aur.model import AnatoBindBrain
    model = AnatoBindBrain(**SMALL).eval().requires_grad_(False)
    meta, arrays = d.collect_crop(model, synthetic_crop(), points=16, seed=1, query_chunk=2)
    out = tmp_path / "new"
    contract = dict(schema=1, n_crops=1, relation_config=dict(d_model=16, layers=1, heads=2),
                    provenance=dict(checkpoint_sha256="a" * 64), recipe=dict(crop=[8, 16, 16]))
    d.create_archive(out, contract, [{"private_fixture_only": True}], model.relation)
    d.write_crop(out, 0, dict(meta, source="bmsr"), arrays)
    d.complete_archive(out)
    assert stat.S_IMODE(out.stat().st_mode) == 0o700
    assert all(stat.S_IMODE(p.stat().st_mode) == 0o600 for p in out.iterdir())
    monkeypatch.setattr(torch, "load", lambda *a, **k: pytest.fail("checkpoint loaded"))
    monkeypatch.setattr(AnatoBindBrain, "__init__", lambda *a, **k: pytest.fail("full model constructed"))
    report = d.replay_archive(out)
    assert report["n_crops"] == 1 and len(report["a_queries"]) == 32
    public = json.dumps(report)
    assert "private_fixture_only" not in public and "patient" not in public
    (out / "crop_000.json").write_text("{}")
    with pytest.raises(ValueError, match="Invalid A/R diagnostic archive"):
        d.replay_archive(out)
    with pytest.raises(FileExistsError):
        d.create_archive(out, contract, [], model.relation)


def test_stable_distance_retains_float32_centroids_and_resolves_nearby_large_coordinates():
    d = api()
    coords = torch.tensor([[[[100000., 100000.125]]], [[[100000., 100000.]]], [[[100000., 100000.]]]])
    events = torch.zeros(26, 1, 1, 2)
    events[:, 0, 0, 0] = 1
    hosts = torch.zeros(13, 1, 1, 2)
    hosts[0, 0, 0, 1] = 1
    from anatobind.aur.relation import geometry
    native = geometry(events, hosts, coords)
    stable = d.stable_distance_geometry(events, hosts, coords)
    torch.testing.assert_close(stable[:, 0, 4], torch.full((26,), .125 / 100), rtol=0, atol=1e-9)
    assert torch.equal(stable[..., :4], native[..., :4]) and torch.equal(stable[..., 5:], native[..., 5:])
    assert torch.equal(stable[:, 1:, 4], native[:, 1:, 4])  # absent-host cap stays unchanged


def incomplete_native_fixture(tmp_path, extra_failure=None):
    d = api()
    tmp_path.mkdir(parents=True, exist_ok=True)
    from anatobind.aur.model import AnatoBindBrain
    model = AnatoBindBrain(**SMALL).eval().requires_grad_(False)
    meta, arrays = d.collect_crop(model, synthetic_crop(), points=16, seed=1, query_chunk=2)
    # Synthetic failed native replay: cached native geometry is intentionally
    # incompatible, while original probabilities/tokens/weights stay intact.
    arrays["dense_geo"][0, 0, 4] += .01
    if extra_failure == "overlap":
        arrays["dense_geo"][0, 0, 0] += .5
    elif extra_failure == "logit":
        arrays["coarse_logits"][0, 0] += .5
    snapshot = tmp_path / "collection_source"
    snapshot.mkdir(mode=0o700)
    source_bytes = b"# synthetic collection-time source\n"
    source_file = snapshot / "collection.py"
    source_file.write_bytes(source_bytes)
    source_file.chmod(0o600)
    source_hash = hashlib.sha256(source_bytes).hexdigest()
    source = tmp_path / "incomplete_native"
    contract = dict(schema=1, n_crops=1, relation_config=dict(d_model=16, layers=1, heads=2),
                    provenance=dict(checkpoint_sha256="a" * 64, source_sha256={"collection.py": source_hash}),
                    recipe=dict(crop=[8, 16, 16]))
    d.create_archive(source, contract, [{"private_fixture_only": True}], model.relation)
    d.write_crop(source, 0, dict(meta, source="bmsr"), arrays)
    manifest = dict(schema=1, source_sha256={"collection.py": source_hash},
                    native_member_sha256=d._hashes(source), native_replay_passed=False, native_complete_present=False)
    d._write_json(snapshot / "snapshot_manifest.json", manifest)
    return source, snapshot


def test_stable_derivation_keeps_native_fail_and_replays_without_image_or_backbone(tmp_path, monkeypatch):
    d = api()
    source, snapshot = incomplete_native_fixture(tmp_path)
    before = d._hashes(source)
    from anatobind.aur.model import AnatoBindBrain
    import nibabel
    monkeypatch.setattr(nibabel, "load", lambda *a, **k: pytest.fail("medical image opened"))
    monkeypatch.setattr(torch, "load", lambda *a, **k: pytest.fail("checkpoint opened"))
    monkeypatch.setattr(AnatoBindBrain, "__init__", lambda *a, **k: pytest.fail("backbone constructed"))
    target = tmp_path / "stable_derived"
    report = d.derive_stable_analysis(source, target, snapshot)
    assert report["native_replay_passed"] is False and report["derived_replay_passed"] is True
    assert before == d._hashes(source) and not (source / "COMPLETE.json").exists()
    assert d.replay_stable_analysis(target) == report
    assert stat.S_IMODE(target.stat().st_mode) == 0o700
    assert all(stat.S_IMODE(p.stat().st_mode) == 0o600 for p in target.iterdir())
    public = json.dumps(report)
    assert "private_fixture_only" not in public
    contract = json.loads((target / "contract.json").read_text())
    assert contract["tolerance"] == {"rtol": d.RTOL, "atol": d.ATOL}
    with pytest.raises(ValueError):
        d.replay_archive(source)  # derived success never repairs the missing native marker
    (target / "crop_000.npz").write_bytes(b"corrupt")
    with pytest.raises(ValueError, match="Invalid stable A/R analysis"):
        d.replay_stable_analysis(target)


def test_stable_derivation_refuses_missing_source_or_snapshot_drift(tmp_path):
    d = api()
    source, snapshot = incomplete_native_fixture(tmp_path)
    (source / "crop_000.npz").unlink()
    with pytest.raises(ValueError, match="Invalid native A/R source"):
        d.derive_stable_analysis(source, tmp_path / "missing", snapshot)
    assert not (tmp_path / "missing").exists()
    source2, snapshot2 = incomplete_native_fixture(tmp_path / "second")
    (snapshot2 / "collection.py").write_bytes(b"changed")
    with pytest.raises(ValueError, match="Invalid native A/R source"):
        d.derive_stable_analysis(source2, tmp_path / "drift", snapshot2)


def test_stable_cli_requires_explicit_source_snapshot_and_supports_replay():
    d = api()
    args = d.parser(replay=True).parse_args(["--out", "fixture-derived", "--derive-from", "fixture-native",
                                           "--source-snapshot", "fixture-source"])
    assert args.derive_from == Path("fixture-native") and args.source_snapshot == Path("fixture-source")


@pytest.mark.parametrize("extra_failure", ["overlap", "logit"])
def test_distance_analysis_refuses_unrelated_native_drift(tmp_path, extra_failure):
    d = api()
    source, snapshot = incomplete_native_fixture(tmp_path, extra_failure=extra_failure)
    target = tmp_path / "refused"
    with pytest.raises(ValueError, match="outside distance feature"):
        d.derive_stable_analysis(source, target, snapshot)
    assert not (target / "COMPLETE.json").exists()
