# tests/test_aur_infer.py
"""Whole-volume inference of AnatoBind-Brain (SSL-first plan T12; spec §8): sliding windows, Gaussian weights, the
assembled entity / lesion maps and the per-instance binding."""
import numpy as np
import pytest
import torch

import anatobind.aur.infer as I
from anatobind.aur.crops import coordinates_mm, extract, local_coordinates, normalise, to_zyx
from anatobind.aur.labels import N_ENTITIES, N_HOST_CLASSES
from anatobind.aur.model import AnatoBindBrain

TINY = dict(embed=32, depths=(1, 1, 1, 1), heads=(1, 1, 1, 1), window=(2, 4, 4), patch=(2, 4, 4), d_model=16, n_events=4,
            mask_dim=4, pixel_dim=8, dec_layers=1, dec_heads=2, rel_layers=1, rel_heads=2, use_checkpoint=False)


def _volume(shape=(12, 36, 40), seed=0):
    """A normalised (z, y, x) volume with a bright blob and the diagonal affine of (2, 1, 1) mm (zyx)."""
    rng = np.random.default_rng(seed)
    img = rng.normal(0, 5, shape).astype(np.float32)
    img[2:10, 6:30, 8:32] += 300.0
    affine = np.diag([1.0, 1.0, 2.0, 1.0])
    return normalise(img), affine


def test_windows_cover_the_volume_with_the_crop_size_and_end_aligned():
    ws = I.windows((10, 40, 33), (8, 16, 16), overlap=0.5)
    assert all(all(b - a == c for (a, b), c in zip(w, (8, 16, 16))) for w in ws)
    covered = np.zeros((10, 40, 33), bool)
    for w in ws:
        covered[tuple(slice(max(a, 0), min(b, n)) for (a, b), n in zip(w, covered.shape))] = True
    assert covered.all()
    starts = sorted({w[1][0] for w in ws})
    assert starts == [0, 8, 16, 24] and sorted({w[2][0] for w in ws}) == [0, 8, 16, 17]       # the last window ends at the edge
    assert sorted({w[0][0] for w in ws}) == [0, 2]
    single = I.windows((6, 16, 16), (8, 16, 16), overlap=0.5)
    assert single == [[(-1, 7), (0, 16), (0, 16)]]                                            # a volume thinner than the crop: one centred window
    with pytest.raises(ValueError, match="overlap"):
        I.windows((10, 40, 33), (8, 16, 16), overlap=1.0)


def test_gaussian_weight_peaks_at_the_centre_and_stays_positive():
    w = I.gaussian_weight((8, 16, 16))
    assert w.shape == (8, 16, 16) and w.dtype == np.float32 and w.max() == pytest.approx(1.0)
    assert np.unravel_index(w.argmax(), w.shape) == (4, 8, 8) and w.min() > 0.0 and w[0, 0, 0] < w[4, 8, 8]


def test_predict_volume_assembles_the_maps_and_is_deterministic():
    torch.manual_seed(0)
    model = AnatoBindBrain(**TINY).eval()
    image, affine = _volume()
    out = I.predict_volume(model, image, affine, crop=(8, 16, 16), device=torch.device("cpu"), batch_size=3)
    assert out["entity"].shape == image.shape and out["entity"].dtype == np.uint8 and out["entity"].max() <= N_ENTITIES
    assert out["lesion_prob"].shape == image.shape and 0.0 <= out["lesion_prob"].min() and out["lesion_prob"].max() <= 1.0
    assert out["entity_max_prob"].shape == image.shape and out["entity_max_prob"].dtype == np.float16
    assert out["seq_probs"].shape == (6,) and out["seq_probs"].sum() == pytest.approx(1.0, abs=1e-3)
    assert out["entity_presence"].shape == (N_ENTITIES,) and out["n_windows"] == len(I.windows(image.shape, (8, 16, 16)))
    again = I.predict_volume(model, image, affine, crop=(8, 16, 16), device=torch.device("cpu"), batch_size=1)
    assert np.array_equal(out["entity"], again["entity"]) and np.allclose(out["lesion_prob"], again["lesion_prob"], atol=1e-5)
    big = I.predict_volume(model, image, affine, crop=(16, 48, 48), device=torch.device("cpu"))    # a crop larger than the volume
    assert big["n_windows"] == 1 and big["entity"].shape == image.shape


def test_predict_volume_equals_the_direct_forward_when_one_window_covers_the_volume():
    torch.manual_seed(1)
    model = AnatoBindBrain(**TINY).eval()
    image, affine = _volume(shape=(8, 32, 32))
    out = I.predict_volume(model, image, affine, crop=(8, 32, 32), device=torch.device("cpu"))
    window = [(0, 8), (0, 32), (0, 32)]
    img, valid = extract(image, window, fill=-1.0)
    coords, local = coordinates_mm(window, affine=affine), local_coordinates(window, image.shape)
    with torch.no_grad():
        o = model(torch.from_numpy(img)[None, None], torch.from_numpy(valid.astype(np.float32))[None],
                  torch.from_numpy(coords.astype(np.float32))[None], torch.from_numpy(local.astype(np.float32))[None])
        ent = model.entity_masks(o).sigmoid()[0].numpy()
        pres = o["event_presence"].sigmoid()[0].numpy()
        ev = model.event_masks(o).sigmoid()[0].numpy()
    expected = np.where(ent.max(0) >= 0.5, ent.argmax(0) + 1, 0).astype(np.uint8)
    assert np.array_equal(out["entity"], expected)
    kept = ev[pres > 0.5]
    lesion = kept.max(0) if len(kept) else np.zeros(image.shape, np.float32)
    assert np.allclose(out["lesion_prob"], lesion, atol=1e-3)


def test_instances_from_the_lesion_map_drop_components_under_the_floor():
    prob = np.zeros((10, 20, 20), np.float32)
    prob[2:6, 2:6, 2:6] = 0.9                      # 64 voxels
    prob[8, 15, 15] = 0.8                          # a single voxel
    inst, rows = I.instances_from_probability(prob, voxel_mm3=1.0, threshold=0.5)
    assert inst.shape == prob.shape and inst.max() == 1 and rows[0]["volume_mm3"] == 64.0 and rows[0]["score"] == pytest.approx(0.9)
    assert rows[0]["box"] == [2, 2, 2, 6, 6, 6] and inst[8, 15, 15] == 0
    empty, none = I.instances_from_probability(np.zeros((4, 4, 4), np.float32), voxel_mm3=1.0)
    assert empty.max() == 0 and none == []


def test_bind_instances_picks_the_query_with_the_largest_overlap_and_returns_host_probabilities():
    torch.manual_seed(2)
    model = AnatoBindBrain(**TINY).eval()
    image, affine = _volume()
    inst = np.zeros(image.shape, np.int32)
    inst[3:6, 10:14, 12:16] = 1
    inst[7:9, 20:24, 26:30] = 2
    rows = I.bind_instances(model, image, affine, inst, crop=(8, 16, 16), device=torch.device("cpu"))
    assert [r["instance"] for r in rows] == [1, 2]
    for r in rows:
        assert r["host_probs"].shape == (N_HOST_CLASSES,) and r["host_probs"].sum() == pytest.approx(1.0, abs=1e-4)
        assert 0 <= r["host"] < N_HOST_CLASSES and r["host"] == int(np.argmax(r["host_probs"]))
        assert 0.0 <= r["query_iou"] <= 1.0 and 0 <= r["query"] < TINY["n_events"]
        assert all(a <= b for a, b in r["window"]) and r["in_crop_share"] == pytest.approx(1.0)
    assert I.bind_instances(model, image, affine, np.zeros(image.shape, np.int32), crop=(8, 16, 16), device=torch.device("cpu")) == []


class _Constant(torch.nn.Module):
    """A stand-in model: constant entity logits (entity 3 everywhere), one present event with a constant mask,
    constant sequence logits; lets the blending be checked exactly."""
    def __init__(self, n_events=2):
        super().__init__()
        self.n_events = n_events
        self.p = torch.nn.Parameter(torch.zeros(1))

    def forward(self, image, valid, coords, local):
        return {"image": image, "valid": valid}

    def entity_masks(self, out, points=None):
        B, _, D, H, W = out["image"].shape
        logits = torch.full((B, N_ENTITIES, D, H, W), -3.0)
        logits[:, 2] = 2.0
        return logits

    def event_masks(self, out, points=None):
        B, _, D, H, W = out["image"].shape
        logits = torch.full((B, self.n_events, D, H, W), -4.0)
        logits[:, 0] = 1.0
        return logits

    def __call__(self, image, valid, coords, local):
        out = self.forward(image, valid, coords, local)
        B = image.shape[0]
        out["event_presence"] = torch.tensor([[3.0, -3.0]]).expand(B, -1)
        out["seq_logits"] = torch.tensor([[0.0, 0.0, 0.0, 5.0, 0.0, 0.0]]).expand(B, -1)
        out["entity_presence"] = torch.zeros(B, N_ENTITIES)
        return out

    def eval(self):
        return self


def test_blending_of_a_constant_model_gives_the_constant_everywhere():
    image, affine = _volume(shape=(10, 36, 40))
    out = I.predict_volume(_Constant(), image, affine, crop=(8, 16, 16), device=torch.device("cpu"), batch_size=2)
    assert out["n_windows"] > 1 and np.all(out["entity"] == 3) and np.allclose(out["lesion_prob"], 1 / (1 + np.exp(-1.0)), atol=1e-5)
    assert np.allclose(out["entity_max_prob"].astype(np.float32), 1 / (1 + np.exp(-2.0)), atol=2e-3) and out["seq_probs"].argmax() == 3


def test_bind_instances_picks_the_query_with_the_largest_overlap(monkeypatch):
    torch.manual_seed(3)
    model = AnatoBindBrain(**TINY).eval()
    image, affine = _volume()
    inst = np.zeros(image.shape, np.int32)
    inst[3:6, 10:14, 12:16] = 1
    def crafted(out, points=None):
        B, _, D, H, W = out["levels"][0]["feat"].shape[0], None, 8, 16, 16
        logits = torch.full((B, TINY["n_events"], D, H, W), -5.0)
        logits[:, 2, 3:6, 6:10, 6:10] = 5.0          # query 2 covers the instance inside the centred crop (window z 0:8, y 4:20, x 6:22)
        logits[:, 1, 0:1, 0:1, 0:1] = 5.0            # query 1 lies elsewhere
        return logits
    monkeypatch.setattr(model, "event_masks", crafted)
    rows = I.bind_instances(model, image, affine, inst, crop=(8, 16, 16), device=torch.device("cpu"))
    assert rows[0]["query"] == 2 and rows[0]["query_iou"] > 0.5 and rows[0]["zero_overlap"] is False
    def nowhere(out, points=None):
        B = out["levels"][0]["feat"].shape[0]
        logits = torch.full((B, TINY["n_events"], 8, 16, 16), -5.0)
        logits[:, 3, 3:6, 6:10, 6:10] = -1.0         # no query above 0.5; query 3 has the largest soft overlap
        return logits
    monkeypatch.setattr(model, "event_masks", nowhere)
    rows = I.bind_instances(model, image, affine, inst, crop=(8, 16, 16), device=torch.device("cpu"))
    assert rows[0]["zero_overlap"] is True and rows[0]["query"] == 3 and rows[0]["query_iou"] == 0.0


@pytest.mark.skipif(not torch.cuda.is_available(), reason="the bf16 autocast path is the CUDA one (CPU autocast lacks some ops)")
def test_the_autocast_path_runs_the_heads_inside_the_context_on_cuda():
    """The review of 2026-10-09 found the heads running outside the autocast on CUDA (dtype error at the first
    window); verified by hand on a card the same day, this test keeps it (it skips on a CPU-only run)."""
    device = torch.device("cuda", 0)
    torch.manual_seed(0)
    model = AnatoBindBrain(**TINY).to(device).eval()
    image, affine = _volume()
    out = I.predict_volume(model, image, affine, crop=(8, 16, 16), device=device, batch_size=2)
    assert out["entity"].shape == image.shape
    inst = np.zeros(image.shape, np.int32)
    inst[3:6, 10:14, 12:16] = 1
    rows = I.bind_instances(model, image, affine, inst, crop=(8, 16, 16), device=device)
    assert len(rows) == 1 and rows[0]["host_probs"].sum() == pytest.approx(1.0, abs=1e-3)
