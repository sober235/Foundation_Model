import pytest

torch = pytest.importorskip("torch")

from anatobind.model.relation import RelationModule, geometry_features

SPACING = torch.tensor([[0.8, 0.625, 0.625]])  # (z, y, x) mm


def _mask_at(shape, voxels, K=2):
    m = torch.zeros(1, K, *shape)
    for k, (z, y, x) in enumerate(voxels):
        m[0, k, z, y, x] = 1.0
    return m


def _box(z0, y0, x0, z1, y1, x1):
    return torch.tensor([[[z0, y0, x0, z1, y1, x1]]], dtype=torch.float32)


# --- geometry ---------------------------------------------------------------

def test_delta_is_millimetres_and_respects_anisotropic_spacing():
    shape = (24, 24, 24)
    box = _box(4, 4, 4, 6, 6, 6)  # centre (5, 5, 5)
    masks = _mask_at(shape, [(15, 5, 5), (5, 5, 15)])
    g = geometry_features(masks, box, SPACING)
    # structure 0 is 10 voxels away along z (0.8 mm) -> 8.0 mm
    torch.testing.assert_close(g[0, 0, 0, :3], torch.tensor([8.0, 0.0, 0.0]))
    # structure 1 is 10 voxels away along x (0.625 mm) -> 6.25 mm
    torch.testing.assert_close(g[0, 1, 0, :3], torch.tensor([0.0, 0.0, 6.25]))


def test_distance_channel_is_the_norm_of_the_delta():
    masks = _mask_at((24, 24, 24), [(15, 5, 5), (5, 5, 15)])
    g = geometry_features(masks, _box(4, 4, 4, 6, 6, 6), SPACING)
    torch.testing.assert_close(g[..., 3], g[..., :3].norm(dim=-1))


def test_ioa_is_intersection_over_box_volume_not_over_union():
    shape = (16, 16, 16)
    masks = torch.zeros(1, 2, *shape)
    masks[0, 0] = 1.0                       # fills the whole volume
    masks[0, 1, 2:4, 2:4, 2:4] = 1.0        # exactly the box
    box = _box(2, 2, 2, 4, 4, 4)
    g = geometry_features(masks, box, SPACING)
    # a box fully inside a mask has IoA 1 regardless of how large the mask is
    assert float(g[0, 0, 0, 4]) == pytest.approx(1.0)
    assert float(g[0, 1, 0, 4]) == pytest.approx(1.0)


def test_ioa_is_zero_for_a_disjoint_structure():
    masks = torch.zeros(1, 1, 16, 16, 16)
    masks[0, 0, 10:12, 10:12, 10:12] = 1.0
    g = geometry_features(masks, _box(2, 2, 2, 4, 4, 4), SPACING)
    assert float(g[0, 0, 0, 4]) == 0.0


def test_geometry_is_detached():
    masks = _mask_at((16, 16, 16), [(8, 8, 8)], K=1).requires_grad_(True)
    g = geometry_features(masks, _box(2, 2, 2, 4, 4, 4), SPACING)
    assert not g.requires_grad


# --- relation module --------------------------------------------------------

@pytest.fixture
def parts():
    torch.manual_seed(0)
    B, K, M, d = 2, 6, 4, 32
    return dict(
        a=torch.randn(B, K, d), u=torch.randn(B, M, d),
        geo=torch.randn(B, K, M, 5),
        present=torch.tensor([[1, 1, 1, 0, 0, 0], [1, 0, 1, 0, 1, 0]], dtype=torch.bool),
        d=d, K=K, M=M,
    )


def test_shapes(parts):
    mod = RelationModule(d_model=parts["d"], K=parts["K"], layers=2, heads=4)
    out = mod(parts["a"], parts["u"], parts["geo"], parts["present"])
    assert out["R"].shape == (2, parts["K"], parts["M"])
    assert out["host_logits"].shape == (2, parts["M"], parts["K"] + 1)


def test_absent_structures_get_zero_host_probability(parts):
    mod = RelationModule(d_model=parts["d"], K=parts["K"], layers=2, heads=4).eval()
    with torch.no_grad():
        out = mod(parts["a"], parts["u"], parts["geo"], parts["present"])
    probs = out["host_logits"].softmax(-1)
    for b in range(2):
        absent = ~parts["present"][b]
        assert float(probs[b][:, :parts["K"]][:, absent].abs().max()) == 0.0
    torch.testing.assert_close(probs.sum(-1), torch.ones(2, parts["M"]))


def test_none_column_survives_when_every_structure_is_absent(parts):
    mod = RelationModule(d_model=parts["d"], K=parts["K"], layers=2, heads=4).eval()
    present = torch.zeros(2, parts["K"], dtype=torch.bool)
    with torch.no_grad():
        out = mod(parts["a"], parts["u"], parts["geo"], present)
    probs = out["host_logits"].softmax(-1)
    assert torch.isfinite(probs).all()
    torch.testing.assert_close(probs[..., -1], torch.ones(2, parts["M"]))


def test_permuting_event_queries_permutes_the_host_logits(parts):
    mod = RelationModule(d_model=parts["d"], K=parts["K"], layers=2, heads=4).eval()
    perm = torch.tensor([2, 0, 3, 1])
    with torch.no_grad():
        base = mod(parts["a"], parts["u"], parts["geo"], parts["present"])["host_logits"]
        swapped = mod(parts["a"], parts["u"][:, perm], parts["geo"][:, :, perm],
                      parts["present"])["host_logits"]
    torch.testing.assert_close(swapped, base[:, perm], atol=1e-5, rtol=1e-4)


def test_gradients_flow_to_both_embeddings(parts):
    mod = RelationModule(d_model=parts["d"], K=parts["K"], layers=2, heads=4)
    a = parts["a"].clone().requires_grad_(True)
    u = parts["u"].clone().requires_grad_(True)
    mod(a, u, parts["geo"], parts["present"])["host_logits"].square().mean().backward()
    assert a.grad is not None and a.grad.abs().sum() > 0
    assert u.grad is not None and u.grad.abs().sum() > 0
