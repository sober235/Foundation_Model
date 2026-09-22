import pytest

torch = pytest.importorskip("torch")

from anatobind.model.relation import IndependentCandidateHead


@pytest.fixture
def parts():
    torch.manual_seed(7)
    B, K, M, d, G = 2, 5, 3, 32, 7
    return {
        "a": torch.randn(B, K, d),
        "u": torch.randn(B, M, d),
        "geo": torch.randn(B, K, M, G),
        "present": torch.tensor(
            [[1, 1, 1, 0, 0], [1, 0, 1, 1, 0]], dtype=torch.bool
        ),
        "B": B, "K": K, "M": M, "d": d, "G": G,
    }


def _head(p, **kwargs):
    return IndependentCandidateHead(
        d_model=p["d"], geometry_channels=p["G"], geo_dim=16, hidden_dim=24, **kwargs
    )


def test_host_competition_output_shapes(parts):
    out = _head(parts)(parts["a"], parts["u"], parts["geo"], parts["present"])
    assert out["pair_repr"].shape == (parts["B"], parts["K"], parts["M"], parts["d"])
    assert out["host_logits"].shape == (parts["B"], parts["M"], parts["K"] + 1)


def test_absent_anatomy_has_zero_probability(parts):
    model = _head(parts).eval()
    with torch.no_grad():
        probs = model(parts["a"], parts["u"], parts["geo"], parts["present"])["host_logits"].softmax(-1)
    for b in range(parts["B"]):
        absent = ~parts["present"][b]
        assert float(probs[b, :, :parts["K"]][:, absent].abs().max()) == 0.0
    torch.testing.assert_close(probs.sum(-1), torch.ones(parts["B"], parts["M"]))


def test_none_is_certain_if_no_anatomy_candidate_is_present(parts):
    model = _head(parts).eval()
    present = torch.zeros_like(parts["present"])
    with torch.no_grad():
        probs = model(parts["a"], parts["u"], parts["geo"], present)["host_logits"].softmax(-1)
    assert torch.isfinite(probs).all()
    torch.testing.assert_close(probs[..., -1], torch.ones(parts["B"], parts["M"]))


def test_event_permutation_only_permutes_event_outputs(parts):
    model = _head(parts).eval()
    perm = torch.tensor([2, 0, 1])
    with torch.no_grad():
        base = model(parts["a"], parts["u"], parts["geo"], parts["present"])["host_logits"]
        swapped = model(
            parts["a"], parts["u"][:, perm], parts["geo"][:, :, perm], parts["present"]
        )["host_logits"]
    torch.testing.assert_close(swapped, base[:, perm])


def test_one_lesion_cannot_change_another_lesions_binding(parts):
    """Pins the minimal-head semantics: no cross-lesion attention/mixing."""
    model = _head(parts).eval()
    changed_u = parts["u"].clone()
    changed_geo = parts["geo"].clone()
    changed_u[:, 0] += 20.0
    changed_geo[:, :, 0] -= 10.0
    with torch.no_grad():
        base = model(parts["a"], parts["u"], parts["geo"], parts["present"])["host_logits"]
        changed = model(parts["a"], changed_u, changed_geo, parts["present"])["host_logits"]
    torch.testing.assert_close(changed[:, 1:], base[:, 1:])


def test_local_evidence_is_optional_but_shape_checked(parts):
    model = _head(parts, local_dim=4)
    local = torch.randn(parts["B"], parts["K"], parts["M"], 4)
    out = model(parts["a"], parts["u"], parts["geo"], parts["present"], local)
    assert torch.isfinite(out["pair_repr"]).all()
    with pytest.raises(ValueError):
        model(parts["a"], parts["u"], parts["geo"], parts["present"])


def test_gradients_reach_anatomy_lesion_geometry_and_local_evidence(parts):
    model = _head(parts, local_dim=4)
    a = parts["a"].clone().requires_grad_(True)
    u = parts["u"].clone().requires_grad_(True)
    geo = parts["geo"].clone().requires_grad_(True)
    local = torch.randn(parts["B"], parts["K"], parts["M"], 4, requires_grad=True)
    logits = model(a, u, geo, parts["present"], local)["host_logits"]
    finite = torch.isfinite(logits)
    loss = logits[finite].square().mean()
    loss.backward()
    for x in (a, u, geo, local):
        assert x.grad is not None and x.grad.abs().sum() > 0
