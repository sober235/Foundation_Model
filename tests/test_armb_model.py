import pytest

torch = pytest.importorskip("torch")

from anatobind.model.armb import ArmBMinimal, boxes_vox_to_norm

SHAPE = (16, 32, 32)


def _batch(n_boxes=(2, 1)):
    b = len(n_boxes)
    seg = torch.zeros(b, *SHAPE, dtype=torch.long)
    seg[:, :6] = 1
    seg[:, 6:10] = 4
    boxes, classes, hosts = [], [], []
    for n in n_boxes:
        boxes.append(torch.tensor([[1.0, 2, 2, 5, 10, 10]] * n)
                     if n else torch.zeros(0, 6))
        classes.append(torch.zeros(n, dtype=torch.long))
        hosts.append(torch.full((n,), 4, dtype=torch.long))
    return {
        "image": torch.randn(b, 1, *SHAPE),
        "seg": seg,
        "present": torch.stack([(seg[i].unique()[:, None] == torch.arange(1, 7)).any(0)
                                for i in range(b)]),
        "spacing_mm": torch.tensor([[0.8, 0.625, 0.625]] * b),
        "boxes": boxes, "box_classes": classes, "host_label": hosts,
    }


@pytest.fixture(scope="module")
def model():
    torch.manual_seed(0)
    return ArmBMinimal(K=6, M=4, d_model=64, embed_dim=32)


def test_box_normalisation_round_trips_the_centre():
    n = boxes_vox_to_norm(torch.tensor([[0.0, 0, 0, 8, 16, 16]]), SHAPE)
    torch.testing.assert_close(n[0, :3], torch.tensor([0.25, 0.25, 0.25]))
    torch.testing.assert_close(n[0, 3:], torch.tensor([0.5, 0.5, 0.5]))


def test_forward_returns_every_head(model):
    out = model(_batch())
    assert out["masks"].shape == (2, 6, *SHAPE)
    assert out["logits"].shape == (2, 4, 3)
    assert out["host_logits"].shape == (2, 4, 7)
    assert out["R"].shape == (2, 6, 4)
    assert len(out["matches"]) == 2


def test_loss_is_finite_and_reaches_the_backbone(model):
    model.zero_grad()
    batch = _batch()
    loss, parts = model.compute_loss(model(batch), batch)
    assert torch.isfinite(loss)
    loss.backward()
    stem = model.backbone.swin.patch_embed.proj.weight
    assert stem.grad is not None and stem.grad.abs().sum() > 0
    for name in ("mask", "presence", "cls", "l1", "giou", "host_ce", "rel_bce"):
        assert name in parts


def test_relation_head_receives_gradient(model):
    model.zero_grad()
    batch = _batch()
    loss, _ = model.compute_loss(model(batch), batch)
    loss.backward()
    w = model.relation.h_host.weight
    assert w.grad is not None and w.grad.abs().sum() > 0


def test_a_sample_with_no_boxes_gives_a_finite_loss(model):
    batch = _batch(n_boxes=(0, 0))
    loss, _ = model.compute_loss(model(batch), batch)
    assert torch.isfinite(loss)


def test_no_nan_under_bfloat16_autocast(model):
    batch = _batch()
    with torch.autocast("cpu", dtype=torch.bfloat16):
        loss, _ = model.compute_loss(model(batch), batch)
    assert torch.isfinite(loss)


def test_host_accuracy_is_reported_over_matched_queries(model):
    batch = _batch()
    hit, total = model.host_accuracy(model(batch), batch)
    assert total == 3  # 2 + 1 matched queries
    assert 0 <= hit <= total


def test_host_accuracy_skips_instances_with_an_unresolved_host(model):
    batch = _batch()
    batch["host_label"][0][0] = 0  # unresolved side -> no truth to score
    _, total = model.host_accuracy(model(batch), batch)
    assert total == 2


def test_an_unresolved_host_still_gives_a_finite_loss(model):
    batch = _batch(n_boxes=(1, 1))
    for h in batch["host_label"]:
        h[:] = 0
    loss, _ = model.compute_loss(model(batch), batch)
    assert torch.isfinite(loss)
