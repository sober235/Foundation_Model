import pytest

torch = pytest.importorskip("torch")

from anatobind.model.backbone import Backbone
from anatobind.model.decoders import FullResMaskHead, UBDecoder, _QueryStack

CHANNELS = (32, 64, 128, 256)
SHAPES = [(2, 32, 8, 8, 8), (2, 64, 4, 4, 4), (2, 128, 2, 2, 2), (2, 256, 1, 1, 1)]


@pytest.fixture
def feats():
    torch.manual_seed(0)
    return [torch.randn(*s) for s in SHAPES]


def test_the_query_stack_returns_every_layer_and_the_last_is_the_default(feats):
    stack = _QueryStack((128, 64, 32), d_model=64, n_queries=5, layers=3, heads=4).eval()
    mems = [feats[2], feats[1], feats[0]]
    with torch.no_grad():
        every, last = stack(mems, return_all=True), stack(mems)
    assert len(every) == 3 and all(q.shape == (2, 5, 64) for q in every)
    torch.testing.assert_close(every[-1], last)


def test_the_lesion_decoder_emits_auxiliary_heads_for_all_but_the_last_layer(feats):
    dec = UBDecoder(CHANNELS, d_model=64, M=6, layers=3, num_classes=4)
    out = dec(feats, aux=True)
    assert out["logits"].shape == (2, 6, 5) and len(out["aux"]) == 2
    for a in out["aux"]:
        assert a["logits"].shape == (2, 6, 5) and a["boxes"].shape == (2, 6, 6)
    assert "aux" not in dec(feats)


def test_the_full_resolution_head_returns_masks_at_the_input_size():
    head = FullResMaskHead(c1=16, d_model=32, mask_dim=8)
    masks = head(torch.randn(2, 6, 32), torch.randn(2, 16, 4, 8, 8), torch.randn(2, 1, 8, 32, 32))
    assert masks.shape == (2, 6, 8, 32, 32)
    masks.square().mean().backward()
    assert head.up.weight.grad.abs().sum() > 0 and head.img[0].weight.grad.abs().sum() > 0


def test_backbone_checkpointing_changes_nothing_but_memory():
    torch.manual_seed(0)
    plain, ckpt = Backbone(embed_dim=16), Backbone(embed_dim=16, use_checkpoint=True)
    ckpt.load_state_dict(plain.state_dict())
    x = torch.randn(1, 1, 32, 64, 64)
    plain.eval(), ckpt.eval()
    with torch.no_grad():
        for a, b in zip(plain(x), ckpt(x)):
            torch.testing.assert_close(a, b)
    ckpt.train()
    sum(f.square().mean() for f in ckpt(x)).backward()
    assert ckpt.swin.patch_embed.proj.weight.grad.abs().sum() > 0


def test_token_centres_follow_the_box_normalisation():
    from anatobind.model.decoders import token_centres
    c = token_centres((2, 1, 4), "cpu")
    assert c.shape == (8, 3)
    torch.testing.assert_close(c[0], torch.tensor([0.25, 0.5, 0.125]))
    torch.testing.assert_close(c[-1], torch.tensor([0.75, 0.5, 0.875]))


def test_coordinates_are_off_by_default_and_change_the_output_when_on(feats):
    torch.manual_seed(0)
    plain = UBDecoder(CHANNELS, d_model=64, M=6, layers=2, num_classes=4).eval()
    torch.manual_seed(0)
    placed = UBDecoder(CHANNELS, d_model=64, M=6, layers=2, num_classes=4, coords=True).eval()
    assert plain.stack.coord is None and placed.stack.coord is not None
    with torch.no_grad():
        assert not torch.allclose(plain(feats)["boxes"], placed(feats)["boxes"])
