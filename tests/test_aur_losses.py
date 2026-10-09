# tests/test_aur_losses.py
import pytest
import torch

import anatobind.aur.losses as L
from anatobind.aur.labels import N_HOST_CLASSES, NO_HOST


def test_points_are_drawn_from_valid_voxels_only():
    valid = torch.zeros(2, 2, 3, 4)
    valid[0, 0] = 1.0
    valid[1, 1, 2, 3] = 1.0
    pts = L.sample_points(valid, 50, torch.Generator().manual_seed(0))
    assert pts.shape == (2, 50) and (pts[0] < 12).all() and (pts[1] == 23).all()
    x = torch.arange(24.0).view(1, 2, 3, 4)
    assert L.gather(x, torch.tensor([[0, 23, 5]])).tolist() == [[0.0, 23.0, 5.0]]
    inst = torch.zeros(1, 2, 3, 4, dtype=torch.long)
    inst[0, 0, 0, :3] = 1                                       # a 3-voxel lesion
    inst[0, 1, 2, 3] = 2                                        # a 1-voxel lesion
    pts = L.sample_points(torch.ones(1, 2, 3, 4), 40, torch.Generator().manual_seed(0), instance=inst)
    got = L.gather(inst, pts)[0]
    assert pts.shape == (1, 40) and (got[:10] == 1).all() and (got[10:20] == 2).all()     # 20 focus points, 10 per instance
    assert L.sample_points(torch.ones(1, 2, 3, 4), 8, torch.Generator().manual_seed(0), instance=torch.zeros(1, 2, 3, 4, dtype=torch.long)).shape == (1, 8)


def test_entity_loss_prefers_the_right_masks_and_skips_lesion_points():
    pts = torch.tensor([[1, 1, 2, 0, 0, 2]])
    ignore = torch.tensor([[False, False, False, False, False, True]])
    present = torch.zeros(1, 32, dtype=torch.bool)
    present[0, :2] = True
    good = torch.full((1, 32, 6), -8.0)
    good[0, 0, :2] = 8.0
    good[0, 1, 2] = 8.0
    bad = -good
    lg = L.entity_loss(good, torch.full((1, 32), -5.0), pts, ignore, present)
    lb = L.entity_loss(bad, torch.full((1, 32), -5.0), pts, ignore, present)
    assert lg["a_mask"] < 0.05 < lb["a_mask"] and lg["a_presence"] > 0
    leaky = good.clone()
    leaky[0, 5, :] = 8.0                                           # an absent entity claiming every point: penalised
    assert L.entity_loss(leaky, torch.full((1, 32), -5.0), pts, ignore, present)["a_mask"] > lg["a_mask"] + 0.1
    worse = good.clone()
    worse[0, 1, 5] = 8.0                                           # wrong at an ignored (lesion) point: no penalty
    assert torch.allclose(L.entity_loss(worse, torch.full((1, 32), -5.0), pts, ignore, present)["a_mask"], lg["a_mask"])


def test_event_matching_and_loss():
    torch.manual_seed(0)
    target = torch.zeros(2, 6)
    target[0, :3] = 1.0
    target[1, 3:] = 1.0
    logits = torch.full((4, 6), -6.0)
    logits[2, 3:] = 6.0                                             # query 2 fits target 1
    logits[3, :3] = 6.0                                             # query 3 fits target 0
    presence = torch.tensor([-3.0, -3.0, 3.0, 3.0])
    qi, ti = L.match_events(presence, logits, target)
    assert sorted(zip(qi.tolist(), ti.tolist())) == [(2, 1), (3, 0)]
    e, t = L.match_events(presence, logits, torch.zeros(0, 6))
    assert e.numel() == 0 and t.numel() == 0
    out, matches = L.event_loss(presence[None], logits[None], [target], torch.ones(1, 6), torch.tensor([True]))
    assert out["u_mask"] < 0.05 and out["u_presence"] < 0.1 and matches[0][0].tolist() == qi.tolist()
    out2, matches2 = L.event_loss(presence[None], logits[None], [target], torch.ones(1, 6), torch.tensor([False]))
    assert out2["u_mask"] == 0 and out2["u_presence"] == 0 and matches2 == [None]
    out3, _ = L.event_loss(presence[None], torch.full_like(logits, -6.0)[None], [target], torch.ones(1, 6), torch.tensor([True]))
    assert out3["u_mask"] > out["u_mask"]
    weighted, _ = L.event_loss(presence[None], logits[None], [target], torch.tensor([[1.0, 1.0, 1.0, 1.0, 1.0, 0.0]]), torch.tensor([True]))
    assert weighted["u_mask"] < 0.05                                 # a point under the volume floor carries no mask loss
    many = torch.eye(6)                                              # 6 instances, 4 queries: only 4 can be matched
    qi6, ti6 = L.match_events(presence, logits, many)
    assert qi6.numel() == 4 and ti6.numel() == 4 and len(set(ti6.tolist())) == 4
    crowded, m6 = L.event_loss(presence[None], logits[None], [many], torch.ones(1, 6), torch.tensor([True]))
    assert torch.isfinite(crowded["u_mask"]) and m6[0][0].numel() == 4


def test_relation_loss_and_total():
    logits = torch.full((2, N_HOST_CLASSES), -4.0)
    logits[0, 0], logits[1, NO_HOST] = 4.0, 4.0
    host = torch.tensor([0, NO_HOST])
    neg = torch.tensor([[1, 2], [-1, -1]])
    good = L.relation_loss(logits, host, neg)
    assert good["r_host"] < 0.01 and good["r_hard"] == 0.0
    logits[0, 1] = 4.5
    bad = L.relation_loss(logits, host, neg)
    assert bad["r_host"] > good["r_host"] and bad["r_hard"] > 0
    z = L.relation_loss(torch.zeros(0, N_HOST_CLASSES), torch.zeros(0, dtype=torch.long), torch.zeros(0, 2, dtype=torch.long))
    assert z["r_host"] == 0 and z["r_hard"] == 0
    t, logged = L.total({"a_mask": torch.tensor(1.0), "r_hard": torch.tensor(2.0)})
    assert float(t) == pytest.approx(1.0 + L.LAMBDA_R * L.LAMBDA_H * 2.0) and logged == {"a_mask": 1.0, "r_hard": 2.0}
    assert L.seq_loss(torch.tensor([[5.0, 0, 0, 0, 0, 0]]), torch.tensor([0]))["s"] < 0.05


def test_points_include_the_ring_around_the_lesions():
    inst = torch.zeros(1, 6, 6, 6, dtype=torch.long)
    inst[0, 2:4, 2:4, 2:4] = 1
    valid = torch.ones(1, 6, 6, 6)
    n = 100
    pts = L.sample_points(valid, n, torch.Generator().manual_seed(0), instance=inst)
    got = L.gather(inst, pts)[0]
    assert (got[:50] == 1).all()                                         # the focus share
    ring = (torch.nn.functional.max_pool3d((inst > 0).float(), 3, stride=1, padding=1)[0] > 0) & (inst[0] == 0)
    boundary = pts[0, 50:70]
    assert (got[50:70] == 0).all() and ring.flatten()[boundary].all()    # the boundary share: background next to the lesion
    pts2 = L.sample_points(valid, n, torch.Generator().manual_seed(0), instance=inst, boundary_share=0.0)
    assert pts2.shape == (1, n)
    with pytest.raises(ValueError, match="differ in shape"):
        L.sample_points(valid, n, instance=torch.zeros(1, 6, 6, 5, dtype=torch.long))


def test_entity_loss_skips_unsupervised_samples():
    pts = torch.tensor([[1, 1, 2, 0, 0, 2], [2, 2, 1, 0, 1, 0]])
    ignore = torch.zeros(2, 6, dtype=torch.bool)
    present = torch.zeros(2, 32, dtype=torch.bool)
    present[:, :2] = True
    logits = torch.randn(2, 32, 6, generator=torch.Generator().manual_seed(1)).requires_grad_()
    presence = torch.zeros(2, 32, requires_grad=True)
    both = L.entity_loss(logits, presence, pts, ignore, present)
    first = L.entity_loss(logits[:1], presence[:1], pts[:1], ignore[:1], present[:1])
    gated = L.entity_loss(logits, presence, pts, ignore, present, a_supervised=torch.tensor([True, False]))
    assert torch.allclose(gated["a_mask"], first["a_mask"]) and torch.allclose(gated["a_presence"], first["a_presence"])
    assert not torch.allclose(gated["a_mask"], both["a_mask"])
    none = L.entity_loss(logits, presence, pts, ignore, present, a_supervised=torch.tensor([False, False]))
    assert none["a_mask"] == 0 and none["a_presence"] == 0
    (none["a_mask"] + none["a_presence"]).backward()
    assert logits.grad is not None and float(logits.grad.abs().sum()) == 0.0
    with pytest.raises(ValueError, match="a_supervised"):
        L.entity_loss(logits, presence, pts, ignore, present, a_supervised=torch.tensor([True]))


def test_matching_reads_only_the_weighted_points():
    target = torch.tensor([[1.0, 1.0, 1.0, 0.0, 0.0, 0.0]])
    logits = torch.full((2, 6), -6.0)
    logits[0, :2] = 6.0                                                  # query 0 misses point 2
    logits[1, :4] = 6.0                                                  # query 1 adds a false point 3
    presence = torch.tensor([3.0, 3.0])
    qi, _ = L.match_events(presence, logits, target)
    assert qi.tolist() == [1]
    qi_w, _ = L.match_events(presence, logits, target, torch.tensor([1.0, 1.0, 0.0, 1.0, 1.0, 1.0]))
    assert qi_w.tolist() == [0]                                          # point 2 carries no weight: query 0 is exact
