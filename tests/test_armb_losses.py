import inspect

import pytest

torch = pytest.importorskip("torch")

import anatobind.model.losses as losses
from anatobind.model.losses import (
    a_loss, box_corners, build_relation_targets, giou3d, hungarian_match, rel_loss, ub_loss,
)


# --- boxes ------------------------------------------------------------------

def test_giou3d_is_one_for_identical_boxes():
    b = torch.tensor([[0.2, 0.2, 0.2, 0.4, 0.4, 0.4]])
    assert float(giou3d(box_corners(b), box_corners(b))) == pytest.approx(1.0, abs=1e-5)


def test_giou3d_is_negative_for_disjoint_boxes():
    a = box_corners(torch.tensor([[0.1, 0.1, 0.1, 0.1, 0.1, 0.1]]))
    b = box_corners(torch.tensor([[0.9, 0.9, 0.9, 0.1, 0.1, 0.1]]))
    assert float(giou3d(a, b)) < 0.0


def test_giou3d_is_bounded_below_by_minus_one():
    a = box_corners(torch.tensor([[0.01, 0.01, 0.01, 0.02, 0.02, 0.02]]))
    b = box_corners(torch.tensor([[0.99, 0.99, 0.99, 0.02, 0.02, 0.02]]))
    assert float(giou3d(a, b)) >= -1.0


# --- matching ---------------------------------------------------------------

def test_hungarian_recovers_a_known_permutation():
    tgt_boxes = torch.tensor([[0.2, 0.2, 0.2, 0.1, 0.1, 0.1],
                              [0.8, 0.8, 0.8, 0.1, 0.1, 0.1]])
    tgt_cls = torch.tensor([0, 1])
    # query 2 carries target 0 and query 0 carries target 1
    boxes = torch.tensor([[0.8, 0.8, 0.8, 0.1, 0.1, 0.1],
                          [0.5, 0.1, 0.9, 0.4, 0.4, 0.4],
                          [0.2, 0.2, 0.2, 0.1, 0.1, 0.1]])
    logits = torch.tensor([[-5.0, 5.0, -5.0], [0.0, 0.0, 5.0], [5.0, -5.0, -5.0]])
    pi, ti = hungarian_match(logits, boxes, tgt_cls, tgt_boxes)
    got = dict(zip(pi.tolist(), ti.tolist()))
    assert got == {2: 0, 0: 1}


def test_hungarian_on_an_empty_target_set_returns_empty_indices():
    pi, ti = hungarian_match(torch.zeros(4, 3), torch.rand(4, 6),
                             torch.zeros(0, dtype=torch.long), torch.zeros(0, 6))
    assert pi.numel() == 0 and ti.numel() == 0


# --- anatomy loss -----------------------------------------------------------

def _perfect_masks(seg, K):
    tgt = torch.stack([(seg == k + 1).float() for k in range(K)], 1)
    return (tgt * 2 - 1) * 20.0, tgt


def test_a_loss_is_zero_on_a_perfect_prediction():
    seg = torch.zeros(1, 4, 4, 4, dtype=torch.long)
    seg[0, :2] = 1
    seg[0, 2:] = 3
    logits, _ = _perfect_masks(seg, 6)
    present = torch.tensor([[1, 0, 1, 0, 0, 0]], dtype=torch.bool)
    out = a_loss(logits, (present.float() * 2 - 1) * 20.0, seg, present)
    assert float(out["mask"]) < 1e-3
    assert float(out["presence"]) < 1e-3


def test_a_loss_ignores_the_masks_of_absent_structures():
    seg = torch.ones(1, 4, 4, 4, dtype=torch.long)
    present = torch.tensor([[1, 0, 0, 0, 0, 0]], dtype=torch.bool)
    logits, _ = _perfect_masks(seg, 6)
    good = a_loss(logits, (present.float() * 2 - 1) * 20.0, seg, present)["mask"]
    logits[:, 1:] = 20.0  # garbage on every absent structure
    bad = a_loss(logits, (present.float() * 2 - 1) * 20.0, seg, present)["mask"]
    torch.testing.assert_close(good, bad)


# --- event loss -------------------------------------------------------------

def test_ub_loss_is_zero_on_a_perfect_prediction():
    tgt_boxes = torch.tensor([[0.3, 0.3, 0.3, 0.2, 0.2, 0.2]])
    tgt_cls = torch.tensor([1])
    boxes = torch.cat([tgt_boxes, torch.tensor([[0.9, 0.9, 0.9, 0.05, 0.05, 0.05]])])
    logits = torch.tensor([[-20.0, 20.0, -20.0], [-20.0, -20.0, 20.0]])
    pi, ti = hungarian_match(logits, boxes, tgt_cls, tgt_boxes)
    out = ub_loss(logits[None], boxes[None], [tgt_cls], [tgt_boxes], [(pi, ti)])
    assert float(out["cls"]) < 1e-3
    assert float(out["l1"]) < 1e-6
    assert float(out["giou"]) < 1e-5


# --- relation targets: the spec 5.1 guard -----------------------------------

def test_host_ce_target_uses_host_label_not_max_ioa():
    """Relation truth is the annotator's host_label; overlap must never define it.

    Structure 2 is the annotated host but structure 5 is the one the box overlaps
    most.  A target built from overlap would say 4; it must say 1.
    """
    K, M = 6, 3
    matches = [(torch.tensor([0]), torch.tensor([0]))]
    host_labels = [torch.tensor([2])]  # segmentation label 2 -> query index 1
    present = torch.ones(1, K, dtype=torch.bool)
    host_target, rel_target, rel_mask = build_relation_targets(
        matches, host_labels, present, K, M, torch.device("cpu")
    )
    assert host_target[0, 0].item() == 1
    assert float(rel_target[0, 1, 0]) == 1.0
    assert float(rel_target[0, 4, 0]) == 0.0


def test_relation_target_builder_takes_no_geometry():
    """A geometry argument would open the door to overlap-derived truth."""
    params = set(inspect.signature(build_relation_targets).parameters)
    for banned in ("geo", "geometry", "ioa", "masks", "overlap"):
        assert banned not in params
    assert "ioa" not in inspect.getsource(build_relation_targets).lower()


def test_an_unknown_host_is_excluded_from_every_relation_term():
    """host_label 0 means the side could not be resolved: unknown, not a label.

    Inventing a host here (or letting 0 - 1 = -1 wrap onto the last query) would
    train the model on a fabricated binding.
    """
    K, M = 6, 2
    matches = [(torch.tensor([0, 1]), torch.tensor([0, 1]))]
    host_labels = [torch.tensor([0, 3])]  # first instance unresolved
    host_target, rel_target, rel_mask = build_relation_targets(
        matches, host_labels, torch.ones(1, K, dtype=torch.bool), K, M,
        torch.device("cpu")
    )
    assert host_target[0, 0].item() == -100
    assert host_target[0, 1].item() == 2
    assert float(rel_target[0, :, 0].sum()) == 0.0
    assert float(rel_mask[0, :, 0].sum()) == 0.0
    assert float(rel_mask[0, :, 1].sum()) == K


def test_unmatched_queries_are_ignored_by_the_host_loss():
    K, M = 6, 4
    matches = [(torch.tensor([1]), torch.tensor([0]))]
    host_target, _, _ = build_relation_targets(
        matches, [torch.tensor([3])], torch.ones(1, K, dtype=torch.bool), K, M,
        torch.device("cpu")
    )
    assert host_target[0, 1].item() == 2
    assert (host_target[0, [0, 2, 3]] == -100).all()


def test_rel_loss_is_zero_on_a_perfect_prediction():
    K, M = 6, 2
    matches = [(torch.tensor([0]), torch.tensor([0]))]
    present = torch.ones(1, K, dtype=torch.bool)
    host_target, rel_target, rel_mask = build_relation_targets(
        matches, [torch.tensor([4])], present, K, M, torch.device("cpu")
    )
    host_logits = torch.full((1, M, K + 1), -20.0)
    host_logits[0, 0, 3] = 20.0
    R = (rel_target * 2 - 1) * 20.0
    out = rel_loss(host_logits, R, host_target, rel_target, rel_mask)
    assert float(out["host_ce"]) < 1e-3
    assert float(out["rel_bce"]) < 1e-3


def test_rel_loss_survives_a_batch_with_no_matches():
    K, M = 6, 2
    host_target, rel_target, rel_mask = build_relation_targets(
        [(torch.zeros(0, dtype=torch.long), torch.zeros(0, dtype=torch.long))],
        [torch.zeros(0, dtype=torch.long)], torch.ones(1, K, dtype=torch.bool),
        K, M, torch.device("cpu")
    )
    out = rel_loss(torch.randn(1, M, K + 1), torch.randn(1, K, M),
                   host_target, rel_target, rel_mask)
    assert torch.isfinite(out["host_ce"]) and torch.isfinite(out["rel_bce"])


def test_total_loss_weights_are_the_documented_ones():
    src = inspect.getsource(losses.total_loss)
    assert "5.0" in src and "2.0" in src  # DETR box weights
    parts = {k: torch.tensor(1.0) for k in
             ("mask", "presence", "cls", "l1", "giou", "host_ce", "rel_bce")}
    total, _ = losses.total_loss(parts)
    assert float(total) == pytest.approx(1 + 1 + 1 + 5 + 2 + 1 + 1)
