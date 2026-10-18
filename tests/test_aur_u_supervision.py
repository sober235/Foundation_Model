"""Detached probes must preserve the official loss, assignment and gradient semantics."""
import numpy as np
import pytest
import torch

from anatobind.aur import losses
from anatobind.aur.dataset import event_targets_at_points
from anatobind.aur.u_supervision_diagnostics import compare_draw_assignments, probe_event_loss


def fixture():
    presence = torch.tensor([3., .5, -3., -3.], requires_grad=True)
    logits = torch.full((4, 6), -6.)
    logits[0, :3] = 6.
    logits[1, 3:] = 6.
    return presence, logits.requires_grad_(), torch.tensor([1, 1, 1, 0, 0, 0]), torch.ones(6), torch.arange(6), 1


def direct(args, supervised=True, profile="unmatched_empty_v1"):
    p, m, ids, w, _, n = args
    p = p.detach().float().clone().requires_grad_()
    m = m.detach().float().clone().requires_grad_()
    out, matches = losses.event_loss(p[None], m[None], event_targets_at_points(ids[None], [n]),
                                     w[None], torch.tensor([supervised]), u_mask_profile=profile)
    total = out["u_mask"] + out["u_presence"]
    if total.requires_grad:
        total.backward()
    return out, matches, np.zeros(m.shape, np.float32) if m.grad is None else m.grad.numpy(), np.zeros(p.shape, np.float32) if p.grad is None else p.grad.numpy()


@pytest.mark.parametrize("profile", ["legacy", "unmatched_empty_v1"])
def test_probe_preserves_official_loss_matches_and_all_gradients(profile):
    args = fixture()
    result = probe_event_loss(*args, profile=profile)
    out, matches, mg, pg = direct(args, profile=profile)
    assert result["matches"] == {"query_indices": matches[0][0].tolist(), "target_indices": matches[0][1].tolist()}
    assert result["query_target_instance"] == [1, None, None, None]
    for key in out:
        assert result["losses"][key] == float(out[key])
        assert result["reconstructed_losses"][key] == pytest.approx(float(out[key]), abs=1e-6, rel=1e-5)
    np.testing.assert_array_equal(result["mask_logit_gradient"], mg)
    np.testing.assert_array_equal(result["presence_logit_gradient"], pg)
    assert result["mask_logit_gradient"].dtype == np.float32
    assert result["presence_logit_gradient"].dtype == np.float32
    assert args[0].grad is None and args[1].grad is None
    assert result["normalization"]["matched_queries"] == 1
    assert result["normalization"]["unmatched_queries"] == 3
    assert result["normalization"]["presence_weight_sum"] == pytest.approx(1.3)
    assert sum(q["presence_contribution"] for q in result["queries"]) == pytest.approx(result["losses"]["u_presence"])
    assert sum(q["mask_contribution"] for q in result["queries"]) == pytest.approx(result["losses"]["u_mask"], abs=1e-6)
    assert result["queries"][0]["presence_target"] == 1.
    assert result["queries"][1]["presence_weight"] == .1
    if profile == "legacy":
        assert np.count_nonzero(mg[1:]) == 0
    else:
        assert mg[1, 3] > 0  # An unmatched query's painted negative voxel is suppressed.
        assert pg[1] > 0 and pg[0] < 0


def test_unsupervised_targets_are_undefined_and_gradients_explicit_zero():
    result = probe_event_loss(*fixture(), u_supervised=False)
    assert result["u_supervised"] is False
    assert result["query_target_instance"] == [None] * 4
    assert result["losses"] == {"u_mask": 0., "u_presence": 0.}
    assert not result["mask_logit_gradient"].any() and not result["presence_logit_gradient"].any()
    for query in result["queries"]:
        for key in ("presence_target", "presence_weight", "mask_bce", "mask_dice", "target_positive_points"):
            assert query[key] is None


@pytest.mark.parametrize("n,ids,weight", [(0, [0]*6, [1]*6), (2, [0]*6, [1]*6), (1, [1,1,1,0,0,0], [0]*6), (6, [1,2,3,4,5,6], [1]*6)])
def test_empty_missed_zero_weight_and_capacity_cases_match_official(n, ids, weight):
    args = list(fixture())
    args[2], args[3], args[5] = torch.tensor(ids), torch.tensor(weight).float(), n
    result = probe_event_loss(*args)
    out, matches, mg, pg = direct(args)
    assert len(result["matches"]["query_indices"]) == min(n, 4)
    np.testing.assert_array_equal(result["mask_logit_gradient"], mg)
    np.testing.assert_array_equal(result["presence_logit_gradient"], pg)
    for key in out:
        assert result["losses"][key] == float(out[key])
        assert result["reconstructed_losses"][key] == pytest.approx(float(out[key]), abs=1e-6, rel=1e-5)
    if not any(weight):
        assert not result["mask_logit_gradient"].any()
    if n == 2:
        assert result["target_point_counts"] == [0, 0]
        assert result["query_target_instance"].count(None) == 2  # Dense targets still get matched.


def test_duplicate_indices_keep_original_gradient_order_and_count_unique_coverage():
    args = list(fixture())
    args[4] = torch.tensor([8, 8, 4, 2, 2, 9])
    args[3] = torch.tensor([1., 1., 0., 1., 1., 0.])
    result = probe_event_loss(*args)
    _, _, mg, _ = direct(args)
    np.testing.assert_array_equal(result["mask_logit_gradient"], mg)
    q = result["queries"][0]
    assert (q["target_positive_points"], q["target_positive_unique"], q["target_positive_effective"], q["target_positive_effective_unique"]) == (3, 2, 2, 1)
    assert (result["normalization"]["effective_points"], result["normalization"]["unique_effective_points"]) == (4, 2)
    assert not mg[:, [2, 5]].any()


def test_matching_stability_counts_target_not_query_status_and_capacity():
    # Targets 1,2,3,4,5: same query, changed query, first only, second only, neither.
    a = {"n_queries": 3, "n_instances": 5, "u_supervised": True, "query_target_instance": [1, 2, 3], "target_point_counts": [1, 0, 1, 0, 0], "target_effective_point_counts": [1, 0, 1, 0, 0]}
    b = dict(a, query_target_instance=[1, 4, 2], target_point_counts=[1, 1, 0, 1, 0], target_effective_point_counts=[1, 1, 0, 1, 0])
    result = compare_draw_assignments(a, b, 5)
    assert {k: result[k] for k in ("n_instances", "both_matched", "same_query", "different_query", "first_only", "second_only", "neither")} == {"n_instances": 5, "both_matched": 2, "same_query": 1, "different_query": 1, "first_only": 1, "second_only": 1, "neither": 1}
    assert result["capacity_overflow"] == 2
    assert result["first_zero_positive_targets"] == 3
    assert result["second_zero_positive_targets"] == 2
    empty = probe_event_loss(torch.zeros(2), torch.zeros(2, 1), torch.zeros(1, dtype=torch.long), torch.ones(1), torch.zeros(1, dtype=torch.long), 0)
    assert compare_draw_assignments(empty, empty, 0)["neither"] == 0
    off = probe_event_loss(*fixture(), u_supervised=False)
    assert compare_draw_assignments(off, off, 1)["both_matched"] is None


@pytest.mark.parametrize("position,bad", [(0, [float("nan")]*4), (1, torch.zeros(4,5)), (2, [1,1,1,0,0,2]), (2, [1.5,1,1,0,0,0]), (3, [.5]*6), (4, [-1,0,1,2,3,4]), (4, [.5]*6), (5, -1), (5, 1.5)])
def test_invalid_numeric_contracts_fail_without_echoing_inputs(position, bad):
    args = list(fixture())
    args[position] = bad
    with pytest.raises(ValueError):
        probe_event_loss(*args)


def test_probe_accepts_numpy_and_detaches_under_no_grad():
    args = [v.detach().numpy() if isinstance(v, torch.Tensor) else v for v in fixture()]
    with torch.no_grad():
        result = probe_event_loss(*args)
    assert result["mask_logit_gradient"][1, 3] > 0


def test_probe_reenables_local_autograd_inside_frozen_inference_context():
    # Frozen-forward callers may use inference_mode, stronger than no_grad.
    with torch.inference_mode():
        args = fixture()
        result = probe_event_loss(*args)
    _, _, mg, pg = direct(fixture())
    np.testing.assert_array_equal(result["mask_logit_gradient"], mg)
    np.testing.assert_array_equal(result["presence_logit_gradient"], pg)


@pytest.mark.parametrize("profile", ["legacy", "unmatched_empty_v1"])
def test_multiple_matches_and_missed_targets_reconstruct_global_matched_mean(profile):
    generator = torch.Generator().manual_seed(77)
    args = (torch.randn(5, generator=generator), torch.randn(5, 9, generator=generator),
            torch.tensor([1,1,2,2,0,0,0,0,0]), torch.tensor([1.,0.,1.,1.,1.,0.,1.,1.,1.]),
            torch.tensor([8,2,3,3,5,6,7,9,1]), 3)
    result = probe_event_loss(*args, profile=profile)
    out, _, mg, pg = direct(args, profile=profile)
    assert result["normalization"]["matched_query_denominator"] == 3
    assert result["normalization"]["unmatched_query_denominator"] == 2
    assert result["target_point_counts"] == [2,2,0]
    assert result["target_effective_point_counts"] == [1,2,0]
    np.testing.assert_array_equal(result["mask_logit_gradient"], mg)
    np.testing.assert_array_equal(result["presence_logit_gradient"], pg)
    for key in out:
        assert result["reconstructed_losses"][key] == pytest.approx(float(out[key]), abs=1e-6, rel=1e-5)
