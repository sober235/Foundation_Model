"""Synthetic checks for the native-anchored shared-head experiment.

These tests catch loss renormalization, rematching the primary arm, unanchored
values, incorrect shared derivatives, and mutation of the real frozen head.
"""
import importlib
import importlib.util
import json
from unittest.mock import patch

import numpy as np
import pytest
import torch
import torch.nn.functional as F

from anatobind.aur import losses


def tangent():
    assert importlib.util.find_spec("scripts.aur_u_background_tangent") is not None, "tangent implementation is required"
    return importlib.import_module("scripts.aur_u_background_tangent")


def grad(value, inputs):
    if not value.requires_grad:
        return tuple(torch.zeros_like(v) for v in inputs)
    got = torch.autograd.grad(value, inputs, allow_unused=True)
    return tuple(torch.zeros_like(v) if g is None else g for v, g in zip(inputs, got))


@pytest.mark.parametrize("n,supervised,weight", [
    (0, True, [1, 1, 1, 1, 1, 1]),
    (1, True, [1, 1, 0, 1, 1, 0]),
    (5, True, [1, 1, 1, 1, 1, 1]),
    (1, True, [0, 0, 0, 0, 0, 0]),
    (1, False, [1, 1, 1, 1, 1, 1]),
])
@pytest.mark.parametrize("profile", ["legacy", "unmatched_empty_v1"])
def test_fixed_loss_matches_official_scalar_and_logit_gradients(n, supervised, weight, profile):
    # The repeated columns model duplicate sampled voxel occurrences. Dropping or
    # deduplicating them, or averaging matched and unmatched queries together, fails.
    p = torch.tensor([2., -.3, -2.], requires_grad=True)
    z = torch.tensor([[3., 3., -.4, -2., -2., 1.],
                      [-2., -2., 3., .2, .2, -1.],
                      [.5, .5, -.7, 2., 2., -2.]], requires_grad=True)
    ids = torch.tensor([1, 1, 2, 0, 0, 3])
    target = (ids[None] == torch.arange(1, n + 1)[:, None]).float()
    w = torch.tensor(weight).float()
    official, matches = losses.event_loss(p[None], z[None], [target], w[None],
                                           torch.tensor([supervised]), u_mask_profile=profile)
    fixed = tangent().fixed_event_loss(p, z, target, w, matches[0],
                                      u_supervised=supervised, profile=profile)
    for key in ("u_presence", "u_mask"):
        torch.testing.assert_close(fixed[key], official[key], rtol=0, atol=0)
    expected = grad(official["u_presence"] + official["u_mask"], (p, z))
    actual = grad(fixed["u_presence"] + fixed["u_mask"], (p, z))
    for a, e in zip(actual, expected):
        torch.testing.assert_close(a, e, rtol=0, atol=0)


def fixture(*, a_supervised=True, u_supervised=True):
    rng = np.random.default_rng(71)
    M, K, D, H, C, P = 3, 32, 4, 5, 2, 7
    random = lambda shape: (rng.normal(size=shape) * .25).astype(np.float32)
    arrays = {
        "head_0_weight": random((H, D)), "head_0_bias": random((H,)),
        "head_2_weight": random((C, H)), "head_2_bias": random((C,)),
        "event_tokens": random((M, D)), "entity_tokens": random((K, D)),
        "event_presence": random((M,)), "entity_presence": random((K,)),
        "u_features": random((2, 2, C, P)), "u_logits": random((2, 2, M, P)),
        "u_points": np.tile(np.array([1, 1, 2, 3, 4, 5, 5], np.int64), (2, 2, 1)),
        "sampled_instances": np.tile(np.array([1, 1, 0, 0, 1, 0, 0], np.int64), (2, 2, 1)),
        "sampled_weights": np.tile(np.array([1, 1, 0, 1, 1, 1, 1], np.float32), (2, 2, 1)),
        "a_logits": random((2, K, P)),
        "a_entity_points": np.tile(np.array([1, 1, 0, 2, 2, 0, 0], np.int64), (2, 1)),
        "a_ignore": np.tile(np.array([False, False, True, False, False, False, False]), (2, 1)),
        "a_present": np.array([True, True] + [False] * (K - 2)),
        "roi_features": np.zeros((0, C), np.float32),
        "roi_reference_responses": np.zeros((0, 2, 2, 2), np.float64),
    }
    # Baseline matcher picks query0; global native logits strongly prefer query1.
    arrays["u_logits"][:, 0, 0] = np.array([5, 5, -5, -5, 5, -5, -5], np.float32)
    arrays["u_logits"][:, 0, 1:] = -3
    arrays["u_logits"][:, 1, 0] = -3
    arrays["u_logits"][:, 1, 1] = arrays["u_logits"][:, 0, 0]
    meta = {"source": "SYNTHETIC", "sequence": "SYNTHETIC", "n_instances": 1,
            "a_supervised": a_supervised, "u_supervised": u_supervised, "roi_rows": []}
    return meta, arrays


HEAD_NAMES = ("head_0_weight", "head_0_bias", "head_2_weight", "head_2_bias")


def independent_parameters(arrays, flat=None):
    params, offset = [], 0
    for name in HEAD_NAMES:
        base = torch.from_numpy(arrays[name])
        params.append(base.clone() if flat is None else flat[offset:offset + base.numel()].reshape(base.shape))
        offset += base.numel()
    return tuple(params)


def independent_projection(params, tokens, features=None):
    w0, b0, w2, b2 = params
    out = F.linear(F.gelu(F.linear(tokens, w0, b0)), w2, b2)
    return out if features is None else out @ features


def independent_losses(meta, arrays, draw, arm, flat, matches):
    params = independent_parameters(arrays, flat)
    anchor = independent_parameters(arrays)
    et = torch.from_numpy(arrays["event_tokens"])
    at = torch.from_numpy(arrays["entity_tokens"])
    feature = torch.from_numpy(arrays["u_features"][draw, arm])
    control = torch.from_numpy(arrays["u_features"][draw, 0])
    # Parentheses matter: subtract the same float32 projection before adding the
    # native value, so finite differences evaluate the declared anchored function.
    z = torch.from_numpy(arrays["u_logits"][draw, arm]) + (
        independent_projection(params, et, feature) - independent_projection(anchor, et, feature))
    az = torch.from_numpy(arrays["a_logits"][draw]) + (
        independent_projection(params, at, control) - independent_projection(anchor, at, control))
    ids = torch.from_numpy(arrays["sampled_instances"][draw, arm])
    target = (ids[None] == torch.arange(1, meta["n_instances"] + 1)[:, None]).float()
    # Use the official scalar implementation, changing only its assignment input.
    with patch("anatobind.aur.losses.match_events", return_value=matches):
        u, _ = losses.event_loss(torch.from_numpy(arrays["event_presence"])[None], z[None], [target],
                                  torch.from_numpy(arrays["sampled_weights"][draw, arm])[None],
                                  torch.tensor([meta["u_supervised"]]), u_mask_profile="unmatched_empty_v1")
    a = losses.entity_loss(az[None], torch.from_numpy(arrays["entity_presence"])[None],
                           torch.from_numpy(arrays["a_entity_points"][draw])[None],
                           torch.from_numpy(arrays["a_ignore"][draw])[None],
                           torch.from_numpy(arrays["a_present"])[None], torch.tensor([meta["a_supervised"]]))
    return u["u_presence"] + u["u_mask"], a["a_mask"] + a["a_presence"]


def test_native_anchor_is_exact_and_tangent_is_functional():
    module = tangent()
    _, arrays = fixture()
    head = torch.nn.Sequential(torch.nn.Linear(4, 5), torch.nn.GELU(), torch.nn.Linear(5, 2))
    for name, parameter in zip(HEAD_NAMES, head.parameters()):
        arrays[name] = parameter
    before = [p.detach().clone() for p in head.parameters()]
    tokens = torch.from_numpy(arrays["event_tokens"])
    features = torch.from_numpy(arrays["u_features"][0, 0])
    native = torch.from_numpy(arrays["u_logits"][0, 0])
    params = tuple(p.detach().float().clone().requires_grad_() for p in head.parameters())
    anchored = module.anchored_logits(params, tokens, features, native, tuple(p.detach().clone() for p in params))
    assert torch.equal(anchored, native)
    assert any(g.abs().sum() > 0 for g in grad(anchored.sum(), params))
    result = module.analyze_crop(fixture()[0], arrays)
    assert all(d["native_anchor_max_abs_error"] == 0 for d in result["summary"]["draws"])
    for p, original in zip(head.parameters(), before):
        assert torch.equal(p, original) and p.grad is None


def test_analysis_preserves_fixed_matches_and_all_numerical_interfaces():
    meta, arrays = fixture()
    original = {k: v.copy() for k, v in arrays.items()}
    with torch.inference_mode():
        result = tangent().analyze_crop(meta, arrays)
    numeric = result["numerics"]
    nphi = sum(arrays[k].size for k in HEAD_NAMES)
    expected = {"g_u": (2, 2, nphi), "g_a": (2, nphi),
                "directions": (2, 2, 2, 2, nphi), "direction_defined": (2, 2, 2, 2),
                "event_jvp": (2, 2, 2, 2, 3, 2), "entity_jvp": (2, 2, 2, 2, 32, 2),
                "a_loss_response": (2, 2, 2, 2), "a_query_loss_response": (2, 2, 2, 2, 32)}
    for name, shape in expected.items():
        assert numeric[name].shape == shape and np.isfinite(numeric[name]).all()
    for draw in result["summary"]["draws"]:
        assert draw["control_matches"] == {"query_indices": [0], "target_indices": [0]}
        assert draw["global_matches"] == {"query_indices": [1], "target_indices": [0]}
        assert draw["assignment"]["different_query"] == 1
        assert draw["control_loss_max_abs_error"] == 0
        assert draw["control_logit_gradient_max_abs_error"] == 0
    np.testing.assert_allclose(numeric["directions"][:, 0, :, 0],
                               -(numeric["g_u"] + numeric["g_a"][:, None]), rtol=0, atol=0)
    np.testing.assert_array_equal(numeric["directions"][:, 1, :, 0], -numeric["g_u"])
    np.testing.assert_allclose(numeric["a_query_loss_response"].sum(-1), numeric["a_loss_response"], rtol=2e-5, atol=2e-7)
    json.dumps(result["summary"], allow_nan=False)
    for k, v in original.items():
        np.testing.assert_array_equal(arrays[k], v)


def test_head_gradients_jvps_and_shared_a_response_match_independent_finite_difference():
    meta, arrays = fixture()
    result = tangent().analyze_crop(meta, arrays)["numerics"]
    baseline = torch.cat([v.flatten() for v in independent_parameters(arrays)])
    rng = np.random.default_rng(90)
    direction = torch.tensor(rng.normal(size=baseline.numel()), dtype=torch.float32)
    direction /= direction.norm()
    eps = .01
    fixed_matches = (torch.tensor([0]), torch.tensor([0]))
    for draw in range(2):
        for arm in range(2):
            up, ap = independent_losses(meta, arrays, draw, arm, baseline + eps * direction, fixed_matches)
            um, am = independent_losses(meta, arrays, draw, arm, baseline - eps * direction, fixed_matches)
            assert np.dot(result["g_u"][draw, arm], direction.numpy()) == pytest.approx(float((up - um) / (2 * eps)), abs=3e-5, rel=.004)
            assert np.dot(result["g_a"][draw], direction.numpy()) == pytest.approx(float((ap - am) / (2 * eps)), abs=3e-5, rel=.004)
            for objective in range(2):
                # Unit directions avoid step sizes depending on gradient magnitude.
                d = torch.from_numpy(result["directions"][draw, objective, arm, 1])
                plus, minus = independent_parameters(arrays, baseline + eps * d), independent_parameters(arrays, baseline - eps * d)
                for name, token_name in (("event_jvp", "event_tokens"), ("entity_jvp", "entity_tokens")):
                    tokens = torch.from_numpy(arrays[token_name])
                    fd = (independent_projection(plus, tokens) - independent_projection(minus, tokens)) / (2 * eps)
                    np.testing.assert_allclose(result[name][draw, objective, arm, 1], fd.numpy(), rtol=.002, atol=8e-6)
                _, ap = independent_losses(meta, arrays, draw, arm, baseline + eps * d, fixed_matches)
                _, am = independent_losses(meta, arrays, draw, arm, baseline - eps * d, fixed_matches)
                assert result["a_loss_response"][draw, objective, arm, 1] == pytest.approx(float((ap - am) / (2 * eps)), abs=3e-5, rel=.004)


@pytest.mark.parametrize("a_supervised,u_supervised", [(False, False), (False, True), (True, False)])
def test_disabled_supervision_has_zero_gradient_and_unit_direction_defined_mask(a_supervised, u_supervised):
    meta, arrays = fixture(a_supervised=a_supervised, u_supervised=u_supervised)
    result = tangent().analyze_crop(meta, arrays)
    num = result["numerics"]
    if not a_supervised:
        assert not num["g_a"].any() and not num["a_loss_response"].any() and not num["a_query_loss_response"].any()
    if not u_supervised:
        assert not num["g_u"].any()
        assert not num["direction_defined"][:, 1, :, 1].any()
        assert not num["directions"][:, 1].any() and not num["event_jvp"][:, 1].any()
    if not (a_supervised or u_supervised):
        assert not num["direction_defined"][..., 1].any()
        assert num["direction_defined"][..., 0].all()  # Raw zero direction is defined.


def test_all_zero_effective_points_produce_zero_head_gradient():
    meta, arrays = fixture(a_supervised=False)
    arrays["sampled_weights"][:] = 0
    result = tangent().analyze_crop(meta, arrays)["numerics"]
    assert not result["g_u"].any() and not result["direction_defined"][..., 1].any()


def test_core_accepts_arrays_before_roi_construction_and_anatomy_query_responses_are_identity_specific():
    meta, arrays = fixture()
    del meta["roi_rows"]
    del arrays["roi_features"], arrays["roi_reference_responses"]
    result = tangent().analyze_crop(meta, arrays)["numerics"]
    base = torch.cat([v.flatten() for v in independent_parameters(arrays)])
    d = torch.from_numpy(result["directions"][0, 1, 1, 1])
    eps = .01
    tokens, features = torch.from_numpy(arrays["entity_tokens"]), torch.from_numpy(arrays["u_features"][0, 0])
    anchor = independent_projection(independent_parameters(arrays), tokens, features)
    moved = [independent_projection(independent_parameters(arrays, base + sign * eps * d), tokens, features) - anchor
             for sign in (1, -1)]
    native = torch.from_numpy(arrays["a_logits"][0])
    for query in range(32):
        losses_q = []
        for residual in moved:
            z = native.clone()
            z[query] += residual[query]  # Only this identity moves; the official loss supplies its normalization.
            a = losses.entity_loss(z[None], torch.from_numpy(arrays["entity_presence"])[None],
                                   torch.from_numpy(arrays["a_entity_points"][0])[None],
                                   torch.from_numpy(arrays["a_ignore"][0])[None],
                                   torch.from_numpy(arrays["a_present"])[None], torch.tensor([True]))
            losses_q.append(sum(a.values()))
        fd = float((losses_q[0] - losses_q[1]) / (2 * eps))
        assert result["a_query_loss_response"][0, 1, 1, 1, query] == pytest.approx(fd, abs=2e-5, rel=.01)


@pytest.mark.parametrize("n", [0, 5])
def test_analysis_reports_empty_targets_and_capacity_overflow_without_dropping_missed_targets(n):
    meta, arrays = fixture(a_supervised=False)
    meta["n_instances"] = n
    arrays["sampled_instances"][:] = 0
    result = tangent().analyze_crop(meta, arrays)
    for draw in result["summary"]["draws"]:
        assert len(draw["control_matches"]["query_indices"]) == min(n, 3)
        assert draw["assignment"]["capacity_overflow"] == max(n - 3, 0)
        assert draw["assignment"]["control_zero_positive_targets"] == n
        assert draw["assignment"]["global_zero_effective_positive_targets"] == n
        assert draw["control_loss_max_abs_error"] == draw["control_logit_gradient_max_abs_error"] == 0


@pytest.mark.parametrize("bad", ["nonfinite", "fractional_weight", "bad_shape"])
def test_invalid_inputs_fail_with_fixed_anonymous_errors(bad):
    meta, arrays = fixture()
    if bad == "nonfinite":
        arrays["u_logits"][0, 0, 0, 0] = np.nan
    elif bad == "fractional_weight":
        arrays["sampled_weights"][0, 0, 0] = .5
    else:
        arrays["entity_tokens"] = arrays["entity_tokens"][:, :3]
    with pytest.raises(ValueError, match="^tangent input contract invalid$"):
        tangent().analyze_crop(meta, arrays)
