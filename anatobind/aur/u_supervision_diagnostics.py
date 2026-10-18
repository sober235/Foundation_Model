"""Private CPU diagnostics for the unchanged sampled U loss.

Official loss values, assignments and detached float32-logit gradients are authoritative.
Independent float64 scalar decomposition uses atol=1e-6, rtol=1e-5. Gradients refer to
sampled logit occurrences, not shared parameters; duplicate indices retain input order.
"""
import math
import numbers

import numpy as np
import torch

from .dataset import event_targets_at_points
from .losses import NO_OBJECT_WEIGHT, event_loss


_RECONSTRUCTION_ATOL = 1e-6
_RECONSTRUCTION_RTOL = 1e-5


def _array(value, name):
    try:
        value = torch.as_tensor(value).detach().cpu()
        if value.is_complex() or not bool(torch.isfinite(value).all()):
            raise ValueError
        return value
    except (TypeError, ValueError, RuntimeError):
        raise ValueError(f"{name} must contain finite real numeric values") from None


def _integer(value, name):
    if isinstance(value, bool) or not isinstance(value, numbers.Integral) or value < 0:
        raise ValueError(f"{name} must be a nonnegative integer")
    return int(value)


def _gradient(leaf):
    return np.zeros(tuple(leaf.shape), dtype=np.float32) if leaf.grad is None else leaf.grad.numpy().copy()


@torch.inference_mode(False)
def probe_event_loss(presence, mask_logits, instance_points, point_weight, point_indices, n_instances,
                     *, u_supervised=True, profile="unmatched_empty_v1"):
    """Probe one crop using fresh CPU leaves and the actual target/loss APIs.

    Sample indices are nonnegative flat voxel indices; their order and multiplicity
    are preserved. Targets of unsupervised queries are undefined (``None``), rather
    than empty masks. ``mask_logit_gradient`` is d(u_mask+u_presence)/d(mask_logits).
    Query mask contributions sum to u_mask, including the unmatched-query mean.
    """
    n_instances = _integer(n_instances, "n_instances")
    if not isinstance(u_supervised, (bool, np.bool_)):
        raise ValueError("u_supervised must be boolean")
    u_supervised = bool(u_supervised)
    p = _array(presence, "presence").to(torch.float32).clone()
    m = _array(mask_logits, "mask_logits").to(torch.float32).clone()
    ids = _array(instance_points, "instance_points")
    weights = _array(point_weight, "point_weight")
    indices = _array(point_indices, "point_indices")
    if p.ndim != 1 or m.ndim != 2 or m.shape[0] != p.numel() or min(m.shape) < 1:
        raise ValueError("presence and mask_logits must have shapes [Q] and [Q,P] with Q,P positive")
    q_count, point_count = m.shape
    if any(tuple(v.shape) != (point_count,) for v in (ids, weights, indices)):
        raise ValueError("point arrays must have shape [P]")
    if not bool(torch.isfinite(p).all() and torch.isfinite(m).all()):
        raise ValueError("logits must be finite in float32")
    if not bool(((weights == 0) | (weights == 1)).all()):
        raise ValueError("point_weight must be binary")
    if not bool(((ids >= 0) & (ids <= n_instances) & (ids == ids.floor())).all()):
        raise ValueError("instance_points must be integer IDs in 0..n_instances")
    if not bool(((indices >= 0) & (indices == indices.floor())).all()) or (
        indices.is_floating_point() and not bool((indices < float(2**63)).all())
    ):
        raise ValueError("point_indices must be nonnegative int64 indices")
    ids, indices, weights = ids.long(), indices.long(), weights.to(torch.float32).clone()
    targets = event_targets_at_points(ids[None], [n_instances])
    with torch.enable_grad():
        p.requires_grad_()
        m.requires_grad_()
        actual, official_matches = event_loss(p[None], m[None], targets, weights[None],
                                               torch.tensor([u_supervised]), u_mask_profile=profile)
        total = actual["u_presence"] + actual["u_mask"]
        if total.requires_grad:
            total.backward()
    mg, pg = _gradient(m), _gradient(p)
    if not all(math.isfinite(float(v)) for v in actual.values()) or not (np.isfinite(mg).all() and np.isfinite(pg).all()):
        raise ValueError("official loss or gradients are nonfinite")
    match = official_matches[0]
    pairs = [] if match is None else list(zip(match[0].tolist(), match[1].tolist()))
    assignment = [None] * q_count
    for q, target in pairs:
        assignment[q] = target + 1
    matched_count, unmatched_count = len(pairs), q_count - len(pairs)
    w = weights.numpy().astype(np.float64)
    idx, labels = indices.numpy(), ids.numpy()
    effective = w > 0
    point_denom = max(float(w.sum()), 1.)
    presence_denom = matched_count + NO_OBJECT_WEIGHT * unmatched_count if u_supervised else None
    target_counts = [int(np.count_nonzero(labels == i)) for i in range(1, n_instances + 1)]
    effective_counts = [int(np.count_nonzero((labels == i) & effective)) for i in range(1, n_instances + 1)]
    p_values, m_values = p.detach().numpy().astype(np.float64), m.detach().numpy().astype(np.float64)
    queries = []
    for q in range(q_count):
        matched = assignment[q] is not None
        prob = float(torch.sigmoid(p.detach()[q]))
        target = (labels == assignment[q]) if matched else np.zeros(point_count, dtype=bool)
        pres_target = float(matched) if u_supervised else None
        pres_weight = (1. if matched else NO_OBJECT_WEIGHT) if u_supervised else None
        pres_bce = float(np.logaddexp(0., p_values[q]) - p_values[q] * pres_target) if u_supervised else None
        pres_contribution = pres_bce * pres_weight / presence_denom if u_supervised else 0.
        mask_enabled = u_supervised and (matched or profile == "unmatched_empty_v1")
        bce = float(((np.logaddexp(0., m_values[q]) - m_values[q] * target) * w).sum() / point_denom) if mask_enabled else None
        dice = None
        contribution = 0.
        if mask_enabled:
            if matched:
                # Official masked_fill(-1e4) makes the sigmoid exactly zero in float32.
                probability = torch.sigmoid(m.detach()[q]).numpy().astype(np.float64) * w
                weighted_target = target * w
                dice = float(1 - (2 * (probability * weighted_target).sum() + 1) /
                             (probability.sum() + weighted_target.sum() + 1))
                contribution = (bce + dice) / max(matched_count, 1)
            else:
                contribution = NO_OBJECT_WEIGHT * bce / max(unmatched_count, 1)
        gradient = mg[q]
        queries.append({
            "query": q, "target_instance": assignment[q], "matched": matched if u_supervised else None,
            "presence_probability": prob, "presence_target": pres_target, "presence_weight": pres_weight,
            "presence_gradient": float(pg[q]), "presence_bce": pres_bce, "presence_contribution": pres_contribution,
            "mask_bce": bce, "mask_dice": dice, "mask_contribution": contribution,
            "target_positive_points": int(target.sum()) if u_supervised else None,
            "target_positive_unique": int(np.unique(idx[target]).size) if u_supervised else None,
            "target_positive_effective": int((target & effective).sum()) if u_supervised else None,
            "target_positive_effective_unique": int(np.unique(idx[target & effective]).size) if u_supervised else None,
            "mask_gradient_sum": float(gradient.astype(np.float64).sum()),
            "mask_gradient_abs_sum": float(np.abs(gradient.astype(np.float64)).sum()),
            "mask_gradient_l2": float(np.linalg.norm(gradient.astype(np.float64))),
            "mask_gradient_positive_points": int((gradient > 0).sum()),
            "mask_gradient_negative_points": int((gradient < 0).sum()),
            "mask_gradient_zero_points": int((gradient == 0).sum()),
        })
    observed = {key: float(value) for key, value in actual.items()}
    reconstructed = {"u_presence": sum(q["presence_contribution"] for q in queries),
                     "u_mask": sum(q["mask_contribution"] for q in queries)}
    if not all(math.isclose(observed[k], reconstructed[k], abs_tol=_RECONSTRUCTION_ATOL,
                            rel_tol=_RECONSTRUCTION_RTOL) for k in observed):
        raise ValueError("independent loss reconstruction exceeds documented tolerance")
    return {
        "u_supervised": u_supervised, "n_queries": q_count, "n_instances": n_instances,
        "matches": None if match is None else {"query_indices": match[0].tolist(), "target_indices": match[1].tolist()},
        "query_target_instance": assignment, "losses": observed, "reconstructed_losses": reconstructed,
        "normalization": {"sampled_points": point_count, "unique_points": int(np.unique(idx).size),
                          "effective_points": int(w.sum()), "unique_effective_points": int(np.unique(idx[effective]).size),
                          "point_denominator": point_denom, "matched_queries": matched_count if u_supervised else None,
                          "unmatched_queries": unmatched_count if u_supervised else None,
                          "matched_query_denominator": max(matched_count, 1) if u_supervised else None,
                          "unmatched_query_denominator": max(unmatched_count, 1) if u_supervised else None,
                          "presence_weight_sum": presence_denom, "supervised_samples": int(u_supervised),
                          "unmatched_mask_weight": NO_OBJECT_WEIGHT if profile == "unmatched_empty_v1" else 0.,
                          "reconstruction_atol": _RECONSTRUCTION_ATOL, "reconstruction_rtol": _RECONSTRUCTION_RTOL},
        "target_point_counts": target_counts, "target_effective_point_counts": effective_counts,
        "queries": queries, "mask_logit_gradient": mg, "presence_logit_gradient": pg,
    }


def compare_draw_assignments(first, second, n_instances):
    """Count target-to-query changes; empty/capacity-limited targets stay in the denominator.

    Inputs must describe the same frozen crop and query ordering. This helper cannot
    verify forward identity; the caller owns frozen-logit/provenance verification.
    Assignment statistics are undefined if either draw is U-unsupervised.
    """
    n_instances = _integer(n_instances, "n_instances")
    q_count = _integer(first["n_queries"], "n_queries")
    if second["n_queries"] != q_count or any(d["n_instances"] != n_instances for d in (first, second)):
        raise ValueError("draw query and target cardinalities must agree")
    assignments = []
    for draw in (first, second):
        values = draw["query_target_instance"]
        if len(values) != q_count:
            raise ValueError("draw assignment length must equal n_queries")
        mapping = {}
        for q, target in enumerate(values):
            if target is not None:
                target = _integer(target, "target_instance")
                if target < 1 or target > n_instances or target in mapping:
                    raise ValueError("draw target assignments must be unique IDs in 1..n_instances")
                mapping[target] = q
        assignments.append(mapping)
    a, b = assignments
    defined = bool(first["u_supervised"] and second["u_supervised"])
    result = {"n_instances": n_instances, "n_queries": q_count, "u_supervised": defined,
              "capacity_overflow": max(n_instances - q_count, 0), "assignment_denominator": n_instances if defined else None}
    counts = {"first_matched": len(a), "second_matched": len(b), "both_matched": len(a.keys() & b.keys()),
              "same_query": sum(a[t] == b[t] for t in a.keys() & b.keys()),
              "different_query": sum(a[t] != b[t] for t in a.keys() & b.keys()),
              "first_only": len(a.keys() - b.keys()), "second_only": len(b.keys() - a.keys()),
              "neither": n_instances - len(a.keys() | b.keys())}
    result.update({k: v if defined else None for k, v in counts.items()})
    for prefix, draw in (("first", first), ("second", second)):
        for key, suffix in (("target_point_counts", "zero_positive_targets"),
                            ("target_effective_point_counts", "zero_effective_positive_targets")):
            values = draw[key]
            if len(values) != n_instances or any(not isinstance(v, numbers.Integral) or v < 0 for v in values):
                raise ValueError("draw target counts must contain one nonnegative integer per target")
            result[f"{prefix}_{suffix}"] = sum(v == 0 for v in values)
    return result
