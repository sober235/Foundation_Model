"""Float32 frozen-feature mask-head tangent, anchored at native sampled logits.

Only detached copies of the four embed_full tensors participate in autograd.
Array axes are draw, objective (combined/U-only), arm (control/global), and
normalization (raw/unit), as applicable. Matching always uses the native logits;
the primary losses hold the control assignment fixed in both arms. This is not a
historical bf16 gradient or a parameter update. No file or model I/O occurs here.
"""
import numbers

import numpy as np
import torch
import torch.nn.functional as F

from anatobind.aur.losses import (
    NO_OBJECT_WEIGHT, U_MASK_PROFILES, dice_loss, entity_loss, event_loss,
)


HEAD_NAMES = ("head_0_weight", "head_0_bias", "head_2_weight", "head_2_bias")
_INPUT_ERROR = "tangent input contract invalid"


def fixed_event_loss(presence, logits, target, weight, matches, *,
                     u_supervised=True, profile="unmatched_empty_v1"):
    """One-sample official U scalar with an explicitly frozen assignment.

    Preserve occurrence multiplicity, point weights, global matched-query mean,
    and the separate unmatched-query mean. Unsupervised rows have zero local
    derivatives and never acquire empty-mask negative supervision.
    """
    if profile not in U_MASK_PROFILES:
        raise ValueError(_INPUT_ERROR)
    zero = (presence.sum() + logits.sum()) * 0
    if not u_supervised:
        return {"u_presence": zero, "u_mask": zero}
    if matches is None:
        raise ValueError(_INPUT_ERROR)
    qi, ti = (torch.as_tensor(v, dtype=torch.long, device=logits.device) for v in matches)
    m = presence.numel()
    if (presence.ndim != 1 or logits.ndim != 2 or logits.shape[0] != m
            or target.ndim != 2 or target.shape[1] != logits.shape[1]
            or tuple(weight.shape) != (logits.shape[1],) or qi.ndim != 1 or ti.shape != qi.shape
            or bool(((qi < 0) | (qi >= m)).any())
            or bool(((ti < 0) | (ti >= target.shape[0])).any())
            or qi.unique().numel() != qi.numel() or ti.unique().numel() != ti.numel()):
        raise ValueError(_INPUT_ERROR)
    pres_target = torch.zeros(m, device=presence.device)
    pres_target[qi] = 1.
    pres_weight = torch.full((m,), NO_OBJECT_WEIGHT, device=presence.device)
    pres_weight[qi] = 1.
    u_presence = (F.binary_cross_entropy_with_logits(presence, pres_target, reduction="none")
                  * pres_weight).sum() / pres_weight.sum()
    u_mask = zero
    if qi.numel():
        l, t, k = logits[qi], target[ti], weight[None]
        bce = (F.binary_cross_entropy_with_logits(l, t, reduction="none") * k).sum(-1) / k.sum(-1).clamp(min=1.)
        dropped = l.masked_fill(k < .5, -1e4)
        term = bce.sum() + dice_loss(dropped[None], (t * k)[None],
                                    torch.ones(1, qi.numel(), device=l.device)) * qi.numel()
        u_mask = term / qi.numel()
    if profile == "unmatched_empty_v1":
        free = torch.ones(m, dtype=torch.bool, device=logits.device)
        free[qi] = False
        if bool(free.any()):
            l, k = logits[free], weight[None]
            bce = (F.binary_cross_entropy_with_logits(l, torch.zeros_like(l), reduction="none")
                   * k).sum(-1) / k.sum(-1).clamp(min=1.)
            u_mask = u_mask + NO_OBJECT_WEIGHT * bce.mean()
    return {"u_presence": u_presence, "u_mask": u_mask}


def head_embeddings(parameters, tokens):
    """The original Linear/GELU/Linear head using functional float32 copies."""
    w0, b0, w2, b2 = parameters
    return F.linear(F.gelu(F.linear(tokens, w0, b0)), w2, b2)


def anchored_logits(parameters, tokens, features, native_logits, reference_parameters):
    """Native values plus a zero-at-reference float32 residual, with fixed anchor."""
    with torch.no_grad():
        anchor = head_embeddings(reference_parameters, tokens) @ features
    return native_logits + (head_embeddings(parameters, tokens) @ features - anchor)


def _tensor(value):
    value = torch.as_tensor(value).detach().cpu().clone()
    if value.is_complex() or not bool(torch.isfinite(value).all()):
        raise ValueError(_INPUT_ERROR)
    return value


def _inputs(meta, arrays):
    names = HEAD_NAMES + ("event_tokens", "entity_tokens", "event_presence", "entity_presence",
                          "u_logits", "u_features", "sampled_instances", "sampled_weights",
                          "a_logits", "a_entity_points", "a_ignore", "a_present")
    try:
        t = {key: _tensor(arrays[key]) for key in names}
        n = meta["n_instances"]
        if isinstance(n, (bool, np.bool_)) or not isinstance(n, numbers.Integral) or n < 0:
            raise ValueError
        if any(not isinstance(meta[key], (bool, np.bool_)) for key in ("a_supervised", "u_supervised")):
            raise ValueError
        w0, b0, w2, b2 = (t[key] for key in HEAD_NAMES)
        if w0.ndim != 2 or w2.ndim != 2:
            raise ValueError
        h, d = w0.shape
        c = w2.shape[0]
        m, k, p = t["event_tokens"].shape[0], t["entity_tokens"].shape[0], t["u_logits"].shape[-1]
        if min(h, d, c, m, k, p) < 1:
            raise ValueError
        shapes = {"head_0_bias": (h,), "head_2_weight": (c, h), "head_2_bias": (c,),
                  "event_tokens": (m, d), "entity_tokens": (k, d),
                  "event_presence": (m,), "entity_presence": (k,),
                  "u_logits": (2, 2, m, p), "u_features": (2, 2, c, p),
                  "sampled_instances": (2, 2, p), "sampled_weights": (2, 2, p),
                  "a_logits": (2, k, p), "a_entity_points": (2, p), "a_ignore": (2, p), "a_present": (k,)}
        if any(tuple(t[key].shape) != shape for key, shape in shapes.items()):
            raise ValueError
        for key, top in (("sampled_instances", n), ("a_entity_points", k)):
            if not bool(((t[key] >= 0) & (t[key] <= top) & (t[key] == t[key].floor())).all()):
                raise ValueError
        for key in ("sampled_weights", "a_ignore", "a_present"):
            if not bool(((t[key] == 0) | (t[key] == 1)).all()):
                raise ValueError
        for key in names:
            if key in ("sampled_instances", "a_entity_points"):
                t[key] = t[key].long()
            elif key in ("a_ignore", "a_present"):
                t[key] = t[key].bool()
            else:
                t[key] = t[key].float()
                if not bool(torch.isfinite(t[key]).all()):
                    raise ValueError
    except (KeyError, TypeError, ValueError, RuntimeError, IndexError):
        raise ValueError(_INPUT_ERROR) from None
    return t, int(n)


def _grad(total, leaf, *, retain_graph=False):
    if not total.requires_grad:
        return torch.zeros_like(leaf)
    value = torch.autograd.grad(total, leaf, allow_unused=True, retain_graph=retain_graph)[0]
    return torch.zeros_like(leaf) if value is None else value


def _match_dict(matches):
    # Disabled supervision has undefined targets, represented by empty lists plus
    # the explicit u_supervised flag in the enclosing summary.
    return {"query_indices": [] if matches is None else matches[0].tolist(),
            "target_indices": [] if matches is None else matches[1].tolist()}


def _loss_values(parts):
    return {key: float(value.detach()) for key, value in parts.items()}


def _official(presence, logits, target, weight, enabled):
    return event_loss(presence[None], logits[None], [target], weight[None],
                      torch.tensor([enabled]), u_mask_profile="unmatched_empty_v1")


def _loss_check(presence, native, target, weight, matches, enabled):
    p, z = presence.clone().requires_grad_(), native.clone().requires_grad_()
    official, _ = _official(p, z, target, weight, enabled)
    fixed = fixed_event_loss(p, z, target, weight, matches, u_supervised=enabled)
    value_error = max(abs(float((official[key] - fixed[key]).detach())) for key in official)
    actual_total, fixed_total = sum(official.values()), sum(fixed.values())
    errors = []
    for leaf in (p, z):
        expected = _grad(actual_total, leaf, retain_graph=True)
        got = _grad(fixed_total, leaf, retain_graph=True)
        errors.append(float((expected - got).abs().max()))
    if value_error != 0. or max(errors) != 0.:
        raise ValueError("tangent control reconstruction failed")
    return value_error, max(errors)


@torch.inference_mode(False)
@torch.enable_grad()
def analyze_crop(meta, arrays):
    """Reconstruct two paired draws without loading a checkpoint or dense arrays.

    Extra ROI and R fields are intentionally left to the measurement/archive
    layers. All returned summaries are JSON safe, but contain private query data;
    the caller owns redaction and must never print this whole result.
    """
    t, n = _inputs(meta, arrays)
    enabled_a, enabled_u = bool(meta["a_supervised"]), bool(meta["u_supervised"])
    parameters = tuple(t[name] for name in HEAD_NAMES)
    sizes, shapes = [p.numel() for p in parameters], [p.shape for p in parameters]
    phi = torch.cat([p.flatten() for p in parameters]).requires_grad_()

    def unpack(flat):
        return tuple(part.view(shape) for part, shape in zip(flat.split(sizes), shapes))

    all_tokens = torch.cat([t["event_tokens"], t["entity_tokens"]])
    m, k, c, nphi = t["event_tokens"].shape[0], t["entity_tokens"].shape[0], parameters[2].shape[0], phi.numel()
    numerics = {"g_u": np.zeros((2, 2, nphi), np.float32), "g_a": np.zeros((2, nphi), np.float32),
                "directions": np.zeros((2, 2, 2, 2, nphi), np.float32),
                "direction_defined": np.ones((2, 2, 2, 2), bool),
                "event_jvp": np.zeros((2, 2, 2, 2, m, c), np.float32),
                "entity_jvp": np.zeros((2, 2, 2, 2, k, c), np.float32),
                "a_loss_response": np.zeros((2, 2, 2, 2), np.float64),
                "a_query_loss_response": np.zeros((2, 2, 2, 2, k), np.float64)}
    draws = []
    for draw in range(2):
        targets, official_parts, matches = [], [], []
        for arm in range(2):
            ids = t["sampled_instances"][draw, arm]
            target = (ids[None] == torch.arange(1, n + 1)[:, None]).float()
            parts, matched = _official(t["event_presence"], t["u_logits"][draw, arm], target,
                                       t["sampled_weights"][draw, arm], enabled_u)
            targets.append(target)
            official_parts.append(parts)
            matches.append(matched[0])
        loss_error, logit_error = _loss_check(t["event_presence"], t["u_logits"][draw, 0], targets[0],
                                             t["sampled_weights"][draw, 0], matches[0], enabled_u)
        fixed_values, anchor_errors, gradients_u = [], [], []
        for arm in range(2):
            z = anchored_logits(unpack(phi), t["event_tokens"], t["u_features"][draw, arm],
                                 t["u_logits"][draw, arm], parameters)
            anchor_errors.append(float((z.detach() - t["u_logits"][draw, arm]).abs().max()))
            parts = fixed_event_loss(t["event_presence"], z, targets[arm], t["sampled_weights"][draw, arm],
                                      matches[0], u_supervised=enabled_u)
            fixed_values.append(_loss_values(parts))
            gradients_u.append(_grad(sum(parts.values()), phi))
        az = anchored_logits(unpack(phi), t["entity_tokens"], t["u_features"][draw, 0],
                              t["a_logits"][draw], parameters)
        anchor_errors.append(float((az.detach() - t["a_logits"][draw]).abs().max()))
        if max(anchor_errors) != 0.:
            raise ValueError("tangent native anchor failed")
        a_parts = entity_loss(az[None], t["entity_presence"][None], t["a_entity_points"][draw][None],
                              t["a_ignore"][draw][None], t["a_present"][None], torch.tensor([enabled_a]))
        a_total = sum(a_parts.values())
        a_logit_gradient = _grad(a_total, az, retain_graph=True)
        gradient_a = _grad(a_total, phi)
        numerics["g_u"][draw] = torch.stack(gradients_u).detach().numpy()
        numerics["g_a"][draw] = gradient_a.detach().numpy()
        response_errors = []
        for objective in range(2):
            for arm in range(2):
                gradient = gradients_u[arm] + gradient_a if objective == 0 else gradients_u[arm]
                # Float64 norm avoids declaring a tiny nonzero float32 gradient
                # zero merely because squaring it underflows.
                norm = float(torch.linalg.vector_norm(gradient.double()))
                for normalization in range(2):
                    defined = normalization == 0 or norm > 0.
                    direction = -gradient if normalization == 0 else (
                        (-gradient.double() / norm).float() if defined else torch.zeros_like(gradient))
                    index = (draw, objective, arm, normalization)
                    numerics["directions"][index] = direction.detach().numpy()
                    numerics["direction_defined"][index] = defined
                    if bool(direction.any()):
                        _, jvp = torch.autograd.functional.jvp(
                            lambda flat: head_embeddings(unpack(flat), all_tokens), phi, direction)
                    else:
                        jvp = torch.zeros((m + k, c), dtype=phi.dtype)
                    numerics["event_jvp"][index] = jvp[:m].detach().numpy()
                    numerics["entity_jvp"][index] = jvp[m:].detach().numpy()
                    query_response = (a_logit_gradient.double()
                                      * (jvp[m:].double() @ t["u_features"][draw, 0].double())).sum(-1)
                    response = float((gradient_a.double() * direction.double()).sum())
                    numerics["a_query_loss_response"][index] = query_response.detach().numpy()
                    numerics["a_loss_response"][index] = response
                    response_errors.append(abs(float(query_response.sum()) - response))
                    if not np.isclose(float(query_response.sum()), response, rtol=2e-5, atol=2e-7):
                        raise ValueError("tangent anatomy response reconstruction failed")
        mappings = [dict(zip(v[1].tolist(), v[0].tolist())) if v is not None else {} for v in matches]
        norm_u = [float(g.double().norm()) for g in gradients_u]
        norm_combined = [float((g + gradient_a).double().norm()) for g in gradients_u]
        cosine = lambda a, b: (float(F.cosine_similarity(a.double(), b.double(), dim=0, eps=1e-300))
                                if bool(a.any() and b.any()) else None)
        assignment = {"capacity_overflow": max(n - m, 0), "assignment_denominator": n if enabled_u else None,
                      "different_query": sum(mappings[0][i] != mappings[1][i] for i in mappings[0].keys() & mappings[1].keys()) if enabled_u else None,
                      "same_query": sum(mappings[0][i] == mappings[1][i] for i in mappings[0].keys() & mappings[1].keys()) if enabled_u else None,
                      "control_only": len(mappings[0].keys() - mappings[1].keys()) if enabled_u else None,
                      "global_only": len(mappings[1].keys() - mappings[0].keys()) if enabled_u else None,
                      "neither": n - len(mappings[0].keys() | mappings[1].keys()) if enabled_u else None}
        for arm, label in enumerate(("control", "global")):
            positive = targets[arm].sum(-1)
            effective = (targets[arm] * t["sampled_weights"][draw, arm][None]).sum(-1)
            assignment[label + "_zero_positive_targets"] = int((positive == 0).sum())
            assignment[label + "_zero_effective_positive_targets"] = int((effective == 0).sum())
        draws.append({"draw": draw, "a_supervised": enabled_a, "u_supervised": enabled_u,
                      "control_matches": _match_dict(matches[0]), "global_matches": _match_dict(matches[1]),
                      "fixed_losses": fixed_values, "official_losses": [_loss_values(v) for v in official_parts],
                      "a_losses": _loss_values(a_parts), "assignment": assignment,
                      "control_loss_max_abs_error": loss_error, "control_logit_gradient_max_abs_error": logit_error,
                      "native_anchor_max_abs_error": max(anchor_errors),
                      "a_query_response_max_abs_error": max(response_errors),
                      "gradient_norms": {"u": norm_u, "a": float(gradient_a.double().norm()), "combined": norm_combined},
                      "gradient_cosines": {"u": cosine(*gradients_u),
                                           "combined": cosine(*(g + gradient_a for g in gradients_u))},
                      "a_query_support": [{"query": q, "enabled": enabled_a, "present": bool(t["a_present"][q]),
                                           "effective_positive_points": int(((t["a_entity_points"][draw] == q + 1)
                                                                              & ~t["a_ignore"][draw]).sum())}
                                          for q in range(k)]})
    if not all(np.isfinite(v).all() for v in numerics.values()):
        raise ValueError("tangent numeric result nonfinite")
    return {"summary": {"n_parameters": nphi, "n_event_queries": m, "n_entity_queries": k, "draws": draws},
            "numerics": numerics}
