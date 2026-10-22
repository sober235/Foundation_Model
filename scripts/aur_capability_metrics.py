"""Bounded TRAIN-only capability arithmetic; private records must stay outside Git.

No loader, checkpoint deserialization, runtime mutation or printing occurs here.
Counters returned by aggregation are JSON-safe; cell records and fitted readouts
are private CPU tensors. Missing denominators are explicitly represented by None.
"""
import math

import torch
import torch.nn.functional as F

from anatobind.aur.labels import ENTITY_HOST
from anatobind.aur.losses import match_events, sample_points


def _require(condition, message="invalid capability target contract"):
    if not condition:
        raise ValueError(message)


def _finite(value):
    if isinstance(value, torch.Tensor):
        _require(bool(torch.isfinite(value).all()), "capability values must be finite")
    else:
        _require(isinstance(value, (int, float)) and not isinstance(value, bool)
                 and math.isfinite(value), "capability values must be finite")


def _batch(batch):
    """Validate the real collate_unified batch-of-one before model use."""
    _require(isinstance(batch, dict))
    mode = batch.get("input_mode")
    _require(mode in ("2d", "3d"))
    entity = batch.get("entity")
    _require(isinstance(entity, torch.Tensor) and entity.ndim == 4 and entity.shape[0] == 1)
    shape = tuple(entity.shape)
    _require(all(n > 0 for n in shape) and (mode != "2d" or shape[1] == 1))
    for key in ("valid", "instance", "a_ignore", "point_weight"):
        item = batch.get(key)
        _require(isinstance(item, torch.Tensor) and tuple(item.shape) == shape)
        _finite(item)
    _finite(entity)
    _require(entity.dtype in (torch.uint8, torch.int16, torch.int32, torch.int64)
             and bool(((entity >= 0) & (entity <= 32)).all()))
    inst = batch["instance"]
    _require(inst.dtype in (torch.uint8, torch.int16, torch.int32, torch.int64))
    for key in ("a_supervised", "u_supervised", "r_supervised"):
        flag = batch.get(key)
        _require(isinstance(flag, torch.Tensor) and tuple(flag.shape) == (1,)
                 and flag.dtype == torch.bool)
    numbers = batch.get("n_instances")
    _require(isinstance(numbers, (list, tuple)) and len(numbers) == 1
             and type(numbers[0]) is int and numbers[0] >= 0)
    n = numbers[0]
    _require(bool(((inst >= 0) & (inst <= n)).all()))
    for key in ("host", "r_eligible"):
        values = batch.get(key)
        _require(isinstance(values, (list, tuple)) and len(values) == 1
                 and isinstance(values[0], torch.Tensor) and tuple(values[0].shape) == (n,))
    host = batch["host"][0]
    _require(host.dtype == torch.long and bool(((host >= -1) & (host < 14)).all())
             and batch["r_eligible"][0].dtype == torch.bool)
    sources = batch.get("source_labels")
    _require(isinstance(sources, list) and len(sources) == 1
             and isinstance(sources[0], str) and bool(sources[0]))
    valid = batch["valid"] > .5
    weighted = valid & (batch["point_weight"] > .5)
    # A crop-local target must carry real valid support, not only a declared id.
    ids = torch.unique(inst[weighted])
    ids = ids[ids > 0]
    _require(ids.numel() == n)
    return valid, weighted


def _counts(labels, mask, classes):
    return torch.bincount(labels[mask].long(), minlength=classes + 1)[1:].cpu().tolist()


def batch_support(batch):
    """Label-only support. Host 0 and no-host 13 are valid; unknown -1 is not."""
    valid, weighted = _batch(batch)
    a = valid & ~batch["a_ignore"].bool() & batch["a_supervised"][0]
    u = weighted & (batch["instance"] > 0) & batch["u_supervised"][0]
    host = batch["host"][0]
    eligible = batch["r_eligible"][0] & (host >= 0)
    eligible = eligible & batch["r_supervised"][0] & batch["u_supervised"][0]
    return {"a_truth": _counts(batch["entity"], a, 32),
            "u_instances": int(torch.unique(batch["instance"][u]).numel()),
            "u_voxels": int(u.sum()),
            "background_voxels": int((weighted & (batch["instance"] == 0)).sum()),
            "r_hosts": torch.bincount(host[eligible], minlength=14).cpu().tolist(),
            "sources": list(batch["source_labels"])}


def _support(item):
    for key, count in (("a_truth", 32), ("r_hosts", 14)):
        _require(isinstance(item.get(key), list) and len(item[key]) == count)
        _require(all(type(n) is int and n >= 0 for n in item[key]))
    for key in ("u_instances", "u_voxels", "background_voxels"):
        _require(type(item.get(key)) is int and item[key] >= 0)
    _require(isinstance(item.get("sources"), list)
             and all(isinstance(s, str) and bool(s) for s in item["sources"]))


def panel_support(supports):
    """Aggregate coverage without exposing source names or observation records."""
    a, r, sources = [0] * 32, [0] * 14, set()
    instances = voxels = background = 0
    all_background = bool(supports)
    for item in supports:
        _support(item)
        a = [x + y for x, y in zip(a, item["a_truth"])]
        r = [x + y for x, y in zip(r, item["r_hosts"])]
        instances += item["u_instances"]
        voxels += item["u_voxels"]
        background += item["background_voxels"]
        sources.update(item["sources"])
        all_background &= item["background_voxels"] > 0
    hosts = sum(n > 0 for n in r)
    passed = all(n > 0 for n in a) and instances >= 4 and hosts >= 2 and len(sources) >= 3 and all_background
    return {"status": "PASS" if passed else "INSUFFICIENT", "observations": len(supports),
            "a_truth": a, "u_instances": instances, "u_voxels": voxels,
            "background_voxels": background, "every_observation_has_background": all_background,
            "r_hosts": r, "supported_r_hosts": hosts, "source_count": len(sources)}


def select_panel(supports, max_observations=16):
    """Deterministic lexicographic coverage, then lower candidate/draw index."""
    _require(type(max_observations) is int and 0 <= max_observations <= 16)
    for item in supports:
        _support(item)
    selected, a, r, sources = [], set(), set(), set()
    remaining = set(range(len(supports)))
    while remaining and len(selected) < max_observations:
        def rank(i):
            item = supports[i]
            return (sum(n > 0 and k not in a for k, n in enumerate(item["a_truth"])),
                    sum(n > 0 and k not in r for k, n in enumerate(item["r_hosts"])),
                    len(set(item["sources"]) - sources), item["u_instances"], -i)
        index = max(remaining, key=rank)
        item = supports[index]
        selected.append(index)
        remaining.remove(index)
        a.update(k for k, n in enumerate(item["a_truth"]) if n > 0)
        r.update(k for k, n in enumerate(item["r_hosts"]) if n > 0)
        sources.update(item["sources"])
    return selected


def _cells(out, batch):
    f2 = out["levels"][1]["feat"].float()
    grid = tuple(f2.shape[2:])
    shape = tuple(batch["entity"].shape[1:])
    _require(len(grid) == 3 and all(s % g == 0 for s, g in zip(shape, grid)))
    stride = tuple(s // g for s, g in zip(shape, grid))
    parts = [f2] + [F.interpolate(out["levels"][k]["feat"].float(), size=grid,
                                mode="trilinear", align_corners=False) for k in (2, 3)]
    features = torch.cat(parts, 1)[0].flatten(1).t()
    _finite(features)
    # Same majority pooling as G1; non-host/background wins ties at index zero.
    lut = torch.tensor([0] + [h + 1 for h in ENTITY_HOST], device=batch["entity"].device)
    host = lut[batch["entity"].long()]
    onehot = F.one_hot(host, 14).permute(0, 4, 1, 2, 3).float()
    anatomy_valid = ((batch["valid"] > .5) & ~batch["a_ignore"].bool()).float()[:, None]
    target = F.avg_pool3d(onehot * anatomy_valid, stride, stride)[0].argmax(0).flatten()
    fraction = F.avg_pool3d(anatomy_valid, stride, stride)[0, 0].flatten()
    feature_valid = out["levels"][1]["valid"]
    _require(tuple(feature_valid.shape) == (1, *grid))
    _finite(feature_valid)
    valid = (feature_valid[0].flatten() > .5) & (fraction >= .5) & (target > 0) & batch["a_supervised"][0]
    per_host = [torch.where(valid & (target == k))[0] for k in range(1, 14)]
    allocations = [0] * 13
    for _ in range(min(512, int(valid.sum()))):
        candidates = [k for k in range(13) if allocations[k] < len(per_host[k])]
        k = min(candidates, key=lambda c: (allocations[c], c))
        allocations[k] += 1
    chosen = []
    for indices, count in zip(per_host, allocations):
        if count:
            ranks = torch.linspace(0, len(indices) - 1, count, device=indices.device).round().long()
            chosen.append(indices[ranks])
    indices = torch.cat(chosen).sort().values if chosen else torch.empty(0, dtype=torch.long, device=features.device)
    return {"features": features[indices].detach().cpu(), "targets": (target[indices] - 1).long().cpu(),
            "parity": (indices % 2).bool().cpu(), "indices": indices.cpu(),
            "grid": grid, "input_mode": batch["input_mode"]}


@torch.no_grad()
def measure_batch(model, batch):
    """One forward, native dense A/union U and fixed sampled instance matching.

    Query matching uses a local CPU generator and never consumes training RNG.
    Relation inputs are actual matched query indices; bind constructs its own
    predicted geometry. All submodule train/eval flags are restored on errors.
    """
    valid, weighted = _batch(batch)
    shape = tuple(batch["entity"].shape)
    for key, channels in (("image", 1), ("coords", 3), ("local", 3)):
        x = batch.get(key)
        _require(isinstance(x, torch.Tensor) and tuple(x.shape) == (1, channels, *shape[1:]))
        _finite(x)
    state = [(module, module.training) for module in model.modules()]
    model.eval()
    try:
        out = model(batch["image"], batch["valid"], batch["coords"], batch["local"], input_mode=batch["input_mode"])
        ap, up = out["entity_presence"], out["event_presence"]
        _require(tuple(ap.shape) == (1, 32) and up.ndim == 2 and up.shape[0] == 1)
        _finite(ap)
        _finite(up)
        a_logits = model.entity_masks(out)
        _require(tuple(a_logits.shape) == (1, 32, *shape[1:]))
        _finite(a_logits)
        scores, predicted = (a_logits.sigmoid() * ap.sigmoid()[:, :, None, None, None]).max(1)
        predicted = (predicted + 1) * (scores > .5)
        keep_a = valid & ~batch["a_ignore"].bool() & batch["a_supervised"][0]
        a = {"truth": _counts(batch["entity"], keep_a, 32), "predicted": _counts(predicted, keep_a, 32),
             "intersection": _counts(predicted, keep_a & (predicted == batch["entity"]), 32)}
        del a_logits, scores, predicted
        u = {"dice_sum": 0., "instances": 0, "present": 0, "unmatched_active": 0,
             "union_truth": 0, "union_predicted": 0, "union_intersection": 0, "observations": 0}
        r = {"truth": [0] * 14, "correct": [0] * 14, "cross_entropy_sum": 0., "eligible": 0}
        if bool(batch["u_supervised"][0]):
            _require(bool(weighted.any()))
            points = sample_points(weighted.cpu(), 16384, generator=torch.Generator().manual_seed(1729),
                                   instance=batch["instance"].cpu()).to(up.device)
            logits = model.event_masks(out)
            _require(tuple(logits.shape) == (1, up.shape[1], *shape[1:]))
            _finite(logits)
            sampled = logits.flatten(2).gather(2, points[:, None].expand(-1, up.shape[1], -1))[0]
            labels = batch["instance"].flatten(1).gather(1, points)[0]
            ids = torch.arange(1, batch["n_instances"][0] + 1, device=labels.device)
            target = labels[None] == ids[:, None]
            qi, ti = match_events(up[0], sampled, target.float())
            active = up[0].sigmoid() > .5
            hard = sampled[qi].sigmoid() > .3
            truth = target[ti]
            denominator = hard.sum(1) + truth.sum(1)
            _require(bool((denominator > 0).all()))
            u["dice_sum"] = float((2 * (hard & truth).sum(1).double() / denominator).sum())
            # Unmatched real targets remain zero Dice and zero recall in this denominator.
            u["instances"] = len(ids)
            u["present"] = int(active[qi].sum())
            u["unmatched_active"] = int(active.sum() - active[qi].sum())
            union = (logits[0, active].sigmoid() > .3).any(0) if bool(active.any()) else torch.zeros_like(valid[0])
            truth_union = batch["instance"][0] > 0
            u.update(union_truth=int((truth_union & weighted[0]).sum()),
                     union_predicted=int((union & weighted[0]).sum()),
                     union_intersection=int((union & truth_union & weighted[0]).sum()), observations=1)
            if bool(batch["r_supervised"][0]) and len(qi):
                host = batch["host"][0].to(qi.device)
                eligible = batch["r_eligible"][0].to(qi.device)[ti] & (host[ti] >= 0)
                targets = host[ti[eligible]]
                if targets.numel():
                    rel = model.bind(out, 0, qi[eligible])
                    _require(tuple(rel.shape) == (len(targets), 14))
                    _finite(rel)
                    correct = targets[rel.argmax(1) == targets]
                    r = {"truth": torch.bincount(targets, minlength=14).cpu().tolist(),
                         "correct": torch.bincount(correct, minlength=14).cpu().tolist(),
                         "cross_entropy_sum": float(F.cross_entropy(rel.float(), targets, reduction="sum")),
                         "eligible": len(targets)}
        return {"a": a, "u": u, "r": r, "cells": _cells(out, batch)}
    finally:
        for module, training in state:
            module.training = training


def _ratio(numerator, denominator):
    return numerator / denominator if denominator > 0 else None


def _dice(truth, predicted, intersection):
    return [_ratio(2 * i, t + p) for t, p, i in zip(truth, predicted, intersection)]


def aggregate_measurements(records):
    """Sufficient-statistic aggregation; never average missing classes into success."""
    a = {k: [0] * 32 for k in ("truth", "predicted", "intersection")}
    u = {k: 0 for k in ("dice_sum", "instances", "present", "unmatched_active", "union_truth",
                        "union_predicted", "union_intersection", "observations")}
    r = {"truth": [0] * 14, "correct": [0] * 14, "cross_entropy_sum": 0., "eligible": 0}
    for record in records:
        for category, accumulator in (("a", a), ("u", u), ("r", r)):
            for key, previous in list(accumulator.items()):
                value = record[category][key]
                if isinstance(previous, list):
                    _require(isinstance(value, list) and len(value) == len(previous))
                    for count in value:
                        _finite(count)
                        _require(type(count) is int and count >= 0)
                    accumulator[key] = [x + y for x, y in zip(previous, value)]
                else:
                    _finite(value)
                    _require(value >= 0)
                    if key not in ("dice_sum", "cross_entropy_sum"):
                        _require(type(value) is int)
                    accumulator[key] += value
    _require(all(i <= min(t, p) for t, p, i in zip(a["truth"], a["predicted"], a["intersection"])))
    _require(u["dice_sum"] <= u["instances"] and u["present"] <= u["instances"]
             and u["union_intersection"] <= min(u["union_truth"], u["union_predicted"]))
    _require(r["eligible"] == sum(r["truth"]) and all(c <= t for c, t in zip(r["correct"], r["truth"])))
    a["dice"] = _dice(a["truth"], a["predicted"], a["intersection"])
    supported = [d for d, t in zip(a["dice"], a["truth"]) if t > 0]
    a["macro"] = _ratio(sum(supported), len(supported))
    u.update(matched_mask_dice=_ratio(u["dice_sum"], u["instances"]),
             present_recall=_ratio(u["present"], u["instances"]),
             unmatched_active_per_observation=_ratio(u["unmatched_active"], u["observations"]),
             union_dice=_ratio(2 * u["union_intersection"], u["union_truth"] + u["union_predicted"]))
    accuracy = [c / t for c, t in zip(r["correct"], r["truth"]) if t > 0]
    r.update(macro_accuracy=_ratio(sum(accuracy), len(accuracy)),
             cross_entropy=_ratio(r["cross_entropy_sum"], r["eligible"]), supported_hosts=len(accuracy),
             majority_accuracy=_ratio(max(r["truth"]), r["eligible"]),
             majority_macro_accuracy=_ratio(1, len(accuracy)))
    return {"a": a, "u": u, "r": r}


def _retention_cells(records):
    features, targets, parity, indices, metadata = [], [], [], [], []
    for record in records:
        cell = record["cells"]
        x, y, p, idx = (cell[k] for k in ("features", "targets", "parity", "indices"))
        _require(all(isinstance(t, torch.Tensor) and t.device.type == "cpu" for t in (x, y, p, idx)))
        _require(x.ndim == 2 and y.shape == p.shape == idx.shape == (len(x),)
                 and y.dtype == torch.long and p.dtype == torch.bool and idx.dtype == torch.long)
        _finite(x)
        _require(bool(((y >= 0) & (y < 13)).all()) and bool((idx >= 0).all())
                 and torch.equal(p, (idx % 2).bool()) and len(torch.unique(idx)) == len(idx))
        _require(not features or features[0].shape[1] == x.shape[1])
        features.append(x.double())
        targets.append(y)
        parity.append(p)
        indices.append(idx)
        metadata.append((len(x), cell.get("grid"), cell.get("input_mode")))
    if not features:
        return torch.empty(0, 0, dtype=torch.float64), torch.empty(0, dtype=torch.long), torch.empty(0, dtype=torch.bool), [], []
    return torch.cat(features), torch.cat(targets), torch.cat(parity), indices, metadata


def fit_retention(records):
    """Fit class-balanced ridge once, using even cells only and fixed ridge .01."""
    x, y, parity, indices, metadata = _retention_cells(records)
    fit_counts = torch.bincount(y[~parity], minlength=13)
    observation_counts = torch.bincount(y[parity], minlength=13)
    if not bool(((fit_counts >= 2) & (observation_counts >= 2)).all()):
        return {"status": "INSUFFICIENT", "fit_counts": fit_counts.tolist(),
                "observation_counts": observation_counts.tolist()}
    fit = x[~parity]
    mean, scale = fit.mean(0), fit.std(0, unbiased=False)
    scale = torch.where(scale > 0, scale, torch.ones_like(scale))
    design = torch.cat(((fit - mean) / scale, torch.ones(len(fit), 1, dtype=torch.float64)), 1)
    target = F.one_hot(y[~parity], 13).double()
    weights = 1. / (13 * fit_counts[y[~parity]].double())
    penalty = torch.eye(design.shape[1], dtype=torch.float64) * .01
    penalty[-1, -1] = 0
    weight = torch.linalg.solve(design.t() @ (design * weights[:, None]) + penalty,
                                design.t() @ (target * weights[:, None]))
    _finite(weight)
    return {"status": "PASS", "mean": mean, "scale": scale, "weight": weight,
            "targets": y.clone(), "parity": parity.clone(), "indices": [i.clone() for i in indices],
            "metadata": metadata, "observation_features": x[parity].clone(),
            "fit_counts": fit_counts.tolist(), "observation_counts": observation_counts.tolist()}


def score_retention(records, readout):
    """Reuse frozen labels, support, normalization and head; no fitting occurs."""
    if readout.get("status") != "PASS":
        return {"status": "INSUFFICIENT", "macro_dice": None, "feature_relative_change": None}
    x, y, parity, indices, metadata = _retention_cells(records)
    _require(metadata == readout["metadata"] and torch.equal(y, readout["targets"])
             and torch.equal(parity, readout["parity"]) and len(indices) == len(readout["indices"])
             and all(torch.equal(i, j) for i, j in zip(indices, readout["indices"])),
             "capability retention cell support changed")
    observed = x[parity]
    design = torch.cat(((observed - readout["mean"]) / readout["scale"],
                        torch.ones(len(observed), 1, dtype=torch.float64)), 1)
    scores = design @ readout["weight"]
    _finite(scores)
    predicted, truth = scores.argmax(1), y[parity]
    tc = torch.bincount(truth, minlength=13).tolist()
    pc = torch.bincount(predicted, minlength=13).tolist()
    inter = torch.bincount(truth[predicted == truth], minlength=13).tolist()
    dice = _dice(tc, pc, inter)
    baseline = readout["observation_features"]
    relative = _ratio(float(torch.linalg.vector_norm(observed - baseline)), float(torch.linalg.vector_norm(baseline)))
    return {"status": "PASS" if relative is not None else "INSUFFICIENT", "macro_dice": sum(dice) / 13,
            "truth": tc, "predicted": pc, "intersection": inter, "dice": dice,
            "feature_relative_change": relative}


def capability_verdict(initial, stage_two, stage_three, retention_baseline, retention_final):
    """Frozen per-mode thresholds. stage_three has explicit initial/final metrics."""
    try:
        three0, three = stage_three["initial"], stage_three["final"]
        for point in (initial, stage_two, three0, three):
            _require(len(point["a"]["truth"]) == 32 and all(n > 0 for n in point["a"]["truth"]))
            _require(point["u"]["instances"] >= 4 and point["r"]["eligible"] > 0
                     and point["r"]["supported_hosts"] >= 2)
            for value in (point["a"]["macro"], point["u"]["matched_mask_dice"],
                          point["u"]["present_recall"], point["u"]["union_dice"],
                          point["r"]["macro_accuracy"], point["r"]["cross_entropy"],
                          point["r"]["majority_macro_accuracy"]):
                _finite(value)
        for retention in (retention_baseline, retention_final):
            _require(retention["status"] == "PASS")
            _finite(retention["macro_dice"])
        _require(three0["r"]["cross_entropy"] > 0)
    except (KeyError, TypeError, ValueError):
        return {"status": "INSUFFICIENT", "evidence": "NOT_EVIDENCE", "checks": {}}
    checks = {}
    for name, final in (("stage_two", stage_two), ("stage_three", three)):
        checks[name + "_a_gain"] = final["a"]["macro"] >= initial["a"]["macro"] + .05
        checks[name + "_u_gain"] = final["u"]["matched_mask_dice"] >= initial["u"]["matched_mask_dice"] + .10
        checks[name + "_presence"] = final["u"]["present_recall"] >= .5
        checks[name + "_union"] = final["u"]["union_dice"] >= .25
    checks["r_accuracy"] = three["r"]["macro_accuracy"] >= max(.5, three["r"]["majority_macro_accuracy"] + .10)
    checks["r_cross_entropy"] = three["r"]["cross_entropy"] <= .8 * three0["r"]["cross_entropy"]
    checks["retention"] = retention_final["macro_dice"] >= retention_baseline["macro_dice"] - .05
    return {"status": "PASS" if all(checks.values()) else "FAIL", "evidence": "NOT_EVIDENCE", "checks": checks}
