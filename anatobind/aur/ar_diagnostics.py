"""Bounded, float32 TRAIN A/R diagnostics, separate from training and U decoding.

The archive is private. Its public aggregate contains only counts and class-level
statistics. A replay reaggregates A sufficient statistics (not raw dense maps)
and independently recomputes R geometry and the small frozen relation head on
CPU. It never opens an image or deserializes a model checkpoint.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time

import numpy as np
import torch
import torch.nn.functional as F

from anatobind.aur.relation import (CandidateCompetition, DIST_CAP_MM, SCALE_MM, geometry,
                                    host_from_entities, host_masks_from_entities)

PAIRS = (("bmsr", "T1c"), ("isles", "DWI"), ("pdgm", "FLAIR"))
SEED = 20261019
RTOL, ATOL = 2e-4, 2e-5
C2_STATE_SHA = "5fbb0d6f9513fdebf66831033c6a8d32710a05ce2ec900656c062a61546e482f"
ARRAY_KEYS = {"entity_tokens", "event_tokens", "seq_token", "entity_presence", "coarse_a", "coarse_u",
              "dense_a", "dense_u", "valid", "coords", "coarse_geo", "dense_geo", "coarse_logits", "dense_logits"}
STABLE_ANALYSIS = "stable_distance_cpu64_v1"
RESULT_KEYS = ("coarse_geo", "dense_geo", "coarse_logits", "dense_logits")


def file_digest(path):
    h = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def state_digest(state):
    h = hashlib.sha256()
    for name, value in sorted(state.items()):
        t = value.detach().cpu().contiguous()
        h.update(json.dumps([name, str(t.dtype), list(t.shape)], separators=(",", ":")).encode())
        h.update(t.reshape(-1).view(torch.uint8).numpy().tobytes())
    return h.hexdigest()


def select_train_rows(rows, val_patients, per_source=1, seed=SEED):
    """Return full TRAIN rows and selected original TRAIN indices, one row/patient.

    Selection uses only split, supervision and source/sequence metadata. It is a
    new contract and may overlap patients from earlier TRAIN diagnostics.
    """
    from anatobind.aur.train import split_rows
    if per_source < 1:
        raise ValueError("Invalid selection budget")
    train, _ = split_rows(rows, val_patients)
    rng = np.random.default_rng(seed)
    selected = []
    for source, sequence in PAIRS:
        candidates = [i for i, r in enumerate(train) if r["source"] == source and r["sequence"] == sequence
                      and r.get("a_supervised", True) and r.get("r_supervised", True) and r["u_supervised"]]
        seen = set()
        for index in rng.permutation(candidates):
            row = train[int(index)]
            if str(row["patient"]) not in seen:
                seen.add(str(row["patient"]))
                selected.append(int(index))
                if len(seen) == per_source:
                    break
        if len(seen) != per_source:
            raise ValueError("Insufficient eligible TRAIN membership")
    return train, selected


def _sum(values):
    return float(values.double().sum())


def entity_query_statistics(blocks, entity, valid, a_ignore, points):
    """Reduce ordered dense-logit query blocks without storing all 32 maps.

    Margins use the GT query minus the largest other query probability. Exact
    argmax ties use the lowest query index, matching torch.argmax. Statistics
    exclude invalid and A-ignore voxels; sample counts retain multiplicity and
    also report unique positive voxel coverage.
    """
    gt = entity.detach().cpu().long().flatten()
    good = (valid.detach().cpu().flatten() > .5) & ~a_ignore.detach().cpu().bool().flatten()
    pts = torch.as_tensor(points, dtype=torch.long).cpu().flatten()
    if gt.numel() != good.numel() or (gt < 0).any() or (gt > 32).any() or (pts < 0).any() or (pts >= gt.numel()).any():
        raise ValueError("Invalid entity diagnostic inputs")
    best = torch.full((gt.numel(),), -1.)
    second = best.clone()
    winner = torch.full_like(gt, -1)
    gt_prob = torch.zeros_like(best)
    rows, next_query = [], 0
    for start, logits in blocks:
        if start != next_query:
            raise ValueError("Dense query blocks must be complete and ordered")
        probs = logits.detach().float().cpu().sigmoid().flatten(1)
        if probs.shape[1] != gt.numel() or not torch.isfinite(probs).all():
            raise ValueError("Invalid dense entity probabilities")
        for prob in probs:
            q = next_query
            if q >= 32:
                raise ValueError("Unexpected anatomy query count")
            positive = good & (gt == q + 1)
            negative = good & (gt != q + 1)
            selected = pts[good[pts] & (gt[pts] == q + 1)]
            count = int(positive.sum())
            rows.append(dict(query=q, positive_voxels=count, negative_voxels=int(negative.sum()),
                             positive_probability_sum=_sum(prob[positive]),
                             negative_probability_sum=_sum(prob[negative]),
                             positive_above_half=int((prob[positive] >= .5).sum()),
                             sampled_positive_points=int(selected.numel()),
                             sampled_unique_positive_voxels=int(selected.unique().numel()),
                             sampled_positive_probability_sum=_sum(prob[selected])))
            gt_prob[positive] = prob[positive]
            higher = prob > best
            second = torch.where(higher, best, torch.maximum(second, prob))
            winner = torch.where(higher, torch.full_like(winner, q), winner)
            best = torch.maximum(best, prob)
            next_query += 1
    if next_query != 32:
        raise ValueError("Incomplete anatomy queries")
    for row in rows:
        pos = good & (gt == row["query"] + 1)
        other = torch.where(winner == row["query"], second, best)
        n = row["positive_voxels"]
        row.update(winner_voxels=int((pos & (winner == row["query"])).sum()),
                   winner_margin_sum=_sum((gt_prob - other)[pos]),
                   positive_probability_mean=row["positive_probability_sum"] / n if n else None,
                   winner_margin_mean=_sum((gt_prob - other)[pos]) / n if n else None,
                   sampled_positive_voxel_fraction=row["sampled_unique_positive_voxels"] / n if n else None,
                   sampled_positive_probability_mean=row["sampled_positive_probability_sum"] / row["sampled_positive_points"]
                   if row["sampled_positive_points"] else None)
    return rows


def pool_probabilities(logits, patch):
    """Sigmoid first, then patch-aligned mean; F1 validity is applied later."""
    if logits.ndim != 4 or any(n % p for n, p in zip(logits.shape[1:], patch)):
        raise ValueError("Unaligned dense probability grid")
    return F.avg_pool3d(logits.float().sigmoid()[None], kernel_size=tuple(patch), stride=tuple(patch))[0]


def stable_distance_geometry(event_probs, host_probs, coords):
    """New CPU analysis only: retain float32 centroids and all other features.

    Evaluate the nearest-host Euclidean norm using explicit float64 coordinate
    differences, never the squared-norm matrix expansion. This is deliberately
    separate from production geometry and the original native replay contract.
    """
    if event_probs.device.type != "cpu":
        raise ValueError("Stable analysis is CPU only")
    result = geometry(event_probs, host_probs, coords)
    e, h, c = event_probs.flatten(1).float(), host_probs.flatten(1).float(), coords.flatten(1).float()
    # Match the original float32 centroid definition exactly; only the norm's
    # arithmetic changes. Do not silently replace centroid summation precision.
    centroids = (e @ c.t()) / e.sum(1).clamp(min=1e-6)[:, None]
    distances = torch.full((e.shape[0], h.shape[0]), DIST_CAP_MM, dtype=torch.float64)
    for j in range(h.shape[0]):
        voxels = c[:, h[j] > .5].t()
        for start in range(0, len(voxels), 4096):
            delta = centroids.double()[:, None] - voxels[start:start + 4096].double()[None]
            nearest = delta.square().sum(-1).sqrt().amin(1)
            distances[:, j] = torch.minimum(distances[:, j], nearest)
    distances = torch.where(result[..., 0] > .05, torch.zeros_like(distances), distances)
    result[..., 4] = (distances / SCALE_MM).to(result.dtype)
    return result


def _relation_pair(relation, entity_tokens, event_tokens, seq_token, presence,
                   coarse_a, coarse_u, dense_a, dense_u, valid, coords, geometry_function=geometry):
    gate = presence.sigmoid()[:, None, None, None]
    host_tokens = host_from_entities(entity_tokens[None])[0]
    result = {}
    for arm, a, u in (("coarse", coarse_a, coarse_u), ("dense", dense_a, dense_u)):
        hosts = host_masks_from_entities((a * gate)[None])[0] * valid
        geo = geometry_function(u * valid, hosts, coords)
        result[f"{arm}_geo"] = geo
        result[f"{arm}_logits"] = relation(event_tokens, host_tokens, seq_token, geo)
    return result


@torch.no_grad()
def relation_from_arrays(relation, arrays, *, distance_backend="native_float32"):
    device = next(relation.parameters()).device
    t = {k: torch.as_tensor(arrays[k], dtype=torch.float32, device=device) for k in
         ("entity_tokens", "event_tokens", "seq_token", "entity_presence", "coarse_a", "coarse_u", "dense_a", "dense_u", "valid", "coords")}
    if distance_backend not in ("native_float32", STABLE_ANALYSIS):
        raise ValueError("Unknown distance backend")
    function = stable_distance_geometry if distance_backend == STABLE_ANALYSIS else geometry
    result = _relation_pair(relation, t["entity_tokens"], t["event_tokens"], t["seq_token"], t["entity_presence"],
                            t["coarse_a"], t["coarse_u"], t["dense_a"], t["dense_u"], t["valid"], t["coords"], function)
    return {k: v.detach().cpu().numpy() for k, v in result.items()}


@torch.no_grad()
def collect_crop(model, crop, *, points=16384, seed=SEED, query_chunk=4):
    """One float32 frozen forward, no U decoder, optimizer or parameter backward."""
    from anatobind.aur.losses import sample_points
    if model.training or any(p.requires_grad or p.grad is not None for p in model.parameters()) or query_chunk < 1:
        raise ValueError("Expected frozen eval model and positive chunk size")
    device = next(model.parameters()).device
    before = state_digest(model.state_dict())
    out = model(*[crop[k][None].to(device, dtype=torch.float32) for k in ("image", "valid", "coords", "local")])
    token_keys = ("entity_embed", "event_embed", "seq_embed")
    token_before = state_digest({k: out[k] for k in token_keys})
    indices = torch.arange(out["event_embed"].shape[1], device=device)
    original = model.bind(out, 0, indices).float()
    pts = sample_points(crop["valid"][None].cpu(), points, torch.Generator().manual_seed(seed),
                        instance=crop["instance"][None].cpu())[0]
    patch = tuple(model.cfg["patch"])
    dense_a = []

    def entity_blocks():
        for start in range(0, 32, query_chunk):
            logits = model.masks.full_masks(out["entity_embed"][:, start:start + query_chunk], out["pix"])[0].float()
            dense_a.append(pool_probabilities(logits, patch))
            yield start, logits.cpu()

    summary = entity_query_statistics(entity_blocks(), crop["entity"], crop["valid"], crop["a_ignore"], pts)
    dense_u = []
    for start in range(0, len(indices), query_chunk):
        logits = model.masks.full_masks(out["event_embed"][:, start:start + query_chunk], out["pix"])[0].float()
        dense_u.append(pool_probabilities(logits, patch))
    coarse = {"coarse": out["pix"]["coarse"]}
    t = dict(entity_tokens=out["entity_embed"][0], event_tokens=out["event_embed"][0], seq_token=out["seq_embed"][0],
             entity_presence=out["entity_presence"][0],
             coarse_a=model.masks.coarse_masks(out["entity_embed"], coarse)[0].sigmoid(),
             coarse_u=model.masks.coarse_masks(out["event_embed"], coarse)[0].sigmoid(),
             dense_a=torch.cat(dense_a), dense_u=torch.cat(dense_u),
             valid=out["levels"][0]["valid"][0], coords=out["levels"][0]["coords"][0])
    paired = _relation_pair(model.relation, t["entity_tokens"], t["event_tokens"], t["seq_token"], t["entity_presence"],
                            t["coarse_a"], t["coarse_u"], t["dense_a"], t["dense_u"], t["valid"], t["coords"])
    torch.testing.assert_close(original, paired["coarse_logits"], rtol=1e-6, atol=1e-6)
    error = float((original - paired["coarse_logits"]).abs().max())
    t.update(paired)
    arrays = {k: v.detach().float().cpu().numpy().copy() for k, v in t.items()}
    token_after = state_digest({k: out[k] for k in token_keys})
    if (not all(np.isfinite(v).all() for v in arrays.values()) or before != state_digest(model.state_dict())
            or token_after != token_before or any(p.grad is not None for p in model.parameters())):
        raise ValueError("Frozen diagnostic controls changed")
    hashes = {k: hashlib.sha256(crop[k].detach().cpu().contiguous().numpy().tobytes()).hexdigest()
              for k in ("image", "valid", "coords", "local", "entity", "a_ignore", "instance", "point_weight")}
    meta = dict(a_queries=summary, n_queries=len(indices), points=int(pts.numel()),
                coarse_control_max_abs=error, state_before=before, state_after=before,
                tokens_before=token_before, tokens_after=token_after, crop_tensor_sha256=hashes)
    return meta, arrays


def _json(path):
    return json.loads(Path(path).read_text())


def _write_json(path, value):
    with os.fdopen(os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600), "w") as stream:
        json.dump(value, stream, sort_keys=True, indent=2, allow_nan=False)
        stream.write("\n")


def _write_npz(path, arrays):
    with os.fdopen(os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600), "wb") as stream:
        np.savez_compressed(stream, **arrays)


def _load_arrays(path):
    with np.load(path, allow_pickle=False) as archive:
        out = {k: archive[k] for k in archive.files}
    if any(v.dtype != np.float32 or not np.isfinite(v).all() for v in out.values()):
        raise ValueError("Invalid numeric arrays")
    return out


def create_archive(out, contract, selection, relation):
    out = Path(out)
    out.mkdir(mode=0o700, parents=True, exist_ok=False)
    if out.is_symlink() or out.stat().st_mode & 0o777 != 0o700:
        raise ValueError("Invalid output permissions")
    _write_json(out / "contract.json", contract)
    _write_json(out / "selection_private.json", selection)
    _write_npz(out / "relation_state.npz", {k: v.detach().float().cpu().numpy() for k, v in relation.state_dict().items()})


def write_crop(out, index, metadata, arrays):
    if set(arrays) != ARRAY_KEYS:
        raise ValueError("Unexpected diagnostic tensor membership")
    _write_npz(Path(out) / f"crop_{index:03d}.npz", arrays)
    _write_json(Path(out) / f"crop_{index:03d}.json", metadata)


def _hashes(out):
    return {p.name: file_digest(p) for p in sorted(Path(out).iterdir()) if p.name != "COMPLETE.json"}


def _aggregate(records, contrasts, provenance):
    counts = ("positive_voxels", "negative_voxels", "positive_above_half", "sampled_positive_points",
              "sampled_unique_positive_voxels", "winner_voxels")
    sums = ("positive_probability_sum", "negative_probability_sum", "sampled_positive_probability_sum", "winner_margin_sum")
    classes = []
    for q in range(32):
        rows = [r["a_queries"][q] for r in records]
        row = dict(query=q, **{k: sum(r[k] for r in rows) for k in counts + sums})
        n = row["positive_voxels"]
        row.update(crops_with_gt=sum(r["positive_voxels"] > 0 for r in rows),
                   positive_probability_mean=row["positive_probability_sum"] / n if n else None,
                   winner_margin_mean=row["winner_margin_sum"] / n if n else None,
                   winner_fraction=row["winner_voxels"] / n if n else None,
                   sampled_positive_voxel_fraction=row["sampled_unique_positive_voxels"] / n if n else None,
                   sampled_positive_probability_mean=row["sampled_positive_probability_sum"] / row["sampled_positive_points"]
                   if row["sampled_positive_points"] else None)
        classes.append(row)
    geo = np.concatenate([a["dense_geo"] - a["coarse_geo"] for a in contrasts])
    logits = np.concatenate([a["dense_logits"] - a["coarse_logits"] for a in contrasts])
    return dict(schema=1, evidence="NOT_EVIDENCE: frozen TRAIN pseudo-label diagnostic; no training or clinical claim",
                n_crops=len(records), sources={s: sum(r.get("source") == s for r in records) for s, _ in PAIRS},
                a_queries=classes, a_replay="Stored dense sufficient statistics; raw dense maps are not archived",
                r=dict(n_queries=int(logits.shape[0]), geometry_mean_abs_by_feature=np.abs(geo).mean((0, 1)).tolist(),
                       geometry_max_abs_by_feature=np.abs(geo).max((0, 1)).tolist(),
                       logit_mean_abs_by_host=np.abs(logits).mean(0).tolist(),
                       logit_max_abs=float(np.abs(logits).max()),
                       choice_flips=sum(int((a["dense_logits"].argmax(1) != a["coarse_logits"].argmax(1)).sum()) for a in contrasts),
                       coarse_control_max_abs=max(r["coarse_control_max_abs"] for r in records),
                       interpretation="Joint predicted A/U geometry swap at fixed tokens; no attribution to A or U alone"),
                provenance=provenance)


def _rebuild(out):
    out = Path(out)
    contract = _json(out / "contract.json")
    relation = CandidateCompetition(**contract["relation_config"]).eval().requires_grad_(False)
    relation.load_state_dict({k: torch.from_numpy(v) for k, v in _load_arrays(out / "relation_state.npz").items()}, strict=True)
    records, contrasts = [], []
    for i in range(contract["n_crops"]):
        meta = _json(out / f"crop_{i:03d}.json")
        arrays = _load_arrays(out / f"crop_{i:03d}.npz")
        if (set(arrays) != ARRAY_KEYS or len(meta["a_queries"]) != 32 or meta["state_before"] != meta["state_after"]
                or meta["tokens_before"] != meta["tokens_after"]):
            raise ValueError("Diagnostic control differs")
        actual = relation_from_arrays(relation, arrays)
        for k, value in actual.items():
            np.testing.assert_allclose(value, arrays[k], rtol=RTOL, atol=ATOL)
        records.append(meta)
        # Aggregate captured values so CPU/GPU rounding cannot rewrite results.
        contrasts.append(arrays)
    return _aggregate(records, contrasts, contract["provenance"])


def complete_archive(out):
    report = _rebuild(out)
    _write_json(Path(out) / "aggregate_anonymous.json", report)
    _write_json(Path(out) / "COMPLETE.json", dict(schema=1, files=_hashes(out)))
    return report


def replay_archive(out):
    try:
        out = Path(out)
        if out.is_symlink() or out.stat().st_mode & 0o777 != 0o700:
            raise ValueError("Invalid archive directory")
        contract = _json(out / "contract.json")
        names = {"contract.json", "selection_private.json", "relation_state.npz", "aggregate_anonymous.json", "COMPLETE.json"}
        names |= {f"crop_{i:03d}.{ext}" for i in range(contract["n_crops"]) for ext in ("json", "npz")}
        if {p.name for p in out.iterdir()} != names or any(p.is_symlink() or p.stat().st_mode & 0o777 != 0o600 for p in out.iterdir()):
            raise ValueError("Invalid archive members")
        if _json(out / "COMPLETE.json") != dict(schema=1, files=_hashes(out)):
            raise ValueError("Archive checksums differ")
        report = _rebuild(out)
        if report != _json(out / "aggregate_anonymous.json"):
            raise ValueError("Archive aggregate differs")
        return report
    except Exception:
        raise ValueError("Invalid A/R diagnostic archive") from None


def _native_names(n_crops):
    return {"contract.json", "selection_private.json", "relation_state.npz"} | {
        f"crop_{i:03d}.{ext}" for i in range(n_crops) for ext in ("json", "npz")}


def _native_copy_name(name):
    return name if name == "relation_state.npz" else "native_" + name


def _write_bytes(path, value):
    with os.fdopen(os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600), "wb") as stream:
        stream.write(value)


def _verified_native_source(source, snapshot):
    """Verify collection-time bytes, never bless an absent native COMPLETE."""
    try:
        source, snapshot = Path(source), Path(snapshot)
        if source.is_symlink() or snapshot.is_symlink() or (source / "COMPLETE.json").exists():
            raise ValueError("Unexpected native completion")
        for directory in (source, snapshot):
            if directory.stat().st_mode & 0o777 != 0o700:
                raise ValueError("Invalid private permissions")
        contract = _json(source / "contract.json")
        manifest = _json(snapshot / "snapshot_manifest.json")
        if manifest["native_replay_passed"] is not False or manifest["native_complete_present"] is not False:
            raise ValueError("Native failure was not preserved")
        expected = manifest["native_member_sha256"]
        if set(expected) != _native_names(contract["n_crops"]):
            raise ValueError("Native member commitment differs")
        if contract["provenance"]["source_sha256"] != manifest["source_sha256"]:
            raise ValueError("Collection source commitment differs")
        for name, digest in manifest["source_sha256"].items():
            path = snapshot / name
            if path.is_symlink() or not path.resolve().is_relative_to(snapshot.resolve()) or file_digest(path) != digest:
                raise ValueError("Collection-time source bytes differ")
        for name, digest in expected.items():
            path = source / name
            if path.is_symlink() or path.stat().st_mode & 0o777 != 0o600 or file_digest(path) != digest:
                raise ValueError("Native member differs")
        return contract, manifest
    except Exception:
        raise ValueError("Invalid native A/R source") from None


def _differences(actual, expected):
    result = {}
    for key in RESULT_KEYS:
        delta = np.abs(actual[key] - expected[key])
        result[key] = dict(max_abs=float(delta.max()),
                           failed_elements=int((delta > ATOL + RTOL * np.abs(expected[key])).sum()))
    return result


def _frozen_relation(out, config):
    relation = CandidateCompetition(**config).eval().requires_grad_(False)
    relation.load_state_dict({k: torch.from_numpy(v) for k, v in _load_arrays(Path(out) / "relation_state.npz").items()}, strict=True)
    return relation


def _rebuild_stable(out):
    out = Path(out)
    contract = _json(out / "contract.json")
    if (contract.get("schema") != 2 or contract.get("analysis_version") != STABLE_ANALYSIS
            or contract.get("native_replay_passed") is not False
            or contract.get("tolerance") != dict(rtol=RTOL, atol=ATOL)):
        raise ValueError("Stable analysis contract differs")
    snapshot_manifest = _json(out / "collection_snapshot_manifest.json")
    if file_digest(out / "collection_snapshot_manifest.json") != contract["collection_snapshot_sha256"]:
        raise ValueError("Snapshot commitment differs")
    if snapshot_manifest["native_member_sha256"] != contract["native_member_sha256"]:
        raise ValueError("Native source commitment differs")
    for name, digest in contract["native_member_sha256"].items():
        if file_digest(out / _native_copy_name(name)) != digest:
            raise ValueError("Original native bytes changed")
    native_contract = _json(out / "native_contract.json")
    if (native_contract["provenance"]["source_sha256"] != snapshot_manifest["source_sha256"]
            or native_contract["n_crops"] != contract["n_crops"]):
        raise ValueError("Native source lineage differs")
    relation = _frozen_relation(out, contract["relation_config"])
    records, stable_arrays, native_arrays, observations = [], [], [], []
    for i in range(contract["n_crops"]):
        metadata = _json(out / f"native_crop_{i:03d}.json")
        native = _load_arrays(out / f"native_crop_{i:03d}.npz")
        derived = _load_arrays(out / f"crop_{i:03d}.npz")
        if set(native) != ARRAY_KEYS or set(derived) != ARRAY_KEYS:
            raise ValueError("Unexpected tensor membership")
        if (metadata["state_before"] != metadata["state_after"] or metadata["tokens_before"] != metadata["tokens_after"]):
            raise ValueError("Frozen native controls differ")
        for key in ARRAY_KEYS - set(RESULT_KEYS):
            if not np.array_equal(native[key], derived[key]):
                raise ValueError("Frozen probability/token inputs changed")
        original_cpu = relation_from_arrays(relation, native)
        for key in RESULT_KEYS:
            comparison, expected = original_cpu[key], native[key]
            if key.endswith("_geo"):
                keep = [k for k in range(15) if k != 4]
                comparison, expected = comparison[..., keep], expected[..., keep]
            if not np.allclose(comparison, expected, rtol=RTOL, atol=ATOL):
                raise ValueError("Native replay mismatch is outside distance feature")
        observations.append(_differences(original_cpu, native))
        actual = relation_from_arrays(relation, native, distance_backend=STABLE_ANALYSIS)
        for key in RESULT_KEYS:
            np.testing.assert_allclose(actual[key], derived[key], rtol=RTOL, atol=ATOL)
        # Both algorithms share all feature definitions except nearest distance.
        for arm in ("coarse", "dense"):
            keep = [k for k in range(15) if k != 4]
            np.testing.assert_array_equal(original_cpu[arm + "_geo"][..., keep], actual[arm + "_geo"][..., keep])
        records.append(metadata)
        stable_arrays.append(derived)
        native_arrays.append(native)
    failed = {key: sum(r[key]["failed_elements"] for r in observations) for key in RESULT_KEYS}
    if not any(failed.values()):
        raise ValueError("Expected native replay failure was not reproduced")
    report = _aggregate(records, stable_arrays, contract["provenance"])
    native_report = _aggregate(records, native_arrays, native_contract["provenance"])
    native_delta = {}
    for key in RESULT_KEYS:
        delta = np.concatenate([np.abs(s[key] - n[key]) for s, n in zip(stable_arrays, native_arrays)])
        native_delta[key] = dict(max_abs=float(delta.max()), mean_abs=float(delta.mean()))
    report.update(analysis_version=STABLE_ANALYSIS, native_replay_passed=False, derived_replay_passed=True,
                  original_native_complete_present=False, native_replay_failed_elements=failed,
                  native_replay_max_abs={k: max(r[k]["max_abs"] for r in observations) for k in RESULT_KEYS},
                  stable_minus_native=native_delta, native_r_contrast=native_report["r"],
                  native_failure_scope="Only nearest-host distance feature4; other geometry and native logits pass original tolerance",
                  source_commitment_sha256=contract["source_commitment_sha256"],
                  method="CPU analysis only: original float32 centroids, float64 explicit difference norms, original head dtype",
                  limitation="Derived success does not change the original native replay FAIL; no new image forward or quality verdict")
    report["r"]["coarse_control_scope"] = "Original native capture control against model.bind; derived CPU head is separately reported"
    return report


def _stable_names(n_crops):
    return ({"contract.json", "collection_snapshot_manifest.json", "relation_state.npz", "native_contract.json",
             "native_selection_private.json", "aggregate_anonymous.json", "RUN.json", "COMPLETE.json"}
            | {f"native_crop_{i:03d}.{ext}" for i in range(n_crops) for ext in ("json", "npz")}
            | {f"crop_{i:03d}.npz" for i in range(n_crops)})


def derive_stable_analysis(source, out, snapshot):
    """Derive a new private CPU contract while preserving every native byte."""
    import resource
    started = time.monotonic()
    source, out, snapshot = Path(source), Path(out), Path(snapshot)
    native_contract, manifest = _verified_native_source(source, snapshot)
    if out.exists() or out.resolve().is_relative_to(source.resolve()):
        raise ValueError("Expected fresh independent derived directory")
    repo = Path(__file__).resolve().parents[2]
    code = ("anatobind/aur/ar_diagnostics.py", "scripts/aur_diagnose_r_geometry.py",
            "anatobind/aur/relation.py", "docs/superpowers/specs/2026-10-19-aur-ar-stable-distance-analysis.md")
    analysis_hashes = {name: file_digest(repo / name) for name in code}
    source_commitment = hashlib.sha256(json.dumps(manifest["native_member_sha256"], sort_keys=True).encode()).hexdigest()
    provenance = dict(native_contract["provenance"], analysis_source_sha256=analysis_hashes)
    contract = dict(schema=2, analysis_version=STABLE_ANALYSIS, native_replay_passed=False,
                    tolerance=dict(rtol=RTOL, atol=ATOL), n_crops=native_contract["n_crops"],
                    relation_config=native_contract["relation_config"], provenance=provenance,
                    native_member_sha256=manifest["native_member_sha256"], source_commitment_sha256=source_commitment,
                    collection_snapshot_sha256=file_digest(snapshot / "snapshot_manifest.json"))
    out.mkdir(mode=0o700, parents=True, exist_ok=False)
    _write_json(out / "contract.json", contract)
    _write_bytes(out / "collection_snapshot_manifest.json", (snapshot / "snapshot_manifest.json").read_bytes())
    for name in manifest["native_member_sha256"]:
        _write_bytes(out / _native_copy_name(name), (source / name).read_bytes())
    relation = _frozen_relation(out, contract["relation_config"])
    for i in range(contract["n_crops"]):
        native = _load_arrays(out / f"native_crop_{i:03d}.npz")
        actual = relation_from_arrays(relation, native, distance_backend=STABLE_ANALYSIS)
        derived = dict(native, **actual)
        _write_npz(out / f"crop_{i:03d}.npz", derived)
    report = _rebuild_stable(out)
    # Do not complete against a source or method that changed mid-analysis.
    _verified_native_source(source, snapshot)
    if analysis_hashes != {name: file_digest(repo / name) for name in code}:
        raise ValueError("Stable analysis source changed")
    _write_json(out / "aggregate_anonymous.json", report)
    _write_json(out / "RUN.json", dict(schema=2, analysis_version=STABLE_ANALYSIS, elapsed_seconds=time.monotonic() - started,
                                       rss_max_kib=int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss),
                                       source_commitment_sha256=source_commitment, native_members_unchanged=True,
                                       native_replay_passed=False, derived_replay_passed=True, cuda_used=False,
                                       images_opened=False, checkpoints_opened=False, backbone_forwards=0,
                                       torch=str(torch.__version__), numpy=np.__version__, tolerance=dict(rtol=RTOL, atol=ATOL)))
    _write_json(out / "COMPLETE.json", dict(schema=2, files=_hashes(out)))
    return replay_stable_analysis(out)


def replay_stable_analysis(out):
    try:
        out = Path(out)
        contract = _json(out / "contract.json")
        if out.is_symlink() or out.stat().st_mode & 0o777 != 0o700 or {p.name for p in out.iterdir()} != _stable_names(contract["n_crops"]):
            raise ValueError("Invalid private stable analysis membership")
        if any(p.is_symlink() or p.stat().st_mode & 0o777 != 0o600 for p in out.iterdir()):
            raise ValueError("Invalid private stable analysis permissions")
        if _json(out / "COMPLETE.json") != dict(schema=2, files=_hashes(out)):
            raise ValueError("Stable analysis checksums differ")
        report = _rebuild_stable(out)
        if report != _json(out / "aggregate_anonymous.json"):
            raise ValueError("Stable analysis aggregate differs")
        return report
    except Exception:
        raise ValueError("Invalid stable A/R analysis") from None


class _SafeParser(argparse.ArgumentParser):
    def error(self, message):
        self.exit(2, "A/R diagnostic incomplete (invalid_arguments)\n")


def parser(replay=False):
    p = _SafeParser(description="New frozen TRAIN A32/R geometry diagnostic; float32, NOT_EVIDENCE")
    p.add_argument("--out", type=Path, required=True, help="Fresh private collection directory, or completed cache for replay")
    if replay:
        p.add_argument("--derive-from", type=Path, help="Original incomplete native cache; derive a separate stable CPU analysis")
        p.add_argument("--source-snapshot", type=Path, help="Verified collection-time source snapshot; required with --derive-from")
    else:
        p.add_argument("--samples", type=Path, required=True)
        p.add_argument("--val-patients", type=Path, required=True)
        p.add_argument("--checkpoint", type=Path, required=True, help="Frozen legacy C2 Stage III export")
        p.add_argument("--per-source", type=int, default=1)
        p.add_argument("--crop", type=int, nargs=3, default=(8, 32, 32), help="New crop contract; default 8 32 32, standard 128 160 160 is explicit")
        p.add_argument("--points", type=int, default=16384)
        p.add_argument("--query-chunk", type=int, default=4)
        p.add_argument("--seed", type=int, default=SEED)
        p.add_argument("--device", default="cpu", help="cpu or a single visible CUDA device such as cuda:0")
        p.add_argument("--max-seconds", type=float, default=1800)
    return p


def _private_output(path):
    repo = Path(__file__).resolve().parents[2]
    output, home = Path(path).resolve(), Path.home().resolve()
    # Home itself has a personal configuration Git repository on this host;
    # reject project checkouts below home and this worktree explicitly.
    project_ancestors = [p for p in (output, *output.parents) if p != home and p.is_relative_to(home)]
    if (not output.is_relative_to(home) or output.is_relative_to(repo)
            or any((p / ".git").exists() for p in project_ancestors)):
        raise ValueError("Private output must be under home and outside project Git checkouts")
    return output


def run_collection(args):
    """The only entry that reads medical arrays or forwards the full model."""
    repo = Path(__file__).resolve().parents[2]
    args.out = _private_output(args.out)
    from anatobind.aur.dataset import AURDataset
    from anatobind.aur.infer import model_from_export
    if args.out.exists() or args.points < 1 or args.query_chunk < 1 or not 0 < args.max_seconds <= 1800:
        raise ValueError("Invalid new diagnostic request")
    if any(c < 1 or c % p for c, p in zip(args.crop, (2, 4, 4))):
        raise ValueError("Invalid 3D crop")
    started = time.monotonic()
    frozen_files = {name: file_digest(getattr(args, name)) for name in ("samples", "val_patients", "checkpoint")}
    torch.set_num_threads(2)
    torch.manual_seed(args.seed)
    rows, val = _json(args.samples), _json(args.val_patients)
    train, selection = select_train_rows(rows, val, args.per_source, args.seed)
    root = Path("/data2/congcong/data/FM_data").resolve()
    inputs = {}
    for i in selection:
        for field in ("image", "anatomy", "lesion"):
            path = train[i].get(field)
            if path:
                real = Path(path).resolve(strict=True)
                if not real.is_relative_to(root):
                    raise ValueError("Input escapes approved data root")
                inputs[str(real)] = file_digest(real)
    model, meta = model_from_export(args.checkpoint, torch.device(args.device))
    model.eval().requires_grad_(False)
    cfg = meta.get("config", {})
    if (meta.get("stage", cfg.get("stage")) != "III" or meta.get("step") != 3000 or meta.get("seen_crops") != 72000
            or cfg.get("u_mask_profile") != "unmatched_empty_v1" or model.cfg.get("dual_input", False)
            or state_digest(dict(list(model.named_parameters()) + list(model.named_buffers()))) != C2_STATE_SHA):
        raise ValueError("Expected frozen legacy C2 export")
    code = ("anatobind/aur/ar_diagnostics.py", "scripts/aur_diagnose_a_queries.py", "scripts/aur_diagnose_r_geometry.py",
            "anatobind/aur/model.py", "anatobind/aur/heads.py", "anatobind/aur/swin.py", "anatobind/aur/rope.py",
            "anatobind/aur/crops.py", "anatobind/aur/dataset.py", "anatobind/aur/targets.py", "anatobind/aur/labels.py",
            "anatobind/aur/losses.py", "anatobind/aur/relation.py", "anatobind/aur/infer.py", "anatobind/aur/train.py",
            "anatobind/eval/lesion_components.py", "anatobind/anatomy/simulate.py", "anatobind/aur/unified.py")
    source_hashes = {name: file_digest(repo / name) for name in code}
    provenance = dict(checkpoint_sha256=frozen_files["checkpoint"], samples_sha256=frozen_files["samples"],
                      val_patients_sha256=frozen_files["val_patients"], source_sha256=source_hashes,
                      selection_sha256=hashlib.sha256(json.dumps(selection).encode()).hexdigest(),
                      model_state_sha256=C2_STATE_SHA,
                      code_sha=subprocess.run(["git", "rev-parse", "HEAD"], cwd=repo, capture_output=True, text=True, check=True).stdout.strip())
    recipe = dict(crop=list(args.crop), seed=args.seed, points=args.points, per_source=args.per_source,
                  crops_per_volume=1, do_augment=False, lesion_share=1.0, epoch=0, query_chunk=args.query_chunk,
                  query_selection="all queries, fixed before forward", dtype="float32",
                  pooling="sigmoid then patch-aligned average, then original F1 valid; no invalid renormalization",
                  selection="new TRAIN contract; possible overlap with earlier TRAIN patients", rtol=RTOL, atol=ATOL)
    contract = dict(schema=1, n_crops=len(selection), provenance=provenance, recipe=recipe,
                    relation_config=dict(d_model=model.cfg["d_model"], layers=model.cfg["rel_layers"], heads=model.cfg["rel_heads"]))
    private = dict(selected_indices=selection, rows=[train[i] for i in selection], input_sha256=inputs)
    create_archive(args.out, contract, private, model.relation)
    ds = AURDataset(train, crop=tuple(args.crop), crops_per_volume=1, do_augment=False, lesion_share=1.0, seed=args.seed)
    for ordinal, index in enumerate(selection):
        if time.monotonic() - started > args.max_seconds:
            raise TimeoutError("Diagnostic deadline exceeded")
        metadata, arrays = collect_crop(model, ds[index][0], points=args.points, seed=args.seed + ordinal, query_chunk=args.query_chunk)
        metadata["source"] = train[index]["source"]
        write_crop(args.out, ordinal, metadata, arrays)
        print(f"Completed frozen A/R crops {ordinal + 1}/{len(selection)}", flush=True)
    if (inputs != {name: file_digest(name) for name in inputs}
            or source_hashes != {name: file_digest(repo / name) for name in code}
            or frozen_files != {name: file_digest(getattr(args, name)) for name in frozen_files}):
        raise ValueError("Input or source changed")
    if time.monotonic() - started > args.max_seconds:
        raise TimeoutError("Diagnostic deadline exceeded")
    complete_archive(args.out)
    print("Frozen TRAIN A/R diagnostic complete; NOT_EVIDENCE", flush=True)
    return 0


def main(argv=None, *, replay=False):
    args = parser(replay).parse_args(argv)
    try:
        if replay:
            torch.set_num_threads(2)
            if args.derive_from:
                if not args.source_snapshot:
                    raise ValueError("Expected collection-time source snapshot")
                derive_stable_analysis(args.derive_from, _private_output(args.out), args.source_snapshot)
                print("Stable CPU A/R analysis complete; original native replay remains FAIL; NOT_EVIDENCE", flush=True)
            else:
                if args.source_snapshot:
                    raise ValueError("Source snapshot applies only to derivation")
                if _json(args.out / "contract.json").get("analysis_version") == STABLE_ANALYSIS:
                    replay_stable_analysis(args.out)
                    print("Stable CPU A/R replay verified; original native replay remains FAIL; NOT_EVIDENCE", flush=True)
                else:
                    replay_archive(args.out)
                    print("Complete CPU A/R replay verified; NOT_EVIDENCE", flush=True)
            return 0
        return run_collection(args)
    except Exception as exc:
        print(f"A/R diagnostic incomplete ({type(exc).__name__}); no completed result accepted", flush=True)
        return 1
