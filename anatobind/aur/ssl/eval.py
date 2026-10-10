"""Frozen-representation probes for gate G1 (SSL-first plan §3.3, T09; decision Q8 of 2026-10-09).

Two probes read a frozen backbone, once for the Stage I weights and once for a random initialisation of the same
configuration, on the same crops:

1. The host readout: a linear 1 x 1 x 1 classifier over the F2-grid features (F2 with F3 and F4 upsampled to it) that
   predicts the majority host class of each cell (13 sided hosts + none), trained on calibration crops of train patients
   and scored on crops of the validation patients by the Dice of every host and the macro mean over the 13 hosts.
2. The lesion separability: a logistic regression over pooled cell features that tells a lesion instance from a
   size-matched normal region of the same host class in the same crop, trained on the calibration crops, scored by the
   AUC on the validation crops, with the lesions stratified by equivalent diameter (< 5 mm, 5-10 mm, > 10 mm).

The frozen features never change with the probe seed; the seeds vary the probe heads, and a patient-level bootstrap
over the validation crops gives the interval of the difference SSL - random. Gate G1 (PROPOSED thresholds): macro Dice
of the SSL backbone at least 0.05 above the random one with the interval of the difference above zero, and the lesion
AUC not below the random one minus 0.02. Everything here is scored against SynthSeg-derived pseudo-labels:
NOT_EVIDENCE for anatomy, an engineering gate only."""
import json
import math
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from anatobind.aur.dataset import load_volume, make_crop
from anatobind.aur.labels import N_HOSTS, entity_to_synthseg, host_map
from anatobind.aur.ssl import checkpoint as CK
from anatobind.aur.swin import SwinBackbone
from anatobind.eval.lesion_components import equivalent_diameter_mm

DICE_GAIN = 0.05            # PROPOSED (decision Q8)
AUC_TOLERANCE = 0.02        # PROPOSED (decision Q8)
SIZE_BINS = ((0.0, 5.0), (5.0, 10.0), (10.0, float("inf")))


def select_rows(rows, val_patients, n_calibration, n_validation, seed, lesion_only=False):
    """(calibration rows from the train patients, validation rows from the validation patients), without replacement,
    one row per volume. With lesion_only, only rows whose U is supervised (the lesion probe)."""
    rng = np.random.default_rng(seed)
    val = {(s, p) for s, ps in val_patients.items() for p in ps}
    pool = [r for r in rows if r["split"] == "train" and (not lesion_only or r["u_supervised"])]
    cal = [r for r in pool if (r["source"], r["patient"]) not in val]
    vl = [r for r in pool if (r["source"], r["patient"]) in val]
    pick = lambda xs, n: [xs[i] for i in rng.choice(len(xs), size=min(n, len(xs)), replace=False)] if xs else []
    return pick(cal, n_calibration), pick(vl, n_validation)


@torch.no_grad()
def cell_features(backbone, crop, device):
    """One crop dict -> (C, D2, H2, W2) float16 features on the F2 grid (F2, F3 and F4 upsampled), the F2 validity."""
    levels = backbone(crop["image"][None].to(device), crop["valid"][None].to(device), crop["coords"][None].to(device), crop["local"][None].to(device))
    f2 = levels[1]["feat"]
    parts = [f2] + [F.interpolate(levels[k]["feat"].float(), size=f2.shape[2:], mode="trilinear", align_corners=False) for k in (2, 3)]
    return torch.cat(parts, 1)[0].half().cpu(), levels[1]["valid"][0].cpu()


def host_cells(crop, grid, patch_stride):
    """The majority host class of every F2 cell (0 none, 1..13 hosts) and the share of lesion voxels per cell."""
    seg = entity_to_synthseg(crop["entity"].numpy())
    hosts = torch.from_numpy(host_map(seg).astype(np.int64))                         # (D, H, W) 0..13
    one_hot = F.one_hot(hosts, N_HOSTS + 1).permute(3, 0, 1, 2).float()[None]        # (1, 14, D, H, W)
    pooled = F.avg_pool3d(one_hot, kernel_size=patch_stride, stride=patch_stride)[0]  # (14, D2, H2, W2)
    lesion = F.avg_pool3d((crop["instance"] > 0).float()[None, None], kernel_size=patch_stride, stride=patch_stride)[0, 0]
    return pooled.argmax(0), lesion


def stride_of(crop_shape, grid):
    return tuple(int(s // g) for s, g in zip(crop_shape, grid))


PROBE_KEYS = ("image", "valid", "coords", "local", "entity", "instance")


class _ProbeCrops(torch.utils.data.Dataset):
    """One crop per row, as gather_probe_data draws it (rng [rng_seed, i]); only the arrays the probes read."""

    def __init__(self, rows, crop, rng_seed, lesion):
        self.rows, self.crop, self.rng_seed, self.lesion = list(rows), tuple(crop), rng_seed, lesion

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        vol = load_volume(self.rows[i])
        c = make_crop(vol, np.random.default_rng([self.rng_seed, i]), crop=self.crop, do_augment=False, lesion_centred=self.lesion)
        return {**{k: c[k] for k in PROBE_KEYS}, "voxel_mm3": vol["voxel_mm3"]}


def _first(items):
    return items[0]


def probe_crops(rows, crop, rng_seed, lesion=False, workers=0):
    """The crops of gather_probe_data, built once (in `workers` loader processes, in order) so that both arms can read
    the same ones."""
    ds = _ProbeCrops(rows, crop, rng_seed, lesion)
    if workers > 0 and len(ds) > 1:
        return list(torch.utils.data.DataLoader(ds, batch_size=1, shuffle=False, num_workers=min(workers, len(ds)), collate_fn=_first))
    return [ds[i] for i in range(len(ds))]


def gather_probe_data(backbone, rows, device, crop, rng_seed, lesion=False, crops=None):
    """Frozen features and targets of one crop per row (no augmentation; lesion-centred when `lesion`); `crops` (from
    probe_crops with the same rows, seed and flag) skips the loading."""
    feats, targets, valids, lesions, meta = [], [], [], [], []
    for i, r in enumerate(rows):
        if crops is not None:
            c = crops[i]
            vol = {"voxel_mm3": c["voxel_mm3"]}
        else:
            vol = load_volume(r)
            c = make_crop(vol, np.random.default_rng([rng_seed, i]), crop=crop, do_augment=False, lesion_centred=lesion)
        f, v = cell_features(backbone, c, device)
        grid = f.shape[1:]
        hosts, les = host_cells(c, grid, stride_of(crop, grid))
        feats.append(f)
        targets.append(hosts)
        valids.append(v > 0.5)
        lesions.append(les)
        meta.append({"case": r["case"], "source": r["source"], "patient": r["patient"], "voxel_mm3": vol["voxel_mm3"],
                     "instances": c["instance"], "entity": c["entity"]})
    return feats, targets, valids, lesions, meta


def train_readout(feats, targets, valids, seed, device, epochs=300, lr=1e-2):
    """A linear classifier over the cells (features standardised by the calibration statistics)."""
    x = torch.cat([f.permute(1, 2, 3, 0)[v] for f, v in zip(feats, valids)]).float()
    y = torch.cat([t[v] for t, v in zip(targets, valids)])
    mean, std = x.mean(0, keepdim=True), x.std(0, keepdim=True).clamp(min=1e-3)
    x = ((x - mean) / std).to(device)
    y = y.to(device)
    g = torch.Generator(device="cpu").manual_seed(seed)
    w = torch.nn.Linear(x.shape[1], N_HOSTS + 1).to(device)
    with torch.no_grad():
        w.weight.copy_(torch.randn(w.weight.shape, generator=g) * 0.01)
        w.bias.zero_()
    opt = torch.optim.Adam(w.parameters(), lr=lr, weight_decay=1e-4)
    for _ in range(epochs):
        opt.zero_grad()
        loss = F.cross_entropy(w(x), y)
        loss.backward()
        opt.step()
    return {"linear": w.cpu(), "mean": mean, "std": std}


@torch.no_grad()
def score_readout(head, feats, targets, valids):
    """Per-crop confusion counts -> per-host Dice (13) and the macro mean; crops returned separately for the bootstrap."""
    per_crop = []
    for f, t, v in zip(feats, targets, valids):
        x = ((f.permute(1, 2, 3, 0)[v].float() - head["mean"]) / head["std"])
        pred = head["linear"](x).argmax(1)
        tgt = t[v]
        inter = torch.stack([((pred == k) & (tgt == k)).sum() for k in range(1, N_HOSTS + 1)]).float()
        ps = torch.stack([(pred == k).sum() for k in range(1, N_HOSTS + 1)]).float()
        ts = torch.stack([(tgt == k).sum() for k in range(1, N_HOSTS + 1)]).float()
        per_crop.append((inter, ps, ts))
    return per_crop


def dice_from_counts(per_crop):
    inter = sum(c[0] for c in per_crop)
    ps = sum(c[1] for c in per_crop)
    ts = sum(c[2] for c in per_crop)
    present = ts > 0
    dice = torch.where(present, 2 * inter / (ps + ts).clamp(min=1.0), torch.full_like(inter, float("nan")))
    return dice, float(dice[present].mean()) if present.any() else float("nan")


def roi_pairs(feats, lesions, targets, valids, meta, rng):
    """(X, y, diameters): for every lesion instance of a crop, the mean feature over its cells (label 1) and over as
    many random lesion-free cells of its majority host class (label 0); the diameter from the instance's voxel count."""
    xs, ys, diam = [], [], []
    for f, les, t, v, m in zip(feats, lesions, targets, valids, meta):
        inst = m["instances"]
        stride = stride_of(inst.shape, f.shape[1:])
        for k in range(1, int(inst.max()) + 1):
            cells = F.max_pool3d((inst == k).float()[None, None], kernel_size=stride, stride=stride)[0, 0] > 0
            cells &= v
            if not cells.any():
                continue
            host = int(torch.mode(t[cells]).values)
            normal = (t == host) & v & (les == 0) & ~cells
            n = int(cells.sum())
            if int(normal.sum()) < n:
                continue
            idx = torch.nonzero(normal.flatten()).flatten()
            take = idx[torch.as_tensor(rng.choice(len(idx), size=n, replace=False))]
            flat = f.permute(1, 2, 3, 0).reshape(-1, f.shape[0]).float()
            xs += [flat[cells.flatten()].mean(0), flat[take].mean(0)]
            ys += [1, 0]
            d = equivalent_diameter_mm(float((inst == k).sum()) * m["voxel_mm3"])
            diam += [d, d]
    if not xs:
        return torch.zeros(0, feats[0].shape[0]), torch.zeros(0, dtype=torch.long), np.zeros(0)
    return torch.stack(xs), torch.tensor(ys), np.array(diam)


def auc(scores, labels):
    """The rank AUC (ties at half)."""
    s, y = np.asarray(scores, float), np.asarray(labels)
    pos, neg = s[y == 1], s[y == 0]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    order = np.argsort(np.concatenate([pos, neg]), kind="mergesort")
    ranks = np.empty(len(order), float)
    ranks[order] = np.arange(1, len(order) + 1)
    # average ranks for ties
    vals = np.concatenate([pos, neg])
    for u in np.unique(vals):
        sel = vals == u
        if sel.sum() > 1:
            ranks[sel] = ranks[sel].mean()
    return float((ranks[:len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


def lesion_probe(cal, val, seed):
    """Logistic regression on the calibration pairs, AUC on the validation pairs, overall and per size bin."""
    from sklearn.linear_model import LogisticRegression
    xc, yc, _ = cal
    xv, yv, dv = val
    if len(yc) < 4 or len(yv) < 2 or len(set(yc.tolist())) < 2:
        return {"auc": float("nan"), "n_train_pairs": int(len(yc)), "n_val_pairs": int(len(yv)), "by_size": {}}
    mean, std = xc.mean(0, keepdim=True), xc.std(0, keepdim=True).clamp(min=1e-3)
    clf = LogisticRegression(C=1.0, max_iter=2000, random_state=seed).fit(((xc - mean) / std).numpy(), yc.numpy())
    scores = clf.decision_function(((xv - mean) / std).numpy())
    out = {"auc": auc(scores, yv.numpy()), "n_train_pairs": int(len(yc)), "n_val_pairs": int(len(yv)), "by_size": {}}
    for lo, hi in SIZE_BINS:
        sel = (dv >= lo) & (dv < hi)
        out["by_size"][f"{lo:g}-{hi:g}mm"] = {"auc": auc(scores[sel], yv.numpy()[sel]) if sel.sum() else float("nan"), "n_pairs": int(sel.sum())}
    return out


def bootstrap_difference(per_crop_a, per_crop_b, patients, n_boot=1000, seed=0):
    """The macro-Dice difference A - B resampled over patients (crops of one patient move together)."""
    rng = np.random.default_rng(seed)
    pts = sorted(set(patients))
    idx = {p: [i for i, q in enumerate(patients) if q == p] for p in pts}
    diffs = []
    for _ in range(n_boot):
        chosen = rng.choice(len(pts), size=len(pts), replace=True)
        sel = [i for c in chosen for i in idx[pts[c]]]
        diffs.append(dice_from_counts([per_crop_a[i] for i in sel])[1] - dice_from_counts([per_crop_b[i] for i in sel])[1])
    diffs = np.array(diffs, float)
    diffs = diffs[np.isfinite(diffs)]
    return {"mean": float(diffs.mean()) if diffs.size else float("nan"), "ci95": [float(np.percentile(diffs, 2.5)), float(np.percentile(diffs, 97.5))] if diffs.size else [float("nan")] * 2}


def load_backbones(checkpoint, device, random_seed=0, **kwargs):
    """{"ssl": the Stage I backbone, "random": a fresh one of the same configuration}, both frozen in eval mode."""
    ssl = SwinBackbone(**kwargs)
    meta = CK.load_backbone(checkpoint, ssl) if checkpoint is not None else None
    torch.manual_seed(random_seed)
    rnd = SwinBackbone(**kwargs)
    out = {"ssl": ssl, "random": rnd}
    for bb in out.values():
        bb.to(device).eval()
        for p in bb.parameters():
            p.requires_grad_(False)
    return out, meta


def run(rows, val_patients, checkpoint, out_dir, device, crop, n_calibration=96, n_validation=48, n_lesion=48, seeds=(0, 1, 2), backbone_kwargs=None,
        lesion=True, share_crops=True, workers=0):
    """The G1 gate (lesion=True), or a host-only probe of a running Stage I (lesion=False: no lesion arm, no verdict;
    used to decide early whether a run is heading the right way, 2026-10-10)."""
    out_dir = Path(out_dir)
    if out_dir.exists():
        raise FileExistsError(f"{out_dir} exists")
    out_dir.mkdir(parents=True)
    backbones, meta = load_backbones(checkpoint, device, **(backbone_kwargs or {}))
    cal_rows, val_rows = select_rows(rows, val_patients, n_calibration, n_validation, seed=11)
    lcal_rows, lval_rows = select_rows(rows, val_patients, n_lesion, n_lesion, seed=12, lesion_only=True) if lesion else ([], [])
    report = {"mode": "gate" if lesion else "host-only probe", "checkpoint": None if checkpoint is None else str(checkpoint), "checkpoint_meta": meta, "crop": list(crop),
              "n_calibration_crops": len(cal_rows), "n_validation_crops": len(val_rows), "n_lesion_calibration": len(lcal_rows), "n_lesion_validation": len(lval_rows),
              "seeds": list(seeds), "thresholds": {"dice_gain": DICE_GAIN, "auc_tolerance": AUC_TOLERANCE, "status": "PROPOSED"}, "arms": {}}
    per_crop = {}
    shared = {}
    if share_crops:                                                   # one load per crop for both arms (identical crops)
        shared["cal"] = probe_crops(cal_rows, crop, 21, False, workers)
        shared["val"] = probe_crops(val_rows, crop, 22, False, workers)
        if lesion:
            shared["lcal"] = probe_crops(lcal_rows, crop, 23, True, workers)
            shared["lval"] = probe_crops(lval_rows, crop, 24, True, workers)
    for name, bb in backbones.items():
        cal = gather_probe_data(bb, cal_rows, device, crop, rng_seed=21, crops=shared.get("cal"))
        val = gather_probe_data(bb, val_rows, device, crop, rng_seed=22, crops=shared.get("val"))
        arm = {"host": {"seeds": []}, "lesion": {"seeds": []}}
        for s in seeds:
            head = train_readout(cal[0], cal[1], cal[2], s, device)
            pc = score_readout(head, val[0], val[1], val[2])
            dice, macro = dice_from_counts(pc)
            arm["host"]["seeds"].append({"seed": s, "macro_dice": macro, "per_host_dice": [None if math.isnan(float(d)) else float(d) for d in dice]})
            per_crop[(name, s)] = pc
        arm["host"]["macro_dice_mean"] = float(np.nanmean([x["macro_dice"] for x in arm["host"]["seeds"]]))
        if not lesion:
            arm["lesion"] = None
            report["arms"][name] = arm
            continue
        lcal = gather_probe_data(bb, lcal_rows, device, crop, rng_seed=23, lesion=True, crops=shared.get("lcal"))
        lval = gather_probe_data(bb, lval_rows, device, crop, rng_seed=24, lesion=True, crops=shared.get("lval"))
        for s in seeds:
            rng = np.random.default_rng(s)
            res = lesion_probe(roi_pairs(lcal[0], lcal[3], lcal[1], lcal[2], lcal[4], rng), roi_pairs(lval[0], lval[3], lval[1], lval[2], lval[4], rng), s)
            arm["lesion"]["seeds"].append({"seed": s, **res})
        aucs = [x["auc"] for x in arm["lesion"]["seeds"] if not math.isnan(x["auc"])]
        arm["lesion"]["auc_mean"] = float(np.mean(aucs)) if aucs else float("nan")
        report["arms"][name] = arm
        if name == "ssl":
            report["val_patients_used"] = len({r["patient"] for r in val_rows})
    if "val_patients_used" not in report:
        report["val_patients_used"] = len({r["patient"] for r in val_rows})
    patients = [r["patient"] for r in val_rows]
    diffs = [bootstrap_difference(per_crop[("ssl", s)], per_crop[("random", s)], patients, seed=s) for s in seeds]
    gain = report["arms"]["ssl"]["host"]["macro_dice_mean"] - report["arms"]["random"]["host"]["macro_dice_mean"]
    lo = min(d["ci95"][0] for d in diffs)
    if not lesion:
        report["g1"] = {"macro_dice_gain": gain, "gain_bootstrap_by_seed": diffs, "gain_ci_low_min": lo,
                        "host_macro_ssl": report["arms"]["ssl"]["host"]["macro_dice_mean"],
                        "host_macro_random": report["arms"]["random"]["host"]["macro_dice_mean"],
                        "host_pass": None, "lesion_pass": None, "pass": None,
                        "note": "host-only probe of a running Stage I: not the G1 verdict",
                        "evidence": "NOT_EVIDENCE: SynthSeg-derived pseudo-labels; engineering probe only"}
        (out_dir / "g1_report.json").write_text(json.dumps(report, indent=1, default=str))
        return report
    auc_ssl, auc_rnd = report["arms"]["ssl"]["lesion"]["auc_mean"], report["arms"]["random"]["lesion"]["auc_mean"]
    report["g1"] = {"macro_dice_gain": gain, "gain_bootstrap_by_seed": diffs, "gain_ci_low_min": lo,
                    "host_pass": bool(gain >= DICE_GAIN and lo > 0.0),
                    "lesion_auc_ssl": auc_ssl, "lesion_auc_random": auc_rnd,
                    "lesion_pass": bool(math.isnan(auc_rnd) or math.isnan(auc_ssl) or auc_ssl >= auc_rnd - AUC_TOLERANCE),
                    "evidence": "NOT_EVIDENCE: SynthSeg-derived pseudo-labels; engineering gate only"}
    report["g1"]["pass"] = report["g1"]["host_pass"] and report["g1"]["lesion_pass"]
    (out_dir / "g1_report.json").write_text(json.dumps(report, indent=1, default=str))
    return report
