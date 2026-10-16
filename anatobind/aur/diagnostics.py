"""Observational validation diagnostics; never a replacement for A/U/R gates."""
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from anatobind.aur import infer as I

GROUPS = (('bmsr', 'T1c'), ('isles', 'DWI'), ('pdgm', 'FLAIR'))


def identity(row):
    return tuple(str(row[k]) for k in ('source', 'patient', 'case', 'sequence'))


def selection_digest(row):
    return hashlib.sha256(json.dumps([0, *identity(row)], ensure_ascii=False, separators=(',', ':')).encode()).hexdigest()


def select_cases(val_rows, saved_cases, per_source=2):
    """Caller supplies the frozen validation rows; reject test rows, duplicates and absent saved truths."""
    if any(r.get('split') == 'test' for r in val_rows):
        raise ValueError('test membership in validation selection')
    keys = [identity(r) for r in val_rows]
    skeys = [identity(r) for r in saved_cases]
    if len(keys) != len(set(keys)) or len(skeys) != len(set(skeys)):
        raise ValueError('duplicate row identities')
    saved = dict(zip(skeys, saved_cases))
    if not set(keys).issubset(saved):
        raise ValueError('validation rows missing from saved evaluation')
    selected = []
    for source, sequence in GROUPS:
        pool = [r for r in val_rows if (r['source'], r['sequence']) == (source, sequence) and r['u_supervised']
                and any(not g.get('ignore') for g in (saved[identity(r)].get('u') or {}).get('gt', []))]
        seen = set()
        for r in sorted(pool, key=lambda r: (selection_digest(r), identity(r))):
            if r['patient'] in seen:
                continue
            selected.append(r)
            seen.add(r['patient'])
            if len(seen) == per_source:
                break
        if len(seen) != per_source:
            raise ValueError('insufficient distinct validation patients')
    return selected


def verify_detections(actual, expected):
    """Fail closed: exact components/boxes/counts, previously frozen volume/score tolerances."""
    key = lambda r: (tuple(r['box']), r['n_voxels'])
    a, b = sorted(actual, key=key), sorted(expected, key=key)
    if len(a) != len(b):
        raise ValueError('baseline reproduction: component count mismatch')
    for x, y in zip(a, b):
        for k in ('box', 'n_voxels'):
            if x[k] != y[k]:
                raise ValueError('baseline reproduction: component geometry mismatch')
        for k, atol in (('volume_mm3', 1e-3), ('score', 1e-5)):
            if not np.isfinite(x[k]) or not np.isfinite(y[k]) or abs(x[k] - y[k]) > atol:
                raise ValueError('baseline reproduction: numeric mismatch')


class WindowAudit:
    """Stream measurements and representative query arrays; reconstruct the original window mean independently.

    All per-query statistics are float32-derived. Saved representative arrays use float16 and are labelled as such;
    they are visualization/debug caches, not used for reproduction or threshold decisions.
    """
    def __init__(self, instance, small, out_dir):
        self.instance, self.small = np.asarray(instance), np.asarray(small)
        self.out_dir = Path(out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.window_rows = []
        self.acc = np.zeros(instance.shape, np.float32)
        self.wsum = np.zeros(instance.shape, np.float32)
        self.total = dict(zip(*np.unique(instance, return_counts=True)))

    def __call__(self, *, window, event_prob, event_presence, entity_prob, valid, weight):
        device = event_prob.device
        local, _ = I.extract(self.instance, window, fill=0)
        small, _ = I.extract(self.small, window, fill=False)
        gt = torch.as_tensor(local, device=device)
        ok = valid > 0
        ignored = torch.as_tensor(small, device=device) & ok
        healthy = (gt == 0) & ok & ~ignored
        keep = event_presence > I.PRESENCE_THRESHOLD
        maximum = event_prob[keep].amax(0) if keep.any() else torch.zeros_like(valid)
        dst, src = I._inside(window, self.instance.shape)
        wb = (weight * valid).cpu().numpy()
        max_np = maximum.cpu().numpy()
        self.acc[dst] += max_np[src] * wb[src]
        self.wsum[dst] += wb[src]
        row = dict(window=[list(w) for w in window], n_present=int(keep.sum()), gt=[])
        qsave = set()
        for threshold in (.3, .5):
            hard = ((event_prob >= threshold) if threshold == .3 else (event_prob > threshold)) & ok
            nv = hard.flatten(1).sum(1)
            max_hard = (maximum >= threshold if threshold == .3 else maximum > threshold) & ok
            for k in np.unique(local):
                if k <= 0:
                    continue
                sel = (gt == int(k)) & ok
                count = int(sel.sum())
                if not count:
                    continue
                if threshold == .3:
                    row['gt'].append(dict(instance=int(k), visible_voxels=count,
                                          visible_fraction=count / self.total[k]))
                g = next(g for g in row['gt'] if g['instance'] == k)
                inter = hard[:, sel].sum(1)
                iou = inter.float() / (nv + count - inter).clamp(min=1)
                q = int(iou.argmax())
                kept_iou = torch.where(keep, iou, torch.full_like(iou, -1))
                qk = int(kept_iou.argmax()) if keep.any() else -1
                union = int((max_hard | sel).sum())
                entry = dict(best_all_iou=float(iou[q]), best_all_query=q, best_all_presence=float(event_presence[q]),
                             best_kept_iou=float(iou[qk]) if qk >= 0 else 0., best_kept_query=qk,
                             max_iou=float((max_hard & sel).sum()) / max(union, 1),
                             max_gt_tp=int((max_hard & sel).sum()), max_gt_fn=int((~max_hard & sel).sum()),
                             best_all_mask_voxels=int(nv[q]), best_all_healthy_fp=int((hard[q] & healthy).sum()),
                             best_all_other_lesion_voxels=int((hard[q] & (gt > 0) & ~sel).sum()),
                             best_all_ignored_voxels=int((hard[q] & ignored).sum()),
                             best_all_gt_mean_prob=float(event_prob[q][sel].mean()),
                             best_all_healthy_mean_prob=float(event_prob[q][healthy].mean()) if healthy.any() else None,
                             max_gt_mean_prob=float(maximum[sel].mean()),
                             gaussian_gt_mean=float(weight[sel].mean()))
                g[f'at_{threshold}'] = entry
                qsave.add(q)
                if qk >= 0:
                    qsave.add(qk)
        # Store a bounded set of representative masks; all query measurements above remain available.
        qids = sorted(qsave)[:8]
        np.savez(self.out_dir / f'window_{len(self.window_rows):03d}.npz',
                 window=np.asarray(window), query_ids=np.asarray(qids),
                 representative_prob_float16=event_prob[qids].cpu().numpy().astype(np.float16),
                 presence=event_presence.cpu().numpy(), maximum=maximum.cpu().numpy(),
                 valid=valid.cpu().numpy().astype(np.uint8), weight=weight.cpu().numpy(), instance=local)
        self.window_rows.append(row)

    def verify_probability(self, probability):
        if np.any(self.wsum <= 0):
            raise ValueError('probability reconstruction: uncovered voxels')
        if not np.allclose(self.acc / self.wsum.clip(min=1e-12), probability, atol=1e-6, rtol=1e-6):
            raise ValueError('probability reconstruction mismatch')

    def finish(self, probability, component_map, dets, score_threshold):
        """Whole-volume voxel/CC statistics; these are not local-query or population detection metrics."""
        rows = []
        for k in sorted(int(k) for k in self.total if k > 0):
            sel = self.instance == k
            overlaps = []
            for d in dets:
                c = component_map == d['instance']
                inter = int((c & sel).sum())
                if inter:
                    overlaps.append(dict(component=d['instance'], intersection=inter,
                                         mask_iou=inter / int((c | sel).sum()), score=d['score'],
                                         passes_score=d['score'] >= score_threshold))
            p = probability[sel]
            rows.append(dict(instance=k, n_voxels=int(sel.sum()), mean_prob=float(p.mean()),
                             threshold_tp=int((p >= .3).sum()), threshold_fn=int((p < .3).sum()), components=overlaps))
        return dict(n_windows=len(self.window_rows), windows=self.window_rows, gt=rows,
                    no_present_windows=sum(w['n_present'] == 0 for w in self.window_rows))
