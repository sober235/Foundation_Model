#!/usr/bin/env python
"""Frozen six-patient validation replay. Private outputs stay local; no training or model changes."""
import argparse
import json
import os
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import torch
import torch.nn.functional as F

from anatobind.aur import infer as I, eval as E
from anatobind.aur.dataset import load_volume
from anatobind.aur.diagnostics import WindowAudit, identity, select_cases, verify_detections, verify_provenance
from anatobind.aur.labels import ENTITY_LABELS, entity_to_synthseg
from anatobind.aur.relation import geometry, host_masks_from_entities
from anatobind.aur.ssl.checkpoint import file_sha256, code_sha
from anatobind.aur.targets import host_targets
from anatobind.aur.train import split_rows

CKPT_SHA = '38f6376333041a75be1cce83e927bc221468761f80ae952e2a3020805f7dcbde'
SAMPLES_SHA = 'fb96acd741aa6c0ee689eb772a4332304501c8d63ebd7ccddcd657f41eb86d5d'
SCORES = {'bmsr': .7, 'isles': .6, 'pdgm': .75}


def clean(value):
    if isinstance(value, dict):
        return {str(k): clean(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [clean(v) for v in value]
    if isinstance(value, np.ndarray):
        return clean(value.tolist())
    if isinstance(value, np.generic):
        return clean(value.item())
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def write(path, data):
    Path(path).write_text(json.dumps(clean(data), indent=1, allow_nan=False))


class BindAudit:
    def __init__(self, volume, pred_seg, out_dir, deadline):
        self.volume, self.pred_seg, self.out_dir, self.deadline = volume, pred_seg, Path(out_dir), deadline
        self.rows = []

    def __call__(self, *, model, out, target, query, window, result):
        if time.monotonic() > self.deadline:
            raise TimeoutError('frozen replay time budget exceeded')
        device = target.device
        with I._autocast(device):
            pix = {'coarse': out['pix']['coarse']}
            # model.bind applies sigmoid/gating in the autocast output dtype, before geometry.float().
            raw_native = model.masks.coarse_masks(out['entity_embed'], pix)[0].sigmoid()
            raw_a = raw_native.float()
            gate_native = out['entity_presence'][0].sigmoid()[:, None, None, None]
            gated_a = raw_native * gate_native
            coarse_u = model.masks.coarse_masks(out['event_embed'][:, query:query + 1], pix)[0].sigmoid()
            valid = out['levels'][0]['valid'][0]
            hosts = host_masks_from_entities(gated_a[None])[0] * valid
            geo = geometry(coarse_u * valid, hosts, out['levels'][0]['coords'][0])
            dense_a = model.entity_masks(out).float().sigmoid()[0]
        amax, aid = dense_a.max(0)
        amap = torch.where(amax >= .5, aid + 1, 0).to(torch.uint8).cpu().numpy()
        truth_entity, _ = I.extract(self.volume['entity'], window, fill=0)
        true_seg = entity_to_synthseg(truth_entity)
        coarse_truth = F.interpolate(torch.as_tensor(truth_entity.copy(), device=device)[None, None].float(),
                                     size=raw_a.shape[1:], mode='nearest')[0, 0].cpu().numpy().astype(np.uint8)
        scores = {}
        for name, p in (('coarse_ungated', raw_a), ('coarse_gated', gated_a.float())):
            mx, ix = p.max(0)
            m = torch.where(mx >= .5, ix + 1, 0).to(torch.uint8).cpu().numpy()
            scores[name] = E.host_dice(entity_to_synthseg(m), entity_to_synthseg(coarse_truth))
        k = result['instance']
        local_host = host_targets(target.cpu().numpy().astype(np.int32), true_seg, self.volume['spacing'])['host'][0]
        row = dict(instance=k, query=query, query_iou=result['query_iou'], in_crop_share=result['in_crop_share'],
                   truth=int(self.volume['hosts']['host'][k - 1]), local_truth=int(local_host), r=result['host'],
                   r_probs=result['host_probs'], window=window, full_local_a=E.host_dice(entity_to_synthseg(amap), true_seg),
                   coarse_a=scores, coarse_overlap_choice=int(geo[0, :, 0].argmax()),
                   coarse_overlap=geo[0, :, 0].cpu().numpy(), coarse_u_mean=float(coarse_u.mean()),
                   entity_presence=out['entity_presence'][0].float().sigmoid().cpu().numpy())
        np.savez(self.out_dir / f'bind_{k:03d}.npz', coarse_a_ungated=raw_a.cpu().numpy(),
                 coarse_a_gated=gated_a.float().cpu().numpy(), coarse_u=coarse_u.float().cpu().numpy(),
                 geometry=geo.cpu().numpy(), valid=valid.cpu().numpy(), coords=out['levels'][0]['coords'][0].cpu().numpy(),
                 target=target.cpu().numpy(), full_a=amap, true_a=truth_entity)
        self.rows.append(row)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    for name in ('samples', 'val-patients', 'saved-eval', 'checkpoint', 'out'):
        ap.add_argument('--' + name, type=Path, required=True)
    ap.add_argument('--gpu', type=int, required=True)
    ap.add_argument('--max-seconds', type=int, default=7200)
    a = ap.parse_args()
    if a.out.exists():
        raise FileExistsError('diagnostic output must be new')
    saved = json.loads((a.saved_eval / 'cases.json').read_text())
    run = json.loads((a.saved_eval / 'aggregate.json').read_text())['run']
    verify_provenance(run, a.samples, a.checkpoint)
    if run['checkpoint_sha256'] != CKPT_SHA or run['samples_sha256'] != SAMPLES_SHA:
        raise ValueError('not the approved evaluated checkpoint/manifest')
    _, val = split_rows(json.loads(a.samples.read_text()), json.loads(a.val_patients.read_text()))
    if {identity(r) for r in val} != {identity(r) for r in saved}:
        raise ValueError('validation membership does not match original evaluation')
    rows = select_cases(val, saved)
    saved_by_id = {identity(r): r for r in saved}
    a.out.mkdir(parents=True)
    os.chmod(a.out, 0o700)
    write(a.out / 'selection_private.json', rows)
    selection_sha = file_sha256(a.out / 'selection_private.json')
    # Freeze configuration before image loading or model inference.
    contract = dict(evidence='NOT_EVIDENCE', checkpoint_sha256=CKPT_SHA, samples_sha256=SAMPLES_SHA,
                    val_patients_sha256=file_sha256(a.val_patients), selection_sha256=selection_sha,
                    code_sha=code_sha(), sources={s: 2 for s in SCORES}, crop=[128, 160, 160], batch_size=2,
                    max_seconds=a.max_seconds, saved_eval=str(a.saved_eval), instance_threshold=.3,
                    presence_threshold=.5, score_thresholds=SCORES, gpu=a.gpu)
    write(a.out / 'contract.json', contract)
    device = torch.device('cuda', a.gpu)
    model, _ = I.model_from_export(a.checkpoint, device)
    t0 = time.monotonic()
    deadline = t0 + min(a.max_seconds, 7200)
    summary = dict(evidence='NOT_EVIDENCE', contract=contract, cases=[], status='running')
    try:
        for i, row in enumerate(rows):
            if time.monotonic() > deadline:
                raise TimeoutError('frozen replay time budget exceeded')
            case_dir = a.out / f'case_{i + 1:02d}'
            case_dir.mkdir()
            start = time.monotonic()
            torch.cuda.reset_peak_memory_stats(device)
            vol = load_volume(row)
            audit = WindowAudit(vol['instance'], vol['small'], case_dir / 'windows')
            def observer(**kw):
                if time.monotonic() > deadline:
                    raise TimeoutError('frozen replay time budget exceeded')
                audit(**kw)
            pred = I.predict_volume(model, vol['image'], vol['affine'], (128, 160, 160), device,
                                    batch_size=2, observer=observer)
            audit.verify_probability(pred['lesion_prob'])
            components, dets = I.instances_from_probability(pred['lesion_prob'], vol['voxel_mm3'])
            expected = saved_by_id[identity(row)]
            verify_detections(dets, expected['u']['dets'])
            pred_seg, true_seg = entity_to_synthseg(pred['entity']), entity_to_synthseg(vol['entity'])
            inside = vol['a_ignore']
            anatomy = dict(all=E.host_dice(pred_seg, true_seg),
                           lesion=E.host_dice(pred_seg[inside], true_seg[inside]),
                           outside_lesion=E.host_dice(pred_seg[~inside], true_seg[~inside]),
                           entity_counts={str(l): dict(truth=int((true_seg == l).sum()), pred=int((pred_seg == l).sum()),
                                                      presence=float(pred['entity_presence'][j])) for j, l in enumerate(ENTITY_LABELS)})
            ba = BindAudit(vol, pred_seg, case_dir, deadline)
            bound = I.bind_instances(model, vol['image'], vol['affine'], vol['instance'], (128, 160, 160), device, observer=ba)
            old_r = {r['instance']: r for r in (expected.get('r') or {}).get('instances', [])}
            for r in bound:
                if r['instance'] in old_r:
                    old = old_r[r['instance']]
                    if r['host'] != old['r'] or abs(r['query_iou'] - old['query_iou']) > 1e-5:
                        raise ValueError('baseline reproduction: controlled R mismatch')
            u = audit.finish(pred['lesion_prob'], components, dets, SCORES[row['source']])
            np.savez(case_dir / 'volume.npz', entity=pred['entity'], lesion_prob=pred['lesion_prob'], instance=vol['instance'],
                     predicted_components=components, true_entity=vol['entity'], a_ignore=vol['a_ignore'], affine=vol['affine'])
            write(case_dir / 'diagnostic.json', dict(a=anatomy, u=u, r=ba.rows, detections=dets))
            item = dict(anonymous_case=i + 1, source=row['source'], sequence=row['sequence'], reproduced=True,
                        n_gt=len(vol['hosts']['host']), n_windows=pred['n_windows'], n_detections=len(dets),
                        seconds=round(time.monotonic() - start, 2), peak_cuda_gib=torch.cuda.max_memory_allocated(device) / 2**30,
                        a_host_macro=anatomy['all']['host_macro'], a_entity_macro=anatomy['all']['entity_macro'])
            summary['cases'].append(item)
            write(a.out / 'summary.json', summary)
            print(json.dumps(item), flush=True)
            del pred, audit, ba, vol
        summary['status'] = 'complete'
    except Exception as exc:
        summary['status'] = 'stopped'
        summary['reason'] = type(exc).__name__ + ': ' + str(exc)
        raise
    finally:
        summary['seconds'] = round(time.monotonic() - t0, 2)
        write(a.out / 'summary.json', summary)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
