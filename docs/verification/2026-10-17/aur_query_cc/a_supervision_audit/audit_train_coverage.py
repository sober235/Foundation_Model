"""Bounded, anonymous label-only supervision audit. Never constructs a model."""
import hashlib
import json
import os
import time
from collections import Counter
from pathlib import Path

import numpy as np
import torch

from anatobind.aur import losses as L
from anatobind.aur.dataset import AURDataset
from anatobind.aur.labels import ENTITY_LABELS
from anatobind.aur.train import split_rows


OUT = Path('/tmp/aur-a-supervision-audit-20261017')
ROOT = Path('/data2/congcong/data/FM_data')
SAMPLES = ROOT / 'derived/aur/samples_1mm_v1.json'
VAL = ROOT / 'derived/aur/ssl_manifest_v1/val_patients.json'
SEED = 20261017
POINT_SEEDS = (0, 1)
N = 16384


def hfile(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def stats(xs):
    return {'n': len(xs), 'mean': float(np.mean(xs)) if xs else None,
            'median': float(np.median(xs)) if xs else None,
            'min': int(min(xs)) if xs else None, 'max': int(max(xs)) if xs else None}


def aggregate(records):
    out = {}
    for k, label in enumerate(ENTITY_LABELS):
        rr = [r for r in records if r['dense'][k] > 0]
        aa = [r for r in rr if r['unignored_dense'][k] > 0]
        q = {'crop_present': len(rr), 'fully_ignored_crop': len(rr) - len(aa),
             'dense_voxels_when_present': stats([r['dense'][k] for r in rr]),
             'unignored_dense_voxels_when_present': stats([r['unignored_dense'][k] for r in rr])}
        for profile in ('shared', 'uniform'):
            pts = [r[profile][seed][k] for r in aa for seed in range(len(POINT_SEEDS))]
            q[profile] = {'positive_points_when_unignored_present': stats(pts),
                          'zero_positive_draws': sum(x == 0 for x in pts),
                          'under_5_positive_draws': sum(x < 5 for x in pts)}
        out[str(label)] = q
    return out


def run():
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    assert not torch.cuda.is_available()
    rows = json.loads(SAMPLES.read_text())
    val = json.loads(VAL.read_text())
    tr, held = split_rows(rows, val)
    val_set = {(s, str(p)) for s, pp in val.items() for p in pp}
    rng = np.random.default_rng(SEED)
    chosen = []
    # Two distinct TRAIN patients per source, A-supervised, U-visible where available.
    # The selection is frozen before any label-volume or crop measurement.
    for source in ('bmsr', 'isles', 'pdgm', 'sibbms'):
        cand = [(i, r) for i, r in enumerate(tr) if r['source'] == source and r.get('a_supervised', True)
                and (source == 'sibbms' or r['u_supervised'])]
        seen = set()
        for j in rng.permutation(len(cand)):
            i, r = cand[int(j)]
            if str(r['patient']) in seen:
                continue
            seen.add(str(r['patient']))
            chosen.append((i, r))
            if len(seen) == 2:
                break
        assert len(seen) == 2
    for _, r in chosen:
        assert r.get('split', 'train') == 'train'
        assert (r['source'], str(r['patient'])) not in val_set
        for key in ('image', 'anatomy', 'lesion'):
            if r.get(key):
                assert Path(r[key]).is_relative_to(ROOT)
                assert Path(r[key]).resolve().is_relative_to(ROOT)
    picked = [r for _, r in chosen]
    ds = AURDataset(picked, crops_per_volume=2, do_augment=True, lesion_share=0.5, seed=SEED)
    records = []
    start = time.monotonic()
    for i in range(len(ds)):
        crops = ds[i]
        for crop in crops:
            entity = crop['entity'].reshape(-1)
            ignore = crop['a_ignore'].reshape(-1)
            dense = torch.bincount(entity, minlength=33)[1:].tolist()
            unignored_dense = torch.bincount(entity[~ignore], minlength=33)[1:].tolist()
            assert crop['a_supervised'].item()
            assert np.array_equal(np.array(dense) > 0, crop['entity_present'].numpy())
            rec = {'source': picked[i]['source'], 'sequence': picked[i]['sequence'], 'dense': dense,
                   'unignored_dense': unignored_dense,
                   'has_instances': bool(crop['n_instances']),
                   'valid_voxels': int(crop['valid'].sum()), 'shared': [], 'uniform': [],
                   'shared_ignored': [], 'uniform_ignored': []}
            for seed in POINT_SEEDS:
                for profile, inst in (('shared', crop['instance'][None]), ('uniform', None)):
                    points = L.sample_points(crop['valid'][None], N,
                        torch.Generator().manual_seed(SEED + seed), instance=inst)
                    idx = points[0]
                    keep = ~ignore[idx]
                    positives = torch.bincount(entity[idx][keep], minlength=33)[1:].tolist()
                    rec[profile].append(positives)
                    rec[profile + '_ignored'].append(int((~keep).sum()))
            records.append(rec)
        print(json.dumps({'completed_train_volumes': i + 1, 'total_train_volumes': len(ds),
                          'elapsed_seconds': round(time.monotonic() - start, 2)}), flush=True)
        del crops
    # Isolate the exact loss behavior when dense presence is true but sampled positives are absent.
    probe = {}
    for shift in (-4.0, 0.0, 4.0):
        x = torch.full((1, 32, 64), -4.0)
        x[0, 25] = shift
        x.requires_grad_(True)
        present = torch.zeros(1, 32, dtype=torch.bool)
        present[0, 25] = True  # label 18: fixed identity, dense presence supplied independently of points
        loss = L.entity_loss(x, torch.zeros(1, 32), torch.zeros(1, 64, dtype=torch.long),
                             torch.zeros(1, 64, dtype=torch.bool), present)['a_mask']
        loss.backward()
        probe[str(shift)] = {'a_mask': float(loss), 'present_query_logit_gradient_sum': float(x.grad[0, 25].sum())}
    report = {'evidence': 'NOT_EVIDENCE', 'purpose': 'training-label point coverage only; no model, inference, training or parameter selection',
              'seed': SEED, 'point_seeds': list(POINT_SEEDS), 'points_per_draw': N, 'crop': [128, 160, 160],
              'do_augment': True, 'crops_per_volume': 2, 'lesion_share': 0.5,
              'selection_recipe': 'seeded permutation per source of A-supervised TRAIN rows, U-visible except SibBMS, take first two distinct patients; validation patients excluded',
              'samples_sha256': hfile(SAMPLES), 'val_patients_sha256': hfile(VAL),
              'code_revision': 'bbfc4ba6bfbc63f089f99a0919bea6a488d756d5',
              'audit_script_sha256': hfile(Path(__file__)),
              'supervision_source_sha256': {str(q): hfile(q) for q in map(Path, (
                  'anatobind/aur/losses.py', 'anatobind/aur/dataset.py',
                  'anatobind/aur/labels.py', 'anatobind/aur/heads.py'))},
              'selected_row_positions_sha256': hashlib.sha256(json.dumps([i for i, _ in chosen]).encode()).hexdigest(),
              'train_row_counts': [{'source': k[0], 'a_supervised': k[1], 'u_supervised': k[2], 'n': n}
                                   for k, n in sorted(Counter((r['source'], bool(r.get('a_supervised', True)), bool(r['u_supervised'])) for r in tr).items())],
              'n_selected_volumes': len(chosen), 'n_crops': len(records),
              'selected_source_sequences': [{'source': k[0], 'sequence': k[1], 'n': n}
                                            for k, n in sorted(Counter((r['source'], r['sequence']) for r in picked).items())],
              'n_crops_with_instances': sum(r['has_instances'] for r in records),
              'all_a_supervised': True, 'validation_patient_intersection': 0,
              'sampling': {p: {'ignored_points': stats([x for r in records for x in r[p + '_ignored']]),
                               'ignored_fraction': float(np.mean([x / N for r in records for x in r[p + '_ignored']]))}
                           for p in ('shared', 'uniform')},
              'by_entity': aggregate(records),
              'by_source': {s: aggregate([r for r in records if r['source'] == s]) for s in ('bmsr', 'isles', 'pdgm', 'sibbms')},
              'by_instance_presence': {str(v): aggregate([r for r in records if r['has_instances'] == v]) for v in (False, True)},
              'zero_sample_positive_loss_probe': probe,
              'elapsed_seconds': round(time.monotonic() - start, 2),
              'limitations': ['8 TRAIN rows / 16 crops only, not prevalence estimates', 'U-visible rows preferentially sampled; training includes A-only sequences',
                              '2 point draws per fixed crop', 'no checkpoint logits, gradients or dense fit measured',
                              'sampler is a mechanism candidate, not established cause of the five missing classes',
                              'all anatomy targets are SynthSeg pseudo-labels']}
    OUT.mkdir(exist_ok=True)
    (OUT / 'aggregate.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps({'status': 'complete', 'n_volumes': len(chosen), 'n_crops': len(records),
                      'n_instance_crops': report['n_crops_with_instances'], 'elapsed_seconds': report['elapsed_seconds']}))


if __name__ == '__main__':
    try:
        run()
    except Exception as exc:
        # Do not expose row or patient identifiers embedded in data-loader errors.
        print(json.dumps({'status': 'failed', 'exception_type': type(exc).__name__}), flush=True)
        raise SystemExit(1)
